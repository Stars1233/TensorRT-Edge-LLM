/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "audioRunner.h"
#include "audioUtils.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

Qwen3OmniAudioRunner::Qwen3OmniAudioRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner()
{
    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "Failed to validate and fill config");

    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    // Initialize audio encoder (engineDir is already the audio subdirectory)
    std::string audioEnginePath = engineDir + "/audio_encoder.engine";
    if (std::filesystem::exists(audioEnginePath))
    {
        LOG_INFO("Loading audio encoder from %s", audioEnginePath.c_str());
        try
        {
            auto mmapReader = std::make_unique<file_io::MmapReader>(audioEnginePath);
            mAudioEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
                mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));
            ELLM_CHECK(mAudioEngine, "Failed to deserialize audio encoder engine");

            mAudioContext = std::unique_ptr<nvinfer1::IExecutionContext>(mAudioEngine->createExecutionContext());
            ELLM_CHECK(mAudioContext, "Failed to create audio encoder context");

            bool const profileSet = mAudioContext->setOptimizationProfileAsync(0, stream);
            ELLM_CHECK(profileSet, "Failed to set optimization profile for audio encoder");
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to load audio encoder: %s", e.what());
            throw;
        }
    }
    else
    {
        throw std::runtime_error("Audio encoder not found at " + audioEnginePath);
    }

    bool const bufferAllocated = allocateBuffer(stream);
    ELLM_CHECK(bufferAllocated, "Failed to allocate buffers");

    LOG_INFO("Qwen3OmniAudioRunner initialized successfully");
}

bool Qwen3OmniAudioRunner::validateAndFillConfig(std::string const& engineDir)
{
    // Read config.json
    std::string configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

    Json jsonConfig;
    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }

    // Parse audio configuration
    if (jsonConfig.contains("audio_config"))
    {
        auto audioConfig = jsonConfig["audio_config"];
        mConfig.melBins = audioConfig.value("num_mel_bins", 128);
        mConfig.audioFeatureDim = audioConfig.value("output_dim", 2560);
        mConfig.nWindow = audioConfig.value("n_window", 50);
        mConfig.nWindowInfer = audioConfig.value("n_window_infer", 200);
    }
    else
    {
        LOG_WARNING("audio_config not found in config.json, using default values");
    }

    // Parse audio special token IDs from top-level config (may differ between Qwen3-Omni and Qwen3-ASR)
    if (jsonConfig.contains("audio_token_id"))
    {
        mConfig.audioTokenId = jsonConfig["audio_token_id"].get<int32_t>();
    }
    if (jsonConfig.contains("audio_start_token_id"))
    {
        mConfig.audioBosTokenId = jsonConfig["audio_start_token_id"].get<int32_t>();
    }
    if (jsonConfig.contains("audio_end_token_id"))
    {
        mConfig.audioEosTokenId = jsonConfig["audio_end_token_id"].get<int32_t>();
    }
    LOG_DEBUG("Audio token IDs: audio_pad=%d, audio_start=%d, audio_end=%d", mConfig.audioTokenId,
        mConfig.audioBosTokenId, mConfig.audioEosTokenId);

    // Parse rope_theta for MRope initialization (from text_config or top-level)
    if (jsonConfig.contains("text_config") && jsonConfig["text_config"].contains("rope_theta"))
    {
        mConfig.mropeTheta = jsonConfig["text_config"]["rope_theta"].get<float>();
    }
    else if (jsonConfig.contains("rope_theta"))
    {
        mConfig.mropeTheta = jsonConfig["rope_theta"].get<float>();
    }

    return true;
}

bool Qwen3OmniAudioRunner::allocateBuffer([[maybe_unused]] cudaStream_t stream)
{
    if (!mAudioEngine || !mAudioContext)
    {
        LOG_ERROR("Cannot allocate buffers - audio encoder not loaded");
        return false;
    }

    // Get max shapes from optimization profile using tensor names directly
    nvinfer1::Dims paddedFeaturesShapeMax
        = mAudioEngine->getProfileShape(binding_names::kAudioPaddedFeatures, 0, nvinfer1::OptProfileSelector::kMAX);
    nvinfer1::Dims paddedMaskIndicesShapeMax
        = mAudioEngine->getProfileShape(binding_names::kAudioPaddedMaskIndices, 0, nvinfer1::OptProfileSelector::kMAX);

    // Extract dimensions
    int64_t const melBins = mConfig.melBins;
    int64_t const audioFeatureDim = mConfig.audioFeatureDim;       // Use config value, not output shape
    int64_t const maxAudioTokens = paddedMaskIndicesShapeMax.d[0]; // Max attention elements = max audio tokens
    int64_t const maxTimeSteps = 6000;                             // Conservative estimate

    // Allocate tensors
    mMelSpectrogram = rt::Tensor({1, melBins, maxTimeSteps}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

    int64_t const maxNumChunks = paddedFeaturesShapeMax.d[0];
    int64_t const maxChunkLen = paddedFeaturesShapeMax.d[2];
    mPaddedFeature = rt::Tensor({maxNumChunks, melBins, maxChunkLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    int64_t const maxLenAfterCNN = (maxChunkLen - 1) / 2 + 1;
    mPaddedMaskAfterCNN = rt::Tensor({maxNumChunks, maxLenAfterCNN}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);

    int64_t const maxValidElements = paddedMaskIndicesShapeMax.d[0];
    mPaddedMaskIndices = rt::Tensor({maxValidElements, 2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64);

    mAfterCNNLens = rt::Tensor({maxNumChunks}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT64);

    mAudioEmbedding = rt::Tensor({maxAudioTokens, audioFeatureDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    return true;
}

bool Qwen3OmniAudioRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, [[maybe_unused]] bool imageOnly)
{
    if (!mropeCosSinOut.has_value())
    {
        LOG_ERROR("mropeCosSinOut is required (this runner is MRope-only).");
        return false;
    }

    std::vector<int64_t> audioTokenLengths;

    // Step 1: Process audio inputs to get embeddings and token lengths
    for (auto const& req : request.requests)
    {
        if (!req.audioBuffers.empty())
        {
            if (!preprocessAudio(req.audioBuffers, audioTokenLengths, stream))
            {
                LOG_ERROR("Audio preprocessing failed");
                return false;
            }
        }
        else
        {
            // No audio in this request, add 0 length
            audioTokenLengths.push_back(0);
        }
    }

    // Step 2: Tokenize and replace audio tokens (similar to QwenViTRunner::textPreprocess)
    textPreprocess(request, batchedInputIds, audioTokenLengths, tokenizer);

    // Step 3: Initialize sequential MRope cache if applicable.
    // For audio+text only (no vision), all 3 MRope dimensions (T, H, W) use identical sequential positions.
    // When a vision runner is also present, QwenViTRunner::preprocess will overwrite the MRope cache
    // with vision-aware position IDs, so this initialization is harmlessly overwritten.
    int64_t const activeBatchSize = static_cast<int64_t>(request.requests.size());
    if (!initializeSequentialMRopeCache(activeBatchSize, mropeCosSinOut.value().get(), stream))
    {
        LOG_ERROR("Failed to initialize sequential MRope cache for audio input.");
        return false;
    }

    return true;
}

void Qwen3OmniAudioRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& audioTokenLengths,
    tokenizer::Tokenizer const* tokenizer)
{
    ELLM_CHECK(audioTokenLengths.size() == request.requests.size(),
        "audioTokenLengths.size() != request.requests.size(), " + std::to_string(audioTokenLengths.size())
            + " != " + std::to_string(request.requests.size()));

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids;

        // Check if already tokenized (incremental mode)
        if (i < batchInputIds.size() && !batchInputIds[i].empty())
        {
            // Already tokenized by another runner, use existing tokens
            ids = batchInputIds[i];
        }
        else
        {
            // First runner to process, tokenize the request
            ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        }

        // Insert audio tokens if present
        if (audioTokenLengths[i] > 0)
        {
            std::vector<int32_t> newIds;
            for (size_t j = 0; j < ids.size(); ++j)
            {
                if (ids[j] == mConfig.audioTokenId)
                {
                    // Replace <|audio_pad|> placeholder with: <|audio_start|> + N×<|audio_pad|> + <|audio_end|>
                    // TRT chat template only has <|audio_pad|> without start/end markers
                    int64_t numAudioTokens = audioTokenLengths[i];

                    newIds.push_back(mConfig.audioBosTokenId);

                    for (int64_t k = 0; k < numAudioTokens; ++k)
                    {
                        newIds.push_back(mConfig.audioTokenId);
                    }

                    newIds.push_back(mConfig.audioEosTokenId);
                }
                else
                {
                    newIds.push_back(ids[j]);
                }
            }

            // Update batchInputIds
            if (i < batchInputIds.size())
            {
                batchInputIds[i] = std::move(newIds);
            }
            else
            {
                batchInputIds.emplace_back(std::move(newIds));
            }
        }
        else
        {
            // No audio in this request, keep original tokens
            if (i < batchInputIds.size())
            {
                batchInputIds[i] = std::move(ids);
            }
            else
            {
                batchInputIds.emplace_back(std::move(ids));
            }
        }
    }
}

bool Qwen3OmniAudioRunner::loadMelSpectrogramFromFile(
    std::string const& filePath, std::string const& format, rt::Tensor& melSpectrogram, cudaStream_t stream)
{
    if (format != "safetensors")
    {
        LOG_ERROR("Only safetensors format is supported. Got: %s", format.c_str());
        return false;
    }

    // Load mel-spectrogram from safetensors file
    std::vector<rt::Tensor> tensors;
    if (!safetensors::loadSafetensors(filePath, tensors, stream))
    {
        LOG_ERROR("Failed to load mel-spectrogram from safetensors file: %s", filePath.c_str());
        return false;
    }

    // Use the first tensor as mel-spectrogram (should contain exactly one tensor)
    check::check(tensors.size() == 1, "Mel-spectrogram safetensors should contain exactly one tensor");
    check::check(tensors[0].getShape().getNumDims() == 3, "Mel-spectrogram should be 3D [1, mel_bins, time_steps]");
    check::check(tensors[0].getDataType() == nvinfer1::DataType::kHALF, "Mel-spectrogram must be FP16 (HALF)");

    melSpectrogram = std::move(tensors[0]);
    return true;
}

bool Qwen3OmniAudioRunner::preprocessAudio(std::vector<rt::audioUtils::AudioData> const& audioBuffers,
    std::vector<int64_t>& audioTokenLengths, cudaStream_t stream)
{
    if (audioBuffers.empty())
    {
        return true;
    }

    if (!mAudioEngine || !mAudioContext)
    {
        LOG_ERROR("Audio encoder not loaded");
        return false;
    }

    // Process each audio clip
    for (auto const& audio : audioBuffers)
    {
        rt::Tensor melSpec;

        // Load pre-computed mel-spectrogram
        // Audio preprocessing (Mel-spectrogram computation) must be done externally
        // using dedicated audio processing libraries (e.g., librosa, torchaudio) in Python
        if (audio.melSpectrogramPath.empty())
        {
            LOG_ERROR(
                "Pre-computed mel-spectrogram path is required. "
                "Audio preprocessing must be done externally using Python or other pipelines.");
            return false;
        }

        if (!loadMelSpectrogramFromFile(audio.melSpectrogramPath, audio.melSpectrogramFormat, melSpec, stream))
        {
            LOG_ERROR("Failed to load mel-spectrogram from file");
            return false;
        }

        int64_t const timeSteps = melSpec.getShape()[2];
        LOG_DEBUG("Mel-spectrogram shape: [%ld, %ld, %ld]", melSpec.getShape()[0], melSpec.getShape()[1], timeSteps);

        // Preprocess for audio encoder
        std::vector<int64_t> afterCNNLens;
        if (!audioUtils::preprocessAudioForEncoder(
                melSpec, mConfig.nWindow, mPaddedFeature, mPaddedMaskAfterCNN, afterCNNLens, stream))
        {
            LOG_ERROR("Failed to preprocess audio for encoder");
            return false;
        }

        // Convert mask to indices
        if (!audioUtils::convertMaskToIndices(mPaddedMaskAfterCNN, mPaddedMaskIndices, stream))
        {
            LOG_ERROR("Failed to convert mask to indices");
            return false;
        }

        LOG_DEBUG("Mask shape: [%ld, %ld], Indices shape: [%ld, %ld]", mPaddedMaskAfterCNN.getShape()[0],
            mPaddedMaskAfterCNN.getShape()[1], mPaddedMaskIndices.getShape()[0], mPaddedMaskIndices.getShape()[1]);

        // Create attention mask with merged windows (matching PyTorch cu_seqlens logic)
        if (!audioUtils::createChunkwiseAttentionMask(
                afterCNNLens, mConfig.nWindow, mConfig.nWindowInfer, mAudioAttentionMask, stream))
        {
            LOG_ERROR("Failed to create attention mask");
            return false;
        }

        LOG_DEBUG(
            "Created attention mask [%ld, %ld]", mAudioAttentionMask.getShape()[0], mAudioAttentionMask.getShape()[1]);

        // Calculate total audio tokens
        int64_t const totalAudioTokens = mPaddedMaskIndices.getShape()[0];

        // Reshape output buffer
        if (!mAudioEmbedding.reshape({totalAudioTokens, mConfig.audioFeatureDim}))
        {
            LOG_ERROR("Failed to reshape audio output");
            return false;
        }

        LOG_DEBUG("Reshaped audio output to [%ld, %d]", totalAudioTokens, mConfig.audioFeatureDim);

        // Set input shapes
        if (!mAudioContext->setInputShape(binding_names::kAudioPaddedFeatures, mPaddedFeature.getShape().getTRTDims()))
        {
            LOG_ERROR("Failed to set padded features input shape");
            return false;
        }

        if (!mAudioContext->setInputShape(
                binding_names::kAudioPaddedMaskIndices, mPaddedMaskIndices.getShape().getTRTDims()))
        {
            LOG_ERROR("Failed to set padded mask indices input shape");
            return false;
        }

        if (!mAudioContext->setInputShape(
                binding_names::kAudioAttentionMask, mAudioAttentionMask.getShape().getTRTDims()))
        {
            LOG_ERROR("Failed to set attention mask input shape");
            return false;
        }

        // Set tensor addresses
        if (!mAudioContext->setTensorAddress(binding_names::kAudioPaddedFeatures, mPaddedFeature.rawPointer()))
        {
            LOG_ERROR("Failed to set padded features input address");
            return false;
        }

        if (!mAudioContext->setTensorAddress(binding_names::kAudioPaddedMaskIndices, mPaddedMaskIndices.rawPointer()))
        {
            LOG_ERROR("Failed to set padded mask indices input address");
            return false;
        }

        if (!mAudioContext->setTensorAddress(binding_names::kAudioAttentionMask, mAudioAttentionMask.rawPointer()))
        {
            LOG_ERROR("Failed to set attention mask input address");
            return false;
        }

        if (!mAudioContext->setTensorAddress(binding_names::kAudioOutput, mAudioEmbedding.rawPointer()))
        {
            LOG_ERROR("Failed to set audio output address");
            return false;
        }

        // Execute audio encoder
        LOG_DEBUG(
            "Executing audio encoder with shapes: "
            "input=[%ld,%ld,%ld], indices=[%ld,2], mask=[%ld,%ld], output=[%ld,%ld]",
            mPaddedFeature.getShape()[0], mPaddedFeature.getShape()[1], mPaddedFeature.getShape()[2],
            mPaddedMaskIndices.getShape()[0], mAudioAttentionMask.getShape()[0], mAudioAttentionMask.getShape()[1],
            mAudioEmbedding.getShape()[0], mAudioEmbedding.getShape()[1]);

        {
            TIME_STAGE(metrics::StageNames::kAUDIO_ENCODER, stream);

            if (!mAudioContext->enqueueV3(stream))
            {
                LOG_ERROR("Audio encoder inference failed");
                return false;
            }
        }

        LOG_DEBUG("Audio encoder inference completed");

        audioTokenLengths.push_back(totalAudioTokens);
        mMultimodalMetrics.recordRun(0, 0, 1, totalAudioTokens);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

bool Qwen3OmniAudioRunner::infer([[maybe_unused]] cudaStream_t stream)
{
    LOG_DEBUG("No-op (inference already done in preprocessAudio)");
    return true;
}

rt::Tensor& Qwen3OmniAudioRunner::getOutputEmbedding()
{
    return mAudioEmbedding;
}

bool Qwen3OmniAudioRunner::preprocessSystemPrompt(std::string const& systemPrompt,
    [[maybe_unused]] tokenizer::Tokenizer const* tokenizer, rt::OptionalOutputTensor mropeCosSinOut,
    cudaStream_t stream)
{
    if (systemPrompt.empty())
    {
        return true;
    }

    if (!mropeCosSinOut.has_value())
    {
        LOG_ERROR("mropeCosSinOut is required for non-empty system prompts.");
        return false;
    }

    // For audio-only MRope models (e.g. Qwen3-ASR), initialize sequential MRope cache
    // for system prompt since no vision runner will fill it.
    // Batch size is always 1 for system prompt KVCache generation.
    return initializeSequentialMRopeCache(1, mropeCosSinOut.value().get(), stream);
}

bool Qwen3OmniAudioRunner::initializeSequentialMRopeCache(
    int64_t activeBatchSize, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream)
{
    if (mConfig.mropeTheta <= 0.0F)
    {
        // Not an MRope model, nothing to do.
        return true;
    }

    auto const ropeShape = ropeRotaryCosSinDevice.getShape();
    int64_t const maxPositionEmbeddings = ropeShape[1];
    int64_t const rotaryDim = ropeShape[2];

    // Allocate host position IDs: [activeBatchSize, 3, maxPositionEmbeddings]
    // For audio+text only, all 3 dimensions (T, H, W) use identical sequential positions [0, 1, 2, ...]
    int64_t const numElements = activeBatchSize * 3 * maxPositionEmbeddings;
    std::vector<int64_t> positionIdsHost(numElements);

    for (int64_t b = 0; b < activeBatchSize; ++b)
    {
        int64_t batchOffset = b * 3 * maxPositionEmbeddings;
        for (int64_t dim = 0; dim < 3; ++dim)
        {
            for (int64_t pos = 0; pos < maxPositionEmbeddings; ++pos)
            {
                positionIdsHost[batchOffset + dim * maxPositionEmbeddings + pos] = pos;
            }
        }
    }

    // Copy position IDs to device
    rt::Tensor positionIdsDevice({activeBatchSize, 3, maxPositionEmbeddings}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kINT64, "sequentialMRopePositionIds");
    CUDA_CHECK(cudaMemcpyAsync(positionIdsDevice.rawPointer(), positionIdsHost.data(), numElements * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));

    // Reshape the cos/sin cache and fill it
    check::check(
        ropeRotaryCosSinDevice.reshape({activeBatchSize, maxPositionEmbeddings, rotaryDim}), "Tensor reshape failed");

    // All audio+text MRope models (Qwen3-ASR, Qwen3-Omni) use interleaved layout.
    bool constexpr interleaved = true;
    try
    {
        kernel::initializeMRopeCosSin(ropeRotaryCosSinDevice.dataPointer<float>(),
            positionIdsDevice.dataPointer<int64_t>(), mConfig.mropeTheta, rotaryDim, maxPositionEmbeddings,
            activeBatchSize, interleaved, mConfig.mropeSectionH, mConfig.mropeSectionW, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("initializeSequentialMRopeCache: kernel launch failed: %s", e.what());
        return false;
    }

    // Synchronize to ensure kernel completes before local positionIdsDevice is freed.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_INFO("Initialized sequential MRope cos/sin cache for %ld batches, %ld positions", activeBatchSize,
        maxPositionEmbeddings);

    return true;
}

} // namespace rt
} // namespace trt_edgellm
