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

#include "nemotronOmniAudioRunner.h"
#include "common/checkMacros.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

NemotronOmniAudioRunner::NemotronOmniAudioRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner()
{
    // Audio runner does not use the visual engine from base class.
    // Load the audio engine instead.
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    std::string const enginePath = engineDir + "/audio_encoder.engine";
    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    mAudioEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    mAudioContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mAudioEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    bool const profileSet = mAudioContext->setOptimizationProfileAsync(0, stream);
    ELLM_CHECK(profileSet, "Failed to set optimization profile for audio engine");

    bool const configValid = validateAndFillConfig(engineDir);
    ELLM_CHECK(configValid, "NemotronOmniAudioRunner: Failed to validate config");
    bool const bufferAllocated = allocateBuffer(stream);
    ELLM_CHECK(bufferAllocated, "NemotronOmniAudioRunner: Failed to allocate buffer");
}

bool NemotronOmniAudioRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json jsonConfig;
    std::string const configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

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

    mModelType = multimodal::ModelType::NEMOTRON_OMNI_AUDIO_ENCODER;

    if (!jsonConfig.contains("sound_config"))
    {
        LOG_ERROR("sound_config not found in config.json");
        return false;
    }
    auto const& soundConfig = jsonConfig["sound_config"];

    if (!soundConfig.contains("num_mel_bins"))
    {
        LOG_ERROR("sound_config.num_mel_bins not found in config.json");
        return false;
    }
    mConfig.melBins = soundConfig["num_mel_bins"].get<int32_t>();

    if (soundConfig.contains("subsampling_factor"))
    {
        mConfig.subsamplingFactor = soundConfig["subsampling_factor"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("sound_config.subsampling_factor not found in config.json");
        return false;
    }

    if (soundConfig.contains("sampling_rate"))
    {
        mConfig.samplingRate = soundConfig["sampling_rate"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("sound_config.sampling_rate not found in config.json");
        return false;
    }

    if (!jsonConfig.contains("sound_context_token_id"))
    {
        LOG_ERROR("sound_context_token_id not found in config.json");
        return false;
    }
    mConfig.soundContextTokenId = jsonConfig["sound_context_token_id"].get<int32_t>();

    if (jsonConfig.contains("llm_config") && jsonConfig["llm_config"].contains("vocab_size"))
    {
        mConfig.vocabSize = jsonConfig["llm_config"]["vocab_size"].get<int32_t>();
    }

    nvinfer1::Dims const inputShapeMax
        = mAudioEngine->getProfileShape("input_features", 0, nvinfer1::OptProfileSelector::kMAX);
    mMaxSeqLen = inputShapeMax.d[1];

    nvinfer1::Dims const outputShape = mAudioEngine->getTensorShape("last_hidden_state");
    mConfig.audioFeatureDim = outputShape.d[2];

    LOG_INFO("NemotronOmniAudioRunner: melBins=%d, maxSeqLen=%ld, hiddenDim=%d, soundTokenId=%d", mConfig.melBins,
        mMaxSeqLen, mConfig.audioFeatureDim, mConfig.soundContextTokenId);

    return true;
}

bool NemotronOmniAudioRunner::allocateBuffer([[maybe_unused]] cudaStream_t stream)
{
    bool status{true};

    mInputFeatures = rt::Tensor({1, mMaxSeqLen, mConfig.melBins}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "NemotronOmniAudioRunner::mInputFeatures");
    status &= mAudioContext->setTensorAddress("input_features", mInputFeatures.rawPointer());

    // Initial output buffer is sized for one full-length clip. preprocess()
    // grows it via resizeEmbeddingForRows() when a batch packs more clips.
    // last_hidden_state is rebound per clip with the row offset, so no
    // setTensorAddress here.
    int64_t const maxEncodedSeqLen = mMaxSeqLen / mConfig.subsamplingFactor;
    mAudioEmbedding = rt::Tensor({maxEncodedSeqLen, static_cast<int64_t>(mConfig.audioFeatureDim)},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::mAudioEmbedding");

    if (!status)
    {
        LOG_ERROR("Failed to set tensor addresses for audio engine");
        return false;
    }

    return true;
}

bool NemotronOmniAudioRunner::resizeEmbeddingForRows(int64_t rows)
{
    int64_t const hidden = mConfig.audioFeatureDim;
    int64_t const requiredBytes = rows * hidden * static_cast<int64_t>(sizeof(__half));
    if (requiredBytes <= mAudioEmbedding.getMemoryCapacity())
    {
        return mAudioEmbedding.reshape({rows, hidden});
    }
    // Existing capacity isn't enough — reallocate. Encoded rows haven't been
    // written yet at this point in preprocess(), so dropping the old buffer
    // is safe.
    mAudioEmbedding = rt::Tensor(
        {rows, hidden}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::mAudioEmbedding");
    return true;
}

bool NemotronOmniAudioRunner::loadMelSpectrogramFromFile(
    std::string const& filePath, std::string const& format, rt::Tensor& melSpectrogram, cudaStream_t stream)
{
    if (format != "safetensors")
    {
        LOG_ERROR("Only safetensors format is supported. Got: %s", format.c_str());
        return false;
    }

    std::vector<rt::Tensor> tensors;
    if (!safetensors::loadSafetensors(filePath, tensors, stream))
    {
        LOG_ERROR("Failed to load mel-spectrogram from: %s", filePath.c_str());
        return false;
    }

    check::check(tensors.size() == 1, "Mel-spectrogram safetensors should contain exactly one tensor");
    check::check(tensors[0].getDataType() == nvinfer1::DataType::kHALF, "Mel-spectrogram must be FP16");

    melSpectrogram = std::move(tensors[0]);
    return true;
}

bool NemotronOmniAudioRunner::encodeSingleClip(
    rt::Tensor const& mel, int64_t destRowOffset, int64_t& encodedRowsOut, cudaStream_t stream)
{
    // Mel shape: [1, rawSeqLen, mel_bins]. Pad rawSeqLen up to a multiple of
    // subsamplingFactor so the 3× stride-2 subsampling produces an integer
    // number of output rows.
    int64_t const rawSeqLen = mel.getShape()[1];
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };
    int64_t const paddedSeqLen = alignUp(rawSeqLen, mConfig.subsamplingFactor);
    int64_t const encodedSeqLen = paddedSeqLen / mConfig.subsamplingFactor;

    check::check(
        mInputFeatures.reshape({1, paddedSeqLen, static_cast<int64_t>(mConfig.melBins)}), "Tensor reshape failed");
    if (paddedSeqLen > rawSeqLen)
    {
        CUDA_CHECK(cudaMemsetAsync(mInputFeatures.rawPointer(), 0, mInputFeatures.getMemoryCapacity(), stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        mInputFeatures.rawPointer(), mel.rawPointer(), mel.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));

    // Bind the encoder output to this clip's slot in mAudioEmbedding so each
    // enqueueV3 writes directly to its destination — no scatter step needed.
    auto* const embeddingBase = static_cast<std::byte*>(mAudioEmbedding.rawPointer());
    int64_t const destByteOffset
        = destRowOffset * static_cast<int64_t>(mConfig.audioFeatureDim) * static_cast<int64_t>(sizeof(__half));

    bool status{true};
    status &= mAudioContext->setInputShape("input_features", mInputFeatures.getShape().getTRTDims());
    status &= mAudioContext->setTensorAddress("last_hidden_state", embeddingBase + destByteOffset);
    if (!status)
    {
        LOG_ERROR("encodeSingleClip: Failed to set encoder bindings");
        return false;
    }

    if (!mAudioContext->enqueueV3(stream))
    {
        LOG_ERROR("encodeSingleClip: Failed to enqueue encoder");
        return false;
    }

    mMultimodalMetrics.recordRun(1, encodedSeqLen);
    encodedRowsOut = encodedSeqLen;
    return true;
}

bool NemotronOmniAudioRunner::encodeAllClips(
    rt::LLMGenerationRequest const& request, std::vector<int64_t>& audioTokenLengths, cudaStream_t stream)
{
    // Two-pass: load every clip's mel spectrogram first so we can size
    // mAudioEmbedding once before any enqueueV3 binds an output address.
    // Mid-loop reallocation would invalidate addresses set on prior clips
    // whose enqueueV3 may not yet have run on the stream.
    std::vector<rt::Tensor> mels;
    int64_t totalEncodedRows = 0;
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };

    for (auto const& req : request.requests)
    {
        for (auto const& audio : req.audioBuffers)
        {
            if (audio.melSpectrogramPath.empty())
            {
                LOG_ERROR("Nemotron-Omni audio runner requires mel-spectrogram input (melSpectrogramPath)");
                return false;
            }
            rt::Tensor mel;
            if (!loadMelSpectrogramFromFile(audio.melSpectrogramPath, audio.melSpectrogramFormat, mel, stream))
            {
                LOG_ERROR("Failed to load mel-spectrogram from %s", audio.melSpectrogramPath.c_str());
                return false;
            }
            int64_t const encodedSeqLen
                = alignUp(mel.getShape()[1], mConfig.subsamplingFactor) / mConfig.subsamplingFactor;
            audioTokenLengths.push_back(encodedSeqLen);
            totalEncodedRows += encodedSeqLen;
            mels.emplace_back(std::move(mel));
        }
    }

    if (totalEncodedRows == 0)
    {
        // No audio in the batch: leave a 0-row view so downstream multimodal
        // scatter has nothing to inject.
        return mAudioEmbedding.reshape({0, static_cast<int64_t>(mConfig.audioFeatureDim)});
    }

    if (!resizeEmbeddingForRows(totalEncodedRows))
    {
        LOG_ERROR("Failed to size mAudioEmbedding for %ld rows", totalEncodedRows);
        return false;
    }

    int64_t rowOffset = 0;
    for (auto const& mel : mels)
    {
        int64_t encodedRows = 0;
        if (!encodeSingleClip(mel, rowOffset, encodedRows, stream))
        {
            return false;
        }
        rowOffset += encodedRows;
    }
    return true;
}

void NemotronOmniAudioRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& audioTokenLengths,
    tokenizer::Tokenizer const* tokenizer)
{
    // Repeat each ``<so_embedding>`` placeholder ``audioTokenLengths[i]``
    // times so the runtime multimodal kernel can inject one audio frame per
    // copy. Reuse ``batchInputIds`` if the ViT runner already tokenized
    // (combined image+audio); otherwise tokenize from scratch.
    bool const alreadyTokenized = batchInputIds.size() == request.requests.size();
    int audioIndex = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids = alreadyTokenized
            ? std::move(batchInputIds[i])
            : tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        check::check(!ids.empty(), "Failed to encode text");

        std::vector<int32_t> newIds;
        newIds.reserve(ids.size());
        for (auto const& id : ids)
        {
            if (id == mConfig.soundContextTokenId)
            {
                int64_t const numAudioTokens = audioTokenLengths.at(audioIndex);
                for (int64_t k = 0; k < numAudioTokens; ++k)
                {
                    newIds.push_back(mConfig.soundContextTokenId);
                }
                ++audioIndex;
            }
            else
            {
                newIds.push_back(id);
            }
        }
        if (alreadyTokenized)
        {
            batchInputIds[i] = std::move(newIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(newIds));
        }
    }
}

bool NemotronOmniAudioRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::OptionalOutputTensor mropeCosSinOut, cudaStream_t stream, [[maybe_unused]] bool imageOnly)
{
    std::vector<int64_t> audioTokenLengths;

    try
    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);
        if (!encodeAllClips(request, audioTokenLengths, stream))
        {
            return false;
        }
        textPreprocess(request, batchedInputIds, audioTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed: %s", e.what());
        return false;
    }

    return true;
}

bool NemotronOmniAudioRunner::infer([[maybe_unused]] cudaStream_t stream)
{
    // Encoder is invoked per-clip from preprocess() above; nothing left to
    // do here. The base interface still requires this method, so keep it
    // as an explicit no-op rather than removing the override.
    return true;
}

rt::Tensor& NemotronOmniAudioRunner::getOutputEmbedding()
{
    return mAudioEmbedding;
}

} // namespace rt
} // namespace trt_edgellm
