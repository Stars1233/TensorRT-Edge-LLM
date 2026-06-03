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

#include "runtime/llmRuntimeUtils.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "runtime/streaming.h" // For SlotStreamState (explicit compactVector instantiation)
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>

using namespace nvinfer1;
namespace trt_edgellm
{
namespace rt
{

std::ostream& operator<<(std::ostream& os, RopeType const& type)
{
    switch (type)
    {
    case RopeType::kDefault: os << "Default"; break;
    case RopeType::kDynamic: os << "Dynamic"; break;
    case RopeType::kLongRope: os << "LongRope"; break;
    case RopeType::kMRope: os << "MRope"; break;
    case RopeType::kNoRope: os << "NoRope"; break;
    }
    return os;
}

std::string formatRopeConfig(RopeConfig const& config)
{
    std::stringstream ss;
    ss << "RopeConfig:"
       << "  type: " << config.type << "  rotaryScale: " << config.rotaryScale
       << "  rotaryTheta: " << config.rotaryTheta << "  maxPositionEmbeddings: " << config.maxPositionEmbeddings;
    if (config.type == RopeType::kLongRope)
    {
        ss << "LongRopeConfig:"
           << "  originalMaxPositionEmbeddings: " << config.longRope.value().originalMaxPositionEmbeddings;
    }
    return ss.str();
}

RopeConfig collectRopeConfig(nlohmann::json const& config)
{
    RopeConfig ropeConfig{};

    // Check for explicit use_rope flag (set by hybrid model export)
    if (config.contains("use_rope") && config["use_rope"].is_boolean() && !config["use_rope"].get<bool>())
    {
        ropeConfig.type = RopeType::kNoRope;
        if (config.contains("max_position_embeddings"))
        {
            ropeConfig.maxPositionEmbeddings = config["max_position_embeddings"].get<int32_t>();
        }
        LOG_INFO("Collected rope config: %s", formatRopeConfig(ropeConfig).c_str());
        return ropeConfig;
    }

    auto ropeScalingIt = config.find("rope_scaling");
    if (ropeScalingIt != config.end())
    {
        auto ropeTypeIt = ropeScalingIt->find("type");
        if (ropeTypeIt == ropeScalingIt->end())
        {
            ropeTypeIt = ropeScalingIt->find("rope_type");
        }
        auto mropeSectionIt = ropeScalingIt->find("mrope_section");
        if (ropeTypeIt != ropeScalingIt->end())
        {
            std::string const ropeTypeStr = ropeTypeIt->get<std::string>();
            if (ropeTypeStr == "mrope" || (ropeTypeStr == "default" && mropeSectionIt != ropeScalingIt->end()))
            {
                // Accept both the legacy HF value ("mrope") and the newer
                // `Qwen2_5_VLVisionConfig` convention ("default" + mrope_section).
                // Talker uses same config (3D position_ids + interleaved MRoPE) as in PyTorch.
                ropeConfig.type = RopeType::kMRope;
            }
            else if (ropeTypeStr == "default" || ropeTypeStr == "llama3")
            {
                // Route the llama3 config to default type.
                ropeConfig.type = RopeType::kDefault;
            }
            else if (ropeTypeStr == "dynamic")
            {
                ropeConfig.type = RopeType::kDynamic;
            }
            else if (ropeTypeStr == "longrope")
            {
                ropeConfig.type = RopeType::kLongRope;
            }
        }

        // Parse long rope scaling parameters when requested
        if (ropeConfig.type == RopeType::kLongRope)
        {
            LongRopeParams params{};
            auto longFactorIt = ropeScalingIt->find("long_factor");
            check::check((longFactorIt != ropeScalingIt->end() && longFactorIt->is_array()),
                "rope_scaling.long_factor must be a non-empty array for longrope");
            params.longFactor = longFactorIt->get<std::vector<float>>();

            auto shortFactorIt = ropeScalingIt->find("short_factor");
            check::check((shortFactorIt != ropeScalingIt->end() && shortFactorIt->is_array()),
                "rope_scaling.short_factor must be a non-empty array for longrope");
            params.shortFactor = shortFactorIt->get<std::vector<float>>();

            check::check(params.longFactor.size() == params.shortFactor.size(),
                "rope_scaling.long_factor size differs from short_factor size");

            check::check(config.contains("original_max_position_embeddings"),
                "original_max_position_embeddings is not specified in the model config");
            params.originalMaxPositionEmbeddings = config["original_max_position_embeddings"].get<int32_t>();

            ropeConfig.longRope = std::move(params);
        }
    }
    else
    {
        LOG_WARNING(
            "rope_scaling is not specified in the model config, using default rope type. This could misalign with the "
            "model configuration, please check the config file to ensure the correctness");
        ropeConfig.type = RopeType::kDefault;
    }

    // Detect RopeTheta
    if (config.contains("rope_theta"))
    {
        ropeConfig.rotaryTheta = config["rope_theta"].get<float>();
    }
    else
    {
        LOG_WARNING("rope_theta is not specified in the model config, using default value: %f", ropeConfig.rotaryTheta);
    }

    // Detect MaxPositionEmbeddings
    if (config.contains("max_position_embeddings"))
    {
        ropeConfig.maxPositionEmbeddings = config["max_position_embeddings"].get<int32_t>();
    }
    else
    {
        LOG_WARNING("max_position_embeddings is not specified in the model config, using default value: %d",
            ropeConfig.maxPositionEmbeddings);
    }

    LOG_INFO("Collected rope config: %s", formatRopeConfig(ropeConfig).c_str());
    return ropeConfig;
}

bool initializeRopeCosSinCache(rt::Tensor& cosSinCache, RopeConfig const& config, cudaStream_t stream) noexcept
{
    if (config.type == RopeType::kMRope)
    {
        LOG_ERROR("MRope is context dependent rope type, which cannot be initialized with basic parameters.");
        return false;
    }
    else if (config.type == RopeType::kLongRope)
    {
        LOG_ERROR("Please use initializeLongRopeCosSinCache instead.");
        return false;
    }

    // Tensor shape: [1, maxLength, rotaryDim]
    if (cosSinCache.getShape().getNumDims() != 3 || cosSinCache.getDataType() != DataType::kFLOAT)
    {
        LOG_ERROR("Persistent RopeCosSinCache should be float tensor with dimensions: [1, maxLength, rotaryDim].");
        return false;
    }
    int64_t ropeMaxLength = cosSinCache.getShape()[1];
    int64_t rotaryDim = cosSinCache.getShape()[2];
    if (config.type == RopeType::kDefault || config.type == RopeType::kDynamic)
    {
        if (config.type == RopeType::kDynamic && ropeMaxLength > config.maxPositionEmbeddings)
        {
            LOG_ERROR("Dynamic rope type with sequence length larger than maxPositionEmbeddings is not supported.");
            return false;
        }
        if (ropeMaxLength > config.maxPositionEmbeddings)
        {
            LOG_WARNING(
                "maxLength %d is greater than maxPositionEmbeddings %d indicated by model config, this could cause "
                "inaccurate generation results",
                ropeMaxLength, config.maxPositionEmbeddings);
        }

        try
        {
            kernel::initializeNormalRopeCosSin(cosSinCache.dataPointer<float>(), config.rotaryTheta, config.rotaryScale,
                rotaryDim, ropeMaxLength, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("CUDA kernel launch for initializeNormalRopeCosSin failed: %s", e.what());
            return false;
        }
    }
    return true;
}

bool initializeNopeCosSinCache(rt::Tensor& cosSinCache, cudaStream_t stream) noexcept
{
    if (cosSinCache.getShape().getNumDims() != 3 || cosSinCache.getDataType() != DataType::kFLOAT)
    {
        LOG_ERROR("NoRope CosSinCache should be float tensor with dimensions: [1, maxLength, rotaryDim].");
        return false;
    }

    int64_t maxLength = cosSinCache.getShape()[1];
    int64_t rotaryDim = cosSinCache.getShape()[2];
    int64_t halfDim = rotaryDim / 2;

    std::vector<float> hostBuf(static_cast<size_t>(maxLength * rotaryDim));
    for (int64_t pos = 0; pos < maxLength; ++pos)
    {
        for (int64_t d = 0; d < halfDim; ++d)
        {
            hostBuf[pos * rotaryDim + d] = 1.0F;
        }
        for (int64_t d = halfDim; d < rotaryDim; ++d)
        {
            hostBuf[pos * rotaryDim + d] = 0.0F;
        }
    }

    try
    {
        CUDA_CHECK(cudaMemcpyAsync(cosSinCache.dataPointer<float>(), hostBuf.data(), hostBuf.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("cudaMemcpyAsync for initializeNopeCosSinCache failed: %s", e.what());
        return false;
    }
    return true;
}

bool initializeLongRopeCosSinCache(
    rt::Tensor& shortCosSinCache, rt::Tensor& longCosSinCache, RopeConfig const& config, cudaStream_t stream)
{

    if (config.type != RopeType::kLongRope)
    {
        LOG_ERROR("This function is only used for initializing LongRope cos/sin cache.");
        return false;
    }

    // Tensor shape: [1, maxLength, rotaryDim]
    if (shortCosSinCache.getShape().getNumDims() != 3 || shortCosSinCache.getDataType() != DataType::kFLOAT
        || longCosSinCache.getShape().getNumDims() != 3 || longCosSinCache.getDataType() != DataType::kFLOAT)
    {
        LOG_ERROR("Persistent RopeCosSinCache should be float tensor with dimensions: [1, maxLength, rotaryDim].");
        return false;
    }

    int64_t ropeMaxLength = shortCosSinCache.getShape()[1];
    int64_t rotaryDim = shortCosSinCache.getShape()[2];

    // Validate long/short factors before creating tensors and launching the kernel
    if (!config.longRope.has_value() || config.longRope->longFactor.empty() || config.longRope->shortFactor.empty())
    {
        LOG_ERROR("LongRope requires non-empty long_factor and short_factor arrays in rope_scaling.");
        return false;
    }

    Coords shape{static_cast<int64_t>(config.longRope->longFactor.size())};

    Tensor longFactorTensor(shape, DeviceType::kGPU, DataType::kFLOAT, "long_factor");
    Tensor shortFactorTensor(shape, DeviceType::kGPU, DataType::kFLOAT, "short_factor");
    CUDA_CHECK(cudaMemcpyAsync(longFactorTensor.rawPointer(), config.longRope->longFactor.data(),
        longFactorTensor.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(shortFactorTensor.rawPointer(), config.longRope->shortFactor.data(),
        shortFactorTensor.getMemoryCapacity(), cudaMemcpyHostToDevice, stream));
    try
    {
        kernel::initializeLongRopeCosSin(shortCosSinCache.dataPointer<float>(), longCosSinCache.dataPointer<float>(),
            shortFactorTensor.dataPointer<float>(), longFactorTensor.dataPointer<float>(), config.rotaryTheta,
            rotaryDim, ropeMaxLength, config.maxPositionEmbeddings, config.longRope->originalMaxPositionEmbeddings,
            stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("CUDA kernel launch for initializeLongRopeCosSin failed: %s", e.what());
        return false;
    }
    return true;
}

//=============================================================================
// CPU Vector Compaction Utility for Batch Eviction
//=============================================================================

template <typename T>
void compactVector(std::vector<int32_t> const& batchMapping, std::vector<T>& vec)
{
    // Validate that vector size matches batchMapping size
    // In batch eviction, batchMapping[i] indicates where batch i should move to
    ELLM_CHECK(vec.size() == batchMapping.size(),
        format::fmtstr("compactVector: vector size (%zu) does not match batchMapping size (%zu)", vec.size(),
            batchMapping.size()));

    std::vector<T> compacted;
    compacted.reserve(vec.size());

    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (batchMapping[i] >= 0)
        {
            compacted.push_back(std::move(vec[i]));
        }
    }

    vec = std::move(compacted);
}

template void compactVector<int8_t>(std::vector<int32_t> const&, std::vector<int8_t>&);
template void compactVector<int32_t>(std::vector<int32_t> const&, std::vector<int32_t>&);
template void compactVector<std::vector<int32_t>>(std::vector<int32_t> const&, std::vector<std::vector<int32_t>>&);
template void compactVector<std::string>(std::vector<int32_t> const&, std::vector<std::string>&);
template void compactVector<std::vector<std::string>>(
    std::vector<int32_t> const&, std::vector<std::vector<std::string>>&);
template void compactVector<SlotStreamState>(std::vector<int32_t> const&, std::vector<SlotStreamState>&);

// Build batch mapping from finished states
// Returns a vector mapping old batch indices to new indices (-1 for evicted batches)
std::vector<int32_t> buildBatchMapping(std::vector<int8_t> const& finishedStates)
{
    int32_t const oldActiveBatch = static_cast<int32_t>(finishedStates.size());
    std::vector<int32_t> mapping(oldActiveBatch);

    int32_t newIdx = 0;
    for (int32_t oldIdx = 0; oldIdx < oldActiveBatch; ++oldIdx)
    {
        if (!finishedStates[oldIdx])
        {
            mapping[oldIdx] = newIdx;
            newIdx++;
        }
        else
        {
            mapping[oldIdx] = -1; // Mark for eviction
        }
    }

    return mapping;
}

//=============================================================================
// Embedding Loading Utilities
//=============================================================================

EmbeddingData loadEmbeddingTable(std::filesystem::path const& embeddingPath, cudaStream_t stream)
{
    ELLM_CHECK(std::filesystem::exists(embeddingPath),
        format::fmtstr("Embedding file not found: %s", embeddingPath.string().c_str()));

    std::vector<rt::Tensor> tensors;
    ELLM_CHECK(safetensors::loadSafetensors(embeddingPath, tensors, stream),
        format::fmtstr("Failed to load embedding file: %s", embeddingPath.string().c_str()));

    // Find tensors by name
    rt::Tensor* embeddingPtr = nullptr;
    rt::Tensor* scalesPtr = nullptr;

    for (auto& tensor : tensors)
    {
        if (tensor.getName() == "embedding")
        {
            embeddingPtr = &tensor;
        }
        else if (tensor.getName() == "embedding_scale")
        {
            scalesPtr = &tensor;
        }
    }

    ELLM_CHECK(embeddingPtr != nullptr,
        format::fmtstr("Embedding file missing 'embedding' tensor: %s", embeddingPath.string().c_str()));

    ELLM_CHECK(embeddingPtr->getShape().getNumDims() == 2,
        format::fmtstr("Embedding tensor must be 2D, got %d dimensions", embeddingPtr->getShape().getNumDims()));

    int64_t vocabSize = embeddingPtr->getShape()[0];
    int64_t hiddenSize = embeddingPtr->getShape()[1];

    EmbeddingData result;

    // Detect FP8 vs FP16 by checking dtype
    if (embeddingPtr->getDataType() == nvinfer1::DataType::kFP8)
    {
        // FP8 format - requires scales
        ELLM_CHECK(scalesPtr != nullptr,
            format::fmtstr("FP8 embedding requires 'embedding_scale' tensor: %s", embeddingPath.string().c_str()));

        ELLM_CHECK(scalesPtr->getDataType() == nvinfer1::DataType::kFLOAT,
            format::fmtstr("embedding_scale must have FP32 dtype, got %d", static_cast<int>(scalesPtr->getDataType())));

        ELLM_CHECK(scalesPtr->getShape().getNumDims() == 2,
            format::fmtstr("embedding_scale must be 2D, got %d dimensions", scalesPtr->getShape().getNumDims()));

        int64_t scaleVocabSize = scalesPtr->getShape()[0];
        int64_t numGroups = scalesPtr->getShape()[1];

        ELLM_CHECK(vocabSize == scaleVocabSize,
            format::fmtstr("Vocab size mismatch: embedding has %ld, scales has %ld", vocabSize, scaleVocabSize));

        ELLM_CHECK(hiddenSize % kFP8EmbeddingBlockSize == 0,
            format::fmtstr("Hidden size %ld must be divisible by block size %ld", hiddenSize, kFP8EmbeddingBlockSize));

        int64_t expectedNumGroups = hiddenSize / kFP8EmbeddingBlockSize;
        ELLM_CHECK(numGroups == expectedNumGroups,
            format::fmtstr("Scale groups mismatch: expected %ld, got %ld", expectedNumGroups, numGroups));

        LOG_INFO(
            "Loaded FP8 embedding: [%ld, %ld], scales: [%ld, %ld]", vocabSize, hiddenSize, scaleVocabSize, numGroups);

        result.table = std::move(*embeddingPtr);
        result.tableScalingFactor = std::move(*scalesPtr);
    }
    else
    {
        // FP16 format
        LOG_INFO("Loaded FP16 embedding: [%ld, %ld]", vocabSize, hiddenSize);

        result.table = std::move(*embeddingPtr);
    }

    return result;
}

int32_t clampMaxGenerateLengthForKVCapacity(std::vector<int32_t> const& effectivePrefillLengths,
    int32_t requestedMaxGenerateLength, int32_t kvCacheCapacity, int32_t kvCacheReserveLength)
{
    check::check(!effectivePrefillLengths.empty(), "effectivePrefillLengths must not be empty");

    int32_t clampedMaxGenerateLength = requestedMaxGenerateLength;
    for (int32_t const prefillLength : effectivePrefillLengths)
    {
        int32_t const availableGenerateLength = std::max(0, kvCacheCapacity - prefillLength - kvCacheReserveLength);
        clampedMaxGenerateLength = std::min(clampedMaxGenerateLength, availableGenerateLength);
    }

    return clampedMaxGenerateLength;
}

rt::Tensor generateMultimodalIndices(rt::Tensor const& inputIds, std::optional<int32_t> audioTokenId,
    std::optional<int32_t> imageTokenId, int32_t vocabSize)
{
    auto const shape = inputIds.getShape();
    check::check(shape.getNumDims() == 2, "inputIds must be 2D tensor");
    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];

    rt::Tensor multimodalIndices({batchSize, seqLen}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);

    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    int32_t* indicesPtr = multimodalIndices.dataPointer<int32_t>();

    int32_t audioIndex = 0;
    int32_t imageIndex = 0;

    for (int64_t b = 0; b < batchSize; ++b)
    {
        for (int64_t s = 0; s < seqLen; ++s)
        {
            int64_t const pos = b * seqLen + s;
            int32_t const tokenId = inputIdsPtr[pos];

            if (audioTokenId.has_value() && tokenId == *audioTokenId)
            {
                indicesPtr[pos] = audioIndex++;
            }
            else if ((imageTokenId.has_value() && tokenId == *imageTokenId) || tokenId >= vocabSize)
            {
                indicesPtr[pos] = imageIndex++;
            }
            else
            {
                indicesPtr[pos] = 0;
            }
        }
    }

    return multimodalIndices;
}

} // namespace rt
} // namespace trt_edgellm
