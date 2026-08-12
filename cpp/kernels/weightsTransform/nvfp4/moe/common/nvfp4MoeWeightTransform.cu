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

#include "kernels/weightsTransform/common/checkpointSourceBatch.h"
#include "nvfp4MoeNormalize.cuh"
#include "nvfp4MoeWeightTransform.h"

#include <algorithm>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kSwigluInterleaveRows = 64;

template <typename T>
__global__ void checkpointSourceBatchCopyKernel(
    CheckpointSourceBatch<uint8_t> sources, T* output, int32_t count, int64_t elementsPerSource)
{
    int64_t const totalElements = static_cast<int64_t>(count) * elementsPerSource;
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < totalElements; index += stride)
    {
        int32_t const sourceIndex = static_cast<int32_t>(index / elementsPerSource);
        auto const* source = reinterpret_cast<T const*>(sources.get(sourceIndex));
        output[index] = source[index - static_cast<int64_t>(sourceIndex) * elementsPerSource];
    }
}

template <typename T>
__global__ void checkpointSourceBatchPaddedCopyKernel(CheckpointSourceBatch<uint8_t> sources, T* output, int32_t count,
    int32_t sourceRows, int32_t sourceRowElements, int32_t outputRows, int32_t outputRowElements)
{
    int64_t const outputElementsPerSource = static_cast<int64_t>(outputRows) * outputRowElements;
    int64_t const totalElements = static_cast<int64_t>(count) * outputElementsPerSource;
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < totalElements; index += stride)
    {
        int32_t const sourceIndex = static_cast<int32_t>(index / outputElementsPerSource);
        int64_t const within = index - static_cast<int64_t>(sourceIndex) * outputElementsPerSource;
        int32_t const row = static_cast<int32_t>(within / outputRowElements);
        int32_t const column = static_cast<int32_t>(within - static_cast<int64_t>(row) * outputRowElements);
        T value{};
        if (row < sourceRows && column < sourceRowElements)
        {
            auto const* source = reinterpret_cast<T const*>(sources.get(sourceIndex));
            value = source[static_cast<int64_t>(row) * sourceRowElements + column];
        }
        output[index] = value;
    }
}

template <typename T>
__global__ void nvfp4Fc1InterleaveSourceBatchKernel(CheckpointSourceBatch<uint8_t> upSources,
    CheckpointSourceBatch<uint8_t> gateSources, T* out, int32_t E, int32_t sourceI, int32_t outputI,
    int32_t rowElements, int32_t concatLayout)
{
    int64_t const outExpertElements = static_cast<int64_t>(2 * outputI) * rowElements;
    int64_t const totalOutElements = static_cast<int64_t>(E) * outExpertElements;
    int64_t outputElement = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; outputElement < totalOutElements; outputElement += stride)
    {
        int32_t const expert = static_cast<int32_t>(outputElement / outExpertElements);
        int64_t const within = outputElement - static_cast<int64_t>(expert) * outExpertElements;
        int32_t const outRow = static_cast<int32_t>(within / rowElements);
        int32_t const column = static_cast<int32_t>(within - static_cast<int64_t>(outRow) * rowElements);

        int32_t sourceRow;
        T const* source;
        if (concatLayout)
        {
            bool const gate = outRow >= outputI;
            sourceRow = gate ? outRow - outputI : outRow;
            source = reinterpret_cast<T const*>(gate ? gateSources.get(expert) : upSources.get(expert));
        }
        else
        {
            int32_t const local = outRow % (2 * kSwigluInterleaveRows);
            bool const gate = local >= kSwigluInterleaveRows;
            sourceRow = (outRow / (2 * kSwigluInterleaveRows)) * kSwigluInterleaveRows
                + (gate ? local - kSwigluInterleaveRows : local);
            source = reinterpret_cast<T const*>(gate ? gateSources.get(expert) : upSources.get(expert));
        }
        out[outputElement] = sourceRow < sourceI ? source[static_cast<int64_t>(sourceRow) * rowElements + column] : T{};
    }
}

__global__ void nvfp4MoeWeightNormalizeSourceBatchKernel(CheckpointSourceBatch<uint8_t> firstWeights,
    CheckpointSourceBatch<uint8_t> secondWeights, CheckpointSourceBatch<uint8_t> firstScales,
    CheckpointSourceBatch<uint8_t> secondScales, detail::ExpertFloatBatch firstAlphas,
    detail::ExpertFloatBatch secondAlphas, uint8_t* output, int32_t count, int32_t sourceRows, int32_t sourceRowBytes,
    int32_t sourceScaleColumns, int32_t outputRows, int32_t outputRowBytes, int32_t paired, int32_t concatenated)
{
    int32_t const outputScaleColumns = outputRowBytes / 8;
    int64_t const blocksPerExpert = static_cast<int64_t>(outputRows) * outputScaleColumns;
    int64_t const totalBlocks = static_cast<int64_t>(count) * blocksPerExpert;
    int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; block < totalBlocks; block += stride)
    {
        int32_t const expert = static_cast<int32_t>(block / blocksPerExpert);
        int64_t const withinExpert = block - static_cast<int64_t>(expert) * blocksPerExpert;
        int32_t const outputRow = static_cast<int32_t>(withinExpert / outputScaleColumns);
        int32_t const scaleColumn
            = static_cast<int32_t>(withinExpert - static_cast<int64_t>(outputRow) * outputScaleColumns);
        uint8_t* outputBlock = output + (static_cast<int64_t>(expert) * outputRows + outputRow) * outputRowBytes
            + static_cast<int64_t>(scaleColumn) * 8;

        detail::ProjectionRow const projection
            = detail::mapProjectionRow(outputRow, outputRows, paired != 0, concatenated != 0);
        if (projection.row >= sourceRows || scaleColumn >= sourceScaleColumns)
        {
#pragma unroll
            for (int32_t byte = 0; byte < 8; ++byte)
            {
                outputBlock[byte] = 0;
            }
            continue;
        }

        uint8_t const* weight = projection.second ? secondWeights.get(expert) : firstWeights.get(expert);
        uint8_t const* scale = projection.second ? secondScales.get(expert) : firstScales.get(expert);
        float const alpha = projection.second ? secondAlphas.get(expert) : firstAlphas.get(expert);
        uint8_t const* sourceBlock
            = weight + static_cast<int64_t>(projection.row) * sourceRowBytes + static_cast<int64_t>(scaleColumn) * 8;
        uint8_t const sourceScale = scale[static_cast<int64_t>(projection.row) * sourceScaleColumns + scaleColumn];
        float const sourceMultiplier = detail::decodeFp8(sourceScale) * alpha;
        float const normalizedScale = detail::normalizedBlockScale(sourceBlock, sourceScale, alpha);
#pragma unroll
        for (int32_t byte = 0; byte < 8; ++byte)
        {
            outputBlock[byte] = detail::normalizePackedFp4(sourceBlock[byte], sourceMultiplier, normalizedScale);
        }
    }
}

} // namespace

cudaError_t launchCheckpointSourceBatchCopy(
    uint8_t const* const* sources, int32_t count, uint8_t* output, int64_t bytesPerSource, cudaStream_t stream)
{
    if (sources == nullptr || output == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch
        || bytesPerSource <= 0)
    {
        return cudaErrorInvalidValue;
    }
    uintptr_t alignmentBits = reinterpret_cast<uintptr_t>(output) | static_cast<uintptr_t>(bytesPerSource);
    for (int32_t index = 0; index < count; ++index)
    {
        if (sources[index] == nullptr)
        {
            return cudaErrorInvalidValue;
        }
        alignmentBits |= reinterpret_cast<uintptr_t>(sources[index]);
    }
    bool const uint2Aligned = alignmentBits % alignof(uint2) == 0;
    bool const wordAligned = alignmentBits % alignof(uint32_t) == 0;
    size_t const elementBytes = uint2Aligned ? sizeof(uint2) : (wordAligned ? sizeof(uint32_t) : 1);
    int64_t const elementsPerSource = bytesPerSource / static_cast<int64_t>(elementBytes);
    int64_t const totalElements = static_cast<int64_t>(count) * elementsPerSource;
    int32_t constexpr threads = 256;
    int32_t const blocks
        = static_cast<int32_t>(std::min<int64_t>((totalElements + threads - 1) / threads, static_cast<int64_t>(65535)));
    auto const sourcePointers = makeCheckpointSourceBatch(sources, count);
    if (uint2Aligned)
    {
        checkpointSourceBatchCopyKernel<<<blocks, threads, 0, stream>>>(
            sourcePointers, reinterpret_cast<uint2*>(output), count, elementsPerSource);
    }
    else if (wordAligned)
    {
        checkpointSourceBatchCopyKernel<<<blocks, threads, 0, stream>>>(
            sourcePointers, reinterpret_cast<uint32_t*>(output), count, elementsPerSource);
    }
    else
    {
        checkpointSourceBatchCopyKernel<<<blocks, threads, 0, stream>>>(
            sourcePointers, output, count, elementsPerSource);
    }
    return cudaGetLastError();
}

cudaError_t launchCheckpointSourceBatchPaddedCopy(uint8_t const* const* sources, int32_t count, uint8_t* output,
    int32_t sourceRows, int32_t sourceColumns, int32_t outputRows, int32_t outputColumns, cudaStream_t stream)
{
    if (sources == nullptr || output == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch || sourceRows <= 0
        || sourceColumns <= 0 || outputRows < sourceRows || outputColumns < sourceColumns)
    {
        return cudaErrorInvalidValue;
    }

    uintptr_t alignmentBits
        = reinterpret_cast<uintptr_t>(output) | static_cast<uintptr_t>(sourceColumns | outputColumns);
    for (int32_t index = 0; index < count; ++index)
    {
        if (sources[index] == nullptr)
        {
            return cudaErrorInvalidValue;
        }
        alignmentBits |= reinterpret_cast<uintptr_t>(sources[index]);
    }
    bool const uint2Aligned = alignmentBits % alignof(uint2) == 0;
    bool const wordAligned = alignmentBits % alignof(uint32_t) == 0;
    int32_t const elementBytes = uint2Aligned ? sizeof(uint2) : (wordAligned ? sizeof(uint32_t) : 1);
    int32_t const sourceRowElements = sourceColumns / elementBytes;
    int32_t const outputRowElements = outputColumns / elementBytes;
    int64_t const totalElements = static_cast<int64_t>(count) * outputRows * outputRowElements;
    int32_t constexpr threads = 256;
    int32_t const blocks
        = static_cast<int32_t>(std::min<int64_t>((totalElements + threads - 1) / threads, static_cast<int64_t>(65535)));
    auto const sourcePointers = makeCheckpointSourceBatch(sources, count);
    if (uint2Aligned)
    {
        checkpointSourceBatchPaddedCopyKernel<<<blocks, threads, 0, stream>>>(sourcePointers,
            reinterpret_cast<uint2*>(output), count, sourceRows, sourceRowElements, outputRows, outputRowElements);
    }
    else if (wordAligned)
    {
        checkpointSourceBatchPaddedCopyKernel<<<blocks, threads, 0, stream>>>(sourcePointers,
            reinterpret_cast<uint32_t*>(output), count, sourceRows, sourceRowElements, outputRows, outputRowElements);
    }
    else
    {
        checkpointSourceBatchPaddedCopyKernel<<<blocks, threads, 0, stream>>>(
            sourcePointers, output, count, sourceRows, sourceColumns, outputRows, outputColumns);
    }
    return cudaGetLastError();
}

cudaError_t launchNvfp4MoeFc1WeightTransformSourceBatch(uint8_t const* const* upSources,
    uint8_t const* const* gateSources, int32_t count, uint8_t* output, int32_t I, int32_t K, int32_t elementBytes,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream)
{
    return launchNvfp4MoeFc1WeightTransformSourceBatchPadded(
        upSources, gateSources, count, output, I, I, K, elementBytes, layout, stream);
}

cudaError_t launchNvfp4MoeFc1WeightTransformSourceBatchPadded(uint8_t const* const* upSources,
    uint8_t const* const* gateSources, int32_t count, uint8_t* output, int32_t sourceI, int32_t outputI, int32_t K,
    int32_t elementBytes, Nvfp4MoeFc1Layout layout, cudaStream_t stream)
{
    bool const concatLayout = layout == Nvfp4MoeFc1Layout::kConcatenated;
    if (upSources == nullptr || gateSources == nullptr || output == nullptr || count <= 0
        || count > kCheckpointSourcesPerLaunch || sourceI <= 0 || outputI < sourceI || K <= 0 || elementBytes <= 0
        || (!concatLayout && outputI % kSwigluInterleaveRows != 0))
    {
        return cudaErrorInvalidValue;
    }
    int64_t const rowBytes = static_cast<int64_t>(K) * elementBytes;
    uintptr_t alignmentBits = reinterpret_cast<uintptr_t>(output) | static_cast<uintptr_t>(rowBytes);
    for (int32_t index = 0; index < count; ++index)
    {
        if (upSources[index] == nullptr || gateSources[index] == nullptr)
        {
            return cudaErrorInvalidValue;
        }
        alignmentBits
            |= reinterpret_cast<uintptr_t>(upSources[index]) | reinterpret_cast<uintptr_t>(gateSources[index]);
    }
    bool const uint2Aligned = alignmentBits % alignof(uint2) == 0;
    size_t const kernelElementBytes = uint2Aligned ? sizeof(uint2) : 1;
    int32_t const rowElements = static_cast<int32_t>(rowBytes / static_cast<int64_t>(kernelElementBytes));
    int64_t const totalElements = static_cast<int64_t>(count) * 2 * outputI * rowElements;
    int32_t constexpr threads = 256;
    int32_t const blocks
        = static_cast<int32_t>(std::min<int64_t>((totalElements + threads - 1) / threads, static_cast<int64_t>(65535)));
    auto const upPointers = makeCheckpointSourceBatch(upSources, count);
    auto const gatePointers = makeCheckpointSourceBatch(gateSources, count);
    if (uint2Aligned)
    {
        nvfp4Fc1InterleaveSourceBatchKernel<<<blocks, threads, 0, stream>>>(upPointers, gatePointers,
            reinterpret_cast<uint2*>(output), count, sourceI, outputI, rowElements, concatLayout ? 1 : 0);
    }
    else
    {
        nvfp4Fc1InterleaveSourceBatchKernel<<<blocks, threads, 0, stream>>>(
            upPointers, gatePointers, output, count, sourceI, outputI, rowElements, concatLayout ? 1 : 0);
    }
    return cudaGetLastError();
}

cudaError_t launchNvfp4MoeWeightNormalizeSourceBatchPadded(uint8_t const* const* firstWeights,
    uint8_t const* const* secondWeights, uint8_t const* const* firstScales, uint8_t const* const* secondScales,
    float const* firstAlphas, float const* secondAlphas, int32_t count, uint8_t* output, int32_t sourceRows,
    int32_t sourceRowBytes, int32_t sourceScaleColumns, int32_t outputRows, int32_t outputRowBytes,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream)
{
    bool const paired = secondWeights != nullptr;
    bool const concatenated = layout == Nvfp4MoeFc1Layout::kConcatenated;
    if (firstWeights == nullptr || firstScales == nullptr || firstAlphas == nullptr || output == nullptr || count <= 0
        || count > kCheckpointSourcesPerLaunch || sourceRows <= 0 || sourceScaleColumns <= 0
        || sourceRowBytes != sourceScaleColumns * 8 || outputRows < (paired ? 2 * sourceRows : sourceRows)
        || outputRowBytes < sourceRowBytes || outputRowBytes % 8 != 0
        || (paired
            && (secondScales == nullptr || secondAlphas == nullptr || outputRows % 2 != 0
                || (!concatenated && outputRows % 128 != 0)))
        || (!paired && (secondScales != nullptr || secondAlphas != nullptr)))
    {
        return cudaErrorInvalidValue;
    }
    for (int32_t index = 0; index < count; ++index)
    {
        if (firstWeights[index] == nullptr || firstScales[index] == nullptr
            || (paired && (secondWeights[index] == nullptr || secondScales[index] == nullptr)))
        {
            return cudaErrorInvalidValue;
        }
    }

    int32_t const outputScaleColumns = outputRowBytes / 8;
    int64_t const totalBlocks = static_cast<int64_t>(count) * outputRows * outputScaleColumns;
    int32_t constexpr threads = 256;
    int32_t const blocks
        = static_cast<int32_t>(std::min<int64_t>((totalBlocks + threads - 1) / threads, static_cast<int64_t>(65535)));
    nvfp4MoeWeightNormalizeSourceBatchKernel<<<blocks, threads, 0, stream>>>(
        makeCheckpointSourceBatch(firstWeights, count),
        makeCheckpointSourceBatch(paired ? secondWeights : nullptr, paired ? count : 0),
        makeCheckpointSourceBatch(firstScales, count),
        makeCheckpointSourceBatch(paired ? secondScales : nullptr, paired ? count : 0),
        detail::makeExpertFloatBatch(firstAlphas, count),
        detail::makeExpertFloatBatch(paired ? secondAlphas : nullptr, paired ? count : 0), output, count, sourceRows,
        sourceRowBytes, sourceScaleColumns, outputRows, outputRowBytes, paired ? 1 : 0, concatenated ? 1 : 0);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
