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
#include "nvfp4MoeScaleTransform.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__global__ void nvfp4MoeScaleTransformSourceBatchKernel(CheckpointSourceBatch<uint8_t> firstSources,
    CheckpointSourceBatch<uint8_t> secondSources, int8_t* mmaLayout, int32_t batch, int32_t sourceRows,
    int32_t sourceKsf, int32_t outputRows, int32_t outputKsf, int32_t mTiles, int32_t kTiles, int32_t concatLayout,
    int32_t paired)
{
    int64_t const globalIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const elementsPerMatrix = static_cast<int64_t>(mTiles) * kTiles * 32 * 4 * 4;
    if (globalIdx >= static_cast<int64_t>(batch) * elementsPerMatrix)
    {
        return;
    }
    int32_t const expert = static_cast<int32_t>(globalIdx / elementsPerMatrix);
    int64_t t = globalIdx % elementsPerMatrix;
    int32_t const i4 = static_cast<int32_t>(t % 4);
    t /= 4;
    int32_t const i3 = static_cast<int32_t>(t % 4);
    t /= 4;
    int32_t const i2 = static_cast<int32_t>(t % 32);
    t /= 32;
    int32_t const kt = static_cast<int32_t>(t % kTiles);
    t /= kTiles;
    int32_t const mt = static_cast<int32_t>(t);
    int32_t const row = mt * 128 + i3 * 32 + i2;
    int32_t const col = kt * 4 + i4;

    uint8_t value = 0;
    if (row < outputRows && col < sourceKsf && col < outputKsf)
    {
        int32_t sourceRow = row;
        uint8_t const* source = firstSources.get(expert);
        if (paired)
        {
            int32_t const outputProjectionRows = outputRows / 2;
            bool useSecond;
            if (concatLayout)
            {
                useSecond = row >= outputProjectionRows;
                sourceRow = useSecond ? row - outputProjectionRows : row;
            }
            else
            {
                int32_t constexpr interleaveRows = 64;
                int32_t const local = row % (2 * interleaveRows);
                useSecond = local >= interleaveRows;
                sourceRow
                    = (row / (2 * interleaveRows)) * interleaveRows + (useSecond ? local - interleaveRows : local);
            }
            source = useSecond ? secondSources.get(expert) : firstSources.get(expert);
        }
        if (sourceRow < sourceRows)
        {
            value = source[static_cast<int64_t>(sourceRow) * sourceKsf + col];
        }
    }
    mmaLayout[globalIdx] = static_cast<int8_t>(value);
}

__global__ void nvfp4MoeScaleNormalizeSourceBatchKernel(CheckpointSourceBatch<uint8_t> firstWeights,
    CheckpointSourceBatch<uint8_t> secondWeights, CheckpointSourceBatch<uint8_t> firstScales,
    CheckpointSourceBatch<uint8_t> secondScales, detail::ExpertFloatBatch firstAlphas,
    detail::ExpertFloatBatch secondAlphas, int8_t* mmaLayout, int32_t batch, int32_t sourceRows, int32_t sourceRowBytes,
    int32_t sourceScaleColumns, int32_t outputRows, int32_t outputScaleColumns, int32_t mTiles, int32_t kTiles,
    int32_t concatenated, int32_t paired)
{
    int64_t const globalIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const elementsPerMatrix = static_cast<int64_t>(mTiles) * kTiles * 32 * 4 * 4;
    if (globalIdx >= static_cast<int64_t>(batch) * elementsPerMatrix)
    {
        return;
    }
    int32_t const expert = static_cast<int32_t>(globalIdx / elementsPerMatrix);
    int64_t offset = globalIdx % elementsPerMatrix;
    int32_t const i4 = static_cast<int32_t>(offset % 4);
    offset /= 4;
    int32_t const i3 = static_cast<int32_t>(offset % 4);
    offset /= 4;
    int32_t const i2 = static_cast<int32_t>(offset % 32);
    offset /= 32;
    int32_t const kt = static_cast<int32_t>(offset % kTiles);
    offset /= kTiles;
    int32_t const mt = static_cast<int32_t>(offset);
    int32_t const outputRow = mt * 128 + i3 * 32 + i2;
    int32_t const scaleColumn = kt * 4 + i4;

    uint8_t value = 0;
    detail::ProjectionRow const projection
        = detail::mapProjectionRow(outputRow, outputRows, paired != 0, concatenated != 0);
    if (outputRow < outputRows && projection.row < sourceRows && scaleColumn < sourceScaleColumns
        && scaleColumn < outputScaleColumns)
    {
        uint8_t const* weight = projection.second ? secondWeights.get(expert) : firstWeights.get(expert);
        uint8_t const* scale = projection.second ? secondScales.get(expert) : firstScales.get(expert);
        float const alpha = projection.second ? secondAlphas.get(expert) : firstAlphas.get(expert);
        uint8_t const* sourceBlock
            = weight + static_cast<int64_t>(projection.row) * sourceRowBytes + static_cast<int64_t>(scaleColumn) * 8;
        uint8_t const sourceScale = scale[static_cast<int64_t>(projection.row) * sourceScaleColumns + scaleColumn];
        value = detail::encodeFp8(detail::normalizedBlockScale(sourceBlock, sourceScale, alpha));
    }
    mmaLayout[globalIdx] = static_cast<int8_t>(value);
}

} // namespace

cudaError_t launchNvfp4MoeScaleTransformSourceBatch(uint8_t const* const* firstSources,
    uint8_t const* const* secondSources, int32_t count, int8_t* mmaLayout, int32_t rows, int32_t Ksf,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream)
{
    int32_t const outputRows = secondSources == nullptr ? rows : 2 * rows;
    return launchNvfp4MoeScaleTransformSourceBatchPadded(
        firstSources, secondSources, count, mmaLayout, rows, Ksf, outputRows, Ksf, layout, stream);
}

cudaError_t launchNvfp4MoeScaleTransformSourceBatchPadded(uint8_t const* const* firstSources,
    uint8_t const* const* secondSources, int32_t count, int8_t* mmaLayout, int32_t sourceRows, int32_t sourceKsf,
    int32_t outputRows, int32_t outputKsf, Nvfp4MoeFc1Layout layout, cudaStream_t stream)
{
    bool const paired = secondSources != nullptr;
    bool const concatLayout = layout == Nvfp4MoeFc1Layout::kConcatenated;
    if (firstSources == nullptr || mmaLayout == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch
        || sourceRows <= 0 || sourceKsf <= 0 || outputRows < (paired ? 2 * sourceRows : sourceRows)
        || outputKsf < sourceKsf || (paired && (outputRows % 2 != 0 || (!concatLayout && outputRows % 128 != 0))))
    {
        return cudaErrorInvalidValue;
    }

    for (int32_t index = 0; index < count; ++index)
    {
        if (firstSources[index] == nullptr || (paired && secondSources[index] == nullptr))
        {
            return cudaErrorInvalidValue;
        }
    }
    int32_t const mTiles = (outputRows + 127) / 128;
    int32_t const kTiles = (outputKsf + 3) / 4;
    int64_t const total = static_cast<int64_t>(count) * mTiles * kTiles * 32 * 4 * 4;
    int32_t constexpr threads = 256;
    int32_t const blocks = static_cast<int32_t>((total + threads - 1) / threads);
    nvfp4MoeScaleTransformSourceBatchKernel<<<blocks, threads, 0, stream>>>(
        makeCheckpointSourceBatch(firstSources, count),
        makeCheckpointSourceBatch(paired ? secondSources : nullptr, paired ? count : 0), mmaLayout, count, sourceRows,
        sourceKsf, outputRows, outputKsf, mTiles, kTiles, concatLayout ? 1 : 0, paired ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t launchNvfp4MoeScaleNormalizeSourceBatchPadded(uint8_t const* const* firstWeights,
    uint8_t const* const* secondWeights, uint8_t const* const* firstScales, uint8_t const* const* secondScales,
    float const* firstAlphas, float const* secondAlphas, int32_t count, int8_t* mmaLayout, int32_t sourceRows,
    int32_t sourceRowBytes, int32_t sourceScaleColumns, int32_t outputRows, int32_t outputScaleColumns,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream)
{
    bool const paired = secondWeights != nullptr;
    bool const concatenated = layout == Nvfp4MoeFc1Layout::kConcatenated;
    if (firstWeights == nullptr || firstScales == nullptr || firstAlphas == nullptr || mmaLayout == nullptr
        || count <= 0 || count > kCheckpointSourcesPerLaunch || sourceRows <= 0 || sourceScaleColumns <= 0
        || sourceRowBytes != sourceScaleColumns * 8 || outputRows < (paired ? 2 * sourceRows : sourceRows)
        || outputScaleColumns < sourceScaleColumns
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

    int32_t const mTiles = (outputRows + 127) / 128;
    int32_t const kTiles = (outputScaleColumns + 3) / 4;
    int64_t const total = static_cast<int64_t>(count) * mTiles * kTiles * 32 * 4 * 4;
    int32_t constexpr threads = 256;
    int32_t const blocks = static_cast<int32_t>((total + threads - 1) / threads);
    nvfp4MoeScaleNormalizeSourceBatchKernel<<<blocks, threads, 0, stream>>>(
        makeCheckpointSourceBatch(firstWeights, count),
        makeCheckpointSourceBatch(paired ? secondWeights : nullptr, paired ? count : 0),
        makeCheckpointSourceBatch(firstScales, count),
        makeCheckpointSourceBatch(paired ? secondScales : nullptr, paired ? count : 0),
        detail::makeExpertFloatBatch(firstAlphas, count),
        detail::makeExpertFloatBatch(paired ? secondAlphas : nullptr, paired ? count : 0), mmaLayout, count, sourceRows,
        sourceRowBytes, sourceScaleColumns, outputRows, outputScaleColumns, mTiles, kTiles, concatenated ? 1 : 0,
        paired ? 1 : 0);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
