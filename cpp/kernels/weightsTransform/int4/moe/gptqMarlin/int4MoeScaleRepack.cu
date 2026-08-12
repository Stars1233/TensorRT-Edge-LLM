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

#include "int4MoeScaleRepack.h"
#include "kernels/weightsTransform/common/checkpointSourceBatch.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__constant__ int32_t cScalePermutation[64] = {0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26,
    34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14,
    22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};

__global__ void int4MoeScaleRepackSourceBatchKernel(CheckpointSourceBatch<uint16_t> firstSources,
    CheckpointSourceBatch<uint16_t> secondSources, uint16_t* marlin, int32_t E, int32_t G, int32_t projectionN,
    int32_t paired)
{
    int32_t const N = paired ? 2 * projectionN : projectionN;
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const count = static_cast<int64_t>(E) * G * N;
    if (index >= count)
    {
        return;
    }
    int32_t const n = static_cast<int32_t>(index % N);
    int64_t const row = index / N;
    int32_t const expert = static_cast<int32_t>(row / G);
    int32_t const group = static_cast<int32_t>(row % G);
    bool const useSecond = paired && n >= projectionN;
    int32_t const sourceN = useSecond ? n - projectionN : n;
    uint16_t const* source = useSecond ? secondSources.get(expert) : firstSources.get(expert);
    int32_t const outputN = (n / 64) * 64 + cScalePermutation[n % 64];
    marlin[row * N + outputN] = source[static_cast<size_t>(group) * projectionN + sourceN];
}

} // namespace

cudaError_t launchInt4MoeScaleRepackSourceBatch(uint16_t const* const* firstSources,
    uint16_t const* const* secondSources, uint16_t* marlinOutput, int32_t count, int32_t G, int32_t projectionN,
    cudaStream_t stream)
{
    bool const paired = secondSources != nullptr;
    int32_t const N = paired ? 2 * projectionN : projectionN;
    if (firstSources == nullptr || marlinOutput == nullptr || count <= 0 || count > kCheckpointSourcesPerLaunch
        || G <= 0 || projectionN <= 0 || (N % 64) != 0)
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
    int64_t const elements = static_cast<int64_t>(count) * G * N;
    int32_t constexpr threads = 256;
    int32_t const blocks = static_cast<int32_t>((elements + threads - 1) / threads);
    int4MoeScaleRepackSourceBatchKernel<<<blocks, threads, 0, stream>>>(makeCheckpointSourceBatch(firstSources, count),
        makeCheckpointSourceBatch(paired ? secondSources : nullptr, paired ? count : 0), marlinOutput, count, G,
        projectionN, paired ? 1 : 0);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
