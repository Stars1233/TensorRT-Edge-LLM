/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gptqActivationPermutation.h"

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kThreads = 256;
constexpr int32_t kWarps = kThreads / 32;

__global__ void gptqActivationPermutationKernel(
    int32_t const* groupIndex, int32_t* permutation, int32_t K, int32_t groupSize)
{
    int32_t const group = static_cast<int32_t>(blockIdx.x);
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    int32_t const lane = thread % 32;
    int32_t const warp = thread / 32;
    __shared__ int32_t warpMatches[kWarps];
    __shared__ int32_t groupMatches;

    if (thread == 0)
    {
        groupMatches = 0;
    }
    __syncthreads();

    for (int32_t base = 0; base < K; base += kThreads)
    {
        int32_t const source = base + thread;
        bool const matches = source < K && groupIndex[source] == group;
        uint32_t const mask = __ballot_sync(0xFFFFFFFFU, matches);
        if (lane == 0)
        {
            warpMatches[warp] = __popc(mask);
        }
        __syncthreads();

        int32_t priorMatches = groupMatches;
        for (int32_t priorWarp = 0; priorWarp < warp; ++priorWarp)
        {
            priorMatches += warpMatches[priorWarp];
        }
        uint32_t const lowerLanes = lane == 0 ? 0U : ((1U << lane) - 1U);
        int32_t const destination = priorMatches + __popc(mask & lowerLanes);
        if (matches && destination < groupSize)
        {
            permutation[static_cast<size_t>(group) * groupSize + destination] = source;
        }
        __syncthreads();

        if (thread == 0)
        {
            for (int32_t currentWarp = 0; currentWarp < kWarps; ++currentWarp)
            {
                groupMatches += warpMatches[currentWarp];
            }
        }
        __syncthreads();
    }
}

} // namespace

cudaError_t launchGptqActivationPermutation(int32_t const* groupIndex, int32_t* permutation, int32_t K,
    int32_t numGroups, int32_t groupSize, cudaStream_t stream)
{
    if (groupIndex == nullptr || permutation == nullptr || K <= 0 || numGroups <= 0 || groupSize <= 0
        || numGroups * groupSize != K)
    {
        return cudaErrorInvalidValue;
    }
    gptqActivationPermutationKernel<<<numGroups, kThreads, 0, stream>>>(groupIndex, permutation, K, groupSize);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
