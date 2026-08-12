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

#include "argmaxKernel.h"

#include <cfloat>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

constexpr int32_t kArgmaxBlockSize = 256;

//! Combine (max, idx) pairs preferring the higher value, then — on an exact
//! tie — the lower index. This monoid is associative, so any warp/block
//! reduction order yields the same result.
__device__ __forceinline__ void argmaxCombine(float& bestVal, int32_t& bestIdx, float otherVal, int32_t otherIdx)
{
    if (otherVal > bestVal || (otherVal == bestVal && otherIdx < bestIdx))
    {
        bestVal = otherVal;
        bestIdx = otherIdx;
    }
}

template <typename T>
__global__ void rowwiseArgmaxKernel(
    T const* __restrict__ input, int32_t* __restrict__ outIndices, int32_t rows, int32_t cols)
{
    int32_t const row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }
    T const* __restrict__ rowPtr = input + static_cast<int64_t>(row) * cols;

    float localMax = -FLT_MAX;
    int32_t localIdx = 0;
    for (int32_t c = threadIdx.x; c < cols; c += blockDim.x)
    {
        argmaxCombine(localMax, localIdx, static_cast<float>(rowPtr[c]), c);
    }

    // Intra-warp reduction.
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        float const otherMax = __shfl_down_sync(0xFFFFFFFF, localMax, offset);
        int32_t const otherIdx = __shfl_down_sync(0xFFFFFFFF, localIdx, offset);
        argmaxCombine(localMax, localIdx, otherMax, otherIdx);
    }

    __shared__ float sharedMaxValues[32];
    __shared__ int32_t sharedMaxIndices[32];
    int32_t const warpId = threadIdx.x / 32;
    int32_t const laneId = threadIdx.x % 32;
    int32_t const numWarps = (blockDim.x + 31) / 32;
    if (laneId == 0)
    {
        sharedMaxValues[warpId] = localMax;
        sharedMaxIndices[warpId] = localIdx;
    }
    __syncthreads();

    // Reduce the per-warp winners in the first warp.
    if (warpId == 0)
    {
        float warpMax = (laneId < numWarps) ? sharedMaxValues[laneId] : -FLT_MAX;
        int32_t warpIdx = (laneId < numWarps) ? sharedMaxIndices[laneId] : 0;
        for (int32_t offset = 16; offset > 0; offset >>= 1)
        {
            float const otherMax = __shfl_down_sync(0xFFFFFFFF, warpMax, offset);
            int32_t const otherIdx = __shfl_down_sync(0xFFFFFFFF, warpIdx, offset);
            argmaxCombine(warpMax, warpIdx, otherMax, otherIdx);
        }
        if (laneId == 0)
        {
            outIndices[row] = warpIdx;
        }
    }
}

} // namespace

template <typename T>
void invokeRowwiseArgmax(T const* input, int32_t rows, int32_t cols, int32_t* outIndices, cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0)
    {
        return;
    }
    rowwiseArgmaxKernel<T><<<rows, kArgmaxBlockSize, 0, stream>>>(input, outIndices, rows, cols);
}

template void invokeRowwiseArgmax<float>(float const*, int32_t, int32_t, int32_t*, cudaStream_t);
template void invokeRowwiseArgmax<__half>(__half const*, int32_t, int32_t, int32_t*, cudaStream_t);

} // namespace kernel
} // namespace trt_edgellm
