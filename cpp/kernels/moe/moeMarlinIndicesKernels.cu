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

#include "common/cudaUtils.h"
#include "moeMarlinIndicesKernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{

// Build Marlin indices from slot lists
__global__ void buildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize)
{
    int32_t expertId = blockIdx.x;
    if (expertId >= numExperts)
        return;

    int32_t count = slotsPerExpertWorkspace[expertId];
    int32_t paddedCount = paddedCounts[expertId];
    if (paddedCount <= 0)
        return;

    int32_t totalSlots = numTokens * topK;
    int32_t outStart = paddedOffsets[expertId];

    for (int32_t i = threadIdx.x; i < paddedCount; i += blockDim.x)
    {
        if (i < count)
        {
            int32_t slot = slotsByExpertWorkspace[expertId * totalSlots + i];
            sortedTokenIds[outStart + i] = slot;
            topkWeightsFlat[outStart + i] = topkWeights[slot];
        }
        else
        {
            sortedTokenIds[outStart + i] = totalSlots;
            topkWeightsFlat[outStart + i] = 0.0f;
        }
    }

    int32_t numBlocks = paddedCount / moeBlockSize;
    int32_t blockOffset = outStart / moeBlockSize;
    for (int32_t b = threadIdx.x; b < numBlocks; b += blockDim.x)
    {
        expertIds[blockOffset + b] = expertId;
    }
}

// Build the degenerate Marlin routing arrays for a dense (single-expert, topK=1) GEMM.
__global__ void buildDenseMarlinIndicesKernel(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedRows, int32_t moeBlockSize)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const stride = gridDim.x * blockDim.x;

    // sortedTokenIds is the identity map with padded-tail slots masked by the out-of-range sentinel numTokens
    // (== numTokens*topK for topK=1). topkWeights is filled with 1.0f only to keep the shared Marlin signature valid.
    for (int32_t i = idx; i < paddedRows; i += stride)
    {
        sortedTokenIds[i] = i < numTokens ? i : numTokens;
        topkWeights[i] = 1.0f;
    }

    int32_t const numBlocks = paddedRows / moeBlockSize;
    for (int32_t b = idx; b < numBlocks; b += stride)
    {
        expertIds[b] = 0;
    }

    if (idx == 0)
    {
        numTokensPostPadded[0] = paddedRows;
    }
}

namespace
{

constexpr int32_t kAggregateThreadsPerBlock{256};

template <typename T>
struct AggregationTypeTraits;

template <>
struct AggregationTypeTraits<half>
{
    static __device__ float toFloat(half const value)
    {
        return __half2float(value);
    }

    static __device__ half fromFloat(float const value)
    {
        return __float2half(value);
    }
};

template <>
struct AggregationTypeTraits<__nv_bfloat16>
{
    static __device__ float toFloat(__nv_bfloat16 const value)
    {
        return __bfloat162float(value);
    }

    static __device__ __nv_bfloat16 fromFloat(float const value)
    {
        return __float2bfloat16_rn(value);
    }
};

// Aggregate slot outputs back to tokens: sum over topK in slot order.
template <typename T>
__global__ void aggregateSlotOutputsKernel(
    T const* slotOutputs, T* aggregatedOutput, int32_t numTokens, int32_t topK, int32_t outDim)
{
    int32_t const tokenId = blockIdx.x;
    int32_t const dimIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (tokenId >= numTokens || dimIdx >= outDim)
    {
        return;
    }

    float accum = 0.0F;
    int32_t const base = tokenId * topK;
    for (int32_t k = 0; k < topK; ++k)
    {
        int32_t const slot = base + k;
        accum += AggregationTypeTraits<T>::toFloat(slotOutputs[slot * outDim + dimIdx]);
    }

    aggregatedOutput[tokenId * outDim + dimIdx] = AggregationTypeTraits<T>::fromFloat(accum);
}

template <typename T>
void launchAggregateSlotOutputsKernelImpl(
    T const* slotOutputs, T* aggregatedOutput, int32_t numTokens, int32_t topK, int32_t outDim, cudaStream_t stream)
{
    dim3 const grid(numTokens, static_cast<uint32_t>(trt_edgellm::divUp(outDim, kAggregateThreadsPerBlock)));
    dim3 const block(kAggregateThreadsPerBlock);
    aggregateSlotOutputsKernel<T><<<grid, block, 0, stream>>>(slotOutputs, aggregatedOutput, numTokens, topK, outDim);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launchBuildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize, cudaStream_t stream)
{
    buildMarlinIndicesKernel<<<numExperts, 256, 0, stream>>>(slotsByExpertWorkspace, slotsPerExpertWorkspace,
        paddedCounts, paddedOffsets, topkWeights, sortedTokenIds, topkWeightsFlat, expertIds, numTokens, topK,
        numExperts, moeBlockSize);
    CUDA_CHECK(cudaGetLastError());
}

void launchBuildDenseMarlinIndicesKernel(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedRows, int32_t moeBlockSize, cudaStream_t stream)
{
    constexpr int32_t kThreadsPerBlock = 256;
    int32_t const grid = static_cast<int32_t>(trt_edgellm::divUp(std::max(paddedRows, 1), kThreadsPerBlock));
    buildDenseMarlinIndicesKernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        sortedTokenIds, expertIds, numTokensPostPadded, topkWeights, numTokens, paddedRows, moeBlockSize);
    CUDA_CHECK(cudaGetLastError());
}

void launchAggregateSlotOutputsKernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens, int32_t topK,
    int32_t outDim, cudaStream_t stream)
{
    launchAggregateSlotOutputsKernelImpl(
        static_cast<half const*>(slotOutputs), static_cast<half*>(aggregatedOutput), numTokens, topK, outDim, stream);
}

void launchAggregateSlotOutputsBf16Kernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens,
    int32_t topK, int32_t outDim, cudaStream_t stream)
{
    launchAggregateSlotOutputsKernelImpl(static_cast<__nv_bfloat16 const*>(slotOutputs),
        static_cast<__nv_bfloat16*>(aggregatedOutput), numTokens, topK, outDim, stream);
}

} // namespace kernel
} // namespace trt_edgellm
