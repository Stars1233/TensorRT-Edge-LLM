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

#include "f16MoeSupportKernels.h"

#if defined(CUTE_DSL_F16_MOE_ENABLED)

#include "common/cudaUtils.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kACT_SWIGLU{2};
constexpr int32_t kACT_RELU2{4};
constexpr int32_t kHALF_PER_VECTOR{8};
constexpr int32_t kSWIGLU_CHUNK_ROWS{64};
constexpr int32_t kSWIGLU_STORAGE_ROWS{2 * kSWIGLU_CHUNK_ROWS};
constexpr int32_t kTHREADS_PER_BLOCK{256};
constexpr int32_t kMAX_TOP_K{8};
constexpr int32_t kMAX_FUSED_ROUTING_TOKENS{kTHREADS_PER_BLOCK};

uint32_t gridSize(int64_t elements)
{
    return static_cast<uint32_t>(divUp(elements, static_cast<int64_t>(kTHREADS_PER_BLOCK)));
}

bool isValidGemmSetup(F16MoeGemmSetup const& setup) noexcept
{
    return setup.metadata.problemShapes != nullptr && setup.metadata.strides != nullptr
        && setup.metadata.addresses != nullptr && setup.input != nullptr && setup.weights != nullptr
        && setup.output != nullptr && setup.n > 0 && setup.k > 0;
}

__global__ void countRoutesKernel(int32_t const* topkIds, int32_t* expertCounts, int32_t routedRows, int32_t numExperts)
{
    int32_t const route = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (route < routedRows)
    {
        int32_t const expert = topkIds[route];
        if (expert >= 0 && expert < numExperts)
        {
            atomicAdd(expertCounts + expert, 1);
        }
    }
}

__device__ void populateGemmMetadata(F16MoeGemmSetup const& setup, int32_t expert, int32_t rowOffset, int32_t rows)
{
    int32_t const shapeBase = expert * 4;
    setup.metadata.problemShapes[shapeBase] = rows;
    setup.metadata.problemShapes[shapeBase + 1] = setup.n;
    setup.metadata.problemShapes[shapeBase + 2] = setup.k;
    setup.metadata.problemShapes[shapeBase + 3] = 1;

    int32_t const strideBase = expert * 3 * 2;
    setup.metadata.strides[strideBase] = setup.k;
    setup.metadata.strides[strideBase + 1] = 1;
    setup.metadata.strides[strideBase + 2] = setup.k;
    setup.metadata.strides[strideBase + 3] = 1;
    setup.metadata.strides[strideBase + 4] = setup.n;
    setup.metadata.strides[strideBase + 5] = 1;

    auto const* input = static_cast<__half const*>(setup.input);
    auto const* weights = static_cast<__half const*>(setup.weights);
    auto* output = static_cast<__half*>(setup.output);
    int32_t const addressBase = expert * 3;
    setup.metadata.addresses[addressBase]
        = reinterpret_cast<int64_t>(input + static_cast<int64_t>(rowOffset) * setup.k);
    setup.metadata.addresses[addressBase + 1]
        = reinterpret_cast<int64_t>(weights + static_cast<int64_t>(expert) * setup.n * setup.k);
    setup.metadata.addresses[addressBase + 2]
        = reinterpret_cast<int64_t>(output + static_cast<int64_t>(rowOffset) * setup.n);
}

template <int32_t TOP_K>
__global__ void fusedBuildRoutingAndGemmMetadataKernel(F16MoeRoutingBuffers buffers, F16MoeGemmSetup fc1Setup,
    F16MoeGemmSetup fc2Setup, int32_t const* topkIds, int32_t numTokens, int32_t numExperts)
{
    __shared__ int32_t sharedExpertCounts[kTHREADS_PER_BLOCK];
    __shared__ int32_t sharedExpertWriteOffsets[kTHREADS_PER_BLOCK];

    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    if (thread < numExperts)
    {
        sharedExpertCounts[thread] = 0;
    }
    __syncthreads();

    int32_t const token = thread;
    if (token < numTokens)
    {
#pragma unroll
        for (int32_t slot = 0; slot < TOP_K; ++slot)
        {
            int32_t const expandedRow = token * TOP_K + slot;
            int32_t const expert = topkIds[expandedRow];
            if (expert >= 0 && expert < numExperts)
            {
                atomicAdd(sharedExpertCounts + expert, 1);
            }
        }
    }
    __syncthreads();

    if (thread == 0)
    {
        int32_t prefix{0};
        buffers.expertOffsets[0] = 0;
        for (int32_t expert = 0; expert < numExperts; ++expert)
        {
            prefix += sharedExpertCounts[expert];
            buffers.expertOffsets[expert + 1] = prefix;
        }
    }
    __syncthreads();

    if (thread < numExperts)
    {
        int32_t const rowOffset = buffers.expertOffsets[thread];
        int32_t const rows = sharedExpertCounts[thread];
        buffers.expertCounts[thread] = rows;
        sharedExpertWriteOffsets[thread] = rowOffset;
        populateGemmMetadata(fc1Setup, thread, rowOffset, rows);
        populateGemmMetadata(fc2Setup, thread, rowOffset, rows);
    }
    __syncthreads();

    if (token < numTokens)
    {
#pragma unroll
        for (int32_t slot = 0; slot < TOP_K; ++slot)
        {
            int32_t const expandedRow = token * TOP_K + slot;
            int32_t const expert = topkIds[expandedRow];
            if (expert >= 0 && expert < numExperts)
            {
                int32_t const sortedRow = atomicAdd(sharedExpertWriteOffsets + expert, 1);
                buffers.sortedToExpanded[sortedRow] = expandedRow;
                buffers.expandedToSorted[expandedRow] = sortedRow;
            }
        }
    }
    __syncthreads();

    if (thread < numExperts)
    {
        buffers.expertWriteOffsets[thread] = sharedExpertWriteOffsets[thread];
    }
}

cudaError_t launchFusedBuildRoutingAndGemmMetadata(F16MoeRoutingBuffers const& buffers, F16MoeGemmSetup const& fc1Setup,
    F16MoeGemmSetup const& fc2Setup, int32_t const* topkIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    cudaStream_t stream)
{
    bool launched{true};
    switch (topK)
    {
    case 1:
        fusedBuildRoutingAndGemmMetadataKernel<1>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    case 2:
        fusedBuildRoutingAndGemmMetadataKernel<2>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    case 3:
        fusedBuildRoutingAndGemmMetadataKernel<3>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    case 4:
        fusedBuildRoutingAndGemmMetadataKernel<4>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    case 5:
        fusedBuildRoutingAndGemmMetadataKernel<5>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    case 6:
        fusedBuildRoutingAndGemmMetadataKernel<6>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    case 7:
        fusedBuildRoutingAndGemmMetadataKernel<7>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    case 8:
        fusedBuildRoutingAndGemmMetadataKernel<8>
            <<<1, kTHREADS_PER_BLOCK, 0, stream>>>(buffers, fc1Setup, fc2Setup, topkIds, numTokens, numExperts);
        break;
    default: launched = false; break;
    }
    if (!launched)
    {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

__global__ void buildOffsetsKernel(int32_t const* expertCounts, int32_t* expertOffsets, int32_t* expertWriteOffsets,
    F16MoeGemmSetup fc1Setup, F16MoeGemmSetup fc2Setup, int32_t numExperts)
{
    if (threadIdx.x == 0)
    {
        int32_t prefix{0};
        expertOffsets[0] = 0;
        for (int32_t expert = 0; expert < numExperts; ++expert)
        {
            prefix += expertCounts[expert];
            expertOffsets[expert + 1] = prefix;
        }
    }
    __syncthreads();

    int32_t const expert = static_cast<int32_t>(threadIdx.x);
    if (expert < numExperts)
    {
        int32_t const rowOffset = expertOffsets[expert];
        expertWriteOffsets[expert] = rowOffset;
        int32_t const rows = expertCounts[expert];
        populateGemmMetadata(fc1Setup, expert, rowOffset, rows);
        populateGemmMetadata(fc2Setup, expert, rowOffset, rows);
    }
}

__global__ void scatterRouteMapKernel(int32_t const* topkIds, int32_t* expertWriteOffsets, int32_t* sortedToExpanded,
    int32_t* expandedToSorted, int32_t routedRows, int32_t numExperts)
{
    int32_t const expandedRow = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (expandedRow < routedRows)
    {
        int32_t const expert = topkIds[expandedRow];
        if (expert >= 0 && expert < numExperts)
        {
            int32_t const sortedRow = atomicAdd(expertWriteOffsets + expert, 1);
            sortedToExpanded[sortedRow] = expandedRow;
            expandedToSorted[expandedRow] = sortedRow;
        }
    }
}

__global__ void gatherHiddenKernel(uint4 const* hiddenStates, uint4* gatheredInput, int32_t const* sortedToExpanded,
    int32_t routedRows, int32_t topK, int32_t vectorsPerRow)
{
    int64_t const vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const numVectors = static_cast<int64_t>(routedRows) * vectorsPerRow;
    if (vector < numVectors)
    {
        int32_t const sortedRow = static_cast<int32_t>(vector / vectorsPerRow);
        int32_t const vectorInRow = static_cast<int32_t>(vector % vectorsPerRow);
        int32_t const expandedRow = sortedToExpanded[sortedRow];
        int32_t const token = expandedRow / topK;
        gatheredInput[vector] = hiddenStates[static_cast<int64_t>(token) * vectorsPerRow + vectorInRow];
    }
}

__global__ void activateFc1Kernel(
    __half const* rawFc1, __half* activatedFc1, int32_t routedRows, int32_t moeInterSize, int32_t activationType)
{
    int64_t const element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const numElements = static_cast<int64_t>(routedRows) * moeInterSize;
    if (element >= numElements)
    {
        return;
    }

    int32_t const row = static_cast<int32_t>(element / moeInterSize);
    int32_t const intermediate = static_cast<int32_t>(element % moeInterSize);
    float result{};
    if (activationType == kACT_SWIGLU)
    {
        int32_t const chunk = intermediate / kSWIGLU_CHUNK_ROWS;
        int32_t const rowInChunk = intermediate % kSWIGLU_CHUNK_ROWS;
        int32_t const upColumn = chunk * kSWIGLU_STORAGE_ROWS + rowInChunk;
        int32_t const gateColumn = upColumn + kSWIGLU_CHUNK_ROWS;
        int32_t const fc1N = 2 * moeInterSize;
        float const up = __half2float(rawFc1[static_cast<int64_t>(row) * fc1N + upColumn]);
        float const gate = __half2float(rawFc1[static_cast<int64_t>(row) * fc1N + gateColumn]);
        result = up * gate / (1.0F + expf(-gate));
    }
    else
    {
        float const projection = __half2float(rawFc1[static_cast<int64_t>(row) * moeInterSize + intermediate]);
        float const positive = projection > 0.0F ? projection : 0.0F;
        result = positive * positive;
    }
    activatedFc1[element] = __float2half_rn(result);
}

__global__ void scatterOutputKernel(__half const* routedOutput, int32_t const* expandedToSorted,
    float const* topkWeights, __half* output, int32_t numTokens, int32_t topK, int32_t hiddenSize)
{
    int64_t const element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const numElements = static_cast<int64_t>(numTokens) * hiddenSize;
    if (element >= numElements)
    {
        return;
    }

    int32_t const token = static_cast<int32_t>(element / hiddenSize);
    int32_t const hidden = static_cast<int32_t>(element % hiddenSize);
    float accumulator{0.0F};
    for (int32_t slot = 0; slot < topK; ++slot)
    {
        int32_t const expandedRow = token * topK + slot;
        int32_t const sortedRow = expandedToSorted[expandedRow];
        accumulator += topkWeights[expandedRow]
            * __half2float(routedOutput[static_cast<int64_t>(sortedRow) * hiddenSize + hidden]);
    }
    output[element] = __float2half_rn(accumulator);
}

} // namespace

cudaError_t buildF16MoeRoutingAndGemmMetadata(F16MoeRoutingBuffers const& buffers, F16MoeGemmSetup const& fc1Setup,
    F16MoeGemmSetup const& fc2Setup, int32_t const* topkIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    cudaStream_t stream) noexcept
{
    if (topkIds == nullptr || buffers.expertCounts == nullptr || buffers.expertOffsets == nullptr
        || buffers.expertWriteOffsets == nullptr || buffers.sortedToExpanded == nullptr
        || buffers.expandedToSorted == nullptr || !isValidGemmSetup(fc1Setup) || !isValidGemmSetup(fc2Setup)
        || numTokens <= 0 || topK <= 0 || topK > kMAX_TOP_K || numTokens > std::numeric_limits<int32_t>::max() / topK
        || numExperts <= 0 || numExperts > kTHREADS_PER_BLOCK)
    {
        return cudaErrorInvalidValue;
    }

    if (numTokens <= kMAX_FUSED_ROUTING_TOKENS)
    {
        return launchFusedBuildRoutingAndGemmMetadata(
            buffers, fc1Setup, fc2Setup, topkIds, numTokens, topK, numExperts, stream);
    }

    int32_t const routedRows = numTokens * topK;
    cudaError_t error
        = cudaMemsetAsync(buffers.expertCounts, 0, static_cast<size_t>(numExperts) * sizeof(int32_t), stream);
    if (error != cudaSuccess)
    {
        return error;
    }
    countRoutesKernel<<<gridSize(routedRows), kTHREADS_PER_BLOCK, 0, stream>>>(
        topkIds, buffers.expertCounts, routedRows, numExperts);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return error;
    }
    buildOffsetsKernel<<<1, kTHREADS_PER_BLOCK, 0, stream>>>(
        buffers.expertCounts, buffers.expertOffsets, buffers.expertWriteOffsets, fc1Setup, fc2Setup, numExperts);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return error;
    }
    scatterRouteMapKernel<<<gridSize(routedRows), kTHREADS_PER_BLOCK, 0, stream>>>(topkIds, buffers.expertWriteOffsets,
        buffers.sortedToExpanded, buffers.expandedToSorted, routedRows, numExperts);
    return cudaGetLastError();
}

cudaError_t gatherF16MoeHiddenRows(void const* hiddenStates, void* gatheredInput, int32_t const* sortedToExpanded,
    int32_t routedRows, int32_t topK, int32_t hiddenSize, cudaStream_t stream) noexcept
{
    if (hiddenStates == nullptr || gatheredInput == nullptr || sortedToExpanded == nullptr || routedRows <= 0
        || topK <= 0 || hiddenSize <= 0 || hiddenSize % kHALF_PER_VECTOR != 0)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const vectorsPerRow = hiddenSize / kHALF_PER_VECTOR;
    int64_t const numVectors = static_cast<int64_t>(routedRows) * vectorsPerRow;
    gatherHiddenKernel<<<gridSize(numVectors), kTHREADS_PER_BLOCK, 0, stream>>>(static_cast<uint4 const*>(hiddenStates),
        static_cast<uint4*>(gatheredInput), sortedToExpanded, routedRows, topK, vectorsPerRow);
    return cudaGetLastError();
}

cudaError_t activateF16Moe(void const* rawFc1, void* activatedFc1, int32_t routedRows, int32_t moeInterSize,
    int32_t activationType, cudaStream_t stream) noexcept
{
    if (rawFc1 == nullptr || activatedFc1 == nullptr || routedRows <= 0 || moeInterSize <= 0
        || (activationType != kACT_SWIGLU && activationType != kACT_RELU2))
    {
        return cudaErrorInvalidValue;
    }
    int64_t const numElements = static_cast<int64_t>(routedRows) * moeInterSize;
    activateFc1Kernel<<<gridSize(numElements), kTHREADS_PER_BLOCK, 0, stream>>>(static_cast<__half const*>(rawFc1),
        static_cast<__half*>(activatedFc1), routedRows, moeInterSize, activationType);
    return cudaGetLastError();
}

cudaError_t scatterF16MoeOutput(void const* routedOutput, int32_t const* expandedToSorted, float const* topkWeights,
    void* output, int32_t numTokens, int32_t topK, int32_t hiddenSize, cudaStream_t stream) noexcept
{
    if (routedOutput == nullptr || expandedToSorted == nullptr || topkWeights == nullptr || output == nullptr
        || numTokens <= 0 || topK <= 0 || hiddenSize <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const numElements = static_cast<int64_t>(numTokens) * hiddenSize;
    scatterOutputKernel<<<gridSize(numElements), kTHREADS_PER_BLOCK, 0, stream>>>(
        static_cast<__half const*>(routedOutput), expandedToSorted, topkWeights, static_cast<__half*>(output),
        numTokens, topK, hiddenSize);
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm

#endif // defined(CUTE_DSL_F16_MOE_ENABLED)
