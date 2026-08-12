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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "eagleAcceptKernels.h"
#include "kernels/common/argmaxKernel.h"
#include "speculativeKernelsUtils.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace trt_edgellm
{
namespace kernel
{

constexpr int32_t kTop1BlocksPerRow{8};

// Internal workspace structure for memory management (similar to SamplingWorkspace)
struct EagleAcceptWorkspace
{
    // Device pointer, owned by external caller - must remain valid for kernel lifetime
    void* ptr;
    // Size in bytes of the workspace buffer
    size_t size;

    // Device pointer to top-1 tokens buffer, owned by external caller
    // Array size: batchSize * numTokens elements (int32_t each)
    int32_t* top1Tokens;

    // Per-lane top-1 candidates for the two-stage argmax. The layout is
    // [batchSize * numTokens, kTop1BlocksPerRow].
    int32_t* top1TempIndices;
    float* top1TempValues;

    EagleAcceptWorkspace()
        : ptr(nullptr)
        , size(0)
        , top1Tokens(nullptr)
        , top1TempIndices(nullptr)
        , top1TempValues(nullptr)
    {
    }

    // Calculate workspace partitioning for given parameters
    void setupWorkspace(void* workspace, size_t workspaceSize, int32_t batchSize, int32_t numTokens)
    {
        // Check that workspace is aligned to 256 bytes for optimal GPU memory access
        ELLM_CHECK(reinterpret_cast<uintptr_t>(workspace) % kSpeculativeWorkspaceAlignment == 0,
            "Workspace must be aligned to 256 bytes");

        ptr = workspace;
        size = workspaceSize;

        // Calculate buffer sizes and offsets
        size_t offset = 0;

        int32_t const totalRows = batchSize * numTokens;

        // Top-1 tokens buffer
        size_t top1TokensSize = alignSpeculativeWorkspaceSize(totalRows * sizeof(int32_t));
        top1Tokens = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += top1TokensSize;

        size_t top1TempIndicesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(int32_t));
        top1TempIndices = reinterpret_cast<int32_t*>(static_cast<char*>(ptr) + offset);
        offset += top1TempIndicesSize;

        size_t top1TempValuesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(float));
        top1TempValues = reinterpret_cast<float*>(static_cast<char*>(ptr) + offset);
        offset += top1TempValuesSize;

        // Validate workspace size
        ELLM_CHECK(offset <= workspaceSize,
            "Eagle workspace size too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
    }
};

// Calculate workspace size for Eagle accept algorithm
size_t getEagleAcceptWorkspaceSize(int32_t batchSize, int32_t numTokens)
{
    int32_t const totalRows = batchSize * numTokens;
    size_t const top1TokensSize = alignSpeculativeWorkspaceSize(totalRows * sizeof(int32_t));
    size_t const top1TempIndicesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(int32_t));
    size_t const top1TempValuesSize = alignSpeculativeWorkspaceSize(totalRows * kTop1BlocksPerRow * sizeof(float));
    return top1TokensSize + top1TempIndicesSize + top1TempValuesSize;
}

namespace
{

// Helper structure for top-1 selection
struct Top1Helper
{
    float value;
    int32_t index;

    __device__ __forceinline__ Top1Helper()
        : value(-FLT_MAX)
        , index(-1)
    {
    }

    __device__ __forceinline__ void update(float elem, int32_t elemId)
    {
        if (elem > value)
        {
            value = elem;
            index = elemId;
        }
    }
};

// Reduction operator for top-1 selection
struct top1MaxOpFunctor
{
    __device__ __forceinline__ Top1Helper operator()(Top1Helper const& a, Top1Helper const& b) const
    {
        return a.value > b.value ? a : b;
    }
};

// Inline function for shared memory alignment
__forceinline__ size_t alignSharedMem(size_t size)
{
    return ((size + 15) / 16) * 16; // Align to 16 bytes
}

__global__ void sequentialAcceptWalkKernel(int32_t const* __restrict__ argmaxResults,
    int32_t const* __restrict__ draftTokenIds, int32_t* __restrict__ acceptedTokenIds,
    int32_t* __restrict__ acceptLength, int32_t verifyLen)
{
    int32_t const batchIdx = blockIdx.x;

    int32_t const* batchArgmax = argmaxResults + batchIdx * verifyLen;
    int32_t const* batchDraft = draftTokenIds + batchIdx * verifyLen;
    int32_t* batchAccepted = acceptedTokenIds + batchIdx * verifyLen;

    batchAccepted[0] = batchArgmax[0];
    int32_t accepted = 1;

    for (int32_t tokenIdx = 1; tokenIdx < verifyLen; ++tokenIdx)
    {
        if (batchArgmax[tokenIdx - 1] == batchDraft[tokenIdx])
        {
            batchAccepted[accepted] = batchArgmax[tokenIdx];
            ++accepted;
        }
        else
        {
            break;
        }
    }

    acceptLength[batchIdx] = accepted;
}

// Stage 1: compute a per-lane top-1 for each verify row. The lane
// partitioning intentionally mirrors sampler::selectAllTopK(topK=1) so greedy
// speculative accept resolves exact-logit ties the same way as vanilla decode.
template <int32_t BLOCK_SIZE, int32_t BLOCKS_PER_ROW>
__global__ void eagleComputeTop1Stage1Kernel(float const* logits, int32_t* top1TempIndices, float* top1TempValues,
    int32_t batchSize, int32_t numTokens, int32_t vocabSize)
{
    typedef cub::BlockReduce<Top1Helper, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const rowIdx = static_cast<int32_t>(blockIdx.x / BLOCKS_PER_ROW);
    auto const blockLane = static_cast<int32_t>(blockIdx.x % BLOCKS_PER_ROW);
    int32_t const totalRows = batchSize * numTokens;

    if (rowIdx >= totalRows)
    {
        return;
    }

    float const* rowLogits = logits + static_cast<int64_t>(rowIdx) * vocabSize;
    Top1Helper partial;

    for (int32_t v = tid + blockLane * BLOCK_SIZE; v < vocabSize; v += BLOCK_SIZE * BLOCKS_PER_ROW)
    {
        partial.update(rowLogits[v], v);
    }

    Top1Helper laneMax = BlockReduce(tempStorage).Reduce(partial, top1MaxOpFunctor());

    if (tid == 0)
    {
        int32_t const tempOffset = rowIdx * BLOCKS_PER_ROW + blockLane;
        top1TempIndices[tempOffset] = laneMax.index;
        top1TempValues[tempOffset] = laneMax.value;
    }
}

// Stage 2: reduce the lane candidates to the final top-1 token. The tie order
// matches the sampler top-k stage: the reduction compares values only.
template <int32_t BLOCK_SIZE, int32_t BLOCKS_PER_ROW>
__global__ void eagleComputeTop1Stage2Kernel(int32_t const* top1TempIndices, float const* top1TempValues,
    int32_t* top1Tokens, int32_t const* vocabMappingTable, int32_t totalRows)
{
    typedef cub::BlockReduce<Top1Helper, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const rowIdx = static_cast<int32_t>(blockIdx.x);

    if (rowIdx >= totalRows)
    {
        return;
    }

    Top1Helper partial;
    for (int32_t i = tid; i < BLOCKS_PER_ROW; i += BLOCK_SIZE)
    {
        int32_t const tempOffset = rowIdx * BLOCKS_PER_ROW + i;
        partial.update(top1TempValues[tempOffset], top1TempIndices[tempOffset]);
    }

    Top1Helper rowMax = BlockReduce(tempStorage).Reduce(partial, top1MaxOpFunctor());

    if (tid == 0)
    {
        int32_t selectedIdx = (rowMax.index != -1) ? rowMax.index : 0;
        if (vocabMappingTable != nullptr)
        {
            selectedIdx = vocabMappingTable[selectedIdx];
        }
        top1Tokens[rowIdx] = selectedIdx;
    }
}

// Helper function to compute tree depth - count total connections (sum of 1s)
__device__ int32_t computeTokenDepth(int32_t tokenIdx, int8_t const* attentionMask, int32_t numTokens)
{
    int32_t depth = 0;
    for (int32_t i = 0; i < numTokens; ++i)
    {
        if (attentionMask[tokenIdx * numTokens + i] == 1)
        {
            depth++;
        }
    }
    return depth;
}

// CUDA kernel for eagle accept algorithm - optimized for concurrent batch processing
__global__ void eagleAcceptKernel(int32_t const* top1Tokens, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedLogitsIndices, int32_t* acceptLength, int32_t batchSize,
    int32_t numTokens, int32_t maxDepth)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const tid = threadIdx.x;
    int32_t const blockSize = blockDim.x;

    if (batchIdx >= batchSize)
        return;

    // Optimized shared memory layout
    extern __shared__ char sharedMem[];
    int32_t* tokenDepths = reinterpret_cast<int32_t*>(sharedMem);

    // Batch-specific pointers for better cache locality
    int8_t const* batchAttentionMask = attentionMask + batchIdx * numTokens * numTokens;
    int32_t const* batchTokenIds = tokenIds + batchIdx * numTokens;
    int32_t const* batchTop1Tokens = top1Tokens + batchIdx * numTokens;

    // Parallel initialization of output arrays
    // Use 0 for padding token IDs instead of -1 to avoid embedding lookup issues in draft model.
    // The actual padding positions will be skipped based on acceptLength.
    for (int32_t i = tid; i < maxDepth; i += blockSize)
    {
        acceptedTokenIds[batchIdx * maxDepth + i] = 0;
        acceptedLogitsIndices[batchIdx * maxDepth + i] = -1;
    }
    if (tid == 0)
    {
        acceptLength[batchIdx] = 0;
    }

    // Parallel computation of token depths with better memory access
    for (int32_t i = tid; i < numTokens; i += blockSize)
    {
        tokenDepths[i] = computeTokenDepth(i, batchAttentionMask, numTokens);
    }
    __syncthreads();

    // Process this batch - use warp-level operations where possible
    int32_t currentDepth = 0;
    int32_t currentTokenIdx = 0;
    int32_t expectedNextDepth = tokenDepths[0] + 1;

    for (int32_t step = 0; step < maxDepth && currentTokenIdx < numTokens; ++step)
    {
        // Step 1: Get precomputed top-1 token (broadcast to all threads)
        int32_t selectedTokenId = batchTop1Tokens[currentTokenIdx];

        // Step 2: Accept the selected token (single thread writes)
        if (tid == 0)
        {
            acceptedTokenIds[batchIdx * maxDepth + currentDepth] = selectedTokenId;
            acceptedLogitsIndices[batchIdx * maxDepth + currentDepth] = currentTokenIdx;
            acceptLength[batchIdx] = currentDepth + 1;
            currentDepth++;
        }

        // Step 3: Parallel tree search with block-level reduction
        __shared__ int32_t nextTokenIdx;
        // Finish prior iteration's reads of nextTokenIdx before re-initializing it.
        __syncthreads();
        if (tid == 0)
        {
            nextTokenIdx = numTokens; // Initialize to invalid value
        }
        __syncthreads();

        // Each thread checks different tokens in parallel
        for (int32_t checkIdx = 1 + tid; checkIdx < numTokens; checkIdx += blockSize)
        {
            if (batchTokenIds[checkIdx] == selectedTokenId && tokenDepths[checkIdx] == expectedNextDepth)
            {
                // Check attention mask: does checkIdx attend to currentTokenIdx?
                int32_t maskOffset = batchIdx * numTokens * numTokens + checkIdx * numTokens + currentTokenIdx;
                if (attentionMask[maskOffset] == 1)
                {
                    // Found a valid next token - use atomic to get the minimum index for deterministic behavior
                    atomicMin(&nextTokenIdx, checkIdx);
                }
            }
        }
        __syncthreads();

        // Step 4: Update for next iteration (all threads participate)
        if (nextTokenIdx < numTokens)
        {
            // Found valid next token in tree, continue from there
            currentTokenIdx = nextTokenIdx;
            expectedNextDepth++;
        }
        else
        {
            // No valid next token found in tree, stop here
            break;
        }
    }
}

// Optimized kernel launcher function using workspace and two-stage approach
void launchEagleAcceptKernel(float const* logits, int32_t const* tokenIds, int8_t const* attentionMask,
    int32_t* acceptedTokenIds, int32_t* acceptedLogitsIndices, int32_t* acceptLength, int32_t const* vocabMappingTable,
    int32_t batchSize, int32_t numTokens, int32_t vocabSize, int32_t maxDepth, void* workspace, size_t workspaceSize,
    cudaStream_t stream)
{
    constexpr int32_t blockSize = 256;

    // Setup workspace partitioning
    EagleAcceptWorkspace ws;
    ws.setupWorkspace(workspace, workspaceSize, batchSize, numTokens);

    // Validate workspace buffer
    assert(ws.top1Tokens != nullptr);

    int32_t const totalRows = batchSize * numTokens;

    // Stage 1/2: Compute top-1 tokens for all positions. The two-stage
    // partitioning matches selectAllTopK(topK=1), which keeps greedy accept
    // deterministic against the vanilla decoder when logits tie exactly.
    dim3 const gridSizeStage1(totalRows * kTop1BlocksPerRow);
    dim3 const blockSizeStage1(blockSize);
    eagleComputeTop1Stage1Kernel<blockSize, kTop1BlocksPerRow><<<gridSizeStage1, blockSizeStage1, 0, stream>>>(
        logits, ws.top1TempIndices, ws.top1TempValues, batchSize, numTokens, vocabSize);

    dim3 const gridSizeStage2Top1(totalRows);
    dim3 const blockSizeStage2Top1(blockSize);
    eagleComputeTop1Stage2Kernel<blockSize, kTop1BlocksPerRow><<<gridSizeStage2Top1, blockSizeStage2Top1, 0, stream>>>(
        ws.top1TempIndices, ws.top1TempValues, ws.top1Tokens, vocabMappingTable, totalRows);

    // Stage 3: Run optimized eagle accept algorithm
    dim3 const gridSizeStage3(batchSize);
    dim3 const blockSizeStage3(blockSize);

    // Calculate shared memory for stage 3 (only token depths - much smaller!)
    size_t sharedMemSizeStage3 = numTokens * sizeof(int32_t);
    sharedMemSizeStage3 = alignSharedMem(sharedMemSizeStage3);

    eagleAcceptKernel<<<gridSizeStage3, blockSizeStage3, sharedMemSizeStage3, stream>>>(ws.top1Tokens, tokenIds,
        attentionMask, acceptedTokenIds, acceptedLogitsIndices, acceptLength, batchSize, numTokens, maxDepth);
}

} // namespace

void sequentialAccept(rt::Tensor const& logits, rt::Tensor const& draftTokenIds, rt::Tensor& acceptedTokenIds,
    rt::Tensor& acceptLength, rt::Tensor& argmaxScratch, int32_t batchSize, int32_t verifyLen, int32_t vocabSize,
    cudaStream_t stream)
{
    int32_t const totalPositions = batchSize * verifyLen;
    int32_t* argmaxResults = static_cast<int32_t*>(argmaxScratch.rawPointer());

    invokeRowwiseArgmax<float>(
        static_cast<float const*>(logits.rawPointer()), totalPositions, vocabSize, argmaxResults, stream);
    CUDA_CHECK(cudaGetLastError());

    sequentialAcceptWalkKernel<<<batchSize, 1, 0, stream>>>(argmaxResults,
        static_cast<int32_t const*>(draftTokenIds.rawPointer()), static_cast<int32_t*>(acceptedTokenIds.rawPointer()),
        static_cast<int32_t*>(acceptLength.rawPointer()), verifyLen);
    CUDA_CHECK(cudaGetLastError());
}

void eagleAccept(rt::Tensor const& logits, rt::Tensor const& tokenIds, rt::Tensor const& attentionMask,
    rt::Tensor& acceptedTokenIds, rt::Tensor& acceptedLogitsIndices, rt::Tensor& acceptLength,
    rt::OptionalInputTensor const& vocabMappingTable, void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    // Validate input shapes
    auto const logitsShape = logits.getShape();
    auto const tokenIdsShape = tokenIds.getShape();
    auto const maskShape = attentionMask.getShape();
    auto const acceptedTokenIdsShape = acceptedTokenIds.getShape();
    auto const acceptedLogitsIndicesShape = acceptedLogitsIndices.getShape();
    auto const acceptLengthShape = acceptLength.getShape();

    check::check(logitsShape.getNumDims() == 2, "logits must be 2D tensor [batch_size * num_tokens, vocab_size]");
    check::check(tokenIdsShape.getNumDims() == 2, "tokenIds must be 2D tensor [batch_size, num_tokens]");
    check::check(maskShape.getNumDims() == 3, "attentionMask must be 3D tensor [batch_size, num_tokens, num_tokens]");
    check::check(acceptedTokenIdsShape.getNumDims() == 2, "acceptedTokenIds must be 2D tensor [batch_size, max_depth]");
    check::check(acceptedLogitsIndicesShape.getNumDims() == 2,
        "acceptedLogitsIndices must be 2D tensor [batch_size, max_depth]");
    check::check(acceptLengthShape.getNumDims() == 1, "acceptLength must be 1D tensor [batch_size]");

    int32_t const batchSize = tokenIdsShape[0];
    int32_t const numTokens = tokenIdsShape[1];
    int32_t const vocabSize = logitsShape[1];
    int32_t const maxDepth = acceptedTokenIdsShape[1];

    check::check(logitsShape[0] == batchSize * numTokens, "logits must be [batch_size * num_tokens, vocab_size]");
    check::check(maskShape[0] == batchSize && maskShape[1] == numTokens && maskShape[2] == numTokens,
        "attentionMask must be [batch_size, num_tokens, num_tokens]");
    check::check(acceptedTokenIdsShape[0] == batchSize && acceptedTokenIdsShape[1] == maxDepth,
        "acceptedTokenIds must be [batch_size, max_depth]");
    check::check(acceptedLogitsIndicesShape[0] == batchSize && acceptedLogitsIndicesShape[1] == maxDepth,
        "acceptedLogitsIndices must be [batch_size, max_depth]");
    check::check(acceptLengthShape[0] == batchSize, "acceptLength length must match batch_size");

    // Validate data types
    check::check(logits.getDataType() == nvinfer1::DataType::kFLOAT, "logits must be FP32");
    check::check(tokenIds.getDataType() == nvinfer1::DataType::kINT32, "tokenIds must be INT32");
    check::check(attentionMask.getDataType() == nvinfer1::DataType::kINT8, "attentionMask must be INT8");
    check::check(acceptedTokenIds.getDataType() == nvinfer1::DataType::kINT32, "acceptedTokenIds must be INT32");
    check::check(
        acceptedLogitsIndices.getDataType() == nvinfer1::DataType::kINT32, "acceptedLogitsIndices must be INT32");
    check::check(acceptLength.getDataType() == nvinfer1::DataType::kINT32, "acceptLength must be INT32");

    // Validate device types - all tensors must be on GPU
    check::check(logits.getDeviceType() == rt::DeviceType::kGPU, "logits must be on GPU device");
    check::check(tokenIds.getDeviceType() == rt::DeviceType::kGPU, "tokenIds must be on GPU device");
    check::check(attentionMask.getDeviceType() == rt::DeviceType::kGPU, "attentionMask must be on GPU device");
    check::check(acceptedTokenIds.getDeviceType() == rt::DeviceType::kGPU, "acceptedTokenIds must be on GPU device");
    check::check(
        acceptedLogitsIndices.getDeviceType() == rt::DeviceType::kGPU, "acceptedLogitsIndices must be on GPU device");
    check::check(acceptLength.getDeviceType() == rt::DeviceType::kGPU, "acceptLength must be on GPU device");
    check::check(maxDepth > 0 && maxDepth <= numTokens, "maxDepth must be positive and <= numTokens");

    // Validate vocab mapping table if provided
    int32_t const* vocabMappingTablePtr = nullptr;
    if (vocabMappingTable.has_value())
    {
        rt::Tensor const& vocabMapTensor = vocabMappingTable.value().get();
        check::check(vocabMapTensor.getDeviceType() == rt::DeviceType::kGPU, "vocabMappingTable must be on GPU device");
        check::check(vocabMapTensor.getDataType() == nvinfer1::DataType::kINT32, "vocabMappingTable must be INT32");
        check::check(vocabMapTensor.getShape().getNumDims() == 1, "vocabMappingTable must be 1D");
        check::check(vocabMapTensor.getShape()[0] == vocabSize, "vocabMappingTable size must match vocab size");
        vocabMappingTablePtr = vocabMapTensor.dataPointer<int32_t>();
    }

    // Get device pointers
    float const* logitsPtr = logits.dataPointer<float>();
    int32_t const* tokenIdsPtr = tokenIds.dataPointer<int32_t>();
    int8_t const* attentionMaskPtr = attentionMask.dataPointer<int8_t>();
    int32_t* acceptedTokenIdsPtr = acceptedTokenIds.dataPointer<int32_t>();
    int32_t* acceptedLogitsIndicesPtr = acceptedLogitsIndices.dataPointer<int32_t>();
    int32_t* acceptLengthPtr = acceptLength.dataPointer<int32_t>();

    // Validate workspace size
    size_t requiredWorkspaceSize = getEagleAcceptWorkspaceSize(batchSize, numTokens);
    ELLM_CHECK(workspaceSize >= requiredWorkspaceSize,
        "Eagle workspace size too small. Required: " + std::to_string(requiredWorkspaceSize)
            + ", provided: " + std::to_string(workspaceSize));

    // Launch kernel
    launchEagleAcceptKernel(logitsPtr, tokenIdsPtr, attentionMaskPtr, acceptedTokenIdsPtr, acceptedLogitsIndicesPtr,
        acceptLengthPtr, vocabMappingTablePtr, batchSize, numTokens, vocabSize, maxDepth, workspace, workspaceSize,
        stream);
}

} // namespace kernel
} // namespace trt_edgellm
