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

#include "kernels/speculative/ddtreeKernels.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "speculativeKernelsUtils.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

constexpr int32_t kCandidateBlockSize{256};
constexpr int32_t kMaskBitsPerWord{32};
constexpr int32_t kRootParent{-1};
constexpr int32_t kPaddingParent{-1};

size_t getCandidateTokenWorkspaceBytes(int32_t batchSize, int32_t dflashBlockSize, int32_t candidateTopK)
{
    return alignSpeculativeWorkspaceSize(
        static_cast<size_t>(batchSize) * dflashBlockSize * candidateTopK * sizeof(int32_t));
}

struct DDTreeBuildWorkspace
{
    int32_t* candidateTokenIds{nullptr};
    float* candidateLogProbs{nullptr};

    void setup(void* workspace, size_t workspaceSize, int32_t batchSize, int32_t dflashBlockSize, int32_t candidateTopK)
    {
        ELLM_CHECK(reinterpret_cast<uintptr_t>(workspace) % kSpeculativeWorkspaceAlignment == 0,
            "DDTree workspace must be 256-byte aligned.");

        size_t offset{0};
        size_t const candidateTokenBytes = getCandidateTokenWorkspaceBytes(batchSize, dflashBlockSize, candidateTopK);
        candidateTokenIds = reinterpret_cast<int32_t*>(static_cast<char*>(workspace) + offset);
        offset += candidateTokenBytes;

        size_t const candidateLogProbBytes = alignSpeculativeWorkspaceSize(
            static_cast<size_t>(batchSize) * dflashBlockSize * candidateTopK * sizeof(float));
        candidateLogProbs = reinterpret_cast<float*>(static_cast<char*>(workspace) + offset);
        offset += candidateLogProbBytes;

        ELLM_CHECK(offset <= workspaceSize,
            "DDTree build workspace too small. Required: " + std::to_string(offset)
                + ", provided: " + std::to_string(workspaceSize));
    }
};

__device__ __forceinline__ bool isBetterCandidate(float lhsScore, int32_t lhsToken, float rhsScore, int32_t rhsToken)
{
    return lhsScore > rhsScore || (lhsScore == rhsScore && lhsToken < rhsToken);
}

__device__ void insertCandidate(float score, int32_t tokenId, float (&topScores)[kDDTreeMaxCandidateTopK],
    int32_t (&topTokenIds)[kDDTreeMaxCandidateTopK], int32_t candidateTopK)
{
    for (int32_t slot = 0; slot < candidateTopK; ++slot)
    {
        if (isBetterCandidate(score, tokenId, topScores[slot], topTokenIds[slot]))
        {
            for (int32_t move = candidateTopK - 1; move > slot; --move)
            {
                topScores[move] = topScores[move - 1];
                topTokenIds[move] = topTokenIds[move - 1];
            }
            topScores[slot] = score;
            topTokenIds[slot] = tokenId;
            break;
        }
    }
}

__global__ void selectDDTreeCandidatesKernel(float const* __restrict__ draftLogits,
    int32_t* __restrict__ candidateTokenIds, float* __restrict__ candidateLogProbs, int32_t dflashBlockSize,
    int32_t vocabSize, int32_t candidateTopK)
{
    int32_t const batchIdx = blockIdx.x / dflashBlockSize;
    int32_t const depthIdx = blockIdx.x % dflashBlockSize;
    int32_t const tid = threadIdx.x;

    int32_t const candidateOffset = (batchIdx * dflashBlockSize + depthIdx) * candidateTopK;
    if (depthIdx == 0)
    {
        for (int32_t slot = tid; slot < candidateTopK; slot += blockDim.x)
        {
            candidateTokenIds[candidateOffset + slot] = 0;
            candidateLogProbs[candidateOffset + slot] = -INFINITY;
        }
        return;
    }

    __shared__ float sTopScores[kCandidateBlockSize * kDDTreeMaxCandidateTopK];
    __shared__ int32_t sTopTokenIds[kCandidateBlockSize * kDDTreeMaxCandidateTopK];
    __shared__ float sLocalMax[kCandidateBlockSize];
    __shared__ float sLocalExpSum[kCandidateBlockSize];
    __shared__ float sGlobalMax;
    __shared__ float sLogSumExp;

    float topScores[kDDTreeMaxCandidateTopK];
    int32_t topTokenIds[kDDTreeMaxCandidateTopK];
    for (int32_t slot = 0; slot < kDDTreeMaxCandidateTopK; ++slot)
    {
        topScores[slot] = -INFINITY;
        topTokenIds[slot] = 0;
    }

    float localMax = -INFINITY;
    int64_t const logitsOffset
        = static_cast<int64_t>(batchIdx) * dflashBlockSize * vocabSize + static_cast<int64_t>(depthIdx) * vocabSize;

    for (int32_t vocabIdx = tid; vocabIdx < vocabSize; vocabIdx += blockDim.x)
    {
        float const score = draftLogits[logitsOffset + vocabIdx];
        localMax = fmaxf(localMax, score);
        insertCandidate(score, vocabIdx, topScores, topTokenIds, candidateTopK);
    }

    sLocalMax[tid] = localMax;
    for (int32_t slot = 0; slot < candidateTopK; ++slot)
    {
        sTopScores[tid * kDDTreeMaxCandidateTopK + slot] = topScores[slot];
        sTopTokenIds[tid * kDDTreeMaxCandidateTopK + slot] = topTokenIds[slot];
    }
    __syncthreads();

    if (tid == 0)
    {
        float globalMax = -INFINITY;
        for (int32_t i = 0; i < blockDim.x; ++i)
        {
            globalMax = fmaxf(globalMax, sLocalMax[i]);
        }
        sGlobalMax = globalMax;
    }
    __syncthreads();

    float localExpSum{0.0F};
    for (int32_t vocabIdx = tid; vocabIdx < vocabSize; vocabIdx += blockDim.x)
    {
        localExpSum += expf(draftLogits[logitsOffset + vocabIdx] - sGlobalMax);
    }
    sLocalExpSum[tid] = localExpSum;
    __syncthreads();

    if (tid == 0)
    {
        float totalExpSum{0.0F};
        for (int32_t i = 0; i < blockDim.x; ++i)
        {
            totalExpSum += sLocalExpSum[i];
        }
        sLogSumExp = sGlobalMax + logf(totalExpSum);

        float bestScores[kDDTreeMaxCandidateTopK];
        int32_t bestTokenIds[kDDTreeMaxCandidateTopK];
        for (int32_t slot = 0; slot < kDDTreeMaxCandidateTopK; ++slot)
        {
            bestScores[slot] = -INFINITY;
            bestTokenIds[slot] = 0;
        }

        for (int32_t threadOffset = 0; threadOffset < blockDim.x; ++threadOffset)
        {
            for (int32_t slot = 0; slot < candidateTopK; ++slot)
            {
                int32_t const localOffset = threadOffset * kDDTreeMaxCandidateTopK + slot;
                insertCandidate(
                    sTopScores[localOffset], sTopTokenIds[localOffset], bestScores, bestTokenIds, candidateTopK);
            }
        }

        for (int32_t slot = 0; slot < candidateTopK; ++slot)
        {
            candidateTokenIds[candidateOffset + slot] = bestTokenIds[slot];
            candidateLogProbs[candidateOffset + slot] = bestScores[slot] - sLogSumExp;
        }
    }
}

__device__ __forceinline__ bool isFiniteScore(float score)
{
    return score > -FLT_MAX * 0.5F;
}

__device__ __forceinline__ bool isBetterTreeExpansion(float lhsScore, int32_t lhsParent, int32_t lhsSlot,
    int32_t lhsToken, float rhsScore, int32_t rhsParent, int32_t rhsSlot, int32_t rhsToken)
{
    if (lhsScore != rhsScore)
    {
        return lhsScore > rhsScore;
    }
    if (lhsParent != rhsParent)
    {
        return lhsParent < rhsParent;
    }
    if (lhsSlot != rhsSlot)
    {
        return lhsSlot < rhsSlot;
    }
    return lhsToken < rhsToken;
}

__global__ void buildDDTreeKernel(int32_t const* __restrict__ rootTokenIds, int32_t const* __restrict__ baseLengths,
    int32_t const* __restrict__ candidateTokenIds, float const* __restrict__ candidateLogProbs,
    int32_t const* __restrict__ draftVocabMappingTable, int32_t* __restrict__ nodeTokenIds,
    int32_t* __restrict__ nodeDepths, int32_t* __restrict__ parentIds, float* __restrict__ nodeScores,
    int32_t* __restrict__ validCounts, int32_t* __restrict__ verifyTokenIds, int32_t* __restrict__ verifyPositionIds,
    int32_t* __restrict__ packedAncestorMask, int8_t* __restrict__ ancestorMask, int32_t* __restrict__ contextLengths,
    int64_t* __restrict__ selectTokenIndices, int32_t dflashBlockSize, int32_t verifySize, int32_t candidateTopK)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const packedMaskLen = (verifySize + kMaskBitsPerWord - 1) / kMaskBitsPerWord;
    int32_t const treeOffset = batchIdx * verifySize;
    int32_t const packedMaskOffset = batchIdx * verifySize * packedMaskLen;
    int32_t const ancestorMaskOffset = batchIdx * verifySize * verifySize;

    if (threadIdx.x != 0)
    {
        return;
    }

    extern __shared__ int32_t nextCandidateSlot[];
    for (int32_t nodeIdx = 0; nodeIdx < verifySize; ++nodeIdx)
    {
        nodeTokenIds[treeOffset + nodeIdx] = 0;
        nodeDepths[treeOffset + nodeIdx] = 0;
        parentIds[treeOffset + nodeIdx] = kPaddingParent;
        nodeScores[treeOffset + nodeIdx] = -INFINITY;
        verifyTokenIds[treeOffset + nodeIdx] = 0;
        verifyPositionIds[treeOffset + nodeIdx] = 0;
        selectTokenIndices[treeOffset + nodeIdx] = nodeIdx;
        nextCandidateSlot[nodeIdx] = 0;
        for (int32_t wordIdx = 0; wordIdx < packedMaskLen; ++wordIdx)
        {
            packedAncestorMask[packedMaskOffset + nodeIdx * packedMaskLen + wordIdx] = 0;
        }
        for (int32_t colIdx = 0; colIdx < verifySize; ++colIdx)
        {
            ancestorMask[ancestorMaskOffset + nodeIdx * verifySize + colIdx] = 0;
        }
    }

    nodeTokenIds[treeOffset] = rootTokenIds[batchIdx];
    nodeDepths[treeOffset] = 0;
    parentIds[treeOffset] = kRootParent;
    nodeScores[treeOffset] = 0.0F;

    int32_t validCount{1};
    int32_t const maxProposalDepth = dflashBlockSize - 1;

    for (int32_t outNodeIdx = 1; outNodeIdx < verifySize; ++outNodeIdx)
    {
        int32_t bestParent{-1};
        int32_t bestSlot{-1};
        int32_t bestToken{0};
        float bestScore{-INFINITY};

        for (int32_t parentIdx = 0; parentIdx < validCount; ++parentIdx)
        {
            int32_t const parentDepth = nodeDepths[treeOffset + parentIdx];
            if (parentDepth >= maxProposalDepth)
            {
                continue;
            }

            int32_t const slot = nextCandidateSlot[parentIdx];
            if (slot >= candidateTopK)
            {
                continue;
            }

            int32_t const childDepth = parentDepth + 1;
            int32_t const candidateOffset = (batchIdx * dflashBlockSize + childDepth) * candidateTopK + slot;
            float const candidateScore = candidateLogProbs[candidateOffset];
            if (!isFiniteScore(candidateScore))
            {
                continue;
            }

            int32_t const candidateToken = candidateTokenIds[candidateOffset];
            float const prefixScore = nodeScores[treeOffset + parentIdx] + candidateScore;
            if (bestParent < 0
                || isBetterTreeExpansion(
                    prefixScore, parentIdx, slot, candidateToken, bestScore, bestParent, bestSlot, bestToken))
            {
                bestParent = parentIdx;
                bestSlot = slot;
                bestToken = candidateToken;
                bestScore = prefixScore;
            }
        }

        if (bestParent < 0)
        {
            break;
        }

        nextCandidateSlot[bestParent] += 1;
        nodeTokenIds[treeOffset + outNodeIdx]
            = draftVocabMappingTable == nullptr ? bestToken : draftVocabMappingTable[bestToken];
        nodeDepths[treeOffset + outNodeIdx] = nodeDepths[treeOffset + bestParent] + 1;
        parentIds[treeOffset + outNodeIdx] = bestParent;
        nodeScores[treeOffset + outNodeIdx] = bestScore;
        validCount += 1;
    }

    validCounts[batchIdx] = validCount;

    int32_t const baseLength = baseLengths[batchIdx];
    contextLengths[batchIdx] = baseLength + verifySize;
    for (int32_t nodeIdx = 0; nodeIdx < validCount; ++nodeIdx)
    {
        verifyTokenIds[treeOffset + nodeIdx] = nodeTokenIds[treeOffset + nodeIdx];
        verifyPositionIds[treeOffset + nodeIdx] = baseLength + nodeDepths[treeOffset + nodeIdx];

        int32_t ancestorIdx = nodeIdx;
        while (ancestorIdx >= 0)
        {
            int32_t const wordIdx = ancestorIdx / kMaskBitsPerWord;
            int32_t const bitIdx = ancestorIdx % kMaskBitsPerWord;
            packedAncestorMask[packedMaskOffset + nodeIdx * packedMaskLen + wordIdx]
                |= static_cast<int32_t>(1U << bitIdx);
            ancestorMask[ancestorMaskOffset + nodeIdx * verifySize + ancestorIdx] = 1;
            ancestorIdx = parentIds[treeOffset + ancestorIdx];
        }
    }
}

void validateGpuTensor(
    rt::Tensor const& tensor, char const* tensorName, nvinfer1::DataType dataType, char const* dataTypeName)
{
    check::check(tensor.getDeviceType() == rt::DeviceType::kGPU, std::string(tensorName) + " must be on GPU.");
    check::check(tensor.getDataType() == dataType, std::string(tensorName) + " must be " + dataTypeName + ".");
}

} // anonymous namespace

size_t getDDTreeBuildWorkspaceSize(
    int32_t batchSize, int32_t dflashBlockSize, int32_t verifySize, int32_t vocabSize, int32_t candidateTopK)
{
    if (batchSize <= 0 || dflashBlockSize <= 0 || verifySize <= 0 || vocabSize <= 0 || candidateTopK <= 0
        || candidateTopK > kDDTreeMaxCandidateTopK || candidateTopK > vocabSize)
    {
        return 0;
    }

    size_t const candidateTokenBytes = getCandidateTokenWorkspaceBytes(batchSize, dflashBlockSize, candidateTopK);
    size_t const candidateLogProbBytes = alignSpeculativeWorkspaceSize(
        static_cast<size_t>(batchSize) * dflashBlockSize * candidateTopK * sizeof(float));
    return candidateTokenBytes + candidateLogProbBytes;
}

void ddtreeBuild(DDTreeBuildParams const& params)
{
    rt::Tensor const& draftLogits = params.inputs.draftLogits;
    rt::Tensor const& rootTokenIds = params.inputs.rootTokenIds;
    rt::Tensor const& baseLengths = params.inputs.baseLengths;
    rt::Tensor const* draftVocabMappingTable = params.inputs.draftVocabMappingTable;
    rt::Tensor& nodeTokenIds = params.outputs.nodeTokenIds;
    rt::Tensor& nodeDepths = params.outputs.nodeDepths;
    rt::Tensor& parentIds = params.outputs.parentIds;
    rt::Tensor& nodeScores = params.outputs.nodeScores;
    rt::Tensor& validCounts = params.outputs.validCounts;
    rt::Tensor& verifyTokenIds = params.outputs.verifyTokenIds;
    rt::Tensor& verifyPositionIds = params.outputs.verifyPositionIds;
    rt::Tensor& packedAncestorMask = params.outputs.packedAncestorMask;
    rt::Tensor& ancestorMask = params.outputs.ancestorMask;
    rt::Tensor& contextLengths = params.outputs.contextLengths;
    rt::Tensor& selectTokenIndices = params.outputs.selectTokenIndices;
    int32_t const candidateTopK = params.candidateTopK;

    validateGpuTensor(draftLogits, "draftLogits", nvinfer1::DataType::kFLOAT, "FLOAT");
    validateGpuTensor(rootTokenIds, "rootTokenIds", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(baseLengths, "baseLengths", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(nodeTokenIds, "nodeTokenIds", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(nodeDepths, "nodeDepths", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(parentIds, "parentIds", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(nodeScores, "nodeScores", nvinfer1::DataType::kFLOAT, "FLOAT");
    validateGpuTensor(validCounts, "validCounts", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(verifyTokenIds, "verifyTokenIds", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(verifyPositionIds, "verifyPositionIds", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(packedAncestorMask, "packedAncestorMask", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(ancestorMask, "ancestorMask", nvinfer1::DataType::kINT8, "INT8");
    validateGpuTensor(contextLengths, "contextLengths", nvinfer1::DataType::kINT32, "INT32");
    validateGpuTensor(selectTokenIndices, "selectTokenIndices", nvinfer1::DataType::kINT64, "INT64");

    auto const logitsShape = draftLogits.getShape();
    auto const nodeShape = nodeTokenIds.getShape();
    auto const ancestorMaskShape = ancestorMask.getShape();
    auto const contextLengthsShape = contextLengths.getShape();
    auto const selectTokenIndicesShape = selectTokenIndices.getShape();
    check::check(logitsShape.getNumDims() == 3, "draftLogits must be [batch, dflashBlockSize, vocabSize].");
    check::check(nodeShape.getNumDims() == 2, "nodeTokenIds must be [batch, verifySize].");

    int32_t const batchSize = static_cast<int32_t>(logitsShape[0]);
    int32_t const dflashBlockSize = static_cast<int32_t>(logitsShape[1]);
    int32_t const vocabSize = static_cast<int32_t>(logitsShape[2]);
    int32_t const verifySize = static_cast<int32_t>(nodeShape[1]);
    int32_t const packedMaskLen = (verifySize + kMaskBitsPerWord - 1) / kMaskBitsPerWord;

    int32_t const* draftVocabMappingTablePtr{nullptr};
    if (draftVocabMappingTable != nullptr)
    {
        validateGpuTensor(*draftVocabMappingTable, "draftVocabMappingTable", nvinfer1::DataType::kINT32, "INT32");
        check::check(draftVocabMappingTable->getShape().getNumDims() == 1, "draftVocabMappingTable must be 1D.");
        check::check(draftVocabMappingTable->getShape()[0] == vocabSize,
            "draftVocabMappingTable length must match draft logits vocabulary size.");
        draftVocabMappingTablePtr = draftVocabMappingTable->dataPointer<int32_t>();
    }

    check::check(batchSize > 0 && dflashBlockSize > 1 && vocabSize > 0, "Invalid DDTree logits shape.");
    check::check(verifySize > 0 && verifySize <= kDDTreeMaxVerifySize, "DDTree supports verifySize <= 128.");
    check::check(candidateTopK > 0 && candidateTopK <= kDDTreeMaxCandidateTopK,
        "DDTree candidateTopK must be in [1, " + std::to_string(kDDTreeMaxCandidateTopK) + "].");
    check::check(candidateTopK <= vocabSize, "DDTree candidateTopK must not exceed vocabSize.");
    check::check(rootTokenIds.getShape().getNumDims() == 1 && rootTokenIds.getShape()[0] == batchSize,
        "rootTokenIds must be [batch].");
    check::check(baseLengths.getShape().getNumDims() == 1 && baseLengths.getShape()[0] == batchSize,
        "baseLengths must be [batch].");
    check::check(nodeDepths.getShape() == nodeShape && parentIds.getShape() == nodeShape
            && nodeScores.getShape() == nodeShape && verifyTokenIds.getShape() == nodeShape
            && verifyPositionIds.getShape() == nodeShape,
        "DDTree node and verify tensors must share [batch, verifySize].");
    check::check(validCounts.getShape().getNumDims() == 1 && validCounts.getShape()[0] == batchSize,
        "validCounts must be [batch].");
    check::check(packedAncestorMask.getShape().getNumDims() == 3 && packedAncestorMask.getShape()[0] == batchSize
            && packedAncestorMask.getShape()[1] == verifySize && packedAncestorMask.getShape()[2] == packedMaskLen,
        "packedAncestorMask must be [batch, verifySize, ceil(verifySize / 32)].");
    check::check(ancestorMaskShape.getNumDims() == 3 && ancestorMaskShape[0] == batchSize
            && ancestorMaskShape[1] == verifySize && ancestorMaskShape[2] == verifySize,
        "ancestorMask must be [batch, verifySize, verifySize].");
    check::check(contextLengthsShape.getNumDims() == 1 && contextLengthsShape[0] == batchSize,
        "contextLengths must be [batch].");
    check::check(selectTokenIndicesShape.getNumDims() == 2 && selectTokenIndicesShape[0] == batchSize
            && selectTokenIndicesShape[1] == verifySize,
        "selectTokenIndices must be [batch, verifySize].");

    size_t const requiredWorkspaceSize
        = getDDTreeBuildWorkspaceSize(batchSize, dflashBlockSize, verifySize, vocabSize, candidateTopK);
    ELLM_CHECK(params.workspaceSize >= requiredWorkspaceSize,
        "DDTree build workspace too small. Required: " + std::to_string(requiredWorkspaceSize)
            + ", provided: " + std::to_string(params.workspaceSize));

    DDTreeBuildWorkspace buildWorkspace;
    buildWorkspace.setup(params.workspace, params.workspaceSize, batchSize, dflashBlockSize, candidateTopK);

    dim3 const candidateGrid(batchSize * dflashBlockSize);
    dim3 const candidateBlock(kCandidateBlockSize);
    selectDDTreeCandidatesKernel<<<candidateGrid, candidateBlock, 0, params.stream>>>(draftLogits.dataPointer<float>(),
        buildWorkspace.candidateTokenIds, buildWorkspace.candidateLogProbs, dflashBlockSize, vocabSize, candidateTopK);
    CUDA_CHECK(cudaGetLastError());

    size_t const buildSharedBytes = static_cast<size_t>(verifySize) * sizeof(int32_t);
    buildDDTreeKernel<<<batchSize, 1, buildSharedBytes, params.stream>>>(rootTokenIds.dataPointer<int32_t>(),
        baseLengths.dataPointer<int32_t>(), buildWorkspace.candidateTokenIds, buildWorkspace.candidateLogProbs,
        draftVocabMappingTablePtr, nodeTokenIds.dataPointer<int32_t>(), nodeDepths.dataPointer<int32_t>(),
        parentIds.dataPointer<int32_t>(), nodeScores.dataPointer<float>(), validCounts.dataPointer<int32_t>(),
        verifyTokenIds.dataPointer<int32_t>(), verifyPositionIds.dataPointer<int32_t>(),
        packedAncestorMask.dataPointer<int32_t>(), ancestorMask.dataPointer<int8_t>(),
        contextLengths.dataPointer<int32_t>(), selectTokenIndices.dataPointer<int64_t>(), dflashBlockSize, verifySize,
        candidateTopK);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
