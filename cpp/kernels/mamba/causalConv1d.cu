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

/*
 * This file contains code derived from causal-conv1d
 * (https://github.com/Dao-AILab/causal-conv1d)
 * Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
 * Licensed under the BSD 3-Clause License.
 *
 * Modifications by NVIDIA:
 * - Adapted causal depthwise conv1d kernel for TensorRT Edge-LLM integration
 * - Added stride, dilation, and padding parameters for generalized conv1d
 * - Added decode-mode kernel (conv_state dot weight)
 * - Added conv state capture and shift-insert kernels
 */

#include "causalConv1d.h"

#include "common/checkMacros.h"
#include "conversion.cuh"

#include <cuda_fp16.h>
#include <stdexcept>

namespace mamba_ssm
{

// Prefill causal conv1d: sliding window with device-adaptive seq-parallel.
// Maintains a shift register of width input values per thread, reading 1 new value per output
// instead of width. Uses gridDim.z to distribute contiguous chunks across SMs.
//
// Two variants: template kWidth for compile-time unroll, runtime width as fallback.
template <typename T, int32_t kWidth>
__global__ void causalConv1dKernelT(T const* __restrict__ x, T const* __restrict__ weight, T const* bias,
    T* __restrict__ out, int32_t seqLen, int32_t outSeqLen, int32_t dim, int32_t padding, int32_t const* contextLengths)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const effectiveSeqLen = contextLengths ? contextLengths[batchIdx] : seqLen;
    float const biasVal = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    float w[kWidth];
#pragma unroll
    for (int32_t k = 0; k < kWidth; ++k)
    {
        w[k] = conversion::toFloat(weight[static_cast<int64_t>(dimIdx) * kWidth + k]);
    }

    int64_t const batchOff = static_cast<int64_t>(batchIdx) * seqLen * dim;
    int64_t const outBatchOff = static_cast<int64_t>(batchIdx) * outSeqLen * dim;

    int32_t const zBlocks = static_cast<int32_t>(gridDim.z);
    int32_t const chunkSize = (outSeqLen + zBlocks - 1) / zBlocks;
    int32_t const chunkStart = static_cast<int32_t>(blockIdx.z) * chunkSize;
    int32_t const chunkEnd = chunkStart + chunkSize < outSeqLen ? chunkStart + chunkSize : outSeqLen;
    if (chunkStart >= outSeqLen)
    {
        return;
    }

    float xBuf[kWidth];
#pragma unroll
    for (int32_t k = 0; k < kWidth - 1; ++k)
    {
        int32_t const inPos = chunkStart + k - padding;
        xBuf[k] = (inPos >= 0 && inPos < effectiveSeqLen)
            ? conversion::toFloat(x[batchOff + static_cast<int64_t>(inPos) * dim + dimIdx])
            : 0.0F;
    }

    for (int32_t outPos = chunkStart; outPos < chunkEnd; ++outPos)
    {
        if (outPos >= effectiveSeqLen)
        {
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], 0.0F);
            xBuf[kWidth - 1] = 0.0F;
        }
        else
        {
            int32_t const newInPos = outPos + kWidth - 1 - padding;
            xBuf[kWidth - 1] = (newInPos >= 0 && newInPos < effectiveSeqLen)
                ? conversion::toFloat(x[batchOff + static_cast<int64_t>(newInPos) * dim + dimIdx])
                : 0.0F;

            float acc = biasVal;
#pragma unroll
            for (int32_t k = 0; k < kWidth; ++k)
            {
                acc += xBuf[k] * w[k];
            }
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], acc);
        }
#pragma unroll
        for (int32_t k = 0; k < kWidth - 1; ++k)
        {
            xBuf[k] = xBuf[k + 1];
        }
    }
}

// Runtime width fallback
template <typename T>
__global__ void causalConv1dKernel(T const* __restrict__ x, T const* __restrict__ weight, T const* bias,
    T* __restrict__ out, int32_t seqLen, int32_t outSeqLen, int32_t dim, int32_t width, int32_t padding,
    int32_t const* contextLengths)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const effectiveSeqLen = contextLengths ? contextLengths[batchIdx] : seqLen;
    float const biasVal = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    constexpr int32_t kMaxWidth = 8;
    float w[kMaxWidth];
    for (int32_t k = 0; k < width && k < kMaxWidth; ++k)
    {
        w[k] = conversion::toFloat(weight[static_cast<int64_t>(dimIdx) * width + k]);
    }

    int64_t const batchOff = static_cast<int64_t>(batchIdx) * seqLen * dim;
    int64_t const outBatchOff = static_cast<int64_t>(batchIdx) * outSeqLen * dim;

    // Contiguous chunk for this z-block
    int32_t const zBlocks = static_cast<int32_t>(gridDim.z);
    int32_t const chunkSize = (outSeqLen + zBlocks - 1) / zBlocks;
    int32_t const chunkStart = static_cast<int32_t>(blockIdx.z) * chunkSize;
    int32_t const chunkEnd = chunkStart + chunkSize < outSeqLen ? chunkStart + chunkSize : outSeqLen;
    if (chunkStart >= outSeqLen)
    {
        return;
    }

    // Pre-fill shift register
    float xBuf[kMaxWidth];
    for (int32_t k = 0; k < width - 1; ++k)
    {
        int32_t const inPos = chunkStart + k - padding;
        xBuf[k] = (inPos >= 0 && inPos < effectiveSeqLen)
            ? conversion::toFloat(x[batchOff + static_cast<int64_t>(inPos) * dim + dimIdx])
            : 0.0F;
    }

    for (int32_t outPos = chunkStart; outPos < chunkEnd; ++outPos)
    {
        if (outPos >= effectiveSeqLen)
        {
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], 0.0F);
            xBuf[width - 1] = 0.0F;
        }
        else
        {
            int32_t const newInPos = outPos + width - 1 - padding;
            xBuf[width - 1] = (newInPos >= 0 && newInPos < effectiveSeqLen)
                ? conversion::toFloat(x[batchOff + static_cast<int64_t>(newInPos) * dim + dimIdx])
                : 0.0F;

            float acc = biasVal;
#pragma unroll
            for (int32_t k = 0; k < width; ++k)
            {
                acc += xBuf[k] * w[k];
            }
            conversion::convertAndStore(&out[outBatchOff + static_cast<int64_t>(outPos) * dim + dimIdx], acc);
        }
#pragma unroll
        for (int32_t k = 0; k < width - 1; ++k)
        {
            xBuf[k] = xBuf[k + 1];
        }
    }
}

void invokeCausalConv1d(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor const& weight,
    trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out, int32_t stride, int32_t padding,
    int32_t dilation, trt_edgellm::rt::OptionalInputTensor contextLengths, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(x.getShape()[0]);
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[2]);
    int32_t const width = static_cast<int32_t>(weight.getShape()[2]);
    int32_t const outSeqLen = static_cast<int32_t>(out.getShape()[1]);

    ELLM_CHECK(x.getDataType() == nvinfer1::DataType::kHALF && weight.getDataType() == nvinfer1::DataType::kHALF
            && out.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");

    bool const isContiguous = (x.getStride(2) == 1 && x.getStride(1) == dim && out.getStride(2) == 1
        && out.getStride(1) == dim && weight.getStride(2) == 1);

    ELLM_CHECK(isContiguous && stride == 1 && dilation == 1 && width <= 8,
        "requires contiguous [B,S,D], stride=1, dilation=1, width<=8.");

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    uint32_t const dimBlocks = static_cast<uint32_t>((dim + kThreads - 1) / kThreads);

    // Adaptive seq-parallel: add z-blocks only when dim-blocks under-utilize the SMs
    int32_t smCount = 0;
    int32_t deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId);
    uint32_t seqBlocks = 1;
    if (dimBlocks < static_cast<uint32_t>(smCount) * 2)
    {
        seqBlocks = (static_cast<uint32_t>(smCount) * 4 + dimBlocks - 1) / dimBlocks;
    }
    if (seqBlocks > static_cast<uint32_t>(outSeqLen))
    {
        seqBlocks = static_cast<uint32_t>(outSeqLen);
    }
    dim3 const grid(batch, dimBlocks, seqBlocks);

    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    int32_t const* clPtr = contextLengths.has_value() ? contextLengths->get().dataPointer<int32_t>() : nullptr;
    half const* xPtr = x.dataPointer<half>();
    half const* wPtr = weight.dataPointer<half>();
    half* outPtr = out.dataPointer<half>();

    switch (width)
    {
    case 2:
        causalConv1dKernelT<half, 2>
            <<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, outPtr, seqLen, outSeqLen, dim, padding, clPtr);
        break;
    case 3:
        causalConv1dKernelT<half, 3>
            <<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, outPtr, seqLen, outSeqLen, dim, padding, clPtr);
        break;
    case 4:
        causalConv1dKernelT<half, 4>
            <<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, outPtr, seqLen, outSeqLen, dim, padding, clPtr);
        break;
    default:
        causalConv1dKernel<half>
            <<<grid, block, 0, stream>>>(xPtr, wPtr, biasPtr, outPtr, seqLen, outSeqLen, dim, width, padding, clPtr);
        break;
    }
    CUDA_CHECK(cudaPeekAtLastError());
}

// Capture last `width` time-steps from x into conv_state (transposed).
template <typename T>
__global__ void captureConvStateKernel(
    T const* x, T* convState, int32_t seqLen, int32_t dim, int32_t width, int32_t const* contextLengths)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int32_t const effectiveSeqLen = contextLengths ? contextLengths[batchIdx] : seqLen;
    int32_t const tailLen = (effectiveSeqLen >= width) ? width : effectiveSeqLen;
    int32_t const tailStart = effectiveSeqLen - tailLen;
    int32_t const dstOffset = width - tailLen;

    for (int32_t t = 0; t < tailLen; ++t)
    {
        int64_t const srcIdx = (static_cast<int64_t>(batchIdx) * seqLen + tailStart + t) * dim + dimIdx;
        int64_t const dstIdx = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width + dstOffset + t;
        convState[dstIdx] = x[srcIdx];
    }
}

void invokeCaptureConvState(trt_edgellm::rt::Tensor const& x, trt_edgellm::rt::Tensor& convState,
    trt_edgellm::rt::OptionalInputTensor contextLengths, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(x.getShape()[0]);
    int32_t const seqLen = static_cast<int32_t>(x.getShape()[1]);
    int32_t const dim = static_cast<int32_t>(x.getShape()[2]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    ELLM_CHECK(x.getDataType() == nvinfer1::DataType::kHALF && convState.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");

    size_t const elemSize = sizeof(half);
    CUDA_CHECK(cudaMemsetAsync(convState.rawPointer(), 0, static_cast<size_t>(batch) * dim * width * elemSize, stream));

    int32_t const* clPtr = contextLengths.has_value() ? contextLengths->get().dataPointer<int32_t>() : nullptr;
    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    captureConvStateKernel<half>
        <<<grid, block, 0, stream>>>(x.dataPointer<half>(), convState.dataPointer<half>(), seqLen, dim, width, clPtr);
    CUDA_CHECK(cudaPeekAtLastError());
}

// Decode kernel: shift conv_state left by 1, insert new column, then dot with weight + bias
template <typename T>
__global__ void causalConv1dDecodeKernel(
    T* convState, T const* newCol, T const* weight, T const* bias, T* output, int32_t dim, int32_t width)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int64_t const rowOffset = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width;
    int64_t const weightOffset = static_cast<int64_t>(dimIdx) * width;
    T* row = convState + rowOffset;

    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    // Shift left and compute dot product
    for (int32_t k = 0; k < width - 1; ++k)
    {
        T val = row[k + 1];
        row[k] = val;
        acc += conversion::toFloat(val) * conversion::toFloat(weight[weightOffset + k]);
    }
    // Insert new column and accumulate last weight element
    T newVal = newCol[static_cast<int64_t>(batchIdx) * dim + dimIdx];
    row[width - 1] = newVal;
    acc += conversion::toFloat(newVal) * conversion::toFloat(weight[weightOffset + width - 1]);

    int64_t const outIdx = static_cast<int64_t>(batchIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);
}

void invokeCausalConv1dDecode(trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::Tensor const& newCol,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    ELLM_CHECK(convState.getDataType() == nvinfer1::DataType::kHALF && newCol.getDataType() == nvinfer1::DataType::kHALF
            && weight.getDataType() == nvinfer1::DataType::kHALF && out.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;
    causalConv1dDecodeKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
        newCol.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(), dim, width);
    CUDA_CHECK(cudaPeekAtLastError());
}

// MTP decode kernel: process T draft tokens, shift+insert+dot per step, checkpoint state.
template <typename T>
__global__ void causalConv1dDecodeMTPKernel(T* convState, T const* newCols, T const* weight, T const* bias, T* output,
    T* intermediateConvStates, int32_t batch, int32_t dim, int32_t width, int32_t numTokens)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.y * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    // Load current conv_state row into registers (width is small, typically 4).
    int64_t const rowOffset = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width;
    float state[8]; // Max supported kernel width (compile-time upper bound).
    for (int32_t k = 0; k < width; ++k)
    {
        state[k] = conversion::toFloat(convState[rowOffset + k]);
    }

    float const biasVal = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;

    // Load weight into registers.
    int64_t const wOffset = static_cast<int64_t>(dimIdx) * width;
    float w[8];
    for (int32_t k = 0; k < width; ++k)
    {
        w[k] = conversion::toFloat(weight[wOffset + k]);
    }

    for (int32_t t = 0; t < numTokens; ++t)
    {
        // Shift state left by 1.
        for (int32_t k = 0; k < width - 1; ++k)
        {
            state[k] = state[k + 1];
        }
        // Insert new token at position width-1.
        int64_t const newColIdx = (static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx;
        state[width - 1] = conversion::toFloat(newCols[newColIdx]);

        // Dot product: output = conv_state · weight + bias.
        float acc = biasVal;
        for (int32_t k = 0; k < width; ++k)
        {
            acc += state[k] * w[k];
        }
        int64_t const outIdx = (static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx;
        conversion::convertAndStore(&output[outIdx], acc);

        // Checkpoint: save intermediate conv_state for rollback.
        // Layout: [batch, T, dim, width]
        int64_t const intermBase = ((static_cast<int64_t>(batchIdx) * numTokens + t) * dim + dimIdx) * width;
        for (int32_t k = 0; k < width; ++k)
        {
            conversion::convertAndStore(&intermediateConvStates[intermBase + k], state[k]);
        }
    }

    // Write final state back to convState.
    for (int32_t k = 0; k < width; ++k)
    {
        conversion::convertAndStore(&convState[rowOffset + k], state[k]);
    }
}

void invokeCausalConv1dDecodeMTP(trt_edgellm::rt::Tensor& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& intermediateConvStates, int32_t T, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);

    ELLM_CHECK(width <= 8, "kernel_size > 8 not supported.");
    ELLM_CHECK(convState.getDataType() == nvinfer1::DataType::kHALF && weight.getDataType() == nvinfer1::DataType::kHALF
            && out.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;

    causalConv1dDecodeMTPKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
        newCols.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(),
        intermediateConvStates.dataPointer<half>(), batch, dim, width, T);
    CUDA_CHECK(cudaPeekAtLastError());
}

// DDTree decode kernel: each node independently reconstructs its conv window by walking
// parent_ids back to root and appending the full root-to-node token path. This avoids
// cross-node synchronization and still executes all tree nodes in one launch.
template <typename T>
__global__ void causalConv1dDecodeDDTreeKernel(T const* __restrict__ convState, T const* __restrict__ newCols,
    T const* __restrict__ weight, T const* __restrict__ bias, T* __restrict__ output, T* __restrict__ convStateOut,
    T* __restrict__ intermediateConvStates, int32_t const* __restrict__ treeParentIds,
    int32_t const* __restrict__ treeDepths, int32_t dim, int32_t width, int32_t verifySeq)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const nodeIdx = blockIdx.y;
    int32_t const dimIdx = static_cast<int32_t>(blockIdx.z * blockDim.x + threadIdx.x);
    if (dimIdx >= dim)
    {
        return;
    }

    int64_t const treeOffset = static_cast<int64_t>(batchIdx) * verifySeq;
    int64_t const stateRowOffset = (static_cast<int64_t>(batchIdx) * dim + dimIdx) * width;
    int64_t const weightOffset = static_cast<int64_t>(dimIdx) * width;

    int32_t const parentIdx = treeParentIds[treeOffset + nodeIdx];
    int32_t const depth = treeDepths[treeOffset + nodeIdx];
    bool const isRoot = nodeIdx == 0 && parentIdx < 0 && depth == 0;
    bool const isValidChild = nodeIdx > 0 && parentIdx >= 0 && parentIdx < nodeIdx && depth > 0;
    bool const isValidNode = isRoot || isValidChild;

    float state[8];
    if (!isValidNode)
    {
        for (int32_t k = 0; k < width; ++k)
        {
            state[k] = conversion::toFloat(convState[stateRowOffset + k]);
        }

        int64_t const outIdx = (treeOffset + nodeIdx) * dim + dimIdx;
        conversion::convertAndStore(&output[outIdx], 0.0F);
        int64_t const intermediateOffset = ((treeOffset + nodeIdx) * dim + dimIdx) * width;
        for (int32_t k = 0; k < width; ++k)
        {
            conversion::convertAndStore(&intermediateConvStates[intermediateOffset + k], state[k]);
        }
        return;
    }

    int32_t pathNodes[8];
    int32_t pathLen{0};
    // The conv window only depends on the latest `width` tokens.  For deeper
    // tree nodes, truncate from the root side and keep the most recent path.
    int32_t const maxPathLen = (depth + 1 < width) ? depth + 1 : width;
    int32_t currentNode = nodeIdx;
    while (pathLen < maxPathLen && currentNode >= 0 && currentNode < verifySeq)
    {
        pathNodes[pathLen] = currentNode;
        ++pathLen;
        if (currentNode == 0)
        {
            break;
        }
        currentNode = treeParentIds[treeOffset + currentNode];
    }

    for (int32_t k = 0; k < width; ++k)
    {
        if (k + pathLen < width)
        {
            state[k] = conversion::toFloat(convState[stateRowOffset + k + pathLen]);
        }
        else
        {
            state[k] = 0.0F;
        }
    }
    for (int32_t pathOffset = 0; pathOffset < pathLen; ++pathOffset)
    {
        int32_t const pathNode = pathNodes[pathOffset];
        int64_t const newColIdx = (treeOffset + pathNode) * dim + dimIdx;
        state[width - 1 - pathOffset] = conversion::toFloat(newCols[newColIdx]);
    }

    float acc = (bias != nullptr) ? conversion::toFloat(bias[dimIdx]) : 0.0F;
    for (int32_t k = 0; k < width; ++k)
    {
        acc += state[k] * conversion::toFloat(weight[weightOffset + k]);
    }
    int64_t const outIdx = (treeOffset + nodeIdx) * dim + dimIdx;
    conversion::convertAndStore(&output[outIdx], acc);

    int64_t const intermediateOffset = ((treeOffset + nodeIdx) * dim + dimIdx) * width;
    for (int32_t k = 0; k < width; ++k)
    {
        conversion::convertAndStore(&intermediateConvStates[intermediateOffset + k], state[k]);
    }

    // Only the root Y-slice copies convStateOut.  grid.z covers every dim, so
    // this is a single race-free copy of [batch, dim, width] per batch item.
    if (nodeIdx == 0)
    {
        for (int32_t k = 0; k < width; ++k)
        {
            convStateOut[stateRowOffset + k] = convState[stateRowOffset + k];
        }
    }
}

void invokeCausalConv1dDecodeDDTree(trt_edgellm::rt::Tensor const& convState, trt_edgellm::rt::Tensor const& newCols,
    trt_edgellm::rt::Tensor const& weight, trt_edgellm::rt::OptionalInputTensor bias, trt_edgellm::rt::Tensor& out,
    trt_edgellm::rt::Tensor& convStateOut, trt_edgellm::rt::Tensor& intermediateConvStates,
    trt_edgellm::rt::Tensor const& treeParentIds, trt_edgellm::rt::Tensor const& treeDepths, cudaStream_t stream)
{
    int32_t const batch = static_cast<int32_t>(convState.getShape()[0]);
    int32_t const dim = static_cast<int32_t>(convState.getShape()[1]);
    int32_t const width = static_cast<int32_t>(convState.getShape()[2]);
    int32_t const verifySeq = static_cast<int32_t>(newCols.getShape()[1]);

    ELLM_CHECK(width <= 8, "kernel_size > 8 not supported.");
    ELLM_CHECK(convState.getDataType() == nvinfer1::DataType::kHALF
            && newCols.getDataType() == nvinfer1::DataType::kHALF && weight.getDataType() == nvinfer1::DataType::kHALF
            && out.getDataType() == nvinfer1::DataType::kHALF && convStateOut.getDataType() == nvinfer1::DataType::kHALF
            && intermediateConvStates.getDataType() == nvinfer1::DataType::kHALF,
        "only FP16 (half) is supported.");
    ELLM_CHECK(treeParentIds.getDataType() == nvinfer1::DataType::kINT32
            && treeDepths.getDataType() == nvinfer1::DataType::kINT32,
        "tree_parent_ids/tree_depths must be INT32.");
    ELLM_CHECK(newCols.getShape()[0] == batch && newCols.getShape()[2] == dim, "newCols must be [B, S, D].");
    ELLM_CHECK(weight.getShape()[0] == dim && weight.getShape()[2] == width, "weight must be [D, 1, W].");
    ELLM_CHECK(out.getShape()[0] == batch && out.getShape()[1] == verifySeq && out.getShape()[2] == dim,
        "out must be [B, S, D].");
    ELLM_CHECK(
        convStateOut.getShape()[0] == batch && convStateOut.getShape()[1] == dim && convStateOut.getShape()[2] == width,
        "convStateOut must be [B, D, W].");
    ELLM_CHECK(intermediateConvStates.getShape()[0] == batch && intermediateConvStates.getShape()[1] == verifySeq
            && intermediateConvStates.getShape()[2] == dim && intermediateConvStates.getShape()[3] == width,
        "intermediateConvStates must be [B, S, D, W].");
    ELLM_CHECK(treeParentIds.getShape()[0] == batch && treeParentIds.getShape()[1] == verifySeq,
        "treeParentIds must be [B, S].");
    ELLM_CHECK(
        treeDepths.getShape()[0] == batch && treeDepths.getShape()[1] == verifySeq, "treeDepths must be [B, S].");

    int32_t constexpr kThreads = 256;
    dim3 const block(kThreads);
    dim3 const grid(batch, verifySeq, static_cast<uint32_t>((dim + kThreads - 1) / kThreads));
    half const* biasPtr = bias.has_value() ? bias->get().dataPointer<half>() : nullptr;

    causalConv1dDecodeDDTreeKernel<half><<<grid, block, 0, stream>>>(convState.dataPointer<half>(),
        newCols.dataPointer<half>(), weight.dataPointer<half>(), biasPtr, out.dataPointer<half>(),
        convStateOut.dataPointer<half>(), intermediateConvStates.dataPointer<half>(),
        treeParentIds.dataPointer<int32_t>(), treeDepths.dataPointer<int32_t>(), dim, width, verifySeq);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace mamba_ssm
