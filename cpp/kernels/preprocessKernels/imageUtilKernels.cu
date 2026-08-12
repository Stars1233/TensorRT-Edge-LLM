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
#include "imageUtilKernels.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace kernel
{

__global__ void normalizeImageKernel(unsigned char const* originalImage, float const* mean, float const* std,
    half* normalizedImage, int64_t const batch, int64_t const height, int64_t const width, int64_t const channels)
{
    // Each thread processes one pixel
    // originalImage format: [batch, height, width, channels]
    // normalizedImage format: [batch, height, width, channels]
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const totalPixels = batch * height * width * channels;
    if (tid >= totalPixels)
        return;

    auto const channel = tid % channels;
    unsigned char val = originalImage[tid];
    float normalized = (val / 255.0f - mean[channel]) / std[channel];
    normalizedImage[tid] = __float2half(normalized);
}

void normalizeImage(rt::Tensor const& originalImage, rt::Tensor const& mean, rt::Tensor const& std,
    rt::Tensor& normalizedImage, cudaStream_t stream)
{
    check::check(originalImage.getDeviceType() == rt::DeviceType::kGPU && mean.getDeviceType() == rt::DeviceType::kGPU
            && std.getDeviceType() == rt::DeviceType::kGPU && normalizedImage.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kUINT8 && mean.getDataType() == DataType::kFLOAT
            && std.getDataType() == DataType::kFLOAT && normalizedImage.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && normalizedImage.getShape().getNumDims() == 4,
        "Input and output tensor shapes shall be [batch, height, width, channels] and [batch, height, width, channels] "
        "respectively.");

    int64_t const batch = originalImage.getShape()[0];
    int64_t const height = originalImage.getShape()[1];
    int64_t const width = originalImage.getShape()[2];
    int64_t const channels = originalImage.getShape()[3];
    int64_t const totalPixels = batch * height * width * channels;
    check::check(
        channels == mean.getShape()[0] && channels == std.getShape()[0], "Channels mismatch for mean and std.");

    // Each CTA get assigned 256 threads.
    uint32_t const blockSize = 256;
    uint32_t const gridSize = static_cast<uint32_t>((totalPixels + blockSize - 1) / blockSize);

    normalizeImageKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<unsigned char>(),
        mean.dataPointer<float>(), std.dataPointer<float>(), normalizedImage.dataPointer<half>(), batch, height, width,
        channels);
}

__global__ void transposeToPatchQwenKernel(half const* originalImage, half* inputPatches, int64_t const T,
    int64_t const H, int64_t const W, int64_t const C, int64_t const temporalPatchSize, int64_t const patchSize,
    int64_t const mergeSize, int64_t const inputOffset)
{
    // This is a naive implementation of 9D transpose.
    // Each CTA get assigned 256 threads. Each thread processes one element
    // Original image format: [T, H, W, C]
    //      T = gridT * temporalPatchSize
    //      H = gridH * mergeSize * patchSize
    //      W = gridW * mergeSize * patchSize
    //      C = channels
    // Transposed format: [seqLength, inputDim]
    //      seqLength = gridT * gridH * gridW * mergeSize * mergeSize
    //      inputDim = C * temporalPatchSize * patchSize * patchSize

    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto const gridT = T / temporalPatchSize;
    auto const gridH = H / (mergeSize * patchSize);
    auto const gridW = W / (mergeSize * patchSize);
    auto const seqLength = gridT * gridH * gridW * mergeSize * mergeSize;
    auto const inputDim = C * temporalPatchSize * patchSize * patchSize;
    auto const totalElements = seqLength * inputDim;

    if (tid >= totalElements)
        return;

    // Calculate which sequence and element this thread handles
    auto const seqIdx = tid / inputDim;
    auto const elemIdx = tid % inputDim;

    // Calculate sequence coordinates
    auto const tIdx = seqIdx / (gridH * gridW * mergeSize * mergeSize);
    auto const hIdx = (seqIdx % (gridH * gridW * mergeSize * mergeSize)) / (gridW * mergeSize * mergeSize);
    auto const wIdx = (seqIdx % (gridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    auto const mergeH = (seqIdx % (mergeSize * mergeSize)) / mergeSize;
    auto const mergeW = seqIdx % mergeSize;

    // Calculate coordinates within the patch
    auto const cIdx = elemIdx / (temporalPatchSize * patchSize * patchSize);
    auto const tPatchIdx = (elemIdx % (temporalPatchSize * patchSize * patchSize)) / (patchSize * patchSize);
    auto const patchH = (elemIdx % (patchSize * patchSize)) / patchSize;
    auto const patchW = elemIdx % patchSize;

    // Calculate source coordinates
    auto const srcT = tIdx * temporalPatchSize + tPatchIdx;
    auto const srcH = hIdx * mergeSize * patchSize + mergeH * patchSize + patchH;
    auto const srcW = wIdx * mergeSize * patchSize + mergeW * patchSize + patchW;
    auto const srcC = cIdx;

    // Calculate indices
    auto const srcIdx = srcT * H * W * C + srcH * W * C + srcW * C + srcC;
    auto const dstIdx = inputOffset + seqIdx * inputDim + elemIdx;

    // Direct copy (coalesced write, strided read)
    inputPatches[dstIdx] = originalImage[srcIdx];
}

__global__ void transposeToPatchGemma4Kernel(half const* originalImage, half* inputPatches, int64_t const H,
    int64_t const W, int64_t const C, int64_t const patchSize, int64_t const inputOffset)
{
    // Gemma4 patchification: channel-last within each patch
    // Original image format: [1, H, W, C]
    // Output format: [numPatches, patchSize * patchSize * C]
    //   where element order within each patch is [patchH, patchW, C] (channels fastest)
    // This matches HuggingFace Gemma4 convert_image_to_patches():
    //   image.reshape(C, pH, ps, pW, ps).permute(1,3,2,4,0).reshape(pH*pW, ps*ps*C)

    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto const gridH = H / patchSize;
    auto const gridW = W / patchSize;
    auto const numPatches = gridH * gridW;
    auto const inputDim = patchSize * patchSize * C;
    auto const totalElements = numPatches * inputDim;

    if (tid >= totalElements)
        return;

    // Calculate which patch and element within patch
    auto const patchIdx = tid / inputDim;
    auto const elemIdx = tid % inputDim;

    // Patch grid coordinates
    auto const hIdx = patchIdx / gridW;
    auto const wIdx = patchIdx % gridW;

    // Element coordinates within patch: [patchH, patchW, C] ordering
    auto const patchH = elemIdx / (patchSize * C);
    auto const patchW = (elemIdx % (patchSize * C)) / C;
    auto const cIdx = elemIdx % C;

    // Source coordinates in [H, W, C] image
    auto const srcH = hIdx * patchSize + patchH;
    auto const srcW = wIdx * patchSize + patchW;
    auto const srcIdx = srcH * W * C + srcW * C + cIdx;
    auto const dstIdx = inputOffset + tid;

    inputPatches[dstIdx] = originalImage[srcIdx];
}

void transposeToPatchGemma4ViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const patchSize, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 2,
        "Input and output tensor shapes shall be [1, H, W, C] and [totalSeqLength, inputDim] respectively.");

    int64_t const H = originalImage.getShape()[1];
    int64_t const W = originalImage.getShape()[2];
    int64_t const C = originalImage.getShape()[3];
    int64_t const inputDim = inputPatches.getShape()[1];

    check::check(inputDim == patchSize * patchSize * C,
        "inputDim must equal patchSize * patchSize * C: inputDim=" + std::to_string(inputDim)
            + ", patchSize*patchSize*C=" + std::to_string(patchSize * patchSize * C));
    check::check(H % patchSize == 0 && W % patchSize == 0, "H and W must be multiples of patchSize");

    int64_t const gridH = H / patchSize;
    int64_t const gridW = W / patchSize;
    int64_t const totalElements = gridH * gridW * inputDim;
    check::check(inputOffset >= 0 && inputOffset + totalElements <= inputPatches.getShape().volume(),
        "inputOffset + totalElements must fit inside inputPatches: inputOffset=" + std::to_string(inputOffset)
            + ", totalElements=" + std::to_string(totalElements)
            + ", capacity=" + std::to_string(inputPatches.getShape().volume()));

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchGemma4Kernel<<<gridSize, blockSize, 0, stream>>>(
        originalImage.dataPointer<half>(), inputPatches.dataPointer<half>(), H, W, C, patchSize, inputOffset);
}

void transposeToPatchQwenViT(rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset,
    int64_t const temporalPatchSize, int64_t const patchSize, int64_t const mergeSize, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 2,
        "Input and output tensor shapes shall be [T, H, W, C] and [totalSeqLength, inputDim] respectively.");
    // Get tensor dimensions
    int64_t const T = originalImage.getShape()[0];
    int64_t const H = originalImage.getShape()[1];
    int64_t const W = originalImage.getShape()[2];
    int64_t const C = originalImage.getShape()[3];
    int64_t const inputDim = inputPatches.getShape()[1];
    int64_t const totalElements = T * H * W * C;

    // Assertions for dimension assumptions
    check::check(inputDim == C * temporalPatchSize * patchSize * patchSize,
        "inputDim must be equal to C * temporalPatchSize * patchSize * patchSize: inputDim=" + std::to_string(inputDim)
            + ", C * temporalPatchSize * patchSize * patchSize="
            + std::to_string(C * temporalPatchSize * patchSize * patchSize));
    check::check(T % temporalPatchSize == 0,
        "T must be multiple of temporalPatchSize: T=" + std::to_string(T)
            + ", temporalPatchSize=" + std::to_string(temporalPatchSize));
    check::check(H % (mergeSize * patchSize) == 0,
        "H must be multiple of mergeSize * patchSize: H=" + std::to_string(H)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));
    check::check(W % (mergeSize * patchSize) == 0,
        "W must be multiple of mergeSize * patchSize: W=" + std::to_string(W)
            + ", mergeSize * patchSize=" + std::to_string(mergeSize * patchSize));

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchQwenKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<half>(),
        inputPatches.dataPointer<half>(), T, H, W, C, temporalPatchSize, patchSize, mergeSize, inputOffset);
}

__global__ void transposeToPatchInternVLPhi4MMKernel(half const* originalImage, half* inputPatches,
    int64_t const inputOffset, int64_t const height, int64_t const width, int64_t const channels,
    int64_t const blockImageSizeH, int64_t const blockImageSizeW)
{
    // This is a naive implementation of 5D transpose.
    // Each CTA get assigned 256 threads. Each thread processes one element
    // Original image format: [1, H, W, C]
    //      H = gridH * blockImageSizeH
    //      W = gridW * blockImageSizeW
    //      C = channels
    // Transposed format: [gridH * gridW, channels, blockSizeH, blockSizeW]

    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const gridH = height / blockImageSizeH;
    auto const gridW = width / blockImageSizeW;
    auto const numBlocks = gridH * gridW;
    auto const totalElements = numBlocks * channels * blockImageSizeH * blockImageSizeW;

    if (tid >= totalElements)
        return;

    // Calculate indices
    auto const gridHIdx = tid / (gridW * channels * blockImageSizeH * blockImageSizeW);
    auto const gridWIdx = (tid % (gridW * channels * blockImageSizeH * blockImageSizeW))
        / (channels * blockImageSizeH * blockImageSizeW);
    auto const cIdx = (tid % (channels * blockImageSizeH * blockImageSizeW)) / (blockImageSizeH * blockImageSizeW);
    auto const blockHIdx = (tid % (blockImageSizeH * blockImageSizeW)) / blockImageSizeW;
    auto const blockWIdx = tid % blockImageSizeW;

    auto const srcHIdx = gridHIdx * blockImageSizeH + blockHIdx;
    auto const srcWIdx = gridWIdx * blockImageSizeW + blockWIdx;
    auto const srcCIdx = cIdx;
    auto const srcIdx = srcHIdx * width * channels + srcWIdx * channels + srcCIdx;
    auto const dstIdx = inputOffset + tid;

    // Direct copy (coalesced write, strided read)
    inputPatches[dstIdx] = originalImage[srcIdx];
}

void transposeToPatchInternVLPhi4MM(
    rt::Tensor const& originalImage, rt::Tensor& inputPatches, int64_t const inputOffset, cudaStream_t stream)
{
    check::check(
        originalImage.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(originalImage.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(originalImage.getShape().getNumDims() == 4 && inputPatches.getShape().getNumDims() == 4,
        "Input and output tensor shapes shall be [1, height, width, channels] and [totalNumBlocks, channels, "
        "blockSizeH, blockSizeW] respectively.");
    check::check(originalImage.getShape()[0] == 1, "Original image shape shall be [1, height, width, channels].");

    int64_t const height = originalImage.getShape()[1];
    int64_t const width = originalImage.getShape()[2];
    int64_t const channels = originalImage.getShape()[3];
    int64_t const blockSizeH = inputPatches.getShape()[2];
    int64_t const blockSizeW = inputPatches.getShape()[3];
    int64_t const totalElements = height * width * channels;

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchInternVLPhi4MMKernel<<<gridSize, blockSize, 0, stream>>>(originalImage.dataPointer<half>(),
        inputPatches.dataPointer<half>(), inputOffset, height, width, channels, blockSizeH, blockSizeW);
}

// Phi4MM Pack All Batched Kernel
namespace
{
// copyHiddenVec
// Purpose:
//   Efficiently copy one token vector of length `hidden` (FP16) from src → dst.
//   Uses vectorized loads/stores (kernel::DVec<half>) to maximize memory
//   throughput on the hidden dimension which is contiguous in memory.
//
// Execution model:
//   - All threads in the CTA cooperate to copy one token:
//       each thread handles chunks in a round-robin fashion with stride = blockDim.x.
//   - Vector width V is chosen by DVec<half>::vec_size (typically 8 halves).
//   - Tail elements (< V) are copied by thread 0 to avoid race conditions.
//
// Rationale:
//   In all pack kernels below, we write tokens consecutively in the output,
//   making writes coalesced. Reads are strided because different tokens are
//   gathered, so vectorizing the hidden copy minimizes the cost of those reads.
__device__ __forceinline__ void copyHiddenVec(
    half const* __restrict__ src, half* __restrict__ dst, int32_t hidden, int32_t threadStride, int32_t threadIdxX)
{
    // Vectorized copy in chunks of 8 halves
    constexpr int32_t V = kernel::DVec<half>::vec_size;
    int32_t const numChunks = hidden / V;
    for (int32_t chunk = threadIdxX; chunk < numChunks; chunk += threadStride)
    {
        kernel::DVec<half> vec;
        vec.load(src + chunk * V);
        vec.store(dst + chunk * V);
    }
    // Tail copy by thread 0
    int32_t const tail = hidden % V;
    if (threadIdxX == 0)
    {
        int32_t const base = numChunks * V;
        for (int32_t i = 0; i < tail; ++i)
        {
            dst[base + i] = src[base + i];
        }
    }
}

// binarySearchImage
// Purpose:
//   Given a global output token index `tokenIdx` and an array `outStart` of size
//   `numImages` where each element denotes the starting output token offset of
//   an image, find the image index that owns `tokenIdx`.
//
// Contract/assumptions:
//   - outStart is monotonically non-decreasing.
//   - The i-th image covers output indices in [outStart[i], outStart[i+1]) for i < numImages-1,
//     and [outStart[numImages-1], totalOutTokens) for the last image.
//
// Returns:
//   The greatest index i such that outStart[i] <= tokenIdx.
__device__ __forceinline__ int32_t binarySearchImage(
    int64_t const* __restrict__ outStart, int32_t numImages, int64_t tokenIdx)
{
    int32_t lo = 0;
    int32_t hi = numImages - 1;
    int32_t ans = numImages - 1;
    while (lo <= hi)
    {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (outStart[mid] <= tokenIdx)
        {
            ans = mid;
            lo = mid + 1;
        }
        else
        {
            hi = mid - 1;
        }
    }
    return ans;
}
} // namespace

__global__ void phi4mmPostprocessVisionTokensKernel(
    half const* __restrict__ src, half* __restrict__ dst, Phi4MMIndex idx, Phi4MMGN gn)
{
    int64_t tokenIdx = static_cast<int64_t>(blockIdx.x);
    if (tokenIdx >= idx.totalOutTokens)
    {
        return;
    }

    int32_t const img = binarySearchImage(idx.dstOutStart, idx.numImages, tokenIdx);
    int64_t const localIdx = tokenIdx - idx.dstOutStart[img];

    int64_t const subLen = idx.subOutLen[img];

    half* dstPtr = dst + tokenIdx * idx.hidden;

    if (localIdx < subLen)
    {
        // sub segment
        int32_t const wb = idx.wBlocks[img];
        int64_t const cols = kTokensPerSidePhi4 * wb;
        int64_t const strideOut = cols + 1;
        int64_t const r = localIdx / strideOut;
        int64_t const c = localIdx % strideOut;
        if (c == cols)
        {
            copyHiddenVec(gn.subGN, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
            return;
        }
        int64_t const bRow = r / kTokensPerSidePhi4;
        int64_t const pRow = r % kTokensPerSidePhi4;
        int64_t const bCol = c / kTokensPerSidePhi4;
        int64_t const pCol = c % kTokensPerSidePhi4;
        int64_t const blockId = bRow * wb + bCol;
        int64_t const patchId = pRow * kTokensPerSidePhi4 + pCol;
        int64_t const srcTokIndex = idx.srcSubStart[img] + blockId * kTokensPerBlockPhi4 + patchId;
        half const* srcPtr = src + srcTokIndex * idx.hidden;
        copyHiddenVec(srcPtr, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
        return;
    }
    else if (localIdx == subLen)
    {
        // single glb_GN
        copyHiddenVec(gn.glbGN, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
        return;
    }
    else
    {
        // glb segment
        int64_t const idx2 = localIdx - (subLen + 1);
        int64_t const cols = kTokensPerSidePhi4;
        int64_t const strideOut = cols + 1;
        int64_t const r = idx2 / strideOut;
        int64_t const c = idx2 % strideOut;
        if (c == cols)
        {
            copyHiddenVec(gn.subGN, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
            return;
        }
        int64_t const srcTokIndex = idx.srcGlbStart[img] + r * kTokensPerSidePhi4 + c;
        half const* srcPtr = src + srcTokIndex * idx.hidden;
        copyHiddenVec(srcPtr, dstPtr, idx.hidden, blockDim.x, threadIdx.x);
        return;
    }
}

void phi4mmPostprocessVisionTokens(rt::Tensor const& srcEmbedding, rt::Tensor& dstEmbedding, Phi4MMIndex const& indices,
    Phi4MMGN const& gn, int64_t totalOutTokens, cudaStream_t stream)
{
    check::check(
        srcEmbedding.getDeviceType() == rt::DeviceType::kGPU && dstEmbedding.getDeviceType() == rt::DeviceType::kGPU,
        "phi4mmPostprocessVisionTokens(): All tensors must be on GPU.");
    check::check(srcEmbedding.getDataType() == DataType::kHALF && dstEmbedding.getDataType() == DataType::kHALF,
        "phi4mmPostprocessVisionTokens(): Embeddings and dstEmbedding must be FP16.");

    int32_t const hidden = static_cast<int32_t>(srcEmbedding.getShape()[1]);
    check::check(hidden == dstEmbedding.getShape()[1],
        "phi4mmPostprocessVisionTokens(): srcEmbedding and dstEmbedding must have the same hidden size.");

    // Require enough space for totalOutTokens * hidden elements of dstEmbedding's data type.
    int64_t const bytesPerElem = static_cast<int64_t>(rt::utils::getTypeSize(dstEmbedding.getDataType()));
    int64_t const requiredBytes = totalOutTokens * static_cast<int64_t>(hidden) * bytesPerElem;
    check::check(requiredBytes <= dstEmbedding.getMemoryCapacity(),
        "phi4mmPostprocessVisionTokens(): Total output tokens exceed dstEmbedding memory capacity.");

    dim3 block(128);
    dim3 grid(static_cast<uint32_t>(totalOutTokens));
    phi4mmPostprocessVisionTokensKernel<<<grid, block, 0, stream>>>(
        srcEmbedding.dataPointer<half>(), dstEmbedding.dataPointer<half>(), indices, gn);
}

__global__ void initRotaryPosEmbQwenKernel(float* rotaryPosEmb, int64_t const T, int64_t const H, int64_t const W,
    int64_t const mergeSize, int64_t const startIdx, int64_t const vitPosEmbDim, float const rotaryBaseFrequency,
    float const scale)
{
    // Each CTA get assigned 256 threads. Each thread processes one element
    // rotaryPosEmb: [totalSeqLength, vitPosEmbDim]
    //     [T, (llmGridH, llmGridW, mergeSize, mergeSize), (2, vitPosEmbDim/2)]
    //     where llmGridH = H / mergeSize, llmGridW = W / mergeSize and position ids is duplicated for T
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const totalElements = T * H * W * vitPosEmbDim;
    if (tid >= totalElements)
        return;

    auto const hwIdx = (tid % (H * W * vitPosEmbDim)) / (vitPosEmbDim);
    auto const hOrWPos = (tid % (vitPosEmbDim)) / (vitPosEmbDim / 2);
    auto const dimIdx = tid % (vitPosEmbDim / 2);

    int64_t const llmGridW = W / mergeSize;
    auto const llmGridHIdx = hwIdx / (llmGridW * mergeSize * mergeSize);
    auto const llmGridWIdx = (hwIdx % (llmGridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    auto const mergeHIdx = (hwIdx % (mergeSize * mergeSize)) / mergeSize;
    auto const mergeWIdx = hwIdx % mergeSize;

    auto const originalHIdx = llmGridHIdx * mergeSize + mergeHIdx;
    auto const originalWIdx = llmGridWIdx * mergeSize + mergeWIdx;
    // 0: H, 1: W
    auto const posId = (hOrWPos == 0) ? originalHIdx : originalWIdx;

    float invFreq = posId * scale / pow(rotaryBaseFrequency, 2 * dimIdx / (float) vitPosEmbDim);
    rotaryPosEmb[startIdx * vitPosEmbDim + tid] = invFreq;
}

void initRotaryPosEmbQwenViT(rt::Tensor& rotaryPosEmb, std::vector<int64_t> const& gridTHW, int64_t const mergeSize,
    int64_t const startIdx, float const rotaryBaseFrequency, float const scale, cudaStream_t stream)
{
    check::check(rotaryPosEmb.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall be GPU for the rotary position embeddings tensor.");
    check::check(rotaryPosEmb.getDataType() == DataType::kFLOAT,
        "Data type shall be float for the rotary position embeddings tensor.");
    check::check(rotaryPosEmb.getShape().getNumDims() == 2,
        "Rotary position embeddings shape shall be [totalSeqLength, vitPosEmbDim].");

    check::check(gridTHW.size() == 3, "gridTHW must have exactly 3 elements [T, H, W]");
    int64_t const T = gridTHW[0];
    int64_t const H = gridTHW[1];
    int64_t const W = gridTHW[2];

    int64_t const vitPosEmbDim = rotaryPosEmb.getShape()[1];
    int64_t const totalElements = T * H * W * vitPosEmbDim;

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    initRotaryPosEmbQwenKernel<<<gridSize, blockSize, 0, stream>>>(
        rotaryPosEmb.dataPointer<float>(), T, H, W, mergeSize, startIdx, vitPosEmbDim, rotaryBaseFrequency, scale);
}

__global__ void initRotaryPosEmbGemma4Kernel(float* rotaryPosEmb, int64_t const* pixelPositionIds,
    int64_t const totalSeqLength, int64_t const headDim, float const rotaryBaseFrequency)
{
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const totalElements = totalSeqLength * headDim;
    if (tid >= totalElements)
        return;

    int64_t const tokenIdx = tid / headDim;
    int64_t const dimIdx = tid % headDim;
    int64_t const axisDim = headDim / 2;
    int64_t const axis = dimIdx / axisDim;
    int64_t const dimInAxis = dimIdx % axisDim;
    int64_t const freqIdx = dimInAxis % (axisDim / 2);
    int64_t const posId = pixelPositionIds[tokenIdx * 2 + axis];

    float const exponent = 2.0F * static_cast<float>(freqIdx) / static_cast<float>(axisDim);
    rotaryPosEmb[tid] = static_cast<float>(posId) / powf(rotaryBaseFrequency, exponent);
}

void initRotaryPosEmbGemma4ViT(
    rt::Tensor& rotaryPosEmb, rt::Tensor const& pixelPositionIds, float rotaryBaseFrequency, cudaStream_t stream)
{
    check::check(rotaryPosEmb.getDeviceType() == rt::DeviceType::kGPU
            && pixelPositionIds.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall be GPU for Gemma4 rotary position tensors.");
    check::check(rotaryPosEmb.getDataType() == DataType::kFLOAT && pixelPositionIds.getDataType() == DataType::kINT64,
        "Data type check failed for Gemma4 rotary position tensors.");
    check::check(rotaryPosEmb.getShape().getNumDims() == 2,
        "Gemma4 rotary position embeddings shape shall be [totalSeqLength, headDim].");
    check::check(pixelPositionIds.getShape().getNumDims() == 2 && pixelPositionIds.getShape()[1] == 2,
        "Gemma4 pixel position ids shape shall be [totalSeqLength, 2].");
    check::check(rotaryPosEmb.getShape()[0] == pixelPositionIds.getShape()[0],
        "Gemma4 rotary position embeddings and pixel position ids must have the same sequence length.");

    int64_t const totalSeqLength = rotaryPosEmb.getShape()[0];
    int64_t const headDim = rotaryPosEmb.getShape()[1];
    check::check(headDim % 4 == 0, "Gemma4 RoPE headDim must be divisible by 4.");

    uint32_t const blockSize = 256;
    uint32_t const gridSize = static_cast<uint32_t>((totalSeqLength * headDim + blockSize - 1) / blockSize);
    initRotaryPosEmbGemma4Kernel<<<gridSize, blockSize, 0, stream>>>(rotaryPosEmb.dataPointer<float>(),
        pixelPositionIds.dataPointer<int64_t>(), totalSeqLength, headDim, rotaryBaseFrequency);
}

__global__ void initPoolingWeightsGemma4Kernel(half* poolingWeights, int64_t const totalPatches,
    int64_t const patchStart, int64_t const softStart, int64_t const patchHeight, int64_t const patchWidth,
    int64_t const poolingKernelSize)
{
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const curPatches = patchHeight * patchWidth;
    if (tid >= curPatches)
        return;

    int64_t const y = tid / patchWidth;
    int64_t const x = tid % patchWidth;
    int64_t const softWidth = patchWidth / poolingKernelSize;
    int64_t const row = softStart + (y / poolingKernelSize) * softWidth + (x / poolingKernelSize);
    int64_t const col = patchStart + tid;
    float const weight = 1.0F / static_cast<float>(poolingKernelSize * poolingKernelSize);
    poolingWeights[row * totalPatches + col] = __float2half(weight);
}

void initPoolingWeightsGemma4ViT(rt::Tensor& poolingWeights, int64_t patchStart, int64_t softStart, int64_t patchHeight,
    int64_t patchWidth, int64_t poolingKernelSize, cudaStream_t stream)
{
    check::check(poolingWeights.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall be GPU for Gemma4 pooling weights tensor.");
    check::check(
        poolingWeights.getDataType() == DataType::kHALF, "Data type shall be half for Gemma4 pooling weights tensor.");
    check::check(poolingWeights.getShape().getNumDims() == 2,
        "Gemma4 pooling weights shape shall be [totalSoftTokens, totalPatches].");
    check::check(poolingKernelSize > 0, "Gemma4 pooling kernel size must be positive.");
    check::check(patchHeight > 0 && patchWidth > 0, "Gemma4 patch grid size must be positive.");
    check::check(patchHeight % poolingKernelSize == 0 && patchWidth % poolingKernelSize == 0,
        "Gemma4 patch grid size must be divisible by pooling kernel size.");

    int64_t const totalSoftTokens = poolingWeights.getShape()[0];
    int64_t const totalPatches = poolingWeights.getShape()[1];
    int64_t const curPatches = patchHeight * patchWidth;
    int64_t const curSoftTokens = (patchHeight / poolingKernelSize) * (patchWidth / poolingKernelSize);
    check::check(patchStart >= 0 && patchStart + curPatches <= totalPatches,
        "Gemma4 pooling patch range exceeds pooling weights shape.");
    check::check(softStart >= 0 && softStart + curSoftTokens <= totalSoftTokens,
        "Gemma4 pooling soft-token range exceeds pooling weights shape.");

    uint32_t const blockSize = 256;
    uint32_t const gridSize = static_cast<uint32_t>((curPatches + blockSize - 1) / blockSize);
    initPoolingWeightsGemma4Kernel<<<gridSize, blockSize, 0, stream>>>(poolingWeights.dataPointer<half>(), totalPatches,
        patchStart, softStart, patchHeight, patchWidth, poolingKernelSize);
}

__global__ void initFastPosEmbedQwenViTKernel(int64_t* fastPosEmbedIdx, half* fastPosEmbedWeight,
    int64_t const llmGridH, int64_t const llmGridW, int64_t const mergeSize, int64_t const numGridPerSide,
    float const lineSpaceH, float const lineSpaceW, int64_t const startIdx, int64_t const totalSeqLength)
{
    // Each CTA get assigned 256 threads. Each thread processes one position in grid
    //     [llmGridH, llmGridW, mergeSize, mergeSize]
    // Each position needs to generate 4 indices and 4 weights
    // fastPosEmbedIdx: [4, totalSeqLength]
    // fastPosEmbedWeight: [4, totalSeqLength]
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto const totalElements = llmGridH * llmGridW * mergeSize * mergeSize;
    if (tid >= totalElements)
        return;

    auto const llmGridHIdx = tid / (llmGridW * mergeSize * mergeSize);
    auto const llmGridWIdx = (tid % (llmGridW * mergeSize * mergeSize)) / (mergeSize * mergeSize);
    auto const mergeHIdx = (tid % (mergeSize * mergeSize)) / mergeSize;
    auto const mergeWIdx = tid % mergeSize;

    float const hIdx = lineSpaceH * (llmGridHIdx * mergeSize + mergeHIdx);
    float const wIdx = lineSpaceW * (llmGridWIdx * mergeSize + mergeWIdx);

    int64_t const hIdxFloor = static_cast<int64_t>(hIdx);
    int64_t const wIdxFloor = static_cast<int64_t>(wIdx);
    int64_t const hIdxCeil = std::min(hIdxFloor + 1, (numGridPerSide - 1));
    int64_t const wIdxCeil = std::min(wIdxFloor + 1, (numGridPerSide - 1));

    float const dh = hIdx - hIdxFloor;
    float const dw = wIdx - wIdxFloor;

    int64_t const baseH = hIdxFloor * numGridPerSide;
    int64_t const baseHCeil = hIdxCeil * numGridPerSide;

    int64_t const targetIdx = startIdx + tid;

    fastPosEmbedIdx[0 * totalSeqLength + targetIdx] = baseH + wIdxFloor;
    fastPosEmbedIdx[1 * totalSeqLength + targetIdx] = baseH + wIdxCeil;
    fastPosEmbedIdx[2 * totalSeqLength + targetIdx] = baseHCeil + wIdxFloor;
    fastPosEmbedIdx[3 * totalSeqLength + targetIdx] = baseHCeil + wIdxCeil;
    fastPosEmbedWeight[0 * totalSeqLength + targetIdx] = __float2half((1 - dh) * (1 - dw));
    fastPosEmbedWeight[1 * totalSeqLength + targetIdx] = __float2half((1 - dh) * dw);
    fastPosEmbedWeight[2 * totalSeqLength + targetIdx] = __float2half(dh * (1 - dw));
    fastPosEmbedWeight[3 * totalSeqLength + targetIdx] = __float2half(dh * dw);
}

void initFastPosEmbedQwenViT(rt::Tensor& fastPosEmbedIdx, rt::Tensor& fastPosEmbedWeight,
    std::vector<int64_t> const& gridTHW, int64_t const mergeSize, int64_t const numGridPerSide, int64_t const startIdx,
    cudaStream_t stream)
{
    check::check(fastPosEmbedIdx.getDeviceType() == rt::DeviceType::kGPU
            && fastPosEmbedWeight.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(
        fastPosEmbedIdx.getDataType() == DataType::kINT64 && fastPosEmbedWeight.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(fastPosEmbedIdx.getShape().getNumDims() == 2 && fastPosEmbedIdx.getShape()[0] == 4,
        "Fast position embeddings index shapes shall be [4, totalSeqLength].");
    check::check(fastPosEmbedWeight.getShape().getNumDims() == 2 && fastPosEmbedWeight.getShape()[0] == 4,
        "Fast position embeddings weight shapes shall be [4, totalSeqLength].");

    int64_t const totalSeqLength = fastPosEmbedIdx.getShape()[1];
    check::check(totalSeqLength == fastPosEmbedWeight.getShape()[1], "Total sequence length mismatch.");

    check::check(gridTHW.size() == 3, "gridTHW must have exactly 3 elements [T, H, W]");
    int64_t const T = gridTHW[0];
    int64_t const H = gridTHW[1];
    int64_t const W = gridTHW[2];
    int64_t const llmGridH = H / mergeSize;
    int64_t const llmGridW = W / mergeSize;
    float const lineSpaceH = static_cast<float>(numGridPerSide - 1) / (H - 1);
    float const lineSpaceW = static_cast<float>(numGridPerSide - 1) / (W - 1);

    uint32_t const blockSize = 256;
    uint32_t const gridSize = (H * W + blockSize - 1) / blockSize;

    // The fast position embedding is spatial (H*W); for a video grid it repeats per temporal frame (HF
    // `pos_embed.repeat(t, 1)`). Qwen3-VL splits frames into T=1 sub-span grids (this loop runs once);
    // Qwen3-Omni passes a single (T, H, W) grid, so write the same spatial pattern at each frame's patch offset.
    for (int64_t t = 0; t < T; ++t)
    {
        initFastPosEmbedQwenViTKernel<<<gridSize, blockSize, 0, stream>>>(fastPosEmbedIdx.dataPointer<int64_t>(),
            fastPosEmbedWeight.dataPointer<half>(), llmGridH, llmGridW, mergeSize, numGridPerSide, lineSpaceH,
            lineSpaceW, startIdx + t * H * W, totalSeqLength);
    }
}

// ---------------------------------------------------------------------------
// GPU bicubic (Catmull-Rom) image resize, anti-aliased for downscaling. For
// downscaling the filter is widened by the scale factor (fscale) so it
// low-passes instead of applying a fixed 4-tap cubic.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float catmullRomWeight(float x)
{
    x = fabsf(x);
    if (x < 1.0f)
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    if (x < 2.0f)
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    return 0.0f;
}

__global__ void resizeCatmullRomHorizKernel(
    unsigned char const* in, float* out, int const Hin, int const Win, int const Wout, int const C)
{
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = static_cast<int64_t>(Hin) * Wout * C;
    if (tid >= total)
        return;
    int const c = static_cast<int>(tid % C);
    int const xo = static_cast<int>((tid / C) % Wout);
    int const h = static_cast<int>(tid / (static_cast<int64_t>(Wout) * C));

    float const scale = static_cast<float>(Win) / static_cast<float>(Wout);
    float const fscale = scale > 1.0f ? scale : 1.0f;
    float const center = (xo + 0.5f) * scale;
    int const xs = static_cast<int>(ceilf(center - 2.0f * fscale - 0.5f));
    int const xe = static_cast<int>(floorf(center + 2.0f * fscale - 0.5f));

    float acc = 0.0f;
    float wsum = 0.0f;
    int64_t const rowOff = static_cast<int64_t>(h) * Win * C;
    for (int xin = xs; xin <= xe; ++xin)
    {
        float const w = catmullRomWeight((xin + 0.5f - center) / fscale);
        int const xc = xin < 0 ? 0 : (xin >= Win ? Win - 1 : xin);
        acc += w * static_cast<float>(in[rowOff + static_cast<int64_t>(xc) * C + c]);
        wsum += w;
    }
    out[tid] = wsum > 0.0f ? acc / wsum : 0.0f;
}

__global__ void resizeCatmullRomVertKernel(
    float const* in, unsigned char* out, int const Hin, int const Hout, int const Wout, int const C)
{
    int64_t const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = static_cast<int64_t>(Hout) * Wout * C;
    if (tid >= total)
        return;
    int const c = static_cast<int>(tid % C);
    int const x = static_cast<int>((tid / C) % Wout);
    int const yo = static_cast<int>(tid / (static_cast<int64_t>(Wout) * C));

    float const scale = static_cast<float>(Hin) / static_cast<float>(Hout);
    float const fscale = scale > 1.0f ? scale : 1.0f;
    float const center = (yo + 0.5f) * scale;
    int const ys = static_cast<int>(ceilf(center - 2.0f * fscale - 0.5f));
    int const ye = static_cast<int>(floorf(center + 2.0f * fscale - 0.5f));

    float acc = 0.0f;
    float wsum = 0.0f;
    int64_t const colOff = static_cast<int64_t>(x) * C + c;
    int64_t const stride = static_cast<int64_t>(Wout) * C;
    for (int yin = ys; yin <= ye; ++yin)
    {
        float const w = catmullRomWeight((yin + 0.5f - center) / fscale);
        int const yc = yin < 0 ? 0 : (yin >= Hin ? Hin - 1 : yin);
        acc += w * in[static_cast<int64_t>(yc) * stride + colOff];
        wsum += w;
    }
    float v = wsum > 0.0f ? acc / wsum : 0.0f;
    v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
    out[tid] = static_cast<unsigned char>(v + 0.5f);
}

void resizeImage(rt::Tensor const& rawImage, rt::Tensor& tmp, rt::Tensor& resizedImage, int64_t const outHeight,
    int64_t const outWidth, InterpolationMode const mode, cudaStream_t stream)
{
    ELLM_CHECK(mode == InterpolationMode::kBICUBIC,
        "GPU resizeImage supports only InterpolationMode::kBICUBIC (Catmull-Rom) for now.");
    ELLM_CHECK(rawImage.getDeviceType() == rt::DeviceType::kGPU && tmp.getDeviceType() == rt::DeviceType::kGPU
            && resizedImage.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for resize tensors.");
    ELLM_CHECK(rawImage.getDataType() == DataType::kUINT8 && tmp.getDataType() == DataType::kFLOAT
            && resizedImage.getDataType() == DataType::kUINT8,
        "Data type check failed for resize tensors (raw u8, tmp f32, out u8).");
    ELLM_CHECK(rawImage.getShape().getNumDims() == 3 && resizedImage.getShape().getNumDims() == 3,
        "rawImage and resizedImage shall be [H, W, C].");

    int const Hin = static_cast<int>(rawImage.getShape()[0]);
    int const Win = static_cast<int>(rawImage.getShape()[1]);
    int const C = static_cast<int>(rawImage.getShape()[2]);
    int const Hout = static_cast<int>(outHeight);
    int const Wout = static_cast<int>(outWidth);

    ELLM_CHECK(tmp.getShape().volume() >= static_cast<int64_t>(Hin) * Wout * C,
        "tmp scratch smaller than Hin * outWidth * C for resize tensors.");
    ELLM_CHECK(resizedImage.getShape().volume() >= static_cast<int64_t>(Hout) * Wout * C,
        "resizedImage smaller than outHeight * outWidth * C for resize tensors.");

    uint32_t const blockSize = 256;
    int64_t const totalH = static_cast<int64_t>(Hin) * Wout * C;
    resizeCatmullRomHorizKernel<<<static_cast<uint32_t>((totalH + blockSize - 1) / blockSize), blockSize, 0, stream>>>(
        rawImage.dataPointer<unsigned char>(), tmp.dataPointer<float>(), Hin, Win, Wout, C);
    int64_t const totalV = static_cast<int64_t>(Hout) * Wout * C;
    resizeCatmullRomVertKernel<<<static_cast<uint32_t>((totalV + blockSize - 1) / blockSize), blockSize, 0, stream>>>(
        tmp.dataPointer<float>(), resizedImage.dataPointer<unsigned char>(), Hin, Hout, Wout, C);
}

void copyImageToDeviceAndResize(unsigned char const* rawHostImage, int64_t const numFrames, int64_t const rawHeight,
    int64_t const rawWidth, int64_t const channels, rt::Tensor& rawScratch, rt::Tensor& tmp, rt::Tensor& dstImage,
    int64_t const outHeight, int64_t const outWidth, cudaStream_t stream)
{
    ELLM_CHECK(rawHostImage != nullptr, "copyImageToDeviceAndResize: raw host image pointer is null.");
    ELLM_CHECK(rawHeight > 0 && rawWidth > 0 && outHeight > 0 && outWidth > 0,
        "copyImageToDeviceAndResize: raw and output dimensions shall be positive.");
    ELLM_CHECK(dstImage.getDeviceType() == rt::DeviceType::kGPU,
        "copyImageToDeviceAndResize: destination image shall be on the GPU.");
    ELLM_CHECK(
        dstImage.getDataType() == DataType::kUINT8, "copyImageToDeviceAndResize: destination image shall be UINT8.");
    ELLM_CHECK(dstImage.reshape({numFrames, outHeight, outWidth, channels}),
        "copyImageToDeviceAndResize destination too small for " + std::to_string(numFrames) + " frames of "
            + std::to_string(outHeight) + "x" + std::to_string(outWidth) + "x" + std::to_string(channels) + ".");

    int64_t const rawFrameBytes = rawHeight * rawWidth * channels;
    int64_t const outFrameBytes = outHeight * outWidth * channels;
    bool const identity = (rawHeight == outHeight && rawWidth == outWidth);

    if (!identity)
    {
        // Each raw side carries its own cap (the engine budget bounds only the resized size); this also
        // bounds the horizontal-pass scratch [rawHeight, outWidth, channels].
        ELLM_CHECK(rawHeight <= kGpuResizeMaxRawDim && rawWidth <= kGpuResizeMaxRawDim,
            "Raw image " + std::to_string(rawHeight) + "x" + std::to_string(rawWidth)
                + " exceeds the GPU-resize budget of " + std::to_string(kGpuResizeMaxRawDim) + "x"
                + std::to_string(kGpuResizeMaxRawDim) + " pixels; downscale the input image.");
        ELLM_CHECK(rawScratch.reshape({rawHeight, rawWidth, channels}),
            "GPU-resize raw scratch too small for a " + std::to_string(rawHeight) + "x" + std::to_string(rawWidth) + "x"
                + std::to_string(channels) + " raw image.");
        ELLM_CHECK(tmp.reshape({rawHeight, outWidth, channels}),
            "GPU-resize horizontal-pass scratch for raw " + std::to_string(rawHeight) + "x" + std::to_string(rawWidth)
                + " resized to " + std::to_string(outHeight) + "x" + std::to_string(outWidth) + " exceeds its budget.");
    }

    auto* const dstBase = static_cast<unsigned char*>(dstImage.rawPointer());
    for (int64_t t = 0; t < numFrames; ++t)
    {
        unsigned char const* srcFrame = rawHostImage + t * rawFrameBytes;
        unsigned char* dstFrame = dstBase + t * outFrameBytes;
        if (identity)
        {
            CUDA_CHECK(cudaMemcpyAsync(dstFrame, srcFrame, rawFrameBytes, cudaMemcpyHostToDevice, stream));
            continue;
        }
        // rawScratch and tmp are reused per frame on the same stream, so the upload-then-resize chain
        // serializes safely; the 3-D view aliases this frame's slot of the 4-D dst.
        CUDA_CHECK(cudaMemcpyAsync(rawScratch.rawPointer(), srcFrame, rawFrameBytes, cudaMemcpyHostToDevice, stream));
        rt::Tensor dstView(dstFrame, {outHeight, outWidth, channels}, rt::DeviceType::kGPU, DataType::kUINT8,
            "kernel::copyImageToDeviceAndResize.dstView");
        resizeImage(rawScratch, tmp, dstView, outHeight, outWidth, InterpolationMode::kBICUBIC, stream);
    }
}

void allocateResizeScratch(int64_t const channels, int64_t const tmpElems, rt::Tensor& rawScratch, rt::Tensor& tmp)
{
    int64_t const rawElems = kGpuResizeMaxRawDim * kGpuResizeMaxRawDim * channels;
    rawScratch = rt::Tensor({rawElems}, rt::DeviceType::kGPU, DataType::kUINT8, "kernel::resizeScratch.rawImage");
    tmp = rt::Tensor({tmpElems}, rt::DeviceType::kGPU, DataType::kFLOAT, "kernel::resizeScratch.tmp");
}

__global__ void transposeToPatchNemotronKernel(half const* blockPixels, half* inputPatches, int64_t const T,
    int64_t const C, int64_t const H, int64_t const W, int64_t const P, int64_t const totalElements)
{
    // Row layout: [group * numPatches + (pi * (W/P) + pj), ((t*C + c)*P + py)*P + px]
    // Source layout: [group*T + t, c, pi*P + py, pj*P + px]
    auto const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= totalElements)
    {
        return;
    }

    int64_t const rowWidth = T * C * P * P;
    int64_t const gridW = W / P;
    int64_t const numPatches = (H / P) * gridW;

    int64_t const row = tid / rowWidth;
    int64_t const col = tid % rowWidth;

    int64_t const group = row / numPatches;
    int64_t const patch = row % numPatches;
    int64_t const pi = patch / gridW;
    int64_t const pj = patch % gridW;

    int64_t const t = col / (C * P * P);
    int64_t const c = (col % (C * P * P)) / (P * P);
    int64_t const py = (col % (P * P)) / P;
    int64_t const px = col % P;

    int64_t const srcIdx = (((group * T + t) * C + c) * H + pi * P + py) * W + pj * P + px;
    inputPatches[tid] = blockPixels[srcIdx];
}

void transposeToPatchNemotronViT(rt::Tensor const& blockPixels, rt::Tensor& inputPatches,
    int64_t const temporalPatchSize, int64_t const patchSize, cudaStream_t stream)
{
    check::check(
        blockPixels.getDeviceType() == rt::DeviceType::kGPU && inputPatches.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(blockPixels.getDataType() == DataType::kHALF && inputPatches.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(blockPixels.getShape().getNumDims() == 4, "blockPixels shape shall be [frames, channels, H, W].");
    check::check(temporalPatchSize > 0, "temporalPatchSize must be positive.");
    check::check(patchSize > 0, "patchSize must be positive.");

    int64_t const frames = blockPixels.getShape()[0];
    int64_t const channels = blockPixels.getShape()[1];
    int64_t const height = blockPixels.getShape()[2];
    int64_t const width = blockPixels.getShape()[3];
    check::check(frames % temporalPatchSize == 0, "Frame count must be a multiple of temporalPatchSize.");
    check::check(height % patchSize == 0 && width % patchSize == 0, "Image dims must be multiples of patchSize.");

    int64_t const totalElements = frames * channels * height * width;
    check::check(
        inputPatches.getShape().volume() >= totalElements, "inputPatches tensor too small for the patch output.");
    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    transposeToPatchNemotronKernel<<<gridSize, blockSize, 0, stream>>>(blockPixels.dataPointer<half>(),
        inputPatches.dataPointer<half>(), temporalPatchSize, channels, height, width, patchSize, totalElements);
}

__global__ void addPosEmbedNemotronKernel(
    half* patchEmbeds, half const* posEmbed, int64_t const perBlockElements, int64_t const totalElements)
{
    auto const tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= totalElements)
    {
        return;
    }
    patchEmbeds[tid] = __hadd(patchEmbeds[tid], posEmbed[tid % perBlockElements]);
}

void addPosEmbedNemotronViT(rt::Tensor& patchEmbeds, rt::Tensor const& posEmbed, cudaStream_t stream)
{
    check::check(
        patchEmbeds.getDeviceType() == rt::DeviceType::kGPU && posEmbed.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(patchEmbeds.getDataType() == DataType::kHALF && posEmbed.getDataType() == DataType::kHALF,
        "Data type check failed for the input tensors.");
    check::check(patchEmbeds.getShape().getNumDims() == 3 && posEmbed.getShape().getNumDims() == 2,
        "patchEmbeds shape shall be [numBlocks, numPatches, hidden] and posEmbed [numPatches, hidden].");
    check::check(
        patchEmbeds.getShape()[1] == posEmbed.getShape()[0] && patchEmbeds.getShape()[2] == posEmbed.getShape()[1],
        "posEmbed dims must match patchEmbeds per-block dims.");

    int64_t const perBlockElements = posEmbed.getShape().volume();
    check::check(perBlockElements > 0, "posEmbed must be non-empty.");
    int64_t const totalElements = patchEmbeds.getShape().volume();
    uint32_t const blockSize = 256;
    uint32_t const gridSize = (totalElements + blockSize - 1) / blockSize;

    addPosEmbedNemotronKernel<<<gridSize, blockSize, 0, stream>>>(
        patchEmbeds.dataPointer<half>(), posEmbed.dataPointer<half>(), perBlockElements, totalElements);
}

__global__ void evsScoresNemotronKernel(
    half const* embeds, float* scores, int64_t const tokensPerGroup, int64_t const hidden)
{
    // One CTA per token (g, s). Cosine dissimilarity vs the same spatial slot in the previous
    // temporal group; group 0 gets the keep-always sentinel 255 (matches HF EVS).
    int64_t const token = blockIdx.x;
    int64_t const group = token / tokensPerGroup;

    if (group == 0)
    {
        if (threadIdx.x == 0)
        {
            scores[token] = 255.0F;
        }
        return;
    }

    half const* cur = embeds + token * hidden;
    half const* prev = embeds + (token - tokensPerGroup) * hidden;

    float dot = 0.0F;
    float normCur = 0.0F;
    float normPrev = 0.0F;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
    {
        float const a = __half2float(cur[i]);
        float const b = __half2float(prev[i]);
        dot += a * b;
        normCur += a * a;
        normPrev += b * b;
    }

    __shared__ float sDot[32];
    __shared__ float sCur[32];
    __shared__ float sPrev[32];
    int const lane = threadIdx.x % 32;
    int const warp = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        normCur += __shfl_down_sync(0xFFFFFFFF, normCur, offset);
        normPrev += __shfl_down_sync(0xFFFFFFFF, normPrev, offset);
    }
    if (lane == 0)
    {
        sDot[warp] = dot;
        sCur[warp] = normCur;
        sPrev[warp] = normPrev;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float d = 0.0F;
        float nc = 0.0F;
        float np = 0.0F;
        int const numWarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < numWarps; ++w)
        {
            d += sDot[w];
            nc += sCur[w];
            np += sPrev[w];
        }
        // torch cosine_similarity clamps each norm to eps before dividing.
        float const eps = 1e-8F;
        float const cos = d / (fmaxf(sqrtf(nc), eps) * fmaxf(sqrtf(np), eps));
        scores[token] = 1.0F - cos;
    }
}

void evsScoresNemotronViT(
    rt::Tensor const& embeds, rt::Tensor& scores, int64_t const tokensPerGroup, cudaStream_t stream)
{
    check::check(embeds.getDeviceType() == rt::DeviceType::kGPU && scores.getDeviceType() == rt::DeviceType::kGPU,
        "Device type shall all be GPU for these tensors.");
    check::check(embeds.getDataType() == DataType::kHALF && scores.getDataType() == DataType::kFLOAT,
        "Data type check failed for the input tensors.");
    check::check(embeds.getShape().getNumDims() == 2, "embeds shape shall be [numTokens, hidden].");
    check::check(tokensPerGroup > 0, "tokensPerGroup must be positive.");
    int64_t const numTokens = embeds.getShape()[0];
    check::check(numTokens % tokensPerGroup == 0, "numTokens must be a multiple of tokensPerGroup.");
    check::check(scores.getShape().volume() >= numTokens, "scores tensor too small.");

    int64_t const hidden = embeds.getShape()[1];
    evsScoresNemotronKernel<<<static_cast<uint32_t>(numTokens), 256, 0, stream>>>(
        embeds.dataPointer<half>(), scores.dataPointer<float>(), tokensPerGroup, hidden);
}

} // namespace kernel
} // namespace trt_edgellm
