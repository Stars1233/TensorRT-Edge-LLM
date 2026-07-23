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

#include "utilKernels.h"

#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace kernel
{

__global__ void calCuQCuKVSeqLensAndKVEndIdxsKernel(int32_t const* inputSeqLen, int32_t const* kvCacheStartIndices,
    int32_t* cuQSeqlen, int32_t* cuKVSeqLens, int32_t* kvCacheEndIndices, int32_t* paddedCuKVSeqLens,
    int32_t runtimeSeqLen, int32_t batchSize)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cuQSeqlen[0] = 0;
        cuKVSeqLens[0] = 0;
        if (paddedCuKVSeqLens != nullptr)
        {
            paddedCuKVSeqLens[0] = 0;
        }

        int32_t runningCuSeqLen = 0;
        int32_t runningCuKvCacheLen = 0;
        int32_t runningPaddedCuKvLen = 0;
        for (int32_t i = 0; i < batchSize; ++i)
        {
            runningCuSeqLen += inputSeqLen[i];
            cuQSeqlen[i + 1] = runningCuSeqLen;

            int32_t kvCacheStartIdx = 0;
            if (kvCacheStartIndices != nullptr)
            {
                kvCacheStartIdx = kvCacheStartIndices[i];
            }

            runningCuKvCacheLen += (kvCacheStartIdx + inputSeqLen[i]);
            cuKVSeqLens[i + 1] = runningCuKvCacheLen;
            // To keep semantic consistency with the packed QKV layout for RoPE, use runtimeSeqLen here.
            int32_t const kvEndIdx = kvCacheStartIdx + runtimeSeqLen;
            kvCacheEndIndices[i] = kvEndIdx;

            if (paddedCuKVSeqLens != nullptr)
            {
                runningPaddedCuKvLen += kvEndIdx;
                paddedCuKVSeqLens[i + 1] = runningPaddedCuKvLen;
            }
        }
    }
}

void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs,
    rt::OptionalOutputTensor paddedCuKVSeqLens, int32_t const runtimeSeqLen, cudaStream_t stream)
{
    int32_t const runtimeBatchSize = static_cast<int32_t>(inputSeqLen.getShape()[0]);

    // Perform necessary shape checks.
    check::check(cuQSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuQSeqLens shall have shape [B+1].");
    check::check(cuKVSeqLens.getShape()[0] == (runtimeBatchSize + 1), "cuKVSeqLens shall have shape [B+1].");
    check::check(kvCacheEndIdxs.getShape()[0] == runtimeBatchSize, "kvCacheEndIdxs shall have shape [B].");

    if (!kvCacheStartIndices.isEmpty())
    {
        check::check(
            kvCacheStartIndices.getShape()[0] == runtimeBatchSize, "KVCacheStartIndices tensor shall have shape [B].");
    }
    else
    {
        // We rely on this nullptr behavior to indicate whether kvCacheStartIndices is available in the kernel.
        check::check(kvCacheStartIndices.rawPointer() == nullptr,
            "KVCacheStartIndices tensor shall be nullptr when it is empty.");
    }

    int32_t* paddedPtr = nullptr;
    if (paddedCuKVSeqLens.has_value())
    {
        rt::Tensor& paddedTensor = paddedCuKVSeqLens.value().get();
        check::check(paddedTensor.getShape()[0] == (runtimeBatchSize + 1), "paddedCuKVSeqLens shall have shape [B+1].");
        paddedPtr = paddedTensor.dataPointer<int32_t>();
    }

    calCuQCuKVSeqLensAndKVEndIdxsKernel<<<1, 1, 0, stream>>>(inputSeqLen.dataPointer<int32_t>(),
        kvCacheStartIndices.dataPointer<int32_t>(), cuQSeqLens.dataPointer<int32_t>(),
        cuKVSeqLens.dataPointer<int32_t>(), kvCacheEndIdxs.dataPointer<int32_t>(), paddedPtr, runtimeSeqLen,
        runtimeBatchSize);
}

// ===== FMHA_v2 CUSTOM_MASK packed-mask builder for vision-block prefill =====
namespace
{

//! Allowed(q, k) predicate for Gemma4 vision-block attention: sliding-causal
//! OR same non-negative vision block.
__device__ __forceinline__ bool isVisionPositionAllowed(
    int32_t const* blockIds, int32_t contextLen, int32_t slidingWindowSize, int32_t q, int32_t k)
{
    if (q >= contextLen || k >= contextLen)
    {
        return false;
    }
    bool const slidingCausal = (k <= q) && (slidingWindowSize <= 0 || k > q - slidingWindowSize);
    int32_t const qBlock = blockIds[q];
    bool const sameVisionBlock = qBlock >= 0 && qBlock == blockIds[k];
    return slidingCausal || sameVisionBlock;
}

//! One thread produces one uint32 of the packed mask.  Grid:
//! (mmasN, mmasMPerSeq, batch), block: 128 threads (one warp group covering a
//! 64x64 tile).  See utilKernels.h for the bit layout contract.
__global__ void buildVisionPackedMaskKernel(int32_t const* __restrict__ visionBlockIds,
    int32_t const* __restrict__ contextLengths, uint32_t* __restrict__ packedMask, int32_t seqLen,
    int32_t slidingWindowSize)
{
    int32_t const mmaN = static_cast<int32_t>(blockIdx.x);
    int32_t const mmaM = static_cast<int32_t>(blockIdx.y);
    int32_t const batch = static_cast<int32_t>(blockIdx.z);
    int32_t const tidx = static_cast<int32_t>(threadIdx.x);
    int32_t const mmasN = static_cast<int32_t>(gridDim.x);
    int32_t const mmasMPerSeq = static_cast<int32_t>(gridDim.y);

    int32_t const contextLen = min(contextLengths[batch], seqLen);
    int32_t const* blockIds = visionBlockIds + static_cast<int64_t>(batch) * seqLen;

    // The base (row, col) covered by this thread within the 64x64 tile.
    int32_t const warp = tidx / 32;
    int32_t const lane = tidx % 32;
    int32_t const row = mmaM * kFMHA_PACKED_MASK_MMA_M + (warp % 4) * 16 + lane / 4;
    int32_t const colBase = mmaN * kFMHA_PACKED_MASK_MMA_N + (lane % 4) * 2;

    uint32_t mask = 0U;
#pragma unroll
    for (int32_t ni = 0; ni < 8; ++ni)
    {
        int32_t const col = colBase + ni * 8;
        mask |= (isVisionPositionAllowed(blockIds, contextLen, slidingWindowSize, row, col) ? 1U : 0U) << (4 * ni + 0);
        mask |= (isVisionPositionAllowed(blockIds, contextLen, slidingWindowSize, row, col + 1) ? 1U : 0U)
            << (4 * ni + 1);
        mask |= (isVisionPositionAllowed(blockIds, contextLen, slidingWindowSize, row + 8, col) ? 1U : 0U)
            << (4 * ni + 2);
        mask |= (isVisionPositionAllowed(blockIds, contextLen, slidingWindowSize, row + 8, col + 1) ? 1U : 0U)
            << (4 * ni + 3);
    }

    // Word layout: [globalMmaM][mmaN][threads].
    int64_t const globalMmaM = static_cast<int64_t>(batch) * mmasMPerSeq + mmaM;
    int64_t const wordIdx = (globalMmaM * mmasN + mmaN) * kFMHA_PACKED_MASK_THREADS_PER_WARP_GROUP + tidx;
    packedMask[wordIdx] = mask;
}

//! Fills cuMaskRows[i] = i * paddedRowsPerSeq (uniform padded rows per batch).
__global__ void fillCuMaskRowsKernel(int32_t* cuMaskRows, int32_t batchSize, int32_t paddedRowsPerSeq)
{
    int32_t const i
        = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) + static_cast<int32_t>(threadIdx.x);
    if (i <= batchSize)
    {
        cuMaskRows[i] = i * paddedRowsPerSeq;
    }
}

//! One thread per (batch, position): expand vision-block IDs into per-position
//! [blockBegin, blockEnd] intervals for the FFPA vision-block overlay prefill.
__global__ void buildVisionBlockRangesKernel(int32_t const* visionBlockIds, int32_t const* contextLengths,
    int32_t* blockBegin, int32_t* blockEnd, int32_t seqLen)
{
    int32_t const pos
        = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x) + static_cast<int32_t>(threadIdx.x);
    int32_t const batch = static_cast<int32_t>(blockIdx.y);
    if (pos >= seqLen)
    {
        return;
    }
    int64_t const base = static_cast<int64_t>(batch) * seqLen;
    int32_t const contextLen = min(contextLengths[batch], seqLen);

    int32_t begin = -1;
    int32_t end = -1;
    if (pos < contextLen)
    {
        int32_t const blockId = visionBlockIds[base + pos];
        if (blockId >= 0)
        {
            // Contiguous-run expansion: blocks are short (a few hundred
            // tokens), so the linear scans are cheap.
            begin = pos;
            end = pos;
            while (begin > 0 && visionBlockIds[base + begin - 1] == blockId)
            {
                --begin;
            }
            while (end + 1 < contextLen && visionBlockIds[base + end + 1] == blockId)
            {
                ++end;
            }
        }
    }
    blockBegin[base + pos] = begin;
    blockEnd[base + pos] = end;
}

} // namespace

void launchBuildVisionPackedMask(int32_t const* visionBlockIds, int32_t const* contextLengths, uint32_t* packedMask,
    int32_t* cuMaskRows, int32_t batchSize, int32_t seqLen, int32_t slidingWindowSize, cudaStream_t stream)
{
    check::check(visionBlockIds != nullptr && contextLengths != nullptr && packedMask != nullptr,
        "Vision packed mask builder received a null pointer");
    check::check(batchSize > 0 && seqLen > 0, "Vision packed mask builder received invalid dimensions");

    int32_t const paddedRowsPerSeq = static_cast<int32_t>(getPackedMaskRowsPerSeq(seqLen));
    int32_t const mmasMPerSeq = paddedRowsPerSeq / kFMHA_PACKED_MASK_MMA_M;
    int32_t const mmasN = static_cast<int32_t>(getPackedMaskRowStrideInBytes(seqLen) * 8) / kFMHA_PACKED_MASK_MMA_N;

    if (cuMaskRows != nullptr)
    {
        uint32_t const fillBlocks = static_cast<uint32_t>((batchSize + 1 + 127) / 128);
        fillCuMaskRowsKernel<<<fillBlocks, 128, 0, stream>>>(cuMaskRows, batchSize, paddedRowsPerSeq);
    }
    dim3 const grid(static_cast<uint32_t>(mmasN), static_cast<uint32_t>(mmasMPerSeq), static_cast<uint32_t>(batchSize));
    buildVisionPackedMaskKernel<<<grid, kFMHA_PACKED_MASK_THREADS_PER_WARP_GROUP, 0, stream>>>(
        visionBlockIds, contextLengths, packedMask, seqLen, slidingWindowSize);
    CUDA_CHECK(cudaGetLastError());
}

void launchBuildVisionBlockRanges(int32_t const* visionBlockIds, int32_t const* contextLengths, int32_t* blockBegin,
    int32_t* blockEnd, int32_t batchSize, int32_t seqLen, cudaStream_t stream)
{
    check::check(visionBlockIds != nullptr && contextLengths != nullptr && blockBegin != nullptr && blockEnd != nullptr,
        "Vision block range expansion received a null pointer");
    check::check(batchSize > 0 && seqLen > 0, "Vision block range expansion received invalid dimensions");

    constexpr int32_t kRANGE_THREADS = 256;
    dim3 const grid(
        static_cast<uint32_t>((seqLen + kRANGE_THREADS - 1) / kRANGE_THREADS), static_cast<uint32_t>(batchSize));
    buildVisionBlockRangesKernel<<<grid, kRANGE_THREADS, 0, stream>>>(
        visionBlockIds, contextLengths, blockBegin, blockEnd, seqLen);
    CUDA_CHECK(cudaGetLastError());
}

// ===== kernel: produce separate K [B, S, H, D] and V [B, S, H, D] =====
// Unified deinterleave kernel: copies tokens [0, dstS) from source [B, 2, H, srcS, D] to
// separate K [B, dstS, H, D] and V [B, dstS, H, D].  When dstS == srcS this is a full copy;
// when dstS < srcS it is a compact copy of only the first dstS tokens.
template <typename T>
__global__ void cvtKVLayoutBHSDToSplitKVKernel(T const* __restrict__ src, // [B, 2, H, srcS, D]
    half* __restrict__ kDst,                                              // [B, dstS, H, D]
    half* __restrict__ vDst,                                              // [B, dstS, H, D]
    float const* __restrict__ kScaleQuantOrig, float const* __restrict__ vScaleQuantOrig, int32_t B, int32_t srcS,
    int32_t dstS, int32_t H, int32_t D)
{
    uint32_t const token = blockIdx.y * blockDim.y + threadIdx.y; // 0 .. dstS-1
    uint32_t const d = blockIdx.x * blockDim.x + threadIdx.x;     // 0 .. D-1

    uint32_t const numHpBlocks = (2 * H + blockDim.z - 1) / blockDim.z;
    uint32_t const batch = blockIdx.z / numHpBlocks;
    uint32_t const hpTile = blockIdx.z % numHpBlocks;
    uint32_t const headPair = hpTile * blockDim.z + threadIdx.z; // 0 .. 2*H-1

    if (batch >= B || headPair >= 2 * H || d >= D || token >= dstS)
        return;

    uint32_t const kv = headPair / H; // 0 = K, 1 = V
    uint32_t const h = headPair % H;

    // src layout: [B, 2, H, srcS, D]
    size_t const srcIdx = (((((size_t) batch * 2 + kv) * H + h) * srcS + token) * D + d);
    // dst layout: [B, dstS, H, D]
    size_t const dstIdx = ((((size_t) batch * dstS + token) * H + h) * D + d);

    half* dst = (kv == 0) ? kDst : vDst;

#if SUPPORTS_FP8
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        float const scale = (kv == 0) ? kScaleQuantOrig[0] : vScaleQuantOrig[0];
        dst[dstIdx] = __float2half(static_cast<float>(src[srcIdx]) * scale);
    }
    else
#endif
    {
        dst[dstIdx] = src[srcIdx];
    }
}

void cvtKVLayoutBHSDToSplitKV(rt::Tensor const& src, rt::Tensor& kDst, rt::Tensor& vDst,
    rt::Tensor const& kvScaleQuantOrig, int32_t seqLen, cudaStream_t stream)
{
    rt::Coords const srcShape = src.getShape();
    int32_t const B = static_cast<int32_t>(srcShape[0]);
    int32_t const H = static_cast<int32_t>(srcShape[2]);
    int32_t const srcS = static_cast<int32_t>(srcShape[3]);
    int32_t const D = static_cast<int32_t>(srcShape[4]);
    // seqLen == 0 means full copy; otherwise compact copy of first seqLen tokens.
    int32_t const dstS = (seqLen > 0) ? seqLen : srcS;

    check::check(srcShape[1] == 2, "Source tensor must have shape [B, 2, H, S, D].");
    check::check(dstS <= srcS, "seqLen must be <= source capacity.");
    check::check(kDst.getDataType() == nvinfer1::DataType::kHALF, "kDst must be FP16.");
    check::check(vDst.getDataType() == nvinfer1::DataType::kHALF, "vDst must be FP16.");

    rt::Coords const kShape = kDst.getShape();
    rt::Coords const vShape = vDst.getShape();
    check::check(kShape[0] == B && kShape[1] == dstS && kShape[2] == H && kShape[3] == D,
        "kDst must have shape [B, dstS, H, D].");
    check::check(vShape[0] == B && vShape[1] == dstS && vShape[2] == H && vShape[3] == D,
        "vDst must have shape [B, dstS, H, D].");

    // Block config with safe thread count (≤ 1024)
    uint32_t const tx = (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
    uint32_t const ty = 4; // token dimension per block
    uint32_t const tz = 1; // process one head-pair per thread in z
    dim3 block(tx, ty, tz);

    // Grid covers only dstS tokens (compact when dstS < srcS).
    uint32_t const hpTilesPerBatch = (2 * H + tz - 1) / tz;
    dim3 grid((D + tx - 1) / tx, // x : feature dim
        (dstS + ty - 1) / ty,    // y : token dim
        hpTilesPerBatch * B);    // z : (batch, headPair)

    if (src.getDataType() == nvinfer1::DataType::kHALF)
    {
        cvtKVLayoutBHSDToSplitKVKernel<half><<<grid, block, 0, stream>>>(src.dataPointer<half>(),
            kDst.dataPointer<half>(), vDst.dataPointer<half>(), nullptr, nullptr, B, srcS, dstS, H, D);
    }
#if SUPPORTS_FP8
    else if (src.getDataType() == nvinfer1::DataType::kFP8)
    {
        check::check(!kvScaleQuantOrig.isEmpty(), "kvScaleQuantOrig is required for FP8 KV cache");
        check::check(kvScaleQuantOrig.getDataType() == nvinfer1::DataType::kFLOAT, "kvScaleQuantOrig must be FP32.");
        check::check(kvScaleQuantOrig.getShape().getNumDims() == 1 && kvScaleQuantOrig.getShape()[0] == 2,
            "kvScaleQuantOrig shall have shape [2] with layout [kScaleQuantOrig, vScaleQuantOrig].");
        float const* const scales = kvScaleQuantOrig.dataPointer<float>();
        float const* const kScaleQuantOrigPtr = scales + 0;
        float const* const vScaleQuantOrigPtr = scales + 1;
        cvtKVLayoutBHSDToSplitKVKernel<__nv_fp8_e4m3><<<grid, block, 0, stream>>>(src.dataPointer<__nv_fp8_e4m3>(),
            kDst.dataPointer<half>(), vDst.dataPointer<half>(), kScaleQuantOrigPtr, vScaleQuantOrigPtr, B, srcS, dstS,
            H, D);
    }
#endif
    else
    {
        check::check(false, "Unsupported KV cache dtype");
    }
}

} // namespace kernel
} // namespace trt_edgellm
