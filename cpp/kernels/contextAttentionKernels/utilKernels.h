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

#pragma once

#include "common/tensor.h"

#include "common/cudaMacros.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace trt_edgellm
{
namespace kernel
{

//! FMHA_v2 packed-mask layout constants (must match the CUSTOM_MASK cubins and
//! TRT-LLM's fmhaPackedMask.cu reference implementation).
//!
//! The packed mask is a bitmask consumed by the flash-attention CUSTOM_MASK
//! kernels (fmha_v2 src/fmha/mask.h, MASK_VERSION 5).  Geometry:
//!  - The Q (row) dimension is padded per sequence to a multiple of 128
//!    (FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT).  `cu_mask_rows[b]` carries the
//!    per-batch prefix sum of padded rows.
//!  - The KV (col) dimension is padded to a multiple of 256
//!    (FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT); the row stride in bytes is
//!    paddedKvLen / 8 (one bit per position).
//!  - Bits are laid out per MMA tile: a CTA warp-group of 128 threads covers a
//!    64-row x 64-col tile with one uint32 per thread.  For thread t
//!    (warp = t/32, warpM = warp%4, lane = t%32) the base position within the
//!    tile is row = warpM*16 + lane/4, col = (lane%4)*2, and bit (4*ni + b)
//!    covers (row + 8*(b>>1), col + 8*ni + (b&1)) for ni in [0, 8).
//!  - uint32 words are ordered [mmaM][mmaN][128 threads] where mmaM indexes
//!    64-row tiles (globally, across the padded rows of all sequences) and
//!    mmaN indexes 64-col tiles.
constexpr int32_t kFMHA_PACKED_MASK_M_ALIGNMENT = 128;
constexpr int32_t kFMHA_PACKED_MASK_N_ALIGNMENT = 256;
constexpr int32_t kFMHA_PACKED_MASK_MMA_M = 64;
constexpr int32_t kFMHA_PACKED_MASK_MMA_N = 64;
constexpr int32_t kFMHA_PACKED_MASK_THREADS_PER_WARP_GROUP = 128;

//! Padded number of packed-mask rows for one sequence of length seqLen.
constexpr int64_t getPackedMaskRowsPerSeq(int64_t seqLen)
{
    return (seqLen + kFMHA_PACKED_MASK_M_ALIGNMENT - 1) / kFMHA_PACKED_MASK_M_ALIGNMENT * kFMHA_PACKED_MASK_M_ALIGNMENT;
}

//! Packed-mask row stride in bytes for a KV length of kvSeqLen (one bit per
//! position, KV dimension padded to a multiple of 256).
constexpr int64_t getPackedMaskRowStrideInBytes(int64_t kvSeqLen)
{
    return (kvSeqLen + kFMHA_PACKED_MASK_N_ALIGNMENT - 1) / kFMHA_PACKED_MASK_N_ALIGNMENT
        * kFMHA_PACKED_MASK_N_ALIGNMENT / 8;
}

//! Total packed-mask size in uint32 words for a batch of batchSize sequences,
//! each padded to seqLen rows and kvSeqLen cols.
constexpr int64_t getPackedMaskSizeInWords(int64_t batchSize, int64_t seqLen, int64_t kvSeqLen)
{
    return batchSize * getPackedMaskRowsPerSeq(seqLen) * getPackedMaskRowStrideInBytes(kvSeqLen)
        / static_cast<int64_t>(sizeof(uint32_t));
}

//! \brief Build the FMHA_v2 CUSTOM_MASK packed mask for Gemma4 vision-block prefill.
//!
//! Allowed(q, k) for q, k < contextLengths[b]:
//!   sliding-causal:  k <= q and (slidingWindowSize <= 0 or k > q - slidingWindowSize), OR
//!   vision block:    visionBlockIds[b][q] >= 0 and visionBlockIds[b][q] == visionBlockIds[b][k]
//! All other positions (including rows/cols >= contextLength and padded
//! rows/cols) have their bits cleared.
//!
//! \param[in]  visionBlockIds     int32_t [B, S]; -1 for text/audio/pad, non-negative
//!                                per contiguous image run.
//! \param[in]  contextLengths     int32_t [B]; actual token count per sequence.
//! \param[out] packedMask         uint32_t [getPackedMaskSizeInWords(B, S, S)] packed mask.
//! \param[out] cuMaskRows         int32_t [B+1]; prefix sum of padded mask rows
//!                                (i * getPackedMaskRowsPerSeq(S)).  Consumed by the kernels
//!                                via params.cu_mask_rows.
//! \param[in]  batchSize          Number of sequences B.
//! \param[in]  seqLen             Padded runtime sequence length S (Q length == KV length).
//! \param[in]  slidingWindowSize  Sliding window size counting the query itself
//!                                (<= 0 means plain causal).
//! \param[in]  stream             CUDA stream.
//! \throws std::runtime_error on invalid arguments.
void launchBuildVisionPackedMask(int32_t const* visionBlockIds, int32_t const* contextLengths, uint32_t* packedMask,
    int32_t* cuMaskRows, int32_t batchSize, int32_t seqLen, int32_t slidingWindowSize, cudaStream_t stream);

//! Expand [B, S] vision-block IDs into per-position [blockBegin, blockEnd]
//! interval tensors for the FFPA vision-block overlay prefill kernel.
//!
//! Each contiguous run of an identical non-negative ID inside the per-batch
//! valid prefix (contextLengths[b], clamped to seqLen) yields
//! blockBegin = run start and blockEnd = run end for every position in the
//! run.  Text/audio positions (ID < 0) and padding positions receive the
//! -1/-1 sentinel (empty interval).  All tensors are [B, S] int32 device
//! buffers.
void launchBuildVisionBlockRanges(int32_t const* visionBlockIds, int32_t const* contextLengths, int32_t* blockBegin,
    int32_t* blockEnd, int32_t batchSize, int32_t seqLen, cudaStream_t stream);

//! \brief Host-side wrapper that launches a lightweight CUDA kernel to compute prefix-sum of sequence lengths
//! and KV cache end indices.
//!
//! \param[in]  inputSeqLen       int32_t tensor with shape [B].  Actual token length of each request.
//! \param[in]  kvCacheStartIndices int32_t tensor with shape [B].  Start index of KV cache for each request.
//!                                (optional, pass in empty tensor to indicate zero start indices)
//! \param[out] cuQSeqLens        int32_t tensor with shape [B+1]. Exclusive prefix-sum of inputSeqLen.
//! \param[out] cuKVSeqLens       int32_t tensor with shape [B+1]. Exclusive prefix-sum of (kvCacheStartIndices[i] +
//!                                inputSeqLen[i]). If kvCacheStartIndices is empty, this will be exclusive prefix-sum
//!                                of inputSeqLen.
//! \param[out] kvCacheEndIdxs    int32_t tensor with shape [B].  Each element equals
//!                                kvCacheStartIndices[i] + runtimeSeqLen (Here we use padding to ease later kernel
//!                                launch).
//! \param[out] paddedCuKVSeqLens (optional) int32_t tensor with shape [B+1]. Exclusive prefix-sum of kvCacheEndIdxs
//!                                (= kvCacheStartIdx + runtimeSeqLen per batch). Pass std::nullopt to skip.
//!                                Background: CuTe DSL FMHA kernel uses bottom_right_align with offset = s_k - s_q.
//!                                Q is padded to runtimeSeqLen for all batches, so we must use padded KV lengths
//!                                (s_k = kvCacheEndIdx per batch) to keep offset non-negative. Using actual s_k
//!                                (< runtimeSeqLen for shorter batches) would produce a negative offset that masks
//!                                out valid KV positions, breaking attention.
//! \param[in]  runtimeSeqLen     Runtime sequence length (equals to the maximum of inputSeqLen).
//! \param[in]  stream            CUDA stream used to launch the kernel.
//! \note kvCacheStartIndices is optional. If it is not provided, kvStartIndices will be assumed to be 0.
//! \throws std::runtime_error if tensor shapes are invalid
void calCuQCuKVSeqLensAndKVEndIdxs(rt::Tensor const& inputSeqLen, rt::Tensor const& kvCacheStartIndices,
    rt::Tensor& cuQSeqLens, rt::Tensor& cuKVSeqLens, rt::Tensor& kvCacheEndIdxs,
    rt::OptionalOutputTensor paddedCuKVSeqLens, int32_t const runtimeSeqLen, cudaStream_t stream);

//! \brief Converts KV cache layout from [B, 2, H, S, D] into separate K and V tensors of shape [B, S, H, D].
//!
//! Splits the interleaved KV source into two independent FP16 output tensors, applying FP8 dequantization when
//! the source is FP8. Used in the chunked-prefill path so that the SEPARATE_Q_K_V FMHA kernels receive
//! separate K and V pointers.
//!
//! \param[in]  src             Source tensor with shape [B, 2, H, S, D].
//! \param[out] kDst            Destination K tensor with shape [B, S, H, D] (FP16).
//! \param[out] vDst            Destination V tensor with shape [B, S, H, D] (FP16).
//! \param[in]  kvScaleQuantOrig Optional packed dequant scale tensor for FP8 KV cache (shape [2], float).
//!             Layout: [kScaleQuantOrig, vScaleQuantOrig]. Pass an empty tensor for FP16 src.
//! \param[in]  stream          CUDA stream to launch the kernel on.
//! \throws std::runtime_error if tensor shapes or data types are invalid.
void cvtKVLayoutBHSDToSplitKV(rt::Tensor const& src, rt::Tensor& kDst, rt::Tensor& vDst,
    rt::Tensor const& kvScaleQuantOrig, int32_t seqLen, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
