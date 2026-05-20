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

#include "ssdVarlenMetadata.h"

#include <cstdio>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace mamba
{

namespace
{

/// Generate seq_idx tensor [B, T]. seq_idx[b, t] = b if t < cl[b] else -1.
/// When d_context_lengths is null, treat all positions as valid (cl[b] = seq_len).
__global__ void buildSeqIdxKernel(int32_t* d_seq_idx, int32_t const* d_context_lengths, int32_t batch, int32_t seq_len)
{
    int32_t const t = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const b = blockIdx.y;
    if (b >= batch || t >= seq_len)
        return;
    int32_t const cl = d_context_lengths ? d_context_lengths[b] : seq_len;
    d_seq_idx[b * seq_len + t] = (t < cl) ? b : -1;
}

/// Fused metadata kernel: prefix-sums chunks_per_seq into seq_chunk_cumsum, then fills
/// chunk_indices/chunk_offsets for the full [0, batch*nchunks_per_batch) upper-bound
/// region. Real slots get global flat IDs, trailing slack gets sentinel -1 so the SSD
/// kernel's chunk_indices[physical_chunk+1] safety read can't false-trigger on garbage.
/// Single block -- batch is small (<= 256), inter-block sync would be wasted.
__global__ void buildChunkMetadataKernel(int32_t* d_seq_chunk_cumsum, // [batch+1] int32
    int32_t* d_chunk_indices,                                         // [batch * nchunks_per_batch] int32
    int32_t* d_chunk_offsets,                                         // [batch * nchunks_per_batch] int32
    int32_t const* d_context_lengths,                                 // [batch] int32 or nullptr
    int32_t batch, int32_t seq_len, int32_t chunk_size, int32_t nchunks_per_batch, int32_t total_slots)
{
    int32_t const tid = threadIdx.x;
    extern __shared__ int32_t smem[];

    if (tid < batch)
    {
        int32_t const cl = d_context_lengths ? d_context_lengths[tid] : seq_len;
        smem[tid] = (cl + chunk_size - 1) / chunk_size;
    }
    __syncthreads();

    // Sequential scan in thread 0 -- batch is small enough that parallel scan setup costs more.
    if (tid == 0)
    {
        int32_t cum = 0;
        d_seq_chunk_cumsum[0] = 0;
        for (int32_t b = 0; b < batch; ++b)
        {
            cum += smem[b];
            d_seq_chunk_cumsum[b + 1] = cum;
        }
        smem[batch] = cum;
    }
    __syncthreads();

    // Real-slot fill: each thread maps its k to (batch_idx, within-seq chunk_idx) and writes
    // the global flat ID to the packed position seq_chunk_cumsum[bp]+cp. Slack slots are
    // skipped here and filled with -1 in the sweep below -- we don't know their destination
    // until cumsum is done.
    for (int32_t k = tid; k < total_slots; k += blockDim.x)
    {
        int32_t const bp = k / nchunks_per_batch;
        int32_t const cp = k - bp * nchunks_per_batch;
        if (cp < smem[bp])
        {
            int32_t const dst = d_seq_chunk_cumsum[bp] + cp;
            d_chunk_indices[dst] = bp * nchunks_per_batch + cp;
            d_chunk_offsets[dst] = 0;
        }
    }
    __syncthreads();

    int32_t const real_end = smem[batch];
    for (int32_t k = real_end + tid; k < total_slots; k += blockDim.x)
    {
        d_chunk_indices[k] = -1;
        d_chunk_offsets[k] = 0;
    }
}

/// Fill valid_lens[0..batch) with uniform seq_len.
__global__ void fillUniformValidLensKernel(int32_t* d_valid_lens, int32_t batch, int32_t seq_len)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch)
    {
        d_valid_lens[idx] = seq_len;
    }
}

} // anonymous namespace

void fillUniformValidLens(int32_t* d_valid_lens, int32_t batch, int32_t seq_len, cudaStream_t stream)
{
    int32_t const block = 64;
    int32_t const grid = (batch + block - 1) / block;
    fillUniformValidLensKernel<<<grid, block, 0, stream>>>(d_valid_lens, batch, seq_len);
}

void buildSSDVarlenMetadata(int32_t* d_seq_idx, int32_t* d_chunk_indices, int32_t* d_chunk_offsets,
    int32_t* d_seq_chunk_cumsum, int32_t const* d_context_lengths, int32_t batch, int32_t seq_len, int32_t chunk_size,
    cudaStream_t stream)
{
    int32_t const nchunks_per_seq = (seq_len + chunk_size - 1) / chunk_size;
    int32_t const total_slots = batch * nchunks_per_seq;

    // Threads >= max(batch, 256); smem fits chunks_per_seq[batch] + cum total at [batch].
    int32_t const threads = (batch <= 256) ? 256 : ((batch + 31) / 32) * 32;
    size_t const smem_bytes = static_cast<size_t>(batch + 1) * sizeof(int32_t);
    buildChunkMetadataKernel<<<1, threads, smem_bytes, stream>>>(d_seq_chunk_cumsum, d_chunk_indices, d_chunk_offsets,
        d_context_lengths, batch, seq_len, chunk_size, nchunks_per_seq, total_slots);

    int32_t const threadsX = 128;
    dim3 const block(threadsX, 1, 1);
    dim3 const grid((seq_len + threadsX - 1) / threadsX, batch, 1);
    buildSeqIdxKernel<<<grid, block, 0, stream>>>(d_seq_idx, d_context_lengths, batch, seq_len);
}

} // namespace mamba
} // namespace trt_edgellm
