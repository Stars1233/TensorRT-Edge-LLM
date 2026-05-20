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
 * Build varlen metadata (seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum)
 * required by the SSD Blackwell kernel compiled with has_varlen=True.
 *
 * For Edge-LLM padded batched layout (each batch row = one sequence),
 * the metadata follows GDN-style auto-synthesis: when context_lengths == nullptr,
 * we synthesize uniform metadata where every batch has nchunks full chunks
 * (cl[b] == seq_len for all b).
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace mamba
{

/// Build varlen metadata fully on-device (CUDA-graph-compatible -- no host sync).
/// Caller passes `num_logical_chunks_upper = batch * nchunks_per_seq` and
/// `num_seqs = batch` directly to the kernel; the trailing slack in
/// chunk_indices is filled with sentinel -1 so the kernel's
/// `chunk_indices[physical_chunk+1]` lookup is safe up to the upper bound.
void buildSSDVarlenMetadata(int32_t* d_seq_idx, int32_t* d_chunk_indices, int32_t* d_chunk_offsets,
    int32_t* d_seq_chunk_cumsum, int32_t const* d_context_lengths, int32_t batch, int32_t seq_len, int32_t chunk_size,
    cudaStream_t stream);

/// Fill `d_valid_lens[0..batch)` with uniform `seq_len` value.
/// Used by the runner when caller's `context_lengths` is null (uniform batch);
/// the kernel's padded_mode end-of-seq clamp still requires a valid_lens tensor.
void fillUniformValidLens(int32_t* d_valid_lens, int32_t batch, int32_t seq_len, cudaStream_t stream);

} // namespace mamba
} // namespace trt_edgellm
