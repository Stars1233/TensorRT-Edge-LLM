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

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! GPU GPTQ → Marlin repack for MoE experts (vLLM ``gptq_marlin_repack`` style).
//!
//! Input:  GPTQ ``qweight`` ``[E, K/8, N]`` int32 (8 INT4 values packed along K).
//! Optional ``qzeros`` ``[E, G, N/8]`` int32 — when non-null, remaps to Marlin's
//! ``(q - 8) * scale`` convention via ``q' = clamp(q - z - zpOffset + 8, 0, 15)``.
//! Output: Marlin ``[E, K/16, 2*N]`` int32 in the Int4MoePlugin layout.
//!
//! Requires ``K % 16 == 0``, ``N % 64 == 0``. When ``qzeros`` is non-null,
//! ``G * groupSize == K``.
//!
//! Algorithm mirrors vLLM ``csrc/.../marlin/gptq_marlin_repack.cu`` (packed GPTQ
//! → Marlin on device, no host nibble expand). Packing indices match Edge-LLM
//! ``pack_int4_awq_marlin`` / ``marlinPackSwizzle``.
cudaError_t launchGptqMarlinRepack(int32_t const* dQweightE_K8_N, int32_t const* dQzerosE_G_N8_orNull, int32_t* dMarlin,
    int32_t E, int32_t N, int32_t K, int32_t numGroups, int32_t groupSize, int32_t zeroPointOffset,
    cudaStream_t stream);

//! Repack non-contiguous expert projections directly from CUDA-mapped
//! checkpoint storage. ``secondQweights`` is null for a single projection.
cudaError_t launchGptqMarlinRepackSourceBatch(int32_t const* const* firstQweights, int32_t const* const* secondQweights,
    int32_t const* const* firstQzeros, int32_t const* const* secondQzeros, int32_t* marlinOutput, int32_t count,
    int32_t projectionN, int32_t K, int32_t numGroups, int32_t groupSize, int32_t zeroPointOffset, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
