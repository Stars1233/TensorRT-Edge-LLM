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

/**
 * @brief Launch kernel to build Marlin indices from slot lists (per-expert).
 */
void launchBuildMarlinIndicesKernel(int32_t const* slotsByExpertWorkspace, int32_t const* slotsPerExpertWorkspace,
    int32_t const* paddedCounts, int32_t const* paddedOffsets, float const* topkWeights, int32_t* sortedTokenIds,
    float* topkWeightsFlat, int32_t* expertIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    int32_t moeBlockSize, cudaStream_t stream);

/**
 * @brief Build the trivial Marlin routing arrays for a dense (single-expert, topK=1) GEMM.
 *
 * The dense case degenerates the MoE routing: every one of the @p numTokens rows maps to expert 0 in order, so
 * sortedTokenIds is the identity 0..numTokens-1 with padded tail slots set to the out-of-range sentinel numTokens
 * (== numTokens*topK, matching marlin_template.h's `idx < prob_m*top_k` masking), expertIds is all zeros, and
 * numTokensPostPadded is the block-aligned row count. topkWeights is filled with 1.0f purely to keep the shared
 * Marlin call signature valid; it is unused because the dense wrapper passes mulTopkWeights=false.
 *
 * @param sortedTokenIds      [paddedRows] (INT32) output slot ids
 * @param expertIds           [paddedRows/moeBlockSize] (INT32) output, all zeros
 * @param numTokensPostPadded [1] (INT32) output, set to paddedRows
 * @param topkWeights         [paddedRows] (FP32) output, all 1.0f (unused by dense path)
 * @param numTokens           Number of real rows M
 * @param paddedRows          Block-aligned row count ceilDiv(M, moeBlockSize)*moeBlockSize
 * @param moeBlockSize        Marlin block size (8 for decode, 32 for prefill)
 */
void launchBuildDenseMarlinIndicesKernel(int32_t* sortedTokenIds, int32_t* expertIds, int32_t* numTokensPostPadded,
    float* topkWeights, int32_t numTokens, int32_t paddedRows, int32_t moeBlockSize, cudaStream_t stream);

/**
 * @brief Launch kernel to aggregate slot outputs back to tokens (sum over topK in slot order).
 */
void launchAggregateSlotOutputsKernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens, int32_t topK,
    int32_t outDim, cudaStream_t stream);

/**
 * @brief Launch the BF16 slot-output aggregation kernel.
 *
 * Each output element is accumulated over topK slots in FP32 and converted to BF16 once at the final store.
 */
void launchAggregateSlotOutputsBf16Kernel(void const* slotOutputs, void* aggregatedOutput, int32_t numTokens,
    int32_t topK, int32_t outDim, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
