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

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Device buffers used to sort token-major routes into expert-major order.
struct F16MoeRoutingBuffers
{
    int32_t* expertCounts{};       //!< INT32 [E].
    int32_t* expertOffsets{};      //!< INT32 [E + 1].
    int32_t* expertWriteOffsets{}; //!< INT32 [E].
    int32_t* sortedToExpanded{};   //!< INT32 [R].
    int32_t* expandedToSorted{};   //!< INT32 [R].
};

//! Device-resident grouped-GEMM descriptors shared by every CuTeDSL variant.
struct F16MoeGemmMetadata
{
    int32_t* problemShapes{}; //!< INT32 [E, 4], storing M, N, K, L.
    int32_t* strides{};       //!< INT32 [E, 3, 2], storing A, B, D strides.
    int64_t* addresses{};     //!< INT64 [E, 3], storing A, B, D addresses.
};

//! Internal setup for one grouped-GEMM stage.
struct F16MoeGemmSetup
{
    F16MoeGemmMetadata metadata{}; //!< Device-resident descriptors for this stage.
    void const* input{};           //!< FP16 expert-contiguous input.
    void const* weights{};         //!< FP16 expert weights.
    void* output{};                //!< FP16 expert-contiguous output.
    int32_t n{};                   //!< Output columns.
    int32_t k{};                   //!< Reduction columns.
};

//! Sort token-major routes by expert and populate both grouped-GEMM stages.
//! Uses a single-CTA fast path for up to 256 tokens and a multi-kernel fallback otherwise.
cudaError_t buildF16MoeRoutingAndGemmMetadata(F16MoeRoutingBuffers const& buffers, F16MoeGemmSetup const& fc1Setup,
    F16MoeGemmSetup const& fc2Setup, int32_t const* topkIds, int32_t numTokens, int32_t topK, int32_t numExperts,
    cudaStream_t stream) noexcept;

//! Gather hidden rows into the expert-major order described by sortedToExpanded.
cudaError_t gatherF16MoeHiddenRows(void const* hiddenStates, void* gatheredInput, int32_t const* sortedToExpanded,
    int32_t routedRows, int32_t topK, int32_t hiddenSize, cudaStream_t stream) noexcept;

//! Apply SwiGLU or ReLU-squared in FP32 and store the activated FP16 intermediate.
cudaError_t activateF16Moe(void const* rawFc1, void* activatedFc1, int32_t routedRows, int32_t moeInterSize,
    int32_t activationType, cudaStream_t stream) noexcept;

//! Weighted token-major route reduction with FP32 accumulation and one FP16 cast.
cudaError_t scatterF16MoeOutput(void const* routedOutput, int32_t const* expandedToSorted, float const* topkWeights,
    void* output, int32_t numTokens, int32_t topK, int32_t hiddenSize, cudaStream_t stream) noexcept;

} // namespace kernel
} // namespace trt_edgellm
