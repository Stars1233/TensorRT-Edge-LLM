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
 * Edge-LLM MoE Marlin W4A16 GEMM Kernel Interface
 *
 * This header provides the rt::Tensor based interface for MoE W4A16 GEMM
 * using Marlin optimized kernels with AWQ quantization format.
 *
 * AWQ (kU4) format:
 * - Uses unsigned 4-bit integers with zero point = 8 baked in
 * - Dequantization: weight_fp16 = (weight_int4 - 8) * scale
 *
 * Note: Weight repacking (AWQ -> Marlin format) should be done in Python
 * using the awq_marlin_repack utility. This interface expects Marlin-packed
 * weights in Marlin format.
 */

#pragma once

#include "common/tensor.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief MoE W4A16 GEMM using Marlin kernel with AWQ format
 *
 * Performs MoE grouped GEMM with 4-bit AWQ quantized weights and 16-bit activations.
 * Uses kU4 format where zero point = 8 is baked into dequantization.
 *
 * Dequantization: weight_fp16 = (weight_int4 - 8) * scale
 *
 * Note: Weights must be transformed into Marlin format using awq_marlin_repack.
 *
 * Internally uses FP32 reduction for numerical accuracy. The workspace buffer
 * must be sized using getMoeMarlinWorkspaceSize() which includes space for both
 * synchronization locks and the FP32 reduction buffer.
 *
 * @param input Input activations [numTokens, hiddenDim] (FP16)
 * @param output Output tensor [numTokens * topK, outDim] (FP16)
 * @param weights Marlin-repacked INT4 weights [numExperts, K/tile, N*tile/pack]
 * @param scales Per-group scales [numExperts, numGroups, outDim] (FP16)
 * @param sortedTokenIds Sorted token indices [numTokensPadded] (INT32)
 * @param expertIds Expert assignment per block [numBlocks] (INT32)
 * @param numTokensPostPadded Total padded token count [1] (INT32)
 * @param topkWeights Routing weights [numTokensPadded] (FP32)
 * @param workspace Workspace buffer sized by getMoeMarlinWorkspaceSize() (INT32)
 * @param moeBlockSize MoE processing block size (8, 16, 32, 48, or 64)
 * @param topK Number of experts per token
 * @param mulTopkWeights Whether to multiply output by topk weights
 * @param stream CUDA stream
 */
void moeAwqW4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& scales, rt::Tensor const& sortedTokenIds, rt::Tensor const& expertIds,
    rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights, rt::Tensor& workspace, int64_t moeBlockSize,
    int64_t topK, bool mulTopkWeights, cudaStream_t stream);

/*!
 * @brief MoE FP16/BF16-A, E2M1-W, FP16/BF16-C GEMM using Marlin.
 *
 * The block scales are raw E4M3 bytes in Marlin order with a fixed K-group
 * size of 16. The per-expert global scales use the activation data type and
 * must already include its exponent adjustment for Marlin's skip-flop E2M1 conversion.
 *
 * @param input Input activations [numTokens, hiddenDim] (FP16 or BF16)
 * @param output Output tensor [numTokens * topK, outDim] (same type as input)
 * @param weights Marlin-packed E2M1 codes [numExperts, K/16, 2*outDim] (INT32 view)
 * @param blockScales Marlin-permuted E4M3 bytes [numExperts, K/16, outDim] (INT8)
 * @param globalScales Per-expert adjusted global scales [numExperts] (same type as input)
 * @param sortedTokenIds Sorted slot indices [maxRoutedRows] (INT32)
 * @param expertIds Expert assignment per routed block (INT32)
 * @param numTokensPostPadded Actual padded routed-row count [1] (INT32)
 * @param topkWeights Routing weights indexed by unpadded slot id (FP32)
 * @param workspace Workspace buffer sized by getMoeMarlinWorkspaceSize() (INT32)
 * @param moeBlockSize MoE processing block size (8, 16, 32, 48, or 64)
 * @param topK Number of experts per input row
 * @param mulTopkWeights Whether to multiply each output slot by its routing weight
 * @param stream CUDA stream
 */
void moeNvfp4A16MarlinGemm(rt::Tensor const& input, rt::Tensor& output, rt::Tensor const& weights,
    rt::Tensor const& blockScales, rt::Tensor const& globalScales, rt::Tensor const& sortedTokenIds,
    rt::Tensor const& expertIds, rt::Tensor const& numTokensPostPadded, rt::Tensor const& topkWeights,
    rt::Tensor& workspace, int64_t moeBlockSize, int64_t topK, bool mulTopkWeights, cudaStream_t stream);

/*!
 * @brief Get required workspace size for MoE Marlin GEMM
 *
 * Returns the total workspace size needed, which includes:
 * - Synchronization locks for thread blocks
 * - FP32 reduction buffer (c_tmp) for numerical accuracy
 *
 * @param numTokensPadded Maximum padded token count
 * @param outDim Output dimension N
 * @param moeBlockSize MoE block size
 * @param numSMs Number of SMs on the device
 * @return Required workspace size in number of int32_t elements
 */
int64_t getMoeMarlinWorkspaceSize(int64_t numTokensPadded, int64_t outDim, int64_t moeBlockSize, int64_t numSMs);

} // namespace kernel
} // namespace trt_edgellm
