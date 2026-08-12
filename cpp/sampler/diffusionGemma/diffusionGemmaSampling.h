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

#include "common/tensor.h"

#include <cstdint>
#include <optional>

namespace trt_edgellm
{

//! Default deterministic seed for DiffusionGemma canvas initialization and re-noising.
inline constexpr uint64_t kDefaultDiffusionRandomSeed{42U};

//! Stateless RNG parameters for DiffusionGemma sampler kernels.
struct DiffusionRandomParams
{
    uint64_t seed{kDefaultDiffusionRandomSeed};
    uint64_t offset{0};
};

//! Scalar parameters for DiffusionGemma canvas initialization.
struct DiffusionCanvasInitParams
{
    int32_t vocabSize{0};
    DiffusionRandomParams random{};
};

//! Scalar parameters for DiffusionGemma entropy-bound acceptance and re-noising.
struct DiffusionCanvasUpdateParams
{
    float entropyThreshold{0.0F};
    float entropyBound{0.0F};
    int32_t stabilityWindow{0};
    bool forceAccept{false};
    int32_t vocabSize{0};
    DiffusionRandomParams random{};
};

/*!
 * \brief Sample DiffusionGemma canvas token IDs from logits with Gumbel-max.
 *
 * Samples each row from ``softmax(logits / temperature)`` using a stateless
 * Gumbel-max transform. The output tensor may be either flat ``[rows]`` or
 * canvas-shaped ``[batch-size, canvas-len]``; its volume must match the input
 * row count.
 *
 * \param[in] input Input tensor [GPU, Float/Half/BF16] with shape [rows, vocab-size]
 * \param[out] sampledIds Sampled token IDs [GPU, Int32], volume [rows]
 * \param[in] temperature Softmax temperature
 * \param[in] stream CUDA stream to execute the kernel
 * \param[in] random RNG seed and offset used by the stateless Gumbel sampler
 */
void sampleDiffusionTokensFromLogits(rt::Tensor const& input, rt::Tensor& sampledIds, float temperature,
    cudaStream_t stream, DiffusionRandomParams const& random = DiffusionRandomParams{});

/*!
 * \brief Compute DiffusionGemma Gumbel samples, argmax token IDs, and entropy from logits.
 *
 * This is equivalent to calling ``selectArgmaxAndComputeEntropy`` and
 * ``sampleDiffusionTokensFromLogits`` with the same logits and RNG parameters,
 * but it computes argmax, Gumbel sample, and entropy in one vocab pass and
 * removes the separate entropy scan.
 *
 * \param[in] input Input tensor [GPU, Float/Half/BF16] with shape [rows, vocab-size]
 * \param[out] sampledIds Gumbel-sampled token IDs [GPU, Int32], volume [rows]
 * \param[out] topIndices Argmax token IDs [GPU, Int32] with shape [rows, 1]
 * \param[out] entropy Entropy values [GPU, Float] with shape [rows]
 * \param[in] temperature Softmax temperature
 * \param[in] stream CUDA stream to execute the kernel
 * \param[in] random RNG seed and offset used by the stateless Gumbel sampler
 */
void sampleDiffusionTokensAndComputeEntropy(rt::Tensor const& input, rt::Tensor& sampledIds, rt::Tensor& topIndices,
    rt::Tensor& entropy, float temperature, cudaStream_t stream,
    DiffusionRandomParams const& random = DiffusionRandomParams{});

/*!
 * \brief Compute only DiffusionGemma argmax token IDs from logits.
 *
 * This is used by the final force-accept denoise step where the update policy
 * commits argmax tokens and does not consume sampled IDs or entropy values.
 *
 * \param[in] input Input tensor [GPU, Float/Half/BF16] with shape [rows, vocab-size]
 * \param[out] topIndices Argmax token IDs [GPU, Int32] with shape [rows, 1]
 * \param[in] temperature Positive logit scaling temperature
 * \param[in] stream CUDA stream to execute the kernel
 */
void selectDiffusionArgmaxFromLogits(
    rt::Tensor const& input, rt::Tensor& topIndices, float temperature, cudaStream_t stream);

/*!
 * \brief Initialize DiffusionGemma denoise canvas and sampler state on GPU.
 *
 * Fills the denoise canvas with random token IDs, resets previous argmax IDs,
 * stability counters, accepted mask, and accepted prefix lengths.
 *
 * \param[out] canvasIds Randomized canvas token IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[out] previousArgmaxIds Previous argmax IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[out] stableCounts Consecutive argmax counters including the current observation [GPU, Int32] with
 * shape [batch-size, canvas-len]. Convergence requires this count to be greater than stabilityWindow,
 * matching HF DiffusionGemma history semantics.
 * \param[out] acceptedMask Accepted-token mask [GPU, Int8] with shape [batch-size, canvas-len]
 * \param[out] prefixLengths Accepted prefix lengths [GPU, Int32] with shape [batch-size]
 * \param[in] params Scalar initialization parameters including vocab size and RNG state
 * \param[in] stream CUDA stream to execute the kernel
 */
void initializeDiffusionCanvas(rt::Tensor& canvasIds, rt::Tensor& previousArgmaxIds, rt::Tensor& stableCounts,
    rt::Tensor& acceptedMask, rt::Tensor& prefixLengths, DiffusionCanvasInitParams const& params, cudaStream_t stream);

/*!
 * \brief Apply DiffusionGemma entropy-bound acceptance and re-noising on GPU.
 *
 * Consumes Gumbel-sampled token IDs, argmax token IDs, and entropy values.
 * The next denoise canvas keeps Gumbel-sampled IDs for positions selected by the
 * entropy-bound budget and uniformly re-noises rejected positions. The argmax
 * canvas is tracked separately for convergence checks and for the causal commit
 * pass.
 *
 * \param[in] sampledIds Gumbel-sampled token IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[in] argmaxIds Top-1 token IDs [GPU, Int32] with shape [batch-size * canvas-len, 1]
 * \param[in] entropy Entropy values [GPU, Float] with shape [batch-size * canvas-len]
 * \param[out] canvasIds Next denoise canvas IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[out] argmaxCanvasIds Argmax canvas IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[out] previousArgmaxIds Previous argmax IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[out] stableCounts Consecutive argmax counters including the current observation [GPU, Int32] with
 * shape [batch-size, canvas-len]. Convergence requires this count to be greater than stabilityWindow,
 * matching HF DiffusionGemma history semantics.
 * \param[out] acceptedMask Entropy-bound budget mask [GPU, Int8] with shape [batch-size, canvas-len]
 * \param[out] prefixLengths Sticky converged prefix lengths [GPU, Int32] with shape [batch-size]
 * \param[in] params Scalar update parameters including entropy policy, force-accept state, vocab size, and RNG state
 * \param[in] stream CUDA stream to execute the kernel
 * \param[in] validCanvasLengths Optional per-batch valid lengths [GPU, Int32] with shape [batch-size].
 */
void diffusionSampleAndUpdateCanvas(rt::Tensor const& sampledIds, rt::Tensor const& argmaxIds,
    rt::Tensor const& entropy, rt::Tensor& canvasIds, rt::Tensor& argmaxCanvasIds, rt::Tensor& previousArgmaxIds,
    rt::Tensor& stableCounts, rt::Tensor& acceptedMask, rt::Tensor& prefixLengths,
    DiffusionCanvasUpdateParams const& params, cudaStream_t stream,
    rt::OptionalInputTensor validCanvasLengths = std::nullopt);

/*!
 * \brief Pack the accepted DiffusionGemma prefix from canvas layout to commit layout on GPU.
 *
 * \param[in] canvasIds Canvas token IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[out] commitCanvasIds Packed commit IDs [GPU, Int32] with shape [batch-size, block-len]
 * \param[in] batchSize Active batch size
 * \param[in] canvasLen Canvas stride
 * \param[in] blockLen Accepted commit length
 * \param[in] stream CUDA stream to execute the kernel
 */
void compactDiffusionCanvas(rt::Tensor const& canvasIds, rt::Tensor& commitCanvasIds, int32_t batchSize,
    int32_t canvasLen, int32_t blockLen, cudaStream_t stream);

/*!
 * \brief Pack per-batch accepted DiffusionGemma prefixes and pad shorter rows.
 *
 * \param[in] canvasIds Canvas token IDs [GPU, Int32] with shape [batch-size, canvas-len]
 * \param[in] commitLengths Per-batch commit lengths [GPU, Int32] with shape [batch-size]
 * \param[out] commitCanvasIds Packed commit IDs [GPU, Int32] with shape [batch-size, max-block-len]
 * \param[in] batchSize Active batch size
 * \param[in] canvasLen Canvas stride
 * \param[in] maxBlockLen Rectangular commit length bound
 * \param[in] padTokenId Token ID used to fill inactive padded positions
 * \param[in] stream CUDA stream to execute the kernel
 */
void compactDiffusionCanvas(rt::Tensor const& canvasIds, rt::Tensor const& commitLengths, rt::Tensor& commitCanvasIds,
    int32_t batchSize, int32_t canvasLen, int32_t maxBlockLen, int32_t padTokenId, cudaStream_t stream);

} // namespace trt_edgellm
