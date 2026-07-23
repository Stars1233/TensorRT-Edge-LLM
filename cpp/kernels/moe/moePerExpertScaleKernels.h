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

/// @brief Apply per-expert multiplicative scale to top-k routing weights.
///
/// After the MoE router selects top-k experts and renormalizes their weights,
/// this kernel applies a learned per-expert scale factor:
///   @code topkWeights[i] *= perExpertScale[topkIds[i]] @endcode
/// for all @p i in [0, numTokens * topK).
///
/// @param[in,out] topkWeights  Flat array of renormalized routing weights,
///                             shape [numTokens * topK]. Modified in-place.
/// @param[in]     topkIds      Expert indices for each slot, shape [numTokens * topK].
///                             Each value must be in [0, numExperts).
/// @param[in]     perExpertScale  Per-expert scale factors, shape [numExperts].
/// @param[in]     numTokens    Number of tokens (batch dimension).
/// @param[in]     topK         Number of selected experts per token.
/// @param[in]     stream       CUDA stream for asynchronous execution.
void applyPerExpertScale(float* topkWeights, int32_t const* topkIds, float const* perExpertScale, int32_t numTokens,
    int32_t topK, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
