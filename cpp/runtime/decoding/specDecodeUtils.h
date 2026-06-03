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
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{
namespace spec_decode_utils
{

//! @brief Check whether a generation request is compatible with greedy-only spec-decode sampling.
//! @return nullptr if compatible; a human-readable reason string if not.
char const* isGreedyCompatible(LLMGenerationRequest const& request) noexcept;

//! @brief Load the draft engine from disk and return an EngineExecutor.
std::unique_ptr<EngineExecutor> loadDraftEngine(std::filesystem::path const& engineDir, DeploymentConfig& deployment);

//! @brief Copy accepted tokens from device buffers into the host-side context token lists.
void appendAcceptedTokens(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor const& deviceAcceptLength, Tensor const& deviceAcceptedTokenIds, int32_t maxAcceptDepth,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t stream);

} // namespace spec_decode_utils
} // namespace rt
} // namespace trt_edgellm
