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
#include "tokenizer/tokenizer.h"

#include <string>
#include <tuple>
#include <vector>

namespace trt_edgellm
{

using SystemPromptCacheKey = std::tuple<std::string, std::string>;

/*! \brief Structure to hold cached system prompt and its KV cache.
 */
struct SystemPromptKVCache
{
    std::string systemPrompt;                     //!< The system prompt text
    std::vector<tokenizer::Rank> tokenizedPrompt; //!< Tokenized version of the system prompt
    std::vector<rt::Tensor> kvCacheLayers;        //!< Per-layer KV cache tensors for the system prompt
    std::vector<rt::Tensor>
        recurrentStateContents;                //!< Cached recurrent states for hybrid layers (empty if not applicable)
    std::vector<rt::Tensor> convStateContents; //!< Cached conv states for hybrid layers (empty if not applicable)
};

inline SystemPromptCacheKey keySystemPromptWithLoraWeights(
    std::string const& systemPrompt, std::string const& loraWeightsName)
{
    return std::make_tuple(systemPrompt, loraWeightsName);
}

} // namespace trt_edgellm
