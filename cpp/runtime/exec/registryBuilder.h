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

#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorRegistry.h"

#include <optional>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Build a TensorRegistry for the base LLM engine.
 *
 * Produces a registry with all tensor specs matching the engine's I/O contract
 * for `llmEngineRunner`. Config flags control which optional tensor groups
 * (deepstack, Mamba/recurrent state, EAGLE, LoRA) are included.
 *
 * @param cfg The engine configuration.
 * @param specDecodeBaseOutputHiddenDim Optional hidden-state output dim for a
 *        SpecDecode base engine. When absent, the legacy EAGLE-3 convention is used.
 * @return A populated TensorRegistry.
 */
TensorRegistry buildRegistryForLLM(
    LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim = std::nullopt);

/*!
 * @brief Build a TensorRegistry for a SpecDecode draft engine.
 *
 * Produces a registry with all tensor specs matching the engine's I/O contract
 * for the draft runner. The draft engine always uses plugin-based KV cache and
 * proposal-attention tensors.
 *
 * @param bundle The deployment configuration. `bundle.draft` and `bundle.specConfig`
 *               must both be set; the draft registry needs the consolidated
 *               SpecDecode settings to size cross-engine bindings (e.g. base
 *               hidden states fed into the draft).
 * @return A populated TensorRegistry.
 */
TensorRegistry buildRegistryForSpecDecodeDraft(DeploymentConfig const& bundle);

} // namespace rt
} // namespace trt_edgellm
