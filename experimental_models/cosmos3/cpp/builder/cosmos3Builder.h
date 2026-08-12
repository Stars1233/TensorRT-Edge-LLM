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

#include <NvInfer.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace cosmos3
{

//! The policy components a full build produces, in dependency order.
std::vector<std::string> const& policyComponents();

//! Build the requested policy components from a package-exported ONNX root and
//! stage the runtime sidecars (tokenizer + token-embedding table) next to the
//! engines, so the engine directory is self-contained — mirrors llm_build.
//!
//! \param onnxRoot  Export root holding one subdirectory per component plus
//!                  ``text_tokenizer/``.
//! \param engineRoot  Output root; each component lands in ``<engineRoot>/<component>``.
//! \param components  Components to build (defaults to all via policyComponents()).
bool buildCosmos3Policy(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components, int32_t maxBatchSize);

//! Builds one Cosmos3 experimental component from a package-exported ONNX contract.
class Cosmos3Builder
{
public:
    Cosmos3Builder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
        std::string const& component, int32_t maxBatchSize);

    bool build();

private:
    bool parseConfig();
    bool setupOptimizationProfile(nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config,
        nvinfer1::INetworkDefinition const& network) const;
    bool inputExists(nvinfer1::INetworkDefinition const& network, std::string const& name) const;

    std::filesystem::path mOnnxDir;
    std::filesystem::path mEngineDir;
    std::string mComponent;
    int32_t mMaxBatchSize{1};
    std::string mOnnxFilename{"model.onnx"};
    std::string mEngineFilename;
    nlohmann::json mModelConfig;
};

} // namespace cosmos3
} // namespace trt_edgellm
