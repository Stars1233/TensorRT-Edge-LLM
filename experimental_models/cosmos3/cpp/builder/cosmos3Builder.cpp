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

#include "builder/cosmos3Builder.h"

#include "builder/builderUtils.h"
#include "common/bindingNames.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

#include <algorithm>
#include <fstream>
#include <system_error>
#include <vector>

namespace trt_edgellm
{
namespace cosmos3
{

namespace
{

nvinfer1::Dims dimsFromJson(nlohmann::json const& shape)
{
    std::vector<int64_t> values = shape.get<std::vector<int64_t>>();
    return builder::createDims(values);
}

std::string defaultEngineFilename(std::string const& component)
{
    return component + ".engine";
}

//! Copy the tokenizer directory and token-embedding table produced by the
//! exporter into the engine root, so the runtime engine directory is
//! self-contained (matches how llm_build stages tokenizer/embedding sidecars).
bool stageRuntimeArtifacts(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components)
{
    // The token-embedding table ships with the und_prefill export; only stage
    // it when that component is part of this build.
    bool const builtUndPrefill = std::find(components.begin(), components.end(), "und_prefill") != components.end();
    if (builtUndPrefill)
    {
        std::filesystem::path const embedSrc = onnxRoot / "und_prefill" / "embed_tokens.safetensors";
        std::filesystem::path const embedDst = engineRoot / "embed_tokens.safetensors";
        if (!file_io::copyFile(embedSrc.string(), embedDst.string()))
        {
            LOG_ERROR("Failed to stage %s to %s", embedSrc.string().c_str(), embedDst.string().c_str());
            return false;
        }
        LOG_INFO("Staged token-embedding table: %s", embedDst.string().c_str());
    }

    std::filesystem::path const tokenizerSrc = onnxRoot / "text_tokenizer";
    if (std::filesystem::exists(tokenizerSrc))
    {
        std::error_code ec;
        std::filesystem::copy(tokenizerSrc, engineRoot / "text_tokenizer",
            std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec);
        if (ec)
        {
            LOG_ERROR(
                "Failed to stage tokenizer directory %s: %s", tokenizerSrc.string().c_str(), ec.message().c_str());
            return false;
        }
        LOG_INFO("Staged tokenizer directory: %s", (engineRoot / "text_tokenizer").string().c_str());
    }
    else
    {
        LOG_WARNING(
            "Tokenizer directory not found at %s; runtime will need it staged manually", tokenizerSrc.string().c_str());
    }
    return true;
}

} // namespace

std::vector<std::string> const& policyComponents()
{
    static std::vector<std::string> const kComponents{"und_prefill", "gen", "vae_encoder"};
    return kComponents;
}

bool buildCosmos3Policy(std::filesystem::path const& onnxRoot, std::filesystem::path const& engineRoot,
    std::vector<std::string> const& components, int32_t maxBatchSize)
{
    for (auto const& component : components)
    {
        Cosmos3Builder builder(onnxRoot / component, engineRoot / component, component, maxBatchSize);
        if (!builder.build())
        {
            LOG_ERROR("Failed to build Cosmos3 %s engine.", component.c_str());
            return false;
        }
        LOG_INFO("Cosmos3 %s engine built successfully.", component.c_str());
    }
    return stageRuntimeArtifacts(onnxRoot, engineRoot, components);
}

Cosmos3Builder::Cosmos3Builder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
    std::string const& component, int32_t maxBatchSize)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mComponent(component)
    , mMaxBatchSize(maxBatchSize)
    , mEngineFilename(defaultEngineFilename(component))
{
}

bool Cosmos3Builder::build()
{
    auto pluginHandles = loadEdgellmPluginLib();

    if (!parseConfig())
    {
        return false;
    }

    auto [trtBuilder, network] = builder::createBuilderAndNetwork();
    if (!trtBuilder || !network)
    {
        return false;
    }

    std::filesystem::path const onnxPath = mOnnxDir / mOnnxFilename;
    auto parser = builder::parseOnnxModel(network.get(), onnxPath.string());
    if (!parser)
    {
        return false;
    }

    LOG_DEBUG("%s", builder::printNetworkInfo(network.get(), ("Cosmos3 " + mComponent).c_str()).c_str());

    auto config = builder::createBuilderConfig(trtBuilder.get());
    if (!config)
    {
        return false;
    }

    if (!setupOptimizationProfile(*trtBuilder, *config, *network))
    {
        return false;
    }

    if (!std::filesystem::exists(mEngineDir) && !std::filesystem::create_directories(mEngineDir))
    {
        LOG_ERROR("Failed to create Cosmos3 engine directory %s", mEngineDir.string().c_str());
        return false;
    }

    std::filesystem::path const enginePath = mEngineDir / mEngineFilename;
    if (!builder::buildAndSerializeEngine(trtBuilder.get(), network.get(), config.get(), enginePath.string()))
    {
        return false;
    }

    nlohmann::json builderConfig = mModelConfig.value("builder_config", nlohmann::json::object());
    builderConfig["max_batch_size"] = mMaxBatchSize;
    if (!builder::saveConfigWithBuilderInfo(mEngineDir, mModelConfig, builderConfig))
    {
        return false;
    }

    return true;
}

bool Cosmos3Builder::parseConfig()
{
    std::filesystem::path const configPath = mOnnxDir / "config.json";
    if (!builder::loadJsonConfig(configPath.string(), mModelConfig))
    {
        return false;
    }

    std::string const modelVersion = mModelConfig.value(trt_edgellm::binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    std::string const configComponent = mModelConfig.value("component", mComponent);
    if (configComponent != mComponent)
    {
        LOG_ERROR("Cosmos3 component mismatch: CLI requested %s but config contains %s", mComponent.c_str(),
            configComponent.c_str());
        return false;
    }
    mOnnxFilename = mModelConfig.value("onnx_filename", mOnnxFilename);
    mEngineFilename = mModelConfig.value("engine_filename", mEngineFilename);
    return true;
}

bool Cosmos3Builder::inputExists(nvinfer1::INetworkDefinition const& network, std::string const& name) const
{
    for (int32_t i = 0; i < network.getNbInputs(); ++i)
    {
        if (network.getInput(i) != nullptr && name == network.getInput(i)->getName())
        {
            return true;
        }
    }
    return false;
}

bool Cosmos3Builder::setupOptimizationProfile(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network) const
{
    nlohmann::json const profileJson = mModelConfig.value("optimization_profile", nlohmann::json::object());
    if (profileJson.empty())
    {
        LOG_INFO("Cosmos3 %s has no dynamic optimization profile", mComponent.c_str());
        return true;
    }

    auto* profile = builder.createOptimizationProfile();
    bool ok = true;
    for (auto const& item : profileJson.items())
    {
        std::string const& name = item.key();
        if (!inputExists(network, name))
        {
            LOG_WARNING("Cosmos3 %s profile entry %s is not an ONNX input; skipping", mComponent.c_str(), name.c_str());
            continue;
        }
        nlohmann::json const& shape = item.value();
        nvinfer1::Dims const minDims = dimsFromJson(shape.at("min"));
        nvinfer1::Dims const optDims = dimsFromJson(shape.at("opt"));
        nvinfer1::Dims maxDims = dimsFromJson(shape.at("max"));
        // The exported contract describes the canonical single-request shapes; --max-batch-size
        // widens the leading (batch) axis of every input's max bound so one engine serves any
        // request batch in [min, maxBatchSize].
        if (mMaxBatchSize > 1 && maxDims.nbDims > 0)
        {
            maxDims.d[0] = std::max(maxDims.d[0], static_cast<int64_t>(mMaxBatchSize));
        }
        ok &= builder::setOptimizationProfile(profile, name.c_str(), minDims, optDims, maxDims);
    }

    if (!ok)
    {
        LOG_ERROR("Failed to set Cosmos3 %s optimization profile", mComponent.c_str());
        return false;
    }

    LOG_DEBUG("%s", builder::printOptimizationProfile(profile, (mComponent + "_profile").c_str(), &network).c_str());
    config.addOptimizationProfile(profile);
    return true;
}

} // namespace cosmos3
} // namespace trt_edgellm
