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

#include "runtime/state/externalWeightManager.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/trtUtils.h"
#include "runtime/exec/engineExecutor.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

using Json = nlohmann::json;

void validateExternalWeightTensor(nvinfer1::ICudaEngine const& engine, Tensor const& tensor)
{
    std::string const& tensorName = tensor.getName();
    ELLM_CHECK(isEngineInput(engine, tensorName), "External weight tensor " + tensorName + " is not an engine input");

    nvinfer1::DataType const expectedType = engine.getTensorDataType(tensorName.c_str());
    ELLM_CHECK(tensor.getDataType() == expectedType,
        "External weight tensor " + tensorName
            + " dtype mismatch. Engine dtype=" + std::to_string(static_cast<int32_t>(expectedType))
            + ", external weight dtype=" + std::to_string(static_cast<int32_t>(tensor.getDataType())));

    // Externalized weights are exported with fully-static shapes (see
    // tensorrt_edgellm/external_weights.py::_add_external_weight_inputs),
    // so the engine's declared shape carries no -1 dims.
    nvinfer1::Dims const expectedDims = engine.getTensorShape(tensorName.c_str());
    ELLM_CHECK(!hasDynamicDims(expectedDims),
        "External weight tensor " + tensorName
            + " has dynamic dims in engine signature; externalized weights must be exported with fixed shapes");
    ELLM_CHECK(dimsEqual(expectedDims, tensor.getShape().getTRTDims()),
        "External weight tensor " + tensorName + " shape mismatch. Engine shape=" + dimsToString(expectedDims)
            + ", external weight shape=" + tensor.getShape().formatString());
}

void loadExternalWeightTensors(std::filesystem::path const& engineDir, std::filesystem::path const& configPath,
    std::vector<Tensor>& externalWeights, cudaStream_t stream)
{
    std::ifstream ifs(configPath);
    ELLM_CHECK(ifs.is_open(), "Failed to open config file for external weights: " + configPath.string());

    Json configJson;
    try
    {
        configJson = Json::parse(ifs);
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error(
            "JSON parse error in " + configPath.string() + " while loading external weights: " + e.what());
    }

    Json const externalWeightFiles = configJson.value("external_weight_files", Json::array());
    ELLM_CHECK(externalWeightFiles.is_array(),
        "external_weight_files must be an array when present in " + configPath.string());
    if (externalWeightFiles.empty())
    {
        externalWeights.clear();
        return;
    }

    std::vector<Tensor> loadedWeights;
    for (auto const& fileEntry : externalWeightFiles)
    {
        if (!fileEntry.is_object() || !fileEntry.contains("file") || !fileEntry["file"].is_string())
        {
            throw std::runtime_error(
                "Malformed external weight file entry in " + configPath.string() + ": " + fileEntry.dump());
        }

        std::filesystem::path const filePath = engineDir / fileEntry["file"].get<std::string>();
        std::vector<Tensor> tensors;
        ELLM_CHECK(safetensors::loadSafetensors(filePath, tensors, stream),
            "Failed to load external weight file: " + filePath.string());

        if (fileEntry.contains("tensors"))
        {
            ELLM_CHECK(fileEntry["tensors"].is_array(),
                "External weight file tensors entry must be an array: " + fileEntry.dump());
            for (auto const& expectedNameJson : fileEntry["tensors"])
            {
                ELLM_CHECK(expectedNameJson.is_string(),
                    "External weight tensor names must be strings in " + configPath.string() + ": "
                        + expectedNameJson.dump());
                std::string const expectedName = expectedNameJson.get<std::string>();
                auto const found = std::find_if(tensors.begin(), tensors.end(),
                    [&expectedName](Tensor const& tensor) { return tensor.getName() == expectedName; });
                ELLM_CHECK(found != tensors.end(),
                    "Expected external weight tensor " + expectedName + " not found in " + filePath.string());
            }
        }

        loadedWeights.reserve(loadedWeights.size() + tensors.size());
        std::move(tensors.begin(), tensors.end(), std::back_inserter(loadedWeights));
    }

    externalWeights = std::move(loadedWeights);
}

} // namespace

void ExternalWeightManager::load(
    std::filesystem::path const& engineDir, std::filesystem::path const& configPath, cudaStream_t stream)
{
    ELLM_CHECK(!mLoaded, "ExternalWeightManager::load called more than once");

    std::vector<Tensor> loadedWeights;
    loadExternalWeightTensors(engineDir, configPath, loadedWeights, stream);

    mWeights = std::move(loadedWeights);
    mLoaded = true;
    if (!mWeights.empty())
    {
        LOG_INFO("Loaded %d externalized model weight tensor(s)", static_cast<int32_t>(mWeights.size()));
    }
}

void ExternalWeightManager::validateAgainstEngine(EngineExecutor const& executor, std::string_view engineLabel)
{
    ELLM_CHECK(mLoaded, "ExternalWeightManager::validateAgainstEngine called before load");
    ELLM_CHECK(!mValidated, "ExternalWeightManager::validateAgainstEngine called more than once");

    auto const& engine = executor.getEngine();
    for (auto const& tensor : mWeights)
    {
        validateExternalWeightTensor(engine, tensor);
    }

    mValidated = true;
    if (!mWeights.empty())
    {
        LOG_INFO("Validated %d externalized model weight tensor(s) for %.*s engine",
            static_cast<int32_t>(mWeights.size()), static_cast<int32_t>(engineLabel.size()), engineLabel.data());
    }
}

void ExternalWeightManager::registerTensorMapEntries(TensorMap& map)
{
    ELLM_CHECK(mValidated, "ExternalWeightManager::registerTensorMapEntries called before validateAgainstEngine");
    ELLM_CHECK(!mRegistered, "ExternalWeightManager::registerTensorMapEntries called more than once");
    for (auto& tensor : mWeights)
    {
        map.set(tensor.getName(), tensor);
    }
    mRegistered = true;
}

} // namespace rt
} // namespace trt_edgellm
