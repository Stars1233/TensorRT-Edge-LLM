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

#include "allReducePlugin.h"

#include "common/logger.h"
#include "common/tensor.h"
#include "plugins/utils/pluginUtils.h"

#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kALL_REDUCE_PLUGIN_VERSION{"1"};
constexpr char const* kALL_REDUCE_PLUGIN_NAME{"AllReducePlugin"};

// Input/output indices
constexpr int32_t kIN_TENSOR_IDX{0};
constexpr int32_t kOUT_TENSOR_IDX{0};

// NCCL data type and op constants
constexpr int32_t kNcclFloat16 = 6;
constexpr int32_t kNcclFloat32 = 7;
constexpr int32_t kNcclBfloat16 = 10;
constexpr int32_t kNcclSum = 0;
constexpr int32_t kNcclSuccess = 0;

// Per-device NCCL state for single-process multi-GPU TP.
// Each GPU device gets its own NCCL communicator handle.
using NcclAllReduceFn = int (*)(void const*, void*, size_t, int, int, void*, cudaStream_t);
static std::unordered_map<int, void*> gNcclCommMap; //!< cudaDevice -> ncclComm_t
static NcclAllReduceFn gNcclAllReduceFn = nullptr;
static std::mutex gCommStateMutex;

} // namespace

void registerNcclCommForAllReducePlugin(int deviceId, void* ncclComm, void* ncclAllReduceFunc) noexcept
{
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    gNcclCommMap[deviceId] = ncclComm;
    gNcclAllReduceFn = reinterpret_cast<NcclAllReduceFn>(ncclAllReduceFunc);
}

// Static class fields initialization
PluginFieldCollection AllReducePluginCreator::mFieldCollection{};
std::vector<PluginField> AllReducePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AllReducePluginCreator);

// ========================== AllReducePlugin Implementation ==========================

AllReducePlugin::AllReducePlugin(std::string const& name, int32_t tpSize)
    : mLayerName(name)
    , mTpSize(tpSize)
{
    LOG_DEBUG("AllReducePlugin created: name=%s, tpSize=%d", name.c_str(), tpSize);
}

AllReducePlugin::AllReducePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    if (fc == nullptr)
    {
        throw std::invalid_argument("AllReducePlugin requires plugin fields");
    }
    auto tpSize = parsePluginScalarField<int32_t>("tp_size", fc);
    if (!tpSize.has_value())
    {
        throw std::invalid_argument("AllReducePlugin requires 'tp_size' field");
    }
    mTpSize = tpSize.value();
    LOG_DEBUG("AllReducePlugin deserialized from fields: name=%s, tpSize=%d", name.c_str(), mTpSize);
}

AllReducePlugin::~AllReducePlugin() {}

IPluginCapability* AllReducePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        return nullptr;
    }
}

IPluginV3* AllReducePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new AllReducePlugin(mLayerName, mTpSize);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePlugin clone failed: %s", e.what());
        return nullptr;
    }
}

int32_t AllReducePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t AllReducePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        if (nbOutputs != 1 || nbInputs != 1)
        {
            return -1;
        }
        outputTypes[kOUT_TENSOR_IDX] = inputTypes[kIN_TENSOR_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t AllReducePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /* shapeInputs */,
    int32_t /* nbShapeInputs */, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        if (nbInputs != 1 || nbOutputs != 1)
        {
            return -1;
        }
        outputs[kOUT_TENSOR_IDX] = inputs[kIN_TENSOR_IDX];
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool AllReducePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != 1 || nbOutputs != 1 || pos >= nbInputs + nbOutputs)
    {
        return false;
    }

    auto const& desc = inOut[pos].desc;
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (desc.type != DataType::kHALF && desc.type != DataType::kFLOAT && desc.type != DataType::kBF16)
    {
        return false;
    }

    if (pos >= nbInputs)
    {
        return desc.type == inOut[kIN_TENSOR_IDX].desc.type;
    }

    return true;
}

int32_t AllReducePlugin::configurePlugin(DynamicPluginTensorDesc const* /* in */, int32_t nbInputs,
    DynamicPluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 1 && nbOutputs == 1) ? 0 : -1;
}

size_t AllReducePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* /* inputs */, int32_t /* nbInputs */,
    DynamicPluginTensorDesc const* /* outputs */, int32_t /* nbOutputs */) const noexcept
{
    return 0;
}

int32_t AllReducePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* /* workspace */, cudaStream_t stream) noexcept
{
    try
    {
        auto const& inDesc = inputDesc[kIN_TENSOR_IDX];

        // Compute total number of elements
        int64_t numElements = 1;
        for (int32_t d = 0; d < inDesc.dims.nbDims; ++d)
        {
            numElements *= inDesc.dims.d[d];
        }

        // Look up NCCL comm for the current CUDA device
        int currentDevice = -1;
        cudaError_t const deviceErr = cudaGetDevice(&currentDevice);
        if (deviceErr != cudaSuccess)
        {
            LOG_ERROR("AllReducePlugin: cudaGetDevice failed: %s", cudaGetErrorString(deviceErr));
            return -1;
        }

        void* ncclComm = nullptr;
        NcclAllReduceFn ncclAllReduceFn = nullptr;
        {
            std::lock_guard<std::mutex> lock(gCommStateMutex);
            auto commIt = gNcclCommMap.find(currentDevice);
            if (commIt != gNcclCommMap.end())
            {
                ncclComm = commIt->second;
            }
            ncclAllReduceFn = gNcclAllReduceFn;
        }

        if (mTpSize <= 1 || ncclComm == nullptr || ncclAllReduceFn == nullptr)
        {
            // No tensor parallelism or no comm for this device: just copy input to output
            size_t const typeSize = rt::utils::getTypeSize(inDesc.type);
            cudaMemcpyAsync(outputs[kOUT_TENSOR_IDX], inputs[kIN_TENSOR_IDX], numElements * typeSize,
                cudaMemcpyDeviceToDevice, stream);
            return 0;
        }


        // Map TRT data type to NCCL data type
        int32_t ncclType = kNcclFloat16;
        if (inDesc.type == DataType::kFLOAT)
        {
            ncclType = kNcclFloat32;
        }
        else if (inDesc.type == DataType::kBF16)
        {
            ncclType = kNcclBfloat16;
        }

        // ---- Profiling: GPU event timing around AllReduce (zero-overhead) ----

        // Perform NCCL all-reduce using the per-device communicator
        int32_t ncclResult = ncclAllReduceFn(
            inputs[kIN_TENSOR_IDX], outputs[kOUT_TENSOR_IDX], numElements, ncclType, kNcclSum, ncclComm, stream);
        if (ncclResult != kNcclSuccess)
        {
            LOG_ERROR("AllReducePlugin: NCCL allReduce failed with error %d on device %d", ncclResult, currentDevice);
            return -1;
        }

        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

int32_t AllReducePlugin::onShapeChange(
    PluginTensorDesc const* /* in */, int32_t nbInputs, PluginTensorDesc const* /* out */, int32_t nbOutputs) noexcept
{
    return (nbInputs == 1 && nbOutputs == 1) ? 0 : -1;
}

IPluginV3* AllReducePlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

PluginFieldCollection const* AllReducePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("tp_size", &mTpSize, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

char const* AllReducePlugin::getPluginName() const noexcept
{
    return kALL_REDUCE_PLUGIN_NAME;
}

char const* AllReducePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AllReducePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* AllReducePlugin::getPluginVersion() const noexcept
{
    return kALL_REDUCE_PLUGIN_VERSION;
}

// ========================== AllReducePluginCreator Implementation ==========================

AllReducePluginCreator::AllReducePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* AllReducePluginCreator::getPluginName() const noexcept
{
    return kALL_REDUCE_PLUGIN_NAME;
}

PluginFieldCollection const* AllReducePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void AllReducePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* AllReducePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* AllReducePluginCreator::getPluginVersion() const noexcept
{
    return kALL_REDUCE_PLUGIN_VERSION;
}

IPluginV3* AllReducePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new AllReducePlugin(name, fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("AllReducePluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

void getNcclRegistrationForDevice(int deviceId, void** ncclComm, void** ncclAllReduceFunc) noexcept
{
    if (ncclComm != nullptr)
    {
        *ncclComm = nullptr;
    }
    if (ncclAllReduceFunc != nullptr)
    {
        *ncclAllReduceFunc = nullptr;
    }
    if (deviceId < 0)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(gCommStateMutex);
    auto it = gNcclCommMap.find(deviceId);
    if (ncclComm != nullptr && it != gNcclCommMap.end())
    {
        *ncclComm = it->second;
    }
    if (ncclAllReduceFunc != nullptr)
    {
        *ncclAllReduceFunc = reinterpret_cast<void*>(gNcclAllReduceFn);
    }
}

void* getNcclCommForDevice(int deviceId) noexcept
{
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    auto it = gNcclCommMap.find(deviceId);
    return (it != gNcclCommMap.end()) ? it->second : nullptr;
}

void* getNcclAllReduceFunc() noexcept
{
    std::lock_guard<std::mutex> lock(gCommStateMutex);
    return reinterpret_cast<void*>(gNcclAllReduceFn);
}

} // namespace plugins
} // namespace trt_edgellm

extern "C" void edgellmRegisterNcclCommForAllReducePlugin(
    int deviceId, void* ncclComm, void* ncclAllReduceFunc) noexcept
{
    trt_edgellm::plugins::registerNcclCommForAllReducePlugin(deviceId, ncclComm, ncclAllReduceFunc);
}

