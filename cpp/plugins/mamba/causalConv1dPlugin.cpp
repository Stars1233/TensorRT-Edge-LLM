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

#include "causalConv1dPlugin.h"

#include "common/logger.h"
#include "kernels/mamba/causalConv1d.h"
#include "plugins/utils/pluginUtils.h"

#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <mutex>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kCAUSAL_CONV_PLUGIN_VERSION{"1"};
constexpr char const* kCAUSAL_CONV_PLUGIN_NAME{"causal_conv1d"};

constexpr int32_t kIN_X_IDX{0};
constexpr int32_t kIN_WEIGHT_IDX{1};
constexpr int32_t kIN_BIAS_IDX{2};
constexpr int32_t kIN_CONV_STATE_IDX{3};
constexpr int32_t kIN_CONTEXT_LENGTHS_IDX{4};
constexpr int32_t kOUT_IDX{0};
constexpr int32_t kOUT_CONV_STATE_IDX{1};
constexpr int32_t kOUT_INTERMEDIATE_CONV_STATES{2};
constexpr int32_t kNUM_INPUTS{5};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_MTP_OPTIONAL_OUTPUTS{1};

std::optional<int32_t> parsePluginIntField(std::string const& fieldName, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        PluginField const& pluginField = fc->fields[i];
        if (fieldName != pluginField.name || pluginField.length != 1 || pluginField.data == nullptr)
        {
            continue;
        }
        if (pluginField.type == PluginFieldType::kINT32)
        {
            return *static_cast<int32_t const*>(pluginField.data);
        }
        if (pluginField.type == PluginFieldType::kINT64)
        {
            return static_cast<int32_t>(*static_cast<int64_t const*>(pluginField.data));
        }
    }
    return std::nullopt;
}

} // namespace

PluginFieldCollection CausalConv1dPluginCreator::mFieldCollection{};
std::vector<PluginField> CausalConv1dPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CausalConv1dPluginCreator);

// ---------------------------------------------------------------------------
// Plugin — construction / destruction
// ---------------------------------------------------------------------------

CausalConv1dPlugin::CausalConv1dPlugin(
    std::string const& name, int32_t stride, int32_t padding, int32_t dilation, int32_t groups, bool useMTP)
    : mLayerName(name)
    , mStride(stride)
    , mPadding(padding)
    , mDilation(dilation)
    , mGroups(groups)
    , mUseMTP(useMTP)
{
}

CausalConv1dPlugin::CausalConv1dPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    auto const* d = static_cast<char const*>(data);
    std::memcpy(&mStride, d, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(&mPadding, d, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(&mDilation, d, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(&mGroups, d, sizeof(int32_t));
    d += sizeof(int32_t);
    // mUseMTP was added later; tolerate older serialized data that is only 4 ints.
    if (length >= 5 * sizeof(int32_t))
    {
        int32_t useMTPInt = 0;
        std::memcpy(&useMTPInt, d, sizeof(int32_t));
        mUseMTP = (useMTPInt != 0);
    }
}

CausalConv1dPlugin::~CausalConv1dPlugin() {}

// ---------------------------------------------------------------------------
// IPluginV2DynamicExt
// ---------------------------------------------------------------------------

IPluginV2DynamicExt* CausalConv1dPlugin::clone() const noexcept
{
    auto* plugin = new CausalConv1dPlugin(mLayerName, mStride, mPadding, mDilation, mGroups, mUseMTP);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

int32_t CausalConv1dPlugin::getNbOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS + (mUseMTP ? kNUM_MTP_OPTIONAL_OUTPUTS : 0);
}

DataType CausalConv1dPlugin::getOutputDataType(
    int32_t index, DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    // All outputs (conv output, conv state, intermediate_conv_states) have the same type as x (FP16).
    (void) index;
    return inputTypes[kIN_X_IDX];
}

DimsExprs CausalConv1dPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == kOUT_IDX)
    {
        // Output[0]: same shape as x [batch, seq_len, dim].
        return inputs[kIN_X_IDX];
    }
    if (outputIndex == kOUT_INTERMEDIATE_CONV_STATES)
    {
        // Only reachable when mUseMTP is true (getNbOutputs() == 3).
        DimsExprs out{};
        // [batch, seq_len, dim, kernel_size]
        out.nbDims = 4;
        out.d[0] = inputs[kIN_X_IDX].d[0];          // batch
        out.d[1] = inputs[kIN_X_IDX].d[1];          // seq_len (T)
        out.d[2] = inputs[kIN_CONV_STATE_IDX].d[1]; // dim
        out.d[3] = inputs[kIN_CONV_STATE_IDX].d[2]; // kernel_size
        return out;
    }
    // Output[1]: conv state output — same shape as conv_state input [batch, dim, kernel].
    return inputs[kIN_CONV_STATE_IDX];
}

bool CausalConv1dPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbOutputs = kNUM_REQUIRED_OUTPUTS + (mUseMTP ? kNUM_MTP_OPTIONAL_OUTPUTS : 0);
    if (nbInputs != kNUM_INPUTS || nbOutputs != expectedNbOutputs)
        return false;
    auto const& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
        return false;
    // INT32: context_lengths only
    if (pos == kIN_CONTEXT_LENGTHS_IDX)
        return desc.type == DataType::kINT32;
    // Everything else (all inputs + all outputs): FP16
    return desc.type == DataType::kHALF;
}

void CausalConv1dPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    if (nbInputs != kNUM_INPUTS)
    {
        LOG_ERROR("causal_conv1d: expected %d inputs, got %d", kNUM_INPUTS, nbInputs);
    }
    if (in[kIN_X_IDX].desc.type != DataType::kHALF)
    {
        LOG_ERROR(
            "causal_conv1d: only FP16 input is supported; got type %d", static_cast<int32_t>(in[kIN_X_IDX].desc.type));
    }
}

size_t CausalConv1dPlugin::getWorkspaceSize([[maybe_unused]] PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CausalConv1dPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{
    auto const& xDesc = inputDesc[kIN_X_IDX];
    auto const& wDesc = inputDesc[kIN_WEIGHT_IDX];
    auto const& outDesc = outputDesc[kOUT_IDX];

    if (xDesc.dims.nbDims != 3 || wDesc.dims.nbDims != 3 || outDesc.dims.nbDims != 3)
    {
        LOG_ERROR("causal_conv1d expects 3D tensors for x/weight/output.");
        return 1;
    }

    int32_t const batch = static_cast<int32_t>(xDesc.dims.d[0]);
    int32_t const seqLen = static_cast<int32_t>(xDesc.dims.d[1]);
    int32_t const dim = static_cast<int32_t>(xDesc.dims.d[2]);
    int32_t const width = static_cast<int32_t>(wDesc.dims.d[2]);

    int32_t const groups = mGroups == 0 ? dim : mGroups;
    if (groups != dim)
    {
        LOG_ERROR("causal_conv1d currently supports depthwise conv only: groups=%d, dim=%d", groups, dim);
        return 1;
    }

    void* convStateOut = outputs[kOUT_CONV_STATE_IDX];

    // MTP mode activates only for short multi-token verification sequences (tree verify),
    // not for normal prefill. Small prefills (seqLen <= kMTPMaxSeqLen) that happen to pass
    // through this path pay a minor cost for intermediate state writes, but it is harmless.
    // TODO: refactor the dispatch logic to explicitly distinguish MTP tree-verify decoding
    // from prefill when 1 < seqLen <= kMTPMaxSeqLen (e.g. pass an execution-phase flag
    // from the runtime instead of relying solely on seq_len range heuristics).
    constexpr int32_t kMTPMaxSeqLen = 8;
    bool const mtpActive = mUseMTP && (seqLen > 1) && (seqLen <= kMTPMaxSeqLen);

    namespace rt = trt_edgellm::rt;

    if (mtpActive)
    {
        // MTP DECODE path: process T draft tokens with per-step state checkpointing.
        // Copy input conv_state -> output conv_state (MTP kernel updates in-place).
        if (convStateOut != inputs[kIN_CONV_STATE_IDX])
        {
            size_t const stateBytes = static_cast<size_t>(batch) * dim * width * sizeof(half);
            cudaMemcpyAsync(convStateOut, inputs[kIN_CONV_STATE_IDX], stateBytes, cudaMemcpyDeviceToDevice, stream);
        }

        auto mtpStateTensor = rt::Tensor{convStateOut, rt::Coords{batch, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpNewColsTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpBiasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto mtpIntermTensor = rt::Tensor{outputs[kOUT_INTERMEDIATE_CONV_STATES], rt::Coords{batch, seqLen, dim, width},
            rt::DeviceType::kGPU, xDesc.type};

        trt_edgellm::rt::OptionalInputTensor mtpBiasOpt = std::optional(std::cref(mtpBiasTensor));
        mamba_ssm::invokeCausalConv1dDecodeMTP(mtpStateTensor, mtpNewColsTensor, mtpWeightTensor, mtpBiasOpt,
            mtpOutTensor, mtpIntermTensor, seqLen, stream);
    }
    else if (seqLen > 1)
    {
        // PREFILL path
        int32_t const outSeqLen = static_cast<int32_t>(outDesc.dims.d[1]);

        auto xTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto weightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto biasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto outTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, outSeqLen, dim}, rt::DeviceType::kGPU, xDesc.type};

        auto contextLengthsTensor = rt::Tensor{const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS_IDX]), rt::Coords{batch},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT32};
        trt_edgellm::rt::OptionalInputTensor contextLengthsOpt = std::optional(std::cref(contextLengthsTensor));

        trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasTensor));
        mamba_ssm::invokeCausalConv1d(
            xTensor, weightTensor, biasOpt, outTensor, mStride, mPadding, mDilation, contextLengthsOpt, stream);

        auto captureXTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, seqLen, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto captureStateTensor
            = rt::Tensor{convStateOut, rt::Coords{batch, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        mamba_ssm::invokeCaptureConvState(captureXTensor, captureStateTensor, contextLengthsOpt, stream);
    }
    else
    {
        // DECODE path (seqLen == 1): copy conv_state to output, then shift+insert and compute dot product.
        if (convStateOut != inputs[kIN_CONV_STATE_IDX])
        {
            size_t const stateBytes = static_cast<size_t>(batch) * dim * width * sizeof(half);
            cudaMemcpyAsync(convStateOut, inputs[kIN_CONV_STATE_IDX], stateBytes, cudaMemcpyDeviceToDevice, stream);
        }

        auto decodeStateTensor
            = rt::Tensor{convStateOut, rt::Coords{batch, dim, width}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeNewColTensor = rt::Tensor{
            const_cast<void*>(inputs[kIN_X_IDX]), rt::Coords{batch, 1, dim}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeWeightTensor = rt::Tensor{const_cast<void*>(inputs[kIN_WEIGHT_IDX]),
            rt::Coords{wDesc.dims.d[0], wDesc.dims.d[1], wDesc.dims.d[2]}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeBiasTensor
            = rt::Tensor{const_cast<void*>(inputs[kIN_BIAS_IDX]), rt::Coords{dim}, rt::DeviceType::kGPU, xDesc.type};
        auto decodeOutTensor
            = rt::Tensor{outputs[kOUT_IDX], rt::Coords{batch, 1, dim}, rt::DeviceType::kGPU, xDesc.type};
        trt_edgellm::rt::OptionalInputTensor decodeBiasOpt = std::optional(std::cref(decodeBiasTensor));
        mamba_ssm::invokeCausalConv1dDecode(
            decodeStateTensor, decodeNewColTensor, decodeWeightTensor, decodeBiasOpt, decodeOutTensor, stream);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

size_t CausalConv1dPlugin::getSerializationSize() const noexcept
{
    return 5 * sizeof(int32_t); // stride, padding, dilation, groups, useMTP
}

void CausalConv1dPlugin::serialize(void* buffer) const noexcept
{
    auto* d = static_cast<char*>(buffer);
    std::memcpy(d, &mStride, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(d, &mPadding, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(d, &mDilation, sizeof(int32_t));
    d += sizeof(int32_t);
    std::memcpy(d, &mGroups, sizeof(int32_t));
    d += sizeof(int32_t);
    int32_t useMTPInt = mUseMTP ? 1 : 0;
    std::memcpy(d, &useMTPInt, sizeof(int32_t));
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

char const* CausalConv1dPlugin::getPluginType() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_NAME;
}

char const* CausalConv1dPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CausalConv1dPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* CausalConv1dPlugin::getPluginVersion() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_VERSION;
}

int32_t CausalConv1dPlugin::initialize() noexcept
{
    return 0;
}

void CausalConv1dPlugin::terminate() noexcept {}

void CausalConv1dPlugin::destroy() noexcept
{
    delete this;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

CausalConv1dPluginCreator::CausalConv1dPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("groups", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_mtp", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* CausalConv1dPluginCreator::getPluginName() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_NAME;
}

PluginFieldCollection const* CausalConv1dPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void CausalConv1dPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* CausalConv1dPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* CausalConv1dPluginCreator::getPluginVersion() const noexcept
{
    return kCAUSAL_CONV_PLUGIN_VERSION;
}

IPluginV2* CausalConv1dPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        int32_t const stride = parsePluginIntField("stride", fc).value_or(1);
        int32_t const padding = parsePluginIntField("padding", fc).value_or(0);
        int32_t const dilation = parsePluginIntField("dilation", fc).value_or(1);
        int32_t const groups = parsePluginIntField("groups", fc).value_or(0);
        bool const useMTP = parsePluginIntField("use_mtp", fc).value_or(0) != 0;
        auto* plugin = new CausalConv1dPlugin(name, stride, padding, dilation, groups, useMTP);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create CausalConv1dPlugin: %s", e.what());
    }
    return nullptr;
}

IPluginV2* CausalConv1dPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* plugin = new CausalConv1dPlugin(name, serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to deserialize CausalConv1dPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
