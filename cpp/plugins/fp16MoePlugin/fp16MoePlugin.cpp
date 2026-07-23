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

#include "fp16MoePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "plugins/utils/pluginUtils.h"

#if defined(CUTE_DSL_F16_MOE_ENABLED)
#include "kernels/moe/f16_cutedsl/cuteDslF16MoeRunner.h"
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{
namespace
{

constexpr int32_t kIN_ROUTER_LOGITS{0};
constexpr int32_t kIN_HIDDEN_STATES{1};
constexpr int32_t kIN_FC1_WEIGHTS{2};
constexpr int32_t kIN_FC2_WEIGHTS{3};
constexpr int32_t kOUT_OUTPUT{4};
constexpr int32_t kNB_INPUTS{4};
constexpr int32_t kNB_OUTPUTS{1};

constexpr char const* kPLUGIN_NAME{"Fp16MoePlugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr int32_t kACT_SWIGLU{2};
constexpr int32_t kACT_RELU2{4};
constexpr int32_t kMAX_TOP_K{8};
constexpr int32_t kROW_ALIGNMENT{128};
constexpr int32_t kINTER_ALIGNMENT{64};

int32_t getRequiredIntField(PluginFieldCollection const* fields, char const* name)
{
    if (fields == nullptr)
    {
        throw std::invalid_argument("Fp16MoePlugin: null PluginFieldCollection");
    }

    bool found{false};
    int32_t value{};
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (field.name != nullptr && std::string(field.name) == name)
        {
            if (found)
            {
                throw std::invalid_argument(std::string("Fp16MoePlugin: duplicate field ") + name);
            }
            if (field.type != PluginFieldType::kINT32 || field.length != 1 || field.data == nullptr)
            {
                throw std::invalid_argument(
                    std::string("Fp16MoePlugin: field ") + name + " must be one non-null INT32 scalar");
            }
            value = *static_cast<int32_t const*>(field.data);
            found = true;
        }
    }
    if (!found)
    {
        throw std::invalid_argument(std::string("Fp16MoePlugin: missing required field ") + name);
    }
    return value;
}

void validateAttributes(int32_t numExperts, int32_t topK, int32_t hiddenSize, int32_t moeInterSize,
    int32_t activationType, int32_t normTopkProb, int32_t maxRoutedRows)
{
    // Match CuteDslF16MoeRunner::kSupportedNumExperts.
    if (numExperts != 128 && numExperts != 256)
    {
        throw std::invalid_argument("Fp16MoePlugin: num_experts must be one of {128, 256}");
    }
    if (topK <= 0 || topK > kMAX_TOP_K)
    {
        throw std::invalid_argument("Fp16MoePlugin: top_k must be in [1, 8]");
    }
    if (hiddenSize <= 0 || hiddenSize % kROW_ALIGNMENT != 0)
    {
        throw std::invalid_argument("Fp16MoePlugin: hidden_size must be a positive multiple of 128");
    }
    if (moeInterSize <= 0 || moeInterSize % kINTER_ALIGNMENT != 0)
    {
        throw std::invalid_argument("Fp16MoePlugin: moe_inter_size must be a positive multiple of 64");
    }
    int64_t const fc1N = activationType == kACT_SWIGLU ? 2LL * moeInterSize : moeInterSize;
    if ((activationType != kACT_SWIGLU && activationType != kACT_RELU2) || fc1N % kROW_ALIGNMENT != 0
        || fc1N > std::numeric_limits<int32_t>::max())
    {
        throw std::invalid_argument(
            "Fp16MoePlugin: activation_type must be 2 (SwiGLU) or 4 (ReLU2), and FC1_N must be an "
            "INT32-representable multiple of 128");
    }
    if (normTopkProb != 0 && normTopkProb != 1)
    {
        throw std::invalid_argument("Fp16MoePlugin: norm_topk_prob must be 0 or 1");
    }
    if (maxRoutedRows < 0)
    {
        throw std::invalid_argument("Fp16MoePlugin: max_routed_rows must be nonnegative");
    }
}

bool validateTokenShape(Dims const& router, Dims const& hidden, int32_t numExperts, char const* profilePoint)
{
    if (router.nbDims != 2 || hidden.nbDims != 3 || router.d[0] <= 0 || router.d[1] != numExperts || hidden.d[0] <= 0
        || hidden.d[1] <= 0 || hidden.d[2] <= 0)
    {
        LOG_ERROR("Fp16MoePlugin: invalid %s profile dimensions", profilePoint);
        return false;
    }
    int64_t const hiddenTokens = static_cast<int64_t>(hidden.d[0]) * hidden.d[1];
    if (hiddenTokens != router.d[0])
    {
        LOG_ERROR("Fp16MoePlugin: %s router_logits d[0]=%d must equal hidden_states B*S=%d*%d=%lld", profilePoint,
            router.d[0], hidden.d[0], hidden.d[1], static_cast<long long>(hiddenTokens));
        return false;
    }
    return true;
}

bool hasExactDims3(Dims const& dims, int32_t d0, int32_t d1, int32_t d2) noexcept
{
    return dims.nbDims == 3 && dims.d[0] == d0 && dims.d[1] == d1 && dims.d[2] == d2;
}

int32_t profileRoutedRows(int64_t maxTokens, int32_t topK)
{
    if (maxTokens <= 0 || maxTokens > std::numeric_limits<int32_t>::max() / topK)
    {
        throw std::overflow_error("Fp16MoePlugin: optimization profile num_tokens*top_k exceeds INT32_MAX");
    }
    return static_cast<int32_t>(maxTokens * topK);
}

} // namespace

PluginFieldCollection Fp16MoePluginCreator::mFieldCollection{};
std::vector<PluginField> Fp16MoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Fp16MoePluginCreator);

Fp16MoePlugin::Fp16MoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize,
    int32_t moeInterSize, int32_t activationType, int32_t normTopkProb, int32_t maxRoutedRows)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mNormTopkProb(normTopkProb)
    , mMaxRoutedRows(maxRoutedRows)
    , mAutoMaxRoutedRows(maxRoutedRows == 0)
{
    validateAttributes(mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType, mNormTopkProb, mMaxRoutedRows);
}

Fp16MoePlugin::Fp16MoePlugin(std::string const& name, PluginFieldCollection const* fields)
    : Fp16MoePlugin(name, getRequiredIntField(fields, "num_experts"), getRequiredIntField(fields, "top_k"),
          getRequiredIntField(fields, "hidden_size"), getRequiredIntField(fields, "moe_inter_size"),
          getRequiredIntField(fields, "activation_type"), getRequiredIntField(fields, "norm_topk_prob"),
          getRequiredIntField(fields, "max_routed_rows"))
{
}

IPluginCapability* Fp16MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Fp16MoePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Fp16MoePlugin(
            mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType, mNormTopkProb, mMaxRoutedRows);
        plugin->mAutoMaxRoutedRows = mAutoMaxRoutedRows;
        plugin->mPersistentBlockCount = mPersistentBlockCount;
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePlugin clone failed: %s", error.what());
        return nullptr;
    }
}

char const* Fp16MoePlugin::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* Fp16MoePlugin::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* Fp16MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Fp16MoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePlugin namespace update failed: %s", error.what());
    }
}

int32_t Fp16MoePlugin::getNbOutputs() const noexcept
{
    return kNB_OUTPUTS;
}

int32_t Fp16MoePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        LOG_ERROR("Fp16MoePlugin: getOutputDataTypes expected four inputs and one output");
        return -1;
    }
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Fp16MoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    (void) shapeInputs;
    (void) nbShapeInputs;
    if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        LOG_ERROR("Fp16MoePlugin: getOutputShapes expected four inputs and one output");
        return -1;
    }
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[kIN_HIDDEN_STATES].d[0];
    outputs[0].d[1] = inputs[kIN_HIDDEN_STATES].d[1];
    outputs[0].d[2] = exprBuilder.constant(mHiddenSize);
    return 0;
}

bool Fp16MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }
    PluginTensorDesc const& tensor = inOut[pos].desc;
    if (tensor.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    int32_t const fc1N = static_cast<int32_t>(mActivationType == kACT_SWIGLU ? 2LL * mMoeInterSize : mMoeInterSize);
    switch (pos)
    {
    case kIN_ROUTER_LOGITS:
        return tensor.type == DataType::kFLOAT && tensor.dims.nbDims == 2 && tensor.dims.d[1] == mNumExperts;
    case kIN_HIDDEN_STATES:
        return tensor.type == DataType::kHALF && tensor.dims.nbDims == 3 && tensor.dims.d[2] == mHiddenSize;
    case kIN_FC1_WEIGHTS:
        return tensor.type == DataType::kHALF && hasExactDims3(tensor.dims, mNumExperts, fc1N, mHiddenSize);
    case kIN_FC2_WEIGHTS:
        return tensor.type == DataType::kHALF && hasExactDims3(tensor.dims, mNumExperts, mHiddenSize, mMoeInterSize);
    case kOUT_OUTPUT:
        return tensor.type == DataType::kHALF && tensor.dims.nbDims == 3 && tensor.dims.d[2] == mHiddenSize;
    default: return false;
    }
}

int32_t Fp16MoePlugin::configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    try
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
        {
            LOG_ERROR("Fp16MoePlugin: configurePlugin expected four inputs and one output");
            return -1;
        }
        if (!validateTokenShape(inputs[kIN_ROUTER_LOGITS].min, inputs[kIN_HIDDEN_STATES].min, mNumExperts, "minimum")
            || !validateTokenShape(inputs[kIN_ROUTER_LOGITS].opt, inputs[kIN_HIDDEN_STATES].opt, mNumExperts, "optimum")
            || !validateTokenShape(
                inputs[kIN_ROUTER_LOGITS].max, inputs[kIN_HIDDEN_STATES].max, mNumExperts, "maximum"))
        {
            return -1;
        }
        if (inputs[kIN_HIDDEN_STATES].min.d[2] != mHiddenSize || inputs[kIN_HIDDEN_STATES].opt.d[2] != mHiddenSize
            || inputs[kIN_HIDDEN_STATES].max.d[2] != mHiddenSize)
        {
            LOG_ERROR("Fp16MoePlugin: hidden_states profile hidden dimension must equal hidden_size=%d", mHiddenSize);
            return -1;
        }

        int32_t const fc1N = static_cast<int32_t>(mActivationType == kACT_SWIGLU ? 2LL * mMoeInterSize : mMoeInterSize);
        if (!hasExactDims3(inputs[kIN_FC1_WEIGHTS].desc.dims, mNumExperts, fc1N, mHiddenSize)
            || !hasExactDims3(inputs[kIN_FC2_WEIGHTS].desc.dims, mNumExperts, mHiddenSize, mMoeInterSize))
        {
            LOG_ERROR("Fp16MoePlugin: weight dimensions do not match the plugin attributes");
            return -1;
        }

#if defined(CUTE_DSL_F16_MOE_ENABLED)
        int32_t const smVersion = getSMVersion();
        if (!CuteDslF16MoeRunner::canImplement(
                mHiddenSize, mMoeInterSize, mNumExperts, mTopK, smVersion, mActivationType))
        {
            LOG_ERROR(
                "Fp16MoePlugin: unsupported configuration H=%d I=%d E=%d top_k=%d activation=%d SM=%d. "
                "The v1 backend requires a matching f16_moe artifact for Ampere SM80/86/87/89, "
                "Blackwell SM100/101/103/110, or Blackwell GeForce SM120/121; E in {128, 256}; top_k in [1,8]; "
                "H%%128=0; I%%64=0; FC1_N%%128=0; and activation 2 (SwiGLU) or 4 (ReLU2).",
                mHiddenSize, mMoeInterSize, mNumExperts, mTopK, mActivationType, smVersion);
            return -1;
        }
#else
        LOG_ERROR(
            "Fp16MoePlugin: f16_moe CuTeDSL artifacts are not linked. Generate them with "
            "kernelSrcs/build_cutedsl.py --kernels f16_moe --gpu_arch <sm_NN> --arch <arch>, then rebuild "
            "with -DENABLE_CUTE_DSL=f16_moe -DCUTE_DSL_ARTIFACT_TAG=<sm_NN>.");
        return -1;
#endif

        int64_t const profileMaxTokens = inputs[kIN_ROUTER_LOGITS].max.d[0];
        int32_t const requiredRoutedRows = profileRoutedRows(profileMaxTokens, mTopK);
        if (mAutoMaxRoutedRows)
        {
            mMaxRoutedRows = std::max(mMaxRoutedRows, requiredRoutedRows);
        }
        else if (mMaxRoutedRows < requiredRoutedRows)
        {
            LOG_ERROR("Fp16MoePlugin: max_routed_rows=%d is smaller than the optimization-profile maximum %lld*%d=%d",
                mMaxRoutedRows, static_cast<long long>(profileMaxTokens), mTopK, requiredRoutedRows);
            return -1;
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePlugin configurePlugin failed: %s", error.what());
        return -1;
    }
    catch (...)
    {
        LOG_ERROR("Fp16MoePlugin configurePlugin failed: unknown error");
        return -1;
    }
}

size_t Fp16MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void) outputs;
    if (inputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        LOG_ERROR("Fp16MoePlugin: getWorkspaceSize expected four inputs and one output");
        return 0;
    }
#if !defined(CUTE_DSL_F16_MOE_ENABLED)
    return 0;
#else
    try
    {
        int64_t const maxTokens64 = inputs[kIN_ROUTER_LOGITS].max.d[0];
        if (maxTokens64 <= 0 || maxTokens64 > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("Fp16MoePlugin: invalid profile maximum token count");
            return 0;
        }
        int32_t const maxTokens = static_cast<int32_t>(maxTokens64);
        int32_t const maxRoutedRows = mMaxRoutedRows > 0 ? mMaxRoutedRows : profileRoutedRows(maxTokens64, mTopK);

        // Reserve aligned slices in the same order consumed by enqueue(): softmax scratch, top-K weights,
        // top-K expert IDs, then the CuTeDSL runner workspace.
        size_t total{};
        size_t const softmaxBytes = kernel::getMoeTopkSoftmaxWorkspaceSize(maxTokens, mNumExperts);
        total = accumulateWorkspaceSize(
            total, {static_cast<int64_t>(std::max<size_t>(softmaxBytes, 1))}, DataType::kINT8);
        total = accumulateWorkspaceSize(total, {maxTokens, mTopK}, DataType::kFLOAT);
        total = accumulateWorkspaceSize(total, {maxTokens, mTopK}, DataType::kINT32);
        size_t const runnerBytes = CuteDslF16MoeRunner::getWorkspaceSize(
            maxRoutedRows, mNumExperts, mHiddenSize, mMoeInterSize, mActivationType);
        if (runnerBytes == 0)
        {
            LOG_ERROR("Fp16MoePlugin: runner returned zero workspace size");
            return 0;
        }
        total = accumulateWorkspaceSize(total, {static_cast<int64_t>(runnerBytes)}, DataType::kINT8);
        return total;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePlugin workspace calculation failed: %s", error.what());
        return 0;
    }
#endif
}

int32_t Fp16MoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    (void) outputDesc;
#if !defined(CUTE_DSL_F16_MOE_ENABLED)
    (void) inputDesc;
    (void) inputs;
    (void) outputs;
    (void) workspace;
    (void) stream;
    LOG_ERROR("Fp16MoePlugin: f16_moe CuTeDSL artifacts are not linked; rebuild with -DENABLE_CUTE_DSL=f16_moe");
    return -1;
#else
    try
    {
        if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr || workspace == nullptr
            || inputs[kIN_ROUTER_LOGITS] == nullptr || inputs[kIN_HIDDEN_STATES] == nullptr
            || inputs[kIN_FC1_WEIGHTS] == nullptr || inputs[kIN_FC2_WEIGHTS] == nullptr || outputs[0] == nullptr)
        {
            LOG_ERROR("Fp16MoePlugin: null descriptor, tensor, output, or workspace");
            return -1;
        }
        if (mPersistentBlockCount <= 0)
        {
            LOG_ERROR("Fp16MoePlugin: runtime context has no cached f16_moe launch configuration");
            return -1;
        }

        Dims const& hiddenDims = inputDesc[kIN_HIDDEN_STATES].dims;
        Dims const& routerDims = inputDesc[kIN_ROUTER_LOGITS].dims;
        if (hiddenDims.nbDims != 3 || routerDims.nbDims != 2 || hiddenDims.d[0] <= 0 || hiddenDims.d[1] <= 0)
        {
            LOG_ERROR("Fp16MoePlugin: invalid runtime hidden/router dimensions");
            return -1;
        }
        int64_t const numTokens64 = static_cast<int64_t>(hiddenDims.d[0]) * hiddenDims.d[1];
        if (numTokens64 != routerDims.d[0] || numTokens64 > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR("Fp16MoePlugin: runtime router token count must equal hidden_states B*S and fit INT32");
            return -1;
        }
        int64_t const routedRows64 = numTokens64 * mTopK;
        if (mMaxRoutedRows <= 0 || routedRows64 > mMaxRoutedRows)
        {
            LOG_ERROR("Fp16MoePlugin: runtime num_tokens*top_k=%lld exceeds resolved max_routed_rows=%d",
                static_cast<long long>(routedRows64), mMaxRoutedRows);
            return -1;
        }
        int32_t const numTokens = static_cast<int32_t>(numTokens64);

        size_t const softmaxBytes = kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);
        std::byte* next = static_cast<std::byte*>(workspace);
        void* const softmaxWorkspace = assignTensorFromWorkspace(
            next, {static_cast<int64_t>(std::max<size_t>(softmaxBytes, 1))}, DataType::kINT8)
                                           .rawPointer();
        float* const topkWeights
            = static_cast<float*>(assignTensorFromWorkspace(next, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
        int32_t* const topkIds
            = static_cast<int32_t*>(assignTensorFromWorkspace(next, {numTokens, mTopK}, DataType::kINT32).rawPointer());
        size_t const runnerBytes = CuteDslF16MoeRunner::getWorkspaceSize(
            mMaxRoutedRows, mNumExperts, mHiddenSize, mMoeInterSize, mActivationType);
        if (runnerBytes == 0)
        {
            LOG_ERROR("Fp16MoePlugin: runner returned zero workspace size at enqueue");
            return -1;
        }
        void* const runnerWorkspace
            = assignTensorFromWorkspace(next, {static_cast<int64_t>(runnerBytes)}, DataType::kINT8).rawPointer();

        rt::Tensor routerTensor(const_cast<void*>(inputs[kIN_ROUTER_LOGITS]), rt::Coords{routerDims},
            rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor topkWeightsTensor(topkWeights, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor topkIdsTensor(topkIds, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        kernel::moeTopkSoftmax(routerTensor, topkWeightsTensor, topkIdsTensor, mTopK,
            softmaxBytes == 0 ? nullptr : softmaxWorkspace, softmaxBytes, stream, mNormTopkProb != 0);
        CUDA_CHECK(cudaGetLastError());

        CuteDslF16MoeParams parameters{};
        parameters.numTokens = numTokens;
        parameters.numExperts = mNumExperts;
        parameters.topK = mTopK;
        parameters.hiddenSize = mHiddenSize;
        parameters.moeInterSize = mMoeInterSize;
        parameters.activationType = mActivationType;
        parameters.persistentBlockCount = mPersistentBlockCount;
        parameters.hiddenStates = inputs[kIN_HIDDEN_STATES];
        parameters.topkIds = topkIds;
        parameters.topkWeights = topkWeights;
        parameters.fc1Weights = inputs[kIN_FC1_WEIGHTS];
        parameters.fc2Weights = inputs[kIN_FC2_WEIGHTS];
        parameters.output = outputs[0];

        int32_t const result = CuteDslF16MoeRunner::run(parameters, runnerWorkspace, stream);
        if (result != 0)
        {
            LOG_ERROR("Fp16MoePlugin: runner failed with code %d", result);
        }
        return result;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePlugin enqueue failed: %s", error.what());
        return -1;
    }
    catch (...)
    {
        LOG_ERROR("Fp16MoePlugin enqueue failed: unknown error");
        return -1;
    }
#endif
}

int32_t Fp16MoePlugin::onShapeChange(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    (void) outputs;
    if (inputs == nullptr || nbInputs != kNB_INPUTS || nbOutputs != kNB_OUTPUTS)
    {
        LOG_ERROR("Fp16MoePlugin: onShapeChange expected four inputs and one output");
        return -1;
    }
    Dims const& hidden = inputs[kIN_HIDDEN_STATES].dims;
    Dims const& router = inputs[kIN_ROUTER_LOGITS].dims;
    if (hidden.nbDims != 3 || router.nbDims != 2 || hidden.d[0] <= 0 || hidden.d[1] <= 0)
    {
        return -1;
    }
    int64_t const tokens = static_cast<int64_t>(hidden.d[0]) * hidden.d[1];
    int64_t const routedRows = tokens * mTopK;
    if (router.d[0] != tokens || mMaxRoutedRows <= 0 || routedRows > mMaxRoutedRows)
    {
        LOG_ERROR("Fp16MoePlugin: runtime shape requires %lld routed rows, resolved cap is %d",
            static_cast<long long>(routedRows), mMaxRoutedRows);
        return -1;
    }
    return 0;
}

IPluginV3* Fp16MoePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    (void) context;
    auto* plugin = static_cast<Fp16MoePlugin*>(clone());
    if (plugin == nullptr)
    {
        return nullptr;
    }
#if defined(CUTE_DSL_F16_MOE_ENABLED)
    plugin->mPersistentBlockCount = CuteDslF16MoeRunner::getPersistentBlockCount();
    if (plugin->mPersistentBlockCount <= 0)
    {
        LOG_ERROR("Fp16MoePlugin: attachToContext failed to cache the f16_moe launch configuration");
        delete plugin;
        return nullptr;
    }
#else
    LOG_ERROR("Fp16MoePlugin: attachToContext requires a build with -DENABLE_CUTE_DSL=f16_moe");
    delete plugin;
    return nullptr;
#endif
    return plugin;
}

PluginFieldCollection const* Fp16MoePlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("top_k", &mTopK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("activation_type", &mActivationType, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("norm_topk_prob", &mNormTopkProb, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("max_routed_rows", &mMaxRoutedRows, PluginFieldType::kINT32, 1);
        mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFieldsToSerialize.fields = mDataToSerialize.data();
        return &mFieldsToSerialize;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePlugin serialization failed: %s", error.what());
        return nullptr;
    }
}

Fp16MoePluginCreator::Fp16MoePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("num_experts", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("top_k", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("hidden_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("moe_inter_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("activation_type", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("norm_topk_prob", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("max_routed_rows", nullptr, PluginFieldType::kINT32, 1);
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Fp16MoePluginCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* Fp16MoePluginCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* Fp16MoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* Fp16MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Fp16MoePluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePluginCreator namespace update failed: %s", error.what());
    }
}

IPluginV3* Fp16MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        if (name == nullptr)
        {
            throw std::invalid_argument("Fp16MoePlugin: null layer name");
        }
        auto* plugin = new Fp16MoePlugin(name, fields);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("Fp16MoePlugin creation failed: %s", error.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
