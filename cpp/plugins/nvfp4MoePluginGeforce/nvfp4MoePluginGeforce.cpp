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

#include "nvfp4MoePluginGeforce.h"

#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/tensor.h"
#include "kernels/moe/moePerExpertScaleKernels.h"
#include "kernels/moe/moeSigmoidGroupTopkKernels.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "plugins/utils/pluginUtils.h"

#include <NvInfer.h>

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
#include "kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeRunner.h"
#endif

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

// Input indices (11 total) and output index — see the plugin doc in the header for a full
// description of each tensor's shape and dtype.
namespace
{
constexpr int32_t kIN_ROUTER_LOGITS{0};
constexpr int32_t kIN_HIDDEN_STATES{1};
constexpr int32_t kIN_FC1_QWEIGHTS{2};
constexpr int32_t kIN_FC1_BLOCKS_SCALE{3};
constexpr int32_t kIN_FC1_ALPHA{4};
constexpr int32_t kIN_FC2_QWEIGHTS{5};
constexpr int32_t kIN_FC2_BLOCKS_SCALE{6};
constexpr int32_t kIN_FC2_ALPHA{7};
constexpr int32_t kIN_INPUT_GLOBAL_SCALE{8};
constexpr int32_t kIN_DOWN_INPUT_SCALE{9};
constexpr int32_t kIN_E_SCORE_CORRECTION_BIAS{10};
constexpr int32_t kOUT_OUTPUT{11};
constexpr int32_t kNbPluginInputs{11};

constexpr char const* kPLUGIN_VERSION{"1"};
constexpr char const* kPLUGIN_NAME{"NvFP4MoEPluginGeforce"};

//! Matches the activation_type attribute encoding (see the header).
constexpr int32_t kACT_IDENTITY{0};
constexpr int32_t kACT_SILU{1};
constexpr int32_t kACT_SWIGLU{2};
constexpr int32_t kACT_GELU{3};
constexpr int32_t kACT_RELU2{4};
constexpr int32_t kACT_GEGLU{5};

constexpr int32_t kBACKEND_AUTO{0};
constexpr int32_t kBACKEND_DECODE{1};
constexpr int32_t kBACKEND_PREFILL{2};

constexpr int32_t kIODT_BF16{0};
constexpr int32_t kIODT_FP16{1};

constexpr int32_t kROUTING_MODE_SOFTMAX_TOPK{0};
constexpr int32_t kROUTING_MODE_SIGMOID_GROUP_TOPK{1};
constexpr int32_t kROUTING_MODE_SOFTMAX_TOPK_POST_SCALE{2};

//! NVFP4 scale-factor group size (kK dim grouping) — block scales have K/sf_vec_size entries.
constexpr int32_t kNvfp4SfVecSize{16};

//! Keep shape inference compilable when the AOT runner is not linked.
#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
constexpr int32_t kCuteDslLevelTileN{CuteDslNvfp4MoeRunner::kLevelTileN};
#else
constexpr int32_t kCuteDslLevelTileN{128};
#endif

inline int32_t ceilDivInt(int32_t a, int32_t b)
{
    return (a + b - 1) / b;
}

inline bool isGatedActivation(int32_t activationType)
{
    return activationType == kACT_SWIGLU || activationType == kACT_GEGLU;
}

//! Number of output rows in FC1 given the gated/non-gated activation.
inline int32_t fc1OutDim(int32_t moeInterSize, int32_t activationType)
{
    return isGatedActivation(activationType) ? 2 * moeInterSize : moeInterSize;
}

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
CuteDslMoeActivation toRunnerActivation(int32_t activationType)
{
    switch (activationType)
    {
    case kACT_IDENTITY: return CuteDslMoeActivation::kIdentity;
    case kACT_SILU: return CuteDslMoeActivation::kSiLU;
    case kACT_SWIGLU: return CuteDslMoeActivation::kSwiGLU;
    case kACT_GELU: return CuteDslMoeActivation::kGeLU;
    case kACT_RELU2: return CuteDslMoeActivation::kReLU2;
    case kACT_GEGLU: return CuteDslMoeActivation::kGeGLU;
    default: return CuteDslMoeActivation::kSwiGLU;
    }
}

CuteDslMoeBackend toRunnerBackend(int32_t backend)
{
    switch (backend)
    {
    case kBACKEND_DECODE: return CuteDslMoeBackend::kDecode;
    case kBACKEND_PREFILL: return CuteDslMoeBackend::kPrefill;
    case kBACKEND_AUTO: [[fallthrough]];
    default: return CuteDslMoeBackend::kAuto;
    }
}

CuteDslMoeIoDtype toRunnerIoDtype(int32_t ioDtype)
{
    return ioDtype == kIODT_BF16 ? CuteDslMoeIoDtype::kBF16 : CuteDslMoeIoDtype::kFP16;
}

//! Compute ``maxNumTokens * topK`` saturated to a positive int32 (1 .. INT32_MAX).
//! Used both for inferring the auto-resolved cap and for validating user-supplied
//! \c max_routed_rows against the optimization profile.
int32_t computeProfileMaxRoutedRows(int64_t maxNumTokens, int32_t topK)
{
    int64_t const routed = maxNumTokens * static_cast<int64_t>(topK);
    int64_t const capped = std::min(routed, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    return static_cast<int32_t>(std::max<int64_t>(1, capped));
}
#endif // CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
} // namespace

PluginFieldCollection NvFP4MoEPluginGeforceCreator::mFieldCollection{};
std::vector<PluginField> NvFP4MoEPluginGeforceCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(NvFP4MoEPluginGeforceCreator);

// Constructors / boilerplate

NvFP4MoEPluginGeforce::NvFP4MoEPluginGeforce(std::string const& name, int32_t numExperts, int32_t topK,
    int32_t hiddenSize, int32_t moeInterSize, int32_t activationType, int32_t nGroup, int32_t topkGroup,
    int32_t normTopkProb, float routedScalingFactor, int32_t routingMode, int32_t backend, int32_t maxRoutedRows,
    int32_t ioDtype)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mRoutingMode(routingMode)
    , mNGroup(nGroup)
    , mTopkGroup(topkGroup)
    , mNormTopkProb(normTopkProb)
    , mRoutedScalingFactor(routedScalingFactor)
    , mBackend(backend)
    , mMaxRoutedRows(maxRoutedRows)
    , mIoDtype(ioDtype)
{
}

NvFP4MoEPluginGeforce::NvFP4MoEPluginGeforce(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName == "num_experts")
        {
            mNumExperts = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "top_k")
        {
            mTopK = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "hidden_size")
        {
            mHiddenSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "moe_inter_size")
        {
            mMoeInterSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "activation_type")
        {
            mActivationType = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "n_group")
        {
            mNGroup = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "topk_group")
        {
            mTopkGroup = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "norm_topk_prob")
        {
            mNormTopkProb = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "routed_scaling_factor")
        {
            mRoutedScalingFactor = *static_cast<float const*>(fc->fields[i].data);
        }
        else if (fieldName == "routing_mode")
        {
            mRoutingMode = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "backend")
        {
            mBackend = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "max_routed_rows")
        {
            mMaxRoutedRows = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "io_dtype")
        {
            mIoDtype = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }
    if (mNumExperts <= 0 || mTopK <= 0 || mHiddenSize <= 0 || mMoeInterSize <= 0)
    {
        throw std::invalid_argument(
            "NvFP4MoEPluginGeforce: num_experts, top_k, hidden_size, "
            "and moe_inter_size must all be > 0");
    }
    // v1 gating — reject values outside the supported subset early so user sees a clear error
    // instead of a silent wrong-answer or a deep-in-the-stack assertion.
    if (mIoDtype != kIODT_FP16)
    {
        throw std::invalid_argument("NvFP4MoEPluginGeforce: v1 only supports io_dtype=1 (FP16)");
    }
    switch (mActivationType)
    {
    case kACT_IDENTITY:
    case kACT_SILU:
    case kACT_SWIGLU:
    case kACT_GELU:
    case kACT_RELU2:
    case kACT_GEGLU: break;
    default:
        throw std::invalid_argument(
            "NvFP4MoEPluginGeforce: activation_type must be 0 (identity), 1 (silu), "
            "2 (swiglu), 3 (gelu), 4 (relu2), or 5 (geglu)");
    }
    if (mBackend != kBACKEND_AUTO && mBackend != kBACKEND_DECODE && mBackend != kBACKEND_PREFILL)
    {
        throw std::invalid_argument("NvFP4MoEPluginGeforce: backend must be 0 (auto), 1 (decode), or 2 (prefill)");
    }
    if (mRoutingMode != kROUTING_MODE_SOFTMAX_TOPK && mRoutingMode != kROUTING_MODE_SIGMOID_GROUP_TOPK
        && mRoutingMode != kROUTING_MODE_SOFTMAX_TOPK_POST_SCALE)
    {
        throw std::invalid_argument(
            "NvFP4MoEPluginGeforce: routing_mode must be 0 (softmax top-k), 1 (sigmoid group top-k), "
            "or 2 (softmax top-k post-scale)");
    }
    if (mRoutingMode == kROUTING_MODE_SIGMOID_GROUP_TOPK)
    {
        if (mNGroup <= 0 || mNumExperts % mNGroup != 0)
        {
            throw std::invalid_argument(
                "NvFP4MoEPluginGeforce: n_group must be positive and evenly divide num_experts");
        }
        if (mTopkGroup <= 0 || mTopkGroup > mNGroup)
        {
            throw std::invalid_argument("NvFP4MoEPluginGeforce: topk_group must be in [1, n_group]");
        }
    }
}

NvFP4MoEPluginGeforce::~NvFP4MoEPluginGeforce() noexcept
{
    // The identity expert table is owned only by the runtime instance returned by
    // attachToContext (which captures \c mGpuAllocator). Build-phase / clone-source
    // / failed-deserialize plugins keep \c mGpuAllocator == nullptr and the dtor
    // is a no-op for them.
    if (mGpuAllocator != nullptr && mIdentityExpertTable != nullptr)
    {
        mGpuAllocator->deallocate(mIdentityExpertTable);
    }
    mIdentityExpertTable = nullptr;
    mGpuAllocator = nullptr;
}

IPluginCapability* NvFP4MoEPluginGeforce::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* NvFP4MoEPluginGeforce::clone() noexcept
{
    try
    {
        // Intentionally do NOT propagate mGpuAllocator / mIdentityExpertTable. Only
        // the runtime instance returned by attachToContext owns the device buffer;
        // copying the pointer here would create a double-free. The clone gets a
        // fresh allocation in attachToContext if it is later attached.
        auto* p = new NvFP4MoEPluginGeforce(mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType,
            mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mRoutingMode, mBackend, mMaxRoutedRows, mIoDtype);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone NvFP4MoEPluginGeforce: %s", e.what());
        return nullptr;
    }
}

char const* NvFP4MoEPluginGeforce::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* NvFP4MoEPluginGeforce::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

char const* NvFP4MoEPluginGeforce::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void NvFP4MoEPluginGeforce::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

int32_t NvFP4MoEPluginGeforce::getNbOutputs() const noexcept
{
    return 1;
}

int32_t NvFP4MoEPluginGeforce::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbOutputs != 1)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce: getOutputDataTypes expected 1 output, got %d", nbOutputs);
        return -1;
    }
    (void) nbInputs;
    (void) inputTypes;
    outputTypes[0] = DataType::kHALF; // v1 always FP16 out.
    return 0;
}

int32_t NvFP4MoEPluginGeforce::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    if (nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR(
            "NvFP4MoEPluginGeforce: getOutputShapes expected %d inputs and 1 output, got %d inputs and %d outputs",
            kNbPluginInputs, nbInputs, nbOutputs);
        return -1;
    }
    (void) shapeInputs;
    (void) nbShapeInputs;
    // Output shares B and S with hidden_states; final dim is the plugin-configured hidden_size.
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[kIN_HIDDEN_STATES].d[0];
    outputs[0].d[1] = inputs[kIN_HIDDEN_STATES].d[1];
    outputs[0].d[2] = exprBuilder.constant(static_cast<int64_t>(mHiddenSize));
    return 0;
}

// Shape / format validation

bool NvFP4MoEPluginGeforce::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != kNbPluginInputs || nbOutputs != 1)
    {
        LOG_ERROR(
            "NvFP4MoEPluginGeforce: supportsFormatCombination expected %d inputs and 1 output, got %d inputs and %d "
            "outputs",
            kNbPluginInputs, nbInputs, nbOutputs);
        return false;
    }

    auto const& td = inOut[pos].desc;

    // All inputs/outputs must be LINEAR — no reformat.
    if (td.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    int32_t const n1 = fc1OutDim(mMoeInterSize, mActivationType);

#define SFC_REJ(reason)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        LOG_DEBUG("NvFP4MoEPluginGeforce supportsFormatCombination rejected pos %d: %s", pos, reason);                 \
        return false;                                                                                                  \
    } while (0)

    switch (pos)
    {
    case kIN_ROUTER_LOGITS:
    {
        if (td.type != DataType::kFLOAT)
            SFC_REJ("ROUTER_LOGITS: type != kFLOAT");
        if (td.dims.nbDims != 2)
            SFC_REJ("ROUTER_LOGITS: nbDims != 2");
        if (td.dims.d[1] != mNumExperts)
            SFC_REJ("ROUTER_LOGITS: d[1] != mNumExperts");
        return true;
    }
    case kIN_HIDDEN_STATES:
    {
        if (td.type != DataType::kHALF)
            SFC_REJ("HIDDEN_STATES: type != kHALF");
        if (td.dims.nbDims != 3)
            SFC_REJ("HIDDEN_STATES: nbDims != 3");
        if (td.dims.d[2] != mHiddenSize)
            SFC_REJ("HIDDEN_STATES: d[2] != mHiddenSize");
        return true;
    }
    case kIN_FC1_QWEIGHTS:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC1_QWEIGHTS: type != kINT8");
        if (td.dims.nbDims != 3)
            SFC_REJ("FC1_QWEIGHTS: nbDims != 3");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC1_QWEIGHTS: d[0] != mNumExperts");
        if (td.dims.d[1] != n1)
            SFC_REJ("FC1_QWEIGHTS: d[1] != n1");
        if (td.dims.d[2] != mHiddenSize / 2)
            SFC_REJ("FC1_QWEIGHTS: d[2] != mHiddenSize/2");
        return true;
    }
    case kIN_FC1_BLOCKS_SCALE:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC1_BLOCKS_SCALE: type != kINT8");
        if (td.dims.nbDims != 6)
            SFC_REJ("FC1_BLOCKS_SCALE: nbDims != 6");
        int32_t const mTiles = ceilDivInt(n1, 128);
        int32_t const kTiles = ceilDivInt(mHiddenSize / kNvfp4SfVecSize, 4);
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC1_BLOCKS_SCALE: d[0] != mNumExperts");
        if (td.dims.d[1] != mTiles)
            SFC_REJ("FC1_BLOCKS_SCALE: d[1] != mTiles");
        if (td.dims.d[2] != kTiles)
            SFC_REJ("FC1_BLOCKS_SCALE: d[2] != kTiles");
        if (td.dims.d[3] != 32 || td.dims.d[4] != 4 || td.dims.d[5] != 4)
            SFC_REJ("FC1_BLOCKS_SCALE: inner dims != [32,4,4]");
        return true;
    }
    case kIN_FC1_ALPHA:
    {
        if (td.type != DataType::kFLOAT)
            SFC_REJ("FC1_ALPHA: type != kFLOAT");
        if (td.dims.nbDims != 1)
            SFC_REJ("FC1_ALPHA: nbDims != 1");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC1_ALPHA: d[0] != mNumExperts");
        return true;
    }
    case kIN_FC2_QWEIGHTS:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC2_QWEIGHTS: type != kINT8");
        if (td.dims.nbDims != 3)
            SFC_REJ("FC2_QWEIGHTS: nbDims != 3");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC2_QWEIGHTS: d[0] != mNumExperts");
        if (td.dims.d[1] != mHiddenSize)
            SFC_REJ("FC2_QWEIGHTS: d[1] != mHiddenSize");
        if (td.dims.d[2] != mMoeInterSize / 2)
            SFC_REJ("FC2_QWEIGHTS: d[2] != mMoeInterSize/2");
        return true;
    }
    case kIN_FC2_BLOCKS_SCALE:
    {
        if (td.type != DataType::kINT8)
            SFC_REJ("FC2_BLOCKS_SCALE: type != kINT8");
        if (td.dims.nbDims != 6)
            SFC_REJ("FC2_BLOCKS_SCALE: nbDims != 6");
        int32_t const mTiles = ceilDivInt(mHiddenSize, 128);
        int32_t const kTiles = ceilDivInt(mMoeInterSize / kNvfp4SfVecSize, 4);
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC2_BLOCKS_SCALE: d[0] != mNumExperts");
        if (td.dims.d[1] != mTiles)
            SFC_REJ("FC2_BLOCKS_SCALE: d[1] != mTiles");
        if (td.dims.d[2] != kTiles)
            SFC_REJ("FC2_BLOCKS_SCALE: d[2] != kTiles");
        if (td.dims.d[3] != 32 || td.dims.d[4] != 4 || td.dims.d[5] != 4)
            SFC_REJ("FC2_BLOCKS_SCALE: inner dims != [32,4,4]");
        return true;
    }
    case kIN_FC2_ALPHA: [[fallthrough]];
    case kIN_INPUT_GLOBAL_SCALE: [[fallthrough]];
    case kIN_DOWN_INPUT_SCALE: [[fallthrough]];
    case kIN_E_SCORE_CORRECTION_BIAS:
    {
        if (td.type != DataType::kFLOAT)
            SFC_REJ("FC2_ALPHA/SCALE group: type != kFLOAT");
        if (td.dims.nbDims != 1)
            SFC_REJ("FC2_ALPHA/SCALE group: nbDims != 1");
        if (td.dims.d[0] != mNumExperts)
            SFC_REJ("FC2_ALPHA/SCALE group: d[0] != mNumExperts");
        return true;
    }
    case kOUT_OUTPUT:
    {
        if (td.type != DataType::kHALF)
            SFC_REJ("OUT_OUTPUT: type != kHALF");
        if (td.dims.nbDims != 3)
            SFC_REJ("OUT_OUTPUT: nbDims != 3");
        if (td.dims.d[2] != mHiddenSize)
            SFC_REJ("OUT_OUTPUT: d[2] != mHiddenSize");
        return true;
    }
    default: SFC_REJ("default: unknown pos");
    }
#undef SFC_REJ
}

int32_t NvFP4MoEPluginGeforce::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    (void) out;
    (void) nbOutputs;
    if (nbInputs != kNbPluginInputs)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce: expected %d inputs, got %d", kNbPluginInputs, nbInputs);
        return -1;
    }

    if (in[kIN_HIDDEN_STATES].max.nbDims == 3)
    {
        int32_t const hiddenSizeFromShape = static_cast<int32_t>(in[kIN_HIDDEN_STATES].max.d[2]);
        if (hiddenSizeFromShape != mHiddenSize)
        {
            LOG_ERROR(
                "NvFP4MoEPluginGeforce: hidden_size attribute (%d) does not match "
                "hidden_states max d[2] (%d). The plugin attribute is authoritative; "
                "rebuild the ONNX graph or fix the network input shape so they agree.",
                mHiddenSize, hiddenSizeFromShape);
            return -1;
        }
    }
    if (static_cast<int32_t>(in[kIN_E_SCORE_CORRECTION_BIAS].max.d[0]) != mNumExperts)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce: e_score_correction_bias d[0] (%d) must equal num_experts (%d)",
            static_cast<int32_t>(in[kIN_E_SCORE_CORRECTION_BIAS].max.d[0]), mNumExperts);
        return -1;
    }
    if (mRoutingMode == kROUTING_MODE_SIGMOID_GROUP_TOPK)
    {
        if (mNGroup <= 0 || mNumExperts % mNGroup != 0)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: n_group (%d) must be positive and evenly divide num_experts (%d)",
                mNGroup, mNumExperts);
            return -1;
        }
        if (mTopkGroup <= 0 || mTopkGroup > mNGroup)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: topk_group (%d) must be in [1, n_group=%d]", mTopkGroup, mNGroup);
            return -1;
        }
    }

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
    // Defer the (sm version, dtype, activation, backend, bounded hidden size,
    // I / E / top_k divisibility) gate to the SM12x fused runner so the plugin
    // and backend can never disagree about what the linked AOT pack supports.
    int32_t const smVersion = trt_edgellm::getSMVersion();
    bool const canImplement = CuteDslNvfp4MoeRunner::canImplement(mHiddenSize, mMoeInterSize, mNumExperts, mTopK,
        smVersion, toRunnerActivation(mActivationType), toRunnerIoDtype(mIoDtype), toRunnerBackend(mBackend));
    if (!canImplement)
    {
        LOG_ERROR(
            "NvFP4MoEPluginGeforce: shape tuple (H=%d, I=%d, E=%d, top_k=%d, sm=%d, "
            "act=%d, io=%d, backend=%d) is not supported by the SM12x CuteDSL fused "
            "runner. Requires io_dtype=FP16, activation in {identity, silu, swiglu, "
            "gelu, relu2}, H %% %d == 0, I %% %d == 0, E > 0, 0 < top_k <= E, and "
            "sm in {120, 121}.",
            mHiddenSize, mMoeInterSize, mNumExperts, mTopK, smVersion, mActivationType, mIoDtype, mBackend,
            kCuteDslLevelTileN, kCuteDslLevelTileN);
        return -1;
    }
#else
    LOG_ERROR(
        "NvFP4MoEPluginGeforce: no CuTeDSL SM12x fused MoE backend is linked -- "
        "rebuild with -DENABLE_CUTE_DSL=nvfp4_fused_moe after generating the "
        "matching AOT artifact via kernelSrcs/build_cutedsl.py");
    return -1;
#endif

    // Runtime-dim consistency: router_logits.d[0] must equal hidden_states.d[0] * d[1].
    int64_t const maxRouter = static_cast<int64_t>(in[kIN_ROUTER_LOGITS].max.d[0]);
    int64_t const maxB = static_cast<int64_t>(in[kIN_HIDDEN_STATES].max.d[0]);
    int64_t const maxS = static_cast<int64_t>(in[kIN_HIDDEN_STATES].max.d[1]);
    if (maxRouter > 0 && maxB > 0 && maxS > 0 && maxRouter != maxB * maxS)
    {
        LOG_ERROR(
            "NvFP4MoEPluginGeforce: router_logits max d[0] (%lld) must equal hidden_states "
            "max d[0]*d[1] (%lld*%lld)",
            static_cast<long long>(maxRouter), static_cast<long long>(maxB), static_cast<long long>(maxS));
        return -1;
    }

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
    // Resolve and validate \c max_routed_rows against the optimization profile.
    int64_t const profileMaxHiddenTokens = std::max<int64_t>(0, maxB) * std::max<int64_t>(0, maxS);
    int64_t const profileMaxTokens = std::max<int64_t>({profileMaxHiddenTokens, std::max<int64_t>(0, maxRouter)});
    if (profileMaxTokens > 0)
    {
        int32_t const profileMaxRoutedRows = computeProfileMaxRoutedRows(profileMaxTokens, mTopK);
        if (mMaxRoutedRows == 0)
        {
            mMaxRoutedRows = profileMaxRoutedRows;
        }
        else if (mMaxRoutedRows < profileMaxRoutedRows)
        {
            LOG_ERROR(
                "NvFP4MoEPluginGeforce: max_routed_rows (%d) is smaller than the "
                "optimization profile maximum num_tokens*top_k (%lld*%d=%d). The "
                "runner workspace would be undersized when the runtime shape "
                "saturates the profile. Either pass max_routed_rows >= %d or pass "
                "0 to auto-resolve from the profile.",
                mMaxRoutedRows, static_cast<long long>(profileMaxTokens), mTopK, profileMaxRoutedRows,
                profileMaxRoutedRows);
            return -1;
        }
    }
    else if (mMaxRoutedRows <= 0)
    {
        LOG_ERROR(
            "NvFP4MoEPluginGeforce: cannot resolve max_routed_rows. The optimization "
            "profile did not provide hidden_states / router_logits max shapes and the "
            "user did not pass an explicit max_routed_rows.");
        return -1;
    }
#endif

    return 0;
}

// Workspace sizing / enqueue

size_t NvFP4MoEPluginGeforce::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    (void) outputs;
    (void) nbOutputs;
    if (nbInputs != kNbPluginInputs)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce: getWorkspaceSize expected %d inputs, got %d", kNbPluginInputs, nbInputs);
        return 0;
    }
    (void) inputs;

#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
    try
    {
        int64_t const maxHiddenTokens = static_cast<int64_t>(inputs[kIN_HIDDEN_STATES].max.d[0])
            * static_cast<int64_t>(inputs[kIN_HIDDEN_STATES].max.d[1]);
        int64_t const maxRouterTokens = static_cast<int64_t>(inputs[kIN_ROUTER_LOGITS].max.d[0]);
        int64_t const maxTokens64 = std::max(maxHiddenTokens, maxRouterTokens);
        int32_t const maxTokens = static_cast<int32_t>(
            std::min<int64_t>(maxTokens64, static_cast<int64_t>(std::numeric_limits<int32_t>::max())));

        int32_t const maxRoutedRows
            = mMaxRoutedRows > 0 ? mMaxRoutedRows : computeProfileMaxRoutedRows(maxTokens64, mTopK);

        size_t total = 0;
        size_t const softmaxWs = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(maxTokens, mNumExperts);
        total = accumulateWorkspaceSize(
            total, rt::Coords{static_cast<int64_t>(std::max<size_t>(softmaxWs, 1))}, DataType::kINT8);
        total = accumulateWorkspaceSize(total, rt::Coords{maxTokens, mTopK}, DataType::kFLOAT);
        total = accumulateWorkspaceSize(total, rt::Coords{maxTokens, mTopK}, DataType::kINT32);
        size_t const runnerWs = CuteDslNvfp4MoeRunner::getWorkspaceSize(
            maxTokens, maxRoutedRows, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, toRunnerBackend(mBackend));
        if (runnerWs == 0)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: SM12x CuTeDSL backend returned zero workspace");
            return 0;
        }
        total = accumulateWorkspaceSize(
            total, rt::Coords{static_cast<int64_t>(std::max<size_t>(runnerWs, 1))}, DataType::kINT8);
        return total;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute NvFP4MoEPluginGeforce workspace size: %s", e.what());
        return 0;
    }
#else
    return 0;
#endif
}

int32_t NvFP4MoEPluginGeforce::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
#ifndef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
    (void) inputDesc;
    (void) outputDesc;
    (void) inputs;
    (void) outputs;
    (void) workspace;
    (void) stream;
    LOG_ERROR(
        "NvFP4MoEPluginGeforce: no CuTeDSL SM12x fused MoE backend is linked; "
        "rebuild with -DENABLE_CUTE_DSL=nvfp4_fused_moe");
    return -1;
#else
    try
    {
        if (!CuteDslNvfp4MoeRunner::loadKernelModules())
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: failed to load CuTe DSL AOT kernel modules");
            return -1;
        }
        if (mIdentityExpertTable == nullptr)
        {
            LOG_ERROR(
                "NvFP4MoEPluginGeforce: identity expert table is not allocated. "
                "attachToContext must run before the first enqueue (TRT V3 plugin "
                "lifecycle). If this plugin is being driven outside TRT, allocate "
                "via IGpuAllocator and call attachToContext first.");
            return -1;
        }

        PluginTensorDesc const& hiddenDesc = inputDesc[kIN_HIDDEN_STATES];
        if (hiddenDesc.dims.nbDims != 3)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: hidden_states must be 3D, got %d", hiddenDesc.dims.nbDims);
            return -1;
        }
        int32_t const batch = hiddenDesc.dims.d[0];
        int32_t const seqLen = hiddenDesc.dims.d[1];
        if (batch < 1 || seqLen < 1)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: batch/seq_len must be >= 1 (got %d, %d)", batch, seqLen);
            return -1;
        }
        int64_t const numTokens64 = static_cast<int64_t>(batch) * static_cast<int64_t>(seqLen);
        if (numTokens64 > std::numeric_limits<int32_t>::max())
        {
            LOG_ERROR(
                "NvFP4MoEPluginGeforce: batch*seq_len (%lld) overflows int32", static_cast<long long>(numTokens64));
            return -1;
        }
        int32_t const numTokens = static_cast<int32_t>(numTokens64);
        if (inputDesc[kIN_ROUTER_LOGITS].dims.d[0] != numTokens)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: router_logits d[0]=%d must equal batch*seq_len=%d",
                inputDesc[kIN_ROUTER_LOGITS].dims.d[0], numTokens);
            return -1;
        }

        int64_t const runtimeRoutedRows64 = numTokens64 * static_cast<int64_t>(mTopK);
        if (mMaxRoutedRows > 0 && runtimeRoutedRows64 > static_cast<int64_t>(mMaxRoutedRows))
        {
            LOG_ERROR(
                "NvFP4MoEPluginGeforce: runtime num_tokens*top_k (%lld) exceeds the "
                "resolved max_routed_rows cap (%d). Rebuild the engine with a profile "
                "whose max shapes cover this case, or pass a larger explicit "
                "max_routed_rows attribute.",
                static_cast<long long>(runtimeRoutedRows64), mMaxRoutedRows);
            return -1;
        }
        int32_t const maxRoutedRows
            = mMaxRoutedRows > 0 ? mMaxRoutedRows : computeProfileMaxRoutedRows(numTokens64, mTopK);

        // ==== Workspace carving (order must match getWorkspaceSize) ====
        size_t const softmaxWs = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);
        std::byte* ws = static_cast<std::byte*>(workspace);
        void* const softmaxScratch = assignTensorFromWorkspace(
            ws, rt::Coords{static_cast<int64_t>(std::max<size_t>(softmaxWs, 1))}, DataType::kINT8)
                                         .rawPointer();
        float* const topkWeightsPtr
            = static_cast<float*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
        int32_t* const topkIdsPtr
            = static_cast<int32_t*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kINT32).rawPointer());
        size_t const runnerWs = CuteDslNvfp4MoeRunner::getWorkspaceSize(
            numTokens, maxRoutedRows, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, toRunnerBackend(mBackend));
        if (runnerWs == 0)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: SM12x CuTeDSL backend returned zero workspace at enqueue");
            return -1;
        }
        void* const runnerScratch = assignTensorFromWorkspace(
            ws, rt::Coords{static_cast<int64_t>(std::max<size_t>(runnerWs, 1))}, DataType::kINT8)
                                        .rawPointer();

        // ==== Step 1: router softmax + top-k + renormalize ====
        rt::Tensor routerLogitsT(const_cast<void*>(inputs[kIN_ROUTER_LOGITS]),
            rt::Coords{inputDesc[kIN_ROUTER_LOGITS].dims}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor topkWeightsT(topkWeightsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        rt::Tensor topkIdsT(topkIdsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        if (mRoutingMode == kROUTING_MODE_SIGMOID_GROUP_TOPK)
        {
            rt::Tensor correctionBiasT(const_cast<void*>(inputs[kIN_E_SCORE_CORRECTION_BIAS]), {mNumExperts},
                rt::DeviceType::kGPU, DataType::kFLOAT);
            rt::OptionalInputTensor correctionBiasOpt = correctionBiasT;
            trt_edgellm::kernel::moeSigmoidGroupTopk(routerLogitsT, topkWeightsT, topkIdsT, mTopK, mNGroup, mTopkGroup,
                mNormTopkProb != 0, mRoutedScalingFactor, stream, correctionBiasOpt);
        }
        else if (mRoutingMode == kROUTING_MODE_SOFTMAX_TOPK)
        {
            // Pre-softmax additive bias (DeepSeek, Qwen3).
            rt::Tensor correctionBiasT(const_cast<void*>(inputs[kIN_E_SCORE_CORRECTION_BIAS]), {mNumExperts},
                rt::DeviceType::kGPU, DataType::kFLOAT);
            rt::OptionalInputTensor biasForSoftmax = correctionBiasT;
            trt_edgellm::kernel::moeTopkSoftmax(routerLogitsT, topkWeightsT, topkIdsT, mTopK,
                softmaxWs > 0 ? softmaxScratch : nullptr, softmaxWs, stream, /*renormalize=*/true, 0.0F,
                biasForSoftmax);
        }
        else
        {
            // kSOFTMAX_TOPK_POST_SCALE: softmax (no bias) + post-renorm multiplicative scale (Gemma4).
            rt::OptionalInputTensor biasForSoftmax;
            trt_edgellm::kernel::moeTopkSoftmax(routerLogitsT, topkWeightsT, topkIdsT, mTopK,
                softmaxWs > 0 ? softmaxScratch : nullptr, softmaxWs, stream, /*renormalize=*/true, 0.0F,
                biasForSoftmax);

            float const* scalePtr = static_cast<float const*>(inputs[kIN_E_SCORE_CORRECTION_BIAS]);
            trt_edgellm::kernel::applyPerExpertScale(topkWeightsPtr, topkIdsPtr, scalePtr, numTokens, mTopK, stream);
        }
        CUDA_CHECK(cudaGetLastError());

        (void) outputDesc;

        // ==== Step 2: fused MoE via SM12x CuTeDSL runner ====
        CuteDslNvfp4MoeParams p{};
        p.numTokens = numTokens;
        p.numExperts = mNumExperts;
        p.topK = mTopK;
        p.hiddenSize = mHiddenSize;
        p.moeInterSize = mMoeInterSize;
        p.maxRoutedRows = maxRoutedRows;
        p.hiddenStates = inputs[kIN_HIDDEN_STATES];
        p.topkIds = topkIdsPtr;
        p.topkWeights = topkWeightsPtr;
        p.fc1QWeights = inputs[kIN_FC1_QWEIGHTS];
        p.fc1BlocksScale = inputs[kIN_FC1_BLOCKS_SCALE];
        p.fc1Alpha = static_cast<float const*>(inputs[kIN_FC1_ALPHA]);
        p.fc2QWeights = inputs[kIN_FC2_QWEIGHTS];
        p.fc2BlocksScale = inputs[kIN_FC2_BLOCKS_SCALE];
        p.fc2Alpha = static_cast<float const*>(inputs[kIN_FC2_ALPHA]);
        p.inputGlobalScale = static_cast<float const*>(inputs[kIN_INPUT_GLOBAL_SCALE]);
        p.downInputScale = static_cast<float const*>(inputs[kIN_DOWN_INPUT_SCALE]);
        // Identity expert tables: two contiguous identity copies in a single buffer
        // owned by this plugin (allocated in attachToContext via IGpuAllocator). The
        // decode wrapper requires non-aliasing weight_expert_ids and
        // global_to_local_expert backing storage.
        p.weightExpertIds = mIdentityExpertTable;
        p.globalToLocalExpertIds = mIdentityExpertTable + mNumExperts;
        p.output = outputs[0];
        p.activation = toRunnerActivation(mActivationType);
        p.ioDtype = toRunnerIoDtype(mIoDtype);
        p.backend = toRunnerBackend(mBackend);

        CuteDslNvfp4MoeRunner runner;
        int32_t const rc = runner.run(p, runnerScratch, stream);
        if (rc != 0)
        {
            LOG_ERROR("NvFP4MoEPluginGeforce: runner.run() failed with code %d", rc);
            return rc;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce enqueue failed: %s", e.what());
        return -1;
    }
#endif // CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
}

int32_t NvFP4MoEPluginGeforce::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    (void) out;
    (void) nbOutputs;
    if (in == nullptr || nbInputs <= kIN_HIDDEN_STATES)
    {
        return 0;
    }
    auto const& hiddenDims = in[kIN_HIDDEN_STATES].dims;
    if (hiddenDims.nbDims != 3)
    {
        return 0;
    }
    int64_t const batch = static_cast<int64_t>(hiddenDims.d[0]);
    int64_t const seqLen = static_cast<int64_t>(hiddenDims.d[1]);
    if (batch <= 0 || seqLen <= 0)
    {
        return 0;
    }
    int64_t const routedRows = batch * seqLen * static_cast<int64_t>(mTopK);
    if (mMaxRoutedRows > 0 && routedRows > static_cast<int64_t>(mMaxRoutedRows))
    {
        LOG_ERROR(
            "NvFP4MoEPluginGeforce::onShapeChange: runtime num_tokens*top_k "
            "(%lld * %lld * %d = %lld) exceeds the resolved max_routed_rows cap (%d). "
            "Rebuild the engine with a profile whose max shapes cover this case, or "
            "pass a larger explicit max_routed_rows attribute.",
            static_cast<long long>(batch), static_cast<long long>(seqLen), mTopK, static_cast<long long>(routedRows),
            mMaxRoutedRows);
        return -1;
    }
    return 0;
}

IPluginV3* NvFP4MoEPluginGeforce::attachToContext(IPluginResourceContext* context) noexcept
{
    auto* plugin = static_cast<NvFP4MoEPluginGeforce*>(clone());
    if (plugin == nullptr)
    {
        return nullptr;
    }
#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
    // attachToContext() is the canonical IPluginV3OneRuntime hook for one-time
    // per-execution-context setup: it runs once per clone after deserialization
    // and before the first enqueue. Loading the AOT modules and allocating
    // per-plugin scratch here keeps cudaMalloc / H2D init out of the timed
    // enqueue path. enqueue() retains a cheap idempotent loadKernelModules()
    // guard for non-TRT callers but does NOT re-allocate the identity table.
    if (!CuteDslNvfp4MoeRunner::loadKernelModules())
    {
        LOG_ERROR("NvFP4MoEPluginGeforce: attachToContext failed to load CuTe DSL AOT kernel modules");
        delete plugin;
        return nullptr;
    }
    if (context == nullptr)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce::attachToContext: TRT did not provide a resource context");
        delete plugin;
        return nullptr;
    }
    IGpuAllocator* alloc = context->getGpuAllocator();
    if (alloc == nullptr)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce::attachToContext: TRT did not provide an IGpuAllocator");
        delete plugin;
        return nullptr;
    }
    if (plugin->mNumExperts <= 0)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce::attachToContext: invalid num_experts (%d)", plugin->mNumExperts);
        delete plugin;
        return nullptr;
    }
    // Allocate the per-plugin identity expert-id table. The decode wrapper
    // requires non-aliasing weight_expert_ids and global_to_local_expert backing
    // storage; we satisfy that with a single 2*E buffer that stores the identity
    // map [0..E-1] twice. The first half is exposed as weight_expert_ids and
    // the second half as global_to_local_expert in CuteDslNvfp4MoeParams.
    int64_t const numEntries = 2 * static_cast<int64_t>(plugin->mNumExperts);
    uint64_t const bytes = static_cast<uint64_t>(numEntries) * sizeof(int32_t);
    void* dev = alloc->allocate(bytes, /*alignment=*/256, AllocatorFlags{0});
    if (dev == nullptr)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce::attachToContext: IGpuAllocator->allocate(%llu) returned null",
            static_cast<unsigned long long>(bytes));
        delete plugin;
        return nullptr;
    }
    // Init [0..E-1, 0..E-1] on the host once and copy into the device table.
    // attachToContext is a one-time setup path, not a hot path, so a synchronous
    // cudaMemcpy is appropriate here -- the alternative (async on a transient
    // stream) would still have to synchronize before the first enqueue and is
    // strictly more code.
    std::vector<int32_t> hostIdentity(static_cast<size_t>(numEntries));
    for (int32_t i = 0; i < plugin->mNumExperts; ++i)
    {
        hostIdentity[static_cast<size_t>(i)] = i;
        hostIdentity[static_cast<size_t>(plugin->mNumExperts + i)] = i;
    }
    cudaError_t const err = cudaMemcpy(dev, hostIdentity.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        LOG_ERROR("NvFP4MoEPluginGeforce::attachToContext: cudaMemcpy failed (%s)", cudaGetErrorString(err));
        alloc->deallocate(dev);
        delete plugin;
        return nullptr;
    }
    plugin->mGpuAllocator = alloc;
    plugin->mIdentityExpertTable = static_cast<int32_t*>(dev);
#else
    (void) context;
#endif // CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
    return plugin;
}

PluginFieldCollection const* NvFP4MoEPluginGeforce::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("top_k", &mTopK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("activation_type", &mActivationType, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("n_group", &mNGroup, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("topk_group", &mTopkGroup, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("norm_topk_prob", &mNormTopkProb, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("routed_scaling_factor", &mRoutedScalingFactor, PluginFieldType::kFLOAT32, 1);
        mDataToSerialize.emplace_back("routing_mode", &mRoutingMode, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("backend", &mBackend, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("max_routed_rows", &mMaxRoutedRows, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("io_dtype", &mIoDtype, PluginFieldType::kINT32, 1);
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize NvFP4MoEPluginGeforce fields: %s", e.what());
        return nullptr;
    }
}

// Creator

NvFP4MoEPluginGeforceCreator::NvFP4MoEPluginGeforceCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("moe_inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("activation_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("n_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topk_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("norm_topk_prob", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("routed_scaling_factor", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("routing_mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backend", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_routed_rows", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("io_dtype", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* NvFP4MoEPluginGeforceCreator::getPluginName() const noexcept
{
    return kPLUGIN_NAME;
}

char const* NvFP4MoEPluginGeforceCreator::getPluginVersion() const noexcept
{
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* NvFP4MoEPluginGeforceCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void NvFP4MoEPluginGeforceCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace ? libNamespace : "";
}

char const* NvFP4MoEPluginGeforceCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

IPluginV3* NvFP4MoEPluginGeforceCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    (void) phase;
    try
    {
        auto* plugin = new NvFP4MoEPluginGeforce(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create NvFP4MoEPluginGeforce: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
