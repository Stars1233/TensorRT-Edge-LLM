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

#include "nvfp4MoePlugin.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/stringUtils.h"
#include "common/tensor.h"
#include "kernels/moe/NvFP4MoEContiguousGemmRunner.h"
#include "kernels/moe/NvFP4MoEFC2FinalizeRunner.h"
#include "kernels/moe/NvFP4MoEUtils.h"
#include "kernels/moe/fp4SupportKernels/alphaCompute.h"
#include "kernels/moe/fp4SupportKernels/buildLayout.h"
#include "kernels/moe/fp4SupportKernels/fp4Quantize.h"
#include "kernels/moe/fp4SupportKernels/moeGather.h"
#include "kernels/moe/fp4SupportKernels/nvfp4MoeTypes.h"
#include "kernels/moe/moeSigmoidGroupTopkKernels.h"
#include "kernels/moe/moeTopkSoftmaxKernels.h"
#include "kernels/moe/nvf4_w4an/kernels.h"
#include "plugins/utils/pluginUtils.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
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

namespace
{
//! Version "1": NvFP4 MoE plugin — FP16 hidden + NVFP4 weights + FP32 router.
//! ``mMoeInterSize`` is the raw FC1 weight N-dimension: ``I`` for non-gated,
//! ``2*I`` for gated (SwiGLU interleaved gate+up). ``nOut = nOutFor()`` is the
//! FC1 output (= FC2 contraction) dim after activation folding.
//! - FC1 up weights: ``[E, H, mMoeInterSize/2]`` INT8 (N-major; 2 FP4 nibbles per byte).
//! - FC2 down weights: ``[E, nOut, H/2]`` INT8 (N-major; 2 FP4 nibbles per byte along H).
//! - FC1 prefill SF atom: M=mMoeInterSize, K=H/16 (raw IEEE FP8 E4M3 bytes, scheme B).
//! - FC2 prefill SF atom: M=H, K=nOut/16 (raw IEEE FP8 E4M3 bytes, scheme B).
//! - FC1 decode SF (slot 9): legacy CuTe DSL decode atom layout (no longer used; decode reads prefill SF at slot 4).
//! - FC2 decode SF (slot 10): legacy CuTe DSL decode atom layout (no longer used; decode reads prefill SF at slot 7).
//! - Per-expert FP32 global scales ``s_max_ex / 448`` for FC1 and FC2.
//! - FP32 length-2 ``hidden_global_scale`` for the internal activation FP4 quants.
//! Dispatch: ``numTokens <= kPrefillDispatchThreshold = 16`` → decode (CuTe DSL W4A16
//! GEMV on atom-layout SF); ``> 16`` → CuteDSL N-major prefill.
constexpr char const* kNVFP4_MOE_PLUGIN_VERSION{"1"};
constexpr char const* kNVFP4_MOE_PLUGIN_NAME{"Nvfp4MoePlugin"};
//! Input count: 11 FP4 MoE tensors (router logits / hidden / prefill + decode SFs / expert
//! payloads + global scales) plus slot [11] \c e_score_correction_bias used by
//! \c moeSigmoidGroupTopk (mode 1) or as an optional bias by \c moeTopkSoftmax (mode 0).
constexpr int32_t kNbPluginInputs{12};
constexpr int32_t kNvfp4MoeQuantizationGroupSize{16};
//! Dispatch threshold: \c numTokens (B·S) > this value → prefill path; otherwise decode.
//! Set to ``16`` — batches of 1..16 tokens take the W4A16 decode GEMV path with
//! prefill atom-layout SF (slots 4/7); larger batches take the CuteDSL
//! N-major prefill path.
constexpr int32_t kPrefillDispatchThreshold{16};
//! CuteDSL FC1/FC2 tile size. Matches the AOT kernel's persistent-tile block size.
constexpr int32_t kPrefillTileSize{128};

//! Map the serialized \c activation_type (0 = ReLU², 1 = SiLU decode / SwiGLU prefill) to
//! the FC1 grouped-GEMM activation enum.
trt_edgellm::kernel::nvfp4_moe::Activation mapActivation(ActivationType const t) noexcept
{
    int32_t const v = static_cast<int32_t>(t);
    if (v == 1)
    {
        return trt_edgellm::kernel::nvfp4_moe::Activation::kSwiglu;
    }
    return trt_edgellm::kernel::nvfp4_moe::Activation::kRelu2;
}

//! FC1 output width along N. For SwiGLU the FC1 emits \c 2·nOut raw values which the fused
//! epilogue folds to \c nOut via `silu(gate) · value`. For ReLU² both are \c moeInterSize.
int32_t nOutFor(ActivationType t, int32_t moeInterSize)
{
    return (mapActivation(t) == trt_edgellm::kernel::nvfp4_moe::Activation::kSwiglu) ? moeInterSize / 2 : moeInterSize;
}

int64_t padUp64(int64_t a, int64_t b)
{
    return ((a + b - 1) / b) * b;
}

//! Upper bound on permuted rows when routing is worst-case: `T + L·(tileSize-1)` slots are
//! needed because each active expert can waste up to `tileSize-1` pad rows to align its
//! group to a tile boundary. Tile-padded again because the GEMM scheduler iterates whole
//! tiles.
int64_t computeMaxPermutedM(int64_t numTokens, int64_t topK, int64_t numLocalExperts, int64_t tileSize)
{
    return padUp64(numTokens * topK + numLocalExperts * (tileSize - 1), tileSize);
}

size_t computeNvfp4MoeDecodeWorkspaceSize(int32_t numTokens, int32_t numExperts, int32_t topK, int32_t moeInterSize,
    int32_t /*hiddenSize*/, int32_t routingMode) noexcept
{
    try
    {
        size_t size = 0;
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kFLOAT);
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kINT32);
        if (routingMode == static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK))
        {
            size_t const softmaxWorkspaceSizeBytes
                = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, numExperts);
            if (softmaxWorkspaceSizeBytes > 0)
            {
                size = accumulateWorkspaceSize(
                    size, rt::Coords{static_cast<int64_t>(softmaxWorkspaceSizeBytes)}, DataType::kINT8);
            }
        }
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
        // CuTe DSL decode GEMV intermediate buffer. Uses moeInterSize (which is 2*I for gated)
        // so the workspace may over-allocate for gated models — harmless.
        size_t const gemvWsBytes
            = trt_edgellm::CuteDslDecodeGemvRunner::getWorkspaceSize(numTokens, topK, moeInterSize);
        size = accumulateWorkspaceSize(size, rt::Coords{static_cast<int64_t>(gemvWsBytes)}, DataType::kINT8);
#endif
        return size;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Nvfp4MoePlugin decode workspace size: %s", e.what());
        return 0;
    }
}

//! Reserve workspace for one prefill call. Sections — in the exact order that
//! \c enqueuePrefill later consumes them via \c assignTensorFromWorkspace :
//!   [A] topkWeights [T, K] FP32
//!   [B] topkIndices [T, K] INT32
//!   [C] softmaxWs (optional) INT8
//!   [D] aFP4      [paddedMSrc, H/2] INT8
//!   [E] aSF       [paddedMSrc * paddedSfColsH] INT8
//!   [F] gathered  [permutedM, H/2] INT8
//!   [G] gatheredSF[permutedM * paddedSfColsH] INT8   (memset-zeroed each call)
//!   [H] fc1Out    [permutedM * nOut] FP16
//!   [I] fc1FP4    [permutedM, nOut/2] INT8
//!   [J] fc1SF     [permutedM * paddedSfColsN] INT8
//!
//! FC1/FC2 α buffers are \b not part of the per-call workspace — they are
//! persistent tensors allocated by \c attachToContext and populated once on
//! the first \c enqueuePrefill.
size_t computeNvfp4MoePrefillWorkspaceSize(int32_t numTokens, int32_t numExperts, int32_t topK, int32_t moeInterSize,
    int32_t hiddenSize, ActivationType activationType, int32_t routingMode) noexcept
{
    try
    {
        int32_t const nOut = nOutFor(activationType, moeInterSize);
        int64_t const paddedMSrc = padUp64(numTokens, 128);
        int64_t const permutedM
            = computeMaxPermutedM(numTokens, topK, numExperts, static_cast<int64_t>(kPrefillTileSize));
        int64_t const paddedSfColsH = padUp64(hiddenSize / 16, 4);
        int64_t const paddedSfColsN = padUp64(nOut / 16, 4);

        size_t size = 0;
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kFLOAT); // [A]
        size = accumulateWorkspaceSize(size, rt::Coords{numTokens, topK}, DataType::kINT32); // [B]
        if (routingMode == static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK))
        {
            size_t const softmaxWs = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, numExperts);
            if (softmaxWs > 0)
            {
                size = accumulateWorkspaceSize(
                    size, rt::Coords{static_cast<int64_t>(softmaxWs)}, DataType::kINT8); // [C]
            }
        }
        size = accumulateWorkspaceSize(size, rt::Coords{paddedMSrc * (hiddenSize / 2)}, DataType::kINT8); // [D]
        size = accumulateWorkspaceSize(size, rt::Coords{paddedMSrc * paddedSfColsH}, DataType::kINT8);    // [E]
        size = accumulateWorkspaceSize(size, rt::Coords{permutedM * (hiddenSize / 2)}, DataType::kINT8);  // [F]
        size = accumulateWorkspaceSize(size, rt::Coords{permutedM * paddedSfColsH}, DataType::kINT8);     // [G]
        size = accumulateWorkspaceSize(size, rt::Coords{permutedM * nOut}, DataType::kHALF);              // [H]
        size = accumulateWorkspaceSize(size, rt::Coords{permutedM * (nOut / 2)}, DataType::kINT8);        // [I]
        size = accumulateWorkspaceSize(size, rt::Coords{permutedM * paddedSfColsN}, DataType::kINT8);     // [J]
        // α buffers are persistent (see attachToContext + first-enqueue init).
        return size;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to compute Nvfp4MoePlugin prefill workspace size: %s", e.what());
        return 0;
    }
}
} // namespace

PluginFieldCollection Nvfp4MoePluginCreator::mFieldCollection{};
std::vector<PluginField> Nvfp4MoePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Nvfp4MoePluginCreator);

// ============================================================================
// Construction / destruction
// ============================================================================

Nvfp4MoePlugin::Nvfp4MoePlugin(std::string const& name, int32_t const numExperts, int32_t const topK,
    int32_t const hiddenSize, int32_t const moeInterSize, ActivationType const activationType, int32_t const nGroup,
    int32_t const topkGroup, int32_t const normTopkProb, float const routedScalingFactor, int32_t const routingMode)
    : mLayerName(name)
    , mNumExperts(numExperts)
    , mTopK(topK)
    , mHiddenSize(hiddenSize)
    , mMoeInterSize(moeInterSize)
    , mActivationType(activationType)
    , mQuantizationGroupSize(kNvfp4MoeQuantizationGroupSize)
    , mNGroup(nGroup)
    , mTopkGroup(topkGroup)
    , mNormTopkProb(normTopkProb)
    , mRoutedScalingFactor(routedScalingFactor)
    , mRoutingMode(routingMode)
{
}

Nvfp4MoePlugin::Nvfp4MoePlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    bool sawMaxTokens{false};
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
            mActivationType = static_cast<ActivationType>(*static_cast<int32_t const*>(fc->fields[i].data));
        }
        else if (fieldName == "quantization_group_size")
        {
            mQuantizationGroupSize = *static_cast<int32_t const*>(fc->fields[i].data);
        }
        else if (fieldName == "max_tokens")
        {
            mMaxTokens = *static_cast<int32_t const*>(fc->fields[i].data);
            sawMaxTokens = true;
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
    }
    if (mQuantizationGroupSize <= 0)
    {
        mQuantizationGroupSize = kNvfp4MoeQuantizationGroupSize;
    }
    if (mQuantizationGroupSize != kNvfp4MoeQuantizationGroupSize)
    {
        throw std::invalid_argument(format::fmtstr("Nvfp4MoePlugin: quantization_group_size must be %d, got %d",
            static_cast<int>(kNvfp4MoeQuantizationGroupSize), static_cast<int>(mQuantizationGroupSize)));
    }
    if (mRoutingMode != static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK)
        && mRoutingMode != static_cast<int32_t>(Nvfp4MoeRoutingMode::kSIGMOID_GROUP_TOPK))
    {
        throw std::invalid_argument(
            format::fmtstr("Nvfp4MoePlugin: routing_mode must be %d (SOFTMAX_TOPK) or %d (SIGMOID_GROUP_TOPK), got %d",
                static_cast<int>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK),
                static_cast<int>(Nvfp4MoeRoutingMode::kSIGMOID_GROUP_TOPK), static_cast<int>(mRoutingMode)));
    }
    if (!sawMaxTokens)
    {
        throw std::invalid_argument(
            "Nvfp4MoePlugin: runtime deserializing constructor requires serialized 'max_tokens' field "
            "(old engines must be re-exported with plugin version 2)");
    }
    if (mMaxTokens <= 0)
    {
        throw std::invalid_argument(format::fmtstr(
            "Nvfp4MoePlugin: serialized max_tokens (%d) must be positive", static_cast<int>(mMaxTokens)));
    }
}

Nvfp4MoePlugin::~Nvfp4MoePlugin() noexcept
{
    // The rt::Tensor layout members are non-owning views over IGpuAllocator memory; free
    // explicitly if we are the attached instance (`mGpuAllocator != nullptr`). Unattached
    // sources (build-phase plugin, clone source that never reached attachToContext, failed
    // deserialize) retain `mGpuAllocator == nullptr` and the dtor is a no-op.
    if (mGpuAllocator != nullptr)
    {
        auto freeIfNonNull = [this](rt::Tensor& t) {
            if (t.rawPointer() != nullptr)
            {
                mGpuAllocator->deallocate(t.rawPointer());
            }
        };
        freeIfNonNull(mLayoutBuffers.tileIdxToGroupIdx);
        freeIfNonNull(mLayoutBuffers.tileIdxToMnLimit);
        freeIfNonNull(mLayoutBuffers.permutedIdxToExpandedIdx);
        freeIfNonNull(mLayoutBuffers.numNonExitingTiles);
        freeIfNonNull(mFC1Alpha);
        freeIfNonNull(mFC2Alpha);
    }
}

IPluginCapability* Nvfp4MoePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3* Nvfp4MoePlugin::clone() noexcept
{
    try
    {
        auto* plugin = new Nvfp4MoePlugin(mLayerName, mNumExperts, mTopK, mHiddenSize, mMoeInterSize, mActivationType,
            mNGroup, mTopkGroup, mNormTopkProb, mRoutedScalingFactor, mRoutingMode);
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->mMaxTokens = mMaxTokens;
        // Intentionally do NOT copy mLayoutBuffers or mGpuAllocator — attachToContext
        // allocates the clone's own layout buffers. This avoids pointer aliasing across
        // source and clone (which would cause double-free).
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to clone Nvfp4MoePlugin: %s", e.what());
        return nullptr;
    }
}

char const* Nvfp4MoePlugin::getPluginName() const noexcept
{
    return kNVFP4_MOE_PLUGIN_NAME;
}

char const* Nvfp4MoePlugin::getPluginVersion() const noexcept
{
    return kNVFP4_MOE_PLUGIN_VERSION;
}

char const* Nvfp4MoePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nvfp4MoePlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

int32_t Nvfp4MoePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Nvfp4MoePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t /*nbOutputs*/, DataType const* /*inputTypes*/, int32_t /*nbInputs*/) const noexcept
{
    outputTypes[0] = DataType::kHALF;
    return 0;
}

int32_t Nvfp4MoePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* /*shapeInputs*/,
    int32_t /*nbShapeInputs*/, DimsExprs* outputs, int32_t /*nbOutputs*/, IExprBuilder& exprBuilder) noexcept
{
    assert(nbInputs == kNbPluginInputs);
    (void) nbInputs;
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[1].d[0];
    outputs[0].d[1] = inputs[1].d[1];
    outputs[0].d[2] = exprBuilder.constant(static_cast<int64_t>(mHiddenSize));
    return 0;
}

// ============================================================================
// Build-time validation
// ============================================================================

bool Nvfp4MoePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(nbInputs == kNbPluginInputs && nbOutputs == 1);
    assert(pos < (nbInputs + nbOutputs));
    (void) nbInputs;
    (void) nbOutputs;

    auto const& td = inOut[pos].desc;
    bool ok = (td.format == TensorFormat::kLINEAR);

    int32_t const nOut = nOutFor(mActivationType, mMoeInterSize);

    bool result = false;
    switch (pos)
    {
    case 0: // router_logits FP32 [T, E]
        result = ok && td.type == DataType::kFLOAT && td.dims.nbDims == 2 && td.dims.d[1] == mNumExperts;
        break;
    case 1: // hidden_states FP16 [B, S, H]
        result = ok && td.type == DataType::kHALF && td.dims.nbDims == 3 && td.dims.d[2] == mHiddenSize;
        break;
    case 2: // hidden_global_scale FP32 [2]
        result = ok && td.type == DataType::kFLOAT && td.dims.nbDims == 1 && td.dims.d[0] == 2;
        break;
    case 3: // up_qweights INT8 [E, H, mMoeInterSize/2] (gated: 2*I/2 = I bytes for interleaved gate+up)
        result = ok && td.type == DataType::kINT8 && td.dims.nbDims == 3 && td.dims.d[0] == mNumExperts
            && td.dims.d[1] == mHiddenSize && td.dims.d[2] == mMoeInterSize / 2;
        break;
    case 4: // up_block_scale INT8 [E, padUp(mMoeInterSize, 128), padUp(H/16, 4)] — atom-layout
        result = ok && td.type == DataType::kINT8 && td.dims.nbDims == 3 && td.dims.d[0] == mNumExperts
            && td.dims.d[1] == padUp64(mMoeInterSize, 128)
            && td.dims.d[2] == padUp64(mHiddenSize / mQuantizationGroupSize, 4);
        break;
    case 5: // up_global_scale FP32 [E]
        result = ok && td.type == DataType::kFLOAT && td.dims.nbDims == 1 && td.dims.d[0] == mNumExperts;
        break;
    case 6: // down_qweights INT8 [E, nOut, H/2]
        result = ok && td.type == DataType::kINT8 && td.dims.nbDims == 3 && td.dims.d[0] == mNumExperts
            && td.dims.d[1] == nOut && td.dims.d[2] == mHiddenSize / 2;
        break;
    case 7: // down_block_scale INT8 [E, padUp(H, 128), padUp(nOut/16, 4)]
        result = ok && td.type == DataType::kINT8 && td.dims.nbDims == 3 && td.dims.d[0] == mNumExperts
            && td.dims.d[1] == padUp64(mHiddenSize, 128) && td.dims.d[2] == padUp64(nOut / mQuantizationGroupSize, 4);
        break;
    case 8: // down_global_scale FP32 [E]
        result = ok && td.type == DataType::kFLOAT && td.dims.nbDims == 1 && td.dims.d[0] == mNumExperts;
        break;
    case 9:  // up_block_scale_decode — legacy, unused by decode path (reads prefill SF at slot 4)
    case 10: // down_block_scale_decode — legacy, unused by decode path (reads prefill SF at slot 7)
        result = ok && td.type == DataType::kINT8;
        break;
    case 11: // e_score_correction_bias FP32 [num_experts]
        result = ok && td.type == DataType::kFLOAT && td.dims.nbDims == 1 && td.dims.d[0] == mNumExperts;
        break;
    case 12: // output FP16 [B, S, H]
        result = ok && td.type == DataType::kHALF && td.dims.nbDims == 3 && td.dims.d[2] == mHiddenSize;
        break;
    default: result = false;
    }
    return result;
}

int32_t Nvfp4MoePlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    if (nbInputs != kNbPluginInputs)
    {
        return -1;
    }
    if (mHiddenSize % 64 != 0)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_size (%d) must be a multiple of 64", mHiddenSize);
        return -1;
    }
    int32_t const nOut = nOutFor(mActivationType, mMoeInterSize);
    // Prefill-path invariants for atom-layout SF: sfCols must be a multiple of 4
    // (k_tile = 4 SF cols). hiddenSize / 16 must be a multiple of 4 → H % 64 == 0 (above).
    // The symmetric constraint on the FC1 output side requires nOut % 64 == 0.
    if (nOut % 64 != 0)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: FC1 output width nOut (%d) must be a multiple of 64 (atom-layout SF k_tile alignment)",
            static_cast<int>(nOut));
        return -1;
    }
    if (mMoeInterSize % 64 != 0)
    {
        LOG_ERROR("Nvfp4MoePlugin: moe_inter_size (%d) must be a multiple of 64", mMoeInterSize);
        return -1;
    }
    if (mTopK < 1 || mNumExperts < mTopK)
    {
        LOG_ERROR("Nvfp4MoePlugin: topK (%d) must be >=1 and numExperts (%d) must be >= topK", mTopK, mNumExperts);
        return -1;
    }
    if (mQuantizationGroupSize != kNvfp4MoeQuantizationGroupSize)
    {
        LOG_ERROR("Nvfp4MoePlugin: quantization_group_size (%d) must be %d", static_cast<int>(mQuantizationGroupSize),
            static_cast<int>(kNvfp4MoeQuantizationGroupSize));
        return -1;
    }
    if (in[1].desc.type != DataType::kHALF)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_states must be FP16. Got type=%d", static_cast<int>(in[1].desc.type));
        return -1;
    }

    // Union across all profiles: the final mMaxTokens must cover every profile's max B×S.
    int64_t const profileMaxTokens = static_cast<int64_t>(in[1].max.d[0]) * static_cast<int64_t>(in[1].max.d[1]);
    if (profileMaxTokens <= 0)
    {
        LOG_ERROR("Nvfp4MoePlugin: hidden_states profile max (%lld × %lld) must have positive batch and seq_len",
            static_cast<long long>(in[1].max.d[0]), static_cast<long long>(in[1].max.d[1]));
        return -1;
    }
    mMaxTokens = std::max(mMaxTokens,
        static_cast<int32_t>(
            std::min<int64_t>(profileMaxTokens, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))));

    // Router token count matches hidden_states at every defined profile when both are bound.
    int64_t const routerTokens = static_cast<int64_t>(in[0].max.d[0]);
    if (routerTokens > 0 && routerTokens != profileMaxTokens)
    {
        LOG_ERROR("Nvfp4MoePlugin: router_logits max d[0] (%lld) must equal hidden_states max d[0]*d[1] (%lld)",
            static_cast<long long>(routerTokens), static_cast<long long>(profileMaxTokens));
        return -1;
    }
    if (static_cast<int32_t>(in[11].max.d[0]) != mNumExperts)
    {
        LOG_ERROR("Nvfp4MoePlugin: e_score_correction_bias d[0] (%d) must equal num_experts (%d)",
            static_cast<int>(in[11].max.d[0]), mNumExperts);
        return -1;
    }
    if (mRoutingMode == static_cast<int32_t>(Nvfp4MoeRoutingMode::kSIGMOID_GROUP_TOPK))
    {
        if (mNGroup <= 0 || mNumExperts % mNGroup != 0)
        {
            LOG_ERROR("Nvfp4MoePlugin: n_group (%d) must be positive and evenly divide num_experts (%d)", mNGroup,
                mNumExperts);
            return -1;
        }
        if (mTopkGroup <= 0 || mTopkGroup > mNGroup)
        {
            LOG_ERROR("Nvfp4MoePlugin: topk_group (%d) must be in [1, n_group=%d]", mTopkGroup, mNGroup);
            return -1;
        }
    }
    return 0;
}

size_t Nvfp4MoePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* /*outputs*/, int32_t /*nbOutputs*/) const noexcept
{
    assert(nbInputs == kNbPluginInputs);
    (void) nbInputs;
    int64_t const maxHiddenTokens = static_cast<int64_t>(inputs[1].max.d[0]) * static_cast<int64_t>(inputs[1].max.d[1]);
    int64_t const maxTokens64 = std::max(static_cast<int64_t>(inputs[0].max.d[0]), maxHiddenTokens);
    int32_t const maxTokens = static_cast<int32_t>(
        std::min<int64_t>(maxTokens64, static_cast<int64_t>(std::numeric_limits<int32_t>::max())));

    size_t const decodeWs
        = computeNvfp4MoeDecodeWorkspaceSize(maxTokens, mNumExperts, mTopK, mMoeInterSize, mHiddenSize, mRoutingMode);
    size_t const prefillWs = computeNvfp4MoePrefillWorkspaceSize(
        maxTokens, mNumExperts, mTopK, mMoeInterSize, mHiddenSize, mActivationType, mRoutingMode);
    return std::max(decodeWs, prefillWs);
}

int32_t Nvfp4MoePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t /*nbInputs*/, PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    int64_t const tokens = static_cast<int64_t>(in[1].dims.d[0]) * static_cast<int64_t>(in[1].dims.d[1]);
    if (tokens > static_cast<int64_t>(mMaxTokens))
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: onShapeChange runtime tokens (%lld) exceeds serialized max_tokens (%d). The engine "
            "must be rebuilt with a profile that covers this shape.",
            static_cast<long long>(tokens), mMaxTokens);
        return -1;
    }
    return 0;
}

// ============================================================================
// attachToContext / serialization
// ============================================================================

IPluginV3* Nvfp4MoePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    try
    {
        auto* cloned = static_cast<Nvfp4MoePlugin*>(this->clone());
        if (cloned == nullptr)
        {
            return nullptr;
        }

        // Construct runners on the clone. Build-phase plugins may have mMaxTokens == 0 at
        // clone time; that is fine because only the runtime-attached instance's enqueue()
        // will execute, and attachToContext only runs after deserialization.
        int32_t const nOut = nOutFor(cloned->mActivationType, cloned->mMoeInterSize);
        cloned->mFC1Runner = std::make_unique<trt_edgellm::kernel::nvfp4_moe::NvFP4MoEContiguousGemmRunner>(
            cloned->mNumExperts, cloned->mTopK, cloned->mMoeInterSize, cloned->mHiddenSize, kPrefillTileSize,
            mapActivation(cloned->mActivationType), trt_edgellm::kernel::nvfp4_moe::OutputDType::kFP16);
        cloned->mFC2Runner
            = std::make_unique<trt_edgellm::kernel::nvfp4_moe::NvFP4MoEFC2FinalizeRunner>(cloned->mNumExperts,
                cloned->mTopK, cloned->mHiddenSize, nOut, trt_edgellm::kernel::nvfp4_moe::OutputDType::kFP16);
        // Load AOT kernel modules once per process (the runners guard with a static flag).
        (void) trt_edgellm::kernel::nvfp4_moe::NvFP4MoEContiguousGemmRunner::loadKernelModules();
        (void) trt_edgellm::kernel::nvfp4_moe::NvFP4MoEFC2FinalizeRunner::loadKernelModules();

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
        // Load CuTe DSL decode GEMV kernel modules.
        (void) trt_edgellm::CuteDslDecodeGemvRunner::loadKernelModules();
#endif

        // Allocate layout buffers for the worst-case profile. Sized for `mMaxTokens` so the
        // same buffers handle every legal runtime shape without reallocation (a
        // potentially-captured runtime stream cannot safely call alloc/free).
        IGpuAllocator* alloc = context->getGpuAllocator();
        if (alloc == nullptr)
        {
            LOG_ERROR("Nvfp4MoePlugin::attachToContext: TRT did not provide an IGpuAllocator");
            delete cloned;
            return nullptr;
        }
        cloned->mGpuAllocator = alloc;

        int64_t const permutedMMax = computeMaxPermutedM(
            cloned->mMaxTokens, cloned->mTopK, cloned->mNumExperts, static_cast<int64_t>(kPrefillTileSize));
        int64_t const numTilesMax = permutedMMax / kPrefillTileSize;

        auto allocI32 = [alloc](int64_t count) -> rt::Tensor {
            uint64_t const bytes = static_cast<uint64_t>(count) * sizeof(int32_t);
            void* ptr = alloc->allocate(bytes, /*alignment=*/256, /*flags=*/AllocatorFlags{0});
            ELLM_CHECK(ptr != nullptr, "Nvfp4MoePlugin::attachToContext: IGpuAllocator->allocate returned null");
            return rt::Tensor(ptr, rt::Coords{count}, rt::DeviceType::kGPU, DataType::kINT32);
        };

        cloned->mLayoutBuffers.tileIdxToGroupIdx = allocI32(numTilesMax);
        cloned->mLayoutBuffers.tileIdxToMnLimit = allocI32(numTilesMax);
        cloned->mLayoutBuffers.permutedIdxToExpandedIdx = allocI32(permutedMMax);
        cloned->mLayoutBuffers.numNonExitingTiles = allocI32(1);

        // Persistent FC1/FC2 α buffers (``[L]`` FP32 each). Contents are
        // computed from constant plugin inputs on the first prefill enqueue
        // and reused forever after — see the ``mAlphaInitialized`` gate below.
        auto allocFP32 = [alloc](int64_t count) -> rt::Tensor {
            uint64_t const bytes = static_cast<uint64_t>(count) * sizeof(float);
            void* ptr = alloc->allocate(bytes, /*alignment=*/256, /*flags=*/AllocatorFlags{0});
            ELLM_CHECK(ptr != nullptr, "Nvfp4MoePlugin::attachToContext: IGpuAllocator->allocate returned null");
            return rt::Tensor(ptr, rt::Coords{count}, rt::DeviceType::kGPU, DataType::kFLOAT);
        };
        cloned->mFC1Alpha = allocFP32(static_cast<int64_t>(cloned->mNumExperts));
        cloned->mFC2Alpha = allocFP32(static_cast<int64_t>(cloned->mNumExperts));
        cloned->mAlphaInitialized = false;

        return cloned;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4MoePlugin::attachToContext failed: %s", e.what());
        return nullptr;
    }
}

PluginFieldCollection const* Nvfp4MoePlugin::getFieldsToSerialize() noexcept
{
    try
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back("num_experts", &mNumExperts, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("top_k", &mTopK, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("hidden_size", &mHiddenSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("moe_inter_size", &mMoeInterSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("activation_type", &mActivationType, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("quantization_group_size", &mQuantizationGroupSize, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("max_tokens", &mMaxTokens, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("n_group", &mNGroup, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("topk_group", &mTopkGroup, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("norm_topk_prob", &mNormTopkProb, PluginFieldType::kINT32, 1);
        mDataToSerialize.emplace_back("routed_scaling_factor", &mRoutedScalingFactor, PluginFieldType::kFLOAT32, 1);
        mDataToSerialize.emplace_back("routing_mode", &mRoutingMode, PluginFieldType::kINT32, 1);
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
        return &mFCToSerialize;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to serialize Nvfp4MoePlugin fields: %s", e.what());
        return nullptr;
    }
}

// ============================================================================
// Enqueue dispatch
// ============================================================================

int32_t Nvfp4MoePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int64_t const numTokens64
            = static_cast<int64_t>(inputDesc[1].dims.d[0]) * static_cast<int64_t>(inputDesc[1].dims.d[1]);
        bool const usePrefill = (numTokens64 > static_cast<int64_t>(kPrefillDispatchThreshold));
        return usePrefill ? enqueuePrefill(inputDesc, outputDesc, inputs, outputs, workspace, stream)
                          : enqueueDecoding(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Nvfp4MoePlugin enqueue failed: %s", e.what());
        return -1;
    }
}

// ============================================================================
// Decode path (numTokens <= 16): top-k softmax + CuTe DSL decode GEMVs
// ============================================================================

int32_t Nvfp4MoePlugin::enqueueDecoding(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (inputDesc[1].dims.nbDims != 3)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: hidden_states must be 3D, got nbDims=%d", static_cast<int>(inputDesc[1].dims.nbDims));
        return -1;
    }
    int32_t const batch = inputDesc[1].dims.d[0];
    int32_t const seqLen = inputDesc[1].dims.d[1];
    int64_t const numTokens64 = static_cast<int64_t>(batch) * static_cast<int64_t>(seqLen);
    int32_t const numTokens = static_cast<int32_t>(numTokens64);

    std::byte* ws = static_cast<std::byte*>(workspace);

    float* topkWeightsPtr
        = static_cast<float*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kFLOAT).rawPointer());
    int32_t* topkIndicesPtr
        = static_cast<int32_t*>(assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kINT32).rawPointer());
    size_t const softmaxWsBytes = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);
    void* softmaxWsPtr = nullptr;
    if (mRoutingMode == static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK) && softmaxWsBytes > 0)
    {
        softmaxWsPtr
            = assignTensorFromWorkspace(ws, {static_cast<int64_t>(softmaxWsBytes)}, DataType::kINT8).rawPointer();
    }

    rt::Tensor routerLogitsTensor(
        const_cast<void*>(inputs[0]), rt::Coords{inputDesc[0].dims}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor topkWeightsTensor(topkWeightsPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor topkIndicesTensor(topkIndicesPtr, {numTokens, mTopK}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor correctionBiasTensor(
        const_cast<void*>(inputs[11]), rt::Coords{mNumExperts}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::OptionalInputTensor correctionBiasOpt = correctionBiasTensor;

    if (mRoutingMode == static_cast<int32_t>(Nvfp4MoeRoutingMode::kSIGMOID_GROUP_TOPK))
    {
        trt_edgellm::kernel::moeSigmoidGroupTopk(routerLogitsTensor, topkWeightsTensor, topkIndicesTensor, mTopK,
            mNGroup, mTopkGroup, mNormTopkProb != 0, mRoutedScalingFactor, stream, correctionBiasOpt);
    }
    else
    {
        trt_edgellm::kernel::moeTopkSoftmax(routerLogitsTensor, topkWeightsTensor, topkIndicesTensor, mTopK,
            softmaxWsPtr, softmaxWsBytes, stream, /*renormalize=*/true, /*dropRate=*/0.0F, correctionBiasOpt);
    }
    CUDA_CHECK(cudaGetLastError());

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    // CuTe DSL decode GEMV — W4A16 NVFP4: N-major weights + prefill atom-layout FP8 E4M3 block scales + FP32 global.
    // For gated (SwiGLU): up_weights [E, H, I] interleaved gate+up, moeInterSize = I (post-fold).
    // For non-gated: up_weights [E, H, I/2], moeInterSize = mMoeInterSize.
    int32_t const nOut = nOutFor(mActivationType, mMoeInterSize);
    bool const isGated = (mapActivation(mActivationType) == trt_edgellm::kernel::nvfp4_moe::Activation::kSwiglu);

    trt_edgellm::CuteDslDecodeGemvParams gemvParams{};
    gemvParams.numTokens = numTokens;
    gemvParams.numExperts = mNumExperts;
    gemvParams.topK = mTopK;
    gemvParams.hiddenSize = mHiddenSize;
    gemvParams.moeInterSize = nOut;
    gemvParams.hiddenStates = static_cast<__half const*>(inputs[1]);
    gemvParams.topkIds = topkIndicesPtr;
    gemvParams.topkWeights = topkWeightsPtr;
    gemvParams.upWeights = inputs[3];
    gemvParams.upScales = inputs[4];                                 // prefill atom-layout FP8 E4M3 SF
    gemvParams.upGlobalScale = static_cast<float const*>(inputs[5]); // FP32 per-expert
    gemvParams.downWeights = inputs[6];
    gemvParams.downScales = inputs[7];                                 // prefill atom-layout FP8 E4M3 SF
    gemvParams.downGlobalScale = static_cast<float const*>(inputs[8]); // FP32 per-expert
    gemvParams.output = static_cast<__half*>(outputs[0]);
    gemvParams.activationKind = (static_cast<int32_t>(mActivationType) == 1) ? 1 : 0; // 0=ReLU2, 1=SiLU
    gemvParams.isGated = isGated;

    // Workspace for intermediate buffer (after router workspace).
    size_t const gemvWsBytes = trt_edgellm::CuteDslDecodeGemvRunner::getWorkspaceSize(numTokens, mTopK, nOut);
    void* gemvWs = assignTensorFromWorkspace(ws, {static_cast<int64_t>(gemvWsBytes)}, DataType::kINT8).rawPointer();

    return mDecodeGemvRunner.run(gemvParams, gemvWs, stream);
#else
    LOG_ERROR("Nvfp4MoePlugin: decode path requires CuTe DSL NVFP4 MoE kernels (CUTE_DSL_NVFP4_MOE_ENABLED)");
    return -1;
#endif
}

// ============================================================================
// Prefill path (numTokens > 16): CuteDSL grouped GEMM pipeline (K0..K5)
// ============================================================================

int32_t Nvfp4MoePlugin::enqueuePrefill(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (mFC1Runner == nullptr || mFC2Runner == nullptr)
    {
        LOG_ERROR("Nvfp4MoePlugin: prefill path invoked on un-attached plugin instance (runners unset)");
        return -1;
    }
    if (mLayoutBuffers.tileIdxToGroupIdx.rawPointer() == nullptr)
    {
        LOG_ERROR("Nvfp4MoePlugin: prefill path invoked without layout buffers (attachToContext missing?)");
        return -1;
    }
    if (inputDesc[1].dims.nbDims != 3)
    {
        LOG_ERROR(
            "Nvfp4MoePlugin: hidden_states must be 3D, got nbDims=%d", static_cast<int>(inputDesc[1].dims.nbDims));
        return -1;
    }

    int32_t const batch = inputDesc[1].dims.d[0];
    int32_t const seqLen = inputDesc[1].dims.d[1];
    int64_t const numTokens64 = static_cast<int64_t>(batch) * static_cast<int64_t>(seqLen);
    if (numTokens64 > static_cast<int64_t>(mMaxTokens))
    {
        LOG_ERROR("Nvfp4MoePlugin: prefill tokens (%lld) exceeds max_tokens (%d)", static_cast<long long>(numTokens64),
            static_cast<int>(mMaxTokens));
        return -1;
    }
    int32_t const numTokens = static_cast<int32_t>(numTokens64);
    int32_t const H = mHiddenSize;
    int32_t const L = mNumExperts;
    int32_t const nOut = nOutFor(mActivationType, mMoeInterSize);
    int64_t const paddedMSrc = padUp64(numTokens, 128);
    int64_t const permutedM = computeMaxPermutedM(numTokens, mTopK, L, static_cast<int64_t>(kPrefillTileSize));
    int64_t const paddedSfColsH = padUp64(H / 16, 4);
    int64_t const paddedSfColsN = padUp64(nOut / 16, 4);

#if SUPPORTS_FP4

    // ---- Workspace carving (order must mirror computeNvfp4MoePrefillWorkspaceSize) ----
    std::byte* ws = static_cast<std::byte*>(workspace);

    rt::Tensor topkWeightsT = assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kFLOAT);
    rt::Tensor topkIndicesT = assignTensorFromWorkspace(ws, {numTokens, mTopK}, DataType::kINT32);
    size_t const softmaxWsBytes = trt_edgellm::kernel::getMoeTopkSoftmaxWorkspaceSize(numTokens, mNumExperts);
    void* softmaxWsPtr = nullptr;
    if (mRoutingMode == static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK) && softmaxWsBytes > 0)
    {
        softmaxWsPtr
            = assignTensorFromWorkspace(ws, {static_cast<int64_t>(softmaxWsBytes)}, DataType::kINT8).rawPointer();
    }
    rt::Tensor aFP4T = assignTensorFromWorkspace(ws, {paddedMSrc * (H / 2)}, DataType::kINT8);
    rt::Tensor aSFT = assignTensorFromWorkspace(ws, {paddedMSrc * paddedSfColsH}, DataType::kINT8);
    rt::Tensor gatheredT = assignTensorFromWorkspace(ws, {permutedM * (H / 2)}, DataType::kINT8);
    rt::Tensor gatheredSFT = assignTensorFromWorkspace(ws, {permutedM * paddedSfColsH}, DataType::kINT8);
    rt::Tensor fc1OutT = assignTensorFromWorkspace(ws, {permutedM * nOut}, DataType::kHALF);
    rt::Tensor fc1FP4T = assignTensorFromWorkspace(ws, {permutedM * (nOut / 2)}, DataType::kINT8);
    rt::Tensor fc1SFT = assignTensorFromWorkspace(ws, {permutedM * paddedSfColsN}, DataType::kINT8);

    // FC1/FC2 α are persistent (allocated in attachToContext). Compute them
    // on the first enqueue only — the inputs that feed α (``hidden_global_scale``
    // and per-expert weight global scales) are constants, so re-running
    // computeFC1Alpha / computeFC2Alpha every call is wasted work.
    if (!mAlphaInitialized)
    {
        trt_edgellm::kernel::computeFC1Alpha(static_cast<float const*>(inputs[2]), static_cast<float const*>(inputs[5]),
            mFC1Alpha.dataPointer<float>(), L, stream);
        CUDA_CHECK(cudaGetLastError());
        trt_edgellm::kernel::computeFC2Alpha(static_cast<float const*>(inputs[2]) + 1,
            static_cast<float const*>(inputs[8]), mFC2Alpha.dataPointer<float>(), L, stream);
        CUDA_CHECK(cudaGetLastError());
        mAlphaInitialized = true;
    }

    // ---- K0a: router (softmax-topk or sigmoid-group-topk) → LOCAL expert indices since L == E ----
    rt::Tensor routerLogitsT(
        const_cast<void*>(inputs[0]), rt::Coords{inputDesc[0].dims}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor correctionBiasT(
        const_cast<void*>(inputs[11]), rt::Coords{mNumExperts}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::OptionalInputTensor correctionBiasOptT = correctionBiasT;
    if (mRoutingMode == static_cast<int32_t>(Nvfp4MoeRoutingMode::kSIGMOID_GROUP_TOPK))
    {
        trt_edgellm::kernel::moeSigmoidGroupTopk(routerLogitsT, topkWeightsT, topkIndicesT, mTopK, mNGroup, mTopkGroup,
            mNormTopkProb != 0, mRoutedScalingFactor, stream, correctionBiasOptT);
    }
    else
    {
        trt_edgellm::kernel::moeTopkSoftmax(routerLogitsT, topkWeightsT, topkIndicesT, mTopK, softmaxWsPtr,
            softmaxWsBytes, stream, /*renormalize=*/true, /*dropRate=*/0.0F, correctionBiasOptT);
    }
    CUDA_CHECK(cudaGetLastError());

    // ---- K0b: GPU layout build ----
    trt_edgellm::kernel::buildLayoutGpu(
        mLayoutBuffers, topkIndicesT.dataPointer<int32_t>(), numTokens, mTopK, L, kPrefillTileSize, stream);
    CUDA_CHECK(cudaGetLastError());

    // ---- M1: pre-zero gatheredSF (gather contract; structural atom-layout padding) ----
    CUDA_CHECK(cudaMemsetAsync(gatheredSFT.rawPointer(), 0, static_cast<size_t>(permutedM * paddedSfColsH), stream));

    // ---- K1: fp4Quantize(hidden, actGsFC1) → aFP4, aSF ----
    // Reshape hidden [B, S, H] → [numTokens, H] so fp4Quantize sees a 2D tensor.
    __half* hiddenPtr = static_cast<__half*>(const_cast<void*>(inputs[1]));
    rt::Tensor hiddenT(hiddenPtr, {numTokens, H}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor actGsFC1T(
        static_cast<float*>(const_cast<void*>(inputs[2])), {1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor actGsFC2T(
        static_cast<float*>(const_cast<void*>(inputs[2])) + 1, {1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    trt_edgellm::kernel::fp4Quantize(hiddenT, actGsFC1T, aFP4T, aSFT, stream);
    CUDA_CHECK(cudaGetLastError());

    // ---- K2: gather permuted FP4 + SF ----
    // ``permutedIdxToExpandedIdx`` is sized for the worst-case ``mMaxTokens``;
    // pass the runtime ``permutedM`` so the kernel grid matches the dst extent.
    trt_edgellm::kernel::launchMoeGather(aFP4T, gatheredT, aSFT, gatheredSFT, mLayoutBuffers.permutedIdxToExpandedIdx,
        static_cast<int32_t>(permutedM), mTopK, H, stream);
    CUDA_CHECK(cudaGetLastError());

    // ---- K3: FC1 grouped GEMM (fused α · acc → FP32 epilogue → activation → FP16 out) ----
    trt_edgellm::kernel::nvfp4_moe::MoELayout layout{};
    layout.tileIdxToGroupIdx = mLayoutBuffers.tileIdxToGroupIdx.dataPointer<int32_t>();
    layout.tileIdxToMnLimit = mLayoutBuffers.tileIdxToMnLimit.dataPointer<int32_t>();
    layout.permutedIdxToExpandedIdx = mLayoutBuffers.permutedIdxToExpandedIdx.dataPointer<int32_t>();
    layout.numNonExitingTiles = mLayoutBuffers.numNonExitingTiles.dataPointer<int32_t>();

    mFC1Runner->run(gatheredT.rawPointer(), /*up_qweights=*/inputs[3], gatheredSFT.rawPointer(),
        /*up_block_scale=*/inputs[4], fc1OutT.rawPointer(), mFC1Alpha.rawPointer(), layout, permutedM, stream);
    CUDA_CHECK(cudaGetLastError());

    // ---- K4: fp4Quantize(fc1Out, actGsFC2) → fc1FP4, fc1SF ----
    //
    // The FC1 output tensor lives in the workspace at size [permutedM, nOut]; we pass it
    // as a 2D tensor so fp4Quantize can iterate the row dimension directly. Note that
    // rows beyond the last active tile are zero-filled (GEMM padding rows produce zeros
    // via K2 zero-fill → `GEMM(0, B) = 0`), so the scan cannot inflate the max-abs.
    rt::Tensor fc1Out2DT(fc1OutT.rawPointer(), {permutedM, nOut}, rt::DeviceType::kGPU, DataType::kHALF);
    trt_edgellm::kernel::fp4Quantize(fc1Out2DT, actGsFC2T, fc1FP4T, fc1SFT, stream);
    CUDA_CHECK(cudaGetLastError());

    // ---- M2: zero output for atomic scatter-reduce ----
    size_t const outputBytes = static_cast<size_t>(numTokens) * static_cast<size_t>(H) * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(outputs[0], 0, outputBytes, stream));

    // ---- K5: FC2 grouped GEMM + scatter-reduce (α from persistent mFC2Alpha) ----
    mFC2Runner->run(fc1FP4T.rawPointer(), /*down_qweights=*/inputs[6], fc1SFT.rawPointer(),
        /*down_block_scale=*/inputs[7], outputs[0], mFC2Alpha.rawPointer(), layout, topkWeightsT.rawPointer(),
        permutedM, numTokens, stream);
    CUDA_CHECK(cudaGetLastError());
    return 0;
#else
    (void) inputs;
    (void) outputs;
    (void) paddedMSrc;
    (void) permutedM;
    (void) paddedSfColsH;
    (void) paddedSfColsN;
    LOG_ERROR("Nvfp4MoePlugin: NVFP4 MoE prefill requires CUDA >= 12.8 (FP4 support)");
    return -1;
#endif
}

// ============================================================================
// Creator
// ============================================================================

Nvfp4MoePluginCreator::Nvfp4MoePluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_experts", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("hidden_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("moe_inter_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("activation_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quantization_group_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("n_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topk_group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("norm_topk_prob", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("routed_scaling_factor", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("routing_mode", nullptr, PluginFieldType::kINT32, 1));

    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* Nvfp4MoePluginCreator::getPluginName() const noexcept
{
    return kNVFP4_MOE_PLUGIN_NAME;
}

PluginFieldCollection const* Nvfp4MoePluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void Nvfp4MoePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* Nvfp4MoePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* Nvfp4MoePluginCreator::getPluginVersion() const noexcept
{
    return kNVFP4_MOE_PLUGIN_VERSION;
}

IPluginV3* Nvfp4MoePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        if (phase == TensorRTPhase::kBUILD)
        {
            // ONNX import: only the public attributes are present. max_tokens is not yet
            // known (filled in by configurePlugin), so use the attribute constructor.
            auto numExperts = parsePluginScalarField<int32_t>("num_experts", fc);
            auto topK = parsePluginScalarField<int32_t>("top_k", fc);
            auto hiddenSize = parsePluginScalarField<int32_t>("hidden_size", fc);
            auto moeInterSize = parsePluginScalarField<int32_t>("moe_inter_size", fc);
            auto activationType = parsePluginScalarField<int32_t>("activation_type", fc);
            if (!numExperts || !topK || !hiddenSize || !moeInterSize || !activationType)
            {
                LOG_ERROR("Nvfp4MoePluginCreator::createPlugin: build phase missing one or more required attributes");
                return nullptr;
            }
            auto nGroup = parsePluginScalarField<int32_t>("n_group", fc);
            auto topkGroup = parsePluginScalarField<int32_t>("topk_group", fc);
            auto normTopkProb = parsePluginScalarField<int32_t>("norm_topk_prob", fc);
            auto routedScalingFactor = parsePluginScalarField<float>("routed_scaling_factor", fc);
            auto routingMode = parsePluginScalarField<int32_t>("routing_mode", fc);
            auto* plugin = new Nvfp4MoePlugin(std::string(name), *numExperts, *topK, *hiddenSize, *moeInterSize,
                static_cast<ActivationType>(*activationType), nGroup.value_or(1), topkGroup.value_or(1),
                normTopkProb.value_or(1), routedScalingFactor.value_or(1.0f),
                routingMode.value_or(static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK)));
            plugin->setPluginNamespace(mNamespace.c_str());
            return plugin;
        }
        // Runtime phase: deserializing constructor — must see `max_tokens`. The ctor throws
        // on missing field so we return nullptr uniformly on any failure here.
        auto* plugin = new Nvfp4MoePlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create Nvfp4MoePlugin (phase=%d): %s", static_cast<int>(phase), e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
