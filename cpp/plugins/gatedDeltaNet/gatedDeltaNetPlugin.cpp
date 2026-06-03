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

#include "gatedDeltaNetPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "plugins/utils/pluginUtils.h"
#ifdef CUTE_DSL_GDN_ENABLED
#include "kernels/gdnKernels/cuteDslGDNRunner.h"
#include "kernels/gdnKernels/gdnKernelUtils.cuh"
#endif

#include <cassert>
#include <cstdint>
#include <mutex>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kGDN_PLUGIN_VERSION{"1"};
constexpr char const* kGDN_PLUGIN_NAME{"gated_delta_net"};

constexpr int32_t kIN_Q_IDX{0};
constexpr int32_t kIN_K_IDX{1};
constexpr int32_t kIN_V_IDX{2};
constexpr int32_t kIN_A_IDX{3};
constexpr int32_t kIN_B_IDX{4};
constexpr int32_t kIN_A_LOG_IDX{5};
constexpr int32_t kIN_DT_BIAS_IDX{6};
constexpr int32_t kIN_H0_SOURCE_IDX{7};
constexpr int32_t kIN_CONTEXT_LENGTHS_IDX{8};
constexpr int32_t kOUT_O_IDX{0};
constexpr int32_t kOUT_H0_SOURCE_IDX{1};
constexpr int32_t kOUT_INTERMEDIATE_STATES_IDX{2};
constexpr int32_t kNUM_INPUTS{9};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{2};
constexpr int32_t kNUM_MTP_OPTIONAL_OUTPUTS{1};

} // namespace

PluginFieldCollection GatedDeltaNetPluginCreator::mFieldCollection{};
std::vector<nvinfer1::PluginField> GatedDeltaNetPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GatedDeltaNetPluginCreator);

// ---------------------------------------------------------------------------
// Plugin constructor — only this block is compilation-guarded.
// When CUTE_DSL_GDN_ENABLED is not set the constructor throws immediately so
// the object can never be constructed; all other methods are shared.
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim, bool useMTP)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mUseMTP(useMTP)
    , mSMVersion(getSMVersion())
{
    if (!CuteDslGDNRunner::canImplement(mKDim, mVDim, mSMVersion))
    {
        LOG_ERROR(
            "Cannot implement GatedDeltaNetPlugin (CuTe DSL): k_dim=%d v_dim=%d SM=%d. "
            "CuTe DSL GDN is only built for k=v=128 and requires SM>=80 (Ampere+). "
            "Use k_dim=v_dim=128 on a supported GPU, or rebuild without CuTe DSL GDN if applicable.",
            mKDim, mVDim, mSMVersion);
        throw std::runtime_error("Cannot implement the GatedDeltaNetPlugin configuration (CuTe DSL GDN).");
    }

    if (!CuteDslGDNRunner::loadKernelModules())
    {
        LOG_ERROR(
            "Failed to load CuTe DSL GDN kernel modules (gdn_decode / gdn_prefill AOT). "
            "Check that the engine was built with ENABLE_CUTE_DSL=gdn (or ALL), AOT .o/.h are present and match the "
            "exported API, and the CUDA driver is compatible.");
        throw std::runtime_error("Cannot load CuTe DSL GDN kernel modules for GatedDeltaNetPlugin.");
    }
}
#else
GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, int32_t kDim, int32_t vDim, bool useMTP)
    : mLayerName(name)
    , mKDim(kDim)
    , mVDim(vDim)
    , mUseMTP(useMTP)
{
    LOG_ERROR("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
    throw std::runtime_error("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
}
#endif // CUTE_DSL_GDN_ENABLED

GatedDeltaNetPlugin::GatedDeltaNetPlugin(std::string const& name, PluginFieldCollection const* fc)
    : mLayerName(name)
{
    mKDim = parsePluginScalarField<int32_t>("k_dim", fc).value_or(128);
    mVDim = parsePluginScalarField<int32_t>("v_dim", fc).value_or(128);
    mUseMTP = parsePluginScalarField<int32_t>("use_mtp", fc).value_or(0) != 0;

#ifdef CUTE_DSL_GDN_ENABLED
    mSMVersion = getSMVersion();
    CuteDslGDNRunner::loadKernelModules();
#else
    LOG_ERROR("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
    throw std::runtime_error("GatedDeltaNet plugin is not available: build with CUTE_DSL_GDN_ENABLED to enable it.");
#endif
}

GatedDeltaNetPlugin::~GatedDeltaNetPlugin() = default;

// ---------------------------------------------------------------------------
// IPluginV3
// ---------------------------------------------------------------------------

IPluginCapability* GatedDeltaNetPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuildV2*>(this);
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

IPluginV3* GatedDeltaNetPlugin::clone() noexcept
{
    try
    {
        auto* p = new GatedDeltaNetPlugin(mLayerName, mKDim, mVDim, mUseMTP);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    catch (...)
    {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// IPluginV3OneCore — metadata
// ---------------------------------------------------------------------------

char const* GatedDeltaNetPlugin::getPluginName() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPlugin::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

char const* GatedDeltaNetPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedDeltaNetPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild — shape / format
// ---------------------------------------------------------------------------

int32_t GatedDeltaNetPlugin::getNbOutputs() const noexcept
{
    return kNUM_REQUIRED_OUTPUTS + (mUseMTP ? kNUM_MTP_OPTIONAL_OUTPUTS : 0);
}

int32_t GatedDeltaNetPlugin::getOutputDataTypes(DataType* outputTypes, [[maybe_unused]] int32_t nbOutputs,
    DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    try
    {
        [[maybe_unused]] int32_t const expectedNbOutputs
            = kNUM_REQUIRED_OUTPUTS + (mUseMTP ? kNUM_MTP_OPTIONAL_OUTPUTS : 0);
        assert(nbOutputs == expectedNbOutputs);
        outputTypes[kOUT_O_IDX] = inputTypes[kIN_Q_IDX];
        outputTypes[kOUT_H0_SOURCE_IDX] = inputTypes[kIN_H0_SOURCE_IDX];
        if (mUseMTP)
        {
            outputTypes[kOUT_INTERMEDIATE_STATES_IDX] = DataType::kFLOAT;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

int32_t GatedDeltaNetPlugin::getOutputShapes(DimsExprs const* inputs, [[maybe_unused]] int32_t nbInputs,
    DimsExprs const* /* shapeInputs */, int32_t /* nbShapeInputs */, DimsExprs* outputs,
    [[maybe_unused]] int32_t nbOutputs, IExprBuilder& /* exprBuilder */) noexcept
{
    try
    {
        [[maybe_unused]] int32_t const expectedNbOutputs
            = kNUM_REQUIRED_OUTPUTS + (mUseMTP ? kNUM_MTP_OPTIONAL_OUTPUTS : 0);
        assert(nbInputs == kNUM_INPUTS);
        assert(nbOutputs == expectedNbOutputs);
        // o has same shape as v: [n, seq_len, hv, v]
        outputs[kOUT_O_IDX] = inputs[kIN_V_IDX];
        // h0_out has same shape as h0_source: [n, hv, k, v]
        outputs[kOUT_H0_SOURCE_IDX] = inputs[kIN_H0_SOURCE_IDX];
        if (mUseMTP)
        {
            // Per-token recurrent checkpoints: [n, seq_len, hv, k, v].
            outputs[kOUT_INTERMEDIATE_STATES_IDX].nbDims = 5;
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[0] = inputs[kIN_Q_IDX].d[0];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[1] = inputs[kIN_Q_IDX].d[1];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[2] = inputs[kIN_V_IDX].d[2];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[3] = inputs[kIN_Q_IDX].d[3];
            outputs[kOUT_INTERMEDIATE_STATES_IDX].d[4] = inputs[kIN_V_IDX].d[3];
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        return -1;
    }
}

bool GatedDeltaNetPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    int32_t const expectedNbOutputs = kNUM_REQUIRED_OUTPUTS + (mUseMTP ? kNUM_MTP_OPTIONAL_OUTPUTS : 0);
    if (nbInputs != kNUM_INPUTS || nbOutputs != expectedNbOutputs)
        return false;
    if (inOut[pos].desc.format != TensorFormat::kLINEAR)
        return false;
    if (pos == kIN_A_LOG_IDX || pos == kIN_H0_SOURCE_IDX)
        return inOut[pos].desc.type == DataType::kFLOAT;
    if (pos == kIN_CONTEXT_LENGTHS_IDX)
        return inOut[pos].desc.type == DataType::kINT32;
    // FP32 outputs: h0_out, intermediate_states (when present)
    if (pos == kNUM_INPUTS + kOUT_H0_SOURCE_IDX)
        return inOut[pos].desc.type == DataType::kFLOAT;
    if (mUseMTP && pos == kNUM_INPUTS + kOUT_INTERMEDIATE_STATES_IDX)
        return inOut[pos].desc.type == DataType::kFLOAT;
    // Everything else: FP16
    return inOut[pos].desc.type == DataType::kHALF;
}

int32_t GatedDeltaNetPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    [[maybe_unused]] DynamicPluginTensorDesc const* out, [[maybe_unused]] int32_t nbOutputs) noexcept
{
    int32_t const expectedNbOutputs = kNUM_REQUIRED_OUTPUTS + (mUseMTP ? kNUM_MTP_OPTIONAL_OUTPUTS : 0);
    if (nbInputs != kNUM_INPUTS)
    {
        LOG_ERROR("gated_delta_net: expected %d inputs, got %d", kNUM_INPUTS, nbInputs);
        return -1;
    }
    if (nbOutputs != expectedNbOutputs)
    {
        LOG_ERROR("gated_delta_net: expected %d outputs, got %d", expectedNbOutputs, nbOutputs);
        return -1;
    }
    if (in[kIN_Q_IDX].desc.type != DataType::kHALF || in[kIN_V_IDX].desc.type != DataType::kHALF)
    {
        LOG_ERROR("gated_delta_net: Q and V must be FP16");
        return -1;
    }
    if (in[kIN_Q_IDX].desc.dims.nbDims != 4 || in[kIN_V_IDX].desc.dims.nbDims != 4)
    {
        LOG_ERROR("gated_delta_net: Q and V must be 4D");
        return -1;
    }
    if (in[kIN_CONTEXT_LENGTHS_IDX].desc.type != DataType::kINT32 || in[kIN_CONTEXT_LENGTHS_IDX].desc.dims.nbDims != 1)
    {
        LOG_ERROR("gated_delta_net: context_lengths must be 1D INT32");
        return -1;
    }
    return 0;
}

size_t GatedDeltaNetPlugin::getWorkspaceSize([[maybe_unused]] DynamicPluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] DynamicPluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    size_t total = 0;

#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    int32_t const maxN = static_cast<int32_t>(inputs[kIN_CONTEXT_LENGTHS_IDX].max.d[0]);
    int32_t const maxHv = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[1]);
    int32_t const kDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[2]);
    int32_t const vDim = static_cast<int32_t>(inputs[kIN_H0_SOURCE_IDX].max.d[3]);

    // cu_seqlens [maxN+1] int32, padded to 128-byte alignment.
    size_t const cuSeqBytes = static_cast<size_t>(maxN + 1) * sizeof(int32_t);
    size_t const cuSeqPadded = (cuSeqBytes + 127u) & ~static_cast<size_t>(127u);
    // h0 scratch [maxN, maxHv, kDim, vDim] f32 — separate buffer for Blackwell h0_out.
    size_t const h0ScratchBytes = static_cast<size_t>(maxN) * maxHv * kDim * vDim * sizeof(float);

    total = cuSeqPadded + h0ScratchBytes;
#endif

    return total;
}

int32_t GatedDeltaNetPlugin::getAliasedInput(int32_t outputIndex) noexcept
{
    if (outputIndex == kOUT_H0_SOURCE_IDX)
    {
        return kIN_H0_SOURCE_IDX;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime — execution
// ---------------------------------------------------------------------------
#ifdef CUTE_DSL_GDN_ENABLED
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    CuteDslGDNRunner::loadKernelModules();

    int64_t const* qDims = inputDesc[kIN_Q_IDX].dims.d;
    int32_t const n = static_cast<int32_t>(qDims[0]);
    int32_t const seq_len = static_cast<int32_t>(qDims[1]);
    int32_t const h = static_cast<int32_t>(qDims[2]);
    int32_t const k_dim = static_cast<int32_t>(qDims[3]);

    int64_t const* vDims = inputDesc[kIN_V_IDX].dims.d;
    int32_t const hv = static_cast<int32_t>(vDims[2]);
    int32_t const v_dim = static_cast<int32_t>(vDims[3]);

    // Determine if this call should use MTP decode path.
    // MTP mode activates only for short multi-token verification sequences (tree verify),
    // not for normal prefill. The MTP kernel writes per-step intermediate states which adds
    // minor overhead; small prefills (seq_len <= kMTPMaxSeqLen) that happen to pass through
    // this path pay a small cost for the intermediate state writes, but it is harmless.
    // TODO: refactor the dispatch logic to explicitly distinguish MTP tree-verify decoding
    // from prefill when 1 < seq_len <= kMTPMaxSeqLen (e.g. pass an execution-phase flag
    // from the runtime instead of relying solely on seq_len range heuristics).
    constexpr int32_t kMTPMaxSeqLen = 8;
    bool const mtpActive = mUseMTP && (seq_len > 1) && (seq_len <= kMTPMaxSeqLen);

    // h0 is batch-dense [n, hv, k, v]
    size_t const h0Bytes = static_cast<size_t>(n) * hv * static_cast<size_t>(k_dim) * v_dim * sizeof(float);
    void* h0Out = outputs[kOUT_H0_SOURCE_IDX];

    // For MTP: the MTP kernel updates h0_source in-place, so we always need the copy
    // (h0Out serves as the working state buffer that the kernel reads/writes).
    // For normal: same logic as before — copy if input != output.
    if (h0Out != inputs[kIN_H0_SOURCE_IDX])
    {
        cudaMemcpyAsync(h0Out, inputs[kIN_H0_SOURCE_IDX], h0Bytes, cudaMemcpyDeviceToDevice, stream);
    }

    GDNParams params{};
    params.q = const_cast<void*>(inputs[kIN_Q_IDX]);
    params.k = const_cast<void*>(inputs[kIN_K_IDX]);
    params.v = const_cast<void*>(inputs[kIN_V_IDX]);
    params.a = const_cast<void*>(inputs[kIN_A_IDX]);
    params.b = const_cast<void*>(inputs[kIN_B_IDX]);
    params.A_log = const_cast<void*>(inputs[kIN_A_LOG_IDX]);
    params.dt_bias = const_cast<void*>(inputs[kIN_DT_BIAS_IDX]);
    params.h0_source = h0Out;
    params.context_lengths = const_cast<void*>(inputs[kIN_CONTEXT_LENGTHS_IDX]);
    params.o = outputs[kOUT_O_IDX];
    params.n = n;
    params.seq_len = seq_len;
    params.h = h;
    params.hv = hv;
    params.k_dim = k_dim;
    params.v_dim = v_dim;
    params.smVersion = mSMVersion;

    if (mtpActive)
    {
        // MTP decode: process all seq_len draft tokens with per-step state caching.
        params.use_mtp = true;
        params.intermediate_states = outputs[kOUT_INTERMEDIATE_STATES_IDX];
    }
    else
    {
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
        // Blackwell prefill: carve cu_seqlens and h0 scratch out of the pre-allocated workspace.
        //   workspace layout: [cu_seqlens: (n+1)*int32, pad to 128B] [h0_scratch: n*hv*k*v*f32]
        if (seq_len > 1 && mSMVersion >= 100)
        {
            size_t const cuSeqBytes = static_cast<size_t>(n + 1) * sizeof(int32_t);
            size_t const cuSeqPadded = (cuSeqBytes + 127u) & ~static_cast<size_t>(127u);

            char* bwBase = static_cast<char*>(workspace);
            launchGdnCalCuSeqLens(inputs[kIN_CONTEXT_LENGTHS_IDX], bwBase, n, stream);
            params.cu_seqlens = bwBase;
            params.h0_scratch = bwBase + cuSeqPadded;
        }
#endif
    }

    CuteDslGDNRunner runner;
    int ret = runner.run(params, stream);

    return (ret == 0) ? 0 : -1;
}
#else
int32_t GatedDeltaNetPlugin::enqueue(PluginTensorDesc const* /* inputDesc */, PluginTensorDesc const* /* outputDesc */,
    void const* const* /* inputs */, void* const* /* outputs */, void* /* workspace */,
    cudaStream_t /* stream */) noexcept
{
    // Constructor already threw; this path should be unreachable.
    return -1;
}
#endif // CUTE_DSL_GDN_ENABLED

int32_t GatedDeltaNetPlugin::onShapeChange(PluginTensorDesc const* /* in */, int32_t /* nbInputs */,
    PluginTensorDesc const* /* out */, int32_t /* nbOutputs */) noexcept
{
    return 0;
}

IPluginV3* GatedDeltaNetPlugin::attachToContext(IPluginResourceContext* /* context */) noexcept
{
    return clone();
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

PluginFieldCollection const* GatedDeltaNetPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("k_dim", &mKDim, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("v_dim", &mVDim, PluginFieldType::kINT32, 1);
    mUseMTPField = mUseMTP ? 1 : 0;
    mDataToSerialize.emplace_back("use_mtp", &mUseMTPField, PluginFieldType::kINT32, 1);

    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------

GatedDeltaNetPluginCreator::GatedDeltaNetPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("k_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("v_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_mtp", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* GatedDeltaNetPluginCreator::getPluginName() const noexcept
{
    return kGDN_PLUGIN_NAME;
}

char const* GatedDeltaNetPluginCreator::getPluginVersion() const noexcept
{
    return kGDN_PLUGIN_VERSION;
}

PluginFieldCollection const* GatedDeltaNetPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

char const* GatedDeltaNetPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void GatedDeltaNetPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

IPluginV3* GatedDeltaNetPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase /* phase */) noexcept
{
    try
    {
        auto* plugin = new GatedDeltaNetPlugin(std::string(name), fc);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("GatedDeltaNetPluginCreator::createPlugin failed: %s", e.what());
        return nullptr;
    }
}

} // namespace plugins
} // namespace trt_edgellm
