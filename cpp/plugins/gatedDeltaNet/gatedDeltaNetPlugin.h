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

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! \brief TensorRT plugin for Gated Delta Net (V2 — IPluginV2DynamicExt).
//!
//! Registered as "gated_delta_net". Dispatches to decode (seq_len==1),
//! prefill (seq_len>1), or MTP verify (use_mtp=true) CuTe DSL kernels.
//! Requires SM80+ and K=V=128.
//!
//! \par Dimension notation
//!   n   = batch size
//!   h   = number of Q/K heads
//!   hv  = number of V heads
//!   k   = head dimension K (must be 128)
//!   v   = head dimension V (must be 128)
//!
//! \par Inputs
//!   [0]  q               [n, seq_len, h,  k]   FP16  query
//!   [1]  k               [n, seq_len, h,  k]   FP16  key
//!   [2]  v               [n, seq_len, hv, v]   FP16  value
//!   [3]  a               [n, seq_len, hv]      FP16  input gate
//!   [4]  b               [n, seq_len, hv]      FP16  output gate
//!   [5]  A_log           [hv]                  FP32  log decay
//!   [6]  dt_bias         [hv]                  FP16  delta-time bias
//!   [7]  h0_source       [n, hv, k, v]         FP32  recurrent state in (batch-dense)
//!   [8]  context_lengths [n]                   INT32 valid token count per batch row
//!
//! \par Outputs
//!   [0]  o               [n, seq_len, hv, v]   FP16  output
//!   [1]  h0_out          [n, hv, k, v]         FP32  recurrent state out
//!   [2]  intermediate_states [n, seq_len, hv, k, v] FP32  (MTP only, optional)
//!        Per-step recurrent state cache for speculative-decoding rollback.
//!        Only populated when use_mtp=true (plugin attribute) and seq_len>1.
//!        When use_mtp=false this output is a 1-element dummy.
class GatedDeltaNetPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    //! \param name         Plugin instance name
    //! \param kDim         Head dimension K (must be 128 for CuTe DSL kernel)
    //! \param vDim         Head dimension V (must be 128 for CuTe DSL kernel)
    GatedDeltaNetPlugin(std::string const& name, int32_t kDim = 128, int32_t vDim = 128, bool useMTP = false);

    //! \brief Deserialization constructor
    GatedDeltaNetPlugin(std::string const& name, void const* data, size_t length);

    GatedDeltaNetPlugin() = delete;
    GatedDeltaNetPlugin(GatedDeltaNetPlugin const&) = delete;
    ~GatedDeltaNetPlugin() override;

    // IPluginV2DynamicExt
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    char const* getPluginType() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;

private:
    std::string mLayerName;
    std::string mNamespace;
    int32_t mKDim{128};    //!< Head dimension K (kernel supports 128 only)
    int32_t mVDim{128};    //!< Head dimension V (kernel supports 128 only)
    bool mUseMTP{false};   //!< Enable MTP output (intermediate_states as 3rd output)
    int32_t mSMVersion{0}; //!< Captured device SM version used for build-time capability checks
};

class GatedDeltaNetPluginCreator : public nvinfer1::IPluginCreator
{
public:
    GatedDeltaNetPluginCreator();
    ~GatedDeltaNetPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
