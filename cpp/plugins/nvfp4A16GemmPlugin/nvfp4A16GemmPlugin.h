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

#include <NvInferRuntime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

/*!
 * @brief TensorRT V3 plugin for a dense FP16-activation / Marlin-packed NVFP4 (W4A16) GEMM.
 *
 * This is a thin wrapper over the MoE Marlin FP16xE2M1 kernel (moeNvfp4A16MarlinGemm): it drives the kernel with a
 * single expert and top_k = 1 so that every input row maps to expert 0. The weights, block scales, and global scale
 * carry a leading expert dimension of 1 to reuse the MoE kernel's shape contract unchanged.
 *
 * gemm_n is the Marlin-padded output dimension (a multiple of 128); the caller slices the logical width downstream.
 * max_m is the optimization-profile token capacity (batch * sequence) used to size the routing/Marlin workspace.
 */
class Nvfp4A16GemmPlugin : public nvinfer1::IPluginV3,
                           public nvinfer1::IPluginV3OneCore,
                           public nvinfer1::IPluginV3OneBuild,
                           public nvinfer1::IPluginV3OneRuntime
{
public:
    Nvfp4A16GemmPlugin(std::string const& name, int32_t gemmN, int32_t gemmK, int32_t maxM);

    Nvfp4A16GemmPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Nvfp4A16GemmPlugin() = delete;
    Nvfp4A16GemmPlugin(Nvfp4A16GemmPlugin const&) = delete;
    Nvfp4A16GemmPlugin& operator=(Nvfp4A16GemmPlugin const&) = delete;
    ~Nvfp4A16GemmPlugin() noexcept override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    int32_t getNbOutputs() const noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    void validateAttributes() const;
    bool validateTensorDesc(int32_t pos, nvinfer1::PluginTensorDesc const& desc) const noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mGemmN{}; //!< Marlin-padded output dimension N (multiple of 128).
    int32_t mGemmK{}; //!< Input dimension K.
    int32_t mMaxM{};  //!< Profile token capacity (batch * sequence) for workspace sizing.

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

//! Creator for Nvfp4A16GemmPlugin.
class Nvfp4A16GemmPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Nvfp4A16GemmPluginCreator();
    ~Nvfp4A16GemmPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
