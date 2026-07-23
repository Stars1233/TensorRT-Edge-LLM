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

namespace kernels
{
class CuteDslGemmNvFp4Runner;
} // namespace kernels

namespace plugins
{

/// Fused row-parallel GEMM + AllReduce plugin for NVFP4 workloads.
///
/// This plugin is NVFP4-only. It runs GEMM and then reduces rank-local output
/// across the tensor-parallel group.
///
/// Inputs:
///   [0] activation FP4 [B,K]             — FP4 quantized activation
///   [1] activation_scale FP32 [B,1,K/16] — combined FP32 scale from DequantizeLinear
///   [2] weight_f4 FP4 [N,K]              — NVFP4 quantized weight
///   [3] weight_f8_scale FP8 [N,K/16]     — per-block FP8 weight scale
///   [4] weight_f32_scale FP32 []          — global FP32 weight scale (scalar)
///
/// Output: [0] result FP16 [B,N] (allreduced)
///
/// Attributes:
///   tp_size:                tensor parallel world size (must be 2)
class FusedNvfp4GemmAllReducePlugin : public nvinfer1::IPluginV3,
                                      public nvinfer1::IPluginV3OneCore,
                                      public nvinfer1::IPluginV3OneBuild,
                                      public nvinfer1::IPluginV3OneRuntime
{
public:
    FusedNvfp4GemmAllReducePlugin(std::string const& name, int32_t tpSize);
    FusedNvfp4GemmAllReducePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    FusedNvfp4GemmAllReducePlugin() = delete;
    FusedNvfp4GemmAllReducePlugin(FusedNvfp4GemmAllReducePlugin const&) = delete;
    ~FusedNvfp4GemmAllReducePlugin() override;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild
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

    // IPluginV3OneRuntime
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

private:
    bool ensureGemmRunners() noexcept;

    std::string mLayerName;
    std::string mNamespace;
    int32_t mTpSize{2};

    // Per-instance cached weight scale in SfAtom tiled layout. TensorRT owns
    // plugin lifetime and destroys the instance only after its enqueued work is complete.
    uint8_t* mCachedWeightScaleTiled{nullptr};
    int64_t mCachedWeightScaleTiledSize{0};
    int32_t mCachedWeightNumRows{0};
    int32_t mCachedWeightNumKBlocks{0};

    kernels::CuteDslGemmNvFp4Runner* mGemmRunner{nullptr};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

class FusedNvfp4GemmAllReducePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    FusedNvfp4GemmAllReducePluginCreator();
    ~FusedNvfp4GemmAllReducePluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept;
    char const* getPluginNamespace() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFieldCollection;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
