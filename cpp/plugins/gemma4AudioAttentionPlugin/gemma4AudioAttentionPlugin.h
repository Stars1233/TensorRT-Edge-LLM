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
#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{

//! \brief TensorRT plugin for Gemma 4 audio-encoder chunked local attention (IPluginV3).
//!
//! Wraps the fused CUDA kernel that computes the full attention body *after* Q/K/V projection
//! and *before* the output projection: per-dim learned Q scaling, fixed K scaling, chunked
//! local context gather, content + relative-position scores, tanh soft-cap, local-causal +
//! padding mask, fp32 softmax, value mix.
//!
//! Inputs (7):
//!   0: qRaw   [B, S, H, D]  (half/bf16/float) — raw query projections
//!   1: kRaw   [B, S, H, D]  (same type)       — raw key projections
//!   2: v      [B, S, H, D]  (same type)       — value projections
//!   3: gamma  [D]           (float)           — per-dim learned query scale
//!   4: relKey [P, H, D]    (same type as q)   — projected relative-position embeddings
//!   5: valid  [B, S]       (bool)            — audio validity mask (true = real token)
//!   6: seqLen [1]          (int32)           — actual sequence length (shape carrier)
//!
//! Outputs (1):
//!   0: out    [B, S, H, D]  (same type as q)  — attention output
//!
//! Plugin attributes (serialized):
//!   chunk_size    (int32) — C, query block size (default 12)
//!   left_horizon  (int32) — L, effective left context (default 12)
//!   context_size  (int32) — M, gathered K/V context size (default 24)
//!   logit_cap     (float) — tanh soft-cap on logits (default 50.0)
class Gemma4AudioAttentionPlugin : public nvinfer1::IPluginV3,
                                   public nvinfer1::IPluginV3OneCore,
                                   public nvinfer1::IPluginV3OneBuild,
                                   public nvinfer1::IPluginV3OneRuntime
{
public:
    Gemma4AudioAttentionPlugin(
        std::string const& name, int32_t chunkSize, int32_t leftHorizon, int32_t contextSize, float logitCap);
    Gemma4AudioAttentionPlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Gemma4AudioAttentionPlugin() = delete;
    Gemma4AudioAttentionPlugin(Gemma4AudioAttentionPlugin const&) = delete;
    ~Gemma4AudioAttentionPlugin() override;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

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
    std::string mLayerName;
    std::string mNamespace;

    int32_t mChunkSize{12};
    int32_t mLeftHorizon{12};
    int32_t mContextSize{24};
    float mLogitCap{50.0f};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
};

//! \brief Factory class for creating Gemma4AudioAttentionPlugin instances.
class Gemma4AudioAttentionPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Gemma4AudioAttentionPluginCreator();
    ~Gemma4AudioAttentionPluginCreator() override = default;

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
