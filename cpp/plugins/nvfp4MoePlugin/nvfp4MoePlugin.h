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

#include "common/tensor.h"
#include "kernels/moe/NvFP4MoEContiguousGemmRunner.h"
#include "kernels/moe/NvFP4MoEFC2FinalizeRunner.h"
#include "kernels/moe/fp4SupportKernels/nvfp4MoeTypes.h"
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
#include "kernels/moe/nvfp4_cutedsl/cuteDslDecodeGemvRunner.h"
#endif

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace plugins
{
//! Router selection kernel chosen by the \c routing_mode plugin attribute.
enum class Nvfp4MoeRoutingMode : int32_t
{
    kSOFTMAX_TOPK = 0,       //!< \c moeTopkSoftmax: softmax over experts + flat top-k + renormalize (default).
    kSIGMOID_GROUP_TOPK = 1, //!< \c moeSigmoidGroupTopk: sigmoid + grouped top-k + renormalize + scale (NemotronH).
};

/*!
 * @brief TensorRT plugin: Nemotron-style MoE MLP with NVFP4 weights.
 *
 * Per expert: \c y_e = down_proj( act( up_proj(x) ) ). No separate gate projection.
 *
 * Dispatch: \c enqueue() picks between two execution paths based on runtime token count
 * (\c numTokens = B × S):
 *   - \c numTokens ≤ 16 — decode path (router + W4A16 GEMV, per-token GEMV kernels).
 *   - \c numTokens > 16 — prefill path (router + GPU layout build + fp4Quantize +
 *     gather + FC1 grouped GEMM + fp4Quantize + FC2 grouped GEMM + scatter-reduce).
 *
 * Routing: both paths consume pre-activation router logits \c [num_tokens, num_experts] and
 * dispatch to one of two router kernels, selected by the \c routing_mode attribute:
 *   - \c 0 (\c kSOFTMAX_TOPK, default): \c moeTopkSoftmax (softmax + flat top-k + renormalize).
 *   - \c 1 (\c kSIGMOID_GROUP_TOPK): \c moeSigmoidGroupTopk (sigmoid + grouped top-k +
 *     renormalize + scale). Uses \c n_group / \c topk_group / \c norm_topk_prob /
 *     \c routed_scaling_factor attributes.
 * \c e_score_correction_bias \c [E] FP32 is used as an optional bias by \c moeTopkSoftmax
 * (mode 0) and as the expert load-balancing bias by \c moeSigmoidGroupTopk (mode 1); pass
 * zeros when no bias is desired.
 *
 * Hidden activations are FP16 only. Any activation NVFP4 quantization (payload + FP8 block
 * scales) needed by the prefill path is computed inside the plugin via \c fp4Quantize; the
 * caller supplies only calibrated global scales.
 */
class Nvfp4MoePlugin : public nvinfer1::IPluginV3,
                       public nvinfer1::IPluginV3OneCore,
                       public nvinfer1::IPluginV3OneBuild,
                       public nvinfer1::IPluginV3OneRuntime
{
public:
    //! Build-phase constructor — populated from ONNX attributes. \c mMaxTokens starts at 0
    //! and is set in \c configurePlugin().
    Nvfp4MoePlugin(std::string const& name, int32_t numExperts, int32_t topK, int32_t hiddenSize, int32_t moeInterSize,
        nvinfer1::ActivationType activationType = static_cast<nvinfer1::ActivationType>(0), int32_t nGroup = 1,
        int32_t topkGroup = 1, int32_t normTopkProb = 1, float routedScalingFactor = 1.0f,
        int32_t routingMode = static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK));

    //! Runtime deserialization constructor. Reads \c max_tokens from the serialized field
    //! collection; a missing field throws.
    Nvfp4MoePlugin(std::string const& name, nvinfer1::PluginFieldCollection const* fc);

    Nvfp4MoePlugin() = delete;
    Nvfp4MoePlugin(Nvfp4MoePlugin const&) = delete;

    ~Nvfp4MoePlugin() noexcept override;

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
    //! Router + CuTe DSL decode GEMV kernels (N-major NVFP4 weights + FP8 block scales + FP32 global scales).
    int32_t enqueueDecoding(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    //! Top-k softmax → GPU layout → fp4Quantize → gather → FC1 GEMM → fp4Quantize → FC2 GEMM+scatter.
    int32_t enqueuePrefill(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    std::string mLayerName;
    //! Empty; ONNX uses domain ``trt`` (``trt::Nvfp4MoePlugin``).
    std::string mNamespace;
    int32_t mNumExperts{};
    int32_t mTopK{};
    int32_t mHiddenSize{};
    int32_t mMoeInterSize{};
    nvinfer1::ActivationType mActivationType{};
    //! Marlin NVFP4 block scales along \c K; only \c 16 is supported.
    int32_t mQuantizationGroupSize{};
    //! NemotronH sigmoid group top-k routing parameters (used when \c mRoutingMode == \c kSIGMOID_GROUP_TOPK).
    int32_t mNGroup{1};
    int32_t mTopkGroup{1};
    int32_t mNormTopkProb{1}; //!< Stored as int32 for serialization; nonzero = true.
    float mRoutedScalingFactor{1.0f};
    //! Router selection kernel: see \c Nvfp4MoeRoutingMode.
    int32_t mRoutingMode{static_cast<int32_t>(Nvfp4MoeRoutingMode::kSOFTMAX_TOPK)};

    //! Max \c B·S captured in \c configurePlugin (union over profiles). Serialized to the
    //! engine so the runtime ctor can size layout buffers.
    int32_t mMaxTokens{0};

    //! Runners for the prefill path. Constructed lazily in \c attachToContext once
    //! \c mMaxTokens / shape parameters are known.
    std::unique_ptr<trt_edgellm::kernel::nvfp4_moe::NvFP4MoEContiguousGemmRunner> mFC1Runner;
    std::unique_ptr<trt_edgellm::kernel::nvfp4_moe::NvFP4MoEFC2FinalizeRunner> mFC2Runner;

    //! Runner for the CuTe DSL decode GEMV path.
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    trt_edgellm::CuteDslDecodeGemvRunner mDecodeGemvRunner{};
#endif

    //! Pre-allocated GPU layout buffers (non-owning \c rt::Tensor views over
    //! \c IGpuAllocator -allocated memory).
    trt_edgellm::kernel::MoELayoutBuffers mLayoutBuffers{};

    //! Persistent FC1/FC2 α buffers (``[L]`` FP32 each), allocated in
    //! \c attachToContext and populated once on the first prefill enqueue.
    //! α depends only on constant plugin inputs (``hidden_global_scale`` and
    //! per-expert ``fc_up_global_scale`` / ``fc_down_global_scale``), so
    //! recomputing every call would be wasted work.
    trt_edgellm::rt::Tensor mFC1Alpha{};
    trt_edgellm::rt::Tensor mFC2Alpha{};
    bool mAlphaInitialized{false};

    //! Allocator captured at \c attachToContext — used to free layout buffers in dtor.
    nvinfer1::IGpuAllocator* mGpuAllocator{nullptr};

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class Nvfp4MoePluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    Nvfp4MoePluginCreator();
    ~Nvfp4MoePluginCreator() override = default;

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
    //! Empty; ONNX domain ``trt`` is separate from creator namespace.
    std::string mNamespace;
};

} // namespace plugins
} // namespace trt_edgellm
