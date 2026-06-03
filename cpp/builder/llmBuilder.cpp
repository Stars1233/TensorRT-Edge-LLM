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

#include "llmBuilder.h"
#include "builderUtils.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"

#include <cstdlib>
#include <fstream>
#include <string>

using namespace trt_edgellm;

namespace trt_edgellm
{
namespace builder
{
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 13)
namespace
{

void appendLunowudFlag(std::string& flags, std::string const& flag)
{
    if (flags.find(flag) != std::string::npos)
    {
        return;
    }
    if (!flags.empty())
    {
        flags += ' ';
    }
    flags += flag;
}

std::string applyMyelinCompileWorkarounds(int32_t maxBatchSize)
{
    std::string lunowudFlags;
    char const* existingLunowud = std::getenv("__LUNOWUD");
    if (existingLunowud)
    {
        lunowudFlags = existingLunowud;
    }
#if NV_TENSORRT_MAJOR == 10 && (NV_TENSORRT_MINOR == 13 || NV_TENSORRT_MINOR == 14)
    appendLunowudFlag(lunowudFlags, "-peep:match_dual_gemm=off");
#endif
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 15)
    appendLunowudFlag(lunowudFlags, "-mlir:autotune:num_threads=1");
    appendLunowudFlag(lunowudFlags, "-mlir:collective:fp4=off");
    appendLunowudFlag(lunowudFlags, "-cask_fusion:async_policy=1");
    if (maxBatchSize == 1)
    {
        appendLunowudFlag(lunowudFlags, "-peep:fc_h_fusion=off");
    }
#endif
    if (existingLunowud || !lunowudFlags.empty())
    {
        setenv("__LUNOWUD", lunowudFlags.c_str(), 1);
    }
    return lunowudFlags;
}

} // namespace
#endif

LLMBuilder::LLMBuilder(
    std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir, LLMBuilderConfig const& config)
    : mOnnxDir(onnxDir)
    , mEngineDir(engineDir)
    , mBuilderConfig(config)
{
}

bool LLMBuilder::build()
{
#if NV_TENSORRT_MAJOR >= 11 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 13)
    std::string const lunowudFlags = applyMyelinCompileWorkarounds(mBuilderConfig.maxBatchSize);
    if (!lunowudFlags.empty())
    {
        LOG_INFO("Using __LUNOWUD=%s", lunowudFlags.c_str());
    }
#endif

    // Load plugin library
    auto pluginHandles = loadEdgellmPluginLib();

    // Parse model config
    if (!parseConfig())
    {
        return false;
    }

    // Create builder and network
    auto [builder, network] = createBuilderAndNetwork();
    if (!builder || !network)
    {
        return false;
    }

    // Determine ONNX file path
    std::string onnxFilePath;
    if (mBuilderConfig.maxLoraRank > 0)
    {
        onnxFilePath = (mOnnxDir / "lora_model.onnx").string();
        LOG_INFO("Parsing LoRA-enabled ONNX model: %s", onnxFilePath.c_str());
    }
    else
    {
        onnxFilePath = (mOnnxDir / "model.onnx").string();
        LOG_INFO("Parsing ONNX model: %s", onnxFilePath.c_str());
    }

    // Parse ONNX model
    auto parser = parseOnnxModel(network.get(), onnxFilePath);
    if (!parser)
    {
        return false;
    }

    // Print network information
    LOG_DEBUG("%s", printNetworkInfo(network.get(), "LLM").c_str());

    LOG_DEBUG(
        "ONNX parsing complete. mNbKVCacheInputs=%d, mNumLinearAttnLayers=%d", mNbKVCacheInputs, mNumLinearAttnLayers);

    // Create builder config
    auto config = createBuilderConfig(builder.get());
    if (!config)
    {
        return false;
    }

    if (mBuilderConfig.profilingDetailed)
    {
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        LOG_INFO("Profiling verbosity set to DETAILED");
    }

    LOG_DEBUG("Builder config created. Setting up optimization profiles...");

    // Setup optimization profiles
    if (!setupLLMOptimizationProfiles(*builder.get(), *config.get(), *network.get()))
    {
        return false;
    }

    // Create engine directory
    if (!std::filesystem::exists(mEngineDir))
    {
        if (!std::filesystem::create_directories(mEngineDir))
        {
            LOG_ERROR("Failed to create directory %s", mEngineDir.string().c_str());
            return false;
        }
        LOG_INFO("Created directory %s for saving LLM engine.", mEngineDir.string().c_str());
    }

    // Determine engine file name
    std::string engineFileName;
    if (mBuilderConfig.eagleDraft)
    {
        engineFileName = "eagle_draft.engine";
    }
    else if (mBuilderConfig.eagleBase)
    {
        engineFileName = "eagle_base.engine";
    }
    else
    {
        engineFileName = "llm.engine";
    }

    // Build and save engine
    std::string const engineFilePath = (mEngineDir / engineFileName).string();
    if (!buildAndSerializeEngine(builder.get(), network.get(), config.get(), engineFilePath))
    {
        return false;
    }

    // Detect number of deepstack embeds from network (for Qwen3VL models)
    mNumDeepstackFeatures = 0;
    for (int32_t idx = 0; idx < network->getNbInputs(); idx++)
    {
        std::string const inputName = network->getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string::npos)
        {
            mNumDeepstackFeatures++;
        }
    }
    if (mNumDeepstackFeatures > 0)
    {
        LOG_INFO("Detected %d deepstack embedding inputs in network (Qwen3VL model)", mNumDeepstackFeatures);
    }

    // Copy files and save builder config
    if (!copyConfig())
    {
        return false;
    }

    if (!copyTokenizerFiles())
    {
        return false;
    }

    if (!copyEagleFiles())
    {
        return false;
    }

    if (!copyVocabMappingFiles())
    {
        return false;
    }

    if (!copyEmbeddingFile())
    {
        return false;
    }

    if (!copyExternalWeightFiles())
    {
        return false;
    }

    return true;
}

bool LLMBuilder::parseConfig()
{
    std::string const jsonPath = (mOnnxDir / "config.json").string();
    if (!loadJsonConfig(jsonPath, mModelConfig))
    {
        return false;
    }

    // Check model version
    std::string modelVersion = mModelConfig.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    mHiddenSize = mModelConfig["hidden_size"].get<int32_t>();
    // For MTP draft, base model outputs hidden_size (1x); for EAGLE3 draft, it outputs hidden_size * 3.
    std::string const modelType = mModelConfig.value("model_type", "");
    mTargetModelOutputHiddenDim = (modelType == "mtp_draft") ? mHiddenSize : mHiddenSize * 3;
    mNumKVHeads = mModelConfig["num_key_value_heads"].get<int32_t>();
    auto numAttentionHeads = mModelConfig["num_attention_heads"].get<int32_t>();

    if (mModelConfig.contains("head_dim"))
    {
        mHeadSize = mModelConfig["head_dim"].get<int32_t>();
    }
    else
    {
        mHeadSize = mHiddenSize / numAttentionHeads;
    }

    if (mModelConfig.contains("partial_rotary_factor"))
    {
        mRotaryDim = static_cast<int64_t>(mModelConfig["partial_rotary_factor"].get<float>() * mHeadSize);
    }
    else
    {
        mRotaryDim = mHeadSize;
    }

    mNumLinearAttnLayers = mModelConfig.value("num_linear_attn_layers", 0);
    mRecurrentStateNumHeads = mModelConfig.value("recurrent_state_num_heads", 0);
    mRecurrentStateHeadDim = mModelConfig.value("recurrent_state_head_dim", 0);
    mRecurrentStateSize = mModelConfig.value("recurrent_state_size", 0);
    mConvDim = mModelConfig.value("conv_dim", 0);
    mConvKernel = mModelConfig.value("conv_kernel", 0);

    // For hybrid models, only attention layers have KV caches
    if (mNumLinearAttnLayers > 0)
    {
        mNbKVCacheInputs = mModelConfig.value("num_attention_layers", mModelConfig["num_hidden_layers"].get<int32_t>());
    }
    else
    {
        mNbKVCacheInputs = mModelConfig["num_hidden_layers"].get<int32_t>();
    }

    // Read trt_native_ops flag from config if present
    if (mModelConfig.contains("trt_native_ops"))
    {
        mBuilderConfig.useTrtNativeOps = mModelConfig["trt_native_ops"].get<bool>();
    }

    return true;
}

bool LLMBuilder::setupLLMOptimizationProfiles(
    nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network)
{
    auto* contextProfile = builder.createOptimizationProfile();
    auto* generationProfile = builder.createOptimizationProfile();

    bool result = true;

    // Setup common profiles
    result &= setupCommonProfiles(*contextProfile, *generationProfile);

    // Setup model-specific profiles
    if (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft)
    {
        result &= setupEagleProfiles(*contextProfile, *generationProfile);
    }
    else
    {
        result &= setupVanillaProfiles(*contextProfile, *generationProfile);
    }

    // Setup intermediate state profiles for MTP base models
    std::string const modelType = mModelConfig.value("model_type", "");
    if (modelType == "mtp_base")
    {
        result &= setupIntermediateRecurrentStateProfiles(*contextProfile, *generationProfile);
        result &= setupIntermediateConvStateProfiles(*contextProfile, *generationProfile);
    }

    // Setup Deepstack profiles for Qwen3VL models
    result &= setupDeepstackProfiles(*contextProfile, *generationProfile, network);

    // Setup lm_head_weight profile for CodePredictor (Qwen3-Omni)
    result &= setupLmHeadWeightProfiles(*contextProfile, *generationProfile, network);

    if (mBuilderConfig.maxLoraRank > 0)
    {
        result &= setupLoraProfiles(*contextProfile, *generationProfile, network);
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles");
        return false;
    }

    LOG_DEBUG("%s", printOptimizationProfile(contextProfile, "context_profile", &network).c_str());
    LOG_DEBUG("%s", printOptimizationProfile(generationProfile, "generation_profile", &network).c_str());

    config.addOptimizationProfile(contextProfile);
    config.addOptimizationProfile(generationProfile);

    return true;
}

bool LLMBuilder::setupCommonProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    // Context lengths
    result &= setOptimizationProfile(&contextProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kContextLengths, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // Rope rotary cos sin
    result &= setOptimizationProfile(&contextProfile, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kRopeCosSin,
        createDims({1, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxKVCacheCapacity, mRotaryDim}));

    // For KVCacheStartIndex, we use zero shape to indicate the kvcache is empty for all sequences in the batch.
    // This can help distinguish the normal prefill and chunked prefill execution.
    result &= setOptimizationProfile(&contextProfile, binding_names::kKVCacheStartIndex, createDims({0}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kKVCacheStartIndex, createDims({1}),
        createDims({mBuilderConfig.maxBatchSize}), createDims({mBuilderConfig.maxBatchSize}));

    // KV cache profiles
    LOG_DEBUG("Setting up KV cache profiles for %d layers...", mNbKVCacheInputs);
    result &= setupKVCacheProfiles(contextProfile, generationProfile);
    LOG_DEBUG("KV cache profiles done. Setting up recurrent state profiles for %d layers...", mNumLinearAttnLayers);

    // Recurrent state profiles for hybrid layers
    result &= setupRecurrentStateProfiles(&contextProfile, &generationProfile);

    LOG_DEBUG("Recurrent state profiles done. Setting up Conv state profiles...");
    // Conv state profiles for recurrent causal conv1d layers
    result &= setupConvStateProfiles(&contextProfile, &generationProfile);
    LOG_DEBUG("Conv state profiles done.");

    return result;
}

bool LLMBuilder::setupVanillaProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;

    // Input embeddings - always dynamic
    result &= setOptimizationProfile(&contextProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}));

    // Last token IDs
    result &= setOptimizationProfile(&contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));

    return result;
}

bool LLMBuilder::setupEagleProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    // TRT-native-ops + EAGLE is a partially-wired path: the Python export
    // emits a 4D bool `attention_mask` [batch, 1, seq_len, seq_len + past_len]
    // (see `llm_model_trtnative.py` with `is_eagle_base=True`), and the
    // `prepareEagleBaseTreeDecodingInputsTrtNative` kernel exists
    // (`cpp/kernels/speculative/eagleUtilKernels.{h,cu}`). However the
    // builder's EAGLE profile setup below and the runtime dispatch in
    // `EagleDecoder::runBaseModelVerification` both
    // hardcode the plugin-path 3D packed-INT32 mask layout. Attempting to
    // build this combination produces a cryptic TRT error:
    //
    //   "Dynamic-shaped input tensor attention_mask has 4 dimensions but
    //    profile 0 has 3 dimensions"
    //
    // Fail fast with an actionable message until the full TRT-native+EAGLE
    // path (builder profile + registry + runtime dispatch) is completed.
    if (mBuilderConfig.useTrtNativeOps && (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft))
    {
        LOG_ERROR(
            "TRT-native-ops + EAGLE speculative decoding is not yet supported. "
            "Re-export the engine ONNX with trt_native_ops=False (plugin attention path).");
        return false;
    }

    bool result = true;

    int const maxTokens
        = mBuilderConfig.eagleDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;

    // Input embeddings
    result &= setOptimizationProfile(&contextProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kInputsEmbeds, createDims({1, 1, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

    // Last token IDs - 2D shape [batch_size, num_selected_tokens]
    result &= setOptimizationProfile(&contextProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLastTokenIds, createDims({1, 1}),
        createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}), createDims({mBuilderConfig.maxBatchSize, maxTokens}));

    if (mBuilderConfig.eagleDraft)
    {
        // Hidden states from draft
        result &= setOptimizationProfile(&contextProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kDraftModelHiddenStates,
            createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));

        // Hidden states input
        result &= setOptimizationProfile(&contextProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mTargetModelOutputHiddenDim}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kBaseModelHiddenStates,
            createDims({1, 1, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mTargetModelOutputHiddenDim}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens, mTargetModelOutputHiddenDim}));
    }

    // Attention mask and position ID
    if (mBuilderConfig.eagleDraft || mBuilderConfig.eagleBase)
    {
        int32_t const attnMaskAlignSize = 32;
        result &= setOptimizationProfile(&contextProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1, 1}), createDims({mBuilderConfig.maxBatchSize, 1, 1}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kAttentionMask, createDims({1, 1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2,
                static_cast<int64_t>(divUp(maxTokens / 2, attnMaskAlignSize) * attnMaskAlignSize)}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens,
                static_cast<int64_t>(divUp(maxTokens, attnMaskAlignSize) * attnMaskAlignSize)}));

        result &= setOptimizationProfile(&contextProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, 1}), createDims({mBuilderConfig.maxBatchSize, 1}));
        result &= setOptimizationProfile(&generationProfile, binding_names::kAttentionPosId, createDims({1, 1}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens / 2}),
            createDims({mBuilderConfig.maxBatchSize, maxTokens}));
    }

    return result;
}

bool LLMBuilder::setupDeepstackProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Dynamically detect all deepstack_embeds inputs in the network
    std::vector<std::string> deepstackInputs;
    for (int32_t idx = 0; idx < network.getNbInputs(); idx++)
    {
        std::string const inputName = network.getInput(idx)->getName();
        if (inputName.find(binding_names::kDeepstackEmbedsTemplate) != std::string::npos)
        {
            deepstackInputs.push_back(inputName);
        }
    }

    // If no deepstack embeds found, return early (not a Qwen3VL model)
    if (deepstackInputs.empty())
    {
        return true;
    }

    LOG_INFO("Detected %zu deepstack embedding inputs", deepstackInputs.size());

    // Setup profiles for all detected deepstack_embeds inputs
    // These have the same shape as inputs_embeds: [batch_size, seq_len, hidden_size]
    for (auto const& deepstackInputName : deepstackInputs)
    {
        LOG_INFO("Setting up optimization profile for %s", deepstackInputName.c_str());

        // Same profile as inputs_embeds
        result &= setOptimizationProfile(&contextProfile, deepstackInputName.c_str(), createDims({1, 1, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mHiddenSize}),
            createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mHiddenSize}));

        if (mBuilderConfig.eagleBase || mBuilderConfig.eagleDraft)
        {
            int const maxTokens
                = mBuilderConfig.eagleDraft ? mBuilderConfig.maxDraftTreeSize : mBuilderConfig.maxVerifyTreeSize;
            result &= setOptimizationProfile(&generationProfile, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, maxTokens / 2, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, maxTokens, mHiddenSize}));
        }
        else
        {
            result &= setOptimizationProfile(&generationProfile, deepstackInputName.c_str(),
                createDims({1, 1, mHiddenSize}), createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}),
                createDims({mBuilderConfig.maxBatchSize, 1, mHiddenSize}));
        }
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupDeepstackProfiles().");
    }

    return result;
}

bool LLMBuilder::setupLmHeadWeightProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;

    // Detect if lm_head_weight input exists (CodePredictor model)
    bool hasLmHeadWeight = false;
    for (int32_t idx = 0; idx < network.getNbInputs(); idx++)
    {
        std::string const inputName = network.getInput(idx)->getName();
        if (inputName == binding_names::kLmHeadWeight)
        {
            hasLmHeadWeight = true;
            break;
        }
    }

    // If no lm_head_weight input found, return early (not a CodePredictor model)
    if (!hasLmHeadWeight)
    {
        return true;
    }

    LOG_INFO("Detected lm_head_weight input (CodePredictor model)");

    // lm_head_weight shape: [vocab_size, hidden_size]
    // For CodePredictor: vocab_size=2048 (codebook size), hidden_size=1024
    // This is a fixed-size weight tensor that gets bound at runtime
    int64_t const vocabSize = mModelConfig["vocab_size"].get<int64_t>();
    int64_t const hiddenSize = mHiddenSize;

    // Both context and generation profiles use the same shape since this is a weight tensor
    result &= setOptimizationProfile(&contextProfile, binding_names::kLmHeadWeight, createDims({vocabSize, hiddenSize}),
        createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}));
    result &= setOptimizationProfile(&generationProfile, binding_names::kLmHeadWeight,
        createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}), createDims({vocabSize, hiddenSize}));

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles for lm_head_weight.");
    }

    return result;
}

bool LLMBuilder::setupLoraProfiles(nvinfer1::IOptimizationProfile& contextProfile,
    nvinfer1::IOptimizationProfile& generationProfile, nvinfer1::INetworkDefinition const& network)
{
    bool result = true;
    if (mBuilderConfig.maxLoraRank == 0)
    {
        LOG_WARNING(
            "Your model has dynamic LoRA, but max LoRA rank is 0. This is equivalent to no LoRA. Please set "
            "--maxLoraRank to a positive value if you want to use LoRA.");
        return true;
    }

    bool findLoraWeights = false;

    for (int i = 0; i < network.getNbInputs(); ++i)
    {
        auto* input = network.getInput(i);
        std::string inputName = input->getName();

        if (inputName.find(binding_names::kLoraAPrefix) != std::string::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_A, the shape is [gemm_k, lora_rank]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_k = dims.d[0];
                result
                    &= setOptimizationProfile(&contextProfile, inputName.c_str(), createDims({gemm_k, 0}), // min shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}),                              // opt shape
                        createDims({gemm_k, mBuilderConfig.maxLoraRank}));                                 // max shape
                result &= setOptimizationProfile(&generationProfile, inputName.c_str(),
                    createDims({gemm_k, 0}),                              // min shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank / 2}), // opt shape
                    createDims({gemm_k, mBuilderConfig.maxLoraRank}));    // max shape
            }
        }
        else if (inputName.find(binding_names::kLoraBPrefix) != std::string::npos)
        {
            if (!findLoraWeights)
            {
                findLoraWeights = true;
            }
            // For lora_B, the shape is [lora_rank, gemm_n]
            auto dims = input->getDimensions();
            if (dims.nbDims == 2)
            {
                int64_t gemm_n = dims.d[1];
                result
                    &= setOptimizationProfile(&contextProfile, inputName.c_str(), createDims({0, gemm_n}), // min shape
                        createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}),                              // opt shape
                        createDims({mBuilderConfig.maxLoraRank, gemm_n}));                                 // max shape
                result &= setOptimizationProfile(&generationProfile, inputName.c_str(),
                    createDims({0, gemm_n}),                              // min shape
                    createDims({mBuilderConfig.maxLoraRank / 2, gemm_n}), // opt shape
                    createDims({mBuilderConfig.maxLoraRank, gemm_n}));    // max shape
            }
        }
    }

    if (!findLoraWeights)
    {
        LOG_ERROR(
            "Failed to find any LoRA weights inputs in the ONNX model. Have you inserted LoRA weights using "
            "tensorrt-edgellm-insert-lora command?");
        return false;
    }

    if (!result)
    {
        LOG_ERROR("Failed to setup optimization profiles at setupLoraProfiles().");
    }

    return result;
}

bool LLMBuilder::setupKVCacheProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    bool result = true;
    if (mBuilderConfig.useTrtNativeOps)
    {
        // TRT attention: separate K and V caches without the "2" dimension
        // Shape: [batch, num_kv_heads, seq_len, head_dim]
        nvinfer1::Dims minKVCacheShape = createDims({1, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});
        nvinfer1::Dims optKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});
        nvinfer1::Dims maxKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});

        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            // K cache bindings
            result &= setOptimizationProfile(&contextProfile, binding_names::formatKCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
            result &= setOptimizationProfile(&generationProfile, binding_names::formatKCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);

            // V cache bindings
            result &= setOptimizationProfile(&contextProfile, binding_names::formatVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
            result &= setOptimizationProfile(&generationProfile, binding_names::formatVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
        }
    }
    else
    {
        // Plugin path: combined KV cache with "2" dimension
        // KV cache shape is [B, 2, num_kv_heads, 0 to max_kv_cache_capacity, head_dim]
        nvinfer1::Dims minKVCacheShape = createDims({1, 2, mNumKVHeads, 0, mHeadSize});
        nvinfer1::Dims optKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, 2, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});
        nvinfer1::Dims maxKVCacheShape
            = createDims({mBuilderConfig.maxBatchSize, 2, mNumKVHeads, mBuilderConfig.maxKVCacheCapacity, mHeadSize});

        for (int i = 0; i < mNbKVCacheInputs; ++i)
        {
            result &= setOptimizationProfile(&contextProfile, binding_names::formatKVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
            result &= setOptimizationProfile(&generationProfile, binding_names::formatKVCacheName(i, true).c_str(),
                minKVCacheShape, optKVCacheShape, maxKVCacheShape);
        }
    }

    return result;
}

bool LLMBuilder::setupRecurrentStateProfiles(
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    if (mNumLinearAttnLayers == 0)
    {
        return true;
    }

    bool result = true;

    // Recurrent state shape: [batch, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
    nvinfer1::Dims minRecurrentShape
        = createDims({1, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims optRecurrentShape = createDims(
        {mBuilderConfig.maxBatchSize, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims maxRecurrentShape = createDims(
        {mBuilderConfig.maxBatchSize, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const recurrentStateName = binding_names::formatRecurrentStateName(i, /*isPast=*/true);
        result &= setOptimizationProfile(
            contextProfile, recurrentStateName.c_str(), minRecurrentShape, optRecurrentShape, maxRecurrentShape);
        result &= setOptimizationProfile(
            generationProfile, recurrentStateName.c_str(), minRecurrentShape, optRecurrentShape, maxRecurrentShape);
    }

    LOG_DEBUG("Set up recurrent state optimization profiles for %d recurrent layers", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupConvStateProfiles(
    nvinfer1::IOptimizationProfile* const contextProfile, nvinfer1::IOptimizationProfile* const generationProfile)
{
    if (mNumLinearAttnLayers == 0 || mConvDim == 0 || mConvKernel == 0)
    {
        return true;
    }

    bool result = true;

    // Conv state shape: [batch, conv_dim, conv_kernel]
    nvinfer1::Dims minConvShape = createDims({1, mConvDim, mConvKernel});
    nvinfer1::Dims optConvShape = createDims({mBuilderConfig.maxBatchSize, mConvDim, mConvKernel});
    nvinfer1::Dims maxConvShape = createDims({mBuilderConfig.maxBatchSize, mConvDim, mConvKernel});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const convStateName = binding_names::formatConvStateName(i, /*isPast=*/true);
        result
            &= setOptimizationProfile(contextProfile, convStateName.c_str(), minConvShape, optConvShape, maxConvShape);
        result &= setOptimizationProfile(
            generationProfile, convStateName.c_str(), minConvShape, optConvShape, maxConvShape);
    }

    LOG_DEBUG("Set up conv state optimization profiles for %d recurrent layers", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupIntermediateRecurrentStateProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    if (mNumLinearAttnLayers == 0)
    {
        return true;
    }

    bool result = true;

    // Intermediate recurrent state shape: [batch, seq_len, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
    nvinfer1::Dims minShape = createDims({1, 1, mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims optCtxShape = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2,
        mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims maxCtxShape = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen,
        mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims optGenShape = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxVerifyTreeSize / 2,
        mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});
    nvinfer1::Dims maxGenShape = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxVerifyTreeSize,
        mRecurrentStateNumHeads, mRecurrentStateHeadDim, mRecurrentStateSize});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const name = binding_names::formatIntermediateRecurrentStateName(i);
        result &= setOptimizationProfile(&contextProfile, name.c_str(), minShape, optCtxShape, maxCtxShape);
        result &= setOptimizationProfile(&generationProfile, name.c_str(), minShape, optGenShape, maxGenShape);
    }

    LOG_DEBUG("Set up intermediate recurrent state profiles for %d recurrent layers (MTP)", mNumLinearAttnLayers);
    return result;
}

bool LLMBuilder::setupIntermediateConvStateProfiles(
    nvinfer1::IOptimizationProfile& contextProfile, nvinfer1::IOptimizationProfile& generationProfile)
{
    if (mNumLinearAttnLayers == 0 || mConvDim == 0 || mConvKernel == 0)
    {
        return true;
    }

    bool result = true;

    // Intermediate conv state shape: [batch, seq_len, conv_dim, conv_kernel]
    nvinfer1::Dims minShape = createDims({1, 1, mConvDim, mConvKernel});
    nvinfer1::Dims optCtxShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen / 2, mConvDim, mConvKernel});
    nvinfer1::Dims maxCtxShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxInputLen, mConvDim, mConvKernel});
    nvinfer1::Dims optGenShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxVerifyTreeSize / 2, mConvDim, mConvKernel});
    nvinfer1::Dims maxGenShape
        = createDims({mBuilderConfig.maxBatchSize, mBuilderConfig.maxVerifyTreeSize, mConvDim, mConvKernel});

    for (int32_t i = 0; i < mNumLinearAttnLayers; ++i)
    {
        std::string const name = binding_names::formatIntermediateConvStateName(i);
        result &= setOptimizationProfile(&contextProfile, name.c_str(), minShape, optCtxShape, maxCtxShape);
        result &= setOptimizationProfile(&generationProfile, name.c_str(), minShape, optGenShape, maxGenShape);
    }

    LOG_DEBUG("Set up intermediate conv state profiles for %d recurrent layers (MTP)", mNumLinearAttnLayers);
    return result;
}

namespace
{
// EAGLE base and draft engines share one engineDir, so their external weight
// files (e.g. external_int4_ffn_weights.safetensors) would otherwise overwrite
// each other. Prefix only the draft files; the base keeps the original names
// (matching the standalone non-EAGLE case), which is enough to avoid the
// collision. copyConfig() and copyExternalWeightFiles() must agree on this name.
std::string externalWeightDstName(std::string const& filename, bool eagleDraft)
{
    if (eagleDraft)
    {
        return "draft_" + filename;
    }
    return filename;
}
} // namespace

bool LLMBuilder::copyConfig()
{
    // Determine config file name based on model type
    std::string configFileName;
    if (mBuilderConfig.eagleDraft)
    {
        configFileName = "draft_config.json";
    }
    else if (mBuilderConfig.eagleBase)
    {
        configFileName = "base_config.json";
    }
    else
    {
        configFileName = "config.json";
    }

    std::string const targetConfigPath = (mEngineDir / configFileName).string();

    // Create a copy of mModelConfig and add builder config
    Json configWithBuilder = mModelConfig;
    configWithBuilder["builder_config"] = mBuilderConfig.toJson();

    // Keep external weight file references in sync with the names written by
    // copyExternalWeightFiles() (draft files get the "draft_" prefix) so the runtime loads the right file.
    if (configWithBuilder.contains("external_weight_files") && configWithBuilder["external_weight_files"].is_array())
    {
        for (auto& fileEntry : configWithBuilder["external_weight_files"])
        {
            if (fileEntry.is_object() && fileEntry.contains("file") && fileEntry["file"].is_string())
            {
                fileEntry["file"]
                    = externalWeightDstName(fileEntry["file"].get<std::string>(), mBuilderConfig.eagleDraft);
            }
        }
    }

    // Add detected num_deepstack_features if present (Qwen3VL models)
    configWithBuilder["num_deepstack_features"] = mNumDeepstackFeatures;

    // Write updated config
    std::ofstream targetConfigFile(targetConfigPath);
    if (!targetConfigFile.is_open())
    {
        LOG_ERROR("Failed to open target config file: %s", targetConfigPath.c_str());
        return false;
    }
    targetConfigFile << configWithBuilder.dump(2);
    targetConfigFile.close();

    LOG_INFO("Copied config.json with builder config to %s", targetConfigPath.c_str());
    return true;
}

bool LLMBuilder::copyTokenizerFiles()
{
    // Eagle3 draft model does not need tokenizer files
    if (mBuilderConfig.eagleDraft)
    {
        return true;
    }

    // Models that use embeddings as input (e.g., Talker, CodePredictor) don't need tokenizer
    bool useEmbeddingsInput = mModelConfig.value("use_embeddings_input", false);
    if (useEmbeddingsInput)
    {
        LOG_INFO("Skipping tokenizer files (model uses embeddings input)");
        return true;
    }

    std::vector<std::string> tokenizerFiles
        = {"tokenizer_config.json", "tokenizer.json", "processed_chat_template.json"};
    bool allSuccess = true;

    for (auto const& filename : tokenizerFiles)
    {
        std::string const srcPath = (mOnnxDir / filename).string();
        std::string const dstPath = (mEngineDir / filename).string();

        if (file_io::copyFile(srcPath, dstPath))
        {
            LOG_INFO("Copied tokenizer file: %s", filename.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy tokenizer file %s", filename.c_str());
            allSuccess = false;
        }
    }

    return allSuccess;
}

bool LLMBuilder::copyEagleFiles()
{
    // Copy d2t.safetensors for Eagle3 draft models only. MTP draft shares vocab with base and has no d2t mapping.
    std::string const modelType = mModelConfig.value("model_type", "");
    if (mBuilderConfig.eagleDraft && modelType != "mtp_draft")
    {
        std::string const d2tPath = (mOnnxDir / "d2t.safetensors").string();
        std::string const targetD2tPath = (mEngineDir / "d2t.safetensors").string();

        if (file_io::copyFile(d2tPath, targetD2tPath))
        {
            LOG_INFO("Copied d2t.safetensors to %s", targetD2tPath.c_str());
        }
        else
        {
            LOG_WARNING("Failed to copy d2t.safetensors to %s", targetD2tPath.c_str());
            return false;
        }
    }

    return true;
}

bool LLMBuilder::copyVocabMappingFiles()
{
    // Copy vocab_map.safetensors if reduced vocabulary is used
    if (mModelConfig.contains(binding_names::kReducedVocabSizeKey)
        && mModelConfig[binding_names::kReducedVocabSizeKey].get<int32_t>() > 0)
    {
        std::string const vocabMapPath = (mOnnxDir / binding_names::kVocabMapFileName).string();
        std::string const targetVocabMapPath = (mEngineDir / binding_names::kVocabMapFileName).string();

        if (file_io::copyFile(vocabMapPath, targetVocabMapPath))
        {
            LOG_INFO("Copied %s to %s", binding_names::kVocabMapFileName, targetVocabMapPath.c_str());
        }
        else
        {
            LOG_WARNING("%s not found in %s. This is expected if reduced vocabulary is not used.",
                binding_names::kVocabMapFileName, mOnnxDir.string().c_str());
        }
    }

    return true;
}

bool LLMBuilder::copyEmbeddingFile()
{
    // Eagle draft model uses shared embedding table from base model, so skip copying
    if (mBuilderConfig.eagleDraft)
    {
        return true;
    }

    // Check if this is a Talker model (has text_projection.safetensors)
    std::filesystem::path const textProjectionPath = mOnnxDir / "text_projection.safetensors";
    if (std::filesystem::exists(textProjectionPath))
    {
        // Talker: copy embedding + text_projection + hidden_projection (optional, text-only TTS omits it)
        LOG_INFO("Detected Talker model, copying projection files...");

        std::vector<std::string> requiredFiles = {"embedding.safetensors", "text_projection.safetensors"};
        std::vector<std::string> optionalFiles = {"text_embedding.safetensors", "hidden_projection.safetensors"};

        bool allSuccess = true;
        for (auto const& filename : requiredFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();

            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_ERROR("Failed to copy %s", filename.c_str());
                allSuccess = false;
            }
        }
        for (auto const& filename : optionalFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();

            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_INFO("Optional %s not found, skipping", filename.c_str());
            }
        }

        return allSuccess;
    }

    // Check if this is a CodePredictor model (has codec_embeddings.safetensors)
    std::filesystem::path const codecEmbedPath = mOnnxDir / "codec_embeddings.safetensors";
    if (std::filesystem::exists(codecEmbedPath))
    {
        LOG_INFO("Detected CodePredictor model, copying codec files...");
        std::vector<std::string> cpRequiredFiles = {"codec_embeddings.safetensors", "lm_heads.safetensors"};
        std::vector<std::string> cpOptionalFiles = {"small_to_mtp_projection.safetensors"};
        bool allSuccess = true;
        for (auto const& filename : cpRequiredFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();
            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_ERROR("Failed to copy required CodePredictor file: %s", filename.c_str());
                allSuccess = false;
            }
        }
        for (auto const& filename : cpOptionalFiles)
        {
            std::string const srcPath = (mOnnxDir / filename).string();
            std::string const dstPath = (mEngineDir / filename).string();
            if (file_io::copyFile(srcPath, dstPath))
            {
                LOG_INFO("Copied %s", filename.c_str());
            }
            else
            {
                LOG_INFO("Optional %s not found, skipping", filename.c_str());
            }
        }
        return allSuccess;
    }

    // Copy embedding.safetensors for vanilla LLM models
    std::string const embeddingPath = (mOnnxDir / "embedding.safetensors").string();
    std::string const targetEmbeddingPath = (mEngineDir / "embedding.safetensors").string();

    if (file_io::copyFile(embeddingPath, targetEmbeddingPath))
    {
        LOG_INFO("Copied embedding.safetensors to %s", targetEmbeddingPath.c_str());
    }
    else
    {
        LOG_ERROR(
            "Failed to copy embedding.safetensors from %s to %s", embeddingPath.c_str(), targetEmbeddingPath.c_str());
        return false;
    }

    return true;
}

bool LLMBuilder::copyExternalWeightFiles()
{
    Json const externalWeightFiles = mModelConfig.value("external_weight_files", Json::array());
    if (!externalWeightFiles.is_array())
    {
        LOG_ERROR("external_weight_files must be an array when present in config.json");
        return false;
    }
    if (externalWeightFiles.empty())
    {
        return true;
    }

    bool allSuccess = true;
    for (auto const& fileEntry : externalWeightFiles)
    {
        if (!fileEntry.is_object() || !fileEntry.contains("file") || !fileEntry["file"].is_string())
        {
            LOG_ERROR("Malformed external weight file entry: %s", fileEntry.dump().c_str());
            return false;
        }
        std::string const filename = fileEntry["file"].get<std::string>();
        std::string const dstFilename = externalWeightDstName(filename, mBuilderConfig.eagleDraft);
        std::filesystem::path const srcPath = mOnnxDir / filename;
        std::filesystem::path const dstPath = mEngineDir / dstFilename;

        if (file_io::copyFile(srcPath.string(), dstPath.string()))
        {
            LOG_INFO("Copied external weight file: %s -> %s", filename.c_str(), dstFilename.c_str());
        }
        else
        {
            LOG_ERROR("Failed to copy external weight file %s from %s to %s", filename.c_str(),
                srcPath.string().c_str(), dstPath.string().c_str());
            allSuccess = false;
        }
    }
    return allSuccess;
}

} // namespace builder
} // namespace trt_edgellm
