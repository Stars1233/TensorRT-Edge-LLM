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

#include "runtime/config/llmEngineConfig.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "common/version.h"
#include "runtime/exec/engineExecutor.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;

namespace
{

//! Helper: read a required field or throw.
template <typename T>
T getRequired(Json const& json, char const* key)
{
    ELLM_CHECK(json.contains(key), std::string("parseEngineConfig: missing required field '") + key + "'");
    return json[key].get<T>();
}

//! Helper: validate that a value is positive.
void requirePositive(int32_t value, char const* fieldName)
{
    ELLM_CHECK(value > 0,
        std::string("parseEngineConfig: invalid ") + fieldName + ": " + std::to_string(value) + " (must be positive)");
}

//! Parse a lowercase dtype token (e.g. from `kv_cache_dtype`, `recurrent_state_dtype`,
//! `conv_state_dtype`) into an `nvinfer1::DataType`.
//!
//! Accepted tokens and their mapping (matches the export pipeline's writer):
//!   "fp16" → kHALF, "fp8" → kFP8, "int8" → kINT8, "bf16" → kBF16, "fp32" → kFLOAT.
//! `fp32` is required for Qwen3.5 GDN recurrent states (the architecture's SSM
//! kernels run in FP32 for numerical stability), so the export writes
//! `recurrent_state_dtype="fp32"` for that model family.
//! Anything else throws with the offending value, the originating field name,
//! and the allowed set.
nvinfer1::DataType parseStateDtype(std::string const& token, char const* fieldName)
{
    if (token == "fp16")
    {
        return nvinfer1::DataType::kHALF;
    }
    if (token == "fp8")
    {
        return nvinfer1::DataType::kFP8;
    }
    if (token == "int8")
    {
        return nvinfer1::DataType::kINT8;
    }
    if (token == "bf16")
    {
        return nvinfer1::DataType::kBF16;
    }
    if (token == "fp32")
    {
        return nvinfer1::DataType::kFLOAT;
    }
    throw std::runtime_error(std::string("parseEngineConfig: invalid ") + fieldName + " '" + token
        + "'. Allowed values: fp16, fp8, int8, bf16, fp32. Re-export the engine with a supported value.");
}

//! Parse a required dtype field. Throws if the field is missing — all current
//! exports (>= 0.7.0) write dtype fields explicitly to config.json.
void parseRequiredStateDtype(Json const& json, char const* key, nvinfer1::DataType& out)
{
    ELLM_CHECK(json.contains(key),
        std::string("parseEngineConfig: config.json missing required field '") + key
            + "'. Re-export the model with the latest llm_export.py to record it.");
    out = parseStateDtype(json[key].get<std::string>(), key);
}

//! Fields shared by base and draft engines. Parses top-level model dims and
//! dtypes (kv_cache_dtype lives at top level — it is a property of the
//! exported weights written by the Python export step, not a builder knob;
//! the C++ builder never emits it inside builder_config). Also parses the
//! common builder_config batch/input/kv limits and RoPE configuration.
//! Engine-type-specific fields (reduced vocab, hybrid state, SpecDecode
//! capacities, LoRA, trt_native_ops) are filled in by the calling parser
//! around this helper.
void parseCoreFields(Json const& configJson, LLMEngineConfig& cfg)
{
    // Top-level model fields.
    cfg.numDecoderLayers = getRequired<int32_t>(configJson, "num_hidden_layers");
    cfg.numKVHeads = getRequired<int32_t>(configJson, "num_key_value_heads");
    cfg.headDim = getRequired<int32_t>(configJson, "head_dim");
    cfg.hiddenSize = getRequired<int32_t>(configJson, "hidden_size");

    // Top-level: kv_cache_dtype. Required — all current exports write this.
    parseRequiredStateDtype(configJson, "kv_cache_dtype", cfg.kvCacheDtype);

    // builder_config: required batch / input / kv limits.
    ELLM_CHECK(configJson.contains("builder_config"), "parseEngineConfig: missing required 'builder_config' section");
    auto const& bc = configJson["builder_config"];
    cfg.maxSupportedBatchSize = getRequired<int32_t>(bc, "max_batch_size");
    cfg.maxSupportedInputLength = getRequired<int32_t>(bc, "max_input_len");
    cfg.maxKVCacheCapacity = getRequired<int32_t>(bc, "max_kv_cache_capacity");

    // RoPE configuration (top-level, derived from full config).
    cfg.ropeConfig = collectRopeConfig(configJson);

    // Positivity checks on required fields (common to both parsers).
    requirePositive(cfg.numDecoderLayers, "num_hidden_layers");
    requirePositive(cfg.numKVHeads, "num_key_value_heads");
    requirePositive(cfg.headDim, "head_dim");
    requirePositive(cfg.hiddenSize, "hidden_size");
    requirePositive(cfg.maxSupportedBatchSize, "max_batch_size");
    requirePositive(cfg.maxSupportedInputLength, "max_input_len");
    requirePositive(cfg.maxKVCacheCapacity, "max_kv_cache_capacity");
    ELLM_CHECK(cfg.maxSupportedInputLength <= cfg.maxKVCacheCapacity,
        "parseEngineConfig: max_input_len (" + std::to_string(cfg.maxSupportedInputLength)
            + ") cannot be greater than max_kv_cache_capacity (" + std::to_string(cfg.maxKVCacheCapacity) + ")");
}

//! Populate `cfg.layerTypes` and `cfg.kvLayerConfigs` from the canonical
//! `layer_types` + `kv_layer_configs` fields when present, or fall back to
//! scalar-broadcast ordering for legacy engines.
//!
//! Canonical path: `kv_layer_configs` present in config.json.
//!   - Both `layer_types` and `kv_layer_configs` must be present and same length.
//!   - Each "attention" entry appends kAttention to layerTypes and the
//!     corresponding `{num_kv_heads, head_dim}` object to kvLayerConfigs.
//!   - Each "mamba" entry appends kMamba to layerTypes (kvLayerConfigs unchanged).
//!
//! Fallback path: `kv_layer_configs` absent.
//!   - If `numLinearAttnLayers > 0` (hybrid): attention layers first, then mamba.
//!   - Otherwise (pure attention): all layers are kAttention with uniform dims.
void populateLayerTypes(Json const& configJson, LLMEngineConfig& cfg)
{
    if (configJson.contains("kv_layer_configs"))
    {
        ELLM_CHECK(configJson.contains("layer_types"),
            "parseEngineConfig: kv_layer_configs requires layer_types in config.json");
        auto const& layerTypesJson = configJson["layer_types"];
        auto const& kvLayerConfigsJson = configJson["kv_layer_configs"];
        ELLM_CHECK(layerTypesJson.size() == kvLayerConfigsJson.size(),
            "parseEngineConfig: layer_types (" + std::to_string(layerTypesJson.size()) + ") and kv_layer_configs ("
                + std::to_string(kvLayerConfigsJson.size()) + ") must have equal length.");
        int32_t const n = static_cast<int32_t>(layerTypesJson.size());
        for (int32_t i = 0; i < n; ++i)
        {
            std::string const typeStr = layerTypesJson[i].get<std::string>();
            if (typeStr == "attention")
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
                auto const& lc = kvLayerConfigsJson[i];
                ELLM_CHECK(!lc.is_null() && lc.contains("num_kv_heads") && lc.contains("head_dim"),
                    "parseEngineConfig: kv_layer_configs[" + std::to_string(i)
                        + "] missing num_kv_heads/head_dim for attention layer");
                cfg.kvLayerConfigs.push_back({lc["num_kv_heads"].get<int32_t>(), lc["head_dim"].get<int32_t>()});
            }
            else if (typeStr == "mamba")
            {
                // Validate both directions (matches legacy/llmEngineRunner.cpp):
                // mamba entries must be null; a non-null object here would be
                // silently discarded, masking a malformed config.
                ELLM_CHECK(kvLayerConfigsJson[i].is_null(),
                    "parseEngineConfig: kv_layer_configs[" + std::to_string(i) + "] must be null for a mamba layer");
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
            }
            else
            {
                ELLM_CHECK(false, "parseEngineConfig: unknown layer type '" + typeStr + "'");
            }
        }
    }
    else
    {
        // Fallback: attention-first then mamba, matching main's
        // llmEngineRunner.cpp scalar-broadcast ordering.
        if (cfg.numLinearAttnLayers > 0)
        {
            for (int32_t i = 0; i < cfg.numAttentionLayers; ++i)
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
                cfg.kvLayerConfigs.push_back({cfg.numKVHeads, cfg.headDim});
            }
            for (int32_t i = 0; i < cfg.numLinearAttnLayers; ++i)
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kMamba);
            }
        }
        else
        {
            for (int32_t i = 0; i < cfg.numDecoderLayers; ++i)
            {
                cfg.layerTypes.push_back(HybridCacheManager::LayerType::kAttention);
                cfg.kvLayerConfigs.push_back({cfg.numKVHeads, cfg.headDim});
            }
        }
    }
}

} // namespace

LLMEngineConfig parseEngineConfig(std::filesystem::path const& configPath)
{
    LOG_INFO("reading %s", configPath.string().c_str());

    std::ifstream ifs(configPath);
    ELLM_CHECK(ifs.is_open(), "parseEngineConfig: failed to open " + configPath.string());

    Json configJson;
    try
    {
        configJson = Json::parse(ifs);
    }
    catch (Json::parse_error const& e)
    {
        throw std::runtime_error("parseEngineConfig: JSON parse error in " + configPath.string() + ": " + e.what());
    }
    ifs.close();

    // Optional version check.
    std::string const modelVersion = configJson.value(binding_names::kEdgellmVersion, "");
    version::checkVersion(modelVersion);

    LLMEngineConfig cfg;

    // Parse speculative decoding type from model_type field.
    std::string const modelType = configJson.value("model_type", "");
    if (modelType == "mtp_base" || modelType == "mtp_draft")
    {
        cfg.specDecodeType = SpecDecodeMode::kMTP;
    }
    else if (modelType == "eagle3_base" || modelType == "eagle3_draft")
    {
        cfg.specDecodeType = SpecDecodeMode::kEAGLE;
    }

    // Shared core fields (layers, kv heads, head_dim, hidden_size, kv_cache_dtype,
    // batch/input/kv limits, RoPE, common positivity checks).
    parseCoreFields(configJson, cfg);

    // --- Base-specific: vocab, rotary dim, deepstack / multimodal, hybrid ---
    cfg.vocabSize = getRequired<int32_t>(configJson, "vocab_size");
    cfg.rotaryDim = static_cast<int32_t>(cfg.headDim * configJson.value("partial_rotary_factor", 1.0F));

    cfg.reducedVocabSize = configJson.value(binding_names::kReducedVocabSizeKey, 0);
    cfg.outputVocabSize = (cfg.reducedVocabSize > 0) ? cfg.reducedVocabSize : cfg.vocabSize;

    cfg.numDeepstackFeatures = configJson.value("num_deepstack_features", 0);
    cfg.audioTokenId = configJson.value("audio_token_id", -1);
    cfg.imageTokenId = configJson.value("image_token_id", -1);

    cfg.numLinearAttnLayers = configJson.value("num_linear_attn_layers", 0);
    cfg.numAttentionLayers = configJson.value("num_attention_layers", cfg.numDecoderLayers);
    cfg.recurrentStateNumHeads = configJson.value("recurrent_state_num_heads", 0);
    cfg.recurrentStateHeadDim = configJson.value("recurrent_state_head_dim", 0);
    cfg.recurrentStateSize = configJson.value("recurrent_state_size", 0);
    cfg.convDim = configJson.value("conv_dim", 0);
    cfg.convKernel = configJson.value("conv_kernel", 0);

    auto const& bc = configJson["builder_config"];
    cfg.maxSupportedLoraRank = bc.value("max_lora_rank", 0);
    cfg.isSpecDecodeBase = (cfg.specDecodeType != SpecDecodeMode::kNONE);

    // Recurrent / conv state dtypes are only meaningful for hybrid engines
    // (Mamba / Nemotron-H / GDN). Mirror the Python export gating exactly:
    // present and parsed strictly iff `num_linear_attn_layers > 0`. Same
    // reasoning as kv_cache_dtype above — these are weight-export properties
    // written to the top level by the Python export step.
    if (cfg.numLinearAttnLayers > 0)
    {
        parseRequiredStateDtype(configJson, "recurrent_state_dtype", cfg.recurrentStateDtype);
        parseRequiredStateDtype(configJson, "conv_state_dtype", cfg.convStateDtype);
    }

    // TRT native ops flag.
    if (bc.contains("trt_native_ops"))
    {
        cfg.useTrtNativeOps = bc["trt_native_ops"].get<bool>();
    }

    // Base-specific positivity checks (beyond parseCoreFields's core set).
    requirePositive(cfg.rotaryDim, "rotary_dim");
    requirePositive(cfg.vocabSize, "vocab_size");
    ELLM_CHECK(cfg.maxSupportedLoraRank >= 0,
        "parseEngineConfig: invalid max_lora_rank: " + std::to_string(cfg.maxSupportedLoraRank)
            + " (must be non-negative)");

    // --- SpecDecode base model ---
    // `max_verify_tree_size` is the base engine's own input budget (the
    // seq_len it accepts for proposal verification), so it must be present
    // in the base's builder_config. `max_draft_tree_size` is the draft
    // engine's input budget and lives in draft_config.json; the builder
    // correctly omits it from base_config.json (see llmBuilder.h), so
    // `cfg.maxDraftTreeSize` stays at 0 on the base — consumers read the
    // authoritative value from the draft engine config (or from
    // `DeploymentConfig::specDecode`, which consolidates both sides).
    if (cfg.isSpecDecodeBase)
    {
        cfg.maxVerifyTreeSize = getRequired<int32_t>(bc, "max_verify_tree_size");
        requirePositive(cfg.maxVerifyTreeSize, "max_verify_tree_size");
    }

    // Populate per-layer type routing from canonical fields or scalar fallback.
    populateLayerTypes(configJson, cfg);

    LOG_INFO("%s", formatEngineConfig(cfg).c_str());
    return cfg;
}

LLMEngineConfig parseDraftEngineConfig(std::filesystem::path const& configPath)
{
    LOG_INFO("reading %s", configPath.string().c_str());

    std::ifstream ifs(configPath);
    ELLM_CHECK(ifs.is_open(), "parseDraftEngineConfig: failed to open " + configPath.string());

    nlohmann::json configJson;
    try
    {
        configJson = nlohmann::json::parse(ifs);
    }
    catch (nlohmann::json::parse_error const& e)
    {
        throw std::runtime_error(
            "parseDraftEngineConfig: JSON parse error in " + configPath.string() + ": " + e.what());
    }
    ifs.close();

    LLMEngineConfig cfg;

    // Parse speculative decoding type from model_type field (draft side).
    std::string const modelType = configJson.value("model_type", "");
    if (modelType == "mtp_base" || modelType == "mtp_draft")
    {
        cfg.specDecodeType = SpecDecodeMode::kMTP;
    }
    else if (modelType == "eagle3_base" || modelType == "eagle3_draft")
    {
        cfg.specDecodeType = SpecDecodeMode::kEAGLE;
    }

    // Shared core fields (layers, kv heads, head_dim, hidden_size, kv_cache_dtype,
    // batch/input/kv limits, RoPE, common positivity checks).
    parseCoreFields(configJson, cfg);

    // --- Draft-specific ---
    cfg.numAttentionLayers = cfg.numDecoderLayers;
    // Apply `partial_rotary_factor` (same convention as parseEngineConfig). The
    // draft engine inherits the base's rotary dim, which can be a fraction of
    // headDim (e.g. Qwen3.5: headDim=256, rotaryDim=64, factor=0.25). Hard-coding
    // `cfg.rotaryDim = cfg.headDim` mismatches the engine's `rope_rotary_cos_sin`
    // binding shape and surfaces as a setInputShape failure on draft prefill.
    cfg.rotaryDim = static_cast<int32_t>(cfg.headDim * configJson.value("partial_rotary_factor", 1.0F));
    cfg.vocabSize = configJson.value("draft_vocab_size", configJson.value("vocab_size", 0));
    cfg.outputVocabSize = cfg.vocabSize;

    // Draft engines do not own hybrid runtime cache state in this path.
    cfg.isSpecDecodeBase = false; // This IS the draft engine, not the base.

    auto const& bc = configJson["builder_config"];
    // Symmetric to the base side (see parseEngineConfig): each engine's
    // builder_config carries only its own sequence budget. The base emits
    // `max_verify_tree_size` (its verification budget); the draft emits
    // `max_draft_tree_size` (its proposal/generation budget). So the draft
    // parser only requires `max_draft_tree_size`; `cfg.maxVerifyTreeSize`
    // stays at 0 on the draft — consumers read the consolidated values
    // from `DeploymentConfig::specDecode`.
    cfg.maxDraftTreeSize = getRequired<int32_t>(bc, "max_draft_tree_size");
    requirePositive(cfg.maxDraftTreeSize, "max_draft_tree_size");

    // Hidden dim the draft engine's `hidden_states_input` binding expects. The
    // python export writes this as `base.hidden_size * 3` for EAGLE-3 (multi-layer
    // concat) and `base.hidden_size` for MTP, so the runtime must honor whatever
    // the draft config carries — deriving it from `base.hiddenSize * 3` is wrong
    // for MTP.
    cfg.baseModelHiddenSize = getRequired<int32_t>(configJson, "base_model_hidden_size");
    requirePositive(cfg.baseModelHiddenSize, "base_model_hidden_size");

    LOG_INFO(
        "parseDraftEngineConfig: hiddenSize=%d vocabSize=%d layers=%d kvHeads=%d headDim=%d "
        "maxBatch=%d maxInputLen=%d maxKVCap=%d maxDraftTreeSize=%d baseModelHiddenSize=%d",
        cfg.hiddenSize, cfg.vocabSize, cfg.numDecoderLayers, cfg.numKVHeads, cfg.headDim, cfg.maxSupportedBatchSize,
        cfg.maxSupportedInputLength, cfg.maxKVCacheCapacity, cfg.maxDraftTreeSize, cfg.baseModelHiddenSize);

    // Populate per-layer type routing from canonical fields or scalar fallback.
    populateLayerTypes(configJson, cfg);

    return cfg;
}

std::string formatEngineConfig(LLMEngineConfig const& cfg)
{
    std::ostringstream ss;
    ss << std::boolalpha;
    ss << "LLMEngineConfig{"
       << " hiddenSize=" << cfg.hiddenSize << " vocabSize=" << cfg.vocabSize
       << " outputVocabSize=" << cfg.outputVocabSize << " numDecoderLayers=" << cfg.numDecoderLayers
       << " numAttentionLayers=" << cfg.numAttentionLayers << " numKVHeads=" << cfg.numKVHeads
       << " headDim=" << cfg.headDim << " rotaryDim=" << cfg.rotaryDim << " maxBatch=" << cfg.maxSupportedBatchSize
       << " maxInputLen=" << cfg.maxSupportedInputLength << " maxKVCapacity=" << cfg.maxKVCacheCapacity
       << " useTrtNativeOps=" << cfg.useTrtNativeOps << " isSpecDecodeBase=" << cfg.isSpecDecodeBase
       << " specDecodeType=" << static_cast<int>(cfg.specDecodeType) << " loraRank=" << cfg.maxSupportedLoraRank;

    if (cfg.numLinearAttnLayers > 0)
    {
        ss << " numLinearAttnLayers=" << cfg.numLinearAttnLayers
           << " recurrentStateDtype=" << static_cast<int>(cfg.recurrentStateDtype)
           << " convStateDtype=" << static_cast<int>(cfg.convStateDtype);
    }
    if (cfg.maxVerifyTreeSize > 0)
    {
        ss << " maxVerifyTreeSize=" << cfg.maxVerifyTreeSize;
    }
    // Only draft engines carry a meaningful `maxDraftTreeSize`; the base side
    // leaves it at 0 (see parseEngineConfig comment), so skip the noise.
    if (cfg.maxDraftTreeSize > 0)
    {
        ss << " maxDraftTreeSize=" << cfg.maxDraftTreeSize;
    }
    ss << " }";
    return ss.str();
}

// ---------------------------------------------------------------------------
// InferenceDims recipe methods
//
// Tier-1 #7 (C++20 designated initializers) is DEFERRED: the repo compiles
// with -std=c++17 -Werror, which rejects designated initializers even as a
// GNU extension. Keep the `/*.field=*/value` comment style until the repo's
// CXX_STANDARD is bumped.
// ---------------------------------------------------------------------------

InferenceDims LLMEngineConfig::prefillDims(int64_t batch, int64_t seqLen, bool kvCacheAllEmpty) const
{
    // seqLen drives the inputs_embeds / KV-write length only.
    //
    // attnMaskSeqLen and packedMaskLen are pinned to 1 (NOT derived from seqLen):
    // the SpecDecode base/draft attention plugin treats a `[B, 1, 1]` mask as a
    // signal to use standard causal attention and ignores the buffer contents,
    // which matches the dummy-mask binding from the pre-refactor runtime. A
    // larger attention shape would be interpreted as a proposal-attention mask
    // and read uninitialized buffer bits, producing garbage outputs.
    //
    // startIndexLen=0 is the plugin-path sentinel for "initial prefill of an
    // empty KV cache"; chunked prefill and TRT-native-ops engines use [batch].
    int64_t const startIndexLen = (!useTrtNativeOps && kvCacheAllEmpty) ? 0 : batch;
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/seqLen,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.startIndexLen=*/startIndexLen,
    };
}

InferenceDims LLMEngineConfig::decodeDims(int64_t batch) const
{
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/1,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/1,
        /*.startIndexLen=*/batch,
    };
}

InferenceDims LLMEngineConfig::specVerifyDims(int64_t batch, int64_t verifySize) const
{
    // verifySize fans out to four fields:
    //   seqLen, selectLen, attnMaskSeqLen (all = verifySize), and packedMaskLen.
    // This is the only recipe where selectLen != 1.
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/verifySize,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/verifySize,
        /*.attnMaskSeqLen=*/verifySize,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(verifySize, 32)),
        /*.startIndexLen=*/batch,
    };
}

InferenceDims LLMEngineConfig::proposalDims(int64_t batch, int64_t proposalSize, int64_t draftTopK) const
{
    // proposalSize fans out to: seqLen, attnMaskSeqLen, and packedMaskLen.
    // draftTopK is selectLen — the draft proposal selects draftTopK tokens per
    // sequence (matches the draft engine's 3D logits output shape
    // [batch, draftTopK, draftVocabSize] and last_token_ids shape
    // [batch, draftTopK]).
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/proposalSize,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/draftTopK,
        /*.attnMaskSeqLen=*/proposalSize,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(proposalSize, 32)),
        /*.startIndexLen=*/batch,
    };
}

InferenceDims LLMEngineConfig::acceptDims(int64_t batch, int64_t acceptLen) const
{
    // Precondition: acceptLen >= 1. Caller computes from the SpecDecode accept result in
    // [1, draftingStep+1]; a zero here would indicate a degenerate batch and
    // will be caught by EngineExecutor::prepare's field-completeness check.
    // acceptLen fans out to: seqLen, attnMaskSeqLen, and packedMaskLen.
    return InferenceDims{
        /*.batch=*/batch,
        /*.seqLen=*/acceptLen,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/acceptLen,
        /*.ropeBatch=*/(ropeConfig.type == RopeType::kMRope) ? batch : 1,
        /*.packedMaskLen=*/static_cast<int64_t>(divUp(acceptLen, 32)),
        /*.startIndexLen=*/batch,
    };
}

void validateAgainstEngine(LLMEngineConfig const& config, EngineExecutor const& executor, char const* engineLabel)
{
    // KV cache binding: validated on layer 0; all layers share the same dtype.
    // TRT-native-ops engines split KV into separate `k_cache_%d` / `v_cache_%d`
    // bindings; plugin-path engines use a combined `past_key_values_%d`. Query
    // whichever the engine actually exposes — asking for the wrong name returns
    // a spurious FLOAT32 (TRT's default for an unknown binding).
    if (config.numAttentionLayers > 0)
    {
        std::string const kvBindingName = config.useTrtNativeOps
            ? binding_names::formatKCacheName(/*layerIdx=*/0, /*isPast=*/true)
            : binding_names::formatKVCacheName(/*layerIdx=*/0, /*isPast=*/true);
        auto const engineDtype = executor.getBindingDataType(kvBindingName.c_str());
        ELLM_CHECK(engineDtype == config.kvCacheDtype,
            std::string("KV cache dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.kvCacheDtype) + ", engine reports " + getDataTypeString(engineDtype)
                + " for binding '" + kvBindingName + "'. Re-export the engine with matching kv_cache_dtype.");
    }

    // Recurrent / conv state bindings (hybrid models only).
    if (config.numLinearAttnLayers > 0)
    {
        std::string const recBindingName = binding_names::formatRecurrentStateName(/*layerIdx=*/0, /*isPast=*/true);
        auto const recEngineDtype = executor.getBindingDataType(recBindingName.c_str());
        ELLM_CHECK(recEngineDtype == config.recurrentStateDtype,
            std::string("Recurrent state dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.recurrentStateDtype) + ", engine reports "
                + getDataTypeString(recEngineDtype) + " for binding '" + recBindingName
                + "'. Re-export the engine with matching recurrent_state_dtype.");

        std::string const convBindingName = binding_names::formatConvStateName(/*layerIdx=*/0, /*isPast=*/true);
        auto const convEngineDtype = executor.getBindingDataType(convBindingName.c_str());
        ELLM_CHECK(convEngineDtype == config.convStateDtype,
            std::string("Conv state dtype mismatch (") + engineLabel + "): config says "
                + getDataTypeString(config.convStateDtype) + ", engine reports " + getDataTypeString(convEngineDtype)
                + " for binding '" + convBindingName + "'. Re-export the engine with matching conv_state_dtype.");
    }
}

InferenceDims LLMEngineConfig::resetDims() const
{
    // Clean-slate binding for CUDA-graph capture reset. ropeBatch is hardcoded
    // to 1 even for MRope — the reset is a placeholder binding, not an
    // inference step.
    // startIndexLen=1 matches batch=1; the reset is a placeholder bind, not a
    // real initial-prefill (which would need shape [0]).
    return InferenceDims{
        /*.batch=*/1,
        /*.seqLen=*/1,
        /*.kvLen=*/maxKVCacheCapacity,
        /*.selectLen=*/1,
        /*.attnMaskSeqLen=*/1,
        /*.ropeBatch=*/1,
        /*.packedMaskLen=*/1,
        /*.startIndexLen=*/1,
    };
}

} // namespace rt
} // namespace trt_edgellm
