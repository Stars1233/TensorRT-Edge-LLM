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

#include "runtime/exec/registryBuilder.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace rt
{

// Note: `sym(...)` and `fixed(...)` helpers are now provided by inferenceDims.h.
// Symbolic-dim registration (formerly `addCommonSymbolicDims` / `addSymbolicDim`)
// is no longer needed — every symbolic reference is a pointer-to-member of
// `InferenceDims`, so the set of dims exists by construction of the type.

TensorRegistry buildRegistryForLLM(LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim)
{
    TensorRegistry reg;

    // ---------------------------------------------------------------
    // Core I/O tensors (always present)
    // ---------------------------------------------------------------

    // inputs_embeds: [batch, seq_len, hiddenSize] HALF
    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(cfg.hiddenSize)}});

    // logits: [batch, outputVocabSize] FLOAT for vanilla, or
    // [batch, seq_len, outputVocabSize] for SpecDecode. The engine binding
    // shape depends on mode, but output address is always set.
    // For the registry we use the common 2D shape; SpecDecode resolves via symbolic dims.
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::batch), fixed(cfg.outputVocabSize)}});

    // context_lengths: [batch] INT32
    reg.addTensor(
        {binding_names::kContextLengths, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::batch)}});

    // last_token_ids: [batch, select_len] INT64 — always [batch, 1] for vanilla, varies for SpecDecode.
    reg.addTensor({binding_names::kLastTokenIds, TensorIO::kInput, nvinfer1::DataType::kINT64,
        {sym(&InferenceDims::batch), sym(&InferenceDims::selectLen)}});

    // kvcache_start_index: [start_index_len] INT32. The engine's context profile
    // uses shape [0] as a sentinel for "initial prefill of an empty KV cache";
    // chunked prefill, decode, and verification use [batch] start offsets.
    // InferenceDims::startIndexLen carries this per-phase: prefillDims sets it to 0
    // when (!useTrtNativeOps && kvCacheAllEmpty), else batch; all other recipes
    // set it to batch. Shape 0 is engine-valid here — TRT reads 0 bytes from
    // the bound address and the engine branches to the initial-prefill path.
    reg.addTensor({binding_names::kKVCacheStartIndex, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::startIndexLen)}});

    // rope_rotary_cos_sin: [rope_batch, kv_len, rotaryDim] FLOAT
    // For non-MRope, rope_batch is always 1 (TRT broadcasts); for MRope, rope_batch = activeBatchSize.
    reg.addTensor({binding_names::kRopeCosSin, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::ropeBatch), sym(&InferenceDims::kvLen), fixed(cfg.rotaryDim)}});

    // ---------------------------------------------------------------
    // Per-layer KV / recurrent / conv state (hybrid routing by layer_types)
    // ---------------------------------------------------------------
    //
    // Walk the absolute decoder-layer indices and emit one spec per layer.
    // For attention layers we use `cfg.kvLayerConfigs[localAttnIdx]` so a model
    // with heterogeneous head configs (Gemma4, Qwen3-Next, etc.) gets the
    // correct per-layer shape. The %d suffix in the binding name is always a
    // LOCAL index (0..numAttn-1 for attention, 0..numMamba-1 for recurrent).
    int32_t localAttnIdx = 0;
    int32_t localMambaIdx = 0;
    for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
    {
        if (cfg.layerTypes[absIdx] == rt::HybridCacheManager::LayerType::kAttention)
        {
            auto const& lc = cfg.kvLayerConfigs[localAttnIdx];
            auto addKVCacheTensor = [&](char const* tmpl, TensorIO io, std::vector<ShapeDim> const& shape) {
                reg.addTensor({std::string(tmpl) + "_" + std::to_string(localAttnIdx), io, cfg.kvCacheDtype, shape});
            };
            if (cfg.useTrtNativeOps)
            {
                // TRT native: separate K / V, 4D [batch, numKVHeads, kv_len, headDim]
                std::vector<ShapeDim> const shape{
                    sym(&InferenceDims::batch), fixed(lc.numKVHeads), sym(&InferenceDims::kvLen), fixed(lc.headDim)};
                addKVCacheTensor(binding_names::kKCacheTemplate, TensorIO::kInput, shape);
                addKVCacheTensor(binding_names::kPresentKCacheTemplate, TensorIO::kOutput, shape);
                addKVCacheTensor(binding_names::kVCacheTemplate, TensorIO::kInput, shape);
                addKVCacheTensor(binding_names::kPresentVCacheTemplate, TensorIO::kOutput, shape);
            }
            else
            {
                // Plugin: combined KV, 5D [batch, 2, numKVHeads, kv_len, headDim]
                std::vector<ShapeDim> const shape{sym(&InferenceDims::batch), fixed(2), fixed(lc.numKVHeads),
                    sym(&InferenceDims::kvLen), fixed(lc.headDim)};
                addKVCacheTensor(binding_names::kPastKeyValuesTemplate, TensorIO::kInput, shape);
                addKVCacheTensor(binding_names::kPresentKeyValuesTemplate, TensorIO::kOutput, shape);
            }
            ++localAttnIdx;
        }
        else // kMamba
        {
            auto addMambaTensor
                = [&](char const* tmpl, TensorIO io, nvinfer1::DataType dtype, std::vector<ShapeDim> const& shape) {
                      reg.addTensor({std::string(tmpl) + "_" + std::to_string(localMambaIdx), io, dtype, shape});
                  };
            // recurrent_state_%d: [batch, recurrentStateNumHeads, recurrentStateHeadDim, recurrentStateSize]
            std::vector<ShapeDim> const recShape{sym(&InferenceDims::batch), fixed(cfg.recurrentStateNumHeads),
                fixed(cfg.recurrentStateHeadDim), fixed(cfg.recurrentStateSize)};
            addMambaTensor(binding_names::kRecurrentStateTemplate, TensorIO::kInput, cfg.recurrentStateDtype, recShape);
            addMambaTensor(
                binding_names::kPresentRecurrentStateTemplate, TensorIO::kOutput, cfg.recurrentStateDtype, recShape);
            // conv_state_%d: [batch, convDim, convKernel]
            std::vector<ShapeDim> const convShape{
                sym(&InferenceDims::batch), fixed(cfg.convDim), fixed(cfg.convKernel)};
            addMambaTensor(binding_names::kConvStateTemplate, TensorIO::kInput, cfg.convStateDtype, convShape);
            addMambaTensor(binding_names::kPresentConvStateTemplate, TensorIO::kOutput, cfg.convStateDtype, convShape);

            // MTP base only: per-layer intermediate state outputs the engine
            // writes during prefill/tree-verify. The builder emits these only
            // when `model_type == "mtp_base"`.
            //
            // intermediate_recurrent_state_%d: [batch, seqLen, recurrentNumHeads, recurrentHeadDim, recurrentStateSize]
            // intermediate_conv_state_%d:      [batch, seqLen, convDim, convKernel]
            if (cfg.specDecodeType == SpecDecodeMode::kMTP)
            {
                std::vector<ShapeDim> const interRecShape{sym(&InferenceDims::batch), sym(&InferenceDims::seqLen),
                    fixed(cfg.recurrentStateNumHeads), fixed(cfg.recurrentStateHeadDim), fixed(cfg.recurrentStateSize)};
                addMambaTensor(binding_names::kIntermediateRecurrentStateTemplate, TensorIO::kOutput,
                    cfg.recurrentStateDtype, interRecShape);
                if (cfg.convDim > 0 && cfg.convKernel > 0)
                {
                    std::vector<ShapeDim> const interConvShape{sym(&InferenceDims::batch), sym(&InferenceDims::seqLen),
                        fixed(cfg.convDim), fixed(cfg.convKernel)};
                    addMambaTensor(binding_names::kIntermediateConvStateTemplate, TensorIO::kOutput, cfg.convStateDtype,
                        interConvShape);
                }
            }
            ++localMambaIdx;
        }
    }

    // ---------------------------------------------------------------
    // Deepstack (Qwen3-VL / Qwen3-Omni)
    // ---------------------------------------------------------------
    if (cfg.numDeepstackFeatures > 0)
    {
        // deepstack_embeds_%d: [batch, seq_len, hiddenSize] HALF — one per feature.
        // DeepstackBinding swaps the backing tensor (real per-request buffer
        // vs. shared zero buffer) between prefill and non-prefill phases.
        reg.addTensor({std::string(binding_names::kDeepstackEmbedsTemplate) + "_%d", TensorIO::kInput,
            nvinfer1::DataType::kHALF, {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(cfg.hiddenSize)},
            /*perLayer=*/cfg.numDeepstackFeatures});
    }

    // ---------------------------------------------------------------
    // SpecDecode verification bindings (base engine side)
    // ---------------------------------------------------------------
    if (cfg.isSpecDecodeBase)
    {
        // hidden_states: output, [batch, outputHiddenDim] for vanilla decode,
        // or [batch, seq_len, outputHiddenDim] for prefill/verification — use symbolic.
        // The concrete output hidden dim is strategy-specific and is consolidated
        // in DeploymentConfig::specDecode.
        int32_t const baseOutputHiddenDim = specDecodeBaseOutputHiddenDim.value_or(cfg.hiddenSize * 3);
        reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
            {sym(&InferenceDims::batch), fixed(baseOutputHiddenDim)}});

        // attention_mask: [batch, attn_seq_len, packed_mask_len] INT32 for proposal verification
        // packed_mask_len = divUp(attn_seq_len, 32): each INT32 stores 32 mask bits.
        // attn_seq_len is decoupled from seq_len so prefill/decode/reset can pin
        // it to 1 (engine then uses standard causal attention) while verify,
        // proposal, and accept use the effective proposal size.
        reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

        // attention_pos_id: [batch, attn_seq_len] INT32
        reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
            {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen)}});
    }

    // ---------------------------------------------------------------
    // LoRA
    // ---------------------------------------------------------------
    // LoRA bindings are dynamic (enumerated from the engine). The registry
    // builder cannot know the exact tensor names at compile time because they
    // depend on which model layers have LoRA adapters. LoRA tensors are handled
    // separately by the EngineExecutor via engine introspection (getLoraWeightsTensorNames).
    // Therefore we intentionally skip LoRA here.

    return reg;
}

TensorRegistry buildRegistryForSpecDecodeDraft(DeploymentConfig const& bundle)
{
    check::check(bundle.draft.has_value(), "buildRegistryForSpecDecodeDraft: bundle.draft must be set");
    check::check(bundle.specConfig.has_value(), "buildRegistryForSpecDecodeDraft: bundle.specConfig must be set");
    TensorRegistry reg;

    LLMEngineConfig const& cfg = *bundle.draft;
    int32_t const draftHiddenSize = bundle.specConfig->draftHiddenSize;
    int32_t const baseOutputHiddenDim = bundle.specConfig->baseOutputHiddenDim;
    int32_t const draftVocabSize = cfg.outputVocabSize;

    // ---------------------------------------------------------------
    // Core I/O tensors
    // ---------------------------------------------------------------

    // inputs_embeds: [batch, seq_len, draftHiddenSize] HALF
    reg.addTensor({binding_names::kInputsEmbeds, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});

    // hidden_states_input (base model hidden states): [batch, seq_len, baseOutputHiddenDim] HALF
    reg.addTensor({binding_names::kBaseModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(baseOutputHiddenDim)}});

    // hidden_states_from_draft (draft model hidden states): [batch, seq_len, draftHiddenSize] HALF
    reg.addTensor({binding_names::kDraftModelHiddenStates, TensorIO::kInput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), sym(&InferenceDims::seqLen), fixed(draftHiddenSize)}});

    // last_token_ids: [batch, select_len] INT64
    reg.addTensor({binding_names::kLastTokenIds, TensorIO::kInput, nvinfer1::DataType::kINT64,
        {sym(&InferenceDims::batch), sym(&InferenceDims::selectLen)}});

    // context_lengths: [batch] INT32
    reg.addTensor(
        {binding_names::kContextLengths, TensorIO::kInput, nvinfer1::DataType::kINT32, {sym(&InferenceDims::batch)}});

    // kvcache_start_index: [start_index_len] INT32. Draft engine always uses
    // the plugin path: shape [0] sentinel during round-0 prefill (empty draft
    // KV cache); [batch] for proposal / accept.
    reg.addTensor({binding_names::kKVCacheStartIndex, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::startIndexLen)}});

    // rope_rotary_cos_sin: [rope_batch, kv_len, rotaryDim] FLOAT
    reg.addTensor({binding_names::kRopeCosSin, TensorIO::kInput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::ropeBatch), sym(&InferenceDims::kvLen), fixed(cfg.rotaryDim)}});

    // attention_mask: [batch, attn_seq_len, packed_mask_len] INT32 — tree decoding mask
    // packed_mask_len = divUp(attn_seq_len, 32): each INT32 stores 32 mask bits.
    // attn_seq_len is decoupled from seq_len so prefill/decode/reset can pin
    // it to 1 (engine then uses standard causal attention) while tree
    // verify/proposal/accept use the effective tree size.
    reg.addTensor({binding_names::kAttentionMask, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen), sym(&InferenceDims::packedMaskLen)}});

    // attention_pos_id: [batch, attn_seq_len] INT32
    reg.addTensor({binding_names::kAttentionPosId, TensorIO::kInput, nvinfer1::DataType::kINT32,
        {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen)}});

    // ---------------------------------------------------------------
    // Outputs
    // ---------------------------------------------------------------

    // logits: [batch, draftVocabSize] FLOAT
    reg.addTensor({binding_names::kLogits, TensorIO::kOutput, nvinfer1::DataType::kFLOAT,
        {sym(&InferenceDims::batch), fixed(draftVocabSize)}});

    // hidden_states (output): [batch, draftHiddenSize] HALF
    reg.addTensor({binding_names::kOutputHiddenStates, TensorIO::kOutput, nvinfer1::DataType::kHALF,
        {sym(&InferenceDims::batch), fixed(draftHiddenSize)}});

    // ---------------------------------------------------------------
    // KV cache (draft engine always uses plugin path)
    // ---------------------------------------------------------------
    //
    // Draft engines today are uniform (single numKVHeads / headDim across all
    // attention layers), and their `layerTypes` are broadcast from
    // `numAttentionLayers` by `parseDraftEngineConfig`. We still walk per-layer
    // so this stays forward-compatible with future heterogeneous draft configs.
    {
        int32_t localAttnIdx = 0;
        for (int32_t absIdx = 0; absIdx < static_cast<int32_t>(cfg.layerTypes.size()); ++absIdx)
        {
            if (cfg.layerTypes[absIdx] != rt::HybridCacheManager::LayerType::kAttention)
            {
                // Current draft engines are not expected to contain Mamba layers. If a
                // future config exercises this branch it's a config error, not a
                // registry-builder concern.
                continue;
            }
            auto const& lc = cfg.kvLayerConfigs[localAttnIdx];
            std::vector<ShapeDim> const shape{sym(&InferenceDims::batch), fixed(2), fixed(lc.numKVHeads),
                sym(&InferenceDims::kvLen), fixed(lc.headDim)};
            auto addKVCacheTensor = [&](char const* tmpl, TensorIO io) {
                reg.addTensor({std::string(tmpl) + "_" + std::to_string(localAttnIdx), io, cfg.kvCacheDtype, shape});
            };
            addKVCacheTensor(binding_names::kPastKeyValuesTemplate, TensorIO::kInput);
            addKVCacheTensor(binding_names::kPresentKeyValuesTemplate, TensorIO::kOutput);
            ++localAttnIdx;
        }
    }

    return reg;
}

} // namespace rt
} // namespace trt_edgellm
