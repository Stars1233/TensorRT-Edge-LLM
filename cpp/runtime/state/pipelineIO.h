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
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/state/sharedResources.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! All tensors flowing through the inference pipeline.
//! POINTER STABILITY INVARIANT: After buildTensorMap() is called, this struct
//! must not be moved, and deepstackEmbeds must not be resized. TensorMap holds
//! Tensor* pointers into these members — any reallocation invalidates them.
struct PipelineIO
{
    // Always present
    Tensor inputsEmbeds;
    Tensor outputLogits;
    Tensor selectTokenIndices;
    Tensor contextLengths;     //!< GPU
    Tensor hostContextLengths; //!< CPU (pinned, [maxBatch] INT32)
    Tensor
        hostSelectTokenIndices; //!< CPU (pinned, [maxBatch, 1] INT64) — pairs with selectTokenIndices for H2D staging

    // Multimodal (resize deepstackEmbeds BEFORE buildTensorMap, never after)
    std::vector<Tensor> deepstackEmbeds;
    Tensor mropeCosSin;

    // Spec decode
    Tensor baseHiddenStates;
    Tensor draftHiddenStatesIn;
    Tensor draftHiddenStatesOut;

    // Streaming output (Qwen3-Omni Talker pipeline). For the vanilla
    // LLM runtime the engine writes its layer-N hidden states into
    // `outputHiddenStates`; the runtime exposes them through
    // `LLMInferenceRuntime::getBaseModelHiddenStates(N)`.
    // `prefillEmbedsBackup` snapshots layer-0 input embeddings before the decode
    // loop reshapes `inputsEmbeds`; it is lazy-allocated on the first request that
    // sets `outputThinkerEmbeddings`. SpecDecode configs reuse `baseHiddenStates`
    // instead and leave these empty.
    Tensor outputHiddenStates;
    Tensor prefillEmbedsBackup;

    // SpecDecode engine-bound tensors (empty for vanilla LLM runtime).
    //! Packed proposal attention mask, [batch, proposalSize, divUp(proposalSize, 32)] INT32.
    //! Written by proposal/verify input preparation kernels; consumed by the base and draft
    //! engines via the `kAttentionMask` binding.
    Tensor packedAttentionMask;
    //! SpecDecode position IDs, [batch, proposalSize] INT32.
    //! Written by proposal/verify input preparation kernels; consumed by the base and draft
    //! engines via the `kAttentionPosId` binding.
    Tensor specDecodePositionIds;

    //! Build PipelineIO for the vanilla single-engine LLM runtime
    //! (basic I/O tensors, deepstack embeds, MRope cos/sin cache).
    static PipelineIO createForLLM(LLMEngineConfig const& cfg, cudaStream_t stream);

    //! Build PipelineIO for a two-engine speculative-decoding runtime
    //! (basic I/O, hidden states, deepstack embeds, MRope cos/sin cache).
    static PipelineIO createForSpecDecode(
        DeploymentConfig const& bundle, int32_t maxRuntimeBatchSize, cudaStream_t stream);
};

void allocateBasicIO(
    PipelineIO& io, int32_t maxBatch, int32_t maxSeq, int32_t hiddenSize, int32_t vocabSize, nvinfer1::DataType dtype);

void allocateDeepstackEmbeds(PipelineIO& io, int32_t numFeatures, int32_t maxBatch, int32_t maxSeq, int32_t hiddenSize,
    nvinfer1::DataType dtype);

void allocateSpecDecodeHiddenStates(PipelineIO& io, int32_t maxBatch, int32_t maxSeq, int32_t baseHiddenDim,
    int32_t draftHiddenDim, nvinfer1::DataType dtype);

void allocateMRope(PipelineIO& io, int32_t maxBatch, int32_t maxKVCacheCapacity, int32_t rotaryDim);

//! Populate a TensorMap from PipelineIO + SharedResources for engine binding.
//!
//! This is the critical glue function that wires all allocated tensors into the
//! name-to-pointer map consumed by TensorRegistry::bindAll().
//!
//! @param map          Output map to populate.
//! @param io           Pipeline I/O tensors.
//! @param res          Shared resources (KV caches, RoPE pool, LoRA, zero buffer).
//! @param cfg          Engine configuration.
//! @param kvCacheIndex Index into res.cacheManagers for the target engine.
void buildTensorMap(
    TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg, int32_t kvCacheIndex);

//! Populate a TensorMap for a SpecDecode draft engine. Delegates to `buildTensorMap`
//! with `kvCacheIndex=1` for the common bindings, then patches in draft-engine-
//! specific bindings (base/draft hidden states in+out, packed proposal attention
//! mask, proposal position IDs).
//!
//! Preconditions: `io` must have been constructed via `PipelineIO::createForSpecDecode`
//! (baseHiddenStates / draftHiddenStatesIn/Out / packedAttentionMask /
//! specDecodePositionIds populated).
//!
//! @param map Output map for the draft engine's bindings.
//! @param io  Pipeline I/O (must be the SpecDecode-flavoured one).
//! @param res Shared resources.
//! @param cfg Draft engine configuration.
void buildTensorMapForSpecDecodeDraft(TensorMap& map, PipelineIO& io, SharedResources& res, LLMEngineConfig const& cfg);

} // namespace rt
} // namespace trt_edgellm
