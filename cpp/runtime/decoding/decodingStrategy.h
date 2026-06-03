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
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/state/systemPromptKVCache.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

enum class DecodingStrategyKind : int32_t
{
    kVanilla,
    kEAGLE,
    kMTP,
};

struct SamplingBuffers
{
    Tensor& workspace;
    Tensor& indices;
    Tensor& scores;
    Tensor& baseVocabMappingTable;
    Tensor& hostPackedTokenIds;
    Tensor& hostSelectedTokenIds;
};

//! Base-engine execution infrastructure: executor, tensor map, KV cache,
//! pipeline I/O, shared resources, and CUDA-graph capture callback.
struct BaseEngineResources
{
    EngineExecutor& executor;
    TensorMap& tensorMap;
    SharedResources& sharedResources;
    HybridCacheManager& cacheManager;
    PipelineIO& pipelineIO;
    std::function<bool(InferenceDims const&, cudaStream_t)> captureGraph;
};

//! Preprocessing resources: embedding lookup, step preparation, deepstack.
struct PreprocessResources
{
    StepPreparer& stepPreparer;
    EmbeddingPreprocessor& embeddingPreprocessor;
    EmbeddingData& embedding;
    Tensor& idsInput;
    DeepstackBinding* deepstack;
};

struct DecodingRuntimeContext
{
    DeploymentConfig& deployment;
    int32_t maxRuntimeBatchSize;

    BaseEngineResources base;
    PreprocessResources preprocess;
    tokenizer::Tokenizer& tokenizer;
    SamplingBuffers sampling;
};

class DecodingStrategy
{
public:
    virtual ~DecodingStrategy() noexcept = default;

    virtual DecodingStrategyKind kind() const noexcept = 0;
    virtual char const* name() const noexcept = 0;
    virtual bool isSpeculative() const noexcept = 0;

    //! Check whether this strategy can handle the given request.
    //! @return nullptr if supported; a human-readable reason string if not.
    virtual char const* unsupportedReason(LLMGenerationRequest const&) const noexcept = 0;

    virtual bool decodeStep(DecodingInferenceContext& context) = 0;
    virtual bool captureCudaGraphs(cudaStream_t stream) = 0;

    virtual int64_t getRequiredContextMemorySize() const noexcept = 0;
    virtual void setContextMemory(Tensor&) = 0;

    // TODO(follow-up): Expose a prefill entry point on DecodingStrategy and remove
    // these system-prompt-specific interfaces. The runtime should call a generic
    // prefill method; system prompt caching becomes an internal optimisation.
    virtual bool hasSystemPromptKVCache(SystemPromptCacheKey const&) const = 0;
    virtual void restoreSystemPromptKVCache(SystemPromptCacheKey const&, int32_t, cudaStream_t) = 0;
    virtual bool runSystemPromptPrefill(DecodingInferenceContext&) = 0;
    virtual void saveSystemPromptKVCache(SystemPromptCacheKey const&, std::string const&,
        std::vector<tokenizer::Rank> const&, int32_t, cudaStream_t) = 0;

    virtual void resetForNewSequences(Tensor&, cudaStream_t) = 0;
    virtual void onBatchEvict(std::vector<int32_t> const&, int32_t, int32_t, Tensor&, cudaStream_t) = 0;
};

} // namespace rt
} // namespace trt_edgellm
