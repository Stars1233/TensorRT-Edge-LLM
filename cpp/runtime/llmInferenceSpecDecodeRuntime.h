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

#include "action/alpamayo1ActionRunner.h"
#include "common/hashUtils.h"
#include "common/tensor.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/eagleDraftEngineRunner.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include <cassert>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{

/*! \brief Structure to hold cached system prompt and its KV cache (unified with recurrent state support)
 */
struct SystemPromptKVCache
{
    std::string systemPrompt;                     //!< The system prompt text
    std::vector<tokenizer::Rank> tokenizedPrompt; //!< Tokenized version of the system prompt
    std::vector<rt::Tensor> kvCacheLayers;        //!< Per-layer KV cache tensors for the system prompt
    std::vector<rt::Tensor>
        recurrentStateContents;                //!< Cached recurrent states for hybrid layers (empty if not applicable)
    std::vector<rt::Tensor> convStateContents; //!< Cached conv states for hybrid layers (empty if not applicable)
};

namespace rt
{
/*!
 * @brief Batch result data for a single sequence
 *
 * Encapsulates all data needed to track a batch's execution results,
 * whether it's active or evicted. Groups related fields together for
 * better cache locality and maintainability.
 */
struct BatchResult
{
    std::vector<int32_t> tokenIds;           //!< Generated token IDs
    std::vector<int32_t> rawBatchedInputIds; //!< Original input token IDs
    int32_t generateLength{0};               //!< Number of tokens generated
    int32_t actualIterations{0};             //!< Number of iterations executed
    int32_t effectivePrefillLength{0};       //!< Effective prefill length (excluding reused KVCache length)
};

/*!
 * @brief Execution context for speculative decode runtime
 *
 * Holds execution information and intermediate metadata during inference.
 * Supports multi-batch inference with independent sequence tracking.
 */
struct SpecDecodeInferenceContext
{
    std::vector<std::string> systemPrompts;               //!< System prompts for each sequence in batch
    std::vector<std::vector<int32_t>> rawBatchedInputIds; //!< Original token IDs before preprocessing (includes padding
                                                          //!< and removal of reused system IDs)
    std::vector<std::vector<int32_t>> tokenIds;           //!< Token IDs for each sequence: [batch_size][seq_length]
    std::vector<int32_t> currentGenerateLengths;          //!< Current generation length for each sequence: [batch_size]
    std::vector<int32_t>
        effectivePrefillLengths;        //!< Effective prefill length (excluding reused KVCache length) [batch_size]
    std::vector<int8_t> finishedStates; //!< Finished state for each sequence: [batch_size] (0=not finished, 1=finished)

    // Completed batch results (saved before eviction for final output)
    // Key: original batch index, Value: complete batch result data
    std::unordered_map<int32_t, BatchResult> completedBatches; //!< Results of completed batches (unified storage)
    std::vector<int32_t> batchIndexMapping;                    //!< Maps current batch index to original index
    std::vector<SlotStreamState> slotStreams;                  //!< Per-slot streaming state (parallel to tokenIds).
    rt::OptionalInputTensor visualEmbeddings;                  //!< Optional visual embeddings
    rt::OptionalInputTensor audioEmbeddings;                   //!< Optional audio embeddings
    rt::OptionalInputTensors deepstackFeatures; //!< Deepstack features for Qwen3-VL (raw features before embedding)
    int32_t generationRound;                    //!< Current generation round (shared across all batches)
    int32_t maxGenerateLength;                  //!< Maximum generation length
    int32_t activeBatchSize;                    //!< Current active batch size
    std::string loraWeightsName{""};            //!< LoRA adapter name used by this request
    cudaStream_t stream;                        //!< CUDA stream

    // Sampling parameters (forwarded from request)
    float temperature{1.0f}; //!< Temperature for sampling
    float topP{1.0f};        //!< Top-P (nucleus) sampling parameter
    int64_t topK{0};         //!< Top-K sampling parameter

    // Thinker embedding output (Qwen3-Omni audio generation)
    bool outputThinkerEmbeddings{false}; //!< Whether to capture hidden states for Talker pipeline

    //! Optional per-token callback invoked after each vanilla decode step
    std::optional<TokenCallback> onTokenGenerated;

    /*!
     * @brief Initialize the context with given parameters
     * @param batchSize Active batch size
     * @param maxGenLength Maximum generation length
     * @param visual Optional visual embeddings
     * @param deepstackFeatures Deepstack features for Qwen3-VL (raw features before embedding)
     * @param loraName LoRA weights name used by this request
     * @param cudaStream CUDA stream for operations
     */
    void initialize(int32_t batchSize, int32_t maxGenLength, rt::OptionalInputTensor const& visual,
        rt::OptionalInputTensors const& deepstackFeatures, std::string const& loraName, cudaStream_t cudaStream);
};

/*!
 * @brief Drafting configuration for Eagle speculative decoding
 *
 * Configuration parameters to drive Eagle spec-decoding.
 */
struct EagleDraftingConfig
{
    int32_t draftingTopK;   //!< Tokens to select from one predecessor for next draft tree level
    int32_t draftingStep;   //!< Number of drafting steps with draft model
    int32_t verifyTreeSize; //!< Number of tokens for base model verification
};

/*!
 * @brief Unified LLM inference runtime with optional Eagle speculative decoding
 *
 * Manages inference pipeline for both standard (vanilla) and Eagle speculative decoding modes.
 * When constructed without a drafting config, operates as a pure vanilla decoding runtime
 * (equivalent to the former LLMInferenceRuntime) with zero draft-model memory overhead.
 * Coordinates base model, optional draft model, and multimodal processing (vision + audio).
 */
class LLMInferenceSpecDecodeRuntime
{
public:
    /*!
     * @brief Construct runtime with Eagle speculative decoding
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param draftingConfig Eagle drafting configuration
     * @param stream CUDA stream for operations
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceSpecDecodeRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, EagleDraftingConfig const& draftingConfig,
        cudaStream_t stream);

    /*!
     * @brief Construct runtime for vanilla-only decoding (no draft model)
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param stream CUDA stream for operations
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceSpecDecodeRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    //! @brief Destructor
    ~LLMInferenceSpecDecodeRuntime() noexcept = default;

    //! @brief Capture CUDA graphs for decoding stages to optimize performance.
    //!
    //! When draft model is present, captures graphs for draft proposal, draft accept token,
    //! base verification, and base vanilla decoding. Without draft model, captures only
    //! vanilla decoding graphs.
    //!
    //! @param stream CUDA stream
    //! @return True if all stage captures succeed, false otherwise
    //! @throws std::runtime_error if a tensor reshape operation fails
    //! @note If capture fails for any stage, the inference can proceed without CUDA graph capture,
    //! but at cost of performance degradation.
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*!
     * @brief Handle generation request
     * @param request Generation request with prompts and parameters
     * @param response Output response with generated tokens and text
     * @param stream CUDA stream
     * @return True on success, false on failure
     * @throws std::runtime_error if an LLM or CUDA operation fails
     */
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false);

    /*!
     * @brief Generate and save system prompt KV cache (public API matching standard runtime signature)
     * @param prompt The system prompt to generate the KVCache
     * @param loraWeightsName The name of the LoRA weights
     * @param stream The CUDA stream used for the generation
     * @return True if the KVCache is generated and saved successfully, false otherwise
     * @throws std::runtime_error if a CUDA operation fails
     */
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

    /*! \brief Set the random seed used when initializing the action diffusion noise trajectory
     *  \param seed Random seed value; has no effect if no action runner is loaded
     */
    void setActionNoiseSeed(int32_t seed) noexcept;

    //! Get LLM prefill stage metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const noexcept
    {
        return mPrefillMetrics;
    }

    //! Get Eagle generation stage metrics (only meaningful when draft model is present)
    metrics::EagleGenerationMetrics const& getEagleGenerationMetrics() const noexcept
    {
        return mEagleGenerationMetrics;
    }

    //! Get vanilla generation stage metrics (only meaningful when no draft model / vanilla path)
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const noexcept
    {
        return mGenerationMetrics;
    }

    //! Get multimodal metrics (returns empty metrics if no multimodal runner)
    metrics::MultimodalMetrics getMultimodalMetrics() const noexcept
    {
        return mVisionRunner ? mVisionRunner->getMultimodalMetrics()
            : mAudioRunner   ? mAudioRunner->getMultimodalMetrics()
                             : metrics::MultimodalMetrics{};
    }

    //! Get the embedding table (for Talker streaming pipeline)
    rt::Tensor const& getEmbeddingTable() const
    {
        return mEmbedding.table;
    }

    //! @brief Get a base model hidden-states buffer for the requested layer index.
    //!
    //! Buffers are owned by the runtime and reused across requests. Layer 0 corresponds to
    //! the post-multimodal input embeddings (backed up before the decode loop reshapes them);
    //! other layer indices correspond to engine-output hidden states (e.g. acceptHiddenLayer
    //! for the Qwen3-Omni Talker, or future MTP layers).
    //!
    //! Lifetime contract:
    //!   - Buffers are sized to {maxRuntimeBatchSize, maxSupportedInputLength, hiddenSize}.
    //!   - Contents are cleared (overwritten) at the start of each handleRequest() call and
    //!     remain valid until the next handleRequest() begins. The buffer is reshaped to
    //!     {activeBatchSize, prefillLength, hiddenSize} for the most recent request — use
    //!     getBaseModelPrefillLength() to query the valid prefill length.
    //!   - The caller is responsible for consuming the data within that window.
    //!
    //! @param layerIdx Layer index. 0 = input embeddings (post-multimodal); other indices are
    //!                 model-specific (e.g. acceptHiddenLayer for Qwen3-Omni Talker).
    //! @return Pointer to the buffer, or nullptr if no buffer is registered for that layer.
    rt::Tensor const* getBaseModelHiddenStates(int32_t layerIdx) const noexcept
    {
        auto it = mHiddenStatesRegistry.find(layerIdx);
        return it != mHiddenStatesRegistry.end() ? it->second : nullptr;
    }

    //! @brief Number of valid prefill tokens in the hidden-states buffers from the most
    //! recent handleRequest() call. Returns 0 if no hidden-states output was requested.
    int32_t getBaseModelPrefillLength() const noexcept
    {
        return mLastPrefillLength;
    }

    //! @brief Per-batch input token IDs from the most recent handleRequest() call.
    //! Cleared at the start of each handleRequest(); valid until the next one begins.
    std::vector<std::vector<int32_t>> const& getBaseModelInputTokenIds() const noexcept
    {
        return mLastInputTokenIds;
    }

    //! @brief Check if draft model is loaded and spec-decode is available
    bool hasDraftModel() const noexcept
    {
        return mDraftEngineRunner != nullptr;
    }

private:
    //! @brief Common initialization logic shared between both constructors
    void initializeCommon(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<EagleDraftingConfig> const& draftingConfig, cudaStream_t stream);

    rt::Tensor mSharedExecContextMemory{};              //!< Shared device memory for all execution contexts
    int32_t mMaxRuntimeBatchSize{1};                    //!< Maximum runtime batch size
    std::optional<EagleDraftingConfig> mDraftingConfig; //!< Eagle drafting configuration (nullopt = no draft)
    LLMEngineRunnerConfig mBaseEngineConfig;            //!< Base engine configuration
    std::optional<EagleDraftEngineRunnerConfig> mDraftEngineConfig; //!< Draft engine configuration (nullopt = no draft)

    std::unique_ptr<LLMEngineRunner> mBaseEngineRunner;            //!< Base model engine runner
    std::unique_ptr<EagleDraftEngineRunner> mDraftEngineRunner;    //!< Draft model engine runner (nullptr = no draft)
    std::unique_ptr<MultimodalRunner> mVisionRunner{nullptr};      //!< Vision multimodal runner (optional)
    std::unique_ptr<MultimodalRunner> mAudioRunner{nullptr};       //!< Audio multimodal runner (optional)
    std::unique_ptr<Alpamayo1ActionRunner> mActionRunner{nullptr}; //!< Action/diffusion head runner (optional)
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;              //!< Tokenizer
    hash_utils::HashMap<std::tuple<std::string, std::string>, SystemPromptKVCache>
        mSystemPromptKVCacheBase; //!< System prompt KVCache for base model
    hash_utils::HashMap<std::tuple<std::string, std::string>, SystemPromptKVCache>
        mSystemPromptKVCacheDraft;         //!< System prompt KVCache for draft model
    std::string mEmptyLoraWeightsName{""}; //!< Empty LoRA weights name for default case

    // Pre-define key runtime GPU tensors and initialize them during construction.
    // [1] I/O Tensors to work with base and eagle draft engine.
    EmbeddingData mEmbedding;                 //!< Embedding table [vocabSize, hiddenSize] and optional FP8 scales
    rt::Tensor mIdsInput;                     //!< Input token IDs (used for embedding lookup)
    rt::Tensor mInputsEmbeds;                 //!< Input embeddings (after embedding lookup)
    std::vector<rt::Tensor> mDeepstackEmbeds; //!< Deepstack embeddings for Qwen3-VL (one per feature)
    rt::Tensor mContextLengthsInput;
    rt::Tensor mLogitsOutput;
    rt::Tensor mDraftTreeSize;
    rt::Tensor mDraftTreeMask;
    rt::Tensor mBaseHiddenStatesOutput;
    // Distinguish draft hidden states input and output since we cannot easily
    // Perform inplace update for hidden states between drafting steps.
    rt::Tensor mDraftHiddenStatesInput;
    rt::Tensor mDraftHiddenStatesOutput;

    // [2] Sampling workspace and output tensors that used across all the sampling operations.
    rt::Tensor mSamplingWorkspace;
    rt::Tensor mSamplingIndices;
    rt::Tensor mSamplingScores;
    rt::Tensor mBaseVocabMappingTable; // Vocab mapping table for base model reduced vocab (empty if not used)

    // [3] Data structures used during Draft tree constructions (only allocated when draft model present).
    // Data tables that store the data structure that can completely describe a multi-layer draft tree.
    rt::Tensor mDraftTokenIdsFullTable;
    rt::Tensor mDraftTokenScoreFullTable;
    rt::Tensor mDraftTokenPredecessorFullTable;
    // Store conversion table (offset) to map from draft-model vocab token id to the original token id.
    // base_id = draft_id + mapping_table[draft_id]
    rt::Tensor mDraftVocabMappingTable;

    rt::Tensor mDraftTreeRootTokenId;
    rt::Tensor mDraftTokenIdsTable;
    rt::Tensor mDraftTokenScoresTable;
    rt::Tensor mDraftTokenIntermediateScores;
    rt::Tensor mDraftTokenIntermediateParents;

    // [4] Data structures that used during base model verification (only allocated when draft model present).
    rt::Tensor mAcceptedTokenIds;
    rt::Tensor mAcceptedTokenIndices;
    rt::Tensor mAcceptLength;

    // [5] Batch eviction support tensors.
    rt::Tensor mDeviceBatchMapping;

    // [6] Host pinned memory tensors for optimized CPU-GPU memory transfers
    rt::Tensor mHostPackedTokenIds;      //!< Host pinned memory for packed token IDs
    rt::Tensor mHostSelectedTokenIds;    //!< Host pinned memory for selected token IDs from sampling
    rt::Tensor mHostAcceptLengths;       //!< Host pinned memory for accept lengths from verification
    rt::Tensor mHostAcceptedTokenIds;    //!< Host pinned memory for accepted token IDs
    rt::Tensor mHostReuseKVCacheLengths; //!< Host pinned memory for reuse KV cache lengths

    // [7] Multimodal support tensors for audio/image token indexing
    rt::Tensor mMultimodalIndices; //!< Multimodal indices tensor [batchSize, seqLen] for audio/image embeddings

    // [8] Base model hidden states portal (Qwen3-Omni audio generation, future MTP).
    //     Buffers are pre-allocated to {maxBS, maxISL, H} and reshaped per request.
    //     mHiddenStatesRegistry maps layer index → buffer; populated per handleRequest().
    //     See getBaseModelHiddenStates() for the lifetime contract.
    rt::Tensor mOutputHiddenStates{};  //!< Engine-output hidden states (layer N = acceptHiddenLayer)
    rt::Tensor mPrefillEmbedsBackup{}; //!< Layer-0 input embeddings backup (post-multimodal)
    std::unordered_map<int32_t, rt::Tensor const*> mHiddenStatesRegistry; //!< Per-request layer→buffer map
    int32_t mLastPrefillLength{0};                                        //!< Valid prefill length in buffers
    std::vector<std::vector<int32_t>> mLastInputTokenIds;                 //!< Per-batch input token IDs

    //! @brief Restore recurrent/conv states from a cached system prompt.
    void restoreRecurrentStates(int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream);

    //! @brief Zero all recurrent/conv states for a given batch index.
    void zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    // Key functions to drive the spec-decode runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    //! @throws std::runtime_error if a CUDA error occurs
    bool runBaseModelPrefill(SpecDecodeInferenceContext& context);

    //! Validate request shape/runtime compatibility.
    bool validateRequestConfig(LLMGenerationRequest const& request);

    //! Prepare per-request runtime state for models built with multimodal support.
    //! Runs multimodal preprocessing when audio or vision inputs are present.
    //! For text-only requests on MRope-based multimodal models, restores text-only RoPE state
    //! and clears stale multimodal request state.
    bool multiModalRuntimePreprocess(
        LLMGenerationRequest const& request, SpecDecodeInferenceContext& context, cudaStream_t stream);

    // Consume the base model hidden states and input token of the sequence. Produce the draft hidden states and logits
    // for the last token of the sequence.
    //! @throws std::runtime_error if tensor shapes do not match, or a CUDA error occurs
    bool runDraftModelPrefill(SpecDecodeInferenceContext& context);

    // Consume the draft hidden states and logits for the last token of the sequence. Produce a speculative draft tree
    // that described by a sequence of draft tokens and tree mask that describe the tree structure.
    //! @throws std::runtime_error if tensor shapes are invalid, or a CUDA operation fails
    bool constructDraftTree(SpecDecodeInferenceContext& context);

    // Consume the speculative draft tree, produce selected tokens and corresponding hidden states.
    //! @throws std::runtime_error if tensor shapes are invalid, or a CUDA operation fails
    bool runBaseModelVerification(SpecDecodeInferenceContext& context);

    // Consume the selected tokens and base model hidden state, produce the draft hidden states and logits for the last
    // token of the accepted sequence.
    //! @throws std::runtime_error if a CUDA operation fails
    bool runDraftModelAcceptToken(SpecDecodeInferenceContext& context);

    // Consume the token sequence & KVCache to produce the next token directly.
    bool runVanillaDecoding(SpecDecodeInferenceContext& context);

    // Consume system prompt, produce the hash table of system prompt KVCache if kv cache reuse is enabled.
    //! @throws std::runtime_error if a CUDA operation fails
    bool genAndSaveSystemPromptKVCache(SpecDecodeInferenceContext& context, int32_t genAndSaveBatchIdx);

    // Consume batched input ids and the hash table of system prompt KVCache, produce the padded input ids and input
    // lengths. Instantiate the KVCache from the hash table if the system prompt has been cached.
    //! @throws std::runtime_error if system prompt is malformed
    bool setUpForPrefillExecution(SpecDecodeInferenceContext& context);

    // Batch eviction support
    //! @brief Perform batch eviction
    //! @param context Inference context
    //! @return True on success, false on failure
    //! @throws std::runtime_error if a CUDA error occurs
    bool performBatchEvict(SpecDecodeInferenceContext& context);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::EagleGenerationMetrics mEagleGenerationMetrics;
    metrics::LLMGenerationMetrics mGenerationMetrics; //!< Vanilla generation metrics (used when no spec-decode)
};

} // namespace rt
} // namespace trt_edgellm
