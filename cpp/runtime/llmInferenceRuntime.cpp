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

#include "llmInferenceRuntime.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/mathUtils.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "kernels/speculative/batchEvictKernels.h"
#include "multimodal/multimodalRunner.h"
#include "multimodal/qwenViTRunner.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/llmRuntimeUtils.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace
{
//! Optimization-profile indices for the composable stack. Profile 0 is prefill, profile 1 is decode
//! (including speculative tree-verification / proposal / accept). These match the profile layout baked
//! into the engines by `llmBuilder`.
constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};

} // namespace

namespace rt
{

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, SpecDecodeDraftingConfig const& draftingConfig,
    cudaStream_t stream)
{
    initializeCommon(engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, stream);
}

LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    initializeCommon(engineDir, multimodalEngineDir, loraWeightsMap, std::nullopt, stream);
}

void LLMInferenceRuntime::initializeCommon(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap,
    std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream)
{
    // -----------------------------------------------------------------------
    // 1. Load shared embedding table (shared between base and draft models).
    // -----------------------------------------------------------------------
    std::filesystem::path const embeddingPath = std::filesystem::path(engineDir) / "embedding.safetensors";
    mEmbedding = loadEmbeddingTable(embeddingPath, stream);

    // -----------------------------------------------------------------------
    // 2. Parse engine configurations and attach user drafting (bundle factory
    //    performs cross-engine consistency and drafting-vs-capacity checks).
    // -----------------------------------------------------------------------
    std::filesystem::path const baseEnginePath = draftingConfig.has_value()
        ? std::filesystem::path(engineDir) / "eagle_base.engine"
        : std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const baseConfigPath = draftingConfig.has_value()
        ? std::filesystem::path(engineDir) / "base_config.json"
        : std::filesystem::path(engineDir) / "config.json";
    std::optional<std::filesystem::path> const draftConfigPath = draftingConfig.has_value()
        ? std::optional<std::filesystem::path>{std::filesystem::path(engineDir) / "draft_config.json"}
        : std::nullopt;

    mDeployment = createDeploymentConfig(baseConfigPath, draftConfigPath, draftingConfig);

    ELLM_CHECK(mDeployment.base.numDeepstackFeatures <= 0 || !multimodalEngineDir.empty(),
        "--multimodalEngineDir is required for VLM engine.");

    // -----------------------------------------------------------------------
    // 3. Construct Runners (registries built internally from the parsed configs).
    // -----------------------------------------------------------------------
    try
    {
        std::optional<int32_t> const specDecodeBaseOutputHiddenDim = mDeployment.specConfig.has_value()
            ? std::optional<int32_t>{mDeployment.specConfig->baseOutputHiddenDim}
            : std::nullopt;
        mBaseExecutor = EngineExecutor::createForLLM(baseEnginePath, mDeployment.base, specDecodeBaseOutputHiddenDim);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize base EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize base EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Base EngineExecutor successfully loaded from %s.", baseEnginePath.c_str());

    // -----------------------------------------------------------------------
    // 4. Validate engine binding dtypes against the parsed configs.
    // -----------------------------------------------------------------------
    validateAgainstEngine(mDeployment.base, *mBaseExecutor, "base");

    // -----------------------------------------------------------------------
    // 5. Set runtime batch size.
    // -----------------------------------------------------------------------
    mMaxRuntimeBatchSize = mDeployment.maxRuntimeBatchSize();
    LOG_INFO("Runtime batch size set to: %d (from engine bundle)", mMaxRuntimeBatchSize);

    // -----------------------------------------------------------------------
    // 6. SharedResources + PipelineIO. PipelineIO is held via unique_ptr so
    //    its address is stable for the TensorMap pointers below (TensorMap
    //    stores non-owning Tensor* into PipelineIO members).
    // -----------------------------------------------------------------------
    bool const hasDraft = draftingConfig.has_value();
    if (hasDraft)
    {
        mSharedResources
            = SharedResources::createForSpecDecode(mDeployment, mMaxRuntimeBatchSize, loraWeightsMap, stream);
        mPipelineIO
            = std::make_unique<PipelineIO>(PipelineIO::createForSpecDecode(mDeployment, mMaxRuntimeBatchSize, stream));
    }
    else
    {
        mSharedResources = SharedResources::createForLLM(mDeployment.base, loraWeightsMap, stream);
        mPipelineIO = std::make_unique<PipelineIO>(PipelineIO::createForLLM(mDeployment.base, stream));
    }
    // Externalized model weights: the SharedResources factory only allocates an
    // empty manager. Load external weights and validate against engine inputs.
    // This handles the base engine; the spec-decode draft engine loads its own
    // external weights from draft_config.json inside the EAGLE/MTP decoder.
    mSharedResources->externalWeightManager->load(std::filesystem::path(engineDir), baseConfigPath, stream);
    mSharedResources->externalWeightManager->validateAgainstEngine(*mBaseExecutor, "base");

    // -----------------------------------------------------------------------
    // 7. Build base TensorMap (kvCacheIndex=0) and publish static external
    //    weight bindings. Speculative decoders add tree-mask / position IDs
    //    to this same map further down.
    // -----------------------------------------------------------------------
    buildTensorMap(mBaseTensorMap, *mPipelineIO, *mSharedResources, mDeployment.base, /*kvCacheIndex=*/0);
    mSharedResources->externalWeightManager->registerTensorMapEntries(mBaseTensorMap);

    // -----------------------------------------------------------------------
    // 8. LoRA: register engine bindings and seed the base tensor map with
    //    dummy / active adapter tensors. Only the base engine carries LoRA
    //    bindings — draft does not.
    // -----------------------------------------------------------------------
    if (mSharedResources->loraManager)
    {
        mSharedResources->loraManager->initializeEngineBindings(*mBaseExecutor);
        mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
    }

    // -----------------------------------------------------------------------
    // 9. Preprocessors.
    // -----------------------------------------------------------------------
    mStepPreparer = std::make_unique<StepPreparer>(mDeployment.base);
    mEmbeddingPre = std::make_unique<EmbeddingPreprocessor>(mEmbedding, mDeployment.base);
    if (mDeployment.base.numDeepstackFeatures > 0)
    {
        mDeepstack = std::make_unique<DeepstackBinding>(mPipelineIO->deepstackEmbeds, mSharedResources->zeroBuffer);
    }

    // -----------------------------------------------------------------------
    // 10. Allocate runtime-local tensors (sampling workspace, host pinned scratch,
    //     batch-eviction mapping). Strategy-specific tensors are owned by strategies.
    // -----------------------------------------------------------------------
    int32_t const effectiveMaxProposalSize = hasDraft ? mDeployment.effectiveMaxDraftProposalSize() : 1;
    int32_t const effectiveDraftTopK = hasDraft ? draftingConfig->draftingTopK : 1;
    int32_t const maxInputLength = hasDraft
        ? std::max(mDeployment.base.maxSupportedInputLength, mDeployment.draft->maxSupportedInputLength)
        : mDeployment.base.maxSupportedInputLength;
    int32_t const maxSamplingSize = hasDraft ? std::max(mMaxRuntimeBatchSize * effectiveMaxProposalSize,
                                                   mMaxRuntimeBatchSize * effectiveDraftTopK * effectiveDraftTopK)
                                             : mMaxRuntimeBatchSize;

    // Reserve enough workspace for sampling, accounting for batch dimension in draft proposal stage.
    // Always include vanilla sampling workspace size because per-request disable_spec_decode
    // can fall back to topK/topP sampling even when draft is loaded.
    int32_t const vanillaSamplingWorkspaceSize
        = static_cast<int32_t>(getTopKtopPSamplingWorkspaceSize(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize,
            SamplingParams(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize, 1.0f, 0, 0.9f)));
    int32_t const maxSamplingWorkspaceSize = hasDraft
        ? std::max({vanillaSamplingWorkspaceSize,
              static_cast<int32_t>(
                  getSelectAllTopKWorkspaceSize(mMaxRuntimeBatchSize, mDeployment.base.outputVocabSize, 1)),
              static_cast<int32_t>(getSelectAllTopKWorkspaceSize(
                  mMaxRuntimeBatchSize * effectiveDraftTopK, mDeployment.draft->outputVocabSize, effectiveDraftTopK))})
        : vanillaSamplingWorkspaceSize;

    try
    {
        mIdsInput = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mIdsInput");

        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
            "LLMInferenceRuntime::mSamplingWorkspace");
        mSamplingIndices = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceRuntime::mSamplingIndices");
        mSamplingScores = rt::Tensor(
            {maxSamplingSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "LLMInferenceRuntime::mSamplingScores");

        // Batch mapping tensor for batch eviction.
        mDeviceBatchMapping = rt::Tensor(
            {mMaxRuntimeBatchSize}, rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceRuntime::mDeviceBatchMapping");

        mHostPackedTokenIds = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostPackedTokenIds");
        mHostSelectedTokenIds = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostSelectedTokenIds");
        mHostReuseKVCacheLengths = rt::Tensor({mMaxRuntimeBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostReuseKVCacheLengths");

        // Pre-allocate multimodal indices tensor (used for audio/vision embedding lookup).
        mMultimodalIndices = rt::Tensor({mMaxRuntimeBatchSize, maxInputLength}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mMultimodalIndices");
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate runtime tensors: %s", e.what());
        throw std::runtime_error("Failed to allocate runtime tensors: " + std::string(e.what()));
    }
    LOG_INFO("Runtime tensors successfully allocated.");

    // -----------------------------------------------------------------------
    // 11. Load optional base model reduced-vocab mapping table.
    // -----------------------------------------------------------------------
    if (mDeployment.base.reducedVocabSize > 0)
    {
        LOG_INFO("Loading vocabulary mapping table for base model reduced vocab size: %d -> %d",
            mDeployment.base.reducedVocabSize, mDeployment.base.vocabSize);
        std::filesystem::path const vocabMapPath = std::filesystem::path(engineDir) / binding_names::kVocabMapFileName;

        std::vector<rt::Tensor> vocabMapTensors;
        ELLM_CHECK(safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream),
            "Failed to load " + std::string(binding_names::kVocabMapFileName) + " from model directory: " + engineDir);

        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == mDeployment.base.reducedVocabSize,
            "vocab_map tensor length should match base model reduced vocab size");
        mBaseVocabMappingTable = std::move(vocabMapTensors[0]);
        LOG_INFO("Base model vocabulary mapping table successfully loaded.");
    }

    // -----------------------------------------------------------------------
    // 12. Tokenizer.
    // -----------------------------------------------------------------------
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    ELLM_CHECK(mTokenizer->loadFromHF(engineDir), "Failed to load tokenizer from model directory: " + engineDir);
    LOG_INFO("Tokenizer successfully loaded from model directory: %s", engineDir.c_str());

    // -----------------------------------------------------------------------
    // 13. Decoding strategies.
    // -----------------------------------------------------------------------
    buildDecodingRuntimeContext();
    mDecoderRegistry = std::make_unique<DecoderRegistry>(
        *mDecodingRuntimeContext, DecoderRegistryConfig{std::filesystem::path(engineDir), draftingConfig, stream});

    // -----------------------------------------------------------------------
    // 14. Optional multimodal runners.
    // -----------------------------------------------------------------------
    if (!multimodalEngineDir.empty())
    {
        auto tryLoadRunner = [&](std::string const& dir, std::string const& name) -> std::unique_ptr<MultimodalRunner> {
            try
            {
                LOG_DEBUG("Attempting to load %s runner from %s", name.c_str(), dir.c_str());
                auto runner = MultimodalRunner::create(
                    dir, mDeployment.base.maxSupportedBatchSize, mDeployment.base.maxKVCacheCapacity, stream);
                LOG_INFO("%s runner successfully initialized", name.c_str());
                return runner;
            }
            catch (std::exception const& e)
            {
                LOG_DEBUG("Failed to load %s runner from %s: %s", name.c_str(), dir.c_str(), e.what());
                return nullptr;
            }
        };

        mAudioRunner = tryLoadRunner(multimodalEngineDir + "/audio", "Audio");
        mVisionRunner = tryLoadRunner(multimodalEngineDir + "/visual", "Visual");
        if (!mVisionRunner)
        {
            mVisionRunner = tryLoadRunner(multimodalEngineDir, "Vision");
        }

        // At least one multimodal runner must be available
        ELLM_CHECK(mAudioRunner || mVisionRunner, "No valid multimodal engine found in " + multimodalEngineDir);

        // Try to load action expert from multimodalEngineDir/action
        try
        {
            std::string actionDir = multimodalEngineDir + "/action";
            LOG_INFO("Attempting to load Action runner from %s", actionDir.c_str());
            mActionRunner = std::make_unique<Alpamayo1ActionRunner>(
                actionDir, stream, mSharedResources->cacheManagers[0]->getKVCacheManager().getConfig());
            LOG_INFO("Alpamayo 1 action expert loaded.");
        }
        catch (std::exception const& e)
        {
            LOG_INFO("Failed to load Action runner from %s: %s", (multimodalEngineDir + "/action").c_str(), e.what());
        }

        // Validate that the action engine's max KV cache capacity matches the LLM engine's.
        if (mActionRunner)
        {
            int32_t const actionMaxKVCacheCapacity = mActionRunner->getMaxKVCacheCapacity();
            int32_t const llmMaxKVCacheCapacity = mDeployment.base.maxKVCacheCapacity;
            ELLM_CHECK(actionMaxKVCacheCapacity == llmMaxKVCacheCapacity,
                format::fmtstr(
                    "Action engine max_kv_cache_capacity (%d) does not match LLM engine max_kv_cache_capacity (%d). "
                    "Re-export and rebuild the action engine with --max_kv_cache_capacity=%d to match the LLM engine.",
                    actionMaxKVCacheCapacity, llmMaxKVCacheCapacity, llmMaxKVCacheCapacity));
        }
    }

    // -----------------------------------------------------------------------
    // 15. Shared execution context memory for all engines (base, optional
    //     draft, and optional vision/audio). All engines execute serially so
    //     they can share a single buffer sized to the max requirement.
    // -----------------------------------------------------------------------
    int64_t const baseContextMemorySize = mBaseExecutor->getRequiredContextMemorySize();
    int64_t const strategyContextMemorySize = mDecoderRegistry ? mDecoderRegistry->getRequiredContextMemorySize() : 0;
    int64_t const visionContextMemorySize = mVisionRunner ? mVisionRunner->getRequiredContextMemorySize() : 0;
    int64_t const audioContextMemorySize = mAudioRunner ? mAudioRunner->getRequiredContextMemorySize() : 0;
    int64_t const actionContextMemorySize = mActionRunner ? mActionRunner->getRequiredContextMemorySize() : 0;
    int64_t const sharedContextMemorySize = std::max({baseContextMemorySize, strategyContextMemorySize,
        visionContextMemorySize, audioContextMemorySize, actionContextMemorySize});
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "LLMInferenceRuntime::mSharedExecContextMemory");
    mBaseExecutor->setContextMemory(mSharedExecContextMemory);
    if (mDecoderRegistry)
    {
        mDecoderRegistry->setContextMemory(mSharedExecContextMemory);
    }
    if (mVisionRunner)
    {
        mVisionRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mAudioRunner)
    {
        mAudioRunner->setContextMemory(mSharedExecContextMemory);
    }
    if (mActionRunner)
    {
        mActionRunner->setContextMemory(mSharedExecContextMemory);
    }
    LOG_INFO(
        "Setup shared execution context memory: %zu bytes (base requires: %zu, strategy requires: %zu, vision "
        "requires: "
        "%zu, audio requires: %zu, action requires: %zu)",
        static_cast<size_t>(sharedContextMemorySize), static_cast<size_t>(baseContextMemorySize),
        static_cast<size_t>(strategyContextMemorySize), static_cast<size_t>(visionContextMemorySize),
        static_cast<size_t>(audioContextMemorySize), static_cast<size_t>(actionContextMemorySize));
}

void LLMInferenceRuntime::buildDecodingRuntimeContext()
{
    BaseEngineResources baseResources{*mBaseExecutor, mBaseTensorMap, *mSharedResources,
        *mSharedResources->cacheManagers[0], *mPipelineIO, [this](InferenceDims const& dims, cudaStream_t stream) {
            return captureBaseGraphWithLoraFanout(dims, stream);
        }};
    PreprocessResources preprocessResources{*mStepPreparer, *mEmbeddingPre, mEmbedding, mIdsInput, mDeepstack.get()};
    SamplingBuffers sampling{mSamplingWorkspace, mSamplingIndices, mSamplingScores, mBaseVocabMappingTable,
        mHostPackedTokenIds, mHostSelectedTokenIds};
    mDecodingRuntimeContext.reset(new DecodingRuntimeContext{
        mDeployment, mMaxRuntimeBatchSize, baseResources, preprocessResources, *mTokenizer, sampling});
}

void LLMInferenceRuntime::setActionNoiseSeed(int32_t seed) noexcept
{
    if (mActionRunner)
    {
        mActionRunner->setNoiseSeed(seed);
    }
}

bool LLMInferenceRuntime::handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response,
    cudaStream_t stream, bool outputThinkerEmbeddings)
{
    // Clear per-request portal state. Buffers themselves stay allocated and are
    // reshaped/overwritten when populated below — see getBaseModelHiddenStates() contract.
    mHiddenStatesRegistry.clear();
    mLastPrefillLength = 0;
    mLastInputTokenIds.clear();

    // Clear per-request response state. On failure (early return) the four vectors
    // stay empty; on success they are repopulated together below to matched sizes.
    response.outputIds.clear();
    response.outputTexts.clear();
    response.outputTrajectories.clear();
    response.finishReasons.clear();

    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    std::string const& loraWeightsName = request.loraWeightsName;

    if (!validateRequestConfig(request))
    {
        return false;
    }

    if (!validateStreamingSubmission(request))
    {
        return false;
    }

    DecodingStrategy& decodingStrategy = mDecoderRegistry->select(request);
    bool const enableSpecDecode = decodingStrategy.isSpeculative();

    // Speculative decoding only supports greedy; override non-default sampling params.
    bool const hasNonDefaultSampling
        = (request.topK > 1 || request.topP < 1.0f || std::fabs(request.temperature - 1.0f) > 1e-3f);
    if (enableSpecDecode && hasNonDefaultSampling)
    {
        LOG_WARNING("Spec-decode active: overriding sampling params to greedy (ignoring temp/topK/topP).");
    }

    int32_t maxGenerateLength = request.maxGenerateLength;

    // Apply chat template for all requests (common for both multimodal and non-multimodal)
    request.formattedRequests.resize(activeBatchSize);
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Apply chat template to populate both formatted system prompt and full formatted prompt
        mTokenizer->applyChatTemplate(request.requests[i], request.formattedRequests[i], request.applyChatTemplate,
            request.addGenerationPrompt, request.enableThinking);
    }

    DecodingInferenceContext context;
    context.initialize(
        activeBatchSize, maxGenerateLength, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName, stream);
    bool const supportsMultimodalInput
        = (mAudioRunner != nullptr) || (mVisionRunner != nullptr) || (mActionRunner != nullptr);

    if (supportsMultimodalInput)
    {
        if (!multiModalRuntimePreprocess(request, context, stream))
        {
            return false;
        }
    }
    else
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
            context.rawBatchedInputIds.emplace_back(
                mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            if (context.rawBatchedInputIds[i].empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
        }
    }

    // Forward sampling params to context; spec-decode forces greedy.
    context.temperature = enableSpecDecode ? 1.0f : request.temperature;
    context.topP = enableSpecDecode ? 1.0f : request.topP;
    context.topK = enableSpecDecode ? 0 : request.topK;
    context.outputThinkerEmbeddings = outputThinkerEmbeddings;
    context.onTokenGenerated = request.onTokenGenerated;

    // Forward per-slot stop strings and cache the longest length to avoid
    // recomputing it on every emitChunks iteration.
    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        context.stopStringsPerSlot[i] = request.requests[i].stopStrings;
        size_t maxLen = 0;
        for (auto const& s : request.requests[i].stopStrings)
        {
            if (s.size() > maxLen)
            {
                maxLen = s.size();
            }
        }
        context.slotStreams[i].maxStopLen = maxLen;
    }

    // The spec-decode path needs extra KV reserve for draft tokens during verification.
    constexpr int32_t kDRAFT_KVCACHE_RESERVE_LENGTH{100};
    int32_t const kvCacheCapacity = enableSpecDecode
        ? std::min(mDeployment.base.maxKVCacheCapacity, mDeployment.draft->maxKVCacheCapacity)
        : mDeployment.base.maxKVCacheCapacity;
    int32_t const kvcReserve = enableSpecDecode ? kDRAFT_KVCACHE_RESERVE_LENGTH : 0;

    // In production, the system-prompt KV cache is saved during warm-up.
    // We disable profiling here to make benchmarking closer to production inference result.
    bool profilingEnabled = getProfilingEnabled();
    if (profilingEnabled)
    {
        setProfilingEnabled(false);
    }

    // Generate system prompt KVCache for each sequence in the batch
    if (request.saveSystemPromptKVCache)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            bool const saveCacheStatus = genAndSaveSystemPromptKVCache(context, i);
            if (!saveCacheStatus)
            {
                LOG_WARNING(
                    "Failed to save system prompt KVCache for request %d in batch. "
                    "Continue to handle the request without saving the system prompt KVCache.",
                    i);
            }
        }
    }

    if (profilingEnabled)
    {
        setProfilingEnabled(true);
    }

    // Conduct the preparation work to handle a new set of sequences, including inputIds packing, input/output tensor
    // preparation, reset the KVCache state, and apply reused prefix KVCache if available.
    if (!setUpForPrefillExecution(context, decodingStrategy))
    {
        LOG_ERROR("Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    // ── Streaming setup ──────────────────────────────────────────────────────
    // Attach first, record in slotStreams only on success — a throw from attach
    // keeps foreign channels out of the finalizer's reach. Seed sentTokenCount
    // to the prompt length so streaming emits only generated tokens.
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (request.streamChannels.empty() || !request.streamChannels[i])
        {
            continue;
        }
        attachStreamChannel(request.streamChannels[i], context.batchIndexMapping[i]);
        auto& slot = context.slotStreams[i];
        slot.channel = request.streamChannels[i];
        slot.sentTokenCount = context.tokenIds[i].size();
        slot.lastEmittedTokenCount = slot.sentTokenCount;
    }
    StreamChannelFinalizer streamFinalizer(context, *mTokenizer);

    int32_t const clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(
        context.effectivePrefillLengths, request.maxGenerateLength, kvCacheCapacity, kvcReserve);
    if (clampedMaxGenerateLength != context.maxGenerateLength)
    {
        context.maxGenerateLength = clampedMaxGenerateLength;
        LOG_WARNING("Reduce max generation length to %d", context.maxGenerateLength);
    }
    if (context.maxGenerateLength <= 0)
    {
        LOG_ERROR("Insufficient KV cache capacity for generation for this request.");
        return false;
    }

    // Prefill from the base model; subsequent iterations are delegated to the selected strategy.
    bool const prefillStatus = runBaseModelPrefill(context);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute prefill step for base model.");
        return false;
    }

    // Populate the base-model hidden-states portal so consumers (Qwen3-Omni Talker via
    // streaming callback or post-handleRequest sequential consumer) can fetch the buffers
    // by layer index. See getBaseModelHiddenStates() / getBaseModelInputTokenIds() for the
    // lifetime contract.
    int32_t prefillSequenceLength = 0;
    if (outputThinkerEmbeddings)
    {
        prefillSequenceLength
            = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());

        // Layer 0: back up post-multimodal input embeddings before the decode loop reshapes
        // mPipelineIO->inputsEmbeds to {BS,1,H} (scrambling the contiguous {BS,prefillLen,H} layout).
        // The backup buffer lives on PipelineIO and is lazy-allocated at maxISL on first
        // streaming request, then reshaped per request — see getBaseModelHiddenStates() lifetime contract.
        rt::Tensor& prefillEmbedsBackup = mPipelineIO->prefillEmbedsBackup;
        if (prefillEmbedsBackup.isEmpty())
        {
            prefillEmbedsBackup = rt::Tensor(
                {mMaxRuntimeBatchSize, mDeployment.base.maxSupportedInputLength, mDeployment.base.hiddenSize},
                rt::DeviceType::kGPU, DataType::kHALF, "PipelineIO::prefillEmbedsBackup");
        }
        check::check(prefillEmbedsBackup.reshape({activeBatchSize, prefillSequenceLength, mDeployment.base.hiddenSize}),
            "Tensor reshape failed");
        size_t const prefillBytes = static_cast<size_t>(activeBatchSize) * prefillSequenceLength
            * mDeployment.base.hiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(prefillEmbedsBackup.rawPointer(), mPipelineIO->inputsEmbeds.rawPointer(),
            prefillBytes, cudaMemcpyDeviceToDevice, stream));

        mLastPrefillLength = prefillSequenceLength;
        mLastInputTokenIds = context.rawBatchedInputIds;
        mHiddenStatesRegistry[0] = &prefillEmbedsBackup;
        // Layer N (acceptHiddenLayer) is registered after the engine-output reshape below.
    }

    // Lambda to check if all batches are finished
    auto checkAllFinished = [&]() {
        // Check if all batches have been evicted
        if (context.activeBatchSize == 0)
        {
            return true;
        }
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (!context.finishedStates[i])
            {
                return false;
            }
        }
        return true;
    };

    // Used for Alpamayo 1
    int32_t trajFutureStartId = 0;
    if (mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
    {
        trajFutureStartId = static_cast<int32_t>(mTokenizer->getTokenId("<|traj_future_start|>"));
    }

    // Lambda to update finish states based on EOS and max_length. Latches
    // terminalReason atomically with the state flip — the !finishedStates guard
    // keeps first-writer-wins semantics relative to applyCancellationToFinishStates.
    auto updateFinishStates = [&]() {
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            if (context.finishedStates[i])
            {
                continue; // Respect first-writer-wins (cancel may have fired).
            }
            auto& s = context.slotStreams[i];
            // terminalReason is set for all slots; non-streaming slots surface it via
            // BatchResult.terminalReason → response.finishReasons.
            if (mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
            {
                if (context.tokenIds[i].size() > 1 && trajFutureStartId >= 0
                    && context.tokenIds[i][context.tokenIds[i].size() - 2] == trajFutureStartId)
                {
                    context.finishedStates[i] = 1;
                    s.terminalReason = FinishReason::kEndId;
                    LOG_DEBUG("Batch %d finished, reason: traj_future_start", i);
                    continue;
                }
            }
            else
            {
                // Check EOS
                if (!context.tokenIds[i].empty() && context.tokenIds[i].back() == mTokenizer->getEosId())
                {
                    context.finishedStates[i] = 1;
                    s.terminalReason = FinishReason::kEndId;
                    LOG_DEBUG("Batch %d finished, reason: EOS", i);
                    continue;
                }
            }
            // Check max length
            if (context.currentGenerateLengths[i] >= context.maxGenerateLength)
            {
                context.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kLength;
                LOG_DEBUG(
                    "Batch %d finished, total tokens=%d, reason: max_length", i, context.currentGenerateLengths[i]);
                continue;
            }
        }

        // Stop-string override pass — runs after EOS/length so it can override
        // kEndId/kLength (user-relevant cause). Cancel/error still win because
        // decodePerSlot skipped the match when those reasons were latched.
        for (int32_t i = 0; i < context.activeBatchSize; ++i)
        {
            auto& s = context.slotStreams[i];
            if (s.stopMatchedThisIter && s.terminalReason != FinishReason::kCancelled
                && s.terminalReason != FinishReason::kError)
            {
                context.finishedStates[i] = 1;
                s.terminalReason = FinishReason::kStopWords;
                LOG_DEBUG("Batch %d finished, reason: stop_words", i);
            }
        }
    };

    // Post-prefill per-iter pipeline:
    //   cancel → decode (emitDelta + stop match) → finalize (EOS/length/stop) → emit
    applyCancellationToFinishStates(context);
    decodePerSlot(context, *mTokenizer);
    updateFinishStates();
    emitChunks(context);

    // If everything finished during prefill, evict once so activeBatchSize reaches 0
    if (checkAllFinished() && context.activeBatchSize > 0)
    {
        bool const batchEvictStatus = performBatchEvict(context, decodingStrategy);
        if (!batchEvictStatus)
        {
            LOG_ERROR("Failed to perform batch eviction.");
            return false;
        }
    }

    while (!checkAllFinished())
    {
        // Observe any consumer cancels at the top of the iteration so they land
        // first in the per-slot terminalReason latch.
        applyCancellationToFinishStates(context);

        if (!decodingStrategy.decodeStep(context))
        {
            LOG_ERROR("Failed to decode tokens with %s decoding strategy.", decodingStrategy.name());
            return false;
        }

        // Per-iter pipeline: decode → finalize finish state → emit chunks.
        decodePerSlot(context, *mTokenizer);
        updateFinishStates();
        emitChunks(context);
        context.generationRound += 1;

        // Perform batch eviction if needed (after verification, before updating finish states)
        bool const batchEvictStatus = performBatchEvict(context, decodingStrategy);
        if (!batchEvictStatus)
        {
            LOG_ERROR("Failed to perform batch eviction.");
            return false;
        }
    }

    if (context.activeBatchSize != 0)
    {
        LOG_ERROR("Eviction failure, there should be no active batch at the end of the inference. activeBatchSize: %d",
            context.activeBatchSize);
        return false;
    }

    // Record metrics - accumulate across all batches (active + evicted)
    int32_t totalReusedTokens = 0;
    int32_t totalComputedTokens = 0;
    int32_t totalGeneratedTokens = 0;
    int32_t totalIterations = 0;

    // Accumulate from completed batches
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t rawPromptLength = static_cast<int32_t>(batchResult.rawBatchedInputIds.size());
        int32_t computedLength = batchResult.effectivePrefillLength;
        totalReusedTokens += (rawPromptLength - computedLength);
        totalComputedTokens += computedLength;
        totalGeneratedTokens += batchResult.generateLength;
        totalIterations += batchResult.actualIterations;
    }

    mPrefillMetrics.recordRun(totalReusedTokens, totalComputedTokens);
    if (enableSpecDecode)
    {
        mSpecDecodeGenerationMetrics.recordRun(totalIterations, totalGeneratedTokens);
    }
    else
    {
        mGenerationMetrics.recordRun(totalGeneratedTokens);
    }

    // Save output ids and decoded texts to response.
    // Maintain original batch order using original batch indices.
    response.outputIds.resize(context.completedBatches.size());
    response.outputTexts.resize(context.completedBatches.size());
    response.outputTrajectories.resize(context.completedBatches.size());
    response.finishReasons.resize(context.completedBatches.size(), FinishReason::kNotFinished);

    // Add outputs from completed batches (using saved original indices)
    for (auto const& [originalIdx, batchResult] : context.completedBatches)
    {
        int32_t genLength = batchResult.generateLength;

        // Log acceptance metrics for evicted batch
        if (enableSpecDecode)
        {
            int32_t const verificationTokens = genLength > 0 ? genLength - 1 : 0;
            float const acceptanceRate = batchResult.actualIterations > 0
                ? static_cast<float>(verificationTokens) / static_cast<float>(batchResult.actualIterations)
                : 0.0f;
            LOG_DEBUG(
                "Batch (completed with SpecDecode, original idx %d) - Acceptance rate: %.3f, Generated tokens: %d, "
                "Iterations: %d",
                originalIdx, acceptanceRate, genLength, batchResult.actualIterations);
        }

        // Extract generated tokens
        int32_t const totalLength = static_cast<int32_t>(batchResult.tokenIds.size());

        check::check(totalLength >= genLength, "Total length should be greater than or equal to generated length");
        response.outputIds[originalIdx] = std::vector<int32_t>(
            batchResult.tokenIds.begin() + (totalLength - genLength), batchResult.tokenIds.end());
        response.outputTexts[originalIdx] = mTokenizer->decode(response.outputIds[originalIdx], true);
        response.finishReasons[originalIdx] = batchResult.terminalReason;

        // Trim this slot's own stop strings from its output text by delegating
        // to applyStopStringMatch with isFinal=true — single source of truth
        // for earliest-position-wins semantics, shared with the streaming path.
        // outputIds is intentionally left intact (full token stream).
        if (originalIdx < static_cast<int32_t>(request.requests.size())
            && !request.requests[originalIdx].stopStrings.empty())
        {
            auto const& slotStops = request.requests[originalIdx].stopStrings;
            size_t maxLen = 0;
            for (auto const& s : slotStops)
            {
                maxLen = std::max(maxLen, s.size());
            }
            auto& text = response.outputTexts[originalIdx];
            auto outcome = applyStopStringMatch(text, slotStops, maxLen, /*isFinal=*/true);
            text = std::move(outcome.emitted);
            if (outcome.stopMatched)
            {
                // emitDelta (incremental) and one-shot Tokenizer::decode can differ at BPE
                // piece boundaries — upgrade the reason if one-shot surfaced a stop the
                // streaming-path matcher missed.
                response.finishReasons[originalIdx] = FinishReason::kStopWords;
            }
        }
    }

    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });
    // If action engine is loaded, run one batched trajectory sample and fill output for all batch items.
    if (hasTrajectoryHistory && mActionRunner && mActionRunner->getModelType() == action::ActionModelType::ALPAMAYO1)
    {
        if (!mVisionRunner)
        {
            LOG_ERROR("Alpamayo1ActionRunner requires a vision runner (e.g. QwenViTRunner) for MRoPE rope deltas.");
            return false;
        }

        multimodal::ModelType const visionType = mVisionRunner->getModelType();
        bool const isQwen3ViT = visionType == multimodal::ModelType::QWEN3_VL;
        if (!isQwen3ViT)
        {
            LOG_ERROR(
                "Alpamayo1ActionRunner requires a Qwen3-VL vision runner but a different vision runner is loaded.");
            return false;
        }
        // MultimodalRunner::create() uses QwenViTRunner only for Qwen3-VL.
        auto* qwenVision = static_cast<rt::QwenViTRunner*>(mVisionRunner.get());
        std::vector<int64_t> const& ropeDeltas = qwenVision->getMropeRopeDeltasPerBatch();
        rt::HybridCacheManager& kvcache = *mSharedResources->cacheManagers[0];
        std::vector<std::vector<rt::FutureTrajectoryPoint>> trajectories
            = mActionRunner->sampleTrajectory(stream, activeBatchSize, kvcache, ropeDeltas);
        if (trajectories.size() != static_cast<size_t>(activeBatchSize))
        {
            LOG_ERROR("Alpamayo1ActionRunner trajectory sampling failed.");
            return false;
        }
        for (size_t i = 0; i < trajectories.size() && i < static_cast<size_t>(activeBatchSize); ++i)
        {
            if (!trajectories[i].empty())
            {
                response.outputTrajectories[i] = std::move(trajectories[i]);
            }
        }
    }

    // Reshape engine-output hidden states to the actual prefill size and register layer N
    // (acceptHiddenLayer) in the portal. The buffers live on PipelineIO; the registry
    // here just records non-owning pointers consumers fetch via getBaseModelHiddenStates().
    if (outputThinkerEmbeddings)
    {
        rt::Tensor& outputHiddenStates = mPipelineIO->outputHiddenStates;
        check::check(outputHiddenStates.reshape({activeBatchSize, prefillSequenceLength, mDeployment.base.hiddenSize}),
            "Tensor reshape failed");
        mHiddenStatesRegistry[request.acceptHiddenLayer] = &outputHiddenStates;
    }

    return true;
}

bool LLMInferenceRuntime::validateRequestConfig(LLMGenerationRequest const& request)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool const hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });

    if (activeBatchSize == 0)
    {
        LOG_ERROR("Empty request with no requests");
        return false;
    }

    if (activeBatchSize > mMaxRuntimeBatchSize)
    {
        LOG_ERROR(
            "Requested batch size %d exceeds maximum supported batch size %d", activeBatchSize, mMaxRuntimeBatchSize);
        return false;
    }
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (request.requests[i].messages.empty())
        {
            LOG_ERROR("Request %d in batch is empty: no messages provided", i);
            return false;
        }
    }
    if (hasAudio && !mAudioRunner)
    {
        LOG_ERROR("Request contains audio input, but this runtime does not have an audio runner.");
        return false;
    }
    if (hasVision && !mVisionRunner)
    {
        LOG_ERROR("Request contains vision input, but this runtime does not have a vision runner.");
        return false;
    }
    if (hasTrajectoryHistory && !mActionRunner)
    {
        LOG_ERROR("Request contains trajectory history input, but this runtime does not have an action runner.");
        return false;
    }

    return true;
}

bool LLMInferenceRuntime::multiModalRuntimePreprocess(
    LLMGenerationRequest const& request, DecodingInferenceContext& context, cudaStream_t stream)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    bool const hasAudio = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.audioBuffers.empty(); });
    bool const hasVision = std::any_of(
        request.requests.begin(), request.requests.end(), [](auto const& req) { return !req.imageBuffers.empty(); });
    bool const hasTrajectoryHistory = std::any_of(request.requests.begin(), request.requests.end(),
        [](auto const& req) { return req.pastTrajectory.has_value(); });

    // Clear request-scoped multimodal state up front so previous requests cannot leak through reused runtime members.
    context.visualEmbeddings = std::nullopt;
    context.audioEmbeddings = std::nullopt;
    context.deepstackFeatures.clear();
    // Treat multimodal indices as request-scoped state. Only request paths that explicitly rebuild
    // mMultimodalIndices for the current request should observe a non-empty tensor downstream.
    check::check(mMultimodalIndices.reshape({0}), "Tensor reshape failed");

    // Mark multimodal preprocessing and inference for NVTX profiling
    NVTX_SCOPED_RANGE(nvtx_multimodal, "MULTIMODAL_PROCESSING", nvtx_colors::ORANGE);

    std::vector<std::vector<int32_t>> batchedInputIds;

    // MRope cos/sin output cache is supplied only for MRope-based runners (QwenViT, Qwen3OmniAudio).
    // Runners with standard RoPE (InternViT, Phi4MMViT) ignore it; see MultimodalRunner::preprocess.
    rt::OptionalOutputTensor mropeCosSinOut = (mDeployment.base.ropeConfig.type == RopeType::kMRope)
        ? rt::OptionalOutputTensor{std::ref(mPipelineIO->mropeCosSin)}
        : std::nullopt;

    // Process audio inputs (if present)
    if (hasAudio && mAudioRunner)
    {
        LOG_INFO("Processing audio inputs");
        if (!mAudioRunner->preprocess(request, batchedInputIds, mTokenizer.get(), mropeCosSinOut, stream))
        {
            LOG_ERROR("Audio preprocessing failed. This request cannot be handled.");
            return false;
        }

        if (!mAudioRunner->infer(stream))
        {
            LOG_ERROR("Audio inference failed. This request cannot be handled.");
            return false;
        }
    }

    // Process vision inputs (if present)
    if (hasVision && mVisionRunner)
    {
        LOG_INFO("Processing vision inputs");
        if (!mVisionRunner->preprocess(request, batchedInputIds, mTokenizer.get(), mropeCosSinOut, stream))
        {
            LOG_ERROR("Vision preprocessing failed. This request cannot be handled.");
            return false;
        }

        if (!mVisionRunner->infer(stream))
        {
            LOG_ERROR("Vision inference failed. This request cannot be handled.");
            return false;
        }
    }

    // Process action inputs (if present)
    if (hasTrajectoryHistory && mActionRunner)
    {
        LOG_INFO("Processing trajectory history inputs");
        if (!mActionRunner->preprocess(request, batchedInputIds, mTokenizer.get()))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Trajectory history preprocessing failed. This request cannot be handled.");
            return false;
        }
    }

    if (!hasAudio && !hasVision)
    {
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            batchedInputIds.push_back(mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, false));
            if (batchedInputIds.back().empty())
            {
                LOG_ERROR("Failed to tokenize input text for request %d in batch", i);
                return false;
            }
        }
        if (mDeployment.base.ropeConfig.type == RopeType::kMRope)
        {
            rt::Tensor& ropeCosSinCache = mPipelineIO->mropeCosSin;
            check::check(ropeCosSinCache.reshape({mDeployment.base.maxSupportedBatchSize,
                             mDeployment.base.maxKVCacheCapacity, mDeployment.base.rotaryDim}),
                "Tensor reshape failed");
            kernel::initializeTextOnlyMRopeCosSin(ropeCosSinCache.dataPointer<float>(),
                mDeployment.base.ropeConfig.rotaryTheta, mDeployment.base.rotaryDim,
                mDeployment.base.maxKVCacheCapacity, mDeployment.base.maxSupportedBatchSize, stream);
        }
    }

    // Get embeddings from independent runners — gate on request having multimodal data,
    // not just runner existence, to avoid leaking stale embeddings from previous requests.
    rt::OptionalInputTensor visionEmbeddings
        = (hasVision && mVisionRunner) ? std::optional{std::ref(mVisionRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensor audioEmbeddings
        = (hasAudio && mAudioRunner) ? std::optional{std::ref(mAudioRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors deepstackFeatures
        = (hasVision && mVisionRunner) ? mVisionRunner->getDeepstackFeatures() : rt::OptionalInputTensors{};

    context.visualEmbeddings = visionEmbeddings;
    context.deepstackFeatures = deepstackFeatures;
    context.audioEmbeddings = audioEmbeddings;

    // Populate system prompts and raw input IDs from batchedInputIds
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        context.systemPrompts[i] = request.formattedRequests[i].formattedSystemPrompt;
        context.rawBatchedInputIds.push_back(batchedInputIds[i]);
    }

    return true;
}

bool LLMInferenceRuntime::runBaseModelPrefill(DecodingInferenceContext& context)
{
    TIME_STAGE(metrics::StageNames::kLLM_PREFILL, context.stream);
    NVTX_SCOPED_RANGE(nvtx_base_prefill,
        ("SPEC_DECODE_BASE_PREFILL[" + std::to_string(context.activeBatchSize) + "]").c_str(), nvtx_colors::BLUE);

    int32_t const activeBatchSize = context.activeBatchSize;
    int32_t const inputIdsLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    int32_t const baseOutputHiddenDim
        = mDeployment.specConfig.has_value() ? mDeployment.specConfig->baseOutputHiddenDim : 0;

    // Reshape IO tensors for this step.
    check::check(mIdsInput.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    check::check(mPipelineIO->hostContextLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(mPipelineIO->inputsEmbeds.reshape({activeBatchSize, inputIdsLength, mDeployment.base.hiddenSize}),
        "Tensor reshape failed");
    check::check(mPipelineIO->outputLogits.reshape({activeBatchSize, mDeployment.base.outputVocabSize}),
        "Tensor reshape failed");
    if (mDeployment.specConfig.has_value())
    {
        // SpecDecode base engines emit target features that feed the draft engine.
        check::check(mPipelineIO->baseHiddenStates.reshape({activeBatchSize, inputIdsLength, baseOutputHiddenDim}),
            "Tensor reshape failed");
    }

    // Populate host-side context lengths with effective (unpadded) prefill lengths and pack tokens.
    int32_t* hostCtxLenData = mPipelineIO->hostContextLengths.dataPointer<int32_t>();
    check::check(mHostPackedTokenIds.reshape({activeBatchSize, inputIdsLength}), "Tensor reshape failed");
    int32_t* hostPackedTokenIdsData = mHostPackedTokenIds.dataPointer<int32_t>();

    // Clear the entire pinned buffer first so trailing pad slots from prior batches don't leak into the
    // multimodal-indices walk, which scans all inputIdsLength positions per row, not just up to context_length.
    std::fill(hostPackedTokenIdsData, hostPackedTokenIdsData + activeBatchSize * inputIdsLength, 0);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        hostCtxLenData[i] = context.effectivePrefillLengths[i];
        std::copy(context.tokenIds[i].begin(), context.tokenIds[i].end(), hostPackedTokenIdsData + i * inputIdsLength);
    }

    CUDA_CHECK(cudaMemcpyAsync(mIdsInput.rawPointer(), hostPackedTokenIdsData,
        activeBatchSize * inputIdsLength * sizeof(int32_t), cudaMemcpyHostToDevice, context.stream));

    // Embedding lookup (text / vision / audio-multimodal) into mPipelineIO->inputsEmbeds;
    // deepstack slots are populated from features or zero-filled depending on the request.
    mEmbeddingPre->embed(mIdsInput, context.visualEmbeddings, context.audioEmbeddings, *mPipelineIO, context.stream);
    mEmbeddingPre->prepareDeepstack(mIdsInput, context.deepstackFeatures, *mPipelineIO, context.stream);

    // Dispatch per-step sequence prep (context lengths H2D, selectTokenIndices).
    mStepPreparer->prepare(
        InferencePhase::kPrefill, activeBatchSize, *mSharedResources->cacheManagers[0], *mPipelineIO, context.stream);
    // Bind real deepstack features for this prefill (no-op when feature absent).
    if (mDeepstack)
    {
        mDeepstack->useRealFeatures(mBaseTensorMap);
    }

    // Execute base prefill through the EngineExecutor. Empty-cache is
    // runtime-dynamic; prefillDims uses it to set InferenceDims::startIndexLen
    // (0 for the "initial prefill" sentinel, else batch).
    bool const baseKVAllEmpty = mSharedResources->cacheManagers[0]->getKVCacheAllEmpty();
    auto const prefillDims = mDeployment.base.prefillDims(activeBatchSize, inputIdsLength, baseKVAllEmpty);
    check::check(mBaseExecutor->prepare(kPrefillProfile, prefillDims, mBaseTensorMap, context.stream),
        "Failed to prepare base model for prefill step.");
    check::check(mBaseExecutor->execute(context.stream), "Failed to execute base model for prefill step.");
    mSharedResources->cacheManagers[0]->commitSequenceLength(mPipelineIO->contextLengths, context.stream);

    // Sampling from the prefill stage logits follows the same policy as vanilla decoding.
    // Speculative decoders reach this code only for greedy-compatible requests; non-greedy
    // requests are routed to the vanilla decode path by handleRequest.
    check::check(mSamplingIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    if (shouldUseNonGreedySampling(context.temperature, context.topK, context.topP))
    {
        SamplingParams params(activeBatchSize, mDeployment.base.outputVocabSize, context.temperature,
            static_cast<int32_t>(context.topK), context.topP);
        topKtopPSamplingFromLogits(
            mPipelineIO->outputLogits, mSamplingIndices, params, mSamplingWorkspace, context.stream);
    }
    else
    {
        constexpr int32_t kSAMPLING_TOP_K = 1;
        selectAllTopK(mPipelineIO->outputLogits, std::nullopt, mSamplingIndices, kSAMPLING_TOP_K, mSamplingWorkspace,
            context.stream);
    }

    // Apply vocabulary mapping if base model uses reduced vocabulary.
    if (mDeployment.base.reducedVocabSize > 0)
    {
        mapReducedVocabToFullVocab(mSamplingIndices, mBaseVocabMappingTable, context.stream);
    }

    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();
    CUDA_CHECK(cudaMemcpyAsync(hostSelectedTokenIdsData, mSamplingIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, context.stream));
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        if (!context.finishedStates[i])
        {
            context.tokenIds[i].push_back(hostSelectedTokenIdsData[i]);
            context.currentGenerateLengths[i] += 1;

            // Fire the per-token callback for the prefill-sampled token. runVanillaDecoding
            // dispatches the callback for every decode token, so emitting here keeps the sequence
            // complete for streaming consumers (e.g. the Qwen3-Omni Thinker-Talker pipeline).
            if (context.onTokenGenerated.has_value())
            {
                bool const isFinished = context.finishedStates[i] != 0;
                TokenCallbackInfo info{hostSelectedTokenIdsData[i], i, context.generationRound, isFinished};
                context.onTokenGenerated.value()(info);
            }
        }
    }
    return true;
}

bool LLMInferenceRuntime::captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream)
{
    auto captureOnce = [&](std::string const& loraName) -> bool {
        if (mSharedResources->loraManager)
        {
            if (loraName.empty())
            {
                mSharedResources->loraManager->resetWeights();
            }
            else
            {
                mSharedResources->loraManager->switchWeights(loraName);
            }
            mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
        }
        if (!mBaseExecutor->prepare(kDecodeProfile, dims, mBaseTensorMap, stream))
        {
            return false;
        }
        return mBaseExecutor->captureGraph(stream);
    };

    bool ok = captureOnce(mEmptyLoraWeightsName);
    if (mDeployment.base.maxSupportedLoraRank > 0 && mSharedResources->loraManager)
    {
        for (auto const& loraWeightsName : mSharedResources->loraManager->getAdapterNames())
        {
            ok &= captureOnce(loraWeightsName);
        }
    }
    return ok;
}

bool LLMInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    return mDecoderRegistry ? mDecoderRegistry->captureCudaGraphs(stream) : true;
}

void LLMInferenceRuntime::restoreRecurrentStates(
    int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream)
{
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    auto const& mambaConfig = mambaMgr.getConfig();

    size_t const recurrentElemSize = rt::utils::getTypeSize(mambaConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mambaConfig.convStateType);
    size_t const recurrentBatchBytes = static_cast<size_t>(mambaConfig.recurrentStateNumHeads
                                           * mambaConfig.recurrentStateHeadDim * mambaConfig.recurrentStateSize)
        * recurrentElemSize;
    size_t const convBatchBytes = static_cast<size_t>(mambaConfig.convDim * mambaConfig.convKernel) * convElemSize;

    for (int32_t layer = 0; layer < mambaMgr.numLayers(); ++layer)
    {
        rt::Tensor& recurrentLayer = mambaMgr.getRecurrentState(layer);
        rt::Tensor& convLayer = mambaMgr.getConvState(layer);

        auto* recurrentDst = static_cast<std::byte*>(recurrentLayer.rawPointer()) + batchIdx * recurrentBatchBytes;
        auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + batchIdx * convBatchBytes;

        if (layer < static_cast<int32_t>(cachedStates.recurrentStateContents.size()))
        {
            CUDA_CHECK(cudaMemcpyAsync(recurrentDst, cachedStates.recurrentStateContents[layer].rawPointer(),
                recurrentBatchBytes, cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(recurrentDst, 0, recurrentBatchBytes, stream));
        }

        if (layer < static_cast<int32_t>(cachedStates.convStateContents.size()))
        {
            CUDA_CHECK(cudaMemcpyAsync(convDst, cachedStates.convStateContents[layer].rawPointer(), convBatchBytes,
                cudaMemcpyDeviceToDevice, stream));
        }
        else
        {
            CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
        }
    }
}

void LLMInferenceRuntime::zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream)
{
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    auto& mambaMgr = cacheMgrBase.getMambaCacheManager();
    auto const& mambaConfig = mambaMgr.getConfig();

    size_t const recurrentElemSize = rt::utils::getTypeSize(mambaConfig.recurrentStateType);
    size_t const convElemSize = rt::utils::getTypeSize(mambaConfig.convStateType);
    size_t const recurrentBatchBytes = static_cast<size_t>(mambaConfig.recurrentStateNumHeads
                                           * mambaConfig.recurrentStateHeadDim * mambaConfig.recurrentStateSize)
        * recurrentElemSize;
    size_t const convBatchBytes = static_cast<size_t>(mambaConfig.convDim * mambaConfig.convKernel) * convElemSize;

    for (int32_t layer = 0; layer < mambaMgr.numLayers(); ++layer)
    {
        rt::Tensor& recurrentLayer = mambaMgr.getRecurrentState(layer);
        rt::Tensor& convLayer = mambaMgr.getConvState(layer);

        auto* recurrentDst = static_cast<std::byte*>(recurrentLayer.rawPointer()) + batchIdx * recurrentBatchBytes;
        auto* convDst = static_cast<std::byte*>(convLayer.rawPointer()) + batchIdx * convBatchBytes;
        CUDA_CHECK(cudaMemsetAsync(recurrentDst, 0, recurrentBatchBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(convDst, 0, convBatchBytes, stream));
    }
}

bool LLMInferenceRuntime::setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy)
{
    NVTX_SCOPED_RANGE(nvtx_setup, "SETUP_PREFILL_EXECUTION", nvtx_colors::PALE_GREEN);

    // LoRA switching goes through the LoRAManager on SharedResources.
    if (mDeployment.base.maxSupportedLoraRank > 0 && mSharedResources->loraManager)
    {
        try
        {
            if (context.loraWeightsName.empty())
            {
                mSharedResources->loraManager->resetWeights();
            }
            else
            {
                mSharedResources->loraManager->switchWeights(context.loraWeightsName);
            }
            mSharedResources->loraManager->refreshTensorMap(mBaseTensorMap);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to switch LoRA weights to %s: %s", context.loraWeightsName.c_str(), e.what());
            return false;
        }
    }

    int32_t const activeBatchSize = context.activeBatchSize;
    std::vector<std::vector<int32_t>> const& batchedInputIds = context.rawBatchedInputIds;
    bool const needsStrategyKVCache = strategy.isSpeculative();
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];

    // Record the length of the reused KVCache for each sequence.
    check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    std::fill(reuseKVCacheLengthsData, reuseKVCacheLengthsData + activeBatchSize, 0);

    context.tokenIds.clear();
    context.tokenIds.resize(activeBatchSize);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto const& prompt = context.systemPrompts[i];
        auto const promptKey = keySystemPromptWithLoraWeights(prompt, context.loraWeightsName);
        if (mSystemPromptKVCacheBase.count(promptKey) > 0)
        {
            auto& precachedKVCacheBase = mSystemPromptKVCacheBase[promptKey];
            auto const& kvCacheLayersBase = precachedKVCacheBase.kvCacheLayers;
            cacheMgrBase.restoreKVCache(kvCacheLayersBase, i, context.stream);

            if (needsStrategyKVCache)
            {
                check::check(strategy.hasSystemPromptKVCache(promptKey),
                    "System prompt cache inconsistency between base and active decoding strategy");
                strategy.restoreSystemPromptKVCache(promptKey, i, context.stream);
            }

            // Restore recurrent/conv states for hybrid models (vanilla path only — spec decode handles this in
            // decoder).
            if (mDeployment.base.numLinearAttnLayers > 0)
            {
                restoreRecurrentStates(i, precachedKVCacheBase, context.stream);
            }

            // Per-layer saved KV tensor shape is [2, numKVHeads, sequenceLength, headDim]; shape[2] == seqLen.
            check::check(!kvCacheLayersBase.empty(), "System prompt KV cache must have at least one layer.");
            auto reuseLength = math::cast<size_t>(kvCacheLayersBase[0].getShape()[2]);
            check::check(reuseLength > 0 && reuseLength < batchedInputIds[i].size(),
                "The reuse length shall be larger than 0 and not exceed the input length.");
            // Reuse N-1 tokens from the cached prefix so the Nth token is treated as real input in prefill;
            // this keeps the draft prefill boundary aligned with the true next-token position.
            auto const effectiveReuseLength = reuseLength - 1;
            reuseKVCacheLengthsData[i] = math::cast<int32_t>(effectiveReuseLength);

            context.tokenIds[i].assign(batchedInputIds[i].begin() + effectiveReuseLength, batchedInputIds[i].end());
            context.effectivePrefillLengths[i] = math::cast<int32_t>(batchedInputIds[i].size() - effectiveReuseLength);

            bool const matchIds = std::equal(precachedKVCacheBase.tokenizedPrompt.begin(),
                precachedKVCacheBase.tokenizedPrompt.end(), batchedInputIds[i].begin());
            if (!matchIds)
            {
                LOG_WARNING(
                    "Though system prompt strings are matched, token_ids are not perfectly aligned."
                    "This may generate incorrect result, please check your system prompt design.");
            }
        }
        else
        {
            context.tokenIds[i] = batchedInputIds[i];
            context.effectivePrefillLengths[i] = static_cast<int32_t>(batchedInputIds[i].size());
            reuseKVCacheLengthsData[i] = 0;

            if (mDeployment.base.numLinearAttnLayers > 0)
            {
                zeroRecurrentStates(i, context.stream);
            }
        }
    }

    int32_t const maxInputLength
        = *std::max_element(context.effectivePrefillLengths.begin(), context.effectivePrefillLengths.end());
    if (maxInputLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR("The max input length (%d) exceeds the max supported input length (%d) of the LLM Engine.",
            maxInputLength, mDeployment.base.maxSupportedInputLength);
        return false;
    }

    mSharedResources->cacheManagers[0]->resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
    if (needsStrategyKVCache)
    {
        strategy.resetForNewSequences(mHostReuseKVCacheLengths, context.stream);
    }
    return true;
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(DecodingInferenceContext& context, int32_t genAndSaveBatchIdx)
{
    std::string const& loraWeightsName = context.loraWeightsName;
    std::string const prompt = context.systemPrompts[genAndSaveBatchIdx];
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);

    if (prompt.empty())
    {
        LOG_DEBUG("The systemPrompt is empty. Skip saving system prompt KVCache.");
        return true;
    }

    DecodingStrategy& cacheStrategy = mDecoderRegistry->cachePrimingStrategy();
    bool const hasDraft = cacheStrategy.isSpeculative();
    auto baseCacheIt = mSystemPromptKVCacheBase.find(promptKey);
    if (baseCacheIt != mSystemPromptKVCacheBase.end() && (!hasDraft || cacheStrategy.hasSystemPromptKVCache(promptKey)))
    {
        LOG_DEBUG("The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }
    if (baseCacheIt != mSystemPromptKVCacheBase.end())
    {
        mSystemPromptKVCacheBase.erase(baseCacheIt);
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());

    if (promptIdsLength > mDeployment.base.maxSupportedInputLength)
    {
        LOG_ERROR("System prompt length (%d) exceeds max supported input length (base=%d)", promptIdsLength,
            mDeployment.base.maxSupportedInputLength);
        return false;
    }

    if (hasDraft && promptIdsLength > mDeployment.draft->maxSupportedInputLength)
    {
        LOG_ERROR("System prompt length (%d) exceeds max supported input length (draft=%d)", promptIdsLength,
            mDeployment.draft->maxSupportedInputLength);
        return false;
    }

    // Temporary single-batch context to reuse the existing prefill functions.
    DecodingInferenceContext tempContext;
    tempContext.initialize(1, 1, context.visualEmbeddings, context.deepstackFeatures, loraWeightsName, context.stream);
    tempContext.systemPrompts[0] = prompt;
    tempContext.rawBatchedInputIds.push_back(tokenizedPrompt);
    tempContext.tokenIds[0] = tokenizedPrompt;

    if (!setUpForPrefillExecution(tempContext, cacheStrategy))
    {
        LOG_ERROR("Prefill execution setup failed for system prompt KVCache generation.");
        return false;
    }

    bool prefillStatus = runBaseModelPrefill(tempContext);
    if (!prefillStatus)
    {
        LOG_ERROR("Failed to execute base model prefill for system prompt KVCache generation.");
        return false;
    }

    // Tokens produced during system KV-cache reuse prefill do not count as generated tokens.
    tempContext.currentGenerateLengths[0] -= 1;

    if (hasDraft)
    {
        bool draftPrefillStatus = cacheStrategy.runSystemPromptPrefill(tempContext);
        if (!draftPrefillStatus)
        {
            LOG_ERROR("Failed to execute draft model prefill for system prompt KVCache generation.");
            return false;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Capture base KV cache content from the new-stack shared KV cache.
    auto& cacheMgrBase = *mSharedResources->cacheManagers[0];
    constexpr int32_t CACHE_BATCH_IDX{0};

    SystemPromptKVCache savedKVCacheBase;
    savedKVCacheBase.systemPrompt = prompt;
    savedKVCacheBase.tokenizedPrompt = tokenizedPrompt;
    savedKVCacheBase.kvCacheLayers = cacheMgrBase.captureKVCache(CACHE_BATCH_IDX, promptIdsLength, context.stream);

    // Save recurrent / conv states for hybrid layers.
    if (mDeployment.base.numLinearAttnLayers > 0)
    {
        savedKVCacheBase.recurrentStateContents = cacheMgrBase.captureRecurrentStates(CACHE_BATCH_IDX, context.stream);
        savedKVCacheBase.convStateContents = cacheMgrBase.captureConvStates(CACHE_BATCH_IDX, context.stream);
    }

    mSystemPromptKVCacheBase.insert({promptKey, std::move(savedKVCacheBase)});

    cacheStrategy.saveSystemPromptKVCache(promptKey, prompt, tokenizedPrompt, promptIdsLength, context.stream);

    CUDA_CHECK(cudaStreamSynchronize(context.stream));
    LOG_DEBUG("System prompt KVCache saved for batch %d: {%s}", genAndSaveBatchIdx, prompt.c_str());

    return true;
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (prompt.empty())
    {
        LOG_DEBUG("The systemPrompt is empty. Skip saving system prompt KVCache.");
        return true;
    }
    auto const promptKey = keySystemPromptWithLoraWeights(prompt, loraWeightsName);
    DecodingStrategy& cacheStrategy = mDecoderRegistry->cachePrimingStrategy();
    if (mSystemPromptKVCacheBase.find(promptKey) != mSystemPromptKVCacheBase.end()
        && (!cacheStrategy.isSpeculative() || cacheStrategy.hasSystemPromptKVCache(promptKey)))
    {
        LOG_DEBUG("The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }
    DecodingInferenceContext tempContext;
    tempContext.initialize(1, 1, std::nullopt, rt::OptionalInputTensors{}, loraWeightsName, stream);
    tempContext.systemPrompts[0] = prompt;
    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    if (tokenizedPrompt.empty())
    {
        LOG_ERROR("Failed to encode system prompt for KVCache generation.");
        return false;
    }
    tempContext.rawBatchedInputIds.push_back(tokenizedPrompt);
    tempContext.tokenIds[0] = tokenizedPrompt;
    return genAndSaveSystemPromptKVCache(tempContext, 0);
}

bool LLMInferenceRuntime::performBatchEvict(DecodingInferenceContext& context, DecodingStrategy& strategy)
{
    // Check if any batch has finished
    bool hasFinishedBatch = false;
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (context.finishedStates[i])
        {
            hasFinishedBatch = true;
            break;
        }
    }

    if (!hasFinishedBatch)
    {
        return true;
    }

    int32_t const oldActiveBatch = context.activeBatchSize;

    // Build batch mapping
    std::vector<int32_t> batchMapping = buildBatchMapping(context.finishedStates);

    // Calculate new active batch size
    int32_t newActiveBatch = 0;
    for (auto newIdx : batchMapping)
    {
        if (newIdx >= 0)
        {
            newActiveBatch = std::max(newActiveBatch, newIdx + 1);
        }
    }

    // Log eviction details
    std::vector<int32_t> evictedIndices;
    for (int32_t i = 0; i < oldActiveBatch; ++i)
    {
        if (batchMapping[i] < 0)
        {
            evictedIndices.push_back(i);
        }
    }
    LOG_DEBUG("Batch eviction: %d active batches to %d remaining (evicted %d batch(es): indices [%s])", oldActiveBatch,
        newActiveBatch, static_cast<int32_t>(evictedIndices.size()),
        [&evictedIndices]() {
            std::string result;
            for (size_t i = 0; i < evictedIndices.size(); ++i)
            {
                if (i > 0)
                {
                    result += ", ";
                }
                result += std::to_string(evictedIndices[i]);
            }
            return result;
        }()
            .c_str());

    // Upload batch mapping to GPU
    check::check(mDeviceBatchMapping.reshape({oldActiveBatch}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBatchMapping.rawPointer(), batchMapping.data(), oldActiveBatch * sizeof(int32_t),
        cudaMemcpyHostToDevice, context.stream));

    // Compact base model caches (KV + Mamba) via the HybridCacheManager single-call API.
    mSharedResources->cacheManagers[0]->compactBatch(
        mDeviceBatchMapping, oldActiveBatch, newActiveBatch, context.stream);
    mSharedResources->cacheManagers[0]->setActiveBatchSize(newActiveBatch);

    // Compact base model's RoPE cache (stored per-batch for MRope on mPipelineIO->mropeCosSin).
    if (mDeployment.base.ropeConfig.type == RopeType::kMRope && newActiveBatch > 0)
    {
        rt::Tensor& baseRopeCache = mPipelineIO->mropeCosSin;
        if (baseRopeCache.getShape().getNumDims() == 3 && baseRopeCache.getShape()[0] == oldActiveBatch)
        {
            kernel::compactTensorBatch(
                baseRopeCache, mDeviceBatchMapping, baseRopeCache, oldActiveBatch, newActiveBatch, context.stream);
            auto const seqLen = static_cast<int32_t>(baseRopeCache.getShape()[1]);
            auto const rotaryDim = static_cast<int32_t>(baseRopeCache.getShape()[2]);
            check::check(baseRopeCache.reshape({newActiveBatch, seqLen, rotaryDim}), "Tensor reshape failed");
        }
    }

    strategy.onBatchEvict(batchMapping, oldActiveBatch, newActiveBatch, mDeviceBatchMapping, context.stream);

    // Compact CPU context
    CUDA_CHECK(cudaStreamSynchronize(context.stream));

    // Save evicted batches' results before compacting (using original batch index)
    for (size_t i = 0; i < batchMapping.size(); ++i)
    {
        if (batchMapping[i] < 0 && context.finishedStates[i])
        {
            // This batch is evicted and finished, save its results with original index
            int32_t originalIdx = context.batchIndexMapping[i];

            // Create and populate BatchResult with all related data
            BatchResult result;
            result.tokenIds = std::move(context.tokenIds[i]);
            result.generateLength = context.currentGenerateLengths[i];
            result.actualIterations = context.generationRound;
            result.rawBatchedInputIds = std::move(context.rawBatchedInputIds[i]);
            result.effectivePrefillLength = context.effectivePrefillLengths[i];
            result.terminalReason = context.slotStreams[i].terminalReason;

            context.completedBatches[originalIdx] = std::move(result);
        }
    }

    rt::compactVector(batchMapping, context.finishedStates);
    rt::compactVector(batchMapping, context.currentGenerateLengths);
    rt::compactVector(batchMapping, context.tokenIds);
    rt::compactVector(batchMapping, context.systemPrompts);
    rt::compactVector(batchMapping, context.rawBatchedInputIds);
    rt::compactVector(batchMapping, context.effectivePrefillLengths);
    rt::compactVector(batchMapping, context.batchIndexMapping);
    rt::compactVector(batchMapping, context.slotStreams);
    rt::compactVector(batchMapping, context.stopStringsPerSlot);

    // Update active batch size
    context.activeBatchSize = newActiveBatch;

    return true;
}

} // namespace rt
} // namespace trt_edgellm
