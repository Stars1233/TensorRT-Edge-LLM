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

#include "qwen3OmniTTSRuntime.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#ifdef CUTE_DSL_GEMM_ENABLED
#include "kernels/talkerMLPKernels/cuteDslGemmRunner.h"
#endif
#include "llmInferenceRuntime.h"
#include "profiling/metrics.h"
#include "profiling/nvtx_wrapper.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace trt_edgellm
{
namespace rt
{

using Json = nlohmann::json;
using namespace talker_constants;

namespace
{
// Helper: Extract MLP weights from safetensors (eliminates code duplication)
bool extractMLPWeightsFromTensors(std::vector<rt::Tensor>& tensors, rt::Tensor& fc1Weight, rt::Tensor& fc1Bias,
    rt::Tensor& fc2Weight, rt::Tensor& fc2Bias, std::string const& projectionName)
{
    constexpr int32_t kExpectedTensorCount = 4;
    check::check(
        tensors.size() == kExpectedTensorCount, projectionName + ".safetensors should contain exactly 4 tensors");

    bool foundFC1Weight = false, foundFC1Bias = false, foundFC2Weight = false, foundFC2Bias = false;

    for (auto& tensor : tensors)
    {
        std::string const& name = tensor.getName();
        if (name.find("fc1.weight") != std::string::npos)
        {
            fc1Weight = std::move(tensor);
            foundFC1Weight = true;
        }
        else if (name.find("fc1.bias") != std::string::npos)
        {
            fc1Bias = std::move(tensor);
            foundFC1Bias = true;
        }
        else if (name.find("fc2.weight") != std::string::npos)
        {
            fc2Weight = std::move(tensor);
            foundFC2Weight = true;
        }
        else if (name.find("fc2.bias") != std::string::npos)
        {
            fc2Bias = std::move(tensor);
            foundFC2Bias = true;
        }
    }

    if (!foundFC1Weight || !foundFC1Bias || !foundFC2Weight || !foundFC2Bias)
    {
        LOG_ERROR("Failed to find all required tensors in %s.safetensors", projectionName.c_str());
        return false;
    }

    return true;
}

// Shared chunk accumulator + emitter for streaming RVQ codes.
// Used by both runTalkerGenerationLoop (per-batch in TTS) and handleStreamingGeneration (bs=1 Omni)
// so chunk semantics (threshold-triggered non-final emit + single final flush) live in one place.
struct ChunkEmitter
{
    int32_t chunkFrames{0};
    std::function<void(std::vector<std::vector<int32_t>> const& chunk, bool isFinal)> onChunk;
    std::vector<std::vector<int32_t>> buffer;

    bool active() const
    {
        return chunkFrames > 0 && static_cast<bool>(onChunk);
    }

    // Append a frame; emit non-final chunk when buffer hits the threshold.
    void append(std::vector<int32_t> const& frame)
    {
        if (!active())
            return;
        buffer.push_back(frame);
        if (static_cast<int32_t>(buffer.size()) >= chunkFrames)
        {
            onChunk(buffer, /*isFinal=*/false);
            buffer.clear();
        }
    }

    // Final flush — always invoked once per active emitter at end-of-stream, even if buffer empty
    // (callers rely on this as the end-of-stream signal).
    void flushFinal()
    {
        if (!active())
            return;
        onChunk(buffer, /*isFinal=*/true);
        buffer.clear();
    }
};
} // anonymous namespace

Qwen3OmniTTSRuntime::Qwen3OmniTTSRuntime(std::string const& talkerEngineDir, std::string const& codePredictorEngineDir,
    std::string const& tokenizerDir, cudaStream_t stream)
    : mStream(stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::init", nvtx_colors::YELLOW);
    LOG_INFO("Initializing Qwen3-Omni Talker runner");
    LOG_INFO("  Talker: %s", talkerEngineDir.c_str());
    LOG_INFO("  CodePredictor: %s", codePredictorEngineDir.c_str());

    // Load tokenizer
    std::filesystem::path const tokenizerPath = tokenizerDir.empty()
        ? std::filesystem::path(talkerEngineDir).parent_path()
        : std::filesystem::path(tokenizerDir);
    LOG_INFO("  Tokenizer: %s", tokenizerPath.string().c_str());
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    bool const tokenizerLoaded = mTokenizer->loadFromHF(tokenizerPath);
    ELLM_CHECK(tokenizerLoaded, "Failed to load tokenizer from: " + tokenizerPath.string());

    bool const configValid = validateAndFillConfig(talkerEngineDir);
    ELLM_CHECK(configValid, "Failed to validate and fill config");

    bool const runnersInitialized = initializeEngineRunners(talkerEngineDir, codePredictorEngineDir);
    ELLM_CHECK(runnersInitialized, "Failed to initialize engine runners");

    // Setup shared execution context memory for Talker and CodePredictor engines.
    // Both use kUSER_MANAGED allocation and require setContextMemory() before execution.
    {
        int64_t const talkerCtxSize = mTalkerExec->getRequiredContextMemorySize();
        int64_t const cpCtxSize = mCodePredictorExec ? mCodePredictorExec->getRequiredContextMemorySize() : 0;
        int64_t const sharedCtxSize = std::max(talkerCtxSize, cpCtxSize);
        LOG_INFO("Setup shared execution context memory: %zu bytes (talker: %zu, code_predictor: %zu)",
            static_cast<size_t>(sharedCtxSize), static_cast<size_t>(talkerCtxSize), static_cast<size_t>(cpCtxSize));
        mSharedExecContextMemory = rt::Tensor({sharedCtxSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
            "Qwen3OmniTTSRuntime::mSharedExecContextMemory");
        bool const talkerCtxSet = mTalkerExec->setContextMemory(mSharedExecContextMemory);
        ELLM_CHECK(talkerCtxSet, "Failed to set context memory for Talker LLM engine");
        ELLM_CHECK(!mCodePredictorExec || mCodePredictorExec->setContextMemory(mSharedExecContextMemory),
            "Failed to set context memory for CodePredictor engine");
    }

    // Determine max batch size from engine configs (use the minimum of Talker and CodePredictor)
    mMaxBatchSize = std::min(mTalkerLLMConfig.maxSupportedBatchSize, mCodePredictorConfig.maxSupportedBatchSize);
    check::check(mMaxBatchSize >= 1, "maxBatchSize must be >= 1");
    LOG_INFO("Max batch size: %d (Talker=%d, CodePredictor=%d)", mMaxBatchSize, mTalkerLLMConfig.maxSupportedBatchSize,
        mCodePredictorConfig.maxSupportedBatchSize);

    bool const codePredictorWeightsLoaded = loadCodePredictorWeights(codePredictorEngineDir);
    ELLM_CHECK(codePredictorWeightsLoaded, "Failed to load CodePredictor weights");

#ifdef CUTE_DSL_GEMM_ENABLED
    bool const cuteDslLoaded = CuteDslGemmRunner::loadKernelModule();
    ELLM_CHECK(cuteDslLoaded, "Failed to load CuTe DSL GEMM kernel module");
#endif

    bool const bufferAllocated = allocateBuffer();
    ELLM_CHECK(bufferAllocated, "Failed to allocate buffers");

    bool const talkerWeightsLoaded = loadTalkerWeights(talkerEngineDir, stream);
    ELLM_CHECK(talkerWeightsLoaded, "Failed to load Talker weights");

    // Load text embedding table (thinker vocab).
    // Used for standalone TTS and for projecting TTS special tokens.
    // TTS: text_embedding.safetensors in talkerEngineDir (copied by builder).
    // Omni: use thinker's embedding.safetensors from tokenizerPath instead.
    {
        std::filesystem::path const textEmbedPath = mIsOmni
            ? tokenizerPath / "embedding.safetensors"
            : std::filesystem::path(talkerEngineDir) / "text_embedding.safetensors";
        LOG_INFO("Loading text embedding from: %s", textEmbedPath.string().c_str());
        std::vector<rt::Tensor> textEmbedTensors;
        bool const textEmbedLoaded = safetensors::loadSafetensors(textEmbedPath, textEmbedTensors, stream);
        ELLM_CHECK(textEmbedLoaded, "Failed to load text embedding from: " + textEmbedPath.string());
        check::check(!textEmbedTensors.empty(), "text embedding file is empty");
        check::check(textEmbedTensors[0].getShape().getNumDims() == 2,
            "text embedding tensor should be 2D [vocabSize, hiddenSize]");
        mTextEmbeddingTable = std::move(textEmbedTensors[0]);
        LOG_INFO("Text embedding table loaded: [%lld, %lld]", mTextEmbeddingTable.getShape()[0],
            mTextEmbeddingTable.getShape()[1]);
    }

    // Note: mTalkerEmbeddingTable is loaded by loadTalkerWeights() above.

    // Load CodePredictor embedding tables from codec_embeddings.safetensors
    // mNumRvqLayers is already set from config.json num_code_groups in validateAndFillConfig()
    {
        std::filesystem::path const embedPath
            = std::filesystem::path(codePredictorEngineDir) / "codec_embeddings.safetensors";
        std::vector<rt::Tensor> allEmbedTensors;
        bool const codecEmbedLoaded = safetensors::loadSafetensors(embedPath, allEmbedTensors, stream);
        ELLM_CHECK(codecEmbedLoaded, "Failed to load codec_embeddings.safetensors from: " + embedPath.string());
        check::check(static_cast<int32_t>(allEmbedTensors.size()) == mNumRvqLayers,
            "codec_embeddings.safetensors has " + std::to_string(allEmbedTensors.size()) + " tensors, expected "
                + std::to_string(mNumRvqLayers) + " (num_code_groups - 1)");
        mCodePredictorEmbeddingTables.resize(mNumRvqLayers);
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            std::string const key = "embedding_" + std::to_string(i);
            auto it = std::find_if(allEmbedTensors.begin(), allEmbedTensors.end(),
                [&key](rt::Tensor const& t) { return t.getName() == key; });
            check::check(it != allEmbedTensors.end(), "Missing key '" + key + "' in codec_embeddings.safetensors");
            check::check(it->getShape().getNumDims() == 2, key + " should be 2D [codebookSize, hiddenSize]");
            mCodePredictorEmbeddingTables[i] = std::move(*it);
        }
    }
    LOG_INFO("Loaded %d CodePredictor embedding tables (from config num_code_groups=%d)", mNumRvqLayers,
        mTalkerConfig.numCodeGroups);

    initializeTTSEmbeddings(stream);

    CUDA_CHECK(cudaEventCreateWithFlags(&mTtfaStart, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&mTtfaEnd, cudaEventDefault));

    LOG_INFO("Qwen3-Omni TTS runtime initialized successfully");
}

Qwen3OmniTTSRuntime::~Qwen3OmniTTSRuntime()
{
#ifdef CUTE_DSL_GEMM_ENABLED
    CuteDslGemmRunner::unloadKernelModule();
#endif
    if (mTtfaStart)
    {
        cudaEventDestroy(mTtfaStart);
    }
    if (mTtfaEnd)
    {
        cudaEventDestroy(mTtfaEnd);
    }
}

bool Qwen3OmniTTSRuntime::initializeEngineRunners(
    std::string const& talkerEngineDir, std::string const& codePredictorEngineDir)
{
    // Load Talker LLM engine via EngineExecutor (migrated from LLMEngineRunner)
    std::filesystem::path talkerEnginePath = std::filesystem::path(talkerEngineDir) / "llm.engine";
    std::filesystem::path talkerConfigPath = std::filesystem::path(talkerEngineDir) / "config.json";

    LOG_INFO("Loading Talker LLM engine from: %s", talkerEnginePath.string().c_str());

    try
    {
        mTalkerLLMConfig = rt::parseEngineConfig(talkerConfigPath);
        mTalkerExec = rt::EngineExecutor::createForLLM(talkerEnginePath, mTalkerLLMConfig);
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mTalkerSharedRes = rt::SharedResources::createForLLM(mTalkerLLMConfig, emptyLoraMap, mStream);
        mTalkerPipelineIO = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(mTalkerLLMConfig, mStream));
        mTalkerStepPreparer = std::make_unique<rt::StepPreparer>(mTalkerLLMConfig);
        rt::buildTensorMap(
            mTalkerTensorMap, *mTalkerPipelineIO, *mTalkerSharedRes, mTalkerLLMConfig, /*kvCacheIndex=*/0);

        LOG_INFO("Talker LLM engine loaded: vocabSize=%d, hiddenSize=%d", mTalkerLLMConfig.vocabSize,
            mTalkerLLMConfig.hiddenSize);
        auto talkerKVType = mTalkerLLMConfig.kvCacheDtype;
        LOG_INFO("Talker KV cache dtype: %s",
            talkerKVType == nvinfer1::DataType::kHALF ? "FP16"
                                                      : (talkerKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN"));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load Talker LLM engine: %s", e.what());
        return false;
    }

    // Load CodePredictor engine via EngineExecutor (migrated from LLMEngineRunner)
    std::filesystem::path codePredictorEnginePath = std::filesystem::path(codePredictorEngineDir) / "llm.engine";
    std::filesystem::path codePredictorConfigPath = std::filesystem::path(codePredictorEngineDir) / "config.json";

    LOG_INFO("Loading CodePredictor engine from: %s", codePredictorEnginePath.string().c_str());

    try
    {
        mCodePredictorConfig = rt::parseEngineConfig(codePredictorConfigPath);
        mCodePredictorExec = rt::EngineExecutor::createForLLM(codePredictorEnginePath, mCodePredictorConfig);
        std::unordered_map<std::string, std::string> emptyLoraMap;
        mCodePredictorSharedRes = rt::SharedResources::createForLLM(mCodePredictorConfig, emptyLoraMap, mStream);
        mCodePredictorPipelineIO
            = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(mCodePredictorConfig, mStream));
        mCodePredictorStepPreparer = std::make_unique<rt::StepPreparer>(mCodePredictorConfig);
        rt::buildTensorMap(mCodePredictorTensorMap, *mCodePredictorPipelineIO, *mCodePredictorSharedRes,
            mCodePredictorConfig, /*kvCacheIndex=*/0);

        // CodePredictor ONNX outputs FP32 logits directly (lm_head applied in-engine via the
        // dynamic lm_head_weight input). We rebind that input per RVQ head before each call.

        // Read CodePredictor dimensions from loaded config (vocab_size==hidden_size since
        // engine output is last_hidden; real codebook_size is inferred from lm_head shape later)
        mTalkerConfig.codePredictorHiddenSize = mCodePredictorConfig.hiddenSize;

        LOG_INFO("CodePredictor engine loaded: vocabSize=%d, hiddenSize=%d, numLayers=%d",
            mCodePredictorConfig.vocabSize, mCodePredictorConfig.hiddenSize, mCodePredictorConfig.numDecoderLayers);
        auto cpKVType = mCodePredictorConfig.kvCacheDtype;
        LOG_INFO("CodePredictor KV cache dtype: %s",
            cpKVType == nvinfer1::DataType::kHALF ? "FP16"
                                                  : (cpKVType == nvinfer1::DataType::kFP8 ? "FP8" : "UNKNOWN"));
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to load CodePredictor engine: %s", e.what());
        return false;
    }

    return true;
}

bool Qwen3OmniTTSRuntime::validateAndFillConfig(std::string const& talkerEngineDir)
{
    // Load config.json from talker directory
    std::filesystem::path configPath = std::filesystem::path(talkerEngineDir) / "config.json";
    LOG_INFO("Loading Talker config from: %s", configPath.string().c_str());

    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.string().c_str());
        return false;
    }

    Json configJson;
    try
    {
        configJson = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config: %s", e.what());
        return false;
    }

    // Model dimensions
    mTalkerConfig.thinkerHiddenSize = configJson.value("thinker_hidden_size", 2048);
    mTalkerConfig.talkerHiddenSize = configJson["hidden_size"].get<int32_t>();
    mTalkerConfig.talkerVocabSize = configJson["vocab_size"].get<int32_t>();

    // CodePredictor RVQ configuration: Omni uses 16 code groups, TTS uses 32
    mTalkerConfig.numCodeGroups = configJson["num_code_groups"].get<int32_t>();
    check::check(mTalkerConfig.numCodeGroups >= 2,
        "num_code_groups must be >= 2, got: " + std::to_string(mTalkerConfig.numCodeGroups));
    mNumRvqLayers = mTalkerConfig.numCodeGroups - 1;
    mNumCodesPerFrame = mTalkerConfig.numCodeGroups;
    LOG_INFO("Config num_code_groups=%d -> RVQ layers=%d, codes per frame=%d", mTalkerConfig.numCodeGroups,
        mNumRvqLayers, mNumCodesPerFrame);

    // Runtime parameters
    mTalkerConfig.maxSeqLen = configJson.value("max_position_embeddings", 8192);

    // Validate dimensions with reasonable limits
    constexpr int32_t kMaxReasonableVocabSize = 200000;
    constexpr int32_t kMaxReasonableHiddenSize = 16384;
    constexpr int32_t kMaxReasonableSeqLen = 131072;

    check::check(mTalkerConfig.talkerVocabSize > 0 && mTalkerConfig.talkerVocabSize < kMaxReasonableVocabSize,
        "Invalid talker vocab size: " + std::to_string(mTalkerConfig.talkerVocabSize));
    check::check(mTalkerConfig.thinkerHiddenSize > 0 && mTalkerConfig.thinkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid thinker hidden size: " + std::to_string(mTalkerConfig.thinkerHiddenSize));
    check::check(mTalkerConfig.talkerHiddenSize > 0 && mTalkerConfig.talkerHiddenSize < kMaxReasonableHiddenSize,
        "Invalid talker hidden size: " + std::to_string(mTalkerConfig.talkerHiddenSize));
    check::check(mTalkerConfig.maxSeqLen > 0 && mTalkerConfig.maxSeqLen < kMaxReasonableSeqLen,
        "Invalid max sequence length: " + std::to_string(mTalkerConfig.maxSeqLen));

    // TTS special tokens (from thinker vocab)
    mTalkerConfig.ttsPadTokenId = configJson.value("tts_pad_token_id", 151671);
    mTalkerConfig.ttsBosTokenId = configJson.value("tts_bos_token_id", 151672);
    mTalkerConfig.ttsEosTokenId = configJson.value("tts_eos_token_id", 151673);

    // Codec special tokens (from talker vocab)
    mTalkerConfig.codecNothinkId = configJson["codec_nothink_id"].get<int32_t>();
    mTalkerConfig.codecThinkBosId = configJson["codec_think_bos_id"].get<int32_t>();
    mTalkerConfig.codecThinkEosId = configJson["codec_think_eos_id"].get<int32_t>();
    mTalkerConfig.codecPadId = configJson["codec_pad_id"].get<int32_t>();
    mTalkerConfig.codecBosId = configJson["codec_bos_id"].get<int32_t>();
    // Support both codec_eos_token_id (original) and codec_eos_id (legacy) for backward compatibility
    if (configJson.contains("codec_eos_token_id"))
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_token_id"].get<int32_t>();
    }
    else
    {
        mTalkerConfig.codecEosId = configJson["codec_eos_id"].get<int32_t>();
    }

    // Speaker ID configuration
    mTalkerConfig.defaultSpeakerId = configJson.value("default_speaker_id", 2301);

    // Thinker→Talker streaming uses this to route engine hidden_states from
    // the Thinker portal into the Talker prefill. Standalone Qwen3-TTS Talker
    // configs (no Thinker) legitimately omit the field; the streaming code
    // path validates the value before use. Sentinel -1 means "unconfigured".
    mTalkerConfig.acceptHiddenLayer = configJson.value("accept_hidden_layer", -1);

    // Load speaker ID mapping if available
    if (configJson.contains("speaker_id") && configJson["speaker_id"].is_object())
    {
        for (auto const& [speaker_name, speaker_id] : configJson["speaker_id"].items())
        {
            mSpeakerIdMap[speaker_name] = speaker_id.get<int32_t>();
        }
        LOG_INFO("Loaded %zu speaker IDs from config", mSpeakerIdMap.size());

        // Log available speakers
        if (!mSpeakerIdMap.empty())
        {
            std::string speakerList;
            for (auto const& [name, id] : mSpeakerIdMap)
            {
                if (!speakerList.empty())
                {
                    speakerList += ", ";
                }
                speakerList += name + ":" + std::to_string(id);
            }
            LOG_DEBUG("Available speakers: %s", speakerList.c_str());
        }
    }

    LOG_INFO("Talker config: vocabSize=%d, hiddenSize=%d, thinkerHiddenSize=%d, defaultSpeaker=%d",
        mTalkerConfig.talkerVocabSize, mTalkerConfig.talkerHiddenSize, mTalkerConfig.thinkerHiddenSize,
        mTalkerConfig.defaultSpeakerId);
    LOG_DEBUG("TTS tokens: pad=%d, bos=%d, eos=%d", mTalkerConfig.ttsPadTokenId, mTalkerConfig.ttsBosTokenId,
        mTalkerConfig.ttsEosTokenId);
    LOG_DEBUG("Codec tokens: skipThink=%d, thinkBos=%d, thinkEos=%d, pad=%d, bos=%d, eos=%d",
        mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, mTalkerConfig.codecEosId);

    return true;
}

bool Qwen3OmniTTSRuntime::loadCodePredictorWeights(std::string const& codePredictorEngineDir)
{
    LOG_INFO("Loading %d CodePredictor lm_head weights", mNumRvqLayers);
    mCodePredictorLmHeadWeights.resize(mNumRvqLayers);
    {
        std::filesystem::path const lmHeadPath = std::filesystem::path(codePredictorEngineDir) / "lm_heads.safetensors";
        std::vector<rt::Tensor> allLmHeadTensors;
        if (!safetensors::loadSafetensors(lmHeadPath, allLmHeadTensors, mStream))
        {
            LOG_ERROR("Failed to load lm_heads.safetensors from: %s", lmHeadPath.string().c_str());
            return false;
        }
        if (static_cast<int32_t>(allLmHeadTensors.size()) != mNumRvqLayers)
        {
            LOG_ERROR("lm_heads.safetensors has %zu entries, expected %d (matching codec_embeddings count)",
                allLmHeadTensors.size(), mNumRvqLayers);
            return false;
        }
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            std::string const weightKey = "lm_head_" + std::to_string(i) + ".weight";
            auto it = std::find_if(allLmHeadTensors.begin(), allLmHeadTensors.end(),
                [&weightKey](rt::Tensor const& t) { return t.getName() == weightKey; });
            if (it == allLmHeadTensors.end())
            {
                LOG_ERROR("Missing key '%s' in lm_heads.safetensors", weightKey.c_str());
                return false;
            }
            if (it->getShape().getNumDims() != 2)
            {
                LOG_ERROR("%s should be 2D [vocabSize, hiddenSize]", weightKey.c_str());
                return false;
            }
            LOG_DEBUG("Loaded %s [%d, %d]", weightKey.c_str(), it->getShape()[0], it->getShape()[1]);
            mCodePredictorLmHeadWeights[i] = std::move(*it);
        }
    }

    mTalkerConfig.codebookSize = static_cast<int32_t>(mCodePredictorLmHeadWeights[0].getShape()[0]);
    LOG_INFO("Loaded %d CodePredictor lm_head weights, codebookSize=%d", mNumRvqLayers, mTalkerConfig.codebookSize);

    // Load small_to_mtp_projection: projects Talker hidden (2048) → CodePredictor input (1024)
    {
        std::filesystem::path const projPath
            = std::filesystem::path(codePredictorEngineDir) / "small_to_mtp_projection.safetensors";
        if (std::filesystem::exists(projPath))
        {
            std::vector<rt::Tensor> projTensors;
            if (!safetensors::loadSafetensors(projPath, projTensors, mStream))
            {
                LOG_ERROR("Failed to load small_to_mtp_projection from: %s", projPath.string().c_str());
                return false;
            }
            bool foundWeight = false, foundBias = false;
            for (auto& t : projTensors)
            {
                if (t.getName() == "weight")
                {
                    mSmallToMtpWeight = std::move(t);
                    foundWeight = true;
                }
                else if (t.getName() == "bias")
                {
                    mSmallToMtpBias = std::move(t);
                    foundBias = true;
                }
            }
            if (!foundWeight || !foundBias)
            {
                LOG_ERROR("Missing 'weight' or 'bias' in small_to_mtp_projection.safetensors");
                return false;
            }
            mUseSmallToMtpProjection = true;
            LOG_INFO("Loaded small_to_mtp_projection: weight=%ldx%ld, bias=%ld", mSmallToMtpWeight.getShape()[0],
                mSmallToMtpWeight.getShape()[1], mSmallToMtpBias.getShape()[0]);
        }
        else if (mTalkerConfig.talkerHiddenSize == mTalkerConfig.codePredictorHiddenSize)
        {
            mUseSmallToMtpProjection = false;
            LOG_INFO("No small_to_mtp_projection needed (talkerHiddenSize == codePredictorHiddenSize = %d)",
                mTalkerConfig.talkerHiddenSize);
        }
        else
        {
            LOG_ERROR(
                "small_to_mtp_projection.safetensors required when talkerHiddenSize (%d) != "
                "codePredictorHiddenSize (%d)",
                mTalkerConfig.talkerHiddenSize, mTalkerConfig.codePredictorHiddenSize);
            return false;
        }
    }

    // CodePredictor embedding tables are already loaded in the constructor (with auto-detection
    // of mNumRvqLayers from codec_embeddings.safetensors).

    return true;
}

bool Qwen3OmniTTSRuntime::allocateBuffer()
{
    LOG_INFO("Allocating Qwen3-Omni TTS Runtime inference workspace buffers (maxBatchSize=%d)...", mMaxBatchSize);

    int64_t const maxSeqLen = mTalkerConfig.maxSeqLen;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const maxBS = mMaxBatchSize;

    try
    {
        // Per-batch prefill workspace (reused sequentially per batch in buildTalkerPrefillFromSegments)
        mThinkerEmbedBuffer = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mThinkerEmbedBuffer");
        mGpuTokenIdsBuffer
            = rt::Tensor({1, maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mGpuTokenIdsBuffer");
        mMLPWorkspace = rt::Tensor(
            {maxSeqLen, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mMLPWorkspace");
        mProjectedBuffer = rt::Tensor(
            {maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mProjectedBuffer");
        // Talker input embeds: [maxBS, maxSeqLen, H] for batched prefill
        mTalkerInputEmbeds = rt::Tensor({maxBS * maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerInputEmbeds");

        // Talker LLM workspace — batched
        mTalkerLogits = rt::Tensor(
            {maxBS, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "mTalkerLogits");
        mTalkerSelectedIndices
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerSelectedIndices");
        mHostSelectedTokenIds
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostSelectedTokenIds");

        // CodePredictor workspace — sized to maxBS so any batch in [1, maxBS] just reshapes per-call.
        // Same pattern as Talker: framework primitives (EngineExecutor / StepPreparer) are
        // batch-size-agnostic; the per-call reshape({activeBatchSize, ...}) makes them work.
        mCodePredictorLogits = rt::Tensor({maxBS, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "mCodePredictorLogits");

        mCodePredictorLogitsPerHead.resize(mNumRvqLayers);
        for (int32_t i = 0; i < mNumRvqLayers; ++i)
        {
            mCodePredictorLogitsPerHead[i] = rt::Tensor({maxBS, mTalkerConfig.codebookSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kFLOAT, "mCodePredictorLogitsPerHead_" + std::to_string(i));
        }

        mCodePredictorPrefillInput = rt::Tensor({maxBS, 2, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorPrefillInput");
        mCodePredictorCodecIds
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorCodecIds");
        mCodePredictorCodecEmbed = rt::Tensor({maxBS, 1, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mCodePredictorCodecEmbed");
        mRawCodecEmbed = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mRawCodecEmbed");
        mSmallToMtpProjectedHidden = rt::Tensor({maxBS, mTalkerConfig.codePredictorHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mSmallToMtpProjectedHidden");
        mCodePredictorSelectedIndices
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mCodePredictorSelectedIndices");
        mHostSelectedCodeIds
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostSelectedCodeIds");
        // Pinned (cudaMallocHost) buffer that accumulates the deferred CP gen samples
        // across all (mNumRvqLayers - 1) steps for up to maxBS active batches, so we
        // can do a single cudaStreamSynchronize per frame instead of one per step.
        // Layout: [step_idx, batch_idx] — step k writes to row k.
        mHostGenCodeBuf = rt::Tensor({static_cast<int64_t>(mNumRvqLayers - 1), maxBS}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kINT32, "mHostGenCodeBuf");
        mHostCodePredictorContextLength
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostCodePredictorContextLength");

        // Residual + decode buffers — batched for Talker engine execution
        mResidualEmbedBuffer = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mResidualEmbedBuffer");
        mTalkerDecodingIds
            = rt::Tensor({maxBS, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mTalkerDecodingIds");
        mTalkerDecodingEmbed = rt::Tensor({maxBS, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerDecodingEmbed");

        // KVCache reset — batched
        mHostReuseKVCacheLengths
            = rt::Tensor({maxBS}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "mHostReuseKVCacheLengths");

        // Sampling workspace — allocated at maxBS for batched topKtopP
        int32_t const defaultTopK{0};
        float const defaultTopP{0.9F};
        trt_edgellm::SamplingParams samplingParams(
            static_cast<int32_t>(maxBS), mTalkerConfig.talkerVocabSize, 1.0f, defaultTopK, defaultTopP);
        int64_t const samplingWorkspaceSize = trt_edgellm::getTopKtopPSamplingWorkspaceSize(
            static_cast<int32_t>(maxBS), mTalkerConfig.talkerVocabSize, samplingParams);
        mSamplingWorkspace = rt::Tensor(
            {samplingWorkspaceSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "mSamplingWorkspace");
        // Per-batch seen codec tokens for repetition penalty
        mSeenCodecTokensBuf = rt::Tensor({maxBS, mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "mSeenCodecTokensBuf");

        // Generation loop workspace — Talker batched, CodePredictor batch=1
        mTalkerHiddenStatesBuffer = rt::Tensor({maxBS, maxSeqLen, talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mTalkerHiddenStatesBuffer");
        mCodePredictorHiddenStatesBuffer = rt::Tensor({maxBS, mNumCodesPerFrame, mTalkerConfig.codePredictorHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodePredictorHiddenStatesBuffer");
        mTalkerLastHidden = rt::Tensor(
            {maxBS, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mTalkerLastHidden");
        mCodecHiddensBuffer = rt::Tensor({maxBS, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mCodecHiddensBuffer");

        // Trailing text hidden buffer: [maxBS * (maxSeqLen+1), H] — per-batch regions for Omni multi-batch
        mStreamingTrailingHidden = rt::Tensor({maxBS * (maxSeqLen + 1), talkerHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, "mStreamingTrailingHidden");

        // Gather/scatter index buffer for multimodal token projection
        mGatherIndicesBuffer
            = rt::Tensor({maxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mGatherIndicesBuffer");

        // Streaming: single-token workspace for appendTrailingToken
        mStreamingTokenId = rt::Tensor({1, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "mStreamingTokenId");
        mStreamingTokenEmbed = rt::Tensor(
            {1, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingTokenEmbed");
        mStreamingProjOut
            = rt::Tensor({1, talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingProjOut");
        mStreamingMlpWork
            = rt::Tensor({1, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "mStreamingMlpWork");

        LOG_INFO("Talker buffers allocated (maxBS=%d, maxSeqLen=%ld, talkerH=%ld, cpH=%d)", mMaxBatchSize, maxSeqLen,
            talkerHiddenSize, mTalkerConfig.codePredictorHiddenSize);
        return true;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate Talker buffers: %s", e.what());
        return false;
    }
}

bool Qwen3OmniTTSRuntime::loadTalkerWeights(std::string const& weightsDir, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::loadTalkerWeights", nvtx_colors::YELLOW);

    // Load text_projection weights
    std::filesystem::path const textProjPath = std::filesystem::path(weightsDir) / "text_projection.safetensors";
    std::vector<rt::Tensor> textTensors;
    if (!safetensors::loadSafetensors(textProjPath, textTensors, stream))
    {
        LOG_ERROR("Failed to load text_projection from: %s", textProjPath.string().c_str());
        return false;
    }
    if (!extractMLPWeightsFromTensors(
            textTensors, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, "text_projection"))
    {
        return false;
    }

    // Load hidden_projection weights (same architecture as text_projection, different weights)
    std::filesystem::path const hiddenProjPath = std::filesystem::path(weightsDir) / "hidden_projection.safetensors";
    if (std::filesystem::exists(hiddenProjPath))
    {
        mIsOmni = true;
        std::vector<rt::Tensor> hiddenTensors;
        if (!safetensors::loadSafetensors(hiddenProjPath, hiddenTensors, stream))
        {
            LOG_ERROR("Failed to load hidden_projection from: %s", hiddenProjPath.string().c_str());
            return false;
        }
        if (!extractMLPWeightsFromTensors(
                hiddenTensors, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight, mHiddenFC2Bias, "hidden_projection"))
        {
            return false;
        }
        LOG_INFO("hidden_projection weights loaded from: %s", hiddenProjPath.string().c_str());
    }
    else
    {
        mIsOmni = false;
        LOG_INFO("hidden_projection.safetensors not found at %s (multimodal token projection unavailable)",
            hiddenProjPath.string().c_str());
    }

    // Note: mTextEmbeddingTable is loaded separately in the constructor with mIsOmni-aware path selection
    // (TTS: text_embedding.safetensors from talkerEngineDir, Omni: embedding.safetensors from tokenizerDir)

    // Load Talker embedding table
    std::filesystem::path const talkerEmbedPath = std::filesystem::path(weightsDir) / "embedding.safetensors";
    std::vector<rt::Tensor> talkerEmbedTensors;
    if (!safetensors::loadSafetensors(talkerEmbedPath, talkerEmbedTensors, stream))
    {
        LOG_ERROR("Failed to load Talker embedding from: %s", talkerEmbedPath.string().c_str());
        return false;
    }
    check::check(talkerEmbedTensors.size() == 1, "Talker embedding.safetensors should contain exactly one tensor");
    check::check(talkerEmbedTensors[0].getShape().getNumDims() == 2,
        "Talker embedding tensor should be 2D [vocabSize, hiddenSize]");
    mTalkerEmbeddingTable = std::move(talkerEmbedTensors[0]);
    LOG_INFO("Talker embedding table loaded: [%lld, %lld]", mTalkerEmbeddingTable.getShape()[0],
        mTalkerEmbeddingTable.getShape()[1]);

    LOG_INFO("Talker weights loaded successfully");
    return true;
}

void Qwen3OmniTTSRuntime::initializeTTSEmbeddings(cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::initializeTTSEmbeddings", nvtx_colors::YELLOW);

    auto const shape = mTextEmbeddingTable.getShape();
    ELLM_CHECK(
        shape.getNumDims() == 2, "Text embedding table must be 2D, got " + std::to_string(shape.getNumDims()) + "D");

    int64_t const vocabSize = shape[0];
    int64_t const thinkerHiddenSize = shape[1];

    ELLM_CHECK(mTalkerConfig.ttsPadTokenId < vocabSize && mTalkerConfig.ttsBosTokenId < vocabSize
            && mTalkerConfig.ttsEosTokenId < vocabSize,
        "TTS token IDs out of vocab range: pad=" + std::to_string(mTalkerConfig.ttsPadTokenId)
            + ", bos=" + std::to_string(mTalkerConfig.ttsBosTokenId)
            + ", eos=" + std::to_string(mTalkerConfig.ttsEosTokenId) + ", vocabSize=" + std::to_string(vocabSize));

    constexpr int32_t kNumTtsTokens = 3;
    std::vector<int32_t> const hostTtsIds
        = {mTalkerConfig.ttsPadTokenId, mTalkerConfig.ttsBosTokenId, mTalkerConfig.ttsEosTokenId};

    rt::Tensor ttsIds({1, kNumTtsTokens}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor ttsRaw({1, kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor ttsProjected(
        {kNumTtsTokens, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor workspace({kNumTtsTokens, thinkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    CUDA_CHECK(cudaMemcpyAsync(
        ttsIds.rawPointer(), hostTtsIds.data(), kNumTtsTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    kernel::embeddingLookup(ttsIds, mTextEmbeddingTable, std::nullopt, ttsRaw, stream);
    // Reshape from [1, 3, hidden] to [3, hidden] for MLP (expects 2D input)
    check::check(ttsRaw.reshape({kNumTtsTokens, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(
        ttsRaw, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, ttsProjected, workspace, stream);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    mTtsPadEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mTtsBosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    mTtsEosEmbed = rt::Tensor({hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    __half* const projectedPtr = static_cast<__half*>(ttsProjected.rawPointer());
    size_t const embedSize = hiddenSize * sizeof(__half);

    CUDA_CHECK(cudaMemcpyAsync(
        mTtsPadEmbed.rawPointer(), projectedPtr + 0 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsBosEmbed.rawPointer(), projectedPtr + 1 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        mTtsEosEmbed.rawPointer(), projectedPtr + 2 * hiddenSize, embedSize, cudaMemcpyDeviceToDevice, stream));

    LOG_INFO("TTS embeddings initialized");
}

bool Qwen3OmniTTSRuntime::projectToTalkerInput(
    rt::Tensor const& thinkerEmbed, int32_t speakerId, rt::Tensor& output, int64_t& outputSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = thinkerEmbed.getShape()[0];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const thinkerHiddenSize = mTalkerConfig.thinkerHiddenSize;

    // N = text tokens after stripping 3-token role prefix and 5-token suffix
    int64_t const N = seqLen - kAssistantPrefixLen - kAssistantTrailingSuffix;
    // Non-streaming prefill: 8 fixed prefix rows + N text rows + 2 suffix rows
    outputSeqLen = kNonStreamingPrefixRows + N + 2; // = seqLen + 2
    LOG_INFO("projectToTalkerInput: seqLen=%ld, N=%ld (stripped prefix=%d suffix=%d), outputSeqLen=%ld, speakerId=%d",
        seqLen, N, kAssistantPrefixLen, kAssistantTrailingSuffix, outputSeqLen, speakerId);

    // Project all tokens via text_projection MLP
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(thinkerEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias, mProjectedBuffer,
        mMLPWorkspace, stream);

    // Fused kernel: build complete non-streaming prefill buffer
    check::check(output.reshape({outputSeqLen, hiddenSize}), "Tensor reshape failed");
    kernel::invokeAssistantPreamble(mProjectedBuffer, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
        mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId, speakerId,
        mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, static_cast<int32_t>(N), output, stream);

    return true;
}

bool Qwen3OmniTTSRuntime::executeTalkerPrefillStep(rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits,
    rt::Tensor& outputHiddenStates, cudaStream_t stream, std::vector<int64_t> const& perBatchContextLengths)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeTalkerPrefillStep", nvtx_colors::PURPLE);

    auto inputShape = inputEmbeds.getShape();
    if (inputShape.getNumDims() != 3)
    {
        LOG_ERROR("executeTalkerPrefillStep: Input must be 3D [batchSize, seqLen, hiddenSize], got %dD",
            inputShape.getNumDims());
        return false;
    }

    int64_t const batchSize = inputEmbeds.getTRTDims().d[0];
    int64_t const seqLen = inputEmbeds.getTRTDims().d[1];

    if (batchSize > mMaxBatchSize)
    {
        LOG_ERROR("executeTalkerPrefillStep: batchSize %ld exceeds maxBatchSize %d", batchSize, mMaxBatchSize);
        return false;
    }

    // Reset Talker KV cache — shape must match batchSize so commitSequenceLength sees consistent activeBatchSize
    auto& talkerCacheMgr = *mTalkerSharedRes->cacheManagers[0];
    check::check(mHostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    std::fill_n(mHostReuseKVCacheLengths.dataPointer<int32_t>(), batchSize, 0);
    talkerCacheMgr.resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // Stage per-batch context lengths into PipelineIO's host buffer.
    // StepPreparer consumes hostContextLengths to fill GPU contextLengths + selectTokenIndices.
    check::check(mTalkerPipelineIO->hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* hostContextLength = mTalkerPipelineIO->hostContextLengths.dataPointer<int32_t>();
    if (!perBatchContextLengths.empty())
    {
        for (int64_t i = 0; i < batchSize; ++i)
        {
            hostContextLength[i] = static_cast<int32_t>(perBatchContextLengths[i]);
        }
    }
    else
    {
        for (int64_t i = 0; i < batchSize; ++i)
        {
            hostContextLength[i] = static_cast<int32_t>(seqLen);
        }
    }
    // Reshape logits to match the prefill batch size (CUDA graph capture may leave it at maxBS).
    check::check(outputLogits.reshape({batchSize, outputLogits.getShape()[outputLogits.getShape().getNumDims() - 1]}),
        "Tensor reshape failed");

    // Bind step-specific tensors into the Talker TensorMap. Engine I/O bindings whose
    // address/shape change per step (inputs_embeds, logits, hidden_states) get rewired here;
    // KV cache and rope cache were registered statically by buildTensorMap.
    mTalkerTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputEmbeds));
    mTalkerTensorMap.set(binding_names::kLogits, outputLogits);
    mTalkerTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);

    // Prepare per-step metadata (selectTokenIndices, contextLengths, kvcache_start_index sentinel).
    mTalkerStepPreparer->prepare(
        rt::InferencePhase::kPrefill, static_cast<int32_t>(batchSize), talkerCacheMgr, *mTalkerPipelineIO, stream);

    bool const kvAllEmpty = talkerCacheMgr.getKVCacheAllEmpty();
    auto const prefillDims = mTalkerLLMConfig.prefillDims(batchSize, seqLen, kvAllEmpty);
    if (!mTalkerExec->prepare(/*prefillProfile=*/0, prefillDims, mTalkerTensorMap, stream))
    {
        LOG_ERROR("Talker prefill prepare failed");
        return false;
    }
    if (!mTalkerExec->execute(stream))
    {
        LOG_ERROR("Talker prefill execute failed");
        return false;
    }
    talkerCacheMgr.commitSequenceLength(mTalkerPipelineIO->contextLengths, stream);
    return true;
}

bool Qwen3OmniTTSRuntime::executeTalkerDecodingStep(
    rt::Tensor const& inputEmbeds, rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeTalkerDecodingStep", nvtx_colors::PURPLE);

    auto inputShape = inputEmbeds.getShape();
    check::check(inputShape.getNumDims() == 3, "executeTalkerDecodingStep: inputEmbeds must be 3D [batch, 1, hidden]");
    int64_t const batchSize = inputShape[0];
    check::check(batchSize <= mMaxBatchSize, "executeTalkerDecodingStep: batchSize exceeds maxBatchSize");

    auto& talkerCacheMgr = *mTalkerSharedRes->cacheManagers[0];

    mTalkerTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputEmbeds));
    mTalkerTensorMap.set(binding_names::kLogits, outputLogits);
    mTalkerTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);

    mTalkerStepPreparer->prepare(
        rt::InferencePhase::kDecode, static_cast<int32_t>(batchSize), talkerCacheMgr, *mTalkerPipelineIO, stream);

    auto const decodeDims = mTalkerLLMConfig.decodeDims(batchSize);
    if (!mTalkerExec->prepare(/*decodeProfile=*/1, decodeDims, mTalkerTensorMap, stream))
    {
        LOG_ERROR("Talker decode prepare failed");
        return false;
    }
    if (!mTalkerExec->execute(stream))
    {
        LOG_ERROR("Talker decode execute failed");
        return false;
    }
    // Vanilla decode advances the KV cache by exactly 1 token per call (matches kVANILLA_DECODE_INCREMENT).
    talkerCacheMgr.commitSequenceLength(/*increment=*/1, stream);
    return true;
}

bool Qwen3OmniTTSRuntime::executeCodePredictorPrefillStep(rt::Tensor const& inputsEmbeds, int32_t lmHeadIdx,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorPrefillStep", nvtx_colors::ORANGE);

    // Batch dim is implicit in input shape — same pattern as executeTalkerPrefillStep / spec decode.
    auto inputShape = inputsEmbeds.getShape();
    check::check(inputShape.getNumDims() == 3,
        "executeCodePredictorPrefillStep: inputsEmbeds must be 3D [batch, seqLen, cpHidden]");
    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    check::check(batchSize <= mMaxBatchSize, "executeCodePredictorPrefillStep: batchSize exceeds maxBatchSize");

    auto& cpCacheMgr = *mCodePredictorSharedRes->cacheManagers[0];

    // Reset CP KV cache for this batch (CP resets every frame — no cross-frame state).
    check::check(mHostReuseKVCacheLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* reuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    std::fill_n(reuseData, batchSize, 0);
    cpCacheMgr.resetForNewSequences(mHostReuseKVCacheLengths, stream);

    // Stage per-batch host context lengths (all = kCodePredictorPrefillSeqLen).
    check::check(mCodePredictorPipelineIO->hostContextLengths.reshape({batchSize}), "Tensor reshape failed");
    int32_t* hostCtxLen = mCodePredictorPipelineIO->hostContextLengths.dataPointer<int32_t>();
    std::fill_n(hostCtxLen, batchSize, static_cast<int32_t>(seqLen));

    int32_t const clampedLmHeadIdx = std::min(lmHeadIdx, mNumRvqLayers - 1);

    mCodePredictorTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputsEmbeds));
    mCodePredictorTensorMap.set(binding_names::kLogits, outputLogits);
    mCodePredictorTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);
    mCodePredictorTensorMap.set(binding_names::kLmHeadWeight, mCodePredictorLmHeadWeights[clampedLmHeadIdx]);

    mCodePredictorStepPreparer->prepare(
        rt::InferencePhase::kPrefill, static_cast<int32_t>(batchSize), cpCacheMgr, *mCodePredictorPipelineIO, stream);

    bool const kvAllEmpty = cpCacheMgr.getKVCacheAllEmpty();
    auto const prefillDims = mCodePredictorConfig.prefillDims(batchSize, seqLen, kvAllEmpty);
    if (!mCodePredictorExec->prepare(/*prefillProfile=*/0, prefillDims, mCodePredictorTensorMap, stream))
    {
        LOG_ERROR("CodePredictor prefill prepare failed");
        return false;
    }
    if (!mCodePredictorExec->execute(stream))
    {
        LOG_ERROR("CodePredictor prefill execute failed");
        return false;
    }
    cpCacheMgr.commitSequenceLength(mCodePredictorPipelineIO->contextLengths, stream);
    return true;
}

bool Qwen3OmniTTSRuntime::executeCodePredictorDecodingStep(rt::Tensor const& inputsEmbeds, int32_t lmHeadIdx,
    rt::Tensor& outputLogits, rt::Tensor& outputHiddenStates, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::executeCodePredictorDecodingStep", nvtx_colors::ORANGE);

    auto inputShape = inputsEmbeds.getShape();
    check::check(
        inputShape.getNumDims() == 3, "executeCodePredictorDecodingStep: inputsEmbeds must be 3D [batch, 1, cpHidden]");
    int64_t const batchSize = inputShape[0];
    check::check(batchSize <= mMaxBatchSize, "executeCodePredictorDecodingStep: batchSize exceeds maxBatchSize");

    auto& cpCacheMgr = *mCodePredictorSharedRes->cacheManagers[0];
    int32_t const clampedLmHeadIdx = std::min(lmHeadIdx, mNumRvqLayers - 1);

    mCodePredictorTensorMap.set(binding_names::kInputsEmbeds, const_cast<rt::Tensor&>(inputsEmbeds));
    mCodePredictorTensorMap.set(binding_names::kLogits, outputLogits);
    mCodePredictorTensorMap.set(binding_names::kOutputHiddenStates, outputHiddenStates);
    mCodePredictorTensorMap.set(binding_names::kLmHeadWeight, mCodePredictorLmHeadWeights[clampedLmHeadIdx]);

    mCodePredictorStepPreparer->prepare(
        rt::InferencePhase::kDecode, static_cast<int32_t>(batchSize), cpCacheMgr, *mCodePredictorPipelineIO, stream);

    auto const decodeDims = mCodePredictorConfig.decodeDims(batchSize);
    if (!mCodePredictorExec->prepare(/*decodeProfile=*/1, decodeDims, mCodePredictorTensorMap, stream))
    {
        LOG_ERROR("CodePredictor decode prepare failed");
        return false;
    }
    if (!mCodePredictorExec->execute(stream))
    {
        LOG_ERROR("CodePredictor decode execute failed");
        return false;
    }
    cpCacheMgr.commitSequenceLength(/*increment=*/1, stream);
    return true;
}

// ========== CUDA Graph Capture ==========

bool Qwen3OmniTTSRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    std::string const emptyLoraWeightsName = "";

    // Talker: capture for all supported batch sizes (1..maxBatchSize).
    // EngineExecutor::captureGraph() hashes the current binding state, so each bs
    // produces its own graph slot automatically.
    bool captureStatus{true};
    auto& talkerCacheMgr = *mTalkerSharedRes->cacheManagers[0];
    for (int32_t bs = 1; bs <= mMaxBatchSize; ++bs)
    {
        // Simulate a mid-sequence decode state for capture (matches `simulateCacheLength=128`),
        // so the captured graph reflects realistic plugin shapes / KV-length math.
        constexpr int32_t kSimulateCacheLength{128};
        std::vector<int32_t> reuseLens(bs, kSimulateCacheLength);
        rt::Tensor simulatedReuse(reuseLens.data(), rt::Coords{bs}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        talkerCacheMgr.resetForNewSequences(simulatedReuse, stream);

        check::check(mResidualEmbedBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(
            mTalkerHiddenStatesBuffer.reshape({bs, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
        check::check(mTalkerLogits.reshape({bs, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");

        mTalkerTensorMap.set(binding_names::kInputsEmbeds, mResidualEmbedBuffer);
        mTalkerTensorMap.set(binding_names::kLogits, mTalkerLogits);
        mTalkerTensorMap.set(binding_names::kOutputHiddenStates, mTalkerHiddenStatesBuffer);

        mTalkerStepPreparer->prepare(rt::InferencePhase::kDecode, bs, talkerCacheMgr, *mTalkerPipelineIO, stream);

        auto const decodeDims = mTalkerLLMConfig.decodeDims(bs);
        captureStatus &= mTalkerExec->prepare(/*decodeProfile=*/1, decodeDims, mTalkerTensorMap, stream);
        captureStatus &= mTalkerExec->captureGraph(stream);
    }
    // Restore Talker KV cache to "empty" state for the first real prefill.
    // The simulated-cache-length init above leaves both the per-batch lengths AND
    // the engine-written stale KV contents in mid-sequence state; resetForNewSequences
    // with zero lengths is what every real prefill expects.
    {
        std::vector<int32_t> zeroLens(mMaxBatchSize, 0);
        rt::Tensor zeroReuse(
            zeroLens.data(), rt::Coords{mMaxBatchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        talkerCacheMgr.resetForNewSequences(zeroReuse, stream);
    }

    // CodePredictor: capture one graph per RVQ head (mNumRvqLayers total).
    // Each graph uses a distinct (lm_head_weight, logits) binding pair, so the
    // EngineExecutor binding hash naturally distinguishes them.
    auto& cpCacheMgr = *mCodePredictorSharedRes->cacheManagers[0];
    check::check(
        mCodePredictorCodecEmbed.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}), "Tensor reshape failed");
    check::check(mCodePredictorHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.codePredictorHiddenSize}),
        "Tensor reshape failed");

    for (int32_t i = 0; i < mNumRvqLayers; ++i)
    {
        // Simulate a mid-sequence CP decode state for capture (matches kSimulateCacheLength=128).
        constexpr int32_t kSimulateCacheLength{128};
        int32_t simLen = kSimulateCacheLength;
        rt::Tensor simReuse(&simLen, rt::Coords{1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        cpCacheMgr.resetForNewSequences(simReuse, stream);

        mCodePredictorTensorMap.set(binding_names::kInputsEmbeds, mCodePredictorCodecEmbed);
        mCodePredictorTensorMap.set(binding_names::kLogits, mCodePredictorLogitsPerHead[i]);
        mCodePredictorTensorMap.set(binding_names::kOutputHiddenStates, mCodePredictorHiddenStatesBuffer);
        mCodePredictorTensorMap.set(binding_names::kLmHeadWeight, mCodePredictorLmHeadWeights[i]);

        mCodePredictorStepPreparer->prepare(
            rt::InferencePhase::kDecode, /*batchSize=*/1, cpCacheMgr, *mCodePredictorPipelineIO, stream);

        auto const decodeDims = mCodePredictorConfig.decodeDims(1);
        captureStatus &= mCodePredictorExec->prepare(/*decodeProfile=*/1, decodeDims, mCodePredictorTensorMap, stream);
        captureStatus &= mCodePredictorExec->captureGraph(stream);
    }
    // Restore CP KV cache to empty so the first real prefill starts from a clean state.
    {
        int32_t zero = 0;
        rt::Tensor zeroReuse(&zero, rt::Coords{1}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32);
        cpCacheMgr.resetForNewSequences(zeroReuse, stream);
    }

    // (mCodePredictorGraphsCaptured flag removed — EngineExecutor's graph cache auto-dispatches per binding hash.)

    if (captureStatus)
    {
        LOG_INFO("Successfully captured decoding CUDA graphs for Talker and all CodePredictor lm_heads.");
    }
    else
    {
        LOG_WARNING("Failed to capture some decoding CUDA graphs. Will use fallback engine execution.");
    }

    return captureStatus;
}

// ========== Audio Generation API ==========

bool Qwen3OmniTTSRuntime::prepareTalkerInput(std::vector<int32_t> const& textTokenIds,
    TalkerGenerationRequest const& request, int64_t& outSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    if (seqLen == 0)
    {
        LOG_ERROR("prepareTalkerInput: empty token ID list");
        return false;
    }
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    check::check(mGpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data(), seqLen * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    check::check(mThinkerEmbedBuffer.reshape({1, seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::embeddingLookup(mGpuTokenIdsBuffer, mTextEmbeddingTable, std::nullopt, mThinkerEmbedBuffer, stream);
    check::check(mThinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");

    // Determine speaker ID
    int32_t speakerId = mTalkerConfig.defaultSpeakerId;
    if (request.speakerId >= 0)
    {
        speakerId = request.speakerId;
    }
    else if (!request.speakerName.empty())
    {
        speakerId = getSpeakerIdByName(request.speakerName);
    }

    // MLP projection: thinker embed → talker input embeds (non-streaming, outputSeqLen = seqLen + 2)
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!projectToTalkerInput(mThinkerEmbedBuffer, speakerId, mTalkerInputEmbeds, outSeqLen, stream))
    {
        LOG_ERROR("MLP projection failed");
        return false;
    }

    // Reshape buffers to 3D [1, seqLen, H] for Talker LLM input
    check::check(mTalkerInputEmbeds.reshape({1, outSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(
        mTalkerHiddenStatesBuffer.reshape({1, outSeqLen, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    return true;
}

bool Qwen3OmniTTSRuntime::handleAudioGeneration(
    std::vector<TalkerGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGeneration", nvtx_colors::PURPLE);
    int32_t const activeBatchSize = static_cast<int32_t>(requests.size());
    LOG_INFO("Starting batched audio generation for %d request(s)", activeBatchSize);

    check::check(activeBatchSize > 0 && activeBatchSize <= mMaxBatchSize,
        "Batch size " + std::to_string(activeBatchSize) + " exceeds max " + std::to_string(mMaxBatchSize));

    response.batchRvqCodes.clear();
    response.numFramesPerSample.clear();
    response.success = false;

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    // CP sampling params come from HF's hardcoded ``code_predictor.generate``
    // defaults; see ``kCPSamplingTemperature`` / ``kCPSamplingTopK`` /
    // ``kCPSamplingTopP`` in qwen3OmniTTSRuntime.h for the source-code links.
    SamplingParams predictorSamplingParams(
        1, mTalkerConfig.codebookSize, kCPSamplingTemperature, kCPSamplingTopK, kCPSamplingTopP);
    SamplingParams singleSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);

    // Build per-batch Talker prefill embeddings into mTalkerInputEmbeds, then run a single
    // batched prefill at bs=activeBatchSize (same pattern as handleAudioGenerationFromThinker).
    // Per-batch sequential prefill is fundamentally wrong for bs>1: it overwrites mTalkerHiddenStatesBuffer
    // slot 0 + Talker KV cache slot 0 each iteration, losing earlier batches' state.
    std::vector<PerBatchTalkerState> states(activeBatchSize);
    std::vector<int64_t> perBatchSeqLens(activeBatchSize);
    int64_t const maxInputSeqLen = mTalkerLLMConfig.maxSupportedInputLength;
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t maxOutSeqLen = 0;

    auto buildOneBatchTts = [&](int32_t b) -> bool {
        LLMGenerationRequest::Request llmReq;
        llmReq.messages = requests[b].messages;
        LLMGenerationRequest::FormattedRequest formatted;
        if (!mTokenizer->applyChatTemplate(llmReq, formatted, requests[b].applyChatTemplate,
                requests[b].addGenerationPrompt, requests[b].enableThinking))
        {
            LOG_ERROR("Chat template failed for batch %d", b);
            return false;
        }
        std::vector<int32_t> const textTokenIds = mTokenizer->encode(formatted.formattedCompleteRequest);

        int64_t seqLen = 0;
        if (!prepareTalkerInput(textTokenIds, requests[b], seqLen, stream))
        {
            LOG_ERROR("Input preparation failed for batch %d", b);
            return false;
        }
        perBatchSeqLens[b] = seqLen;
        maxOutSeqLen = std::max(maxOutSeqLen, seqLen);
        return true;
    };

    // Build batches N-1..1 first, stash each into slot (b * maxInputSeqLen).
    for (int32_t b = activeBatchSize - 1; b >= 1; --b)
    {
        if (!buildOneBatchTts(b))
            return false;
        __half* const slot = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + b * maxInputSeqLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(slot, mTalkerInputEmbeds.rawPointer(),
            perBatchSeqLens[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    // Build batch 0 last — stays at slot 0.
    if (!buildOneBatchTts(0))
        return false;

    // Re-pack from slots [b * maxInputSeqLen] → contiguous [BS, maxOutSeqLen, H]; pad with zeros.
    __half* const inputBase = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());
    for (int32_t b = activeBatchSize - 1; b >= 0; --b)
    {
        __half* const src = inputBase + (b == 0 ? 0 : b * maxInputSeqLen * hiddenSize);
        __half* const dst = inputBase + b * maxOutSeqLen * hiddenSize;
        if (src != dst)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                dst, src, perBatchSeqLens[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        if (perBatchSeqLens[b] < maxOutSeqLen)
        {
            CUDA_CHECK(cudaMemsetAsync(dst + perBatchSeqLens[b] * hiddenSize, 0,
                (maxOutSeqLen - perBatchSeqLens[b]) * hiddenSize * sizeof(__half), stream));
        }
    }

    // Single batched Talker prefill at bs=activeBatchSize.
    check::check(mTalkerInputEmbeds.reshape({activeBatchSize, maxOutSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(
        mTalkerHiddenStatesBuffer.reshape({activeBatchSize, maxOutSeqLen, hiddenSize}), "Tensor reshape failed");
    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        if (!executeTalkerPrefillStep(
                mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream, perBatchSeqLens))
        {
            LOG_ERROR("Batched Talker prefill failed");
            return false;
        }
    }

    // Per-batch logit adjust on [BS, vocab] slices.
    check::check(mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        rt::Tensor logitsSlice(static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
            rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
        rt::Tensor seenBufSlice(mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity,
            rt::Coords{mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
        kernel::invokeTalkerLogitAdjust(seenBufSlice, logitsSlice, mTalkerConfig.talkerVocabSize - 1024,
            mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0, repetitionPenalty, stream);
    }

    // Batched sampling at bs=activeBatchSize.
    trt_edgellm::topKtopPSamplingFromLogits(
        mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams, mSamplingWorkspace, stream);
    check::check(mHostSelectedTokenIds.reshape({activeBatchSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].codecToken = hostTokens[b];
        states[b].rvqCodes.reserve(requests[b].maxAudioLength);
        LOG_INFO("Batch %d: first codec token: %d", b, states[b].codecToken);
    }
    (void) singleSamplingParams; // bs=1-only param kept for streaming path; unused here after batching.

    // First codec token sampled — record TTFA-end so streaming CLIs can compute time-to-first-code.
    // Same point as handleAudioGenerationFromThinker / handleStreamingGeneration; unconditional record
    // is cheap (~µs) and lets non-profiling callers still consume the event for streaming latency.
    if (mTtfaEnd)
    {
        CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
    }

    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t effectiveMaxFrames = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const safe = std::max(1, talkerKVCapacity - static_cast<int32_t>(perBatchSeqLens[b]));
        effectiveMaxFrames = std::max(effectiveMaxFrames, std::min(requests[b].maxAudioLength, safe));
    }

    std::vector<rt::Tensor const*> trailingPtrs(activeBatchSize, nullptr);

    // Wire optional per-request streaming callbacks. Empty vector when no request has streaming on
    // (zero overhead for the non-streaming path).
    std::vector<PerBatchStreamingHandler> streamingHandlers;
    for (auto const& r : requests)
    {
        if (r.streamingChunkFrames > 0 && r.onChunkReady)
        {
            streamingHandlers.resize(activeBatchSize);
            break;
        }
    }
    if (!streamingHandlers.empty())
    {
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            streamingHandlers[b].chunkFrames = requests[b].streamingChunkFrames;
            streamingHandlers[b].onChunk = requests[b].onChunkReady;
        }
    }

    if (!runTalkerGenerationLoop(states, activeBatchSize, effectiveMaxFrames, talkerSamplingParams,
            predictorSamplingParams, repetitionPenalty, trailingPtrs, stream, perBatchSeqLens, streamingHandlers))
    {
        return false;
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        response.batchRvqCodes.push_back(std::move(states[b].rvqCodes));
        response.numFramesPerSample.push_back(states[b].talkerFrames);
    }

    int32_t totalFrames = 0;
    for (auto const& s : states)
        totalFrames += s.talkerFrames;
    mMultimodalMetrics.recordRun(0, 0, activeBatchSize, totalFrames);

    response.success = true;
    return true;
}

bool Qwen3OmniTTSRuntime::handleAudioGenerationFromThinker(
    std::vector<OmniGenerationRequest> const& requests, TalkerGenerationResponse& response, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::handleAudioGenerationFromThinker", nvtx_colors::PURPLE);

    int32_t const activeBatchSize = static_cast<int32_t>(requests.size());
    LOG_INFO("Starting batched Omni audio generation for %d request(s)", activeBatchSize);

    check::check(activeBatchSize > 0 && activeBatchSize <= mMaxBatchSize,
        "Batch size " + std::to_string(activeBatchSize) + " exceeds max " + std::to_string(mMaxBatchSize));

    response.batchRvqCodes.clear();
    response.numFramesPerSample.clear();
    response.success = false;

    // Sampling params from requests[0], applied uniformly (matches LLMInferenceRuntime design)
    auto const& req0 = requests[0];
    float const talkerTemperature = (req0.talkerTemperature > 0) ? req0.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (req0.talkerTopK > 0) ? req0.talkerTopK : 50;
    float const talkerTopP = (req0.talkerTopP > 0) ? req0.talkerTopP : 1.0f;
    float const repetitionPenalty = req0.repetitionPenalty;

    SamplingParams talkerSamplingParams(
        activeBatchSize, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(
        1, mTalkerConfig.codebookSize, kCPSamplingTemperature, kCPSamplingTopK, kCPSamplingTopP);

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const trailingStride = mTalkerConfig.maxSeqLen + 1;
    std::vector<PerBatchTalkerState> states(activeBatchSize);
    std::vector<rt::Tensor> perBatchTrailingViews(activeBatchSize);
    std::vector<rt::Tensor const*> trailingPtrs(activeBatchSize, nullptr);

    // Phase 1: Build per-batch Talker prefill embeddings.
    // buildTalkerPrefillFromSegments writes to mTalkerInputEmbeds starting at row 0 (the "scratch"
    // region). After building each batch we relocate the result to a safe per-batch slot at
    // row ((b+1) * maxInputSeqLen) so that the next build's scratch write doesn't clobber it.
    // Batch 0 is special: it's built last (after all others are safely stashed), so its data
    // can stay in place at row 0.
    int64_t const maxInputSeqLen = mTalkerLLMConfig.maxSupportedInputLength;
    std::vector<int64_t> perBatchSeqLen(activeBatchSize);
    int64_t maxOutSeqLen = 0;

    // Process batches 1..N-1 first (stash each), then batch 0 last (stays in scratch = row 0)
    auto buildOneBatch = [&](int32_t b) -> bool {
        auto const& req = requests[b];
        if (req.fullText.empty() && req.textTokenIds.empty())
        {
            LOG_ERROR("Omni request batch %d has empty fullText and no textTokenIds", b);
            return false;
        }

        std::vector<int32_t> const textTokenIds
            = req.textTokenIds.empty() ? mTokenizer->encode(req.fullText) : req.textTokenIds;

        int32_t speakerId = mTalkerConfig.defaultSpeakerId;
        if (req.speakerId >= 0)
            speakerId = req.speakerId;
        else if (!req.speakerName.empty())
            speakerId = getSpeakerIdByName(req.speakerName);

        __half* const batchTrailingPtr
            = static_cast<__half*>(mStreamingTrailingHidden.rawPointer()) + b * trailingStride * hiddenSize;
        rt::Tensor batchTrailingBuf(
            batchTrailingPtr, rt::Coords{trailingStride, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        CUDA_CHECK(cudaMemsetAsync(batchTrailingPtr, 0, trailingStride * hiddenSize * sizeof(__half), stream));

        int32_t trailingCount = 0;
        int64_t outSeqLen = 0;
        if (!buildTalkerPrefillFromSegments(textTokenIds, req.thinkerPrefillEmbeds, req.thinkerHiddenStates,
                req.prefillLength, mTextEmbeddingTable, speakerId, batchTrailingBuf, trailingCount, outSeqLen, stream))
        {
            return false;
        }

        finalizeTrailing(batchTrailingBuf, trailingCount, stream);
        trailingCount++;

        perBatchSeqLen[b] = outSeqLen;
        maxOutSeqLen = std::max(maxOutSeqLen, outSeqLen);

        perBatchTrailingViews[b]
            = rt::Tensor(batchTrailingPtr, rt::Coords{static_cast<int64_t>(trailingCount), hiddenSize},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        trailingPtrs[b] = &perBatchTrailingViews[b];
        states[b].rvqCodes.reserve(req.maxAudioLength);
        return true;
    };

    // Build batches 1..N-1 first and stash each to slot (b * maxInputSeqLen)
    for (int32_t b = activeBatchSize - 1; b >= 1; --b)
    {
        if (!buildOneBatch(b))
            return false;

        __half* const slotPtr = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + b * maxInputSeqLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(slotPtr, mTalkerInputEmbeds.rawPointer(),
            perBatchSeqLen[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }
    // Build batch 0 last — result stays at row 0 (no copy needed)
    if (!buildOneBatch(0))
        return false;

    // Phase 2: Assemble contiguous padded [BS, maxOutSeqLen, H] from per-batch slots.
    // Slot layout: batch 0 at row 0, batch b>=1 at row (b * maxInputSeqLen).
    // Target layout: batch b at row (b * maxOutSeqLen).
    // Process in reverse so high-address batches are moved first (avoids overlap).
    __half* const base = static_cast<__half*>(mTalkerInputEmbeds.rawPointer());
    for (int32_t b = activeBatchSize - 1; b >= 0; --b)
    {
        __half* const src = base + (b == 0 ? 0 : b * maxInputSeqLen * hiddenSize);
        __half* const dst = base + b * maxOutSeqLen * hiddenSize;
        if (src != dst)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                dst, src, perBatchSeqLen[b] * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }
        if (perBatchSeqLen[b] < maxOutSeqLen)
        {
            CUDA_CHECK(cudaMemsetAsync(dst + perBatchSeqLen[b] * hiddenSize, 0,
                (maxOutSeqLen - perBatchSeqLen[b]) * hiddenSize * sizeof(__half), stream));
        }
    }

    // Single batched Talker prefill with per-batch context lengths
    check::check(mTalkerInputEmbeds.reshape({activeBatchSize, maxOutSeqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({activeBatchSize, maxOutSeqLen, mTalkerConfig.talkerHiddenSize}),
        "Tensor reshape failed");

    {
        TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
        if (!executeTalkerPrefillStep(
                mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream, perBatchSeqLen))
        {
            LOG_ERROR("Batched Talker prefill failed");
            return false;
        }
    }

    // Phase 3: Per-batch logit adjustment and sampling from batched logits [BS, vocabSize]
    check::check(mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
    check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        rt::Tensor logitsSlice(static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
            rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

        kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, logitsSlice, mTalkerConfig.talkerVocabSize - 1024,
            mTalkerConfig.talkerVocabSize, mTalkerConfig.codecEosId, 0, repetitionPenalty, stream);
    }

    trt_edgellm::topKtopPSamplingFromLogits(
        mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams, mSamplingWorkspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].codecToken = hostTokens[b];
        LOG_INFO("Omni batch %d: first codec token: %d, trailing=%d, seqLen=%ld", b, states[b].codecToken,
            static_cast<int32_t>(perBatchTrailingViews[b].getShape()[0]), perBatchSeqLen[b]);
    }

    // Always record (unconditional, matches handleAudioGeneration / handleStreamingGeneration):
    // streaming callers consume the event for TTFC measurement even without profiling enabled.
    // Cost is negligible (~µs per request).
    if (mTtfaEnd)
    {
        CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
    }

    int32_t const talkerKVCapacity = mTalkerLLMConfig.maxKVCacheCapacity;
    int32_t effectiveMaxFrames = 0;
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        int32_t const eff = std::min(requests[b].maxAudioLength,
            std::max(1, talkerKVCapacity - static_cast<int32_t>(requests[b].textTokenIds.size())));
        effectiveMaxFrames = std::max(effectiveMaxFrames, eff);
    }

    if (!runTalkerGenerationLoop(states, activeBatchSize, effectiveMaxFrames, talkerSamplingParams,
            predictorSamplingParams, repetitionPenalty, trailingPtrs, stream, perBatchSeqLen))
    {
        return false;
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        response.batchRvqCodes.push_back(std::move(states[b].rvqCodes));
        response.numFramesPerSample.push_back(states[b].talkerFrames);
    }

    int32_t totalFrames = 0;
    bool anyHitEos = false;
    for (auto const& s : states)
    {
        totalFrames += s.talkerFrames;
        if (s.codecToken == mTalkerConfig.codecEosId)
        {
            anyHitEos = true;
        }
    }
    mMultimodalMetrics.recordRun(0, 0, activeBatchSize, totalFrames);

    if (getProfilingEnabled())
    {
        int32_t const codesPerFrame = mTalkerConfig.numCodeGroups;
        auto talkerPrefillData = gTimer.getTimingData(metrics::StageNames::kTALKER_PREFILL);
        float prefillMs = talkerPrefillData ? talkerPrefillData->getTotalGpuTimeMs() : 0.0f;
        int32_t prefillSeqLen = perBatchSeqLen.empty() ? 0 : static_cast<int32_t>(perBatchSeqLen[0]);

        mOmniTalkerMetrics.recordRun(totalFrames, totalFrames * codesPerFrame, prefillMs, prefillSeqLen,
            anyHitEos ? "eos" : "max_length", false);

        auto talkerGenData = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float talkerGenMs = talkerGenData ? talkerGenData->getTotalGpuTimeMs() : 0.0f;
        float audioDurationS
            = static_cast<float>(totalFrames * kAudioSamplesPerFrame) / static_cast<float>(kAudioSampleRate);
        mOmniLatencyMetrics.audioDurationSeconds = audioDurationS;
        mOmniLatencyMetrics.audioSamples = static_cast<int64_t>(totalFrames) * kAudioSamplesPerFrame;
        mOmniLatencyMetrics.sampleRate = kAudioSampleRate;
        mOmniLatencyMetrics.realTimeFactor = (talkerGenMs > 0.0f) ? (audioDurationS / (talkerGenMs / 1000.0f)) : 0.0f;
    }

    response.success = true;
    return true;
}

// ========== Shared Generation Loop ==========

bool Qwen3OmniTTSRuntime::runTalkerGenerationLoop(std::vector<PerBatchTalkerState>& states, int32_t activeBatchSize,
    int32_t maxFrames, SamplingParams const& talkerSamplingParams, SamplingParams const& predictorSamplingParams,
    float repetitionPenalty, std::vector<rt::Tensor const*> const& trailingTextHiddens, cudaStream_t stream,
    std::vector<int64_t> const& prefillSeqLens, std::vector<PerBatchStreamingHandler> const& streamingHandlers)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runTalkerGenerationLoop", nvtx_colors::PURPLE);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t unfinished = activeBatchSize;

    // Per-batch streaming chunk emitters (no-op for batches that have streaming disabled).
    std::vector<ChunkEmitter> emitters(activeBatchSize);
    if (!streamingHandlers.empty())
    {
        check::check(static_cast<int32_t>(streamingHandlers.size()) == activeBatchSize,
            "streamingHandlers size mismatch with activeBatchSize");
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            emitters[b].chunkFrames = streamingHandlers[b].chunkFrames;
            emitters[b].onChunk = streamingHandlers[b].onChunk;
        }
    }

    // Initialize per-batch seen token tracking. numSeenTokens starts at 0 so the repetition
    // penalty kernel reads no entries on the first decode frame, avoiding a read of the
    // unseeded GPU buffer. The in-loop code below records tokens into both the host set and
    // GPU buffer starting from the first iteration.
    // NOTE: the host set / GPU buffer are still off-by-one (host set tracks the token entering
    // each iteration while the GPU buffer receives the newly sampled token). That functional
    // drift is pre-existing and tracked as a follow-up together with the standalone-TTS
    // Known Chinese-prompt prefill-EOS anomaly; pending technical follow-up.
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        states[b].numSeenTokens = 0;
    }

    int32_t globalFrame = 0;
    {
        TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);

        while (unfinished > 0 && globalFrame < maxFrames)
        {
            // ---- Phase A: extract per-batch Talker last hidden into a stacked [activeBS, H] buffer ----
            auto const& fullShape = mTalkerHiddenStatesBuffer.getShape();
            int64_t const paddedSeqDim = fullShape[1];
            int64_t const hDim = fullShape[2];
            check::check(
                mTalkerLastHidden.reshape({activeBatchSize, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
            std::vector<int32_t> activeBatchCodes(activeBatchSize, 0);
            std::vector<bool> activeMask(activeBatchSize, true);
            int32_t activeBatchPresentCount = 0;
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                {
                    activeMask[b] = false;
                    continue;
                }
                ++activeBatchPresentCount;
                activeBatchCodes[b] = states[b].codecToken;

                int64_t const actualSeqDim
                    = (!prefillSeqLens.empty() && paddedSeqDim > 1) ? prefillSeqLens[b] : paddedSeqDim;
                rt::Tensor batchHiddenView(
                    static_cast<__half*>(mTalkerHiddenStatesBuffer.rawPointer()) + b * paddedSeqDim * hDim,
                    rt::Coords{1, actualSeqDim, hDim}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
                rt::Tensor outSlice(
                    static_cast<__half*>(mTalkerLastHidden.rawPointer()) + b * mTalkerConfig.talkerHiddenSize,
                    rt::Coords{1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
                if (!extractTalkerLastHidden(batchHiddenView, outSlice, stream))
                {
                    LOG_ERROR("extractTalkerLastHidden failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    activeMask[b] = false;
                    --activeBatchPresentCount;
                }
            }

            // ---- Phase B: CodePredictor — single batched call for all active batches.
            // The unified runCodePredictorGenerationForFrame handles activeBatchSize=1..maxBS
            // uniformly (same engine call, just different batch dim). Finished batches still
            // occupy a slot (dummy work) until a future evict pass is added.
            std::vector<std::vector<int32_t>> framesPerBatch;
            if (!runCodePredictorGenerationForFrame(activeBatchSize, activeBatchCodes, mTalkerLastHidden,
                    predictorSamplingParams, framesPerBatch, stream))
            {
                LOG_ERROR("CodePredictor failed at frame %d", globalFrame);
                return false;
            }

            // ---- Phase C: per-batch residual (kernel is per-batch; runtime loops over slots) ----
            int64_t const codecHiddensRowStride = mNumCodesPerFrame * mTalkerConfig.talkerHiddenSize;
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished || !activeMask[b])
                    continue;

                states[b].rvqCodes.push_back(std::move(framesPerBatch[b]));

                // Streaming chunk accumulate (no-op for non-streaming batches). Final flush happens
                // post-loop for every active emitter so isFinal=true is emitted exactly once per
                // streaming batch regardless of which exit path was taken.
                emitters[b].append(states[b].rvqCodes.back());

                rt::Tensor residualSlice(
                    static_cast<__half*>(mResidualEmbedBuffer.rawPointer()) + b * mTalkerConfig.talkerHiddenSize,
                    rt::Coords{1, 1, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

                // Per-batch view into mCodecHiddensBuffer[b] for this batch's residual computation.
                rt::Tensor codecHiddensView(
                    static_cast<__half*>(mCodecHiddensBuffer.rawPointer()) + b * codecHiddensRowStride,
                    rt::Coords{1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
                    nvinfer1::DataType::kHALF);

                if (!computeResidualConnection(codecHiddensView, states[b].rvqCodes.back(), trailingTextHiddens[b],
                        globalFrame, residualSlice, stream))
                {
                    LOG_ERROR("Residual connection failed for batch %d frame %d", b, globalFrame);
                    states[b].finished = true;
                    states[b].talkerError = true;
                    unfinished--;
                    continue;
                }
            }

            // Batched Talker decode step: mResidualEmbedBuffer [BS, 1, H] → mTalkerLogits [BS, vocabSize]
            check::check(mResidualEmbedBuffer.reshape({activeBatchSize, 1, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");
            check::check(mTalkerHiddenStatesBuffer.reshape({activeBatchSize, 1, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");

            if (!executeTalkerDecodingStep(mResidualEmbedBuffer, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
            {
                LOG_ERROR("Batched Talker decoding step failed at frame %d", globalFrame);
                return false;
            }

            // Per-batch logit adjustment (different seenTokens per batch)
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Create views into batch b's logits row
                rt::Tensor logitsSlice(
                    static_cast<float*>(mTalkerLogits.rawPointer()) + b * mTalkerConfig.talkerVocabSize,
                    rt::Coords{1, mTalkerConfig.talkerVocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);

                // Per-batch seenTokens buffer
                rt::Tensor seenBufSlice(
                    mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity,
                    rt::Coords{mTalkerLLMConfig.maxKVCacheCapacity}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);

                kernel::invokeTalkerLogitAdjust(seenBufSlice, logitsSlice, mTalkerConfig.talkerVocabSize - 1024,
                    mTalkerConfig.talkerVocabSize, codecEosId, states[b].numSeenTokens, repetitionPenalty, stream);
            }

            // Batched sampling
            check::check(
                mTalkerLogits.reshape({activeBatchSize, mTalkerConfig.talkerVocabSize}), "Tensor reshape failed");
            check::check(mTalkerSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
                mSamplingWorkspace, stream, 42, static_cast<uint64_t>(globalFrame + 1));
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
                activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Per-batch: update state, check EOS
            int32_t* hostTokens = mHostSelectedTokenIds.dataPointer<int32_t>();
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                if (states[b].finished)
                    continue;

                // Track seen token for repetition penalty
                if (states[b].seenTokenSet.insert(states[b].codecToken).second)
                {
                    int32_t* seenBuf
                        = mSeenCodecTokensBuf.dataPointer<int32_t>() + b * mTalkerLLMConfig.maxKVCacheCapacity;
                    CUDA_CHECK(cudaMemcpyAsync(seenBuf + states[b].numSeenTokens,
                        mTalkerSelectedIndices.dataPointer<int32_t>() + b, sizeof(int32_t), cudaMemcpyDeviceToDevice,
                        stream));
                    states[b].numSeenTokens++;
                }

                states[b].codecToken = hostTokens[b];
                states[b].talkerFrames++;

                if (states[b].codecToken == codecEosId || states[b].talkerFrames >= maxFrames)
                {
                    states[b].finished = true;
                    unfinished--;
                }
            }
            globalFrame++;
        }
    }

    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        bool const hitEos = (states[b].codecToken == codecEosId);
        LOG_INFO("Batch %d: %d audio frames (exit: %s)", b, states[b].talkerFrames, hitEos ? "EOS" : "maxFrames");
    }

    // Final per-batch flush: every active emitter gets exactly one isFinal=true callback (with
    // remaining buffered codes, or empty buffer as an end-of-stream signal).
    for (auto& emitter : emitters)
    {
        emitter.flushFinal();
    }

    return true;
}

bool Qwen3OmniTTSRuntime::runSingleTalkerDecodeFrame(int32_t& codecToken, SamplingParams const& talkerSamplingParams,
    SamplingParams const& predictorSamplingParams, rt::Tensor const* trailingPtr, int32_t frameIdx,
    std::unordered_set<int32_t>& seenTokenSet, int32_t& numSeenTokens, float repetitionPenalty,
    std::vector<std::vector<int32_t>>& rvqCodes, cudaStream_t stream)
{
    int32_t const codecEosId = mTalkerConfig.codecEosId;

    // Streaming TTFA path: always bs=1. Reset CP KV is now done inside runCodePredictorGenerationForFrame.
    if (!extractTalkerLastHidden(mTalkerHiddenStatesBuffer, mTalkerLastHidden, stream))
    {
        LOG_ERROR("extractTalkerLastHidden failed at frame %d", frameIdx);
        return false;
    }

    // Wrap mTalkerLastHidden as a [1, talkerH] batched view for the unified CP call.
    rt::Tensor talkerLastBatched(mTalkerLastHidden.rawPointer(), rt::Coords{1, mTalkerConfig.talkerHiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    std::vector<std::vector<int32_t>> framesPerBatch;
    if (!runCodePredictorGenerationForFrame(/*activeBatchSize=*/1, std::vector<int32_t>{codecToken}, talkerLastBatched,
            predictorSamplingParams, framesPerBatch, stream))
    {
        LOG_ERROR("CodePredictor generation failed at frame %d", frameIdx);
        return false;
    }

    rvqCodes.push_back(std::move(framesPerBatch[0]));

    // Per-batch view (batch 0) into mCodecHiddensBuffer for residual.
    rt::Tensor codecHiddensView(mCodecHiddensBuffer.rawPointer(),
        rt::Coords{1, mNumCodesPerFrame, mTalkerConfig.talkerHiddenSize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF);
    if (!computeResidualConnection(
            codecHiddensView, rvqCodes.back(), trailingPtr, frameIdx, mResidualEmbedBuffer, stream))
    {
        LOG_ERROR("Residual connection failed at frame %d", frameIdx);
        return false;
    }

    check::check(mResidualEmbedBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");
    check::check(mTalkerHiddenStatesBuffer.reshape({1, 1, mTalkerConfig.talkerHiddenSize}), "Tensor reshape failed");

    if (!executeTalkerDecodingStep(mResidualEmbedBuffer, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
    {
        LOG_ERROR("Talker decoding step failed at frame %d", frameIdx);
        return false;
    }

    kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
        mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
    trt_edgellm::topKtopPSamplingFromLogits(mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams,
        mSamplingWorkspace, stream, 42, static_cast<uint64_t>(frameIdx + 1));
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(), sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (seenTokenSet.insert(codecToken).second)
    {
        CUDA_CHECK(cudaMemcpyAsync(mSeenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
            mTalkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        ++numSeenTokens;
    }

    codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
    return true;
}

bool Qwen3OmniTTSRuntime::runCodePredictorGenerationForFrame(int32_t activeBatchSize,
    std::vector<int32_t> const& codecTokensPerBatch, rt::Tensor const& talkerLastHiddenBatched,
    SamplingParams const& samplingParams, std::vector<std::vector<int32_t>>& outputCodesPerBatch, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::runCodePredictorGenerationForFrame", nvtx_colors::ORANGE);

    int64_t const talkerH = mTalkerConfig.talkerHiddenSize;
    int64_t const cpH = mTalkerConfig.codePredictorHiddenSize;
    int32_t const codebookSize = mTalkerConfig.codebookSize;

    check::check(activeBatchSize >= 1 && activeBatchSize <= mMaxBatchSize,
        "runCodePredictorGenerationForFrame: activeBatchSize out of range");
    check::check(static_cast<int32_t>(codecTokensPerBatch.size()) == activeBatchSize,
        "codecTokensPerBatch.size() must equal activeBatchSize");
    if (!mUseSmallToMtpProjection)
    {
        check::check(talkerH == cpH, "no-projection CP path requires talkerH == cpH");
    }

    // Helper: returns a [activeBatchSize, cpH] tensor view containing srcTalkerSpace2D projected
    // into CP hidden space. When mUseSmallToMtpProjection is false (talkerH == cpH), the source
    // view IS already the cp view — returned directly. Otherwise, invokeLinearLayer writes the
    // projected output into projectScratch (which must be sized [activeBatchSize, cpH]).
    auto projectToCpView = [&](rt::Tensor const& srcTalkerSpace2D, rt::Tensor& projectScratch) -> rt::Tensor const* {
        if (!mUseSmallToMtpProjection)
        {
            return &srcTalkerSpace2D;
        }
        check::check(projectScratch.reshape({activeBatchSize, cpH}), "Tensor reshape failed");
        kernel::invokeLinearLayer(srcTalkerSpace2D, mSmallToMtpWeight, mSmallToMtpBias, projectScratch, stream);
        return &projectScratch;
    };

    // Output containers: code_0 (from Talker) + 1..mNumRvqLayers (from CP).
    outputCodesPerBatch.assign(activeBatchSize, std::vector<int32_t>{});
    for (int32_t b = 0; b < activeBatchSize; ++b)
    {
        outputCodesPerBatch[b].reserve(mNumCodesPerFrame);
        outputCodesPerBatch[b].push_back(codecTokensPerBatch[b]);
    }

    // ---- Step 1: Batched lookup of code_0 embeddings (Talker codec_embedding, talkerH-dim) ----
    check::check(mCodePredictorCodecIds.reshape({activeBatchSize, 1}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(mCodePredictorCodecIds.rawPointer(), codecTokensPerBatch.data(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    check::check(mRawCodecEmbed.reshape({activeBatchSize, 1, talkerH}), "Tensor reshape failed");
    kernel::embeddingLookup(mCodePredictorCodecIds, mTalkerEmbeddingTable, std::nullopt, mRawCodecEmbed, stream);

    // ---- Step 2: Build [activeBS, 2, cpH] prefill input — slot 0 = proj(talker_h), slot 1 = proj(code_0_embed)
    check::check(mCodePredictorPrefillInput.reshape({activeBatchSize, 2, cpH}), "Tensor reshape failed");
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_PREFILL, stream);

        // 2D views into the [activeBS, 1, talkerH] sources to feed projectToCpView.
        rt::Tensor talkerInput2D(const_cast<__half*>(static_cast<__half const*>(talkerLastHiddenBatched.rawPointer())),
            rt::Coords{activeBatchSize, talkerH}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor rawCodec2D(mRawCodecEmbed.rawPointer(), rt::Coords{activeBatchSize, talkerH}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF);

        rt::Tensor const* talkerCpView = projectToCpView(talkerInput2D, mSmallToMtpProjectedHidden);
        rt::Tensor const* codecCpView = projectToCpView(rawCodec2D, mCodePredictorCodecEmbed);

        // Interleave the two per-batch [bs, cpH] tensors into [bs, 2, cpH] prefill input.
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            __half* dst = static_cast<__half*>(mCodePredictorPrefillInput.rawPointer()) + b * 2 * cpH;
            __half const* talkerSrc = static_cast<__half const*>(talkerCpView->rawPointer()) + b * cpH;
            __half const* codecSrc = static_cast<__half const*>(codecCpView->rawPointer()) + b * cpH;
            CUDA_CHECK(cudaMemcpyAsync(dst, talkerSrc, cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst + cpH, codecSrc, cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }

        // ---- Step 3: CP prefill engine call (produces logits for code_1) ----
        check::check(mCodePredictorHiddenStatesBuffer.reshape({activeBatchSize, 2, cpH}), "Tensor reshape failed");
        rt::Tensor& logitsHead0 = mCodePredictorLogitsPerHead[0];
        if (!executeCodePredictorPrefillStep(
                mCodePredictorPrefillInput, /*lmHeadIdx=*/0, logitsHead0, mCodePredictorHiddenStatesBuffer, stream))
        {
            return false;
        }

        // ---- Step 4: Sample code_1 (batched) ----
        check::check(logitsHead0.reshape({activeBatchSize, codebookSize}), "Tensor reshape failed");
        check::check(mCodePredictorSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
        SamplingParams const perCallParams(
            activeBatchSize, codebookSize, samplingParams.temperature, samplingParams.topK, samplingParams.topP);
        trt_edgellm::topKtopPSamplingFromLogits(
            logitsHead0, mCodePredictorSelectedIndices, perCallParams, mSamplingWorkspace, stream);
        check::check(mHostSelectedCodeIds.reshape({activeBatchSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedCodeIds.rawPointer(), mCodePredictorSelectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int32_t* hostCodes = mHostSelectedCodeIds.dataPointer<int32_t>();
        for (int32_t b = 0; b < activeBatchSize; ++b)
        {
            outputCodesPerBatch[b].push_back(hostCodes[b]); // code_1
        }
    } // end kCODEPREDICTOR_PREFILL scope

    // ---- Step 5: Mid-RVQ codec hiddens buffer for residual.
    // Layout: mCodecHiddensBuffer[maxBS, mNumCodesPerFrame, talkerH].
    // Positions 0 and mNumRvqLayers are filled by computeResidualConnection (Talker embed lookup).
    // Positions 1..mNumRvqLayers-1 are filled here from the per-step raw codec embedding.
    check::check(mCodecHiddensBuffer.reshape({activeBatchSize, mNumCodesPerFrame, talkerH}), "Tensor reshape failed");

    // ---- Step 6: Decode loop for codes 2 .. mNumRvqLayers (batched) ----
    //
    // Tokens flow GPU-only across the loop: each step's embeddingLookup reads
    // the PREVIOUS step's sample directly from ``mCodePredictorSelectedIndices``
    // (device tensor) instead of round-tripping through host memory.  Per-step
    // sample is async-D2H'd into row ``step - 2`` of ``mHostGenCodeBuf`` on
    // pinned host memory; one ``cudaStreamSynchronize`` at the end of the loop
    // drains all 14 steps × activeBatchSize codes.
    //
    // CP has no EOS / no per-token streaming callback, so the host doesn't need
    // any sample value during the inner loop — making this loop a strictly
    // safer place to defer sync than the Thinker/Talker loops.  AR ordering is
    // preserved by stream order: step k's topK kernel happens-before step k+1's
    // embeddingLookup on the same CUDA stream.
    //
    // Entry condition: ``mCodePredictorSelectedIndices`` already holds code_1
    // from the prefill block above (its topK wrote there), so step 2 reads it
    // correctly.  See the prefill block for the post-sample state.
    int32_t* const hostGenBuf = mHostGenCodeBuf.dataPointer<int32_t>();
    int64_t const hostGenBufStride = mMaxBatchSize; // row stride of mHostGenCodeBuf
    int32_t const numGenSteps = mNumRvqLayers - 1;
    {
        TIME_STAGE(metrics::StageNames::kCODEPREDICTOR_GENERATION, stream);
        for (int32_t step = 2; step <= mNumRvqLayers; ++step)
        {
            int32_t const embedIdx = step - 2;  // step=2 → codec_embedding[0]
            int32_t const lmHeadIdx = step - 1; // step=2 → lm_head[1]

            // Lookup code_(step-1) for each batch from the PREVIOUS step's
            // sampled-token device tensor — skip the host round-trip via
            // outputCodesPerBatch.back() + H2D copy that the original
            // implementation did, since outputCodesPerBatch isn't drained
            // until end-of-loop under the deferred-sync scheme.
            check::check(mCodePredictorSelectedIndices.reshape({activeBatchSize, 1}), "Tensor reshape failed");
            check::check(mRawCodecEmbed.reshape({activeBatchSize, 1, talkerH}), "Tensor reshape failed");
            kernel::embeddingLookup(mCodePredictorSelectedIndices, mCodePredictorEmbeddingTables[embedIdx],
                std::nullopt, mRawCodecEmbed, stream);

            // Save raw embedding to mCodecHiddensBuffer[b][step-1] for residual (matches PyTorch position mapping).
            int32_t const savePos = step - 1; // 1..mNumRvqLayers-1
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                __half* dst = static_cast<__half*>(mCodecHiddensBuffer.rawPointer())
                    + (b * mNumCodesPerFrame + savePos) * talkerH;
                __half const* src = static_cast<__half const*>(mRawCodecEmbed.rawPointer()) + b * talkerH;
                CUDA_CHECK(cudaMemcpyAsync(dst, src, talkerH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            }

            // Project raw codec embed → mCodePredictorCodecEmbed [activeBS, 1, cpH].
            rt::Tensor rawCodec2D(mRawCodecEmbed.rawPointer(), rt::Coords{activeBatchSize, talkerH},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
            rt::Tensor const* projectedView = projectToCpView(rawCodec2D, mSmallToMtpProjectedHidden);

            check::check(mCodePredictorCodecEmbed.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                __half* dst = static_cast<__half*>(mCodePredictorCodecEmbed.rawPointer()) + b * cpH;
                __half const* src = static_cast<__half const*>(projectedView->rawPointer()) + b * cpH;
                CUDA_CHECK(cudaMemcpyAsync(dst, src, cpH * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
            }

            // CP decode engine call.
            rt::Tensor& logitsForHead = mCodePredictorLogitsPerHead[lmHeadIdx];
            check::check(mCodePredictorHiddenStatesBuffer.reshape({activeBatchSize, 1, cpH}), "Tensor reshape failed");
            if (!executeCodePredictorDecodingStep(
                    mCodePredictorCodecEmbed, lmHeadIdx, logitsForHead, mCodePredictorHiddenStatesBuffer, stream))
            {
                return false;
            }

            // Sample code_step.
            check::check(logitsForHead.reshape({activeBatchSize, codebookSize}), "Tensor reshape failed");
            SamplingParams const perCallParams(
                activeBatchSize, codebookSize, samplingParams.temperature, samplingParams.topK, samplingParams.topP);
            trt_edgellm::topKtopPSamplingFromLogits(
                logitsForHead, mCodePredictorSelectedIndices, perCallParams, mSamplingWorkspace, stream);

            // Async D2H to row (step - 2) × maxBS of pinned host buffer;
            // deliberately NO cudaStreamSynchronize here — the value will be
            // drained after the loop.  Pinned (cudaMallocHost) backing makes
            // this truly async.
            int32_t const rowIdx = step - 2;
            CUDA_CHECK(
                cudaMemcpyAsync(hostGenBuf + rowIdx * hostGenBufStride, mCodePredictorSelectedIndices.rawPointer(),
                    activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        }

        // Single sync per frame: blocks until all numGenSteps × activeBatchSize
        // D2H copies (and the sampling kernels they depend on) have completed.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int32_t row = 0; row < numGenSteps; ++row)
        {
            for (int32_t b = 0; b < activeBatchSize; ++b)
            {
                outputCodesPerBatch[b].push_back(hostGenBuf[row * hostGenBufStride + b]); // code_(row+2)
            }
        }
    }

    return true;
}

bool Qwen3OmniTTSRuntime::computeResidualConnection(rt::Tensor const& codecHiddensThisBatch,
    std::vector<int32_t> const& codes, rt::Tensor const* trailingTextHidden, int32_t generationStep,
    rt::Tensor& outputResidual, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TalkerRunner::computeResidualConnection", nvtx_colors::BLUE);

    // PyTorch (modeling_qwen3_omni.py:3429-3434):
    //   if generation_step < trailing_text_hidden.shape[1]:
    //       inputs_embeds += trailing_text_hidden[:, generation_step]
    //   else:
    //       inputs_embeds += tts_pad_embed
    //
    // codecHiddensThisBatch: [1, mNumCodesPerFrame, talkerH] view into per-batch slot of mCodecHiddensBuffer.
    // The kernel reads middle positions [1..mNumRvqLayers-1] (filled by CP decode loop) and
    // fills positions 0 and mNumRvqLayers via Talker/CP embedding lookup using codes[0] / codes[last].

    check::check(static_cast<int32_t>(codes.size()) == mNumCodesPerFrame,
        "Expected " + std::to_string(mNumCodesPerFrame) + " codes, got " + std::to_string(codes.size()));

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    if (!outputResidual.reshape({1, 1, hiddenSize}))
    {
        check::check(
            outputResidual.getShape() == rt::Coords{1, 1, hiddenSize}, "Non-owning residual tensor has wrong shape");
    }

    __half const* addend = mTtsPadEmbed.dataPointer<__half>();
    if (trailingTextHidden != nullptr)
    {
        int64_t const trailingLen = trailingTextHidden->getShape()[0];
        if (generationStep < trailingLen)
        {
            addend = static_cast<__half const*>(trailingTextHidden->rawPointer()) + generationStep * hiddenSize;
        }
    }

    kernel::invokeResidualConnection(codecHiddensThisBatch, mTalkerEmbeddingTable,
        mCodePredictorEmbeddingTables[mNumRvqLayers - 1], codes[0], codes[mNumRvqLayers], addend, outputResidual,
        stream);

    return true;
}

bool Qwen3OmniTTSRuntime::extractTalkerLastHidden(
    rt::Tensor const& talkerHiddenStates, rt::Tensor& outputLastHidden, cudaStream_t stream)
{
    auto const& shape = talkerHiddenStates.getShape();
    int32_t const numDims = shape.getNumDims();

    if (numDims != 3)
    {
        LOG_ERROR("extractTalkerLastHidden: Expected 3D tensor [batchSize, seqLen, hiddenSize], got %dD", numDims);
        return false;
    }

    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];
    int64_t const hiddenSize = shape[2];

    // Owning tensors get reshaped; non-owning views must already have the right shape.
    if (!outputLastHidden.reshape({batchSize, hiddenSize}))
    {
        check::check(outputLastHidden.getShape() == rt::Coords{batchSize, hiddenSize},
            "extractTalkerLastHidden: non-owning output has wrong shape");
    }

    // Extract last token for each batch: output[b] = input[b, seqLen-1, :]
    size_t const copySize = hiddenSize * sizeof(__half);
    for (int64_t b = 0; b < batchSize; ++b)
    {
        size_t const srcOffset = (b * seqLen + seqLen - 1) * hiddenSize * sizeof(__half);
        size_t const dstOffset = b * hiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(outputLastHidden.rawPointer()) + dstOffset,
            static_cast<char const*>(talkerHiddenStates.rawPointer()) + srcOffset, copySize, cudaMemcpyDeviceToDevice,
            stream));
    }

    return true;
}

int32_t Qwen3OmniTTSRuntime::getSpeakerIdByName(std::string const& speakerName) const
{
    auto it = mSpeakerIdMap.find(speakerName);
    if (it != mSpeakerIdMap.end())
    {
        return it->second;
    }

    LOG_WARNING(
        "Speaker '%s' not found, using default speaker ID %d", speakerName.c_str(), mTalkerConfig.defaultSpeakerId);
    return mTalkerConfig.defaultSpeakerId;
}

// ═══════════════════════════════════════════════════════════════════════════
//        Shared Decode Frame + Prefill Construction
// ═══════════════════════════════════════════════════════════════════════════

bool Qwen3OmniTTSRuntime::buildTalkerPrefillFromSegments(std::vector<int32_t> const& textTokenIds,
    rt::Tensor const* prefillEmbedPtr, rt::Tensor const* prefillHiddenPtr, int32_t prefillLen,
    rt::Tensor const& thinkerEmbedTable, int32_t speakerId, rt::Tensor& trailingTextHidden, int32_t& trailingCount,
    int64_t& outSeqLen, cudaStream_t stream)
{
    int64_t const seqLen = static_cast<int64_t>(textTokenIds.size());
    int64_t const thinkerHiddenSize = mTextEmbeddingTable.getShape()[1];
    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;

    // Step 1: Build layer-0 embeddings in mThinkerEmbedBuffer
    check::check(mThinkerEmbedBuffer.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");

    if (prefillEmbedPtr != nullptr)
    {
        int32_t const copyLen = std::min(prefillLen, static_cast<int32_t>(seqLen));
        size_t const prefillBytes = copyLen * thinkerHiddenSize * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(mThinkerEmbedBuffer.rawPointer(), prefillEmbedPtr->rawPointer(), prefillBytes,
            cudaMemcpyDeviceToDevice, stream));

        int32_t const genLen = static_cast<int32_t>(seqLen) - copyLen;
        if (genLen > 0)
        {
            check::check(mGpuTokenIdsBuffer.reshape({1, genLen}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data() + copyLen,
                genLen * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            __half* genDst = static_cast<__half*>(mThinkerEmbedBuffer.rawPointer()) + copyLen * thinkerHiddenSize;
            rt::Tensor genEmbedView(genDst, rt::Coords{1, static_cast<int64_t>(genLen), thinkerHiddenSize},
                rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
            kernel::embeddingLookup(mGpuTokenIdsBuffer, thinkerEmbedTable, std::nullopt, genEmbedView, stream);
        }
    }
    else
    {
        check::check(mGpuTokenIdsBuffer.reshape({1, seqLen}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(mGpuTokenIdsBuffer.rawPointer(), textTokenIds.data(), seqLen * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        rt::Tensor embedView(mThinkerEmbedBuffer.rawPointer(), rt::Coords{1, seqLen, thinkerHiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        kernel::embeddingLookup(mGpuTokenIdsBuffer, thinkerEmbedTable, std::nullopt, embedView, stream);
    }

    // Step 2: Project ALL tokens through text_projection MLP
    check::check(mProjectedBuffer.reshape({seqLen, hiddenSize}), "Tensor reshape failed");
    check::check(mMLPWorkspace.reshape({seqLen, thinkerHiddenSize}), "Tensor reshape failed");
    kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mProjectedBuffer, mMLPWorkspace, stream);

    // Step 3: Parse segments by <|im_start|> positions
    std::vector<SegmentInfo> segments;
    for (int64_t i = 0; i < seqLen; ++i)
    {
        if (textTokenIds[i] == kImStartTokenId)
        {
            int32_t const roleId = (i + 1 < seqLen) ? textTokenIds[i + 1] : -1;
            segments.push_back({i, seqLen, roleId});
        }
    }
    for (size_t s = 0; s + 1 < segments.size(); ++s)
    {
        segments[s].endPos = segments[s + 1].startPos;
    }

    std::vector<size_t> userSegmentIndices;
    int64_t assistantSegIdx = -1;
    for (size_t s = 0; s < segments.size(); ++s)
    {
        if (segments[s].roleId == kSystemRoleId)
            continue;
        else if (segments[s].roleId == kUserRoleId)
            userSegmentIndices.push_back(s);
        else if (segments[s].roleId == kAssistantRoleId)
            assistantSegIdx = static_cast<int64_t>(s);
    }

    if (assistantSegIdx < 0)
    {
        LOG_ERROR("buildTalkerPrefillFromSegments: could not find assistant segment");
        return false;
    }

    // Step 4: Project multimodal tokens via hidden_projection using gather/scatter kernels
    bool const hasHiddenProjection = (mHiddenFC1Weight.getShape().volume() > 0);
    if (hasHiddenProjection && prefillHiddenPtr != nullptr)
    {
        std::vector<int32_t> mmPositions;
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& seg = segments[sIdx];
            for (int64_t i = seg.startPos; i < seg.endPos && i < prefillLen; ++i)
            {
                int32_t const tid = textTokenIds[i];
                if (tid == kAudioTokenId || tid == kImageTokenId || tid == kVideoTokenId)
                    mmPositions.push_back(static_cast<int32_t>(i));
            }
        }

        if (!mmPositions.empty())
        {
            int64_t const numMM = static_cast<int64_t>(mmPositions.size());
            // Safe to repurpose mThinkerEmbedBuffer here: step 1 filled it with full-sequence
            // layer-0 embeddings, but those have already been projected into mTalkerInputEmbeds
            // by step 2's invokeTalkerMLP call. The reshape below overwrites that data, which is
            // intentional — we only need it as scratch for the gather/MLP/scatter chain.
            check::check(mThinkerEmbedBuffer.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            check::check(mMLPWorkspace.reshape({numMM, thinkerHiddenSize}), "Tensor reshape failed");
            check::check(mTalkerInputEmbeds.reshape({numMM, hiddenSize}), "Tensor reshape failed");

            // Upload indices and use vectorized gather kernel instead of per-row cudaMemcpy
            check::check(mGatherIndicesBuffer.reshape({numMM}), "Tensor reshape failed");
            CUDA_CHECK(cudaMemcpyAsync(mGatherIndicesBuffer.rawPointer(), mmPositions.data(), numMM * sizeof(int32_t),
                cudaMemcpyHostToDevice, stream));

            rt::Tensor hiddenSource(const_cast<void*>(prefillHiddenPtr->rawPointer()),
                rt::Coords{static_cast<int64_t>(prefillLen), thinkerHiddenSize}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kHALF);
            kernel::invokeGather(hiddenSource, mGatherIndicesBuffer, mThinkerEmbedBuffer, stream);

            kernel::invokeTalkerMLP(mThinkerEmbedBuffer, mHiddenFC1Weight, mHiddenFC1Bias, mHiddenFC2Weight,
                mHiddenFC2Bias, mTalkerInputEmbeds, mMLPWorkspace, stream);

            // Scatter projected multimodal embeddings back into mProjectedBuffer
            kernel::invokeScatter(mTalkerInputEmbeds, mGatherIndicesBuffer, mProjectedBuffer, stream);
        }
    }
    else if (!hasHiddenProjection || prefillHiddenPtr == nullptr)
    {
        // Zero out multimodal rows as fallback
        for (size_t sIdx : userSegmentIndices)
        {
            auto const& seg = segments[sIdx];
            for (int64_t i = seg.startPos; i < seg.endPos; ++i)
            {
                int32_t const tid = textTokenIds[i];
                if (tid == kAudioTokenId || tid == kImageTokenId || tid == kVideoTokenId)
                {
                    __half* rowPtr = static_cast<__half*>(mProjectedBuffer.rawPointer()) + i * hiddenSize;
                    CUDA_CHECK(cudaMemsetAsync(rowPtr, 0, hiddenSize * sizeof(__half), stream));
                }
            }
        }
    }

    // Step 5: Build Talker prefill input — user segments + restructured assistant preamble
    auto const& assistantSeg = segments[assistantSegIdx];
    constexpr int32_t kAssistantRestructuredLen = kNonStreamingPrefixRows + 1;
    constexpr int32_t kAssistantTrailingOffset = kAssistantPrefixLen + 1;

    int64_t userTotalLen = 0;
    for (size_t sIdx : userSegmentIndices)
        userTotalLen += (segments[sIdx].endPos - segments[sIdx].startPos);

    outSeqLen = userTotalLen + kAssistantRestructuredLen;
    check::check(mTalkerInputEmbeds.reshape({outSeqLen, hiddenSize}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemsetAsync(mTalkerInputEmbeds.rawPointer(), 0, outSeqLen * hiddenSize * sizeof(__half), stream));

    int64_t outOffset = 0;
    for (size_t sIdx : userSegmentIndices)
    {
        auto const& seg = segments[sIdx];
        int64_t const segLen = seg.endPos - seg.startPos;
        __half const* src = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + seg.startPos * hiddenSize;
        __half* dst = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outOffset * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(dst, src, segLen * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        outOffset += segLen;
    }

    // Build restructured assistant preamble via invokeAssistantPreamble
    {
        __half const* const assistantProjPtr
            = static_cast<__half const*>(mProjectedBuffer.rawPointer()) + assistantSeg.startPos * hiddenSize;
        constexpr int64_t kPreambleFullLen = kNonStreamingPrefixRows + 1 + 2;
        int64_t const requiredCapacity = (outSeqLen + kPreambleFullLen) * hiddenSize * sizeof(__half);
        check::check(static_cast<int64_t>(mTalkerInputEmbeds.getMemoryCapacity()) >= requiredCapacity,
            "mTalkerInputEmbeds too small for preamble scratch");
        __half* const scratchPtr = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + outSeqLen * hiddenSize;
        rt::Tensor preambleScratch(
            scratchPtr, rt::Coords{kPreambleFullLen, hiddenSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        int64_t const assistantInputLen = kAssistantPrefixLen + 1;
        rt::Tensor assistantSlice(const_cast<__half*>(assistantProjPtr), rt::Coords{assistantInputLen, hiddenSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        kernel::invokeAssistantPreamble(assistantSlice, mTtsPadEmbed, mTtsBosEmbed, mTtsEosEmbed, mTalkerEmbeddingTable,
            mTalkerConfig.codecNothinkId, mTalkerConfig.codecThinkBosId, mTalkerConfig.codecThinkEosId, speakerId,
            mTalkerConfig.codecPadId, mTalkerConfig.codecBosId, 1, preambleScratch, stream);

        __half* const aOut = static_cast<__half*>(mTalkerInputEmbeds.rawPointer()) + userTotalLen * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(aOut, scratchPtr, kAssistantRestructuredLen * hiddenSize * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Step 6: Fill trailing text hidden states
    int64_t const assistantSegLen = assistantSeg.endPos - assistantSeg.startPos;
    trailingCount = std::min(static_cast<int32_t>(assistantSegLen - kAssistantTrailingOffset),
        static_cast<int32_t>(trailingTextHidden.getShape()[0]) - 1);
    if (trailingCount > 0)
    {
        __half const* trailSrc = static_cast<__half const*>(mProjectedBuffer.rawPointer())
            + (assistantSeg.startPos + kAssistantTrailingOffset) * hiddenSize;
        CUDA_CHECK(cudaMemcpyAsync(trailingTextHidden.rawPointer(), trailSrc,
            trailingCount * hiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
    }

    LOG_INFO("buildTalkerPrefillFromSegments: outSeqLen=%ld (user=%ld + assistant=%d), trailing=%d", outSeqLen,
        userTotalLen, kAssistantRestructuredLen, trailingCount);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//        Incremental Trailing Hidden Helpers (for streaming)
// ═══════════════════════════════════════════════════════════════════════════

void Qwen3OmniTTSRuntime::appendTrailingToken(int32_t tokenId, rt::Tensor const& thinkerEmbedTable,
    rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream)
{
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;

    // Upload token ID (reuse pre-allocated GPU buffer)
    CUDA_CHECK(
        cudaMemcpyAsync(mStreamingTokenId.rawPointer(), &tokenId, sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // embed_tokens(tokenId) → mStreamingTokenEmbed [1, thinkerH]
    // embeddingLookup expects [1, 1, H] output; mStreamingTokenEmbed is [1, H] — same memory, just reshape for kernel
    rt::Tensor embedView(mStreamingTokenEmbed.rawPointer(), rt::Coords{1, 1, mTalkerConfig.thinkerHiddenSize},
        rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    kernel::embeddingLookup(mStreamingTokenId, thinkerEmbedTable, std::nullopt, embedView, stream);

    // text_projection: mStreamingTokenEmbed [1, thinkerH] → mStreamingProjOut [1, talkerH]
    kernel::invokeTalkerMLP(mStreamingTokenEmbed, mTextFC1Weight, mTextFC1Bias, mTextFC2Weight, mTextFC2Bias,
        mStreamingProjOut, mStreamingMlpWork, stream);

    // Write to trailingTextHidden[trailingIdx]
    __half* dst = static_cast<__half*>(trailingTextHidden.rawPointer()) + trailingIdx * talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(
        dst, mStreamingProjOut.rawPointer(), talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

void Qwen3OmniTTSRuntime::finalizeTrailing(rt::Tensor& trailingTextHidden, int32_t trailingIdx, cudaStream_t stream)
{
    int64_t const talkerHiddenSize = mTalkerConfig.talkerHiddenSize;
    __half* dst = static_cast<__half*>(trailingTextHidden.rawPointer()) + trailingIdx * talkerHiddenSize;
    CUDA_CHECK(cudaMemcpyAsync(
        dst, mTtsEosEmbed.rawPointer(), talkerHiddenSize * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
}

// ═══════════════════════════════════════════════════════════════════════════
//        Thinker-Talker Streaming Pipeline (single CUDA stream)
// ═══════════════════════════════════════════════════════════════════════════

bool Qwen3OmniTTSRuntime::handleStreamingGeneration(LLMInferenceRuntime& thinkerRuntime,
    LLMGenerationRequest& thinkerRequest, LLMGenerationResponse& thinkerResponse,
    ThinkerTalkerStreamingConfig const& streamingConfig, OmniGenerationRequest const& omniBaseRequest,
    TalkerGenerationResponse& talkerResponse, cudaStream_t stream)
{
    NVTX_SCOPED_RANGE(nvtx_range, "TTSRuntime::handleStreamingGeneration", nvtx_colors::PURPLE);

    LOG_INFO(
        "Starting Thinker-Talker streaming pipeline (prefillThreshold=%d)", streamingConfig.talkerPrefillThreshold);

    talkerResponse.batchRvqCodes.clear();
    talkerResponse.numFramesPerSample.clear();
    talkerResponse.success = false;

    float const talkerTemperature = (omniBaseRequest.talkerTemperature > 0) ? omniBaseRequest.talkerTemperature : 0.9f;
    int32_t const talkerTopK = (omniBaseRequest.talkerTopK > 0) ? omniBaseRequest.talkerTopK : 50;
    float const talkerTopP = (omniBaseRequest.talkerTopP > 0) ? omniBaseRequest.talkerTopP : 1.0f;
    float const repetitionPenalty = omniBaseRequest.repetitionPenalty;

    SamplingParams talkerSamplingParams(1, mTalkerConfig.talkerVocabSize, talkerTemperature, talkerTopK, talkerTopP);
    SamplingParams predictorSamplingParams(
        1, mTalkerConfig.codebookSize, kCPSamplingTemperature, kCPSamplingTopK, kCPSamplingTopP);

    int32_t const codecEosId = mTalkerConfig.codecEosId;
    int32_t numSeenTokens = 0;
    std::unordered_set<int32_t> seenTokenSet;

    struct StreamingState
    {
        std::vector<int32_t> assistantTokens;
        bool thinkerFinished{false};
        bool talkerPrefillDone{false};
        bool talkerError{false};
        int32_t talkerFrames{0};
        int32_t codecToken{-1};
        int32_t trailingIdx{0};
        std::vector<std::vector<int32_t>> rvqCodes;
        std::vector<int32_t> inputIds;
    };
    StreamingState state;
    state.rvqCodes.reserve(omniBaseRequest.maxAudioLength);

    // Shared chunk emitter — same accumulator used by runTalkerGenerationLoop's TTS streaming path.
    ChunkEmitter emitter{streamingConfig.codecChunkFrames, streamingConfig.onAudioChunkReady, {}};

    int64_t const hiddenSize = mTalkerConfig.talkerHiddenSize;
    int64_t const trailingStride = mTalkerConfig.maxSeqLen + 1;
    int32_t const maxTrailingLen = static_cast<int32_t>(trailingStride);

    // Streaming uses batch slot 0 of the trailing buffer
    size_t const slot0Bytes = trailingStride * hiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMemsetAsync(mStreamingTrailingHidden.rawPointer(), 0, slot0Bytes, stream));

    rt::Tensor const& thinkerEmbedTable = thinkerRuntime.getEmbeddingTable();

    int32_t speakerId = mTalkerConfig.defaultSpeakerId;
    if (omniBaseRequest.speakerId >= 0)
    {
        speakerId = omniBaseRequest.speakerId;
    }
    else if (!omniBaseRequest.speakerName.empty())
    {
        speakerId = getSpeakerIdByName(omniBaseRequest.speakerName);
    }

    int32_t const prefillThreshold = streamingConfig.talkerPrefillThreshold;
    int32_t const maxAudioLength = omniBaseRequest.maxAudioLength;

    // Reset reuse lengths to batch=1 for streaming (per-batch independent Talker)
    check::check(mHostReuseKVCacheLengths.reshape({1}), "Tensor reshape failed");
    int32_t* reuseData = mHostReuseKVCacheLengths.dataPointer<int32_t>();
    reuseData[0] = 0;

    auto makeTrailingView = [&]() -> rt::Tensor {
        return rt::Tensor(mStreamingTrailingHidden.rawPointer(),
            rt::Coords{static_cast<int64_t>(state.trailingIdx), hiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF);
    };

    // ===== Per-token callback for the Thinker decode loop =====
    // SAFETY: This lambda captures stack-local state by reference. It is ONLY safe because
    // thinkerRuntime.handleRequest() invokes the callback synchronously on the same thread
    // (inside cudaStreamSynchronize in the decode loop). Never call this callback asynchronously.
    auto userCallback = thinkerRequest.onTokenGenerated;

    thinkerRequest.onTokenGenerated = [&, userCallback](TokenCallbackInfo const& info) {
        if (userCallback.has_value())
        {
            userCallback.value()(info);
        }
        if (info.batchIdx != 0 || state.talkerError)
        {
            return;
        }

        state.assistantTokens.push_back(info.tokenId);
        state.thinkerFinished = info.isFinished;

        int32_t const numAssistantTokens = static_cast<int32_t>(state.assistantTokens.size());

        if (!state.talkerPrefillDone && numAssistantTokens >= prefillThreshold)
        {
            LOG_INFO("Thinker produced %d assistant tokens, triggering Talker prefill", numAssistantTokens);

            // Reset KV caches
            auto& talkerCacheManager = *mTalkerSharedRes->cacheManagers[0];
            auto& cpCacheManager = *mCodePredictorSharedRes->cacheManagers[0];
            talkerCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
            cpCacheManager.resetForNewSequences(mHostReuseKVCacheLengths, stream);
            {
                auto& talkerKVManager = talkerCacheManager.getKVCacheManager();
                for (int32_t i = 0; i < talkerKVManager.numLayers(); ++i)
                {
                    rt::Tensor& layerKV = talkerKVManager.getCombinedKVCache(i);
                    CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
                }
                auto& cpKVManager = cpCacheManager.getKVCacheManager();
                for (int32_t i = 0; i < cpKVManager.numLayers(); ++i)
                {
                    rt::Tensor& layerKV = cpKVManager.getCombinedKVCache(i);
                    CUDA_CHECK(cudaMemsetAsync(layerKV.rawPointer(), 0, layerKV.getMemoryCapacity(), stream));
                }
            }

            // Build combined token IDs — fetch Thinker input IDs from the runtime portal.
            auto const& thinkerInputs = thinkerRuntime.getBaseModelInputTokenIds();
            auto& textTokenIds = state.inputIds;
            if (textTokenIds.empty() && !thinkerInputs.empty())
            {
                textTokenIds = thinkerInputs[0];
            }
            textTokenIds.insert(textTokenIds.end(), state.assistantTokens.begin(), state.assistantTokens.end());

            // Use buildTalkerPrefillFromSegments for segment parsing, MLP projection, and prefill assembly
            auto prefillEmbedPtr = thinkerRuntime.getBaseModelHiddenStates(0);
            auto prefillHiddenPtr = thinkerRuntime.getBaseModelHiddenStates(thinkerRequest.acceptHiddenLayer);
            int32_t const prefillLen = thinkerRuntime.getBaseModelPrefillLength();

            int64_t outSeqLen = 0;
            if (!buildTalkerPrefillFromSegments(textTokenIds, prefillEmbedPtr, prefillHiddenPtr, prefillLen,
                    thinkerEmbedTable, speakerId, mStreamingTrailingHidden, state.trailingIdx, outSeqLen, stream))
            {
                state.talkerError = true;
                return;
            }

            // Talker prefill
            check::check(mTalkerInputEmbeds.reshape({1, outSeqLen, hiddenSize}), "Tensor reshape failed");
            check::check(mTalkerHiddenStatesBuffer.reshape({1, outSeqLen, mTalkerConfig.talkerHiddenSize}),
                "Tensor reshape failed");

            {
                TIME_STAGE(metrics::StageNames::kTALKER_PREFILL, stream);
                if (!executeTalkerPrefillStep(mTalkerInputEmbeds, mTalkerLogits, mTalkerHiddenStatesBuffer, stream))
                {
                    LOG_ERROR("Talker prefill failed during streaming pipeline");
                    state.talkerError = true;
                    return;
                }
            }

            kernel::invokeTalkerLogitAdjust(mSeenCodecTokensBuf, mTalkerLogits, mTalkerConfig.talkerVocabSize - 1024,
                mTalkerConfig.talkerVocabSize, codecEosId, numSeenTokens, repetitionPenalty, stream);
            // Streaming Talker runs at batch=1; size selectedIndices to match talkerSamplingParams
            // (allocated at {maxBS, 1}, must be {1, 1} for the sampler's shape check).
            check::check(mTalkerSelectedIndices.reshape({1, 1}), "Tensor reshape failed");
            trt_edgellm::topKtopPSamplingFromLogits(
                mTalkerLogits, mTalkerSelectedIndices, talkerSamplingParams, mSamplingWorkspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mTalkerSelectedIndices.rawPointer(),
                sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            state.codecToken = mHostSelectedTokenIds.dataPointer<int32_t>()[0];
            if (seenTokenSet.insert(state.codecToken).second)
            {
                CUDA_CHECK(cudaMemcpyAsync(mSeenCodecTokensBuf.dataPointer<int32_t>() + numSeenTokens,
                    mTalkerSelectedIndices.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
                ++numSeenTokens;
            }
            state.talkerPrefillDone = true;
            // Always record (matches handleAudioGeneration / handleAudioGenerationFromThinker):
            // streaming callers consume the event for TTFC measurement even without profiling.
            if (mTtfaEnd)
            {
                CUDA_CHECK(cudaEventRecord(mTtfaEnd, stream));
            }
            LOG_INFO("Talker prefill done, first codec token: %d", state.codecToken);

            // Generate frame 0 immediately after prefill
            if (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                rt::Tensor trailingView = makeTrailingView();
                rt::Tensor const* trailingPtr = (state.trailingIdx > 0) ? &trailingView : nullptr;

                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        trailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty, state.rvqCodes,
                        stream))
                {
                    state.talkerError = true;
                    return;
                }
                state.talkerFrames++;
            }
        }
        else if (state.talkerPrefillDone && numAssistantTokens > prefillThreshold)
        {
            // Incremental: append new trailing token + run Talker decode step
            if (state.trailingIdx >= maxTrailingLen - 1)
            {
                LOG_WARNING("Trailing buffer full (%d/%d), skipping token append", state.trailingIdx, maxTrailingLen);
            }
            else
            {
                appendTrailingToken(
                    info.tokenId, thinkerEmbedTable, mStreamingTrailingHidden, state.trailingIdx, stream);
                state.trailingIdx++;
            }

            if (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                rt::Tensor trailingView = makeTrailingView();
                rt::Tensor const* trailingPtr = (state.trailingIdx > 0) ? &trailingView : nullptr;

                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        trailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty, state.rvqCodes,
                        stream))
                {
                    state.talkerError = true;
                    return;
                }
                state.talkerFrames++;
                emitter.append(state.rvqCodes.back());
            }
        }
    };

    // ===== Run Thinker with the callback installed =====
    auto hiddenLayers = getThinkerHiddenLayerIndices();
    if (hiddenLayers[1] < 0)
    {
        LOG_ERROR(
            "Talker config is missing 'accept_hidden_layer'; Thinker->Talker streaming requires it to "
            "match the layer index the Thinker engine emits on its hidden_states output.");
        return false;
    }
    thinkerRequest.generateAudio = true;
    thinkerRequest.acceptHiddenLayer = hiddenLayers[1];
    bool thinkerSuccess = thinkerRuntime.handleRequest(thinkerRequest, thinkerResponse, stream, true);

    if (!thinkerSuccess)
    {
        LOG_ERROR("Thinker handleRequest failed in streaming pipeline");
        return false;
    }

    // ===== After Thinker finishes: finalize trailing and flush remaining Talker frames =====
    if (state.talkerPrefillDone && !state.talkerError)
    {
        if (state.trailingIdx < maxTrailingLen)
        {
            finalizeTrailing(mStreamingTrailingHidden, state.trailingIdx, stream);
            state.trailingIdx++;
        }
        else
        {
            LOG_WARNING("Trailing buffer full, cannot append tts_eos");
        }

        LOG_INFO("Thinker done. Flushing remaining Talker frames (current: %d, codec=%d, trailingIdx=%d)",
            state.talkerFrames, state.codecToken, state.trailingIdx);

        rt::Tensor flushTrailingView = makeTrailingView();
        rt::Tensor const* flushTrailingPtr = (state.trailingIdx > 0) ? &flushTrailingView : nullptr;

        while (state.codecToken != codecEosId && state.talkerFrames < maxAudioLength)
        {
            {
                TIME_STAGE(metrics::StageNames::kTALKER_GENERATION, stream);
                if (!runSingleTalkerDecodeFrame(state.codecToken, talkerSamplingParams, predictorSamplingParams,
                        flushTrailingPtr, state.talkerFrames, seenTokenSet, numSeenTokens, repetitionPenalty,
                        state.rvqCodes, stream))
                {
                    break;
                }
            }
            state.talkerFrames++;
            emitter.append(state.rvqCodes.back());
        }

        emitter.flushFinal();
    }
    else
    {
        LOG_WARNING("Thinker finished but Talker prefill was never triggered (only %zu assistant tokens)",
            state.assistantTokens.size());
    }

    bool const hitEos = (state.codecToken == codecEosId);
    LOG_INFO("Streaming pipeline: %d audio frames (exit: %s, codec=%d)", state.talkerFrames,
        hitEos ? "EOS" : "maxAudioLength", state.codecToken);

    talkerResponse.batchRvqCodes.push_back(std::move(state.rvqCodes));
    talkerResponse.numFramesPerSample.push_back(state.talkerFrames);
    talkerResponse.success = state.talkerPrefillDone && !state.talkerError;

    mMultimodalMetrics.recordRun(0, 0, 1, state.talkerFrames);

    if (getProfilingEnabled())
    {
        int32_t const codesPerFrame = mTalkerConfig.numCodeGroups;
        auto talkerPrefillData = gTimer.getTimingData(metrics::StageNames::kTALKER_PREFILL);
        float prefillMs = talkerPrefillData ? talkerPrefillData->getTotalGpuTimeMs() : 0.0f;

        mOmniTalkerMetrics.recordRun(state.talkerFrames, state.talkerFrames * codesPerFrame, prefillMs,
            state.talkerPrefillDone ? static_cast<int32_t>(state.assistantTokens.size() + 30) : 0,
            hitEos ? "eos" : "max_length", true);

        auto talkerGenData = gTimer.getTimingData(metrics::StageNames::kTALKER_GENERATION);
        float talkerGenMs = talkerGenData ? talkerGenData->getTotalGpuTimeMs() : 0.0f;
        float audioDurationS = static_cast<float>(state.talkerFrames * 1920) / 24000.0f;
        mOmniLatencyMetrics.audioDurationSeconds = audioDurationS;
        mOmniLatencyMetrics.audioSamples = static_cast<int64_t>(state.talkerFrames) * 1920;
        mOmniLatencyMetrics.sampleRate = 24000;
        mOmniLatencyMetrics.realTimeFactor = (talkerGenMs > 0.0f) ? (audioDurationS / (talkerGenMs / 1000.0f)) : 0.0f;
    }

    return true;
}

} // namespace rt
} // namespace trt_edgellm
