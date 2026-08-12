/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/hashUtils.h"
#include "runtime/decoding/decodingStrategy.h"
#include "runtime/state/externalWeightManager.h"

#include <filesystem>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

class DFlashDecoder final : public DecodingStrategy
{
public:
    DFlashDecoder(DecodingRuntimeContext& runtime, std::filesystem::path const& engineDir,
        SpecDecodeDraftingConfig const& draftingConfig, std::unique_ptr<EngineExecutor> draftExecutor,
        cudaStream_t stream);

    DecodingStrategyKind kind() const noexcept override
    {
        return DecodingStrategyKind::kDFlash;
    }

    char const* name() const noexcept override
    {
        return "dflash";
    }

    bool isSpeculative() const noexcept override
    {
        return true;
    }

    bool decodeStep(DecodingInferenceContext& context) override;
    bool captureCudaGraphs(cudaStream_t stream) override;

    int64_t getRequiredContextMemorySize() const noexcept override;
    void setContextMemory(Tensor& memory) override;

    bool hasSystemPromptKVCache(SystemPromptCacheKey const& key) const override;
    void restoreSystemPromptKVCache(SystemPromptCacheKey const& key, int32_t batchIdx, cudaStream_t stream) override;
    bool runSystemPromptPrefill(DecodingInferenceContext& context) override;
    void saveSystemPromptKVCache(SystemPromptCacheKey const& key, std::string const& prompt,
        std::vector<tokenizer::Rank> const& tokenizedPrompt, int32_t promptIdsLength, cudaStream_t stream) override;

    void resetForNewSequences(Tensor& reuseLengths, cudaStream_t stream) override;
    void onBatchEvict(std::vector<int32_t> const& batchMapping, int32_t oldActiveBatch, int32_t newActiveBatch,
        Tensor& deviceBatchMapping, cudaStream_t stream, BatchCompactionMode mode) override;

private:
    bool runDraftForward(DecodingInferenceContext& context);
    bool prepareDFlashVerifyInputs(DecodingInferenceContext& context);
    bool captureDraftCudaGraphs(cudaStream_t stream);
    bool buildTreeVerifyInputs(DecodingInferenceContext& context);
    bool runBaseVerification(DecodingInferenceContext& context);
    bool executeBaseVerification(DecodingInferenceContext& context, int32_t verifySize);
    void reshapeBaseVerificationForCapture(int32_t batchSize, int32_t verifySize, bool includeTreeMetadata);
    void prepareLinearBaseVerificationMetadata(int32_t batchSize, int32_t verifySize, cudaStream_t stream);
    void copyVerifyTokenIdsToBaseInput(int32_t batchSize, int32_t verifySize, cudaStream_t stream);
    void runBaseVerificationEmbeddingLookup(
        int32_t batchSize, int32_t verifySize, cudaStream_t stream, bool reshapeGemmaPleOutputs);
    bool capturePreparedBaseVerification(int32_t batchSize, int32_t verifySize, cudaStream_t stream);
    void reshapeBaseVerificationInputsOutputs(int32_t batchSize, int32_t verifySize);
    void prepareCommonBaseVerificationInputs(int32_t batchSize, int32_t verifySize);
    void commitAcceptedTreePath(DecodingInferenceContext& context, int32_t verifySize, int32_t maxAcceptLength);
    bool checkCudaLastError(char const* stage) const;

    DecodingRuntimeContext& mRuntime;
    HybridCacheManager& mDraftCacheManager;

    std::unique_ptr<EngineExecutor> mDraftExecutor;
    TensorMap mDraftTensorMap;
    ExternalWeightManager mDraftExternalWeightManager;

    Tensor mDraftInputsEmbeds;        //!< [B, blockSize, draftHiddenSize] FP16
    Tensor mDraftTargetHidden;        //!< Compact scratch for [B, <= blockSize, baseOutputHiddenDim] FP16
    Tensor mDraftPrefillTargetHidden; //!< Lazy scratch for non-compact round-0 target hidden FP16, max batch reserve
    Tensor mDraftOutputLogits;        //!< [B, blockSize, vocabSize] FP32

    Tensor mDraftPackedAttentionMask; //!< [B, blockSize, divUp(blockSize,32)] INT32
    Tensor mDraftAttentionPosId;      //!< [B, blockSize] INT32
    Tensor mDraftContextLengths;      //!< [B] INT32
    Tensor mDraftDeltaLenCommit;      //!< [B] INT32
    Tensor mDraftDeltaLens;           //!< [B] INT32

    Tensor mDraftTokenIds;          //!< [B, blockSize] INT32
    Tensor mHostDraftInputIds;      //!< [B, blockSize] INT32 CPU
    Tensor mHostLastAcceptedTokens; //!< [B] INT32 CPU
    Tensor mHostDeltaLens;          //!< [B] INT32 CPU
    Tensor mLastAcceptedTokens;     //!< [B] INT32 GPU

    hash_utils::HashMap<SystemPromptCacheKey, SystemPromptKVCache> mSystemPromptKVCacheDraft;

    Tensor mTreeTokenIds;         //!< [B, verifySize] INT32, DDTree node tokens
    Tensor mTreeNodeScores;       //!< [B, verifySize] FP32, DDTree prefix scores
    Tensor mValidCounts;          //!< [B] INT32, DDTree valid node counts
    Tensor mVerifyTokenIds;       //!< [B, verifyTokenCount] INT32
    Tensor mVerifyTreeMask;       //!< [B, verifyTokenCount, verifyTokenCount] INT8
    Tensor mAcceptedTokenIds;     //!< [B, maxAcceptBufferSize] INT32
    Tensor mAcceptedTokenIndices; //!< [B, maxAcceptBufferSize] INT32, verify logits/KV indices
    Tensor mAcceptLength;         //!< [B] INT32
    Tensor mHostAcceptLengths;    //!< [B] INT32 (CPU)
    Tensor mHostAcceptedTokenIds; //!< [B, maxAcceptBufferSize] INT32 (CPU)
    Tensor mBuildWorkspace;       //!< DDTree build workspace bytes

    //! DFlash-specific parameters. Linear DFlash treats mBlockSize as the base
    //! verify window length and copies mProposalLen = mVerifySize - 1 draft tokens
    //! after the anchor. DDTree uses mBlockSize as the draft block horizon.
    int32_t mBlockSize{16};
    int32_t mProposalLen{15};
    int32_t mVerifySize{16};
    int32_t mCandidateTopK{1};
    int32_t mMaskTokenId{0};
    int32_t mDraftHiddenSize{0};
    int32_t mBaseOutputHiddenDim{0};
    int32_t mDraftVocabSize{0};

    bool mUseDDTree{false};

    //! Draft vocab map [reducedVocabSize] INT32 (GPU). Active when draft
    //! lm_head uses a reduced vocabulary. Sized to zero otherwise.
    Tensor mDraftVocabMappingTable;
    bool mHasDraftVocabMap{false};
};

} // namespace rt
} // namespace trt_edgellm
