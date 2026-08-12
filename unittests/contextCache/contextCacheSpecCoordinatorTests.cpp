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

#include "runtime/state/contextCache/contextCacheCoordinator.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/kvPageTable.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

constexpr int32_t kMAX_BATCH{2};
constexpr int32_t kMAX_SEQUENCE_LENGTH{512};

std::vector<int32_t> makeTokens(int32_t count)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

LLMEngineConfig makeAttentionConfig(char const* modelType, int32_t layers)
{
    LLMEngineConfig config;
    config.modelType = modelType;
    config.hiddenSize = 64;
    config.numDecoderLayers = layers;
    config.numAttentionLayers = layers;
    config.numKVHeads = 1;
    config.headDim = 8;
    config.rotaryDim = 8;
    config.maxSupportedBatchSize = kMAX_BATCH;
    config.maxSupportedInputLength = kMAX_SEQUENCE_LENGTH;
    config.maxKVCacheCapacity = kMAX_SEQUENCE_LENGTH;
    int64_t const minimumActivePages = computeMinimumKvPoolPages(kMAX_BATCH, kMAX_SEQUENCE_LENGTH);
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES, "Test KV pool page count must fit int32.");
    config.kvPoolPages = static_cast<int32_t>(minimumActivePages);
    config.kvCacheDtype = DataType::kHALF;
    config.layerTypes.assign(static_cast<size_t>(layers), HybridCacheManager::LayerType::kAttention);
    config.kvLayerConfigs.assign(static_cast<size_t>(layers), KVLayerConfig{config.numKVHeads, config.headDim});
    return config;
}

DeploymentConfig makeEagleDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig("spec-coordinator-base", 3);
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.base.isSpecDecodeBase = true;
    deployment.base.specTargetLayerIds = {0, 2};
    deployment.draft = makeAttentionConfig("spec-coordinator-draft", 2);
    deployment.draft->specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize * 2;
    SpecDecodeConfig spec;
    spec.draftingTopK = 2;
    spec.draftingStep = 2;
    spec.verifySize = 4;
    deployment.specConfig = spec;
    return deployment;
}

HybridCacheManager::Config makeCacheConfig(LLMEngineConfig const& engine)
{
    KVCacheManager::Config kvConfig{engine.numAttentionLayers, engine.maxSupportedBatchSize, engine.maxKVCacheCapacity,
        engine.kvLayerConfigs, engine.kvCacheDtype, engine.kvPoolPages};
    MambaCacheManager::Config mambaConfig{/*.numRecurrentLayers=*/0, engine.maxSupportedBatchSize};
    return HybridCacheManager::Config{
        engine.layerTypes, std::move(kvConfig), std::move(mambaConfig), engine.maxSupportedBatchSize};
}

class ContextCacheSpecCoordinatorTests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        mDeployment = makeEagleDeployment();
        mBaseCache = std::make_unique<HybridCacheManager>(makeCacheConfig(mDeployment.base), mStream);
        mDraftCache = std::make_unique<HybridCacheManager>(makeCacheConfig(*mDeployment.draft), mStream);
        mBasePageTable = makePageTable(*mBaseCache);
        mDraftPageTable = makePageTable(*mDraftCache);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        createCoordinator();
    }

    void TearDown() override
    {
        if (mCoordinator != nullptr)
        {
            EXPECT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
            mCoordinator.reset();
        }
        mDraftPageTable.reset();
        mBasePageTable.reset();
        mDraftCache.reset();
        mBaseCache.reset();
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    std::unique_ptr<KVPageTable> makePageTable(HybridCacheManager& cache)
    {
        KVCacheManager const& kv = cache.getKVCacheManager();
        auto table = std::make_unique<KVPageTable>(kMAX_BATCH, pagesPerSlot(kv.maxCapPadded()), kv.numPages());
        table->setIdentity();
        table->upload(mStream);
        return table;
    }

    void createCoordinator(ContextCacheCoordinator::StreamSynchronizer synchronizer = {})
    {
        ContextCachePhysicalResources resources{*mBaseCache, *mBasePageTable, mDraftCache.get(), mDraftPageTable.get()};
        mCoordinator
            = std::make_unique<ContextCacheCoordinator>(ContextCacheConfig{/*.enabled=*/true, /*.maxRecords=*/16},
                mDeployment, validateContextCacheDeployment(mDeployment), resources, mStream, std::move(synchronizer));
    }

    ContextCacheCoordinator::AdmissionResult begin(
        std::vector<int32_t> tokens, ContextCacheExecutionMode executionMode = ContextCacheExecutionMode::kEAGLE)
    {
        return beginBatch({std::move(tokens)}, executionMode);
    }

    ContextCacheCoordinator::AdmissionResult beginBatch(std::vector<std::vector<int32_t>> batch,
        ContextCacheExecutionMode executionMode = ContextCacheExecutionMode::kEAGLE)
    {
        ContextCacheBatchAdmission admission;
        admission.executionMode = executionMode;
        for (auto& tokens : batch)
        {
            admission.sequences.push_back(ContextCacheSequenceAdmission{std::move(tokens), {}});
        }
        ContextCacheCoordinator::BeginRequestResult result = mCoordinator->beginRequest(admission, mStream);
        EXPECT_EQ(result.status, ContextCacheCoordinatorStatus::kOk);
        EXPECT_TRUE(result.admission.has_value());
        return std::move(*result.admission);
    }

    void freezePrefill(ContextCacheCoordinator::AdmissionResult& request, int32_t inputLength, int32_t lookahead = 9001)
    {
        ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
        // The runtime path has already consumed the base-prefill synchronization before it enqueues EAGLE
        // draft initialization. The coordinator deliberately keeps that later draft work pending.
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&lookahead, 1, inputLength}};
        std::vector<int32_t> commonLengths{inputLength};
        ASSERT_EQ(mCoordinator->finalizePrefillPublication(request.request, progress, &commonLengths),
            ContextCacheCoordinatorStatus::kOk);
    }

    void finalizeVanillaPrefill(
        ContextCacheCoordinator::AdmissionResult& request, int32_t inputLength, int32_t lookahead = 9001)
    {
        ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{&lookahead, 1, inputLength}};
        ASSERT_EQ(
            mCoordinator->finalizePrefillPublication(request.request, progress), ContextCacheCoordinatorStatus::kOk);
    }

    void completeDecode(ContextCacheCoordinator::AdmissionResult& request, std::vector<int32_t> const& acceptedTokens,
        int32_t committedLength, int32_t commonLength, bool publishTerminal)
    {
        ASSERT_EQ(mCoordinator->prepareDecodeStep(request.request), ContextCacheCoordinatorStatus::kOk);
        // EAGLE verification owns the existing round synchronization that makes all preceding row uploads and draft
        // work ready before this callback.
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        std::vector<ContextCacheSequenceAdvance> progress{ContextCacheSequenceAdvance{
            acceptedTokens.data(), static_cast<int32_t>(acceptedTokens.size()), committedLength}};
        std::vector<int32_t> commonLengths{commonLength};
        std::vector<int32_t> const completedSlots = publishTerminal ? std::vector<int32_t>{0} : std::vector<int32_t>{};
        ASSERT_EQ(mCoordinator->completeDecodeStep(request.request, progress, completedSlots, &commonLengths),
            ContextCacheCoordinatorStatus::kOk);
    }

    cudaStream_t mStream{};
    DeploymentConfig mDeployment;
    std::unique_ptr<HybridCacheManager> mBaseCache;
    std::unique_ptr<HybridCacheManager> mDraftCache;
    std::unique_ptr<KVPageTable> mBasePageTable;
    std::unique_ptr<KVPageTable> mDraftPageTable;
    std::unique_ptr<ContextCacheCoordinator> mCoordinator;
};

TEST_F(ContextCacheSpecCoordinatorTests, FirstVerificationPublishesPairAndNextRequestUsesFullPageReplay)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto producer = begin(makeTokens(kInputLength));
    freezePrefill(producer, kInputLength);
    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);

    completeDecode(producer, {41, 42}, kInputLength + 2, kInputLength, false);
    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);
    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().front());
    EXPECT_EQ(record.basePagePath.size(), 2U);
    EXPECT_EQ(record.draftPagePath.size(), 2U);
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(77);
    ContextCacheMetrics const metricsBeforeConsumer = mCoordinator->metrics();
    auto consumer = begin(std::move(continuation));
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{kTOKENS_PER_PAGE});
    ContextCacheMetrics const metricsAfterConsumer = mCoordinator->metrics();
    EXPECT_EQ(
        metricsAfterConsumer.matchedTokens, metricsBeforeConsumer.matchedTokens + static_cast<uint64_t>(kInputLength));
    EXPECT_EQ(metricsAfterConsumer.specFullPageReplays, metricsBeforeConsumer.specFullPageReplays + 1U);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, DecodeEndPublishesOnlyTheCommonBaseDraftBoundary)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE - 1};
    auto producer = begin(makeTokens(kInputLength));
    freezePrefill(producer, kInputLength);
    completeDecode(producer, {41, 42}, kInputLength + 2, kInputLength, false);
    completeDecode(producer, {43, 44}, kInputLength + 4, kInputLength + 2, true);

    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().back());
    EXPECT_EQ(record.basePagePath.size(), 2U);
    EXPECT_EQ(record.draftPagePath.size(), 2U);
    EXPECT_FALSE(record.exactCheckpointLength.has_value());
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, PrefillOnlyCompletionSynchronizesDraftBeforePublishingAndCompaction)
{
    ASSERT_EQ(mCoordinator->shutdown(), ContextCacheCoordinatorStatus::kOk);
    mCoordinator.reset();
    int32_t synchronizations{};
    createCoordinator([&](cudaStream_t stream) {
        ++synchronizations;
        return cudaStreamSynchronize(stream);
    });

    constexpr int32_t kInputLength{kTOKENS_PER_PAGE};
    auto request = begin(makeTokens(kInputLength));
    freezePrefill(request, kInputLength);
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "specCoordinatorMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(request.request, {-1}, 0, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(request.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_EQ(synchronizations, 2);
    EXPECT_EQ(mCoordinator->manager().records().size(), 1U);
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, AbandonedPendingEagleInitializationDoesNotPublishPair)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    {
        auto request = begin(makeTokens(kInputLength));
        freezePrefill(request, kInputLength);
        EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    }

    EXPECT_EQ(mCoordinator->manager().records().size(), 0U);
    ContextCacheMetrics const metricsBeforeNext = mCoordinator->metrics();
    auto next = begin(makeTokens(kInputLength));
    EXPECT_EQ(next.prefillStarts, std::vector<int32_t>{0});
    ContextCacheMetrics const metricsAfterNext = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterNext.matchedTokens, metricsBeforeNext.matchedTokens);
    EXPECT_EQ(metricsAfterNext.specFullPageReplays, metricsBeforeNext.specFullPageReplays);
    EXPECT_EQ(mCoordinator->finish(next.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, VanillaRequestReusesBaseSideOfPairedRecordWithoutTouchingDraftRows)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto producer = begin(makeTokens(kInputLength));
    freezePrefill(producer, kInputLength);
    completeDecode(producer, {41, 42}, kInputLength + 2, kInputLength, false);
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(77);
    ContextCacheMetrics const metricsBeforeVanillaConsumer = mCoordinator->metrics();
    auto consumer = begin(std::move(continuation), ContextCacheExecutionMode::kVanilla);
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{kInputLength});
    ContextCacheMetrics const metricsAfterVanillaConsumer = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterVanillaConsumer.matchedTokens,
        metricsBeforeVanillaConsumer.matchedTokens + static_cast<uint64_t>(kInputLength));
    EXPECT_EQ(metricsAfterVanillaConsumer.specFullPageReplays, metricsBeforeVanillaConsumer.specFullPageReplays);

    std::vector<int32_t> const draftRowBefore(
        mDraftPageTable->hostRow(0), mDraftPageTable->hostRow(0) + mDraftPageTable->maxPagesPerSeq());
    finalizeVanillaPrefill(consumer, kInputLength + 1);
    EXPECT_TRUE(std::equal(draftRowBefore.begin(), draftRowBefore.end(), mDraftPageTable->hostRow(0)));
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "vanillaOnEagleMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(consumer.request, {-1}, 0, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(consumer.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_TRUE(std::equal(draftRowBefore.begin(), draftRowBefore.end(), mDraftPageTable->hostRow(0)));
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> eagleContinuation = makeTokens(kInputLength);
    eagleContinuation.push_back(77);
    eagleContinuation.push_back(78);
    ContextCacheMetrics const metricsBeforeEagleConsumer = mCoordinator->metrics();
    auto eagleConsumer = begin(std::move(eagleContinuation));
    EXPECT_EQ(eagleConsumer.prefillStarts, std::vector<int32_t>{kTOKENS_PER_PAGE});
    ContextCacheMetrics const metricsAfterEagleConsumer = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterEagleConsumer.matchedTokens,
        metricsBeforeEagleConsumer.matchedTokens + static_cast<uint64_t>(kInputLength));
    EXPECT_EQ(metricsAfterEagleConsumer.specFullPageReplays, metricsBeforeEagleConsumer.specFullPageReplays + 1U);
    EXPECT_EQ(mCoordinator->finish(eagleConsumer.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, FirstRoundCompactionRemovesTerminalSlotAndKeepsEagleSurvivorExecutable)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto request = beginBatch({makeTokens(kInputLength), makeTokens(kInputLength)});
    ASSERT_EQ(mCoordinator->preparePrefill(request.request), ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::vector<int32_t> lookahead{9001, 9002};
    std::vector<ContextCacheSequenceAdvance> progress{
        ContextCacheSequenceAdvance{&lookahead[0], 1, kInputLength},
        ContextCacheSequenceAdvance{&lookahead[1], 1, kInputLength},
    };
    std::vector<int32_t> commonLengths{kInputLength, kInputLength};
    ASSERT_EQ(mCoordinator->finalizePrefillPublication(request.request, progress, &commonLengths),
        ContextCacheCoordinatorStatus::kOk);

    std::vector<int32_t> const survivingDraftRow(
        mDraftPageTable->hostRow(1), mDraftPageTable->hostRow(1) + mDraftPageTable->maxPagesPerSeq());
    Tensor deviceMapping({kMAX_BATCH}, rt::DeviceType::kGPU, DataType::kINT32, "firstRoundEagleMapping");
    ASSERT_EQ(mCoordinator->beginBatchCompaction(request.request, {-1, 0}, 1, deviceMapping),
        ContextCacheCoordinatorStatus::kOk);
    ASSERT_EQ(mCoordinator->compactBatch(request.request), ContextCacheCoordinatorStatus::kOk);
    EXPECT_TRUE(std::equal(survivingDraftRow.begin(), survivingDraftRow.end(), mDraftPageTable->hostRow(0)));

    completeDecode(request, {41, 42}, kInputLength + 2, kInputLength, false);
    EXPECT_EQ(mCoordinator->finish(request.request), ContextCacheCoordinatorStatus::kOk);
}

TEST_F(ContextCacheSpecCoordinatorTests, EagleRequestDoesNotConsumeBaseOnlyRecord)
{
    constexpr int32_t kInputLength{2 * kTOKENS_PER_PAGE};
    auto producer = begin(makeTokens(kInputLength), ContextCacheExecutionMode::kVanilla);
    finalizeVanillaPrefill(producer, kInputLength);
    EXPECT_EQ(mCoordinator->finish(producer.request), ContextCacheCoordinatorStatus::kOk);

    ASSERT_EQ(mCoordinator->manager().records().size(), 1U);
    CacheRecord const& record
        = mCoordinator->manager().records().get(mCoordinator->manager().records().lruToMru().front());
    EXPECT_EQ(record.basePagePath.size(), 2U);
    EXPECT_TRUE(record.draftPagePath.empty());

    std::vector<int32_t> continuation = makeTokens(kInputLength);
    continuation.push_back(77);
    ContextCacheMetrics const metricsBeforeConsumer = mCoordinator->metrics();
    auto consumer = begin(std::move(continuation));
    EXPECT_EQ(consumer.prefillStarts, std::vector<int32_t>{0});
    ContextCacheMetrics const metricsAfterConsumer = mCoordinator->metrics();
    EXPECT_EQ(metricsAfterConsumer.matchedTokens, metricsBeforeConsumer.matchedTokens);
    EXPECT_EQ(metricsAfterConsumer.specFullPageReplays, metricsBeforeConsumer.specFullPageReplays);
    EXPECT_EQ(mCoordinator->finish(consumer.request), ContextCacheCoordinatorStatus::kOk);
}

} // namespace
