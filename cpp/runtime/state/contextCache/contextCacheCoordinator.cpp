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
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/contextCache/hybridSnapshotStorage.h"
#include "runtime/state/kvPageTable.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace trt_edgellm
{
namespace rt
{
namespace
{

bool isHybridKind(ContextCacheDeploymentKind kind) noexcept
{
    return kind == ContextCacheDeploymentKind::kHybrid || kind == ContextCacheDeploymentKind::kPureRecurrent;
}

bool isSpecKind(ContextCacheDeploymentKind kind) noexcept
{
    return kind == ContextCacheDeploymentKind::kEAGLE;
}

bool hasAttention(ContextCacheDeploymentKind kind) noexcept
{
    return kind != ContextCacheDeploymentKind::kPureRecurrent;
}

bool shouldLogDegradation(uint64_t count) noexcept
{
    return count != 0U && (count & (count - 1U)) == 0U;
}

int32_t snapshotSlotCount(int64_t budgetBytes, size_t bytesPerSlot, char const* label)
{
    ELLM_CHECK(budgetBytes >= 0, std::string("Context cache ") + label + " budget must be non-negative");
    ELLM_CHECK(bytesPerSlot > 0, std::string("Context cache ") + label + " has an empty snapshot schema");
    uint64_t const slots = static_cast<uint64_t>(budgetBytes) / static_cast<uint64_t>(bytesPerSlot);
    ELLM_CHECK(slots <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        std::string("Context cache ") + label + " budget produces too many snapshot slots");
    return static_cast<int32_t>(slots);
}

ResourceDemand makeResourceCapacities(ContextCachePhysicalResources const& resources,
    ContextCacheDeploymentKind deploymentKind, ContextCacheConfig const& config)
{
    ELLM_CHECK(config.recurrentSnapshotPoolBytes >= 0 && config.partialKvSnapshotPoolBytes >= 0,
        "Context cache snapshot budgets must be non-negative");
    int32_t const draftPages = resources.draftPageTable == nullptr ? 0 : resources.draftPageTable->numPages();
    bool const hybrid = isHybridKind(deploymentKind);
    int32_t recurrentSlots{};
    int32_t partialKvSlots{};
    if (hybrid)
    {
        recurrentSlots = snapshotSlotCount(config.recurrentSnapshotPoolBytes,
            HybridSnapshotStorage::recurrentBytesPerSlot(resources.baseCache.getMambaCacheManager().getConfig()),
            "recurrent snapshot pool");
        if (deploymentKind == ContextCacheDeploymentKind::kHybrid)
        {
            partialKvSlots = snapshotSlotCount(config.partialKvSnapshotPoolBytes,
                HybridSnapshotStorage::partialKvBytesPerSlot(resources.baseCache.getKVCacheManager().getConfig()),
                "partial-KV snapshot pool");
        }
    }
    return ResourceDemand{hasAttention(deploymentKind) ? resources.basePageTable.numPages() : 0, draftPages,
        recurrentSlots, partialKvSlots};
}

bool hasBlockIdentity(BlockKeyExtras const& extras) noexcept
{
    return !extras.media.empty() || extras.adapter.has_value() || extras.positionDigest.has_value()
        || extras.customEmbeddingDigest.has_value() || extras.isolationDigest.has_value();
}

std::vector<BlockHash> hashRequestFullBlocks(
    int32_t const* tokens, size_t tokenCount, BlockKeyExtras const& extras, Hash128 const* perPositionMediaHash)
{
    if (!hasBlockIdentity(extras))
    {
        return hashFullBlocks(tokens, tokenCount, kTOKENS_PER_PAGE, {}, perPositionMediaHash);
    }

    size_t const pageSize = static_cast<size_t>(kTOKENS_PER_PAGE);
    size_t const blockCount = tokenCount / pageSize;
    std::vector<BlockHash> hashes;
    hashes.reserve(blockCount);
    BlockHash parent = kCHAIN_ROOT;
    for (size_t block = 0; block < blockCount; ++block)
    {
        Hash128 const* blockMediaHash
            = perPositionMediaHash != nullptr ? perPositionMediaHash + block * pageSize : nullptr;
        parent = hashBlock(parent, tokens + block * pageSize, pageSize, extras, blockMediaHash);
        hashes.push_back(parent);
    }
    return hashes;
}

BlockHash hashHybridCandidatePrefix(int32_t const* tokens, int32_t exactLength,
    std::vector<BlockHash> const& fullBlockHashes, BlockKeyExtras const& extras,
    Hash128 const* perPositionMediaHash = nullptr)
{
    ELLM_CHECK(exactLength > 0, "Hybrid context cache candidate length must be positive");
    size_t const pageSize = static_cast<size_t>(kTOKENS_PER_PAGE);
    size_t const tokenCount = static_cast<size_t>(exactLength);
    size_t const precedingFullBlocks = tokenCount / pageSize;
    size_t const partialTokenCount = tokenCount % pageSize;
    ELLM_CHECK(precedingFullBlocks <= fullBlockHashes.size(),
        "Hybrid context cache candidate exceeds the computed full-block hash chain");

    if (partialTokenCount == 0U)
    {
        ELLM_CHECK(precedingFullBlocks > 0U, "Hybrid context cache candidate has no terminal full block");
        return fullBlockHashes[precedingFullBlocks - 1U];
    }

    // The complete prefix is already represented by the chained full-block hash; extend it only with the partial tail.
    BlockHash const parent = precedingFullBlocks == 0U ? kCHAIN_ROOT : fullBlockHashes[precedingFullBlocks - 1U];
    Hash128 const* tailMediaHash
        = perPositionMediaHash != nullptr ? perPositionMediaHash + precedingFullBlocks * pageSize : nullptr;
    return hashBlock(parent, tokens + precedingFullBlocks * pageSize, partialTokenCount, extras, tailMediaHash);
}

BlockHash hashRequestExactPrefix(int32_t const* tokens, size_t tokenCount, BlockKeyExtras const& extras,
    Hash128 const* perPositionMediaHash = nullptr)
{
    std::vector<BlockHash> const fullBlockHashes
        = hashRequestFullBlocks(tokens, tokenCount, extras, perPositionMediaHash);
    return tokenCount % static_cast<size_t>(kTOKENS_PER_PAGE) == 0U
        ? (fullBlockHashes.empty() ? kCHAIN_ROOT : fullBlockHashes.back())
        : hashHybridCandidatePrefix(
              tokens, static_cast<int32_t>(tokenCount), fullBlockHashes, extras, perPositionMediaHash);
}

//! Trim trailing reused pages from a vanilla reuse plan when media tokens span across the reuse boundary.
//!
//! Without an encoder embedding cache, the runtime cannot slice cached ViT/audio embeddings to provide only the
//! suffix portion. If the reuse boundary falls inside a contiguous media run, the suffix would contain media
//! placeholders whose embedding indices don't start at zero in the full encoder output. This trims back until
//! the boundary no longer splits a media run.
void trimMediaBoundaryPages(ReusePlan& plan, Hash128 const* perPositionMediaHash, size_t tokenCount)
{
    if (perPositionMediaHash == nullptr || plan.basePageBindings.empty())
    {
        return;
    }

    int32_t const pageSize = kTOKENS_PER_PAGE;
    Hash128 const kZERO{};

    while (!plan.basePageBindings.empty())
    {
        int32_t const reuseLen = static_cast<int32_t>(plan.basePageBindings.size()) * pageSize;
        // Check if the token at the reuse boundary (first suffix token) is a media token.
        bool const suffixStartsWithMedia
            = static_cast<size_t>(reuseLen) < tokenCount && perPositionMediaHash[reuseLen] != kZERO;
        // Check if the last token of the last reused page is a media token.
        bool const lastReusedIsMedia = reuseLen > 0 && perPositionMediaHash[reuseLen - 1] != kZERO;

        if (suffixStartsWithMedia && lastReusedIsMedia)
        {
            // Media run spans across the reuse boundary — trim the last page.
            plan.basePageBindings.pop_back();
            plan.matchedBlockHashes.pop_back();
        }
        else
        {
            break;
        }
    }

    // Update plan fields to reflect trimmed state.
    int64_t const reusablePageCount = static_cast<int64_t>(plan.basePageBindings.size());
    int64_t const totalInputPages
        = (static_cast<int64_t>(tokenCount) + static_cast<int64_t>(pageSize) - 1) / static_cast<int64_t>(pageSize);
    plan.reuseTokenLength = static_cast<int32_t>(reusablePageCount * static_cast<int64_t>(pageSize));
    plan.matchedTokenLength = plan.reuseTokenLength;
    plan.demand.baseKvPages = static_cast<int32_t>(totalInputPages - reusablePageCount);
    if (plan.basePageBindings.empty())
    {
        plan.kind = ReusePlanKind::kNoReusablePrefix;
    }
    else
    {
        plan.kind = ReusePlanKind::kStandard;
    }
}

int32_t pageCountForStateLength(int32_t stateLength)
{
    ELLM_CHECK(stateLength >= 0, "Context cache state length must be non-negative");
    int64_t const pages
        = (static_cast<int64_t>(stateLength) + kTOKENS_PER_PAGE - 1) / static_cast<int64_t>(kTOKENS_PER_PAGE);
    ELLM_CHECK(pages <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "Context cache state requires too many pages");
    return static_cast<int32_t>(pages);
}

void validateCompactionMapping(std::vector<int32_t> const& oldToNew, int32_t oldBatchSize, int32_t newBatchSize)
{
    ELLM_CHECK(static_cast<int32_t>(oldToNew.size()) == oldBatchSize,
        "Context cache compaction mapping must describe every active slot");
    ELLM_CHECK(newBatchSize >= 0 && newBatchSize <= oldBatchSize,
        "Context cache compaction has an invalid destination batch size");

    int32_t nextDestination = 0;
    for (int32_t const destination : oldToNew)
    {
        ELLM_CHECK(destination == -1 || (destination >= 0 && destination < newBatchSize),
            "Context cache compaction contains an invalid destination");
        if (destination >= 0)
        {
            ELLM_CHECK(destination == nextDestination,
                "Context cache compaction must preserve survivor order for in-place state movement");
            ++nextDestination;
        }
    }
    ELLM_CHECK(nextDestination == newBatchSize, "Context cache compaction does not cover every destination");
}

class ActiveRequestRollback final
{
public:
    explicit ActiveRequestRollback(bool& active) noexcept
        : mActive(active)
    {
    }

    ~ActiveRequestRollback() noexcept
    {
        if (mArmed)
        {
            mActive = false;
        }
    }

    void dismiss() noexcept
    {
        mArmed = false;
    }

private:
    bool& mActive;
    bool mArmed{true};
};

} // namespace

struct ContextCacheCoordinator::RequestHandle::Impl
{
    struct RequestSlotToken
    {
        explicit RequestSlotToken(ContextCacheCoordinator& coordinator) noexcept
            : owner(&coordinator)
        {
        }

        RequestSlotToken(RequestSlotToken&& other) noexcept
            : owner(std::exchange(other.owner, nullptr))
        {
        }

        RequestSlotToken& operator=(RequestSlotToken&&) = delete;
        RequestSlotToken(RequestSlotToken const&) = delete;
        RequestSlotToken& operator=(RequestSlotToken const&) = delete;

        ~RequestSlotToken() noexcept
        {
            if (owner != nullptr)
            {
                owner->mRequestActive = false;
            }
        }

        ContextCacheCoordinator* owner{};
    };

    enum class Phase : uint8_t
    {
        kAdmitted,
        kExecuting,
        kFinishing,
    };

    struct StagedHybridPublication
    {
        HybridCheckpointKey checkpoint;
        HybridSnapshotReservation snapshots;
        PublicationPoint point{};
    };

    struct SequenceState
    {
        CacheRequestLease lease;
        std::vector<int32_t> tokenIds;
        BlockKeyExtras keyExtras;
        std::vector<Hash128> perPositionMediaHash;
        int32_t reuseTokenLength{};
        ContextCacheLookupPolicy lookupPolicy{ContextCacheLookupPolicy::kUseCache};
        ContextCacheCommitPolicy commitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};
        int32_t committedStateLength{};
        int32_t publishedFullBlockCount{};
        int32_t publishedExactLength{};
        int32_t commonStateLength{};
        std::optional<int32_t> frozenSpecPrefillLength;
        std::optional<StagedHybridPublication> stagedHybridPublication;
    };

    explicit Impl(ContextCacheCoordinator& coordinator, cudaStream_t requestStream) noexcept
        : requestSlot(coordinator)
        , owner(&coordinator)
        , stream(requestStream)
    {
    }

    //! Declared before sequences so its destructor clears the admission flag only after every lease is released.
    RequestSlotToken requestSlot;
    ContextCacheCoordinator* owner{};
    cudaStream_t stream{};
    ContextCacheExecutionMode executionMode{ContextCacheExecutionMode::kVanilla};
    Phase phase{Phase::kAdmitted};
    bool deviceWorkPending{};
    bool specAwaitingFirstCompletion{};
    std::vector<int32_t> pendingCompactionMapping;
    int32_t pendingCompactionBatchSize{-1};
    Tensor const* pendingDeviceBatchMapping{};
    std::vector<SequenceState> sequences;
};

struct ContextCacheCoordinator::AcquireSequenceResult
{
    std::optional<CacheRequestLease> lease;
    AcquireStatus status{AcquireStatus::kInsufficientCapacity};
    ReusePlan plan;
    bool forcedCold{};
};

ContextCacheCoordinator::RequestHandle::RequestHandle(std::unique_ptr<Impl> impl) noexcept
    : mImpl(std::move(impl))
{
}

ContextCacheCoordinator::RequestHandle::RequestHandle(RequestHandle&& other) noexcept = default;

ContextCacheCoordinator::RequestHandle::~RequestHandle() noexcept
{
    if (mImpl != nullptr)
    {
        ContextCacheCoordinator* const owner = mImpl->owner;
        owner->abandon(std::move(mImpl));
    }
}

bool ContextCacheCoordinator::RequestHandle::valid() const noexcept
{
    return mImpl != nullptr;
}

ContextCacheCoordinator::ContextCacheCoordinator(ContextCacheConfig const& config, DeploymentConfig const& deployment,
    ContextCacheDeploymentKind deploymentKind, ContextCachePhysicalResources resources, cudaStream_t stream,
    StreamSynchronizer synchronizer)
    : mDeploymentKind(deploymentKind)
    , mManager(kTOKENS_PER_PAGE, makeResourceCapacities(resources, mDeploymentKind, config), config.maxRecords)
    , mBaseCache(resources.baseCache)
    , mBasePageTable(resources.basePageTable)
    , mDraftCache(resources.draftCache)
    , mDraftPageTable(resources.draftPageTable)
    , mStream(stream)
    , mSynchronizer(std::move(synchronizer))
{
    ELLM_CHECK(config.enabled, "ContextCacheCoordinator requires an enabled ContextCacheConfig");
    ELLM_CHECK(mDeploymentKind == ContextCacheDeploymentKind::kVanilla || isHybridDeployment() || isSpecDeployment(),
        "This context-cache integration slice admits vanilla, recurrent, and EAGLE deployments");
    if (isSpecDeployment())
    {
        ELLM_CHECK(mDraftCache != nullptr && mDraftPageTable != nullptr && deployment.specConfig.has_value(),
            "EAGLE context reuse requires validated draft cache resources and speculative configuration");
        mSpecVerifySize = deployment.specConfig->verifySize;
        int64_t const draftWorkingTokens = static_cast<int64_t>(deployment.specConfig->draftingStep)
            * static_cast<int64_t>(deployment.specConfig->draftingTopK);
        ELLM_CHECK(mSpecVerifySize > 0 && draftWorkingTokens > 0
                && draftWorkingTokens <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            "EAGLE context reuse has invalid speculative working-set geometry");
        mSpecDraftWorkingTokens = static_cast<int32_t>(draftWorkingTokens);
    }
    else
    {
        ELLM_CHECK(mDraftCache == nullptr && mDraftPageTable == nullptr,
            "Non-speculative context-cache deployment cannot bind draft physical resources");
    }

    KVCacheManager const& baseKv = mBaseCache.getKVCacheManager();
    ELLM_CHECK(baseKv.numPages() == mBasePageTable.numPages(),
        "Context cache base pool and page table have different physical page counts");
    ELLM_CHECK(baseKv.getConfig().maxBatchSize == deployment.base.maxSupportedBatchSize,
        "Context cache base pool and deployment have different maximum batch sizes");
    if (deploymentHasAttention())
    {
        ELLM_CHECK(mBasePageTable.maxPagesPerSeq() == pagesPerSlot(baseKv.maxCapPadded()),
            "Context cache base page-table row does not match the engine KV capacity");
    }

    if (isSpecDeployment())
    {
        KVCacheManager const& draftKv = mDraftCache->getKVCacheManager();
        ELLM_CHECK(draftKv.numPages() == mDraftPageTable->numPages(),
            "Context cache draft pool and page table have different physical page counts");
        ELLM_CHECK(draftKv.getConfig().maxBatchSize == deployment.draft->maxSupportedBatchSize
                && mDraftPageTable->maxPagesPerSeq() == pagesPerSlot(draftKv.maxCapPadded()),
            "Context cache draft resources do not match the draft engine geometry");
    }

    if (isHybridDeployment())
    {
        int32_t const recurrentSlots = mManager.pools().capacity(ResourceType::kRecurrentSnapshot);
        int32_t const partialKvSlots = mManager.pools().capacity(ResourceType::kPartialKvSnapshot);
        ELLM_CHECK(recurrentSlots > 0, "Hybrid context reuse requires at least one recurrent snapshot slot");
        if (mDeploymentKind == ContextCacheDeploymentKind::kHybrid)
        {
            ELLM_CHECK(
                partialKvSlots > 0, "Hybrid attention context reuse requires at least one partial-KV snapshot slot");
        }
        mHybridSnapshots = std::make_unique<HybridSnapshotStorage>(mBaseCache, recurrentSlots, partialKvSlots);
    }

    if (!mSynchronizer)
    {
        mSynchronizer = [](cudaStream_t requestStream) { return cudaStreamSynchronize(requestStream); };
    }
    mHostReuseLengths = std::make_unique<Tensor>(Coords{deployment.base.maxSupportedBatchSize}, DeviceType::kCPU,
        nvinfer1::DataType::kINT32, "ContextCacheCoordinator::hostReuseLengths");
}

ContextCacheCoordinator::~ContextCacheCoordinator() noexcept
{
    if (mQuarantinedRequest != nullptr || mRequestActive)
    {
        LOG_ERROR(
            "ContextCacheCoordinator destroyed with unresolved request ownership; shutdown must prove "
            "quiescence before physical cache destruction.");
        std::terminate();
    }
}

ContextCacheCoordinator::AcquireSequenceResult ContextCacheCoordinator::acquireSequence(
    ContextCacheSequenceAdmission const& admission, ContextCacheExecutionMode executionMode,
    ContextCacheLookupPolicy lookupPolicy)
{
    ELLM_CHECK(!admission.tokenIds.empty(), "Context cache cannot admit an empty token sequence");
    ELLM_CHECK(admission.tokenIds.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "Context cache input contains too many tokens");
    if (deploymentHasAttention())
    {
        int32_t const inputPages = pageCountForStateLength(static_cast<int32_t>(admission.tokenIds.size()));
        ELLM_CHECK(inputPages <= mBasePageTable.maxPagesPerSeq(),
            "Context cache input exceeds the engine page-table capacity");
        ELLM_CHECK(
            executionMode != ContextCacheExecutionMode::kEAGLE || inputPages <= mDraftPageTable->maxPagesPerSeq(),
            "Context cache input exceeds the draft engine page-table capacity");
    }
    auto const planningStart = std::chrono::steady_clock::now();
    Hash128 const* mediaHashPtr
        = admission.perPositionMediaHash.empty() ? nullptr : admission.perPositionMediaHash.data();
    std::vector<BlockHash> const hashes = hashRequestFullBlocks(
        admission.tokenIds.data(), admission.tokenIds.size(), admission.keyExtras, mediaHashPtr);
    std::vector<HybridCheckpointCandidate> hybridCandidates;
    if (isHybridDeployment() && lookupPolicy == ContextCacheLookupPolicy::kUseCache)
    {
        std::vector<int32_t> const candidateLengths
            = mManager.hybridCandidateLengths(static_cast<int32_t>(admission.tokenIds.size()));
        hybridCandidates.reserve(candidateLengths.size());
        for (int32_t const candidateLength : candidateLengths)
        {
            hybridCandidates.push_back(HybridCheckpointCandidate{candidateLength,
                hashHybridCandidatePrefix(
                    admission.tokenIds.data(), candidateLength, hashes, admission.keyExtras, mediaHashPtr)});
        }
    }
    auto acquire = [&](ContextCacheLookupPolicy policy) {
        if (isHybridDeployment())
        {
            return mManager.acquireHybrid(hybridCandidates, hashes, static_cast<int32_t>(admission.tokenIds.size()),
                deploymentHasAttention(), policy);
        }
        if (executionMode == ContextCacheExecutionMode::kEAGLE)
        {
            ELLM_CHECK(isSpecDeployment(), "EAGLE cache planning requires an EAGLE deployment");
            return mManager.acquireSpec(hashes, static_cast<int32_t>(admission.tokenIds.size()), policy);
        }
        return mManager.acquireVanilla(hashes, static_cast<int32_t>(admission.tokenIds.size()), policy);
    };

    AcquireResult acquired = acquire(lookupPolicy);
    bool const cacheDerivedPlan
        = acquired.plan.reuseTokenLength > 0 || acquired.plan.kind == ReusePlanKind::kFullInputRewind;
    bool forcedCold = false;
    if (acquired.status == AcquireStatus::kInsufficientCapacity && cacheDerivedPlan
        && lookupPolicy == ContextCacheLookupPolicy::kUseCache)
    {
        acquired = acquire(ContextCacheLookupPolicy::kBypass);
        forcedCold = true;
    }

    // When media-aware hashing is active, trim trailing reused pages that would split a contiguous media run
    // across the reuse boundary. Without an encoder embedding cache, the runtime cannot supply correctly-offset
    // embeddings for partial media context in the suffix.
    if (!admission.perPositionMediaHash.empty() && acquired.plan.reuseTokenLength > 0)
    {
        trimMediaBoundaryPages(acquired.plan, admission.perPositionMediaHash.data(), admission.tokenIds.size());
    }

    auto const planningEnd = std::chrono::steady_clock::now();
    mMetrics.planningNanoseconds += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(planningEnd - planningStart).count());
    return AcquireSequenceResult{std::move(acquired.lease), acquired.status, std::move(acquired.plan), forcedCold};
}

ContextCacheCoordinator::BeginRequestResult ContextCacheCoordinator::beginRequest(
    ContextCacheBatchAdmission const& admission, cudaStream_t stream)
{
    ELLM_CHECK(stream == mStream, "Context cache request stream differs from the coordinator construction stream");
    ELLM_CHECK(admission.lookupPolicy == ContextCacheLookupPolicy::kUseCache
            || admission.lookupPolicy == ContextCacheLookupPolicy::kBypass,
        "Context cache admission has an invalid lookup policy");
    ELLM_CHECK(admission.commitPolicy == ContextCacheCommitPolicy::kIncludingGeneratedTokens
            || admission.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly,
        "Context cache admission has an invalid commit policy");
    ELLM_CHECK(admission.executionMode == ContextCacheExecutionMode::kVanilla
            || admission.executionMode == ContextCacheExecutionMode::kEAGLE,
        "Context cache admission has an invalid execution mode");
    ELLM_CHECK(admission.executionMode != ContextCacheExecutionMode::kEAGLE || isSpecDeployment(),
        "EAGLE context-cache execution requires an EAGLE deployment");
    if (mPoisoned)
    {
        return BeginRequestResult{ContextCacheCoordinatorStatus::kPoisoned, std::nullopt};
    }

    if (mRequestActive)
    {
        return BeginRequestResult{ContextCacheCoordinatorStatus::kRequestFailed, std::nullopt};
    }
    mRequestActive = true;

    ActiveRequestRollback activeRollback(mRequestActive);
    auto request = std::make_unique<RequestHandle::Impl>(*this, stream);
    activeRollback.dismiss();
    request->executionMode = admission.executionMode;
    int32_t maxBatchSize = mBaseCache.getKVCacheManager().getConfig().maxBatchSize;
    if (admission.executionMode == ContextCacheExecutionMode::kEAGLE)
    {
        maxBatchSize = std::min(maxBatchSize, mDraftCache->getKVCacheManager().getConfig().maxBatchSize);
    }
    ELLM_CHECK(!admission.sequences.empty() && admission.sequences.size() <= static_cast<size_t>(maxBatchSize),
        "Context cache admission batch size is outside the engine range");
    request->sequences.reserve(admission.sequences.size());
    std::vector<int32_t> prefillStarts;
    prefillStarts.reserve(admission.sequences.size());

    for (ContextCacheSequenceAdmission const& sequenceAdmission : admission.sequences)
    {
        AcquireSequenceResult acquired
            = acquireSequence(sequenceAdmission, admission.executionMode, admission.lookupPolicy);
        if (acquired.status != AcquireStatus::kAcquired || !acquired.lease.has_value())
        {
            return BeginRequestResult{ContextCacheCoordinatorStatus::kRequestFailed, std::nullopt};
        }

        RequestHandle::Impl::SequenceState sequence;
        sequence.lease = std::move(*acquired.lease);
        sequence.tokenIds = sequenceAdmission.tokenIds;
        sequence.keyExtras = sequenceAdmission.keyExtras;
        sequence.perPositionMediaHash = sequenceAdmission.perPositionMediaHash;
        sequence.reuseTokenLength = acquired.plan.reuseTokenLength;
        sequence.lookupPolicy = admission.lookupPolicy;
        sequence.commitPolicy = admission.commitPolicy;
        sequence.committedStateLength = acquired.plan.reuseTokenLength;
        sequence.commonStateLength = acquired.plan.reuseTokenLength;
        request->sequences.push_back(std::move(sequence));
        prefillStarts.push_back(acquired.plan.reuseTokenLength);
        ++mMetrics.admittedSequences;
        mMetrics.mediaAwareSequences += static_cast<uint64_t>(!sequenceAdmission.perPositionMediaHash.empty());
        mMetrics.matchedTokens += static_cast<uint64_t>(acquired.plan.matchedTokenLength);
        mMetrics.reusedTokens += static_cast<uint64_t>(acquired.plan.reuseTokenLength);
        mMetrics.hitSequences += static_cast<uint64_t>(acquired.plan.matchedTokenLength > 0);
        mMetrics.lookupBypassSequences += static_cast<uint64_t>(
            admission.lookupPolicy == ContextCacheLookupPolicy::kBypass || acquired.forcedCold);
        if (acquired.forcedCold)
        {
            ++mMetrics.forcedColdSequences;
            if (shouldLogDegradation(mMetrics.forcedColdSequences))
            {
                LOG_WARNING("Context cache forced-cold fallback count reached %llu",
                    static_cast<unsigned long long>(mMetrics.forcedColdSequences));
            }
        }
        switch (acquired.plan.kind)
        {
        case ReusePlanKind::kStandard: ++mMetrics.standardPlans; break;
        case ReusePlanKind::kNoReusablePrefix: ++mMetrics.noReusablePrefixPlans; break;
        case ReusePlanKind::kFullInputRewind: ++mMetrics.fullInputRewindPlans; break;
        }
        mMetrics.specFullPageReplays
            += static_cast<uint64_t>(acquired.plan.specReplayMode == SpecReplayMode::kFullPage);
    }

    AdmissionResult admitted{RequestHandle(std::move(request)), std::move(prefillStarts)};
    return BeginRequestResult{ContextCacheCoordinatorStatus::kOk, std::move(admitted)};
}

ContextCacheCoordinator::RequestHandle::Impl& ContextCacheCoordinator::checkedImpl(RequestHandle& request) const
{
    ELLM_CHECK(request.mImpl != nullptr && request.mImpl->owner == this,
        "Context cache request handle is invalid or belongs to another coordinator");
    return *request.mImpl;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::preparePrefill(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kAdmitted && !impl.deviceWorkPending,
        "Context cache prefill preparation requires a newly admitted request");

    std::vector<KVPageTableRowUpdate> rows;
    std::vector<KVPageTableRowUpdate> draftRows;
    if (deploymentHasAttention())
    {
        rows.reserve(impl.sequences.size());
    }
    if (isSpecRequest(impl))
    {
        draftRows.reserve(impl.sequences.size());
    }
    ELLM_CHECK(mHostReuseLengths->reshape({static_cast<int64_t>(impl.sequences.size())}),
        "Context cache reuse-length tensor reshape failed");
    int32_t* const reuseLengths = mHostReuseLengths->dataPointer<int32_t>();
    for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
    {
        auto const& sequence = impl.sequences[slot];
        if (deploymentHasAttention())
        {
            auto const& pages = sequence.lease.basePages();
            rows.push_back(KVPageTableRowUpdate{static_cast<int32_t>(slot), pages.empty() ? nullptr : pages.data(),
                static_cast<int32_t>(pages.size())});
        }
        if (isSpecRequest(impl))
        {
            auto const& pages = sequence.lease.draftPages();
            ELLM_CHECK(pages.size() == sequence.lease.basePages().size(),
                "EAGLE context-cache prefill requires equal base and draft page paths");
            draftRows.push_back(KVPageTableRowUpdate{static_cast<int32_t>(slot), pages.empty() ? nullptr : pages.data(),
                static_cast<int32_t>(pages.size())});
        }
        reuseLengths[slot] = sequence.reuseTokenLength;
    }

    impl.deviceWorkPending = true;
    if (deploymentHasAttention())
    {
        mBasePageTable.setRows(rows);
        mBasePageTable.upload(impl.stream);
    }
    if (isSpecRequest(impl))
    {
        mDraftPageTable->setRows(draftRows);
        mDraftPageTable->upload(impl.stream);
    }
    if (isHybridDeployment())
    {
        ELLM_CHECK(mHybridSnapshots != nullptr, "Hybrid context cache has no snapshot storage");
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            std::optional<int32_t> const recurrentSnapshot = sequence.lease.recurrentSnapshotSlot();
            if (!recurrentSnapshot.has_value())
            {
                mHybridSnapshots->zeroRecurrent(static_cast<int32_t>(slot), impl.stream);
                continue;
            }

            mHybridSnapshots->restoreRecurrent(*recurrentSnapshot, static_cast<int32_t>(slot), impl.stream);
            if (std::optional<int32_t> const partialSnapshot = sequence.lease.partialKvSnapshotSlot();
                partialSnapshot.has_value())
            {
                int32_t const validTokenCount = sequence.reuseTokenLength % kTOKENS_PER_PAGE;
                size_t const destinationIndex = static_cast<size_t>(sequence.reuseTokenLength / kTOKENS_PER_PAGE);
                ELLM_CHECK(validTokenCount > 0 && destinationIndex < sequence.lease.basePages().size(),
                    "Hybrid partial-KV restore has no private destination page");
                mHybridSnapshots->restorePartialKv(
                    *partialSnapshot, sequence.lease.basePages()[destinationIndex], validTokenCount, impl.stream);
            }
            ++mMetrics.hybridRestores;
        }
    }
    mBaseCache.resetForNewSequences(*mHostReuseLengths, impl.stream);
    if (isSpecRequest(impl))
    {
        mDraftCache->resetForNewSequences(*mHostReuseLengths, impl.stream);
    }
    impl.phase = RequestHandle::Impl::Phase::kExecuting;
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::enqueuePrefillCaptures(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && impl.deviceWorkPending,
        "Context cache prefill capture requires pending prefill work");
    if (!isHybridDeployment())
    {
        return ContextCacheCoordinatorStatus::kOk;
    }

    for (int32_t slot = 0; slot < static_cast<int32_t>(impl.sequences.size()); ++slot)
    {
        int32_t const exactLength = static_cast<int32_t>(impl.sequences[static_cast<size_t>(slot)].tokenIds.size());
        reserveHybridCapture(impl, slot, exactLength, PublicationPoint::kPrefillEnd);
    }
    enqueueHybridCaptures(impl);
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::applyAdvances(
    RequestHandle::Impl& request, std::vector<ContextCacheSequenceAdvance> const& advances)
{
    ELLM_CHECK(
        advances.size() == request.sequences.size(), "Context cache advances must describe every active sequence");
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        ContextCacheSequenceAdvance const& delta = advances[slot];
        auto& sequence = request.sequences[slot];
        ELLM_CHECK(delta.acceptedTokenCount >= 0, "Context cache accepted-token count must be non-negative");
        ELLM_CHECK(delta.acceptedTokenCount == 0 || delta.acceptedTokenIds != nullptr,
            "Context cache accepted-token delta has a null token pointer");
        size_t const acceptedCount = static_cast<size_t>(delta.acceptedTokenCount);
        ELLM_CHECK(sequence.tokenIds.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) - acceptedCount,
            "Context cache token history exceeds int32");
        size_t const updatedTokenCount = sequence.tokenIds.size() + acceptedCount;
        ELLM_CHECK(delta.committedStateLength >= sequence.committedStateLength
                && static_cast<size_t>(delta.committedStateLength) <= updatedTokenCount,
            "Context cache committed state length is not a monotonic materialized prefix");
    }

    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        ContextCacheSequenceAdvance const& delta = advances[slot];
        auto& sequence = request.sequences[slot];
        if (delta.acceptedTokenCount > 0)
        {
            sequence.tokenIds.insert(sequence.tokenIds.end(), delta.acceptedTokenIds,
                delta.acceptedTokenIds + static_cast<std::ptrdiff_t>(delta.acceptedTokenCount));
        }
        sequence.committedStateLength = delta.committedStateLength;
    }
    return ContextCacheCoordinatorStatus::kOk;
}

void ContextCacheCoordinator::reserveHybridCapture(
    RequestHandle::Impl& request, int32_t slot, int32_t exactLength, PublicationPoint point)
{
    ELLM_CHECK(isHybridDeployment() && mHybridSnapshots != nullptr,
        "Hybrid context capture requires a recurrent deployment and snapshot storage");
    ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(request.sequences.size()),
        "Hybrid context capture slot is outside the active batch");
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    ELLM_CHECK(
        !sequence.stagedHybridPublication.has_value(), "Hybrid context capture already has an unpublished snapshot");
    ELLM_CHECK(exactLength > 0 && static_cast<size_t>(exactLength) <= sequence.tokenIds.size(),
        "Hybrid context capture length is outside the logical token history");
    if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass || exactLength <= sequence.publishedExactLength
        || (point == PublicationPoint::kDecodeEnd
            && sequence.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly))
    {
        return;
    }

    bool const needsPartialKvSnapshot = deploymentHasAttention() && exactLength % kTOKENS_PER_PAGE != 0;
    std::optional<HybridSnapshotReservation> const reservation
        = mManager.reserveHybridSnapshots(sequence.lease, needsPartialKvSnapshot);
    if (!reservation.has_value())
    {
        ++mMetrics.hybridSnapshotPressureSkips;
        if (shouldLogDegradation(mMetrics.hybridSnapshotPressureSkips))
        {
            LOG_WARNING("Context cache hybrid snapshot-pressure skip count reached %llu",
                static_cast<unsigned long long>(mMetrics.hybridSnapshotPressureSkips));
        }
        return;
    }

    Hash128 const* exactMediaPtr
        = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
    HybridCheckpointKey const checkpoint{hashRequestExactPrefix(sequence.tokenIds.data(),
                                             static_cast<size_t>(exactLength), sequence.keyExtras, exactMediaPtr),
        exactLength};
    sequence.stagedHybridPublication = RequestHandle::Impl::StagedHybridPublication{checkpoint, *reservation, point};
}

void ContextCacheCoordinator::enqueueHybridCaptures(RequestHandle::Impl& request)
{
    ELLM_CHECK(isHybridDeployment() && mHybridSnapshots != nullptr,
        "Hybrid context capture requires a recurrent deployment and snapshot storage");
    for (size_t slot = 0; slot < request.sequences.size(); ++slot)
    {
        auto& sequence = request.sequences[slot];
        if (!sequence.stagedHybridPublication.has_value())
        {
            continue;
        }
        auto const& staged = *sequence.stagedHybridPublication;
        mHybridSnapshots->captureRecurrent(
            staged.snapshots.recurrentSnapshotSlot, static_cast<int32_t>(slot), request.stream);
        if (staged.snapshots.partialKvSnapshotSlot.has_value())
        {
            int32_t const validTokenCount = staged.checkpoint.exactLength % kTOKENS_PER_PAGE;
            size_t const sourceIndex = static_cast<size_t>(staged.checkpoint.exactLength / kTOKENS_PER_PAGE);
            ELLM_CHECK(validTokenCount > 0 && sourceIndex < sequence.lease.basePages().size(),
                "Hybrid partial-KV capture has no materialized source page");
            mHybridSnapshots->capturePartialKv(*staged.snapshots.partialKvSnapshotSlot,
                sequence.lease.basePages()[sourceIndex], validTokenCount, request.stream);
        }
    }
}

void ContextCacheCoordinator::publishReadyHybridEndpoints(RequestHandle::Impl& request)
{
    for (auto& sequence : request.sequences)
    {
        if (!sequence.stagedHybridPublication.has_value())
        {
            continue;
        }
        auto const staged = *sequence.stagedHybridPublication;
        ELLM_CHECK(staged.checkpoint.exactLength <= sequence.committedStateLength,
            "Hybrid context snapshot exceeds the ready committed boundary");
        size_t const fullTokenCount = static_cast<size_t>(staged.checkpoint.exactLength / kTOKENS_PER_PAGE)
            * static_cast<size_t>(kTOKENS_PER_PAGE);
        Hash128 const* seqMediaPtr
            = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
        std::vector<BlockHash> hashes
            = hashRequestFullBlocks(sequence.tokenIds.data(), fullTokenCount, sequence.keyExtras, seqMediaPtr);
        PublishResult const result = mManager.publishHybrid(
            sequence.lease, HybridPublishRequest{std::move(hashes), staged.checkpoint, staged.snapshots});
        recordPublication(result.status);
        mManager.retireHybridSnapshotReservation(sequence.lease, staged.snapshots);
        sequence.stagedHybridPublication.reset();
        sequence.publishedExactLength = staged.checkpoint.exactLength;
        sequence.publishedFullBlockCount = result.publishedBaseFullBlockCount;
    }
}

void ContextCacheCoordinator::publishReadyEndpoint(RequestHandle::Impl& request, int32_t slot, PublicationPoint point)
{
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass
        || (point == PublicationPoint::kDecodeEnd
            && sequence.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly))
    {
        return;
    }

    int32_t const publishBlocks = sequence.committedStateLength / kTOKENS_PER_PAGE;
    if (publishBlocks <= sequence.publishedFullBlockCount)
    {
        return;
    }

    size_t const tokenCount = static_cast<size_t>(publishBlocks) * static_cast<size_t>(kTOKENS_PER_PAGE);
    Hash128 const* pubMediaPtr = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
    std::vector<BlockHash> hashes
        = hashRequestFullBlocks(sequence.tokenIds.data(), tokenCount, sequence.keyExtras, pubMediaPtr);
    PublishResult const result
        = mManager.publish(sequence.lease, PublishRequest{std::move(hashes), sequence.committedStateLength});
    recordPublication(result.status);
    sequence.publishedFullBlockCount = result.publishedBaseFullBlockCount;
}

void ContextCacheCoordinator::publishSpecEndpoint(
    RequestHandle::Impl& request, int32_t slot, int32_t commonStateLength, PublicationPoint point)
{
    ELLM_CHECK(isSpecRequest(request), "Speculative endpoint publication requires an EAGLE request");
    auto& sequence = request.sequences[static_cast<size_t>(slot)];
    ELLM_CHECK(commonStateLength >= sequence.publishedFullBlockCount * kTOKENS_PER_PAGE
            && commonStateLength <= sequence.committedStateLength,
        "EAGLE common state is outside the ready base-state boundary");
    if (sequence.lookupPolicy == ContextCacheLookupPolicy::kBypass
        || (point == PublicationPoint::kDecodeEnd
            && sequence.commitPolicy == ContextCacheCommitPolicy::kPrefillStateOnly))
    {
        return;
    }

    int32_t const publishBlocks = commonStateLength / kTOKENS_PER_PAGE;
    if (publishBlocks <= sequence.publishedFullBlockCount)
    {
        return;
    }
    size_t const tokenCount = static_cast<size_t>(publishBlocks) * static_cast<size_t>(kTOKENS_PER_PAGE);
    Hash128 const* specMediaPtr
        = sequence.perPositionMediaHash.empty() ? nullptr : sequence.perPositionMediaHash.data();
    std::vector<BlockHash> hashes
        = hashRequestFullBlocks(sequence.tokenIds.data(), tokenCount, sequence.keyExtras, specMediaPtr);
    PublishResult const result = mManager.publish(sequence.lease, PublishRequest{std::move(hashes), commonStateLength});
    recordPublication(result.status);
    sequence.publishedFullBlockCount = result.publishedBaseFullBlockCount;
    ++mMetrics.specPairPublications;
}

void ContextCacheCoordinator::recordPublication(PublishStatus status) noexcept
{
    ++mMetrics.publicationAttempts;
    switch (status)
    {
    case PublishStatus::kPublished:
        ++mMetrics.committedPublications;
        ++mMetrics.publishedEndpoints;
        break;
    case PublishStatus::kExistingRecord:
        ++mMetrics.existingPublications;
        ++mMetrics.publishedEndpoints;
        break;
    }
}

void ContextCacheCoordinator::publishFrozenSpecPrefill(RequestHandle::Impl& request)
{
    ELLM_CHECK(isSpecRequest(request), "Frozen speculative publication requires an EAGLE request");
    for (int32_t slot = 0; slot < static_cast<int32_t>(request.sequences.size()); ++slot)
    {
        auto& sequence = request.sequences[static_cast<size_t>(slot)];
        if (!sequence.frozenSpecPrefillLength.has_value())
        {
            continue;
        }
        int32_t const readyLength = std::min(*sequence.frozenSpecPrefillLength, sequence.commonStateLength);
        publishSpecEndpoint(request, slot, readyLength, PublicationPoint::kPrefillEnd);
        sequence.frozenSpecPrefillLength.reset();
    }
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::terminalizeSpecInitialization(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    if (!impl.specAwaitingFirstCompletion)
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    ELLM_CHECK(isSpecRequest(impl) && impl.deviceWorkPending,
        "EAGLE initialization terminalization requires pending draft work");
    ContextCacheCoordinatorStatus const syncStatus = synchronizeRequest(request);
    if (syncStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return syncStatus;
    }
    publishFrozenSpecPrefill(impl);
    impl.specAwaitingFirstCompletion = false;
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::finalizePrefillPublication(RequestHandle& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const* commonStateLengths)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && impl.deviceWorkPending,
        "Context cache prefill finalization requires pending prefill work");
    ELLM_CHECK(
        advances.size() == impl.sequences.size(), "Context cache prefill advances must describe every active sequence");
    for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
    {
        auto const& sequence = impl.sequences[slot];
        auto const& delta = advances[slot];
        ELLM_CHECK(delta.acceptedTokenCount == 1
                && delta.committedStateLength == static_cast<int32_t>(sequence.tokenIds.size()),
            "Context cache prefill progress does not describe a complete input plus one lookahead");
    }
    if (isSpecRequest(impl))
    {
        ELLM_CHECK(commonStateLengths != nullptr && commonStateLengths->size() == impl.sequences.size(),
            "EAGLE prefill finalization requires one common state length per sequence");
    }
    else
    {
        ELLM_CHECK(commonStateLengths == nullptr || commonStateLengths->empty(),
            "Non-speculative prefill supplied speculative common-state lengths");
        impl.deviceWorkPending = false;
    }
    applyAdvances(impl, advances);
    if (isSpecRequest(impl))
    {
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            int32_t const commonLength = (*commonStateLengths)[slot];
            ELLM_CHECK(commonLength == sequence.committedStateLength,
                "EAGLE draft initialization did not materialize the complete prompt boundary");
            sequence.commonStateLength = commonLength;
            sequence.frozenSpecPrefillLength = sequence.committedStateLength;
        }
        impl.specAwaitingFirstCompletion = true;
    }
    else if (isHybridDeployment())
    {
        for (auto& sequence : impl.sequences)
        {
            mManager.releaseRestoredHybridSnapshots(sequence.lease);
        }
        publishReadyHybridEndpoints(impl);
    }
    else
    {
        for (int32_t slot = 0; slot < static_cast<int32_t>(impl.sequences.size()); ++slot)
        {
            publishReadyEndpoint(impl, slot, PublicationPoint::kPrefillEnd);
        }
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::prepareDecodeStep(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting
            && (!impl.deviceWorkPending || (isSpecRequest(impl) && impl.specAwaitingFirstCompletion)),
        "Context cache decode preparation requires terminal prior work");

    if (isSpecRequest(impl))
    {
        struct PageDemand
        {
            int32_t basePages{};
            int32_t draftPages{};
        };
        std::vector<PageDemand> demands;
        demands.reserve(impl.sequences.size());
        for (auto const& sequence : impl.sequences)
        {
            ELLM_CHECK(sequence.committedStateLength <= std::numeric_limits<int32_t>::max() - mSpecVerifySize
                    && sequence.committedStateLength <= std::numeric_limits<int32_t>::max() - mSpecDraftWorkingTokens,
                "EAGLE context-cache working-set length overflow before decode");
            int32_t const basePages = pageCountForStateLength(sequence.committedStateLength + mSpecVerifySize);
            int32_t const draftPages = pageCountForStateLength(sequence.committedStateLength + mSpecDraftWorkingTokens);
            if (basePages > mBasePageTable.maxPagesPerSeq() || draftPages > mDraftPageTable->maxPagesPerSeq())
            {
                impl.phase = RequestHandle::Impl::Phase::kFinishing;
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            demands.push_back(PageDemand{basePages, draftPages});
        }

        std::vector<KVPageTableRowUpdate> baseRows;
        std::vector<KVPageTableRowUpdate> draftRows;
        baseRows.reserve(impl.sequences.size());
        draftRows.reserve(impl.sequences.size());
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            int32_t const baseGrowth
                = demands[slot].basePages - static_cast<int32_t>(sequence.lease.basePages().size());
            int32_t const draftGrowth
                = demands[slot].draftPages - static_cast<int32_t>(sequence.lease.draftPages().size());
            if (!mManager.growSpecPages(sequence.lease, std::max(0, baseGrowth), std::max(0, draftGrowth)))
            {
                impl.phase = RequestHandle::Impl::Phase::kFinishing;
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            if (baseGrowth > 0)
            {
                auto const& basePages = sequence.lease.basePages();
                baseRows.push_back(KVPageTableRowUpdate{
                    static_cast<int32_t>(slot), basePages.data(), static_cast<int32_t>(basePages.size())});
            }
            if (draftGrowth > 0)
            {
                auto const& draftPages = sequence.lease.draftPages();
                draftRows.push_back(KVPageTableRowUpdate{
                    static_cast<int32_t>(slot), draftPages.data(), static_cast<int32_t>(draftPages.size())});
            }
        }
        impl.deviceWorkPending = true;
        if (!baseRows.empty())
        {
            mBasePageTable.setRows(baseRows);
            mBasePageTable.upload(impl.stream);
        }
        if (!draftRows.empty())
        {
            mDraftPageTable->setRows(draftRows);
            mDraftPageTable->upload(impl.stream);
        }
        return ContextCacheCoordinatorStatus::kOk;
    }

    for (auto const& sequence : impl.sequences)
    {
        ELLM_CHECK(sequence.committedStateLength < std::numeric_limits<int32_t>::max(),
            "Context cache sequence length overflow before decode");
    }
    if (deploymentHasAttention())
    {
        std::vector<KVPageTableRowUpdate> rows;
        rows.reserve(impl.sequences.size());
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto& sequence = impl.sequences[slot];
            int32_t const requiredPages = pageCountForStateLength(sequence.committedStateLength + 1);
            if (requiredPages > mBasePageTable.maxPagesPerSeq())
            {
                impl.phase = RequestHandle::Impl::Phase::kFinishing;
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            int32_t const currentPages = static_cast<int32_t>(sequence.lease.basePages().size());
            if (requiredPages > currentPages && !mManager.growBasePages(sequence.lease, requiredPages - currentPages))
            {
                impl.phase = RequestHandle::Impl::Phase::kFinishing;
                return ContextCacheCoordinatorStatus::kRequestFailed;
            }
            if (requiredPages > currentPages)
            {
                auto const& pages = sequence.lease.basePages();
                rows.push_back(
                    KVPageTableRowUpdate{static_cast<int32_t>(slot), pages.data(), static_cast<int32_t>(pages.size())});
            }
        }
        if (!rows.empty())
        {
            mBasePageTable.setRows(rows);
            mBasePageTable.upload(impl.stream);
        }
    }
    impl.deviceWorkPending = true;
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::completeDecodeStep(RequestHandle& request,
    std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
    std::vector<int32_t> const* commonStateLengths)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && impl.deviceWorkPending,
        "Context cache decode completion requires pending decode work");
    ELLM_CHECK(
        advances.size() == impl.sequences.size(), "Context cache decode advances must describe every active sequence");
    if (isSpecRequest(impl))
    {
        ELLM_CHECK(commonStateLengths != nullptr && commonStateLengths->size() == impl.sequences.size(),
            "EAGLE decode completion requires one common state length per sequence");
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto const& sequence = impl.sequences[slot];
            auto const& delta = advances[slot];
            int32_t const commonLength = (*commonStateLengths)[slot];
            int64_t const expectedCommittedStateLength
                = static_cast<int64_t>(sequence.committedStateLength) + static_cast<int64_t>(delta.acceptedTokenCount);
            int64_t const expectedTokenCount = static_cast<int64_t>(sequence.committedStateLength) + 1;
            ELLM_CHECK(delta.acceptedTokenCount > 0
                    && expectedCommittedStateLength <= std::numeric_limits<int32_t>::max()
                    && delta.committedStateLength == expectedCommittedStateLength && expectedTokenCount >= 0
                    && sequence.tokenIds.size() == static_cast<size_t>(expectedTokenCount),
                "EAGLE decode progress violates the committed-plus-lookahead invariant");
            ELLM_CHECK(commonLength >= sequence.commonStateLength && commonLength <= delta.committedStateLength,
                "EAGLE common state is not a monotonic base-state prefix");
        }
    }
    else
    {
        ELLM_CHECK(commonStateLengths == nullptr || commonStateLengths->empty(),
            "Non-speculative decode supplied speculative common-state lengths");
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            auto const& sequence = impl.sequences[slot];
            auto const& delta = advances[slot];
            int64_t const expectedStateLength = static_cast<int64_t>(sequence.committedStateLength) + 1;
            ELLM_CHECK(delta.acceptedTokenCount == 1 && expectedStateLength <= std::numeric_limits<int32_t>::max()
                    && delta.committedStateLength == expectedStateLength
                    && sequence.tokenIds.size() == static_cast<size_t>(expectedStateLength),
                "Context cache vanilla decode progress violates the committed-plus-lookahead invariant");
        }
    }
    applyAdvances(impl, advances);
    if (isSpecRequest(impl))
    {
        for (size_t slot = 0; slot < impl.sequences.size(); ++slot)
        {
            impl.sequences[slot].commonStateLength = (*commonStateLengths)[slot];
        }
        // EAGLE verification already performs the round synchronization. It also proves the ordered draft
        // initialization and any page-table uploads that preceded the first verification.
        impl.deviceWorkPending = false;
        impl.specAwaitingFirstCompletion = false;
        publishFrozenSpecPrefill(impl);
    }
    std::vector<uint8_t> published(impl.sequences.size(), 0U);
    for (int32_t const slot : publishableCompletedSlots)
    {
        ELLM_CHECK(slot >= 0 && slot < static_cast<int32_t>(impl.sequences.size()),
            "Context cache completed slot is outside the active batch");
        ELLM_CHECK(published[static_cast<size_t>(slot)] == 0U, "Context cache completed slot appears more than once");
        published[static_cast<size_t>(slot)] = 1U;
        if (isHybridDeployment())
        {
            int32_t const exactLength = impl.sequences[static_cast<size_t>(slot)].committedStateLength;
            reserveHybridCapture(impl, slot, exactLength, PublicationPoint::kDecodeEnd);
        }
        else if (isSpecRequest(impl))
        {
            auto const& sequence = impl.sequences[static_cast<size_t>(slot)];
            publishSpecEndpoint(impl, slot, sequence.commonStateLength, PublicationPoint::kDecodeEnd);
        }
    }

    if (isSpecRequest(impl))
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    if (isHybridDeployment())
    {
        bool const hasCaptures = std::any_of(impl.sequences.begin(), impl.sequences.end(),
            [](auto const& sequence) { return sequence.stagedHybridPublication.has_value(); });
        if (hasCaptures)
        {
            enqueueHybridCaptures(impl);
            ++mMetrics.hybridCaptureSynchronizations;
            ContextCacheCoordinatorStatus const syncStatus = synchronizeRequest(request);
            if (syncStatus != ContextCacheCoordinatorStatus::kOk)
            {
                return syncStatus;
            }
            publishReadyHybridEndpoints(impl);
        }
        else
        {
            impl.deviceWorkPending = false;
        }
    }
    else
    {
        impl.deviceWorkPending = false;
        for (int32_t const slot : publishableCompletedSlots)
        {
            publishReadyEndpoint(impl, slot, PublicationPoint::kDecodeEnd);
        }
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::beginBatchCompaction(
    RequestHandle& request, std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    if (isSpecRequest(impl))
    {
        ContextCacheCoordinatorStatus const terminalStatus = terminalizeSpecInitialization(request);
        if (terminalStatus != ContextCacheCoordinatorStatus::kOk)
        {
            return terminalStatus;
        }
    }
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && !impl.deviceWorkPending,
        "Context cache compaction preparation requires terminal model work");
    int32_t const oldBatchSize = static_cast<int32_t>(impl.sequences.size());
    validateCompactionMapping(oldToNew, oldBatchSize, newBatchSize);

    impl.deviceWorkPending = true;
    impl.pendingCompactionMapping = oldToNew;
    impl.pendingCompactionBatchSize = newBatchSize;
    impl.pendingDeviceBatchMapping = &deviceBatchMapping;
    ELLM_CHECK(deviceBatchMapping.reshape({oldBatchSize}), "Context cache batch-mapping tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(deviceBatchMapping.rawPointer(), oldToNew.data(),
        static_cast<size_t>(oldBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, impl.stream));
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::synchronizeRequest(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    if (!impl.deviceWorkPending)
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    cudaError_t const status = mSynchronizer(impl.stream);
    if (status != cudaSuccess)
    {
        quarantine(request);
        return ContextCacheCoordinatorStatus::kPoisoned;
    }
    impl.deviceWorkPending = false;
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::compactBatch(RequestHandle& request)
{
    RequestHandle::Impl& impl = checkedImpl(request);
    ELLM_CHECK(impl.phase == RequestHandle::Impl::Phase::kExecuting && impl.deviceWorkPending,
        "Context cache compaction requires prepared pending work");
    ELLM_CHECK(impl.pendingCompactionBatchSize >= 0 && impl.pendingDeviceBatchMapping != nullptr,
        "Context cache compaction is missing its authoritative mapping");
    std::vector<int32_t> const& oldToNew = impl.pendingCompactionMapping;
    int32_t const newBatchSize = impl.pendingCompactionBatchSize;
    Tensor const& deviceBatchMapping = *impl.pendingDeviceBatchMapping;
    int32_t const oldBatchSize = static_cast<int32_t>(impl.sequences.size());

    ELLM_CHECK(std::none_of(impl.sequences.begin(), impl.sequences.end(),
                   [](auto const& sequence) {
                       return sequence.stagedHybridPublication.has_value()
                           || sequence.frozenSpecPrefillLength.has_value();
                   }),
        "Context cache cannot compact a batch with an unpublished staged endpoint");

    if (deploymentHasAttention())
    {
        mBasePageTable.compactRows(oldToNew, newBatchSize);
        mBasePageTable.upload(impl.stream);
    }
    if (isSpecRequest(impl))
    {
        mDraftPageTable->compactRows(oldToNew, newBatchSize);
        mDraftPageTable->upload(impl.stream);
    }
    mBaseCache.compactBatchSlotState(deviceBatchMapping, oldBatchSize, newBatchSize, impl.stream);
    if (isSpecRequest(impl))
    {
        mDraftCache->compactBatchSlotState(deviceBatchMapping, oldBatchSize, newBatchSize, impl.stream);
    }
    ContextCacheCoordinatorStatus const syncStatus = synchronizeRequest(request);
    if (syncStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return syncStatus;
    }

    std::vector<RequestHandle::Impl::SequenceState> survivors(static_cast<size_t>(newBatchSize));
    for (int32_t oldSlot = 0; oldSlot < oldBatchSize; ++oldSlot)
    {
        int32_t const newSlot = oldToNew[static_cast<size_t>(oldSlot)];
        if (newSlot >= 0)
        {
            survivors[static_cast<size_t>(newSlot)] = std::move(impl.sequences[static_cast<size_t>(oldSlot)]);
        }
    }
    impl.sequences = std::move(survivors);
    impl.pendingCompactionMapping.clear();
    impl.pendingCompactionBatchSize = -1;
    impl.pendingDeviceBatchMapping = nullptr;
    mBaseCache.setActiveBatchSize(newBatchSize);
    if (isSpecRequest(impl))
    {
        mDraftCache->setActiveBatchSize(newBatchSize);
    }
    if (newBatchSize == 0)
    {
        impl.phase = RequestHandle::Impl::Phase::kFinishing;
    }
    return ContextCacheCoordinatorStatus::kOk;
}

void ContextCacheCoordinator::quarantine(RequestHandle& request) noexcept
{
    mPoisoned = true;
    if (mQuarantinedRequest != nullptr)
    {
        std::terminate();
    }
    mQuarantinedRequest = std::move(request.mImpl);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::finish(RequestHandle& request)
{
    if (!request.valid())
    {
        return ContextCacheCoordinatorStatus::kOk;
    }
    if (isSpecRequest(checkedImpl(request)))
    {
        ContextCacheCoordinatorStatus const terminalStatus = terminalizeSpecInitialization(request);
        if (terminalStatus != ContextCacheCoordinatorStatus::kOk)
        {
            return terminalStatus;
        }
    }
    ContextCacheCoordinatorStatus const syncStatus = synchronizeRequest(request);
    if (syncStatus != ContextCacheCoordinatorStatus::kOk)
    {
        return syncStatus;
    }

    request.mImpl.reset();
    return ContextCacheCoordinatorStatus::kOk;
}

void ContextCacheCoordinator::abandon(std::unique_ptr<RequestHandle::Impl> request) noexcept
{
    if (request == nullptr)
    {
        return;
    }
    if (!request->deviceWorkPending)
    {
        request.reset();
        return;
    }

    cudaError_t status{cudaErrorUnknown};
    try
    {
        status = mSynchronizer(request->stream);
    }
    catch (...)
    {
        status = cudaErrorUnknown;
    }
    if (status == cudaSuccess)
    {
        request->deviceWorkPending = false;
        request.reset();
        return;
    }

    mPoisoned = true;
    if (mQuarantinedRequest != nullptr)
    {
        std::terminate();
    }
    mQuarantinedRequest = std::move(request);
}

ContextCacheCoordinatorStatus ContextCacheCoordinator::shutdown() noexcept
{
    if (mQuarantinedRequest != nullptr)
    {
        cudaError_t status{cudaErrorUnknown};
        try
        {
            status = mSynchronizer(mQuarantinedRequest->stream);
        }
        catch (...)
        {
            status = cudaErrorUnknown;
        }
        if (status != cudaSuccess)
        {
            return ContextCacheCoordinatorStatus::kPoisoned;
        }
        mQuarantinedRequest->deviceWorkPending = false;
        mQuarantinedRequest.reset();
    }
    if (mRequestActive)
    {
        return ContextCacheCoordinatorStatus::kRequestFailed;
    }
    return ContextCacheCoordinatorStatus::kOk;
}

ContextCacheMetrics ContextCacheCoordinator::metrics() const noexcept
{
    ContextCacheMetrics result = mMetrics;
    ResourcePools const& pools = mManager.pools();
    result.currentRecords = static_cast<uint64_t>(mManager.records().size());
    result.baseKvPages = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kBaseKvPage), pools.capacity(ResourceType::kBaseKvPage)};
    result.draftKvPages = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kDraftKvPage), pools.capacity(ResourceType::kDraftKvPage)};
    result.recurrentSnapshots = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kRecurrentSnapshot), pools.capacity(ResourceType::kRecurrentSnapshot)};
    result.partialKvSnapshots = ContextCachePoolMetrics{
        pools.freeCount(ResourceType::kPartialKvSnapshot), pools.capacity(ResourceType::kPartialKvSnapshot)};
    ContextCacheManagerMetrics const& managerMetrics = mManager.metrics();
    result.evictedRecords = managerMetrics.evictedRecords;
    result.reclaimedBaseKvPages = managerMetrics.reclaimedBaseKvPages;
    result.reclaimedDraftKvPages = managerMetrics.reclaimedDraftKvPages;
    result.reclaimedRecurrentSnapshots = managerMetrics.reclaimedRecurrentSnapshots;
    result.reclaimedPartialKvSnapshots = managerMetrics.reclaimedPartialKvSnapshots;
    return result;
}

bool ContextCacheCoordinator::isHybridDeployment() const noexcept
{
    return isHybridKind(mDeploymentKind);
}

bool ContextCacheCoordinator::isSpecDeployment() const noexcept
{
    return isSpecKind(mDeploymentKind);
}

bool ContextCacheCoordinator::isSpecRequest(RequestHandle::Impl const& request) const noexcept
{
    return request.executionMode == ContextCacheExecutionMode::kEAGLE;
}

bool ContextCacheCoordinator::deploymentHasAttention() const noexcept
{
    return hasAttention(mDeploymentKind);
}

ContextCacheManager const& ContextCacheCoordinator::manager() const noexcept
{
    return mManager;
}

} // namespace rt
} // namespace trt_edgellm
