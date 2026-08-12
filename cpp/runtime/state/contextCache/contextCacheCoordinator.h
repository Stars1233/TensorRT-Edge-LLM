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

#include "runtime/state/contextCache/contextCacheConfig.h"
#include "runtime/state/contextCache/contextCacheDeployment.h"
#include "runtime/state/contextCache/contextCacheManager.h"
#include "runtime/state/contextCache/contextCacheMetrics.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class HybridCacheManager;
class HybridSnapshotStorage;
class KVPageTable;
class Tensor;

//! Validated physical resources borrowed from SharedResources for the coordinator lifetime.
struct ContextCachePhysicalResources
{
    HybridCacheManager& baseCache;
    KVPageTable& basePageTable;
    HybridCacheManager* draftCache{};
    KVPageTable* draftPageTable{};
};

//! One complete logical input admitted to the cache before its executable suffix is derived.
struct ContextCacheSequenceAdmission
{
    std::vector<int32_t> tokenIds;
    //! Request-wide non-token identity; LoRA/isolation identity is constant for the sequence.
    BlockKeyExtras keyExtras;
    //! Per-position media content hash. Empty means text-only. When non-empty, must have tokenIds.size() entries.
    //! A non-zero Hash128 at position i causes the block hash to consume that 128-bit digest instead of the token ID.
    std::vector<Hash128> perPositionMediaHash;
};

//! Decoder mode selected before cache lookup. A deployment with EAGLE engines may execute either mode per request;
//! vanilla may reuse the base side of a paired record, while EAGLE requires a paired base/draft hit.
enum class ContextCacheExecutionMode : uint8_t
{
    kVanilla,
    kEAGLE,
};

//! One serialized runtime request. Bypass still uses managed private pages but neither looks up nor publishes state.
struct ContextCacheBatchAdmission
{
    std::vector<ContextCacheSequenceAdmission> sequences;
    ContextCacheExecutionMode executionMode{ContextCacheExecutionMode::kVanilla};
    ContextCacheLookupPolicy lookupPolicy{ContextCacheLookupPolicy::kUseCache};
    ContextCacheCommitPolicy commitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};
};

//! Host-visible sequence advance observed after an existing stream synchronization.
struct ContextCacheSequenceAdvance
{
    int32_t const* acceptedTokenIds{};
    int32_t acceptedTokenCount{};
    //! Greatest logical token boundary whose model state is materialized in the bound cache.
    int32_t committedStateLength{};
};

enum class ContextCacheCoordinatorStatus : uint8_t
{
    kOk,
    kRequestFailed,
    kPoisoned,
};

//! Owns the complete host/device lifecycle around the CUDA-free ContextCacheManager.
//!
//! The coordinator has no worker and is deliberately single-request-at-a-time under the runtime's serialized request
//! contract. Normal publication occurs only at host-visible completion points already present in the runtime. An
//! abnormal exit drains the bound stream before releasing active page references; a failed drain quarantines the
//! entire request and poisons the coordinator until runtime-owned shutdown can establish quiescence.
class ContextCacheCoordinator final
{
public:
    using StreamSynchronizer = std::function<cudaError_t(cudaStream_t)>;

    class RequestHandle final
    {
    public:
        RequestHandle(RequestHandle&& other) noexcept;
        RequestHandle& operator=(RequestHandle&& other) = delete;
        RequestHandle(RequestHandle const&) = delete;
        RequestHandle& operator=(RequestHandle const&) = delete;
        ~RequestHandle() noexcept;

        bool valid() const noexcept;

    private:
        friend class ContextCacheCoordinator;
        struct Impl;

        explicit RequestHandle(std::unique_ptr<Impl> impl) noexcept;
        std::unique_ptr<Impl> mImpl;
    };

    struct AdmissionResult
    {
        RequestHandle request;
        //! Per-sequence logical token offset at which runtime prefill begins.
        std::vector<int32_t> prefillStarts;
    };

    struct BeginRequestResult
    {
        ContextCacheCoordinatorStatus status{ContextCacheCoordinatorStatus::kRequestFailed};
        std::optional<AdmissionResult> admission;
    };

    ContextCacheCoordinator(ContextCacheConfig const& config, DeploymentConfig const& deployment,
        ContextCacheDeploymentKind deploymentKind, ContextCachePhysicalResources resources, cudaStream_t stream,
        StreamSynchronizer synchronizer = {});
    ~ContextCacheCoordinator() noexcept;
    ContextCacheCoordinator(ContextCacheCoordinator const&) = delete;
    ContextCacheCoordinator& operator=(ContextCacheCoordinator const&) = delete;

    BeginRequestResult beginRequest(ContextCacheBatchAdmission const& admission, cudaStream_t stream);

    //! Bind every admitted row and reset logical cache lengths to the selected reuse boundaries.
    ContextCacheCoordinatorStatus preparePrefill(RequestHandle& request);
    //! Reserve and enqueue hybrid prefill-end snapshots before the runtime's existing prefill synchronization.
    ContextCacheCoordinatorStatus enqueuePrefillCaptures(RequestHandle& request);
    //! Apply the post-prefill sequence advance and publish every ready full-block endpoint.
    //! For EAGLE, commonStateLengths caps publication at the prefix materialized by both base and draft state.
    ContextCacheCoordinatorStatus finalizePrefillPublication(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances,
        std::vector<int32_t> const* commonStateLengths = nullptr);
    //! Grow and upload every row needed for the next decode working set before model execution.
    ContextCacheCoordinatorStatus prepareDecodeStep(RequestHandle& request);
    //! Apply the post-decode sequence advance after the decoder's existing synchronization.
    //! For EAGLE, commonStateLengths excludes any unmaterialized accepted suffix and speculative lookahead.
    ContextCacheCoordinatorStatus completeDecodeStep(RequestHandle& request,
        std::vector<ContextCacheSequenceAdvance> const& advances, std::vector<int32_t> const& publishableCompletedSlots,
        std::vector<int32_t> const* commonStateLengths = nullptr);
    //! Validate and upload the one authoritative old-to-new mapping before any old-slot compaction work.
    ContextCacheCoordinatorStatus beginBatchCompaction(
        RequestHandle& request, std::vector<int32_t> const& oldToNew, int32_t newBatchSize, Tensor& deviceBatchMapping);
    //! Compact slot-addressed state/page-table rows, retire leases, and consume the existing eviction sync.
    ContextCacheCoordinatorStatus compactBatch(RequestHandle& request);
    //! Consume a normally completed request. This is idempotent for an already-empty handle.
    ContextCacheCoordinatorStatus finish(RequestHandle& request);
    //! Drain quarantined ownership before physical cache resources are destroyed.
    ContextCacheCoordinatorStatus shutdown() noexcept;

    ContextCacheMetrics metrics() const noexcept;
    ContextCacheManager const& manager() const noexcept;

private:
    enum class PublicationPoint : uint8_t
    {
        kPrefillEnd,
        kDecodeEnd,
    };

    struct AcquireSequenceResult;

    AcquireSequenceResult acquireSequence(ContextCacheSequenceAdmission const& admission,
        ContextCacheExecutionMode executionMode, ContextCacheLookupPolicy lookupPolicy);
    ContextCacheCoordinatorStatus applyAdvances(
        RequestHandle::Impl& request, std::vector<ContextCacheSequenceAdvance> const& advances);
    void publishReadyEndpoint(RequestHandle::Impl& request, int32_t slot, PublicationPoint point);
    void reserveHybridCapture(RequestHandle::Impl& request, int32_t slot, int32_t exactLength, PublicationPoint point);
    void enqueueHybridCaptures(RequestHandle::Impl& request);
    void publishReadyHybridEndpoints(RequestHandle::Impl& request);
    void publishSpecEndpoint(
        RequestHandle::Impl& request, int32_t slot, int32_t commonStateLength, PublicationPoint point);
    void recordPublication(PublishStatus status) noexcept;
    void publishFrozenSpecPrefill(RequestHandle::Impl& request);
    ContextCacheCoordinatorStatus terminalizeSpecInitialization(RequestHandle& request);
    RequestHandle::Impl& checkedImpl(RequestHandle& request) const;
    bool isHybridDeployment() const noexcept;
    bool isSpecDeployment() const noexcept;
    bool isSpecRequest(RequestHandle::Impl const& request) const noexcept;
    bool deploymentHasAttention() const noexcept;
    ContextCacheCoordinatorStatus synchronizeRequest(RequestHandle& request);
    void abandon(std::unique_ptr<RequestHandle::Impl> request) noexcept;
    void quarantine(RequestHandle& request) noexcept;

    ContextCacheDeploymentKind mDeploymentKind{};
    ContextCacheManager mManager;
    std::unique_ptr<HybridSnapshotStorage> mHybridSnapshots;
    HybridCacheManager& mBaseCache;
    KVPageTable& mBasePageTable;
    HybridCacheManager* mDraftCache{};
    KVPageTable* mDraftPageTable{};
    int32_t mSpecVerifySize{};
    int32_t mSpecDraftWorkingTokens{};
    cudaStream_t mStream{};
    StreamSynchronizer mSynchronizer;
    ContextCacheMetrics mMetrics;
    std::unique_ptr<Tensor> mHostReuseLengths;
    bool mRequestActive{};
    bool mPoisoned{};
    //! Declared after mManager so quarantined leases are destroyed first during normal shutdown.
    std::unique_ptr<RequestHandle::Impl> mQuarantinedRequest;
};

} // namespace rt
} // namespace trt_edgellm
