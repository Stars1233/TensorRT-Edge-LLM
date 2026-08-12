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

#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/contextCache/contextCacheTypes.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Preallocated device storage for exact hybrid checkpoints.
//!
//! ResourcePools owns snapshot slot IDs and refcounts. This class owns only the corresponding device slabs and
//! performs stream-ordered D2D copies between one immutable snapshot slot and one live batch/page slot. Deployment
//! compatibility is fixed by the owning runtime; this physical storage has no lookup or schema policy.
class HybridSnapshotStorage
{
public:
    HybridSnapshotStorage(HybridCacheManager& cacheManager, int32_t recurrentSlotCount, int32_t partialKvSlotCount);
    HybridSnapshotStorage(HybridSnapshotStorage const&) = delete;
    HybridSnapshotStorage& operator=(HybridSnapshotStorage const&) = delete;

    static size_t recurrentBytesPerSlot(MambaCacheManager::Config const& config);
    static size_t partialKvBytesPerSlot(KVCacheManager::Config const& config);

    void zeroRecurrent(int32_t batchSlot, cudaStream_t stream);
    void captureRecurrent(int32_t snapshotSlot, int32_t batchSlot, cudaStream_t stream);
    void restoreRecurrent(int32_t snapshotSlot, int32_t batchSlot, cudaStream_t stream);
    void capturePartialKv(int32_t snapshotSlot, PageId sourcePage, int32_t validTokenCount, cudaStream_t stream);
    void restorePartialKv(int32_t snapshotSlot, PageId destinationPage, int32_t validTokenCount, cudaStream_t stream);

    int32_t recurrentSlotCount() const noexcept;
    int32_t partialKvSlotCount() const noexcept;

private:
    void validateBatchSlot(int32_t batchSlot) const;
    void validateRecurrentSlots(int32_t snapshotSlot, int32_t batchSlot) const;
    void validatePartialSlots(int32_t snapshotSlot, PageId page, int32_t validTokenCount) const;

    HybridCacheManager& mCacheManager;
    int32_t mRecurrentSlotCount{};
    int32_t mPartialKvSlotCount{};
    std::vector<Tensor> mRecurrentSnapshots;
    std::vector<Tensor> mConvSnapshots;
    std::vector<Tensor> mPartialKvSnapshots;
};

} // namespace rt
} // namespace trt_edgellm
