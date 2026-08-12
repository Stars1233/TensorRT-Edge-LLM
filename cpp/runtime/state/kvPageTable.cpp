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

#include "runtime/state/kvPageTable.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{

namespace
{
//! V id of a K id: the sentinel maps to itself, a live id shifts by numPages.
int32_t deriveV(int32_t k, int32_t numPages)
{
    return k == kUNUSED_PAGE_ENTRY ? kUNUSED_PAGE_ENTRY : k + numPages;
}
} // namespace

KVPageTable::KVPageTable(int32_t maxBatch, int32_t maxPagesPerSeq, int32_t numPages)
    : mMaxBatch(maxBatch)
    , mMaxPagesPerSeq(maxPagesPerSeq)
    , mNumPages(numPages)
{
    check::check(maxBatch > 0, "KVPageTable: maxBatch must be positive.");
    check::check(maxPagesPerSeq > 0, "KVPageTable: maxPagesPerSeq must be positive.");
    check::check(numPages > 0, "KVPageTable: numPages must be positive.");
    check::check(numPages <= kMAX_KV_POOL_PAGES,
        "KVPageTable: numPages exceeds the largest pool whose derived V page ids fit int32.");

    mHost.assign(static_cast<size_t>(maxBatch) * 2 * maxPagesPerSeq, kUNUSED_PAGE_ENTRY);
    mHostScratch.resize(mHost.size(), kUNUSED_PAGE_ENTRY);
    // Device storage is uninitialized until the first upload, so every row initially needs a copy.
    mDirtyRows.resize(static_cast<size_t>(maxBatch), 1U);
    mDevice = rt::Tensor(
        Coords{maxBatch, 2, maxPagesPerSeq}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "KVPageTable::kernelView");
    mUploadStaging = rt::Tensor(Coords{maxBatch, 2, maxPagesPerSeq}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
        "KVPageTable::uploadStaging");
    CUDA_CHECK(cudaEventCreateWithFlags(&mUploadComplete, cudaEventDisableTiming));
}

KVPageTable::~KVPageTable() noexcept
{
    if (mUploadComplete != nullptr)
    {
        if (mUploadPending)
        {
            static_cast<void>(cudaEventSynchronize(mUploadComplete));
        }
        static_cast<void>(cudaEventDestroy(mUploadComplete));
    }
}

void KVPageTable::setIdentity()
{
    for (int32_t b = 0; b < mMaxBatch; ++b)
    {
        int32_t* kRow = mutableHostRow(b);
        int32_t* vRow = kRow + mMaxPagesPerSeq;
        bool changed = false;
        for (int32_t j = 0; j < mMaxPagesPerSeq; ++j)
        {
            int32_t const k = b * mMaxPagesPerSeq + j;
            changed = changed || kRow[j] != k || vRow[j] != deriveV(k, mNumPages);
            kRow[j] = k;
            vRow[j] = deriveV(k, mNumPages);
        }
        if (changed)
        {
            mDirtyRows[static_cast<size_t>(b)] = 1U;
        }
    }
    mIsIdentity = true;
}

void KVPageTable::setRow(int32_t slot, int32_t const* kPageIds, int32_t count)
{
    KVPageTableRowUpdate const update{slot, kPageIds, count};
    applyRows(&update, 1U);
}

void KVPageTable::setRows(std::vector<KVPageTableRowUpdate> const& updates)
{
    applyRows(updates.data(), updates.size());
}

void KVPageTable::applyRows(KVPageTableRowUpdate const* updates, size_t count)
{
    std::vector<uint8_t> slotsSeen(static_cast<size_t>(mMaxBatch), 0U);
    for (size_t updateIndex = 0; updateIndex < count; ++updateIndex)
    {
        KVPageTableRowUpdate const& update = updates[updateIndex];
        ELLM_CHECK(update.slot >= 0 && update.slot < mMaxBatch,
            "KVPageTable::setRows: slot out of range: " + std::to_string(update.slot));
        ELLM_CHECK(update.count >= 0 && update.count <= mMaxPagesPerSeq,
            "KVPageTable::setRows: count out of range for slot " + std::to_string(update.slot));
        ELLM_CHECK(update.count == 0 || update.kPageIds != nullptr,
            "KVPageTable::setRows: non-empty row has a null page-id pointer at slot " + std::to_string(update.slot));
        ELLM_CHECK(slotsSeen[static_cast<size_t>(update.slot)] == 0U,
            "KVPageTable::setRows: slot appears more than once: " + std::to_string(update.slot));
        slotsSeen[static_cast<size_t>(update.slot)] = 1U;

        for (int32_t pageIndex = 0; pageIndex < update.count; ++pageIndex)
        {
            int32_t const pageId = update.kPageIds[pageIndex];
            ELLM_CHECK(pageId >= 0 && pageId < mNumPages,
                "KVPageTable::setRows: page id " + std::to_string(pageId) + " out of range [0, "
                    + std::to_string(mNumPages) + ") at slot " + std::to_string(update.slot) + ", index "
                    + std::to_string(pageIndex));
        }
    }

    if (count == 0U)
    {
        return;
    }

    mIsIdentity = false;
    for (size_t updateIndex = 0; updateIndex < count; ++updateIndex)
    {
        KVPageTableRowUpdate const& update = updates[updateIndex];
        int32_t* kRow = mutableHostRow(update.slot);
        int32_t* vRow = kRow + mMaxPagesPerSeq;
        bool changed = false;
        for (int32_t pageIndex = 0; pageIndex < update.count; ++pageIndex)
        {
            int32_t const pageId = update.kPageIds[pageIndex];
            int32_t const vPageId = deriveV(pageId, mNumPages);
            changed = changed || kRow[pageIndex] != pageId || vRow[pageIndex] != vPageId;
            kRow[pageIndex] = pageId;
            vRow[pageIndex] = vPageId;
        }
        for (int32_t pageIndex = update.count; pageIndex < mMaxPagesPerSeq; ++pageIndex)
        {
            changed = changed || kRow[pageIndex] != kUNUSED_PAGE_ENTRY || vRow[pageIndex] != kUNUSED_PAGE_ENTRY;
            kRow[pageIndex] = kUNUSED_PAGE_ENTRY;
            vRow[pageIndex] = kUNUSED_PAGE_ENTRY;
        }
        if (changed)
        {
            mDirtyRows[static_cast<size_t>(update.slot)] = 1U;
        }
    }
}

void KVPageTable::compactRows(std::vector<int32_t> const& oldToNew, int32_t newBatch)
{
    ELLM_CHECK(
        oldToNew.size() <= static_cast<size_t>(mMaxBatch), "KVPageTable::compactRows: mapping exceeds max batch size.");
    ELLM_CHECK(newBatch >= 0 && newBatch <= mMaxBatch, "KVPageTable::compactRows: new batch size is out of range.");

    std::vector<uint8_t> destinationSeen(static_cast<size_t>(newBatch), 0U);
    for (int32_t const destination : oldToNew)
    {
        if (destination < 0)
        {
            ELLM_CHECK(destination == -1, "KVPageTable::compactRows: invalid retired-slot marker.");
            continue;
        }
        ELLM_CHECK(destination < newBatch, "KVPageTable::compactRows: destination is out of range.");
        ELLM_CHECK(destinationSeen[static_cast<size_t>(destination)] == 0U,
            "KVPageTable::compactRows: destination appears more than once.");
        destinationSeen[static_cast<size_t>(destination)] = 1U;
    }
    ELLM_CHECK(std::all_of(destinationSeen.begin(), destinationSeen.end(), [](uint8_t seen) { return seen != 0U; }),
        "KVPageTable::compactRows: mapping does not cover every destination.");

    std::fill(mHostScratch.begin(), mHostScratch.end(), kUNUSED_PAGE_ENTRY);
    size_t const rowElements = static_cast<size_t>(2 * mMaxPagesPerSeq);
    for (size_t oldSlot = 0; oldSlot < oldToNew.size(); ++oldSlot)
    {
        int32_t const newSlot = oldToNew[oldSlot];
        if (newSlot < 0)
        {
            continue;
        }
        std::copy_n(mHost.data() + oldSlot * rowElements, rowElements,
            mHostScratch.data() + static_cast<size_t>(newSlot) * rowElements);
    }

    for (int32_t slot = 0; slot < mMaxBatch; ++slot)
    {
        size_t const offset = static_cast<size_t>(slot) * rowElements;
        if (!std::equal(mHost.begin() + offset, mHost.begin() + offset + rowElements, mHostScratch.begin() + offset))
        {
            mDirtyRows[static_cast<size_t>(slot)] = 1U;
        }
    }
    mHost.swap(mHostScratch);
    mIsIdentity = false;
}

bool KVPageTable::checkInvariants(std::string& error) const
{
    for (int32_t b = 0; b < mMaxBatch; ++b)
    {
        int32_t const* kRow = hostRow(b);
        bool sawSentinel = false;
        for (int32_t j = 0; j < mMaxPagesPerSeq; ++j)
        {
            int32_t const k = kRow[j];
            if (k == kUNUSED_PAGE_ENTRY)
            {
                sawSentinel = true;
                continue;
            }
            if (sawSentinel)
            {
                error = "KVPageTable: live page id " + std::to_string(k) + " follows sentinel at slot "
                    + std::to_string(b) + ", index " + std::to_string(j);
                return false;
            }
            if (k < 0 || k >= mNumPages)
            {
                error = "KVPageTable: page id " + std::to_string(k) + " out of range [0, " + std::to_string(mNumPages)
                    + ") at slot " + std::to_string(b) + ", index " + std::to_string(j);
                return false;
            }
        }
    }
    return true;
}

bool KVPageTable::upload(cudaStream_t stream)
{
    if (std::none_of(mDirtyRows.begin(), mDirtyRows.end(), [](uint8_t dirty) { return dirty != 0U; }))
    {
        return false;
    }

    std::string error;
    ELLM_CHECK(checkInvariants(error), "KVPageTable::upload: " + error);

    if (mUploadPending)
    {
        CUDA_CHECK(cudaEventSynchronize(mUploadComplete));
        mUploadPending = false;
    }

    size_t const rowElements = static_cast<size_t>(2 * mMaxPagesPerSeq);
    int32_t* const staging = mUploadStaging.dataPointer<int32_t>();
    int32_t* const device = mDevice.dataPointer<int32_t>();
    int32_t rangeBegin = 0;
    while (rangeBegin < mMaxBatch)
    {
        while (rangeBegin < mMaxBatch && mDirtyRows[static_cast<size_t>(rangeBegin)] == 0U)
        {
            ++rangeBegin;
        }
        if (rangeBegin == mMaxBatch)
        {
            break;
        }

        int32_t rangeEnd = rangeBegin + 1;
        while (rangeEnd < mMaxBatch && mDirtyRows[static_cast<size_t>(rangeEnd)] != 0U)
        {
            ++rangeEnd;
        }

        size_t const elementOffset = static_cast<size_t>(rangeBegin) * rowElements;
        size_t const elementCount = static_cast<size_t>(rangeEnd - rangeBegin) * rowElements;
        std::copy_n(mHost.data() + elementOffset, elementCount, staging + elementOffset);
        CUDA_CHECK(cudaMemcpyAsync(device + elementOffset, staging + elementOffset, elementCount * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream));
        rangeBegin = rangeEnd;
    }

    CUDA_CHECK(cudaEventRecord(mUploadComplete, stream));
    mUploadPending = true;
    std::fill(mDirtyRows.begin(), mDirtyRows.end(), 0U);
    return true;
}

rt::Tensor const& KVPageTable::kernelView() const
{
    return mDevice;
}

rt::Tensor& KVPageTable::kernelView()
{
    return mDevice;
}

int32_t const* KVPageTable::hostRow(int32_t slot) const
{
    return mHost.data() + static_cast<size_t>(slot) * 2 * mMaxPagesPerSeq;
}

int32_t* KVPageTable::mutableHostRow(int32_t slot)
{
    return mHost.data() + static_cast<size_t>(slot) * 2 * mMaxPagesPerSeq;
}

} // namespace rt
} // namespace trt_edgellm
