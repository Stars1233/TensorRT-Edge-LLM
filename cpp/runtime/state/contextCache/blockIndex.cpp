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

#include "runtime/state/contextCache/blockIndex.h"

#include "common/checkMacros.h"
#include "runtime/state/contextCache/cacheRecord.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

BaseLookupResult BaseBlockIndex::lookupPrefix(std::vector<BlockHash> const& hashes) const
{
    BaseLookupResult result;
    result.pageIds.reserve(hashes.size());
    result.matchedHashes.reserve(hashes.size());
    for (BlockHash const& hash : hashes)
    {
        auto const iter = mForward.find(hash);
        if (iter == mForward.end())
        {
            break;
        }
        result.pageIds.push_back(iter->second);
        result.matchedHashes.push_back(hash);
    }
    return result;
}

std::optional<PageId> BaseBlockIndex::lookup(BlockHash const& hash) const
{
    auto const iter = mForward.find(hash);
    if (iter == mForward.end())
    {
        return std::nullopt;
    }
    return iter->second;
}

BaseInsertResult BaseBlockIndex::insert(BlockHash hash, PageId proposedPage)
{
    auto const existing = mForward.find(hash);
    if (existing != mForward.end())
    {
        return BaseInsertResult{existing->second, false};
    }
    ELLM_CHECK(proposedPage >= 0, "Cannot index a negative context cache page ID");
    ELLM_CHECK(mReverse.find(proposedPage) == mReverse.end(), "Context cache page is already indexed by another block");

    // Updating the forward and reverse maps is one transaction. If the reverse insertion allocates and throws, the
    // catch below removes the forward entry so readers never observe a one-sided mapping.
    auto const [forwardIter, inserted] = mForward.emplace(hash, proposedPage);
    if (!inserted)
    {
        return BaseInsertResult{forwardIter->second, false};
    }
    try
    {
        auto const [reverseIter, reverseInserted] = mReverse.emplace(proposedPage, hash);
        (void) reverseIter;
        ELLM_CHECK(reverseInserted, "Context cache page is already indexed by another block");
    }
    catch (...)
    {
        mForward.erase(forwardIter);
        throw;
    }

    return BaseInsertResult{proposedPage, true};
}

void BaseBlockIndex::erasePage(PageId page)
{
    auto const reverseIter = mReverse.find(page);
    if (reverseIter == mReverse.end())
    {
        return;
    }

    auto const forwardIter = mForward.find(reverseIter->second);
    if (forwardIter != mForward.end() && forwardIter->second == page)
    {
        mForward.erase(forwardIter);
    }
    mReverse.erase(reverseIter);
}

size_t BaseBlockIndex::size() const noexcept
{
    return mForward.size();
}

void DraftPathIndex::insert(CacheRecord const& record)
{
    ELLM_CHECK(record.id != 0, "Cannot index draft state without a context cache record ID");
    ELLM_CHECK(!record.draftPagePath.empty(), "Cannot index a context cache record without paired draft state");
    ELLM_CHECK(record.draftPagePath.size() == record.logicalBlockHashes.size(),
        "Context cache draft path must cover its full logical path");

    size_t const pathCount = record.draftPagePath.size();
    mForward.reserve(mForward.size() + pathCount);
    std::vector<BlockHash> createdKeys;
    std::vector<std::pair<BlockHash, DraftPathMatch>> insertedMatches;
    createdKeys.reserve(pathCount);
    insertedMatches.reserve(pathCount);
    try
    {
        for (size_t index = 0; index < pathCount; ++index)
        {
            BlockHash const terminalHash = record.logicalBlockHashes[index];
            DraftPathMatch const match{record.id, static_cast<int32_t>(index + 1)};
            auto const [entry, inserted] = mForward.try_emplace(terminalHash);
            ELLM_CHECK(std::find(entry->second.begin(), entry->second.end(), match) == entry->second.end(),
                "Context cache draft path record is already indexed");
            if (inserted)
            {
                createdKeys.push_back(terminalHash);
            }
            entry->second.push_back(match);
            insertedMatches.emplace_back(terminalHash, match);
        }
    }
    catch (...)
    {
        // Preserve the strong exception guarantee: undo every match and now-empty key created by this call before
        // rethrowing an allocation or invariant failure.
        for (auto inserted = insertedMatches.rbegin(); inserted != insertedMatches.rend(); ++inserted)
        {
            auto entry = mForward.find(inserted->first);
            if (entry == mForward.end())
            {
                continue;
            }
            auto const match = std::find(entry->second.begin(), entry->second.end(), inserted->second);
            if (match != entry->second.end())
            {
                entry->second.erase(match);
            }
        }
        for (auto created = createdKeys.rbegin(); created != createdKeys.rend(); ++created)
        {
            auto entry = mForward.find(*created);
            if (entry != mForward.end() && entry->second.empty())
            {
                mForward.erase(entry);
            }
        }
        throw;
    }
}

std::optional<DraftPathMatch> DraftPathIndex::lookupLongest(
    std::vector<BlockHash> const& hashes, int32_t maxBlockCount) const
{
    ELLM_CHECK(maxBlockCount >= 0 && static_cast<size_t>(maxBlockCount) <= hashes.size(),
        "Context cache draft lookup block count exceeds its hash path");
    for (int32_t blockCount = maxBlockCount; blockCount > 0; --blockCount)
    {
        auto const entry = mForward.find(hashes[static_cast<size_t>(blockCount - 1)]);
        if (entry != mForward.end() && !entry->second.empty())
        {
            auto const match = std::find_if(entry->second.rbegin(), entry->second.rend(),
                [blockCount](DraftPathMatch const& candidate) { return candidate.pathBlockCount == blockCount; });
            if (match != entry->second.rend())
            {
                return *match;
            }
        }
    }
    return std::nullopt;
}

bool DraftPathIndex::contains(BlockHash const& terminalHash, DraftPathMatch const& match) const
{
    auto const entry = mForward.find(terminalHash);
    return entry != mForward.end()
        && std::find(entry->second.begin(), entry->second.end(), match) != entry->second.end();
}

void DraftPathIndex::erase(CacheRecord const& record)
{
    if (record.draftPagePath.empty())
    {
        return;
    }
    for (int32_t blockCount = 1; blockCount <= static_cast<int32_t>(record.draftPagePath.size()); ++blockCount)
    {
        BlockHash const terminalHash = record.logicalBlockHashes[static_cast<size_t>(blockCount - 1)];
        auto entry = mForward.find(terminalHash);
        ELLM_CHECK(entry != mForward.end(), "Context cache draft path index is missing a record boundary");
        DraftPathMatch const match{record.id, blockCount};
        auto const candidate = std::find(entry->second.begin(), entry->second.end(), match);
        ELLM_CHECK(candidate != entry->second.end(), "Context cache draft path index is missing a record candidate");
        entry->second.erase(candidate);
        if (entry->second.empty())
        {
            mForward.erase(entry);
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
