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

#include <cstdint>
#include <limits>

namespace trt_edgellm::rt
{

//! Page size P for the paged-KV layout (tokens per page).
constexpr int32_t kTOKENS_PER_PAGE{128};

//! Sentinel value for an unallocated page-table entry.
constexpr int32_t kUNUSED_PAGE_ENTRY{-1};

//! Largest K-page count for which both the derived V ids and the CuTe FMHA combined
//! K/V page count (`2 * numPages`) fit a positive int32.
constexpr int64_t kMAX_KV_POOL_PAGES{std::numeric_limits<int32_t>::max() / 2};

//! Largest token capacity whose page-aligned padded value still fits int32.
constexpr int32_t kMAX_KV_CACHE_CAPACITY = (std::numeric_limits<int32_t>::max() / kTOKENS_PER_PAGE) * kTOKENS_PER_PAGE;

//! Number of pages a single slot's padded token capacity spans.
inline int32_t pagesPerSlot(int32_t maxCapPadded)
{
    return maxCapPadded / kTOKENS_PER_PAGE;
}

//! Maximum number of pages one sequence's KV capacity spans (the page table's last dim).
inline int32_t computeMaxPagesPerSeq(int32_t maxKVCacheCapacity)
{
    return static_cast<int32_t>((static_cast<int64_t>(maxKVCacheCapacity) + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE);
}

//! Computes the paged-KV pool's minimum active pages:
//! `maxBatchSize * ceil(maxKVCacheCapacity / kTOKENS_PER_PAGE)`.
//! Identity-mapped active slots occupy these first pages. A build may serialize extra retained pages for
//! cross-request reuse, but its engine profile, runtime registry, allocation, and page table must all use that same
//! count. Callers validate positive inputs and validate against `kMAX_KV_POOL_PAGES` before narrowing to int32.
inline int64_t computeMinimumKvPoolPages(int64_t maxBatchSize, int64_t maxKVCacheCapacity)
{
    int64_t const pagesPerSequence
        = (maxKVCacheCapacity + static_cast<int64_t>(kTOKENS_PER_PAGE) - 1) / kTOKENS_PER_PAGE;
    return maxBatchSize * pagesPerSequence;
}

} // namespace trt_edgellm::rt
