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

#include "runtime/state/ropeCache.h"

#include "common/logger.h"

namespace trt_edgellm
{
namespace rt
{

bool RopeCache::configsMatch(RopeConfig const& a, int32_t rotaryDimA, int32_t maxSeqLenA, RopeConfig const& b,
    int32_t rotaryDimB, int32_t maxSeqLenB) noexcept
{
    if (a.type != b.type)
    {
        return false;
    }
    if (rotaryDimA != rotaryDimB)
    {
        return false;
    }
    if (maxSeqLenA != maxSeqLenB)
    {
        return false;
    }
    // For NoRope, all params are irrelevant — the cache is always identity.
    if (a.type == RopeType::kNoRope)
    {
        return true;
    }
    // Compare numerical parameters.
    if (a.rotaryScale != b.rotaryScale)
    {
        return false;
    }
    if (a.rotaryTheta != b.rotaryTheta)
    {
        return false;
    }
    if (a.maxPositionEmbeddings != b.maxPositionEmbeddings)
    {
        return false;
    }
    // LongRope-specific comparison.
    if (a.type == RopeType::kLongRope)
    {
        bool const aHasLong = a.longRope.has_value();
        bool const bHasLong = b.longRope.has_value();
        if (aHasLong != bHasLong)
        {
            return false;
        }
        if (aHasLong)
        {
            auto const& la = *a.longRope;
            auto const& lb = *b.longRope;
            if (la.originalMaxPositionEmbeddings != lb.originalMaxPositionEmbeddings)
            {
                return false;
            }
            if (la.longFactor != lb.longFactor)
            {
                return false;
            }
            if (la.shortFactor != lb.shortFactor)
            {
                return false;
            }
        }
    }
    return true;
}

rt::Tensor& RopeCache::getOrCreate(RopeConfig const& config, int32_t rotaryDim, int32_t maxSeqLen, cudaStream_t stream)
{
    // Check for an existing compatible entry.
    for (auto& entry : mEntries)
    {
        if (configsMatch(entry.config, entry.rotaryDim, entry.maxSeqLen, config, rotaryDim, maxSeqLen))
        {
            LOG_DEBUG("RopeCache: reusing existing entry (rotaryDim=%d, maxSeqLen=%d)", rotaryDim, maxSeqLen);
            return entry.tensor;
        }
    }

    // Create a new entry.
    LOG_INFO("RopeCache: creating new entry (rotaryDim=%d, maxSeqLen=%d)", rotaryDim, maxSeqLen);

    Entry newEntry;
    newEntry.config = config;
    newEntry.rotaryDim = rotaryDim;
    newEntry.maxSeqLen = maxSeqLen;

    // Allocate the cos/sin cache tensor: shape [1, maxSeqLen, rotaryDim].
    newEntry.tensor
        = rt::Tensor({1, maxSeqLen, rotaryDim}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "RopeCache::cosSinCache");

    // Initialize the cache based on RoPE type.
    if (config.type == RopeType::kNoRope)
    {
        initializeNopeCosSinCache(newEntry.tensor, stream);
    }
    else
    {
        initializeRopeCosSinCache(newEntry.tensor, config, stream);
    }

    mEntries.push_back(std::move(newEntry));
    return mEntries.back().tensor;
}

size_t RopeCache::size() const noexcept
{
    return mEntries.size();
}

} // namespace rt
} // namespace trt_edgellm
