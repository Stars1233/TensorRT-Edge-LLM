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

#include "common/tensor.h"
#include "runtime/llmRuntimeUtils.h"

#include <cstdint>
#include <deque>

namespace trt_edgellm
{
namespace rt
{

//! Pool of immutable RoPE cos/sin cache tensors with automatic deduplication.
//!
//! `getOrCreate(config, ...)` returns a reference to an existing GPU tensor
//! if the same RoPE configuration has been seen before, otherwise creates and
//! caches a new one.  This saves GPU memory when multiple engines share the
//! same RoPE parameters (e.g. base + draft in EAGLE).
//!
//! Implementation note: entries are stored in a `std::deque` so that
//! previously returned `Tensor&` references remain stable when new entries
//! are added.
class RopeCache
{
public:
    //! @brief Default constructor
    RopeCache() = default;

    //! Deleted copy to avoid accidental duplication of GPU resources.
    RopeCache(RopeCache const&) = delete;
    RopeCache& operator=(RopeCache const&) = delete;

    //! Allow move.
    RopeCache(RopeCache&&) noexcept = default;
    RopeCache& operator=(RopeCache&&) noexcept = default;

    //! Obtain (or create) a RoPE cos/sin cache tensor for the given configuration.
    //!
    //! @param config       RoPE configuration (type, theta, scale, maxPositionEmbeddings, ...).
    //! @param rotaryDim    Number of dimensions that undergo rotation.
    //! @param maxSeqLen    Maximum sequence length for the cache.
    //! @param stream       CUDA stream used when a new tensor must be initialized.
    //! @return Reference to the cached GPU tensor (stable across future calls).
    rt::Tensor& getOrCreate(RopeConfig const& config, int32_t rotaryDim, int32_t maxSeqLen, cudaStream_t stream);

    //! @brief Return the number of cached entries.
    size_t size() const noexcept;

private:
    //! A single cache entry binding a RoPE configuration to its GPU tensor.
    struct Entry
    {
        RopeConfig config{};
        int32_t rotaryDim{};
        int32_t maxSeqLen{};
        rt::Tensor tensor{};
    };

    //! Check whether two RoPE configurations are semantically equivalent.
    static bool configsMatch(RopeConfig const& a, int32_t rotaryDimA, int32_t maxSeqLenA, RopeConfig const& b,
        int32_t rotaryDimB, int32_t maxSeqLenB) noexcept;

    //! Deque provides reference stability — existing `Tensor&` references
    //! remain valid when new entries are pushed.
    std::deque<Entry> mEntries{};
};

} // namespace rt
} // namespace trt_edgellm
