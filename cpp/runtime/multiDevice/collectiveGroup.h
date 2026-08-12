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

#include "runtime/multiDevice/parallelConfig.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Raw collective data type. Values match NCCL's public enum values.
enum class CollectiveDataType
{
    kInt8 = 2,
    kInt32 = 4,
    kInt64 = 5,
    kFloat16 = 6,
    kFloat32 = 7,
    kBfloat16 = 10,
};

//! Non-owning view over one logical communication group.
class CollectiveGroup
{
public:
    CollectiveGroup(ParallelGroupConfig config, std::vector<void*> backendHandles);

    ParallelGroupConfig const& config() const noexcept
    {
        return mConfig;
    }

    ParallelType type() const noexcept
    {
        return mConfig.type;
    }

    int32_t size() const noexcept
    {
        return mConfig.size;
    }

    int32_t rank() const noexcept
    {
        return mConfig.rank;
    }

    int32_t localDevice() const noexcept
    {
        return mConfig.localDevice;
    }

    void* backendHandle(int32_t rank) const noexcept;
    bool broadcast(int32_t rank, void* buffer, int32_t count, CollectiveDataType dataType, int32_t rootRank,
        cudaStream_t stream) const noexcept;
    bool allGather(int32_t rank, void const* sendBuffer, void* recvBuffer, int32_t sendCount,
        CollectiveDataType dataType, cudaStream_t stream) const noexcept;
    bool allReduceSum(int32_t rank, void const* sendBuffer, void* recvBuffer, int32_t count,
        CollectiveDataType dataType, cudaStream_t stream) const noexcept;

    bool broadcastInt32(int32_t rank, void* buffer, int32_t count, int32_t rootRank, cudaStream_t stream) const noexcept
    {
        return broadcast(rank, buffer, count, CollectiveDataType::kInt32, rootRank, stream);
    }

private:
    ParallelGroupConfig mConfig{};
    std::vector<void*> mBackendHandles{};
};

} // namespace rt
} // namespace trt_edgellm
