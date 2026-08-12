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

#include "runtime/multiDevice/collectiveGroup.h"

#include "common/logger.h"
#include "runtime/multiDevice/ncclCollectiveBackend.h"

#include <utility>

namespace trt_edgellm
{
namespace rt
{

CollectiveGroup::CollectiveGroup(ParallelGroupConfig config, std::vector<void*> backendHandles)
    : mConfig(std::move(config))
    , mBackendHandles(std::move(backendHandles))
{
}

void* CollectiveGroup::backendHandle(int32_t rank) const noexcept
{
    if (rank < 0 || rank >= static_cast<int32_t>(mBackendHandles.size()))
    {
        return nullptr;
    }
    return mBackendHandles[rank];
}

bool CollectiveGroup::broadcast(int32_t rank, void* buffer, int32_t count, CollectiveDataType dataType,
    int32_t rootRank, cudaStream_t stream) const noexcept
{
    if (mConfig.size <= 1)
    {
        return true;
    }
    if (buffer == nullptr || count <= 0)
    {
        LOG_ERROR("CollectiveGroup[%s]: invalid broadcast buffer/count.", mConfig.typeName());
        return false;
    }

    void* ncclComm = backendHandle(rank);
    if (ncclComm == nullptr)
    {
        LOG_ERROR(
            "CollectiveGroup[%s rank %d/%d]: NCCL broadcast is unavailable.", mConfig.typeName(), rank, mConfig.size);
        return false;
    }

    int32_t const result = NcclCollectiveBackend::broadcastRaw(
        buffer, buffer, static_cast<size_t>(count), dataType, rootRank, ncclComm, stream);
    if (result != NcclCollectiveBackend::kSuccess)
    {
        if (result == NcclCollectiveBackend::kUnavailable)
        {
            LOG_ERROR("CollectiveGroup[%s rank %d/%d]: NCCL broadcast function is unavailable.", mConfig.typeName(),
                rank, mConfig.size);
            return false;
        }
        LOG_ERROR("CollectiveGroup[%s rank %d/%d]: NCCL broadcast failed with error %d.", mConfig.typeName(), rank,
            mConfig.size, result);
        return false;
    }
    return true;
}

bool CollectiveGroup::allGather(int32_t rank, void const* sendBuffer, void* recvBuffer, int32_t sendCount,
    CollectiveDataType dataType, cudaStream_t stream) const noexcept
{
    if (mConfig.size <= 1)
    {
        return true;
    }
    if (sendBuffer == nullptr || recvBuffer == nullptr || sendCount <= 0)
    {
        LOG_ERROR("CollectiveGroup[%s]: invalid all-gather buffer/count.", mConfig.typeName());
        return false;
    }

    void* ncclComm = backendHandle(rank);
    if (ncclComm == nullptr)
    {
        LOG_ERROR(
            "CollectiveGroup[%s rank %d/%d]: NCCL all-gather is unavailable.", mConfig.typeName(), rank, mConfig.size);
        return false;
    }

    int32_t const result = NcclCollectiveBackend::allGatherRaw(
        sendBuffer, recvBuffer, static_cast<size_t>(sendCount), dataType, ncclComm, stream);
    if (result != NcclCollectiveBackend::kSuccess)
    {
        if (result == NcclCollectiveBackend::kUnavailable)
        {
            LOG_ERROR("CollectiveGroup[%s rank %d/%d]: NCCL all-gather function is unavailable.", mConfig.typeName(),
                rank, mConfig.size);
            return false;
        }
        LOG_ERROR("CollectiveGroup[%s rank %d/%d]: NCCL all-gather failed with error %d.", mConfig.typeName(), rank,
            mConfig.size, result);
        return false;
    }
    return true;
}

bool CollectiveGroup::allReduceSum(int32_t rank, void const* sendBuffer, void* recvBuffer, int32_t count,
    CollectiveDataType dataType, cudaStream_t stream) const noexcept
{
    if (mConfig.size <= 1)
    {
        return true;
    }
    if (sendBuffer == nullptr || recvBuffer == nullptr || count <= 0)
    {
        LOG_ERROR("CollectiveGroup[%s]: invalid all-reduce buffer/count.", mConfig.typeName());
        return false;
    }

    void* ncclComm = backendHandle(rank);
    if (ncclComm == nullptr)
    {
        LOG_ERROR(
            "CollectiveGroup[%s rank %d/%d]: NCCL all-reduce is unavailable.", mConfig.typeName(), rank, mConfig.size);
        return false;
    }

    int32_t const result = NcclCollectiveBackend::allReduceRaw(
        sendBuffer, recvBuffer, static_cast<size_t>(count), dataType, ncclComm, stream);
    if (result != NcclCollectiveBackend::kSuccess)
    {
        if (result == NcclCollectiveBackend::kUnavailable)
        {
            LOG_ERROR("CollectiveGroup[%s rank %d/%d]: NCCL all-reduce function is unavailable.", mConfig.typeName(),
                rank, mConfig.size);
            return false;
        }
        LOG_ERROR("CollectiveGroup[%s rank %d/%d]: NCCL all-reduce failed with error %d.", mConfig.typeName(), rank,
            mConfig.size, result);
        return false;
    }
    return true;
}

} // namespace rt
} // namespace trt_edgellm
