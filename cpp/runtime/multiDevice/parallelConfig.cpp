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

#include "runtime/multiDevice/parallelConfig.h"

#include <algorithm>
#include <sstream>

namespace trt_edgellm
{
namespace rt
{

namespace
{

void appendDeviceList(std::ostringstream& ss, std::vector<int32_t> const& devices)
{
    ss << "[";
    for (size_t index = 0; index < devices.size(); ++index)
    {
        if (index > 0)
        {
            ss << ", ";
        }
        ss << devices[index];
    }
    ss << "]";
}

} // namespace

char const* parallelTypeName(ParallelType type) noexcept
{
    switch (type)
    {
    case ParallelType::kTensor: return "tensor";
    }
    return "unknown";
}

char const* parallelLaunchModeName(ParallelLaunchMode mode) noexcept
{
    switch (mode)
    {
    case ParallelLaunchMode::kThread: return "thread";
    case ParallelLaunchMode::kMpi: return "mpi";
    }
    return "unknown";
}

char const* ParallelGroupConfig::typeName() const noexcept
{
    return parallelTypeName(type);
}

std::string ParallelGroupConfig::toString() const
{
    std::ostringstream ss;
    ss << "ParallelGroupConfig{type=" << typeName() << ", size=" << size << ", rank=" << rank
       << ", globalRank=" << globalRank << ", localDevice=" << localDevice << "}";
    return ss.str();
}

std::string ParallelConfig::toString() const
{
    std::ostringstream ss;
    ss << "ParallelConfig{tensorParallelSize=" << tensorParallelSize
       << ", launchMode=" << parallelLaunchModeName(launchMode) << ", devices=";
    appendDeviceList(ss, devices);
    ss << "}";
    return ss.str();
}

std::string ParallelMapping::toString() const
{
    std::ostringstream ss;
    ss << "ParallelMapping{worldSize=" << worldSize << ", globalRank=" << globalRank << ", localDevice=" << localDevice
       << ", tensorParallelSize=" << tensorParallelSize << ", tensorParallelRank=" << tensorParallelRank << "}";
    return ss.str();
}

bool isInlineSingleRank(ParallelLaunchMode launchMode, int32_t worldSize, size_t localRankCount) noexcept
{
    return launchMode == ParallelLaunchMode::kThread && worldSize == 1 && localRankCount == 1;
}

bool isFullLocalParallelGroup(int32_t groupSize, std::vector<int32_t> const& localRanks) noexcept
{
    if (groupSize <= 0 || static_cast<int32_t>(localRanks.size()) != groupSize)
    {
        return false;
    }
    for (int32_t rank = 0; rank < groupSize; ++rank)
    {
        if (localRanks[static_cast<size_t>(rank)] != rank)
        {
            return false;
        }
    }
    return true;
}

int32_t parallelGroupSize(ParallelConfig const& config, ParallelType type) noexcept
{
    switch (type)
    {
    case ParallelType::kTensor: return std::max(1, config.tensorParallelSize);
    }
    return 1;
}

int32_t parallelWorldSize(ParallelConfig const& config) noexcept
{
    return parallelGroupSize(config, ParallelType::kTensor);
}

int32_t parallelGroupRank(ParallelConfig const& config, ParallelType type, int32_t globalRank) noexcept
{
    int32_t rank = std::max(0, globalRank);
    int32_t const tensorSize = parallelGroupSize(config, ParallelType::kTensor);

    switch (type)
    {
    case ParallelType::kTensor: return rank % tensorSize;
    }
    return 0;
}

ParallelMapping makeParallelMapping(ParallelConfig const& config, int32_t globalRank, int32_t localDevice) noexcept
{
    ParallelMapping mapping{};
    mapping.worldSize = parallelWorldSize(config);
    mapping.globalRank = std::max(0, globalRank);
    mapping.localDevice = localDevice;
    mapping.tensorParallelSize = parallelGroupSize(config, ParallelType::kTensor);
    mapping.tensorParallelRank = parallelGroupRank(config, ParallelType::kTensor, mapping.globalRank);
    return mapping;
}

ParallelGroupConfig makeParallelGroupConfig(
    ParallelConfig const& config, ParallelType type, int32_t globalRank, int32_t localDevice)
{
    ParallelGroupConfig groupConfig{};
    groupConfig.type = type;
    groupConfig.size = parallelGroupSize(config, type);
    groupConfig.rank = parallelGroupRank(config, type, globalRank);
    groupConfig.globalRank = globalRank;
    groupConfig.localDevice = localDevice;
    return groupConfig;
}

std::vector<ParallelGroupConfig> activeParallelGroups(
    ParallelConfig const& config, int32_t globalRank, int32_t localDevice)
{
    std::vector<ParallelGroupConfig> groups;
    for (ParallelType const type : {ParallelType::kTensor})
    {
        if (parallelGroupSize(config, type) > 1)
        {
            groups.push_back(makeParallelGroupConfig(config, type, globalRank, localDevice));
        }
    }
    return groups;
}

} // namespace rt
} // namespace trt_edgellm
