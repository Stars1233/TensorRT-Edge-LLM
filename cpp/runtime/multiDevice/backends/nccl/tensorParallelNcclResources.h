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

#include "runtime/multiDevice/multiDevicePluginResources.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Create and register the NCCL execution-path resources for one tensor-parallel group.
std::unique_ptr<PluginAllReducePathResources> createTensorParallelNcclResources(int32_t tpSize,
    std::vector<int32_t> localRanks, std::vector<int32_t> localDevices, std::vector<void*> ncclComms,
    bool ownsNcclComms);

} // namespace rt
} // namespace trt_edgellm
