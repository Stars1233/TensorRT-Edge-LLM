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

#include "common/allReducePath.h"
#include "runtime/multiDevice/parallelConfig.h"

#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

class RuntimeCollectiveResources;

//! Resource capability owned by one TensorRT plugin AllReduce execution path.
class PluginAllReducePathResources
{
public:
    virtual ~PluginAllReducePathResources() noexcept = default;

    virtual AllReducePathType type() const noexcept = 0;
    virtual bool registered() const noexcept = 0;

    //! Optional runtime-collective view supplied by communication paths such as NCCL.
    virtual RuntimeCollectiveResources* runtimeCollectives() noexcept
    {
        return nullptr;
    }
};

//! Non-owning runtime collective capability backed by communicator handles.
class RuntimeCollectiveResources
{
public:
    virtual ~RuntimeCollectiveResources() noexcept = default;

    virtual void* communicatorForRank(int32_t rank) const noexcept = 0;

    //! Abort communicators owned by this resource. Externally owned communicators are never aborted implicitly.
    //! Returns true when this resource owns the communicators and initiated (or already completed) their abort.
    virtual bool abortOwnedCommunicators() noexcept = 0;
};

//! Non-hot-path owner for plugin communication resources associated with one parallel group.
class MultiDevicePluginResources
{
public:
    virtual ~MultiDevicePluginResources() noexcept = default;

    virtual ParallelType type() const noexcept = 0;
    virtual int32_t size() const noexcept = 0;
    virtual RuntimeCollectiveResources const* runtimeCollectives() const noexcept = 0;
    virtual bool hasAllReducePath(AllReducePathType type) const noexcept = 0;
    virtual bool abortOwnedRuntimeCollectives() noexcept = 0;
};

} // namespace rt
} // namespace trt_edgellm
