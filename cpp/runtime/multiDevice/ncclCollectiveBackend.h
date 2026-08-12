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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

enum class CollectiveDataType;

//! NCCL unique ID storage without exposing NCCL headers to runtime consumers.
struct NcclUniqueId
{
    char internal[128];
};

//! Dynamically loaded NCCL collective backend shared by parallel collectives and plugin resource registration.
namespace NcclCollectiveBackend
{
inline constexpr int32_t kSuccess = 0;
inline constexpr int32_t kUnavailable = -1;

//! Load NCCL and all required symbols, or throw on failure.
void load();

//! Return true if NCCL has already been loaded successfully.
bool isLoaded() noexcept;

//! Create one NCCL unique ID.
void getUniqueId(NcclUniqueId& uniqueId);

//! Initialize one rank's communicator from an exchanged NCCL unique ID.
void initRank(void** comm, int32_t size, NcclUniqueId const& uniqueId, int32_t rank);

//! Initialize all ranks in one process using ncclCommInitAll.
std::vector<void*> initAll(std::vector<int32_t> const& devices);

//! Destroy one communicator handle.
void destroyComm(void* comm) noexcept;

//! Abort one communicator handle to release peers blocked in a collective.
void abortComm(void* comm) noexcept;

//! Abort a group of communicator handles.
void abortComms(std::vector<void*> const& comms) noexcept;

//! Destroy a group of communicator handles.
void destroyComms(std::vector<void*> const& comms) noexcept;

void allReduce(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType, void* comm,
    cudaStream_t stream);
void allGather(void const* sendBuffer, void* recvBuffer, size_t sendCount, CollectiveDataType dataType, void* comm,
    cudaStream_t stream);
void broadcast(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType, int32_t rootRank,
    void* comm, cudaStream_t stream);

//! Raw noexcept NCCL calls for callers that need to convert failures into status returns.
int32_t allReduceRaw(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType, void* comm,
    cudaStream_t stream) noexcept;
int32_t allGatherRaw(void const* sendBuffer, void* recvBuffer, size_t sendCount, CollectiveDataType dataType,
    void* comm, cudaStream_t stream) noexcept;
int32_t broadcastRaw(void const* sendBuffer, void* recvBuffer, size_t count, CollectiveDataType dataType,
    int32_t rootRank, void* comm, cudaStream_t stream) noexcept;

//! Function pointer used by TensorRT plugins that call ncclAllReduce directly.
void* allReduceFunction() noexcept;

} // namespace NcclCollectiveBackend

} // namespace rt
} // namespace trt_edgellm
