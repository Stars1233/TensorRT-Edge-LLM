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

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

//! Maximum number of source tensors carried in one CUDA launch. This is not a
//! checkpoint/model limit: host orchestration splits an arbitrary source count
//! into consecutive launches of this size.
inline constexpr int32_t kCheckpointSourcesPerLaunch = 8;

//! One launch's source pointers. Named fields keep the pointers in CUDA kernel
//! parameter space; an indexed member array makes nvcc materialize the array in
//! every thread's local memory.
template <typename T>
struct CheckpointSourceBatch
{
    T const* source0{};
    T const* source1{};
    T const* source2{};
    T const* source3{};
    T const* source4{};
    T const* source5{};
    T const* source6{};
    T const* source7{};

    __device__ __forceinline__ T const* get(int32_t index) const
    {
        switch (index)
        {
        case 0: return source0;
        case 1: return source1;
        case 2: return source2;
        case 3: return source3;
        case 4: return source4;
        case 5: return source5;
        case 6: return source6;
        case 7: return source7;
        default: return nullptr;
        }
    }
};

template <typename T>
CheckpointSourceBatch<T> makeCheckpointSourceBatch(T const* const* sources, int32_t count)
{
    assert(count >= 0 && count <= kCheckpointSourcesPerLaunch);
    assert(count == 0 || sources != nullptr);
    CheckpointSourceBatch<T> result{};
    result.source0 = count > 0 ? sources[0] : nullptr;
    result.source1 = count > 1 ? sources[1] : nullptr;
    result.source2 = count > 2 ? sources[2] : nullptr;
    result.source3 = count > 3 ? sources[3] : nullptr;
    result.source4 = count > 4 ? sources[4] : nullptr;
    result.source5 = count > 5 ? sources[5] : nullptr;
    result.source6 = count > 6 ? sources[6] : nullptr;
    result.source7 = count > 7 ? sources[7] : nullptr;
    return result;
}

} // namespace kernel
} // namespace trt_edgellm
