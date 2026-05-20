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
 *
 * This file contains code derived from FlashInfer (https://github.com/flashinfer-ai/flashinfer)
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Licensed under the Apache License, Version 2.0.
 *
 * Modifications by NVIDIA:
 * - Extracted common utilities needed for Mamba selective state update kernel
 * - Renamed namespace from flashinfer::mamba to mamba_ssm
 * - Removed unused dispatch helpers (dispatchRatio, dispatchDimDstateTokens)
 * - Replaced FLASHINFER_CHECK with direct std::runtime_error throws
 */
#ifndef MAMBA_COMMON_CUH_
#define MAMBA_COMMON_CUH_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/checkMacros.h"

namespace mamba_ssm
{

constexpr unsigned kWARP_SIZE = 32;

// =============================================================================
// Common types and utilities
// =============================================================================

// Simple packed vector type for loading N elements of type T
template <typename T, int N = sizeof(float4) / sizeof(T)>
struct alignas(N * sizeof(T)) PackedAligned
{
    T val[N];
    static constexpr int count = N;
    using dtype = T;
};

template <class load_t>
__device__ __forceinline__ auto make_zeros() -> load_t
{
    load_t ret{};
#pragma unroll
    for (int i = 0; i < ret.count; i++)
        ret.val[i] = typename load_t::dtype{}; // default initialization
    return ret;
};

// Computes the vector load size that ensures full warp utilization.
// Avoids cases like: dstate=64, load_t = sizeof(float4)/sizeof(f16), warpsize=32 (32 * 8 > 64)
// in which case a part of the warp would be idle.
template <typename T, int DSTATE>
inline constexpr auto getVectorLoadSizeForFullUtilization() -> unsigned
{
    static_assert(sizeof(float4) >= sizeof(T));
    constexpr unsigned maxHardwareLoadSize = sizeof(float4) / sizeof(T);
    constexpr unsigned maxLogicalLoadSize = (unsigned) DSTATE / kWARP_SIZE;
    return maxHardwareLoadSize < maxLogicalLoadSize ? maxHardwareLoadSize : maxLogicalLoadSize;
}

__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int s = kWARP_SIZE / 2; s > 0; s /= 2)
    {
        val += __shfl_down_sync(UINT32_MAX, val, s);
    }
    return val;
}

__forceinline__ __device__ float softplus(float x)
{
    return __logf(1.f + __expf(x));
}

__device__ __forceinline__ float thresholded_softplus(float dt_value)
{
    constexpr float threshold = 20.f;
    return (dt_value <= threshold) ? softplus(dt_value) : dt_value;
}

// =============================================================================
// Dispatch helpers
// =============================================================================

// Format an integer_sequence as a comma-separated string for error messages
template <int... Values>
std::string format_sequence(std::integer_sequence<int, Values...>)
{
    std::ostringstream oss;
    bool first = true;
    ((oss << (first ? (first = false, "") : ", ") << Values), ...);
    return oss.str();
}

// Helper: try each DSTATE for a given DIM
template <int DIM, typename ParamsType, typename KernelLauncher, int FirstDstate, int... RestDstates>
bool tryDstate(ParamsType& params, KernelLauncher&& launcher, std::integer_sequence<int, FirstDstate, RestDstates...>)
{
    if (params.dstate == FirstDstate)
    {
        launcher.template operator()<DIM, FirstDstate>();
        return true;
    }
    if constexpr (sizeof...(RestDstates) > 0)
        return tryDstate<DIM>(
            params, std::forward<KernelLauncher>(launcher), std::integer_sequence<int, RestDstates...>{});
    return false;
}

// Helper: try each DIM value
template <typename ParamsType, typename KernelLauncher, int... AllowedDstates, int FirstDim, int... RestDims>
bool tryDim(ParamsType& params, std::integer_sequence<int, AllowedDstates...> dstates_seq, KernelLauncher&& launcher,
    std::integer_sequence<int, FirstDim, RestDims...>)
{
    if (params.dim == FirstDim)
    {
        bool dispatched = tryDstate<FirstDim>(params, std::forward<KernelLauncher>(launcher), dstates_seq);
        if (!dispatched)
        {
            std::ostringstream oss;
            oss << "Unsupported dstate value: " << params.dstate
                << ".\nSupported values: " << format_sequence(dstates_seq);
            throw std::runtime_error(oss.str());
        }
        return true;
    }
    if constexpr (sizeof...(RestDims) > 0)
        return tryDim(
            params, dstates_seq, std::forward<KernelLauncher>(launcher), std::integer_sequence<int, RestDims...>{});
    return false;
}

// Helper function to dispatch dim and dstate with a kernel launcher
template <typename ParamsType, typename KernelLauncher, int... AllowedDims, int... AllowedDstates>
void dispatchDimDstate(ParamsType& params, std::integer_sequence<int, AllowedDims...> dims_seq,
    std::integer_sequence<int, AllowedDstates...> dstates_seq, KernelLauncher&& launcher)
{
    bool dim_dispatched = tryDim(
        params, dstates_seq, std::forward<KernelLauncher>(launcher), std::integer_sequence<int, AllowedDims...>{});
    if (!dim_dispatched)
    {
        std::ostringstream oss;
        oss << "Unsupported dim value: " << params.dim << ".\nSupported values: " << format_sequence(dims_seq);
        throw std::runtime_error(oss.str());
    }
}

// =============================================================================
// Alignment checks
// =============================================================================

// Check alignment for common input variables (x, z, B, C)
// Works for both STP (SelectiveStateUpdateParams) and MTP (SelectiveStateMTPParams)
template <typename input_t, typename ParamsType>
void check_ptr_alignment_input_vars(ParamsType const& params)
{
    using load_input_t = PackedAligned<input_t>;
    auto const align = std::to_string(sizeof(load_input_t));
    ELLM_CHECK(reinterpret_cast<uintptr_t>(params.x) % sizeof(load_input_t) == 0,
        "x pointer must be aligned to " + align + " bytes");
    ELLM_CHECK((params.x_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
        "x batch stride must be aligned to " + align + " bytes");
    if (params.z)
    {
        ELLM_CHECK(reinterpret_cast<uintptr_t>(params.z) % sizeof(load_input_t) == 0,
            "z pointer must be aligned to " + align + " bytes");
        ELLM_CHECK((params.z_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
            "z batch stride must be aligned to " + align + " bytes");
    }
    ELLM_CHECK(reinterpret_cast<uintptr_t>(params.B) % sizeof(load_input_t) == 0,
        "B pointer must be aligned to " + align + " bytes");
    ELLM_CHECK(reinterpret_cast<uintptr_t>(params.C) % sizeof(load_input_t) == 0,
        "C pointer must be aligned to " + align + " bytes");
    ELLM_CHECK((params.B_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
        "B batch stride must be aligned to " + align + " bytes");
    ELLM_CHECK((params.C_stride_batch * sizeof(input_t)) % sizeof(load_input_t) == 0,
        "C batch stride must be aligned to " + align + " bytes");
}

} // namespace mamba_ssm

#endif // MAMBA_COMMON_CUH_
