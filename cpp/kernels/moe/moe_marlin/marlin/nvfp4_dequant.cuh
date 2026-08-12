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

// NVFP4 (E2M1) dequantization for Marlin. This bit-level conversion does not
// rely on CUDA's native FP4 types, so the weight-only fallback remains usable
// on pre-Blackwell GPUs.

#pragma once
#include "kernels/moe/moe_marlin/marlin/dequant.h"

namespace MARLIN_NAMESPACE_NAME
{

template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kFE2M1f.id(), true>(int32_t const q, half2* fragB)
{
    // skip_flop=true omits correction for the exponent-bias delta (15 - 1 = 14). Direct field mapping yields
    // E2M1(q) * 2^-14; the caller must supply the compensating 2^14 factor through its scales.
    constexpr int32_t kFP4_EXPONENT{2};
    constexpr int32_t kFP16_EXPONENT{5};
    constexpr int32_t kRIGHT_SHIFT{kFP16_EXPONENT - kFP4_EXPONENT};
    constexpr int32_t kMASK{0x70007000};

    int32_t const out1 = (q & 0x80008000) | ((q & kMASK) >> kRIGHT_SHIFT);
    int32_t const shiftedQ = q << 4;
    int32_t const out2 = (shiftedQ & 0x80008000) | ((shiftedQ & kMASK) >> kRIGHT_SHIFT);

    // Reverse indexing is intentional because the weights are Marlin-permuted.
    fragB[1] = *reinterpret_cast<half2 const*>(&out1);
    fragB[0] = *reinterpret_cast<half2 const*>(&out2);
}

template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kFE2M1f.id(), false>(int32_t const q, half2* fragB)
{
    dequant<half2, trt_edgellm::marlin_dtypes::kFE2M1f.id(), true>(q, fragB);

    // skip_flop=false restores the omitted bias delta by multiplying the true-path fragments by 2^14.
    constexpr int32_t kFP4_EXPONENT{2};
    constexpr int32_t kFP16_EXPONENT{5};
    constexpr int32_t kBIAS_OFFSET{(1 << (kFP16_EXPONENT - 1)) - (1 << (kFP4_EXPONENT - 1))};
    // Construct 2^kBIAS_OFFSET exactly before converting it to FP16.
    half2 const bias = __float2half2_rn(static_cast<float>(1 << kBIAS_OFFSET));

    fragB[1] = __hmul2(fragB[1], bias);
    fragB[0] = __hmul2(fragB[0], bias);
}

template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kFE2M1f.id(), true>(
    int32_t const q, nv_bfloat162* fragB)
{
    // skip_flop=true omits correction for the exponent-bias delta (127 - 1 = 126). Direct field mapping yields
    // E2M1(q) * 2^-126; block and global scale adjustments supply 2^7 and 2^119, respectively.
    constexpr int32_t kFP4_EXPONENT{2};
    constexpr int32_t kBF16_EXPONENT{8};
    constexpr int32_t kRIGHT_SHIFT{kBF16_EXPONENT - kFP4_EXPONENT};
    constexpr int32_t kMASK{0x70007000};

    int32_t const out1 = (q & 0x80008000) | ((q & kMASK) >> kRIGHT_SHIFT);
    int32_t const shiftedQ = q << 4;
    int32_t const out2 = (shiftedQ & 0x80008000) | ((shiftedQ & kMASK) >> kRIGHT_SHIFT);

    // Reverse indexing is intentional because the weights are Marlin-permuted.
    fragB[1] = *reinterpret_cast<nv_bfloat162 const*>(&out1);
    fragB[0] = *reinterpret_cast<nv_bfloat162 const*>(&out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kFE2M1f.id(), false>(
    int32_t const q, nv_bfloat162* fragB)
{
    dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kFE2M1f.id(), true>(q, fragB);

    // skip_flop=false restores the omitted bias delta by multiplying the true-path fragments by 2^126.
    constexpr int32_t kFP4_EXPONENT{2};
    constexpr int32_t kBF16_EXPONENT{8};
    constexpr int32_t kBIAS_OFFSET{(1 << (kBF16_EXPONENT - 1)) - (1 << (kFP4_EXPONENT - 1))};
    // Construct the FP32 bit pattern for 2^kBIAS_OFFSET; 127 is the IEEE-754 FP32 exponent bias.
    constexpr uint32_t kBIAS{(kBIAS_OFFSET + 127) << 23};
    nv_bfloat162 const bias = __float2bfloat162_rn(*reinterpret_cast<float const*>(&kBIAS));

    fragB[1] = __hmul2(fragB[1], bias);
    fragB[0] = __hmul2(fragB[0], bias);
}

} // namespace MARLIN_NAMESPACE_NAME
