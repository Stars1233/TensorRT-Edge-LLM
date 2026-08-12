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

#include "common/cudaMacros.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4
{

__device__ __forceinline__ uint16_t e4m3ToFloat16Times128(uint32_t const raw)
{
    uint16_t const sign = static_cast<uint16_t>((raw & 0x80U) << 8);
    uint32_t const exponent = (raw >> 3) & 0xFU;
    uint32_t const mantissa = raw & 0x7U;
    if (exponent != 0)
    {
        // E4M3 normal: (1.m) * 2^(exponent - 7) * 2^7.
        return static_cast<uint16_t>(sign | ((exponent + 15U) << 10) | (mantissa << 7));
    }
    if (mantissa == 0)
    {
        return sign;
    }

    // E4M3 subnormal * 2^7 is mantissa / 4. Normalize the three-bit
    // mantissa explicitly so every finite E4M3 scale remains exact in FP16.
    uint32_t const leadingBit = mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    uint32_t const fp16Exponent = 13U + leadingBit;
    uint32_t const fp16Mantissa = (mantissa - (1U << leadingBit)) << (10U - leadingBit);
    return static_cast<uint16_t>(sign | (fp16Exponent << 10) | fp16Mantissa);
}

__device__ __forceinline__ uint32_t e4m3x2ToFloat16x2Times128(uint16_t const raw)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && SUPPORTS_FP8
    uint32_t converted;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(converted) : "h"(raw));
    constexpr uint32_t kTIMES_128_SCALE{0x58005800U};
    half2 const result
        = __hmul2(*reinterpret_cast<half2 const*>(&converted), *reinterpret_cast<half2 const*>(&kTIMES_128_SCALE));
    return *reinterpret_cast<uint32_t const*>(&result);
#else
    uint32_t const low = static_cast<uint32_t>(e4m3ToFloat16Times128(raw & 0xFFU));
    uint32_t const high = static_cast<uint32_t>(e4m3ToFloat16Times128((raw >> 8) & 0xFFU));
    return low | (high << 16);
#endif
}

__device__ __forceinline__ uint16_t e4m3ToBfloat16Times128(uint32_t const raw)
{
    uint16_t const sign = static_cast<uint16_t>((raw & 0x80U) << 8);
    uint32_t const exponent = (raw >> 3) & 0xFU;
    uint32_t const mantissa = raw & 0x7U;
    if (exponent != 0)
    {
        // E4M3 normal: (1.m) * 2^(exponent - 7) * 2^7.
        return static_cast<uint16_t>(sign | ((exponent + 127U) << 7) | (mantissa << 4));
    }
    if (mantissa == 0)
    {
        return sign;
    }

    // E4M3 subnormal * 2^7 is mantissa / 4. Normalize the three-bit
    // mantissa explicitly so every positive E4M3 code remains exact in BF16.
    uint32_t const leadingBit = mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    uint32_t const bf16Exponent = 125U + leadingBit;
    uint32_t const bf16Mantissa = (mantissa - (1U << leadingBit)) << (7U - leadingBit);
    return static_cast<uint16_t>(sign | (bf16Exponent << 7) | bf16Mantissa);
}

__device__ __forceinline__ uint32_t e4m3x2ToBfloat16x2Times128(uint16_t const raw)
{
#if defined(EDGELLM_CUDA_VERSION) && (EDGELLM_CUDA_VERSION >= 13020)                                                   \
    && ((defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))                            \
        || (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1100)))
    // PTX 9.2 can convert two packed E4M3 values to BF16 and multiply both by
    // packed UE8M0 scale factors. UE8M0 0x86 is 2^7, preserving the existing
    // Marlin scale/global-scale exponent compensation exactly.
    constexpr uint16_t kTIMES_128_SCALE{0x8686U};
    uint32_t result;
    asm volatile("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %0, %1, %2;\n"
        : "=r"(result)
        : "h"(raw), "h"(kTIMES_128_SCALE));
    return result;
#else
    uint32_t const low = static_cast<uint32_t>(e4m3ToBfloat16Times128(raw & 0xFFU));
    uint32_t const high = static_cast<uint32_t>(e4m3ToBfloat16Times128((raw >> 8) & 0xFFU));
    return low | (high << 16);
#endif
}

} // namespace nvfp4
} // namespace kernel
} // namespace trt_edgellm
