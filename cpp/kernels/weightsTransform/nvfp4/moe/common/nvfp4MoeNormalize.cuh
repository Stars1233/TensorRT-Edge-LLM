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
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/cudaMacros.h"

#include <cstdint>
#include <cuda_bf16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace detail
{

struct ExpertFloatBatch
{
    float value0{1.0F};
    float value1{1.0F};
    float value2{1.0F};
    float value3{1.0F};
    float value4{1.0F};
    float value5{1.0F};
    float value6{1.0F};
    float value7{1.0F};

    __device__ __forceinline__ float get(int32_t index) const
    {
        switch (index)
        {
        case 0: return value0;
        case 1: return value1;
        case 2: return value2;
        case 3: return value3;
        case 4: return value4;
        case 5: return value5;
        case 6: return value6;
        case 7: return value7;
        default: return 1.0F;
        }
    }
};

inline ExpertFloatBatch makeExpertFloatBatch(float const* values, int32_t count)
{
    ExpertFloatBatch result;
    result.value0 = count > 0 && values != nullptr ? values[0] : 1.0F;
    result.value1 = count > 1 && values != nullptr ? values[1] : 1.0F;
    result.value2 = count > 2 && values != nullptr ? values[2] : 1.0F;
    result.value3 = count > 3 && values != nullptr ? values[3] : 1.0F;
    result.value4 = count > 4 && values != nullptr ? values[4] : 1.0F;
    result.value5 = count > 5 && values != nullptr ? values[5] : 1.0F;
    result.value6 = count > 6 && values != nullptr ? values[6] : 1.0F;
    result.value7 = count > 7 && values != nullptr ? values[7] : 1.0F;
    return result;
}

struct ProjectionRow
{
    int32_t row;
    bool second;
};

__device__ __forceinline__ ProjectionRow mapProjectionRow(
    int32_t outputRow, int32_t outputRows, bool paired, bool concatenated)
{
    if (!paired)
    {
        return {outputRow, false};
    }
    if (concatenated)
    {
        int32_t const projectionRows = outputRows / 2;
        bool const second = outputRow >= projectionRows;
        return {second ? outputRow - projectionRows : outputRow, second};
    }
    int32_t constexpr interleaveRows = 64;
    int32_t const local = outputRow % (2 * interleaveRows);
    bool const second = local >= interleaveRows;
    int32_t const row = (outputRow / (2 * interleaveRows)) * interleaveRows + (second ? local - interleaveRows : local);
    return {row, second};
}

__device__ __forceinline__ float decodeFp4(uint8_t nibble)
{
    float magnitude;
    switch (nibble & 0x7U)
    {
    case 0: magnitude = 0.0F; break;
    case 1: magnitude = 0.5F; break;
    case 2: magnitude = 1.0F; break;
    case 3: magnitude = 1.5F; break;
    case 4: magnitude = 2.0F; break;
    case 5: magnitude = 3.0F; break;
    case 6: magnitude = 4.0F; break;
    default: magnitude = 6.0F; break;
    }
    return (nibble & 0x8U) != 0 ? -magnitude : magnitude;
}

__device__ __forceinline__ float roundToBf16(float value)
{
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float decodeFp8(uint8_t value)
{
#if SUPPORTS_FP8
    return static_cast<float>(*reinterpret_cast<__nv_fp8_e4m3 const*>(&value));
#else
    uint32_t const exponent = (value >> 3) & 0xFU;
    uint32_t const mantissa = value & 0x7U;
    if (exponent == 0xFU && mantissa == 0x7U)
    {
        return __int_as_float(0x7FFFFFFF);
    }
    float decoded = exponent == 0
        ? static_cast<float>(mantissa) * (1.0F / 512.0F)
        : ldexpf(1.0F + static_cast<float>(mantissa) * 0.125F, static_cast<int32_t>(exponent) - 7);
    return (value & 0x80U) != 0 ? -decoded : decoded;
#endif
}

__device__ __forceinline__ uint8_t encodeFp8(float value)
{
#if SUPPORTS_FP8
    __nv_fp8_e4m3 const encoded(value);
    return *reinterpret_cast<uint8_t const*>(&encoded);
#else
    uint8_t const sign = static_cast<uint8_t>((__float_as_uint(value) >> 24) & 0x80U);
    if (value != value)
    {
        return sign | 0x7FU;
    }
    float const magnitude = fabsf(value);
    if (magnitude >= 448.0F)
    {
        return sign | 0x7EU;
    }
    if (magnitude < (1.0F / 64.0F))
    {
        int32_t const mantissa = __float2int_rn(magnitude * 512.0F);
        return sign | static_cast<uint8_t>(mantissa >= 8 ? 0x08U : mantissa);
    }

    int32_t exponent;
    float const significand = frexpf(magnitude, &exponent) * 2.0F;
    --exponent;
    int32_t mantissa = __float2int_rn((significand - 1.0F) * 8.0F);
    if (mantissa == 8)
    {
        mantissa = 0;
        ++exponent;
    }
    uint32_t const encoded = static_cast<uint32_t>((exponent + 7) << 3) | static_cast<uint32_t>(mantissa);
    return sign | static_cast<uint8_t>(encoded > 0x7EU ? 0x7EU : encoded);
#endif
}

__device__ __forceinline__ float normalizedBlockScale(uint8_t const* packedBlock, uint8_t sourceScale, float alpha)
{
    float const sourceMultiplier = decodeFp8(sourceScale) * alpha;
    float maxAbs = 0.0F;
#pragma unroll
    for (int32_t byte = 0; byte < 8; ++byte)
    {
        uint8_t const packed = packedBlock[byte];
        float const low = roundToBf16(decodeFp4(packed & 0xFU) * sourceMultiplier);
        float const high = roundToBf16(decodeFp4(packed >> 4) * sourceMultiplier);
        maxAbs = fmaxf(maxAbs, fmaxf(fabsf(low), fabsf(high)));
    }
    return fmaxf(maxAbs / 6.0F, 1.0e-12F);
}

__device__ __forceinline__ uint8_t quantizeFp4(float value, float blockScale)
{
    float const scaled = fminf(fabsf(value / blockScale), 6.0F);
    uint8_t magnitude = 0;
    magnitude += scaled > 0.25F;
    magnitude += scaled > 0.75F;
    magnitude += scaled > 1.25F;
    magnitude += scaled > 1.75F;
    magnitude += scaled > 2.5F;
    magnitude += scaled > 3.5F;
    magnitude += scaled > 5.0F;
    return magnitude | (value < 0.0F ? 0x8U : 0U);
}

__device__ __forceinline__ uint8_t normalizePackedFp4(uint8_t packed, float sourceMultiplier, float blockScale)
{
    float const low = roundToBf16(decodeFp4(packed & 0xFU) * sourceMultiplier);
    float const high = roundToBf16(decodeFp4(packed >> 4) * sourceMultiplier);
    return quantizeFp4(low, blockScale) | static_cast<uint8_t>(quantizeFp4(high, blockScale) << 4);
}

} // namespace detail
} // namespace kernel
} // namespace trt_edgellm
