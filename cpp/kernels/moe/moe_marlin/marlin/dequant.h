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
 * Adapted from https://github.com/vllm-project/vllm/blob/main/csrc/moe/marlin_moe_wna16/
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/* Fast Dequantization (Converting INT4/INT8/FP4/FP8 to FP16/BF16)

The process of fast dequantization can be summarized as a combination
of bitwise operations and floating-point computations:

weight =>(bit_op / bitwise operations)=>
f16_value =>(flop / floating-point computation)=>
dequantized_weight

Since the dequantized weights typically require subtracting the zero point and
applying a scale factor, the floating-point computation step can be fused with
the zero-point subtraction and scaling operations.

The following are the parts that need to be modified for the fused operation
of zero-point subtraction and scaling.

## INT4 => FP16/BF16 or INT8 => FP16

The floating-point computation is `__hsub2`

If has zero points:

    flop(bit_op(weight)) - flop(bit_op(zp))
  = sub(bit_op(weight), bias) - sub(bit_op(zp), bias)
  = bit_op(weight) - bit_op(zp)

so we don't need additional modification.

If has float zero points:

    flop(bit_op(weight)) - fzp
  = sub(bit_op(weight), bias) - fzp
  = bit_op(weight) - (fzp + bias)

where the `fzp + bias` can be computed at weight loading. But this
may have accuracy issue, so we should not use this in most cases.

If has not zero points:

    scale(flop(bit_op(weight)))
  = scale(sub(bit_op(weight), bias))
  = scale(bit_op(weight)) - scale(bias)
  = fma(bit_op(weight), scale_factor, scale(bias))

where the `scale(bias)` can be cached. But this may have accuracy issue,
so we should not use this in most cases.


## INT8 => BF16

INT8 => BF16 is a special case, it use byte_perm instead of flop.
We cannot fused byte_perm with scaling.


## FP4/FP8 => FP16/BF16

    scale(flop(bit_op(weight)))
  = scale(mul(bit_op(weight), multiplier))
  = mul(bit_op(weight), scale_factor * multiplier)

where `scale_factor * multiplier` can be computed at weight loading.

*/

#pragma once
#include "common/cudaMacros.h"

#include "marlin_dtypes.cuh"
#include "nvfp4_scale.cuh"

namespace MARLIN_NAMESPACE_NAME
{

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c)
{
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a)
{
    uint32_t res;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(res) : "r"(a), "n"(start_byte), "n"(mask));
    return res;
}

template <typename scalar_t2, trt_edgellm::marlin_dtypes::ScalarTypeId w_type_id, bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

//
// Efficiently dequantize 4bit values packed in an int32 value into a full
// B-fragment of 4 fp16 values. We mostly follow the strategy in the link below,
// with some small changes:
// - FP16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L215-L287
// - BF16:
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h#L327-L385
//
template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kU4B8.id(), true>(int q, half2* frag_b)
{
    int const MASK = 0x000f000f;
    int const EX = 0x64006400;
    // Guarantee that the `(a & b) | c` operations are LOP3s.
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
    q >>= 4;
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

    frag_b[0] = *reinterpret_cast<half2*>(&lo);
    frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kU4B8.id(), false>(int q, half2* frag_b)
{
    int const LO = 0x000f000f;
    int const HI = 0x00f000f0;
    int const EX = 0x64006400;
    // Guarantee that the `(a & b) | c` operations are LOP3s.
    // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    // clang-format on
    // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
    // directly into `SUB` and `ADD`.
    int const SUB = 0x64086408;
    int const MUL = 0x2c002c00;
    int const ADD = 0xd480d480;
    frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2 const*>(&SUB));
    frag_b[1] = __hfma2(
        *reinterpret_cast<half2*>(&hi), *reinterpret_cast<half2 const*>(&MUL), *reinterpret_cast<half2 const*>(&ADD));
}

template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kU4.id(), true>(int q, half2* frag_b)
{
    dequant<half2, trt_edgellm::marlin_dtypes::kU4B8.id(), true>(q, frag_b);
}

// Edge-LLM: AWQ/modelopt uses zero point = 8 baked into dequant magic numbers
// (same as GPTQ/kU4B8). This follows modelopt semantics where zero point is
// always 8, encoded in SUB=0x64086408 and ADD=0xd480d480 (like NEG_72 in
// dequantize.cuh). No explicit b_zeros tensor needed.
template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kU4.id(), false>(int q, half2* frag_b)
{
    int const LO = 0x000f000f;
    int const HI = 0x00f000f0;
    int const EX = 0x64006400;
    // Guarantee that the `(a & b) | c` operations are LOP3s.
    // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    // clang-format on
    // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
    // directly into `SUB` and `ADD`.
    // Edge-LLM: Use same constants as kU4B8 to bake in zero point = 8
    int const SUB = 0x64086408;
    int const MUL = 0x2c002c00;
    int const ADD = 0xd480d480;
    frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2 const*>(&SUB));
    frag_b[1] = __hfma2(
        *reinterpret_cast<half2*>(&hi), *reinterpret_cast<half2 const*>(&MUL), *reinterpret_cast<half2 const*>(&ADD));
}

template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kU4B8.id(), true>(int q, nv_bfloat162* frag_b)
{
    static constexpr uint32_t MASK = 0x000f000f;
    static constexpr uint32_t EX = 0x43004300;

    // Guarantee that the `(a & b) | c` operations are LOP3s.
    // clang-format off
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
    // clang-format on

    frag_b[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
    frag_b[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
}

template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kU4B8.id(), false>(int q, nv_bfloat162* frag_b)
{
    dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kU4B8.id(), true>(q, frag_b);

    static constexpr uint32_t SUB = 0x43084308;

    frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<nv_bfloat162 const*>(&SUB));
    frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<nv_bfloat162 const*>(&SUB));
}

template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kU4.id(), true>(int q, nv_bfloat162* frag_b)
{
    dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kU4B8.id(), true>(q, frag_b);
}

// Edge-LLM: AWQ/modelopt BF16 dequant with zero point = 8 baked in
template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kU4.id(), false>(int q, nv_bfloat162* frag_b)
{
    dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kU4.id(), true>(q, frag_b);

    // Edge-LLM: Use same SUB constant as kU4B8 to bake in zero point = 8
    static constexpr uint32_t SUB = 0x43084308;

    frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<nv_bfloat162 const*>(&SUB));
    frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<nv_bfloat162 const*>(&SUB));
}

template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kFE4M3fn.id(), true>(int q, half2* frag_b)
{
    // Constants for FP8 (E4M3) and FP16 formats
    constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;
    constexpr int RIGHT_SHIFT = FP16_EXPONENT - FP8_EXPONENT;
    constexpr int MASK = 0x7F007F00;

    // Extract and shift FP8 values to FP16 format
    int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 8;
    int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<half2 const*>(&Out1);
    frag_b[0] = *reinterpret_cast<half2 const*>(&Out2);
}

template <>
__device__ inline void dequant<half2, trt_edgellm::marlin_dtypes::kFE4M3fn.id(), false>(int q, half2* frag_b)
{
    dequant<half2, trt_edgellm::marlin_dtypes::kFE4M3fn.id(), true>(q, frag_b);

    // Constants for FP8 (E4M3) and FP16 formats
    constexpr int FP8_EXPONENT = 4, FP16_EXPONENT = 5;

    // Construct and apply exponent bias
    constexpr int BIAS_OFFSET = (1 << (FP16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
    half2 const bias_reg = __float2half2_rn(float(1 << BIAS_OFFSET));

    // Convert to half2 and apply bias
    frag_b[1] = __hmul2(frag_b[1], bias_reg);
    frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kFE4M3fn.id(), true>(
    int q, nv_bfloat162* frag_b)
{
    // Constants for FP8 (E4M3) and BF16 formats
    constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
    constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;

    constexpr int MASK = 0x7F007F00;

    // Extract and shift FP8 values to BF16 format
    int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 8;
    int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<nv_bfloat162 const*>(&Out1);
    frag_b[0] = *reinterpret_cast<nv_bfloat162 const*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kFE4M3fn.id(), false>(
    int q, nv_bfloat162* frag_b)
{
    dequant<nv_bfloat162, trt_edgellm::marlin_dtypes::kFE4M3fn.id(), true>(q, frag_b);

    // Constants for FP8 (E4M3) and BF16 formats
    constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;

    // Construct and apply exponent bias
    constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
    // Add 127 (float exponent bias) to BIAS_OFFSET and shift to float exponent
    // position
    constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
    nv_bfloat162 const bias_reg = __float2bfloat162_rn(*reinterpret_cast<float const*>(&BIAS));

    // Convert to bfloat162 and apply bias
    frag_b[1] = __hmul2(frag_b[1], bias_reg);
    frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

#if SUPPORTS_FP8

template <>
__device__ inline void dequant<__nv_fp8x4_e4m3, trt_edgellm::marlin_dtypes::kU4B8.id(), true>(
    int q, __nv_fp8x4_e4m3* frag_b)
{
    int s = q & 0x08080808;
    int Out1 = ((q & 0x07070707) | (s << 4)) + (s >> 3);
    q >>= 4;
    s = q & 0x08080808;
    int Out2 = ((q & 0x07070707) | (s << 4)) + (s >> 3);

    frag_b[0] = *reinterpret_cast<__nv_fp8x4_e4m3 const*>(&Out1);
    frag_b[1] = *reinterpret_cast<__nv_fp8x4_e4m3 const*>(&Out2);
}

template <typename scalar_t2, trt_edgellm::marlin_dtypes::ScalarTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int32_t const q, scalar_t2* frag_b);

template <>
__device__ inline void dequant_fp8_scales<half2, trt_edgellm::marlin_dtypes::kFE4M3fn.id()>(
    int32_t const q, half2* frag_b)
{
    uint32_t const scales = static_cast<uint32_t>(q);
    constexpr uint32_t kPAIR_SCALE_BYTES{0x3120U};
    // Reorder [b0, b1, b2, b3] to [b0, b2, b1, b3] so each packed
    // conversion produces one Marlin FP16 scale fragment.
    uint32_t const pairedScales = __byte_perm(scales, scales, kPAIR_SCALE_BYTES);
    uint32_t const out2 = trt_edgellm::kernel::nvfp4::e4m3x2ToFloat16x2Times128(static_cast<uint16_t>(pairedScales));
    uint32_t const out1
        = trt_edgellm::kernel::nvfp4::e4m3x2ToFloat16x2Times128(static_cast<uint16_t>(pairedScales >> 16));

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<half2 const*>(&out1);
    frag_b[0] = *reinterpret_cast<half2 const*>(&out2);
};

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, trt_edgellm::marlin_dtypes::kFE4M3fn.id()>(
    int32_t const q, nv_bfloat162* frag_b)
{
    uint32_t const scales = static_cast<uint32_t>(q);
    constexpr uint32_t kPAIR_SCALE_BYTES{0x3120U};
    // Reorder [b0, b1, b2, b3] to [b0, b2, b1, b3] so each packed
    // conversion produces one Marlin BF16 scale fragment.
    uint32_t const pairedScales = __byte_perm(scales, scales, kPAIR_SCALE_BYTES);
    uint32_t const out2 = trt_edgellm::kernel::nvfp4::e4m3x2ToBfloat16x2Times128(static_cast<uint16_t>(pairedScales));
    uint32_t const out1
        = trt_edgellm::kernel::nvfp4::e4m3x2ToBfloat16x2Times128(static_cast<uint16_t>(pairedScales >> 16));

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<nv_bfloat162 const*>(&out1);
    frag_b[0] = *reinterpret_cast<nv_bfloat162 const*>(&out2);
}

#endif

} // namespace MARLIN_NAMESPACE_NAME
