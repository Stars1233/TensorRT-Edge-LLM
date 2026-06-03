# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVFP4 MoE decode GEMV kernel for SM100/SM110 (CuTe DSL).

W4A16 NVFP4 format (N-major):
  weights  [E, K, N/2] uint8 packed FP4 E2M1x2 (N-paired: lo nibble = even N, hi = odd N)
  scales   [E, padUp(N,128), padUp(K/16,4)] FP8 E4M3 atom-layout block scales (prefill layout)
  global_scales [E] FP32 per-expert global scale
  Dequant: weight_f32 = fp4_to_f32(nibble) * fp8_to_f32(block_scale) * global_scale

Optimized scalar GEMV for MoE decode with multi-token support.

Key optimizations:
  - K-parallel: K_WARPS threads share the K-reduction, reducing loop iterations
  - Vectorized loads: 4 weight bytes loaded as uint32 per thread per iteration
  - Shared-memory reduction: partial sums reduced across K_WARPS via smem

Thread mapping (TILE_N=128, VEC_N=4, K_WARPS=16):
  BLOCK_SIZE = (TILE_N / VEC_N) * K_WARPS = 32 * 16 = 512 threads
  tidx = threadIdx.x % 32    → N position (0..31)
  kw   = threadIdx.x / 32    → K chunk    (0..15)
  Each thread computes VEC_N=4 output elements over K_half/K_WARPS iterations.

Composable compile-time specialization via orthogonal parameters:
  activation: "none", "relu2", "silu" — applied to input activations
  gated: False/True — when True, reads two input buffers: act(gate) * up
  output_atomic: False/True — fp16 direct store vs score-weighted fp16 atomicAdd

Pipeline configurations:
  Nemotron (relu2):  up(none,F,F) → down(relu2,F,T)
  Mixtral  (silu):   up(none,F,F) → down(silu,F,T)
  LLaMA SwiGLU:      2×up(none,F,F) → down(silu,T,T)

Weight layout [E, K/2, N] is shared with the N-major prefill GEMM kernel —
no weight duplication needed between prefill and decode paths.
"""

from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

TILE_N = 128  # output elements per CTA
VEC_N = 4  # output elements per thread (vectorized weight load)
K_WARPS = 16  # K-parallel factor (threads sharing K reduction)
N_THREADS = TILE_N // VEC_N  # = 32 threads along N dimension
BLOCK_SIZE = N_THREADS * K_WARPS  # = 512 threads per CTA
TILE_N_PAD = TILE_N + 4  # padded stride to avoid smem bank conflicts
SF_VEC_SIZE = 16  # NVFP4 block scale: 1 scale per 16 K elements
SMEM_FLOATS = K_WARPS * TILE_N_PAD  # reduction buffer size in f32 elements

# ============================================================================
# PTX intrinsics
# ============================================================================



@dsl_user_op
def _get_ptr_as_int64(
    tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None
) -> Int64:
    """Get the memory address of tensor[offset] as Int64.

    WARNING: Uses ptrtoint which strips address space information.
    For SMEM tensors use _get_smem_ptr_as_int32 instead.
    """
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def _get_smem_ptr_as_int32(
    tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None
) -> Int32:
    """Get shared-memory byte address of tensor[offset] as Int32.

    Uses Pointer.toint() which preserves the SMEM address space (addrspace 3),
    returning a 32-bit address suitable for st.shared.*/ld.shared.* instructions.
    """
    elem_ptr = tensor.iterator + Int32(offset)
    return elem_ptr.toint(loc=loc, ip=ip)


@dsl_user_op
def _ld_global_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load single byte from global memory, zero-extended to u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_u16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load u16 from global memory, zero-extended to u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load u32 from global memory (4 packed weight bytes)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_v2u32(addr: Int64, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Load 8 bytes as 2 x u32 (4 packed fp16 scales)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip))
    hi = Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip))
    return lo, hi


@dsl_user_op
def _u32_lo_f16_to_f32(packed: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract low fp16 from u32 (bits [15:0]) and convert to f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            """{
                .reg .f16 lo, hi;
                mov.b32 {lo, hi}, $1;
                cvt.f32.f16 $0, lo;
            }""",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _u32_hi_f16_to_f32(packed: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract high fp16 from u32 (bits [31:16]) and convert to f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            """{
                .reg .f16 lo, hi;
                mov.b32 {lo, hi}, $1;
                cvt.f32.f16 $0, hi;
            }""",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )



@dsl_user_op
def _ld_global_f16_as_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """Load fp16 from global memory and convert to f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            """{
                .reg .f16 tmp;
                ld.global.b16 tmp, [$1];
                cvt.f32.f16 $0, tmp;
            }""",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_1xfp8e4m3_to_f32(
    byte_val: Uint32, *, loc=None, ip=None
) -> Float32:
    """Convert a single FP8 E4M3 byte (in low 8 bits of u32) to float32.

    Packs the byte into a b16 (duplicated) then uses cvt.rn.f16x2.e4m3x2
    and extracts the low f16, converting to f32. Requires SM >= 89.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(byte_val).ir_value(loc=loc, ip=ip)],
            """{
                .reg .b16 pair;
                .reg .b32 h2;
                .reg .f16 flo, fhi;
                cvt.u16.u32 pair, $1;
                cvt.rn.f16x2.e4m3x2 h2, pair;
                mov.b32 {flo, fhi}, h2;
                cvt.f32.f16 $0, flo;
            }""",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_4xfp8e4m3_to_4xf32(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Convert 4 packed FP8 E4M3 bytes (u32) to 4 float32 values.

    Splits u32 into two b16 halves, converts each pair of E4M3 values
    to f16x2 via cvt.rn.f16x2.e4m3x2, then converts to f32.
    Requires SM >= 89.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """{
            .reg .b16 pair_lo, pair_hi;
            .reg .b32 h2_lo, h2_hi;
            .reg .f16 f0, f1, f2, f3;
            mov.b32 {pair_lo, pair_hi}, $4;
            cvt.rn.f16x2.e4m3x2 h2_lo, pair_lo;
            cvt.rn.f16x2.e4m3x2 h2_hi, pair_hi;
            mov.b32 {f0, f1}, h2_lo;
            mov.b32 {f2, f3}, h2_hi;
            cvt.f32.f16 $0, f0;
            cvt.f32.f16 $1, f1;
            cvt.f32.f16 $2, f2;
            cvt.f32.f16 $3, f3;
        }""",
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    s0 = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    s1 = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    s2 = Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    s3 = Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return s0, s1, s2, s3


@dsl_user_op
def _ld_global_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """Load float32 from global memory."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Load int32 from global memory."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_e2m1x2_to_f32x2(
    packed_byte: Uint32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert packed NVFP4 E2M1x2 byte to two float32 values.

    Uses PTX cvt.f16x2.e2m1x2 (requires SM >= 100).
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed_byte).ir_value(loc=loc, ip=ip)],
        """{
            .reg .f16 lo, hi;
            .reg .b32 h2;
            .reg .b8 bv;
            cvt.u8.u32 bv, $2;
            cvt.rn.f16x2.e2m1x2 h2, bv;
            mov.b32 {lo, hi}, h2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }""",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    f0 = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    f1 = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    return f0, f1



@dsl_user_op
def _st_global_f16_from_f32(
    addr: Int64, val: Float32, *, loc=None, ip=None
):
    """Convert f32 to fp16 and store to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .f16 tmp;
            cvt.rn.f16.f32 tmp, $1;
            st.global.b16 [$0], tmp;
        }""",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atomic_add_global_f32(
    addr: Int64, val: Float32, *, loc=None, ip=None
):
    """Atomic f32 add to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _silu_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """Clamped SiLU: x * sigmoid(x), via PTX ex2.approx + rcp.approx.

    Clamps input to [-50, 50] to avoid exp overflow.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            """{
                .reg .f32 xc, neg_xc, l2e_neg, exp_neg, denom, inv_denom;
                max.f32 xc, $1, 0fC2480000;
                min.f32 xc, xc, 0f42480000;
                neg.f32 neg_xc, xc;
                mul.f32 l2e_neg, neg_xc, 0f3FB8AA3B;
                ex2.approx.f32 exp_neg, l2e_neg;
                add.f32 denom, exp_neg, 0f3F800000;
                rcp.approx.f32 inv_denom, denom;
                mul.f32 $0, xc, inv_denom;
            }""",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atomic_add_global_f16_from_f32(
    addr: Int64, val: Float32, *, loc=None, ip=None
):
    """Convert f32 to fp16, then fp16 atomicAdd to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .f16 tmp;
            cvt.rn.f16.f32 tmp, $1;
            red.global.add.noftz.f16 [$0], tmp;
        }""",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fmax_f32(
    a: Float32, b: Float32, *, loc=None, ip=None
) -> Float32:
    """PTX max.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmaf_f32(
    a: Float32, b: Float32, c: Float32, *, loc=None, ip=None
) -> Float32:
    """PTX fma.rn.f32  →  a * b + c."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_shared_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Store f32 to shared memory at 32-bit smem address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load f32 from shared memory at 32-bit smem address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ============================================================================
# Shared memory storage
# ============================================================================


@cute.struct
class SmemStorage:
    """Shared memory for K-parallel reduction.

    Layout: [K_WARPS][TILE_N_PAD] float32 with padding to avoid bank conflicts.
    Total: K_WARPS * TILE_N_PAD * 4 = 16 * 132 * 4 = 8448 bytes.
    """
    red_buf: cute.struct.MemRange[cutlass.Float32, SMEM_FLOATS]


# ============================================================================
# Kernel class
# ============================================================================


SMEM_FLOATS_SWIGLU = 2 * K_WARPS * TILE_N_PAD  # doubled: gate + up partial sums


@cute.struct
class SmemStorageSwiglu:
    """Doubled shared memory for SwiGLU K-parallel reduction.

    Layout: [2][K_WARPS][TILE_N_PAD] float32 — gate partials then up partials.
    Total: 2 * K_WARPS * TILE_N_PAD * 4 = 2 * 16 * 132 * 4 = 16896 bytes.
    """
    red_buf: cute.struct.MemRange[cutlass.Float32, SMEM_FLOATS_SWIGLU]


class NvFP4MoeGemvSwigluKernel:
    """Fused gate+up NVFP4 MoE decode GEMV kernel with SwiGLU epilogue.

    Reads interleaved FC1 weight [E, K/2, 2*N_out] where columns [0,N_out) are
    gate_proj and columns [N_out,2*N_out) are up_proj. Produces
    output[n] = SiLU(gate_sum[n]) * up_sum[n] — one kernel replaces separate
    gate_proj + up_proj launches.

    Input is indexed by [token_idx] (projection mode, not reduce mode).
    Output is direct fp16 store (not atomic) to [numTokens*topK, N_out].
    input2 and gate_scores are unused (pass-through for signature compatibility).
    """

    def __init__(self):
        pass

    @cute.jit
    def __call__(
        self,
        input_t: cute.Tensor,          # [numTokens, K] fp16
        input2_t: cute.Tensor,         # unused (signature compat)
        weights_t: cute.Tensor,        # [total_weight_bytes] uint8 flat
        scales_t: cute.Tensor,         # [total_scale_bytes] fp8 e4m3 (uint8) flat
        global_scales_t: cute.Tensor,  # [E] fp32 per-expert global scales
        expert_ids_t: cute.Tensor,     # [numTokens * TopK] int32
        gate_scores_t: cute.Tensor,    # unused (signature compat)
        output_t: cute.Tensor,         # [numTokens * topK * N_out] fp16 flat
        K: cutlass.Int32,              # reduction dim (full, even, >=2)
        N_out: cutlass.Int32,          # output dim = I (half of weight N)
        topK: cutlass.Int32,           # number of topK experts per token
        numTokens: cutlass.Int32,      # number of tokens (batch dim)
        stream: "cuda.CUstream",
    ):
        """Host-side launch wrapper."""
        grid_x = (N_out + Int32(TILE_N - 1)) // Int32(TILE_N)
        grid_y = topK
        grid_z = numTokens

        K_half = K // Int32(2)
        N_full = N_out * Int32(2)

        self.kernel(
            input_t,
            input2_t,
            weights_t,
            scales_t,
            global_scales_t,
            expert_ids_t,
            gate_scores_t,
            output_t,
            K,
            K_half,
            N_out,
            N_full,
            topK,
            numTokens,
        ).launch(
            grid=[grid_x, grid_y, grid_z],
            block=[BLOCK_SIZE, 1, 1],
            cluster=[1, 1, 1],
            smem=SmemStorageSwiglu.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_t: cute.Tensor,          # [numTokens, K] fp16 flat
        input2_t: cute.Tensor,         # unused
        weights_t: cute.Tensor,        # flat uint8
        scales_t: cute.Tensor,         # flat fp8 e4m3 (uint8) — atom layout
        global_scales_t: cute.Tensor,  # [E] fp32 per-expert global scales
        expert_ids_t: cute.Tensor,     # flat int32
        gate_scores_t: cute.Tensor,    # unused
        output_t: cute.Tensor,         # flat fp16
        K: Int32,
        K_half: Int32,
        N_out: Int32,
        N_full: Int32,
        topK: Int32,
        numTokens: Int32,
    ):
        """Device kernel — fused gate+up with SwiGLU epilogue.

        Grid: (ceil(N_out/128), TopK, numTokens)
        Block: 512 threads (32 N-threads x 16 K-warps)
        Smem: 16896 bytes (2x reduction buffer for gate + up)
        """
        # ---- Shared memory allocation ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SmemStorageSwiglu)
        red_layout = cute.make_layout(SMEM_FLOATS_SWIGLU, stride=1)
        red_t = storage.red_buf.get_tensor(red_layout)

        # ---- Thread / block indices ----
        tid, _, _ = cute.arch.thread_idx()
        tidx = tid % Int32(N_THREADS)     # 0..31: N-position index
        kw = tid // Int32(N_THREADS)      # 0..15: K-chunk index

        bx, k_slot, token_idx = cute.arch.block_idx()
        n_base = bx * Int32(TILE_N) + tidx * Int32(VEC_N)

        n_local = tidx * Int32(VEC_N)  # 0, 4, 8, ..., 124

        # Shared memory base address (32-bit, preserves addrspace 3)
        smem_base = _get_smem_ptr_as_int32(red_t, Int32(0))
        # Gate partials: offset 0; Up partials: offset SMEM_FLOATS (K_WARPS * TILE_N_PAD)
        smem_up_offset = Int32(SMEM_FLOATS * 4)  # byte offset to up section

        if n_base < N_out:
            # ---- Expert / pointer setup ----
            eids_base = _get_ptr_as_int64(expert_ids_t, Int32(0))
            expert_id = _ld_global_i32(
                eids_base + Int64(token_idx * topK + k_slot) * Int64(4)
            )

            input_base = _get_ptr_as_int64(input_t, Int32(0))
            weights_base = _get_ptr_as_int64(weights_t, Int32(0))
            scales_base = _get_ptr_as_int64(scales_t, Int32(0))

            # Expert weight base: expert * K_half * N_full = expert * K * N_out bytes
            expert_w_offset = Int64(expert_id) * Int64(K_half) * Int64(N_full)

            # Prefill atom-layout scale stride (M=N_full, K_sf=K/16)
            num_sf_cols = K >> Int32(4)
            padded_sf_cols = ((num_sf_cols + Int32(3)) >> Int32(2)) << Int32(2)
            padded_m = ((N_full + Int32(127)) >> Int32(7)) << Int32(7)
            sf_bytes_per_ex = Int64(padded_m) * Int64(padded_sf_cols)
            atom_expert_stride = ((sf_bytes_per_ex + Int64(3)) >> Int64(2)) << Int64(2)
            expert_s_offset = Int64(expert_id) * atom_expert_stride

            # Per-expert global scale (FP32)
            g_base = _get_ptr_as_int64(global_scales_t, Int32(0))
            g_scale = _ld_global_f32(g_base + Int64(expert_id) * Int64(4))

            # Input row base: projection mode (indexed by token_idx)
            input_row_base = input_base + Int64(token_idx) * Int64(K) * Int64(2)

            # ---- K-parallel chunk assignment ----
            chunk = (K_half + Int32(K_WARPS - 1)) // Int32(K_WARPS)
            kb_start = kw * chunk
            kb_end = (kw + Int32(1)) * chunk
            if kb_end > K_half:
                kb_end = K_half

            # ---- Main dot-product loop (dual accumulation) ----
            gate_acc0 = Float32(0.0)
            gate_acc1 = Float32(0.0)
            gate_acc2 = Float32(0.0)
            gate_acc3 = Float32(0.0)
            up_acc0 = Float32(0.0)
            up_acc1 = Float32(0.0)
            up_acc2 = Float32(0.0)
            up_acc3 = Float32(0.0)

            # N-major addressing for interleaved (up, gate) weights at 32-col
            # granularity along the N axis — matches the layout the CuTeDSL
            # SwiGLU prefill kernel expects (see
            # `kernelSrcs/nvfp4_moe_cutedsl/README.md` SwiGLU section and the
            # repack at
            # `tensorrt_edgellm/checkpoint/repacking.py`
            # `repack_nvfp4_qwen3_moe_experts_thor`).
            #
            # For output position p (in [0, N_out)) within a 32-col chunk:
            #   chunk_idx = p >> 5
            #   in_chunk_byte = (p & 31) >> 1
            #   up_byte_in_buffer   = chunk_idx * 32 + in_chunk_byte
            #   gate_byte_in_buffer = up_byte_in_buffer + 16
            # i.e. each 32-byte chunk holds the up-proj half in bytes [0,16)
            # and the gate-proj half in bytes [16,32).
            #
            # `n_half_addr` points to the up_proj byte for the current thread's
            # n_base; `up_byte_off = 16` steps from up to gate within the chunk.
            N_half_full = N_full >> Int32(1)
            n_half_addr = Int64(n_base >> Int32(1)) + (Int64(n_base >> Int32(5)) << Int64(4))
            up_byte_off = Int64(16)  # within-chunk byte step from up half to gate half

            kb = kb_start
            while kb < kb_end:
                # K-pair base: K_half * N_full bytes per expert.
                w_pair_base = (
                    weights_base + expert_w_offset
                    + Int64(kb) * Int64(N_full)
                )

                # ---- Input load (2 fp16 elements per kb) ----
                h_offset = Int64(kb) * Int64(4)  # 2*kb fp16 elements x 2 bytes
                a0 = _ld_global_f16_as_f32(input_row_base + h_offset)
                a1 = _ld_global_f16_as_f32(input_row_base + h_offset + Int64(2))

                # ---- Up weight loads: lower half of each 32-byte chunk ----
                up_k0 = _ld_global_u16(w_pair_base + n_half_addr)
                up_k1 = _ld_global_u16(w_pair_base + Int64(N_half_full) + n_half_addr)
                ub0_k0 = up_k0 & Uint32(0xFF)
                ub1_k0 = (up_k0 >> Uint32(8)) & Uint32(0xFF)
                ub0_k1 = up_k1 & Uint32(0xFF)
                ub1_k1 = (up_k1 >> Uint32(8)) & Uint32(0xFF)

                # ---- Gate weight loads: upper half of each 32-byte chunk ----
                gate_k0 = _ld_global_u16(w_pair_base + n_half_addr + up_byte_off)
                gate_k1 = _ld_global_u16(w_pair_base + Int64(N_half_full) + n_half_addr + up_byte_off)
                gb0_k0 = gate_k0 & Uint32(0xFF)
                gb1_k0 = (gate_k0 >> Uint32(8)) & Uint32(0xFF)
                gb0_k1 = gate_k1 & Uint32(0xFF)
                gb1_k1 = (gate_k1 >> Uint32(8)) & Uint32(0xFF)

                # ---- Prefill atom-layout scales (M=N_full, K_sf=K/16) ----
                # SF stores one byte per N column. With the interleaved layout
                # the up-proj scale for output position n_base lives at
                # M = chunk_idx*64 + (n_base & 31), and the matching gate-proj
                # scale at M = up_M + 32. n_base is aligned to VEC_N=4, and the
                # kernel rounds M down to multiples of 16 inside the atom
                # offset computation, so (n_base & 31) collapses to
                # (n_base & 16) — i.e. 0 if n_base%32<16 else 16.
                sf_col = Int32(kb) >> Int32(3)
                inner_k = sf_col & Int32(3)
                k_tile = sf_col >> Int32(2)
                num_k_tiles = (num_sf_cols + Int32(3)) >> Int32(2)

                # Up scale: M_up = (chunk_idx << 6) | (n_base & 16)
                up_leader = (((n_base >> Int32(5)) << Int32(6))
                              | (n_base & Int32(16)))
                up_inner_m = (up_leader & Int32(127)) >> Int32(5)
                up_outer_m = up_leader & Int32(31)
                up_m_tile = up_leader >> Int32(7)
                up_atom_off = (
                    Int64(up_m_tile) * Int64(num_k_tiles) * Int64(512)
                    + Int64(k_tile) * Int64(512)
                    + Int64(up_outer_m) * Int64(16)
                    + Int64(up_inner_m) * Int64(4)
                    + Int64(inner_k)
                )
                up_sf_byte = _ld_global_u8(scales_base + expert_s_offset + up_atom_off)
                up_sc = _cvt_1xfp8e4m3_to_f32(up_sf_byte) * g_scale

                # Gate scale: M_gate = M_up + 32
                gate_leader = up_leader + Int32(32)
                gate_inner_m = (gate_leader & Int32(127)) >> Int32(5)
                gate_outer_m = gate_leader & Int32(31)
                gate_m_tile = gate_leader >> Int32(7)
                gate_atom_off = (
                    Int64(gate_m_tile) * Int64(num_k_tiles) * Int64(512)
                    + Int64(k_tile) * Int64(512)
                    + Int64(gate_outer_m) * Int64(16)
                    + Int64(gate_inner_m) * Int64(4)
                    + Int64(inner_k)
                )
                gate_sf_byte = _ld_global_u8(scales_base + expert_s_offset + gate_atom_off)
                gate_sc = _cvt_1xfp8e4m3_to_f32(gate_sf_byte) * g_scale

                # ---- Decode N-major gate FP4 and accumulate ----
                # b0_kX: lo=w(kX, n_base), hi=w(kX, n_base+1)
                # b1_kX: lo=w(kX, n_base+2), hi=w(kX, n_base+3)
                gw0_k0_lo, gw0_k0_hi = _cvt_e2m1x2_to_f32x2(gb0_k0)
                gw1_k0_lo, gw1_k0_hi = _cvt_e2m1x2_to_f32x2(gb1_k0)
                gw0_k1_lo, gw0_k1_hi = _cvt_e2m1x2_to_f32x2(gb0_k1)
                gw1_k1_lo, gw1_k1_hi = _cvt_e2m1x2_to_f32x2(gb1_k1)

                gw0_k0_lo = gw0_k0_lo * gate_sc
                gw0_k0_hi = gw0_k0_hi * gate_sc
                gw1_k0_lo = gw1_k0_lo * gate_sc
                gw1_k0_hi = gw1_k0_hi * gate_sc
                gw0_k1_lo = gw0_k1_lo * gate_sc
                gw0_k1_hi = gw0_k1_hi * gate_sc
                gw1_k1_lo = gw1_k1_lo * gate_sc
                gw1_k1_hi = gw1_k1_hi * gate_sc

                gate_acc0 = _fmaf_f32(a0, gw0_k0_lo, _fmaf_f32(a1, gw0_k1_lo, gate_acc0))
                gate_acc1 = _fmaf_f32(a0, gw0_k0_hi, _fmaf_f32(a1, gw0_k1_hi, gate_acc1))
                gate_acc2 = _fmaf_f32(a0, gw1_k0_lo, _fmaf_f32(a1, gw1_k1_lo, gate_acc2))
                gate_acc3 = _fmaf_f32(a0, gw1_k0_hi, _fmaf_f32(a1, gw1_k1_hi, gate_acc3))

                # ---- Decode N-major up FP4 and accumulate ----
                uw0_k0_lo, uw0_k0_hi = _cvt_e2m1x2_to_f32x2(ub0_k0)
                uw1_k0_lo, uw1_k0_hi = _cvt_e2m1x2_to_f32x2(ub1_k0)
                uw0_k1_lo, uw0_k1_hi = _cvt_e2m1x2_to_f32x2(ub0_k1)
                uw1_k1_lo, uw1_k1_hi = _cvt_e2m1x2_to_f32x2(ub1_k1)

                uw0_k0_lo = uw0_k0_lo * up_sc
                uw0_k0_hi = uw0_k0_hi * up_sc
                uw1_k0_lo = uw1_k0_lo * up_sc
                uw1_k0_hi = uw1_k0_hi * up_sc
                uw0_k1_lo = uw0_k1_lo * up_sc
                uw0_k1_hi = uw0_k1_hi * up_sc
                uw1_k1_lo = uw1_k1_lo * up_sc
                uw1_k1_hi = uw1_k1_hi * up_sc

                up_acc0 = _fmaf_f32(a0, uw0_k0_lo, _fmaf_f32(a1, uw0_k1_lo, up_acc0))
                up_acc1 = _fmaf_f32(a0, uw0_k0_hi, _fmaf_f32(a1, uw0_k1_hi, up_acc1))
                up_acc2 = _fmaf_f32(a0, uw1_k0_lo, _fmaf_f32(a1, uw1_k1_lo, up_acc2))
                up_acc3 = _fmaf_f32(a0, uw1_k0_hi, _fmaf_f32(a1, uw1_k1_hi, up_acc3))

                kb = kb + Int32(1)

            # ---- Store gate partial sums to smem (first half) ----
            smem_byte_off = (kw * Int32(TILE_N_PAD) + n_local) * Int32(4)
            _st_shared_f32(smem_base + smem_byte_off, gate_acc0)
            _st_shared_f32(smem_base + smem_byte_off + Int32(4), gate_acc1)
            _st_shared_f32(smem_base + smem_byte_off + Int32(8), gate_acc2)
            _st_shared_f32(smem_base + smem_byte_off + Int32(12), gate_acc3)

            # ---- Store up partial sums to smem (second half) ----
            _st_shared_f32(smem_base + smem_up_offset + smem_byte_off, up_acc0)
            _st_shared_f32(smem_base + smem_up_offset + smem_byte_off + Int32(4), up_acc1)
            _st_shared_f32(smem_base + smem_up_offset + smem_byte_off + Int32(8), up_acc2)
            _st_shared_f32(smem_base + smem_up_offset + smem_byte_off + Int32(12), up_acc3)

        # All threads must reach the barrier
        cute.arch.sync_threads()

        # ---- Reduction + SwiGLU epilogue: warp 0 sums across K_WARPS ----
        if n_base < N_out:
            if kw == Int32(0):
                gate_sum0 = Float32(0.0)
                gate_sum1 = Float32(0.0)
                gate_sum2 = Float32(0.0)
                gate_sum3 = Float32(0.0)
                up_sum0 = Float32(0.0)
                up_sum1 = Float32(0.0)
                up_sum2 = Float32(0.0)
                up_sum3 = Float32(0.0)

                for i in cutlass.range_constexpr(K_WARPS):
                    rd_off = (Int32(i * TILE_N_PAD) + n_local) * Int32(4)
                    gate_sum0 = gate_sum0 + _ld_shared_f32(smem_base + rd_off)
                    gate_sum1 = gate_sum1 + _ld_shared_f32(smem_base + rd_off + Int32(4))
                    gate_sum2 = gate_sum2 + _ld_shared_f32(smem_base + rd_off + Int32(8))
                    gate_sum3 = gate_sum3 + _ld_shared_f32(smem_base + rd_off + Int32(12))
                    up_sum0 = up_sum0 + _ld_shared_f32(smem_base + smem_up_offset + rd_off)
                    up_sum1 = up_sum1 + _ld_shared_f32(smem_base + smem_up_offset + rd_off + Int32(4))
                    up_sum2 = up_sum2 + _ld_shared_f32(smem_base + smem_up_offset + rd_off + Int32(8))
                    up_sum3 = up_sum3 + _ld_shared_f32(smem_base + smem_up_offset + rd_off + Int32(12))

                # SwiGLU: output = SiLU(gate) * up
                out0 = _silu_f32(gate_sum0) * up_sum0
                out1 = _silu_f32(gate_sum1) * up_sum1
                out2 = _silu_f32(gate_sum2) * up_sum2
                out3 = _silu_f32(gate_sum3) * up_sum3

                # Direct fp16 store: output[token_idx * topK + k_slot, n]
                out_base = _get_ptr_as_int64(output_t, Int32(0))
                out_row = out_base + Int64(token_idx * topK + k_slot) * Int64(N_out) * Int64(2)
                _st_global_f16_from_f32(
                    out_row + Int64(n_base) * Int64(2), out0
                )
                _st_global_f16_from_f32(
                    out_row + Int64(n_base + Int32(1)) * Int64(2), out1
                )
                _st_global_f16_from_f32(
                    out_row + Int64(n_base + Int32(2)) * Int64(2), out2
                )
                _st_global_f16_from_f32(
                    out_row + Int64(n_base + Int32(3)) * Int64(2), out3
                )


class NvFP4MoeGemvKernel:
    """Optimized NVFP4 MoE decode GEMV kernel.

    K-parallel vectorized dot-product for MoE decode with multi-token support.
    Each CTA computes TILE_N=128 output elements using 512 threads
    (32 N-threads x 16 K-warps).

    Composable compile-time specialization:
      activation: "none", "relu2", "silu" — applied to input activations
      gated: when True, reads two input buffers: act(gate) * up
      output_atomic: False = fp16 direct store; True = score-weighted fp16 atomicAdd
    """

    def __init__(
        self,
        activation: str = "none",
        gated: bool = False,
        output_atomic: bool = False,
    ):
        assert activation in ("none", "relu2", "silu"), f"Unknown activation: {activation}"
        self._activation = activation
        self._gated = gated
        self._output_atomic = output_atomic
        self._apply_relu2 = activation == "relu2"
        self._apply_silu = activation == "silu"

    @cute.jit
    def __call__(
        self,
        input_t: cute.Tensor,          # [rows, K] fp16
        input2_t: cute.Tensor,         # [rows, K] fp16 (second input for gated; ignored if not gated)
        weights_t: cute.Tensor,        # [total_weight_bytes] uint8 flat
        scales_t: cute.Tensor,         # [total_scale_bytes] fp8 e4m3 (uint8) flat
        global_scales_t: cute.Tensor,  # [E] fp32 per-expert global scales
        expert_ids_t: cute.Tensor,     # [numTokens * TopK] int32
        gate_scores_t: cute.Tensor,    # [numTokens * TopK] fp32
        output_t: cute.Tensor,         # [output_elems] fp16 flat
        K: cutlass.Int32,              # reduction dim (full, even, >=2)
        N: cutlass.Int32,              # output dim
        topK: cutlass.Int32,           # number of topK experts per token
        numTokens: cutlass.Int32,      # number of tokens (batch dim)
        stream: "cuda.CUstream",
    ):
        """Host-side launch wrapper."""
        grid_x = (N + Int32(TILE_N - 1)) // Int32(TILE_N)
        grid_y = topK
        grid_z = numTokens

        K_half = K // Int32(2)

        self.kernel(
            input_t,
            input2_t,
            weights_t,
            scales_t,
            global_scales_t,
            expert_ids_t,
            gate_scores_t,
            output_t,
            K,
            K_half,
            N,
            topK,
            numTokens,
        ).launch(
            grid=[grid_x, grid_y, grid_z],
            block=[BLOCK_SIZE, 1, 1],
            cluster=[1, 1, 1],
            smem=SmemStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_t: cute.Tensor,          # [rows, K] fp16 flat
        input2_t: cute.Tensor,         # [rows, K] fp16 flat (gated second input)
        weights_t: cute.Tensor,        # flat uint8
        scales_t: cute.Tensor,         # flat fp8 e4m3 (uint8) — atom layout
        global_scales_t: cute.Tensor,  # [E] fp32 per-expert global scales
        expert_ids_t: cute.Tensor,     # flat int32
        gate_scores_t: cute.Tensor,    # flat fp32
        output_t: cute.Tensor,         # flat fp16
        K: Int32,
        K_half: Int32,
        N: Int32,
        topK: Int32,
        numTokens: Int32,
    ):
        """Device kernel body.

        Grid: (ceil(N/128), TopK, numTokens)
        Block: 512 threads (32 N-threads x 16 K-warps)
        Smem: 8448 bytes for K-parallel reduction buffer
        """
        # ---- Shared memory allocation ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SmemStorage)
        red_layout = cute.make_layout(SMEM_FLOATS, stride=1)
        red_t = storage.red_buf.get_tensor(red_layout)

        # ---- Thread / block indices ----
        tid, _, _ = cute.arch.thread_idx()
        tidx = tid % Int32(N_THREADS)     # 0..31: N-position index
        kw = tid // Int32(N_THREADS)      # 0..15: K-chunk index

        bx, k_slot, token_idx = cute.arch.block_idx()
        n_base = bx * Int32(TILE_N) + tidx * Int32(VEC_N)

        # n_local is the thread's N offset within the CTA tile (for smem indexing)
        n_local = tidx * Int32(VEC_N)  # 0, 4, 8, ..., 124

        # Shared memory base address (32-bit, preserves addrspace 3)
        smem_base = _get_smem_ptr_as_int32(red_t, Int32(0))

        if n_base < N:
            # ---- Expert / pointer setup ----
            eids_base = _get_ptr_as_int64(expert_ids_t, Int32(0))
            expert_id = _ld_global_i32(
                eids_base + Int64(token_idx * topK + k_slot) * Int64(4)
            )

            input_base = _get_ptr_as_int64(input_t, Int32(0))
            weights_base = _get_ptr_as_int64(weights_t, Int32(0))
            scales_base = _get_ptr_as_int64(scales_t, Int32(0))

            # Expert weight base: expert * K_half * N = expert * K * N_half bytes
            expert_w_offset = Int64(expert_id) * Int64(K_half) * Int64(N)

            # Prefill atom-layout scale stride per expert (M=N, K_sf=K/16):
            #   paddedM = ceil(N / 128) * 128
            #   numSfCols = K / 16
            #   paddedSfCols = ceil(numSfCols / 4) * 4
            #   stride = ceil(paddedM * paddedSfCols / 4) * 4  (int32-aligned)
            num_sf_cols = K >> Int32(4)
            padded_sf_cols = ((num_sf_cols + Int32(3)) >> Int32(2)) << Int32(2)
            padded_m = ((N + Int32(127)) >> Int32(7)) << Int32(7)
            sf_bytes_per_ex = Int64(padded_m) * Int64(padded_sf_cols)
            atom_expert_stride = ((sf_bytes_per_ex + Int64(3)) >> Int64(2)) << Int64(2)
            expert_s_offset = Int64(expert_id) * atom_expert_stride

            # Per-expert global scale (FP32)
            g_base = _get_ptr_as_int64(global_scales_t, Int32(0))
            g_scale = _ld_global_f32(g_base + Int64(expert_id) * Int64(4))

            # Input row base addressing
            if cutlass.const_expr(self._output_atomic):
                # Reduce mode: input indexed by [token_idx * topK + k_slot]
                input_row_base = input_base + Int64(token_idx * topK + k_slot) * Int64(K) * Int64(2)
            else:
                # Projection mode: input indexed by [token_idx]
                input_row_base = input_base + Int64(token_idx) * Int64(K) * Int64(2)

            # Second input base (for gated mode)
            if cutlass.const_expr(self._gated):
                input2_base = _get_ptr_as_int64(input2_t, Int32(0))
                if cutlass.const_expr(self._output_atomic):
                    input2_row_base = input2_base + Int64(token_idx * topK + k_slot) * Int64(K) * Int64(2)
                else:
                    input2_row_base = input2_base + Int64(token_idx) * Int64(K) * Int64(2)

            # Pre-load gate score for output_atomic (score pre-multiplication)
            if cutlass.const_expr(self._output_atomic):
                gate_addr = _get_ptr_as_int64(gate_scores_t, Int32(0))
                score = _ld_global_f32(
                    gate_addr + Int64(token_idx * topK + k_slot) * Int64(4)
                )

            # ---- K-parallel chunk assignment ----
            chunk = (K_half + Int32(K_WARPS - 1)) // Int32(K_WARPS)
            kb_start = kw * chunk
            kb_end = (kw + Int32(1)) * chunk
            if kb_end > K_half:
                kb_end = K_half

            # ---- Main dot-product loop ----
            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            acc2 = Float32(0.0)
            acc3 = Float32(0.0)

            # N-major addressing: N_half = N/2, n_half_addr = n_base/2 (byte offset)
            N_half = N >> Int32(1)
            n_half_addr = Int64(n_base >> Int32(1))

            kb = kb_start
            while kb < kb_end:
                # K-pair base address: same as old w_row_addr since K_half*N = K*N_half
                w_pair_base = (
                    weights_base + expert_w_offset
                    + Int64(kb) * Int64(N)
                )

                # ---- Input pre-processing ----
                h_offset = Int64(kb) * Int64(4)  # 2*kb fp16 elements x 2 bytes

                if cutlass.const_expr(self._gated):
                    # Gated mode: act(gate) * up
                    gate0 = _ld_global_f16_as_f32(input_row_base + h_offset)
                    gate1 = _ld_global_f16_as_f32(input_row_base + h_offset + Int64(2))
                    up0 = _ld_global_f16_as_f32(input2_row_base + h_offset)
                    up1 = _ld_global_f16_as_f32(input2_row_base + h_offset + Int64(2))

                    if cutlass.const_expr(self._apply_silu):
                        a0 = _silu_f32(gate0) * up0
                        a1 = _silu_f32(gate1) * up1
                    elif cutlass.const_expr(self._apply_relu2):
                        g0 = _fmax_f32(gate0, Float32(0.0))
                        g1 = _fmax_f32(gate1, Float32(0.0))
                        a0 = g0 * g0 * up0
                        a1 = g1 * g1 * up1
                    else:
                        a0 = gate0 * up0
                        a1 = gate1 * up1
                else:
                    # Non-gated mode
                    a0_raw = _ld_global_f16_as_f32(input_row_base + h_offset)
                    a1_raw = _ld_global_f16_as_f32(input_row_base + h_offset + Int64(2))

                    if cutlass.const_expr(self._apply_silu):
                        a0 = _silu_f32(a0_raw)
                        a1 = _silu_f32(a1_raw)
                    elif cutlass.const_expr(self._apply_relu2):
                        a0 = _fmax_f32(a0_raw, Float32(0.0))
                        a0 = a0 * a0
                        a1 = _fmax_f32(a1_raw, Float32(0.0))
                        a1 = a1 * a1
                    else:
                        a0 = a0_raw
                        a1 = a1_raw

                # Score pre-multiplication for atomic output modes
                if cutlass.const_expr(self._output_atomic):
                    a0 = a0 * score
                    a1 = a1 * score

                # N-major weight loads: 2 x u16 from k0 and k1 rows
                # Each u16 = 2 bytes = 4 nibbles = 4 N-adjacent weight values
                packed_k0 = _ld_global_u16(w_pair_base + n_half_addr)
                packed_k1 = _ld_global_u16(w_pair_base + Int64(N_half) + n_half_addr)
                b0_k0 = packed_k0 & Uint32(0xFF)
                b1_k0 = (packed_k0 >> Uint32(8)) & Uint32(0xFF)
                b0_k1 = packed_k1 & Uint32(0xFF)
                b1_k1 = (packed_k1 >> Uint32(8)) & Uint32(0xFF)

                # Prefill atom-layout scale load (M=N, K_sf=K/16):
                # m_idx from N position, k_idx from K group.
                # All 4 N-positions are in the same 16-element N-block (VEC_N=4 < 16)
                # so they share one scale value.
                leader_row_i32 = (n_base >> Int32(4)) << Int32(4)
                sf_col = Int32(kb) >> Int32(3)
                inner_k = sf_col & Int32(3)
                inner_m = (leader_row_i32 & Int32(127)) >> Int32(5)
                outer_m = leader_row_i32 & Int32(31)
                k_tile = sf_col >> Int32(2)
                num_k_tiles = (num_sf_cols + Int32(3)) >> Int32(2)
                m_tile = leader_row_i32 >> Int32(7)
                atom_off = (
                    Int64(m_tile) * Int64(num_k_tiles) * Int64(512)
                    + Int64(k_tile) * Int64(512)
                    + Int64(outer_m) * Int64(16)
                    + Int64(inner_m) * Int64(4)
                    + Int64(inner_k)
                )
                sf_byte = _ld_global_u8(scales_base + expert_s_offset + atom_off)
                sc_val = _cvt_1xfp8e4m3_to_f32(sf_byte) * g_scale

                # Decode N-major FP4: lo nibble = even N, hi nibble = odd N
                # b0_kX: lo=w(kX, n_base), hi=w(kX, n_base+1)
                # b1_kX: lo=w(kX, n_base+2), hi=w(kX, n_base+3)
                w0_k0_lo, w0_k0_hi = _cvt_e2m1x2_to_f32x2(b0_k0)
                w1_k0_lo, w1_k0_hi = _cvt_e2m1x2_to_f32x2(b1_k0)
                w0_k1_lo, w0_k1_hi = _cvt_e2m1x2_to_f32x2(b0_k1)
                w1_k1_lo, w1_k1_hi = _cvt_e2m1x2_to_f32x2(b1_k1)

                w0_k0_lo = w0_k0_lo * sc_val
                w0_k0_hi = w0_k0_hi * sc_val
                w1_k0_lo = w1_k0_lo * sc_val
                w1_k0_hi = w1_k0_hi * sc_val
                w0_k1_lo = w0_k1_lo * sc_val
                w0_k1_hi = w0_k1_hi * sc_val
                w1_k1_lo = w1_k1_lo * sc_val
                w1_k1_hi = w1_k1_hi * sc_val

                # Accumulate: acc_i += a0 * w(k0, n_base+i) + a1 * w(k1, n_base+i)
                acc0 = _fmaf_f32(a0, w0_k0_lo, _fmaf_f32(a1, w0_k1_lo, acc0))
                acc1 = _fmaf_f32(a0, w0_k0_hi, _fmaf_f32(a1, w0_k1_hi, acc1))
                acc2 = _fmaf_f32(a0, w1_k0_lo, _fmaf_f32(a1, w1_k1_lo, acc2))
                acc3 = _fmaf_f32(a0, w1_k0_hi, _fmaf_f32(a1, w1_k1_hi, acc3))

                kb = kb + Int32(1)

            # ---- Store partial sums to shared memory ----
            smem_byte_off = (kw * Int32(TILE_N_PAD) + n_local) * Int32(4)
            _st_shared_f32(smem_base + smem_byte_off, acc0)
            _st_shared_f32(smem_base + smem_byte_off + Int32(4), acc1)
            _st_shared_f32(smem_base + smem_byte_off + Int32(8), acc2)
            _st_shared_f32(smem_base + smem_byte_off + Int32(12), acc3)

        # All threads must reach the barrier (including n_base >= N)
        cute.arch.sync_threads()

        # ---- Reduction: warp 0 (kw == 0) sums across K_WARPS ----
        if n_base < N:
            if kw == Int32(0):
                sum0 = Float32(0.0)
                sum1 = Float32(0.0)
                sum2 = Float32(0.0)
                sum3 = Float32(0.0)

                for i in cutlass.range_constexpr(K_WARPS):
                    rd_off = (Int32(i * TILE_N_PAD) + n_local) * Int32(4)
                    sum0 = sum0 + _ld_shared_f32(smem_base + rd_off)
                    sum1 = sum1 + _ld_shared_f32(smem_base + rd_off + Int32(4))
                    sum2 = sum2 + _ld_shared_f32(smem_base + rd_off + Int32(8))
                    sum3 = sum3 + _ld_shared_f32(smem_base + rd_off + Int32(12))

                # ---- Write output ----
                out_base = _get_ptr_as_int64(output_t, Int32(0))

                if cutlass.const_expr(self._output_atomic):
                    # Atomic fp16 add: output[token_idx, n] (topK experts accumulate)
                    out_row = out_base + Int64(token_idx) * Int64(N) * Int64(2)
                    _atomic_add_global_f16_from_f32(
                        out_row + Int64(n_base) * Int64(2), sum0
                    )
                    _atomic_add_global_f16_from_f32(
                        out_row + Int64(n_base + Int32(1)) * Int64(2), sum1
                    )
                    _atomic_add_global_f16_from_f32(
                        out_row + Int64(n_base + Int32(2)) * Int64(2), sum2
                    )
                    _atomic_add_global_f16_from_f32(
                        out_row + Int64(n_base + Int32(3)) * Int64(2), sum3
                    )
                else:
                    # Direct fp16 store: output[token_idx * topK + k_slot, n]
                    out_row = out_base + Int64(token_idx * topK + k_slot) * Int64(N) * Int64(2)
                    _st_global_f16_from_f32(
                        out_row + Int64(n_base) * Int64(2), sum0
                    )
                    _st_global_f16_from_f32(
                        out_row + Int64(n_base + Int32(1)) * Int64(2), sum1
                    )
                    _st_global_f16_from_f32(
                        out_row + Int64(n_base + Int32(2)) * Int64(2), sum2
                    )
                    _st_global_f16_from_f32(
                        out_row + Int64(n_base + Int32(3)) * Int64(2), sum3
                    )
