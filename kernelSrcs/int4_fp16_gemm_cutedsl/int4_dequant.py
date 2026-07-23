# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared INT4 (W4A16) fragment-word dequant device helper.

``_dequant_int4_word`` is the single B-dequant path for both the prefill GEMM
(``int4_fp16_gemm_ampere.py``) and the decode GEMV (``int4_fp16_gemv_ampere.py``)
kernels: they consume the *same* offline fragment-order uint32 weight buffer
(``int4_reference.repack_b_for_tile``, bN=128/bK=64), so a single dequant keeps
the two kernels bit-identical and avoids duplication.
"""

from __future__ import annotations

from cutlass import Float16, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _dequant_int4_word(q, scale, *, loc=None, ip=None):
    """Dequant one 32-bit word of four biased-INT4 nibbles into two scaled
    ``f16x2`` register words, at minimal op count -- two ``lop3.b32`` (low/high
    nibble masks), one ``sub.f16x2``, one ``fma.rn.f16x2`` (the ``0x2c00``
    multiplier corrects the high mask's implicit x16), and a single shared scale
    splat.  Returns the two ``Uint32`` words ready for a vector store into the B
    fragment.

    This is the only B dequant path: the repacked B word is already in exact
    MMA-fragment order (the swizzle is done offline, see ``int4_reference``), so
    ``dequant(word)`` / ``dequant(word >> 8)`` peel the two N-groups directly --
    no per-fragment fixup, nibble shuffle, or scalar extract.
    """
    i32 = Int32.mlir_type
    u32 = Uint32.mlir_type
    source = Uint32(q).ir_value(loc=loc, ip=ip)
    ex = arith.constant(u32, 0x64006400, loc=loc, ip=ip)
    lut = arith.constant(u32, (0xF0 & 0xCC) | 0xAA, loc=loc, ip=ip)
    lo_mask = arith.constant(u32, 0x000F000F, loc=loc, ip=ip)
    hi_mask = arith.constant(u32, 0x00F000F0, loc=loc, ip=ip)

    def _lop3(mask):
        return llvm.inline_asm(
            i32, [source, mask, ex, lut], "lop3.b32 $0, $1, $2, $3, $4;",
            "=r,r,n,n,n", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    lo = _lop3(lo_mask)
    hi = _lop3(hi_mask)
    sub = arith.constant(u32, 0x64086408, loc=loc, ip=ip)
    lo = llvm.inline_asm(
        i32, [lo, sub], "sub.f16x2 $0, $1, $2;", "=r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    mul = arith.constant(u32, 0x2C002C00, loc=loc, ip=ip)
    add = arith.constant(u32, 0xD480D480, loc=loc, ip=ip)
    hi = llvm.inline_asm(
        i32, [hi, mul, add], "fma.rn.f16x2 $0, $1, $2, $3;", "=r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    vec_t = ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    sv = vector.broadcast(vec_t, Float16(scale).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    lo_s = arith.mulf(llvm.bitcast(vec_t, lo, loc=loc, ip=ip), sv, loc=loc, ip=ip)
    hi_s = arith.mulf(llvm.bitcast(vec_t, hi, loc=loc, ip=ip), sv, loc=loc, ip=ip)
    return (
        Uint32(llvm.bitcast(i32, lo_s, loc=loc, ip=ip)),
        Uint32(llvm.bitcast(i32, hi_s, loc=loc, ip=ip)),
    )
