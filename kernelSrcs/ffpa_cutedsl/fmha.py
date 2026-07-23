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

# Origin: Adapted from xlite-dev/ffpa-attn (Apache-2.0):
#   https://github.com/xlite-dev/ffpa-attn
#   csrc/cuffpa/prefill.cuh + csrc/cuffpa/ffpa_attn_fwd.cuh
# Copyright (c) DefTruth & the xlite-dev community (Apache-2.0)

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, Type

_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import numpy as np
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack


"""FFPA-style Ampere-floor FMHA forward (CuTe-DSL), AOT-export build.

Whole-tile variant ported from xlite-dev/ffpa-attn — keeps the full
``(Br, D)`` Q tile, ``(Bc, D)`` K tile and ``(Bc, D)`` V tile in SMEM for
the lifetime of each CTA.  Structural skeleton is the upstream NVIDIA
CuTeDSL Ampere FA2 example
(``examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py``);
the FFPA-style register accounting + the SMEM-feasibility relaxations are
the variant's own.

Layout: Q / K / V / O are all ``BSND = (batch, seq, num_head, head_dim)``,
bf16 or fp16.  Per-CTA tile: one ``(Q_tile, head, batch)`` triplet sweeps
all KV tiles internally with FA2-style online softmax (rescale-on-shift).

Variant axes baked at compile time:
  * ``head_dim``  — whole-D MMA partitioning; runtime mode-3 not honored.
  * ``is_causal`` — separate compiled kernel for causal vs dense masking.

Runtime-dynamic axes (no recompile needed across these): batch size,
seqlen_q, seqlen_k, num_head (Q heads), ``num_kv_heads`` (GQA — passed as a
kernel argument; the group size is ``num_head / num_kv_heads``, ``1`` = MHA),
tensor strides.

Variable-length batches: two ``(B+1,)`` Int32 cumulative-length tensors
(``mCuSeqLenQ`` / ``mCuSeqLenK``, same convention as
``fmha_cutedsl_blackwell/fmha.py``) carry the *logical* per-sequence lengths;
``mQ.shape[1]`` / ``mK.shape[1]`` remain the physical (padded) extents.  Per
batch ``b``: ``seqlen_q_b = cu_q[b+1] - cu_q[b]``, ``seqlen_k_b = cu_k[b+1] -
cu_k[b]`` and the causal mask is bottom-right aligned with offset
``seqlen_k_b - seqlen_q_b`` (0 for plain right-padded prefill, the KV-cache
prefix length for chunked prefill).  Padding K/V positions are never
attended by valid rows, and padding Q rows are residual-masked in the masked
steps — fixing the Gemma4 BS>1 NaN corruption at the source.

Vision-block overlay (Gemma4 Unified): two optional ``(B, S_q)`` Int32
tensors ``mBlockBegin`` / ``mBlockEnd`` carry, per query row, an extra
allowed KV interval ``[blockBegin[q], blockEnd[q]]`` (sentinel ``-1`` /
``-1`` for text rows — the empty interval).  Every image-placeholder run
gets one contiguous block that contains the row's own diagonal position
(``blockBegin[q] <= q + offset_b <= blockEnd[q]``), so
``allow(q, k) = causal(q, k) OR blockBegin[q] <= k <= blockEnd[q]``.  KV
tiles between the causal diagonal and the Q tile's largest ``blockEnd``
are additionally visited through the masked path (blocks are short —
O(hundreds of tokens) past the diagonal — so the extra work is bounded).
Passing ``None`` for both tensors compiles the overlay away entirely: the
generated kernel and its AOT C ABI are identical to the plain causal
variant.
"""


class FFPAFmhaAmpere:

    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
        skip_rescale: bool = False,
        hybrid_exp2: bool = False,
    ):
        """Initialize the FFPA Ampere kernel.

        All contiguous dimensions must be at least 16 bytes aligned, which
        means ``head_dim`` should be a multiple of 8.

        GQA is a runtime axis: the kernel takes ``num_kv_heads`` as a launch
        argument (see ``__call__``), so a single compiled kernel serves MHA
        and any GQA group size without recompiling.

        :param head_dim: head dimension
        :param m_block_size: ``Br`` — query tile rows per CTA
        :param n_block_size: ``Bc`` — key/value tile rows per CTA
        :param num_threads: CTA thread count
        :param is_causal: enable causal mask
        :param skip_rescale: clamp per-row rescale factor to 1.0 when within
            ``2^-8`` of unity.  Trades one FFMA for a stabilized rescale, in
            line with the upstream xlite-dev/ffpa-attn tuning.
        :param hybrid_exp2: scaffold for FA4-style 75 % MUFU + 25 % polynomial
            exp2 in softmax.  Not validated in this variant; defer.
        """
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._skip_rescale = skip_rescale
        self._hybrid_exp2 = hybrid_exp2

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
    ) -> bool:
        """Check whether the (dtype, tile, threads) combo is implementable.

        Relaxed from the upstream FA2 example for D=512 on sm_86: the
        defensive ``(m_block * 2) % num_threads == 0`` guard is dropped
        because it rejects valid configs like ``Br=32`` + ``128`` threads
        where ``tQKV`` partitioning still tiles cleanly.  The SMEM capacity
        check uses sm_80's 99 KB opt-in floor, which is also the budget on
        sm_86 / sm_87 / sm_89.
        """
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False

        # Q tile (Br * D) + K tile + V tile, all bf16/fp16.
        smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False

        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mCuSeqLenQ: cute.Tensor,
        mCuSeqLenK: cute.Tensor,
        mBlockBegin: Optional[cute.Tensor],
        mBlockEnd: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        num_kv_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Configure SMEM / tiled-copy / tiled-mma and launch the kernel.

        All four Q/K/V/O tensors share dtype (fp16 or bf16) and BSND layout
        ``(B, S, H, D)``.  Strides match a contiguous ``B*S*H*D`` packing
        with ``D`` innermost.

        ``mCuSeqLenQ`` / ``mCuSeqLenK`` are ``(B+1,)`` Int32 cumulative
        sequence lengths carrying the logical per-batch valid lengths (see
        module docstring).  For uniform dense batches pass
        ``[0, S, 2S, ...]`` — the masking then degenerates to the padded
        extents.

        ``mBlockBegin`` / ``mBlockEnd`` are optional ``(B, S_q)`` Int32
        vision-block interval tensors (see module docstring).  Pass ``None``
        for both to compile the plain causal kernel (unchanged codegen and
        AOT ABI); pass both to compile the vision-block overlay variant.

        ``num_kv_heads`` is the number of K/V heads (``mK``/``mV`` mode-2
        extent).  GQA group size is ``num_head_q / num_kv_heads`` and is
        computed at runtime inside the kernel; ``num_kv_heads == num_head_q``
        is plain MHA.  The caller must ensure ``num_head_q % num_kv_heads == 0``.
        """
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type == mV.element_type == mO.element_type
            )
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr((mBlockBegin is None) != (mBlockEnd is None)):
            raise TypeError(
                "mBlockBegin and mBlockEnd must both be provided or both be None"
            )
        if cutlass.const_expr(mBlockBegin is not None and not self._is_causal):
            raise TypeError(
                "The vision-block overlay is only defined for the causal variant"
            )
        self._dtype: Type[cutlass.Numeric] = mQ.element_type
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )

        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )

        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQKV_layout: thread layout for QKV load
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        # tO_layout: thread layout for O store
        tO_layout = tQKV_layout

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        # gmem_tiled_copy_QKV: tiled copy for QKV load
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        # gmem_tiled_copy_O: tiled copy for O store
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled mma
        # ///////////////////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid_dim: (m_block, batch_size, num_head)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mCuSeqLenQ,
            mCuSeqLenK,
            mBlockBegin,
            mBlockEnd,
            softmax_scale_log2,
            num_kv_heads,
            sQ_layout,
            sKV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mCuSeqLenQ: cute.Tensor,
        mCuSeqLenK: cute.Tensor,
        mBlockBegin: Optional[cute.Tensor],
        mBlockEnd: Optional[cute.Tensor],
        softmax_scale_log2: cutlass.Float32,
        num_kv_heads: cutlass.Int32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """FA2 kernel body: prologue cp.async Q/K, KV outer loop in reverse,
        ``compute_one_n_block`` for BMM1 → softmax_rescale → BMM2.  Epilogue
        rmem → smem (aliased over sQ) → gmem.
        """
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()
        # GQA: group_size = H_q / H_kv (runtime); each K/V head is shared
        # across `kv_group_size` consecutive Q heads, so the K/V head index for
        # this CTA's Q head is `num_head // kv_group_size`.  Matches the
        # `q_head * H_kv / H_q` convention of the FP32 BSHD reference.
        kv_group_size = mQ.shape[2] // num_kv_heads
        num_head_kv = num_head // kv_group_size

        # Per-batch logical lengths (varlen): mQ.shape[1]/mK.shape[1] stay the
        # physical padded extents (used for OOB predicates); the cu_seqlen
        # tensors carry the valid lengths.  offset_b is the bottom-right causal
        # offset (0 for right-padded own-KV prefill; the KV-cache prefix length
        # for chunked prefill).
        seqlen_q_b = mCuSeqLenQ[batch_size + 1] - mCuSeqLenQ[batch_size]
        seqlen_k_b = mCuSeqLenK[batch_size + 1] - mCuSeqLenK[batch_size]
        offset_b = seqlen_k_b - seqlen_q_b

        n_block_max = cute.ceil_div(seqlen_k_b, self._n_block_size)
        if self._is_causal:
            n_block_max = min(
                cute.ceil_div(
                    (m_block + 1) * self._m_block_size + offset_b,
                    self._n_block_size,
                ),
                n_block_max,
            )
        # Skip whole-padding Q tiles: when the tile's FIRST row is already at
        # or past seqlen_q_b, every row of this m_block is a padding row and
        # there is nothing to compute.  The skip works by forcing
        # n_block_max = 0, which makes the KV loop below run zero iterations:
        # no K/V block is prefetched (the `n_block >= 0` guards), acc_O keeps
        # its zero fill and row_sum stays 0, so normalize_softmax's
        # `row_sum == 0` guard keeps scale = 1 and the epilogue stores exact
        # zeros for the whole tile.  Tiles that straddle seqlen_q_b are NOT
        # skipped: their valid rows compute normally, and their padding rows
        # are masked per-row in the masked steps (see softmax_rescale_O) —
        # those rows end up bounded (convex combinations of valid V rows),
        # not exact zeros, matching the CuTe DSL FMHA behaviour.
        n_block_max = (
            0 if m_block * self._m_block_size >= seqlen_q_b else n_block_max
        )
        # Vision-block overlay: extend the per-CTA KV upper bound so tiles
        # covering [.., blockEnd[q]] beyond the causal diagonal are visited.
        # The bound is the max blockEnd over this Q tile's valid rows; the
        # extension tiles (extra_mask_steps of them) always take the masked
        # path since most of their entries lie past the causal limit.  With
        # no blocks in the tile (blockEnd == -1 everywhere, or a whole-padding
        # Q tile whose row scan is empty) the extension is exactly zero and
        # the traversal matches the plain causal kernel.
        n_block_max_causal = n_block_max
        extra_mask_steps = 0
        if cutlass.const_expr(mBlockEnd is not None):
            block_end_tile_max = cutlass.Int32(-1)
            row_lo = m_block * self._m_block_size
            row_hi = cutlass.min(row_lo + self._m_block_size, seqlen_q_b)
            row_hi = cutlass.max(row_hi, row_lo)
            for row in cutlass.range(row_lo, row_hi, 1, unroll=1):
                block_end_tile_max = cutlass.max(
                    block_end_tile_max, mBlockEnd[batch_size, row]
                )
            n_block_ext = cutlass.min(
                cute.ceil_div(block_end_tile_max + 1, self._n_block_size),
                cute.ceil_div(seqlen_k_b, self._n_block_size),
            )
            n_block_max = cutlass.max(n_block_max, n_block_ext)
            extra_mask_steps = n_block_max - n_block_max_causal
        n_block = n_block_max - 1

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # Q is indexed by num_head (Q head); K/V by num_head_kv (shared head).
        # ///////////////////////////////////////////////////////////////////////////////
        gQ = cute.local_tile(
            mQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        gK = cute.local_tile(
            mK[batch_size, None, num_head_kv, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_size, None, num_head_kv, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()

        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        # Transposed view of V: (head_dim, n_block_size) for the BMM2 mma.
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: mark indices that need to copy when problem_shape isn't a
        # multiple of tile_shape.
        # ///////////////////////////////////////////////////////////////////////////////
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, num_head_kv, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )

        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        # Only the k-tile of the predicate is materialised; m/n predicates use
        # the first tile inline (-2-3 % perf gain vs. allocating the whole tile).
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ.shape[0][1],
                    cute.size(tQsQ, mode=[1]),
                    cute.size(tQsQ, mode=[2]),
                ),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tKsK.shape[0][1],
                    cute.size(tKsK, mode=[1]),
                    cute.size(tKsK, mode=[2]),
                ),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )
        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch Prologue — start async loads of the last mn-tile (mn residue handled here).
        # ///////////////////////////////////////////////////////////////////////////////
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                tQsQ[None, m, None].fill(0)
        # n_block == -1 only for a degenerate empty batch (seqlen_k_b == 0);
        # skip the K prefetch entirely — the KV loop below runs zero steps and
        # the epilogue stores the zero-initialised acc_O.
        #
        # NaN hardening: rows of the boundary K/V tile
        # at logical positions >= seqlen_k_b are ZERO-FILLED, not just score
        # masked.  Score masking alone zeroes the softmax weight, but BMM2
        # still multiplies P(=0) x V, and IEEE 0 x NaN = NaN — padding rows can
        # legitimately hold NaN/Inf mid-network (fp16 overflow of garbage
        # embeddings in non-attention layers), which would poison the whole
        # boundary q-tile.  Zero-filled K/V make the masked columns inert
        # (0 x 0 = 0) regardless of what padding memory contains.  The
        # prologue loads the boundary tile (first processed block), so the
        # logical bound is applied here and in the first V load below; interior
        # blocks are entirely below seqlen_k_b and keep the fast full copies.
        if n_block >= 0:
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][1], seqlen_k_b):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, n, None, n_block],
                        tKsK[None, n, None],
                        pred=tKVpKV[None, n, None],
                    )
                else:
                    tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////////////////
        # Online-softmax intermediate state: row_max and row_sum (per-row scalars).
        # ///////////////////////////////////////////////////////////////////////////////
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        basic_params = SimpleNamespace(
            m_block=m_block,
            n_block=n_block,
            mQ=mQ,
            mK=mK,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q_b=seqlen_q_b,
            seqlen_k_b=seqlen_k_b,
            offset_b=offset_b,
            mBlockBegin=mBlockBegin,
            mBlockEnd=mBlockEnd,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK,
            tKsK=tKsK,
            tVgV=tVgV,
            tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK,
            tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # Two flavours of N-block iteration: masking (the block straddling the
        # per-batch seqlen_k_b boundary, and the causal tail) and unmasking.
        # Always at least one masking step.  Causal needs ceil(Br/Bc) diagonal
        # blocks +1: with a per-batch bottom-right offset the diagonal band
        # straddles one extra K block when offset_b % Bc != 0.  mask_steps is
        # a compile-time constant (this loop is unrolled) while offset_b is
        # runtime data, so the +1 is budgeted unconditionally — for aligned
        # offsets the extra masked pass simply finds nothing to mask (one
        # 16-key block per CTA takes the masked path instead of the fast
        # path; correctness over a cheap no-op).
        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size) + 1

        for n_tile in cutlass.range_constexpr(mask_steps):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            if cutlass.const_expr(self._is_causal):
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=(n_tile == 0),
                        in_mask_steps=True,
                    )
            else:
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=True,
                        in_mask_steps=True,
                    )

        # Vision-block extension tiles: KV tiles between the causal diagonal
        # and the Q tile's largest blockEnd.  Every one of them needs the
        # masked path (rows without a block are fully masked there; block
        # rows are allowed only inside their [blockBegin, blockEnd]
        # interval).  extra_mask_steps is the compile-time constant 0 when
        # the overlay is disabled, so this loop vanishes and the traversal
        # is bit-identical to the plain causal kernel.
        for n_tile in range(mask_steps, mask_steps + extra_mask_steps, 1):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            if n_block >= 0:
                self.compute_one_n_block(
                    basic_params,
                    mma_params,
                    gmem_copy_params,
                    smem_copy_params,
                    softmax_params,
                    is_first_n_block=False,
                    in_mask_steps=True,
                )

        # Remaining K-tiles in reverse order — no k-residue handling needed.
        for n_tile in range(mask_steps + extra_mask_steps, n_block_max, 1):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            self.compute_one_n_block(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                is_first_n_block=False,
                in_mask_steps=False,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue: normalize acc_O, rmem → smem (aliased over sQ) → gmem.
        # ///////////////////////////////////////////////////////////////////////////////
        self.normalize_softmax(acc_O, row_sum)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(
            smem_copy_atom_O,
            taccOrO,
            taccOsO,
        )
        gO = cute.local_tile(
            mO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(
            gmem_tiled_copy_O,
            tOsO,
            tOrO,
        )
        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """One outer N-block: BMM1 (Q @ K^T → S) → online softmax → BMM2 (P @ V → O)."""
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        # First tile: load V into smem with the n-residue predicate (otherwise
        # a single vectorised copy is enough — the `if` here is a constexpr).
        # First tile == the boundary K/V block: zero-fill V rows at logical
        # positions >= seqlen_k_b (NaN hardening — see the prologue K load).
        if is_first_n_block:
            for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                if cute.elem_less(
                    gmem_copy_params.tKVcKV[0, n, 0][1],
                    basic_params.seqlen_k_b,
                ):
                    cute.copy(
                        gmem_copy_params.gmem_tiled_copy_QKV,
                        gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                        gmem_copy_params.tVsV[None, n, None],
                        pred=gmem_copy_params.tKVpKV[None, n, None],
                    )
                else:
                    gmem_copy_params.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                gmem_copy_params.tVsV,
                pred=gmem_copy_params.tKVpKV,
            )

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # S = Q @ K^T  (BMM1)
        # ///////////////////////////////////////////////////////////////////////////////
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        if basic_params.n_block > 0:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                gmem_copy_params.tKsK,
                pred=gmem_copy_params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # Online softmax: rescale row_max/row_sum/acc_O, compute P = softmax(S).
        # ///////////////////////////////////////////////////////////////////////////////
        self.softmax_rescale_O(
            basic_params,
            mma_params,
            softmax_params,
            acc_S,
            is_first_n_block,
            in_mask_steps,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # ///////////////////////////////////////////////////////////////////////////////
        # O += P @ V  (BMM2)
        # ///////////////////////////////////////////////////////////////////////////////
        # mma.m16n8k16 layout: rearrange P from (4, MMA_M, MMA_N)
        # to ((4, 2), MMA_M, MMA_N / 2) to match the operand-A shape.
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Apply online softmax to ``acc_S`` and rescale ``acc_O``.

        Distinguishes ``is_first_n_block`` (no prior row_max to fold in) from
        subsequent blocks, and masked vs. unmasked steps.  When
        ``skip_rescale`` is enabled and ``exp2(m_prev - m_cur) ≈ 1``, the
        rescale factor is snapped to exactly ``1.0`` to suppress precision
        wobble (see FFPA upstream).
        """
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor(
                (
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[1],
                    basic_params.mQ.shape[2],
                    basic_params.mK.shape[1],
                )
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_size, None, basic_params.num_head, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            if cutlass.const_expr(in_mask_steps):
                # Per-batch varlen masking:
                #   residual K: mask index_k >= seqlen_k_b (padding keys)
                #   causal:     bottom-right aligned, limit = qi + offset_b + 1
                #   residual Q: padding rows (qi >= seqlen_q_b) fully masked
                #     (masked steps only — in unmasked blocks padding rows
                #     still accumulate bounded valid-V combinations)
                row_idx = tScS_mn[r, 0][1]
                if cutlass.const_expr(not self._is_causal):
                    col_idx_limit = basic_params.seqlen_k_b
                else:
                    col_idx_limit = cutlass.min(
                        row_idx + 1 + basic_params.offset_b, basic_params.seqlen_k_b
                    )
                col_idx_limit = (
                    0 if row_idx + 1 > basic_params.seqlen_q_b else col_idx_limit
                )
                if cutlass.const_expr(basic_params.mBlockBegin is not None):
                    # Vision-block overlay: this row's extra allowed KV
                    # interval.  The 0/-1 default keeps the interval empty
                    # for padding rows (also guarding the (B, S_q) gather
                    # against out-of-bounds row indices in residual tiles);
                    # text rows store the -1/-1 sentinel which is empty for
                    # the same reason (no key satisfies col <= -1).
                    block_begin_r = cutlass.Int32(0)
                    block_end_r = cutlass.Int32(-1)
                    if row_idx < basic_params.seqlen_q_b:
                        block_begin_r = basic_params.mBlockBegin[
                            basic_params.batch_size, row_idx
                        ]
                        block_end_r = cutlass.min(
                            basic_params.mBlockEnd[
                                basic_params.batch_size, row_idx
                            ],
                            basic_params.seqlen_k_b - 1,
                        )
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        col_idx = tScS_mn[0, c][3]
                        # Masked iff past the causal/varlen limit AND outside
                        # the block interval (two writes for keys outside the
                        # interval on both compares are harmless).
                        if cute.elem_less(col_idx_limit, col_idx + 1):
                            if cute.elem_less(col_idx, block_begin_r):
                                acc_S_mn[r, c] = -cutlass.Float32.inf
                            if cute.elem_less(block_end_r, col_idx):
                                acc_S_mn[r, c] = -cutlass.Float32.inf
                else:
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        if cute.elem_less(col_idx_limit, tScS_mn[0, c][3] + 1):
                            acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            # Snap a fully-masked row's -inf max to 0 so exp2 yields 0 (not
            # NaN).  Hits the causal diagonal tail and varlen-masked padding
            # rows (which non-causal masking can now also fully mask).
            row_max_cur_row = (
                0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
            )

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                # skip_rescale: snap a near-unity rescale factor (within 2^-8)
                # to exactly 1.0 — trades one FFMA for a stabilized rescale.
                if cutlass.const_expr(self._skip_rescale):
                    prev_minus_cur_exp = (
                        cutlass.Float32(1.0)
                        if prev_minus_cur_exp > cutlass.Float32(0.99609375)
                        else prev_minus_cur_exp
                    )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
    ):
        """Final softmax normalisation: ``acc_O[r, :] /= row_sum[r]``."""
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )

            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret a ``(MMA, MMA_M, MMA_N)`` accumulator as ``(M, N)``."""
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """Reduce across the four threads holding the same column of an MMA fragment."""
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)


# ---------------------------------------------------------------------------
# Host-side helpers (CuPy/NumPy; no torch dependency)
# ---------------------------------------------------------------------------


def _cutlass_to_cupy_dtype(cutlass_dtype):
    if cutlass_dtype == cutlass.Float16:
        return cp.float16
    if cutlass_dtype == cutlass.BFloat16:
        # CuPy lacks a native bf16 dtype; store as raw uint16 bytes — `from_dlpack`
        # plus `element_type = cutlass.BFloat16` interpret it correctly device-side.
        return cp.uint16
    raise ValueError(f"Unsupported cutlass dtype for CuPy: {cutlass_dtype}")


def _create_bsnd_tensor(
    b: int,
    s: int,
    h: int,
    d: int,
    dtype: Type[cutlass.Numeric],
    *,
    fill_random: bool,
):
    """Allocate a BSND CuPy tensor and return the ``cute.Tensor`` wrapper.

    For bf16 the storage is `uint16` (CuPy has no native bf16); the
    returned `cute.Tensor` is tagged ``element_type = cutlass.BFloat16`` so
    the kernel sees the correct dtype.  Strides match a contiguous
    ``(B, S, H, D)`` packing with ``D`` innermost.
    """
    shape = (b, s, h, d)
    cp_dtype = _cutlass_to_cupy_dtype(dtype)
    if fill_random:
        if dtype == cutlass.Float16:
            arr = cp.random.uniform(-1.0, 1.0, shape).astype(cp_dtype)
        else:
            # bf16: pack a small fp32 random tensor into the upper 16 bits.
            f32 = cp.random.uniform(-1.0, 1.0, shape).astype(cp.float32)
            arr = cp.ascontiguousarray(
                (f32.view(cp.uint32) >> 16).astype(cp.uint16)
            )
    else:
        arr = cp.zeros(shape, dtype=cp_dtype)

    t = from_dlpack(arr, assumed_align=16)
    t.element_type = dtype
    # B / S / H are runtime-dynamic; D is compile-time-known per AOT variant.
    # `mark_compact_shape_dynamic` alone handles the dynamic-shape marking and
    # propagates the static D=512 contribution into the outer strides — that
    # static factor is what lets the IR verifier prove 16-byte alignment for
    # the 128-bit cp.async source pointer.  We deliberately do NOT call
    # `mark_layout_dynamic` here because it would mark every stride dynamic
    # (including the one carrying the static D=512), which strips the
    # alignment-relevant info and fails IR verification at cute.compile time.
    so = (0, 1, 2, 3)
    t = (
        t.mark_compact_shape_dynamic(mode=0, stride_order=so)
        .mark_compact_shape_dynamic(mode=1, stride_order=so)
        .mark_compact_shape_dynamic(mode=2, stride_order=so)
    )
    return t, arr


def _create_block_range_tensor(batch_size: int, seqlen: int, fill: int = -1):
    """(B, S) Int32 vision-block interval tensor (``mBlockBegin`` / ``mBlockEnd``).

    The -1 fill is the text-row sentinel: the ``[begin, end]`` interval is
    empty, so the kernel degenerates to plain causal masking.  B and S are
    runtime-dynamic; the row stride is derived from S (compact packing).
    """
    arr = cp.full((batch_size, seqlen), fill, dtype=cp.int32)
    t = from_dlpack(arr, assumed_align=16)
    so = (0, 1)
    t = t.mark_compact_shape_dynamic(mode=0, stride_order=so).mark_compact_shape_dynamic(
        mode=1, stride_order=so
    )
    return t, arr


def _create_cu_seqlens_tensor(batch_size: int, seqlen: int):
    """(B+1,) Int32 cumulative sequence lengths for uniform per-batch ``seqlen``.

    Matches the ``fmha_cutedsl_blackwell/fmha.py`` convention: element ``b``
    holds the running total of the first ``b`` sequence lengths, so the
    per-batch length is ``cu[b+1] - cu[b]``.  Uniform lengths make the varlen
    masking degenerate to the padded extents (dense behaviour).
    """
    arr = cp.arange(batch_size + 1, dtype=cp.int32) * seqlen
    t = from_dlpack(arr, assumed_align=16)
    t = t.mark_layout_dynamic(leading_dim=0).mark_compact_shape_dynamic(
        mode=0, stride_order=(0,)
    )
    return t, arr


# ---------------------------------------------------------------------------
# run(): test + AOT export entry point
# ---------------------------------------------------------------------------


def run(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    softmax_scale: float = 0.0,
    m_block_size: int = 64,
    n_block_size: int = 16,
    num_threads: int = 128,
    is_causal: bool = False,
    kv_group_size: int = 1,
    vision_block: bool = False,
    skip_rescale: bool = True,
    hybrid_exp2: bool = False,
    warmup_iterations: int = 3,
    iterations: int = 10,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    export_only: bool = False,
    output_dir: str = "./ffpa_aot_artifacts",
    file_name: str = "ffpa",
    function_prefix: str = "ffpa",
    **kwargs,
):
    """Compile (+ optionally test/benchmark or export) the FFPA Ampere kernel.

    AOT export uses dummy placeholder shapes; only ``head_dim``, ``is_causal``,
    ``vision_block`` and the (Br, Bc, threads) tuning are baked at compile
    time — the rest, including ``num_kv_heads`` (GQA), are runtime-dynamic.
    ``kv_group_size`` here only selects the dummy
    ``num_kv_heads = num_head // kv_group_size`` used to trace the kernel; it
    is not baked in.  ``vision_block=True`` compiles the Gemma4 vision-block
    overlay variant, which adds the two ``(B, S_q)`` Int32 ``mBlockBegin`` /
    ``mBlockEnd`` tensors to the kernel ABI (see module docstring).
    """
    _tag = f"[{file_name}]"

    if not FFPAFmhaAmpere.can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
    ):
        raise ValueError(
            f"{_tag} Unsupported config: dtype={dtype}, head_dim={head_dim}, "
            f"Br={m_block_size}, Bc={n_block_size}, threads={num_threads}, "
            f"is_causal={is_causal}"
        )

    if num_head % kv_group_size != 0:
        raise ValueError(
            f"{_tag} num_head ({num_head}) must be divisible by kv_group_size ({kv_group_size})"
        )

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU is required to run this kernel.")

    if softmax_scale <= 0.0:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if export_only:
        print(
            f"{_tag} Compiling FFPA Ampere: dtype={dtype}, head_dim={head_dim}, "
            f"is_causal={is_causal}, kv_group={kv_group_size}, "
            f"Br={m_block_size}, Bc={n_block_size}, threads={num_threads}, "
            f"skip_rescale={skip_rescale}"
        )
    else:
        print(f"{_tag} Running FFPA Ampere FMHA forward with:")
        print(f"{_tag}   dtype={dtype}, head_dim={head_dim}")
        print(f"{_tag}   B={batch_size}, S_q={seqlen_q}, S_k={seqlen_k}, "
              f"H_q={num_head}, kv_group={kv_group_size}")
        print(f"{_tag}   softmax_scale={softmax_scale}, is_causal={is_causal}")
        print(f"{_tag}   Br={m_block_size}, Bc={n_block_size}, threads={num_threads}")
        print(f"{_tag}   skip_rescale={skip_rescale}, hybrid_exp2={hybrid_exp2}")
        print(f"{_tag}   warmup={warmup_iterations}, iterations={iterations}, "
              f"skip_ref_check={skip_ref_check}, use_cold_l2={use_cold_l2}")

    h_q = num_head
    h_kv = h_q // kv_group_size

    q_dyn, q_arr = _create_bsnd_tensor(
        batch_size, seqlen_q, h_q, head_dim, dtype, fill_random=not export_only
    )
    k_dyn, k_arr = _create_bsnd_tensor(
        batch_size, seqlen_k, h_kv, head_dim, dtype, fill_random=not export_only
    )
    v_dyn, v_arr = _create_bsnd_tensor(
        batch_size, seqlen_k, h_kv, head_dim, dtype, fill_random=not export_only
    )
    o_dyn, o_arr = _create_bsnd_tensor(
        batch_size, seqlen_q, h_q, head_dim, dtype, fill_random=False
    )
    cu_q_dyn, cu_q_arr = _create_cu_seqlens_tensor(batch_size, seqlen_q)
    cu_k_dyn, cu_k_arr = _create_cu_seqlens_tensor(batch_size, seqlen_k)
    block_begin_dyn = None
    block_end_dyn = None
    if vision_block:
        block_begin_dyn, _ = _create_block_range_tensor(batch_size, seqlen_q)
        block_end_dyn, _ = _create_block_range_tensor(batch_size, seqlen_q)

    fa2_fwd = FFPAFmhaAmpere(
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        is_causal=is_causal,
        skip_rescale=skip_rescale,
        hybrid_exp2=hybrid_exp2,
    )

    current_stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    # Optional ptxas pass-through (FFPA_PTXAS_VERBOSE=1 / FFPA_PTXAS_OPTS=...).
    _ptx_parts = []
    if os.getenv("FFPA_PTXAS_VERBOSE"):
        _ptx_parts.append("-v")
    _extra_ptx = os.getenv("FFPA_PTXAS_OPTS", "")
    if _extra_ptx:
        _ptx_parts.append(_extra_ptx)
    compile_options = (
        {"options": f"--ptxas-options={','.join(_ptx_parts)}"} if _ptx_parts else {}
    )

    print(f"{_tag} Compiling kernel...")
    t0 = time.time()
    compiled_fa2 = cute.compile(
        fa2_fwd,
        q_dyn,
        k_dyn,
        v_dyn,
        o_dyn,
        cu_q_dyn,
        cu_k_dyn,
        block_begin_dyn,
        block_end_dyn,
        softmax_scale,
        h_kv,
        current_stream,
        **compile_options,
    )
    print(f"{_tag} Compilation time: {time.time() - t0:.4f}s")

    if export_only:
        os.makedirs(output_dir, exist_ok=True)
        compiled_fa2.export_to_c(
            file_path=output_dir,
            file_name=file_name,
            function_prefix=function_prefix,
        )
        print(f"{_tag} Exported to {output_dir}/{file_name}.h and {file_name}.o")
        return None

    # Run + (optional) sanity check.  No low-precision NumPy reference is
    # bundled here — correctness is covered by the C++ unit test
    # `unittests/cuteDslFFPARunnerTest.cpp` (compares against a FP32 BSHD
    # reference); this CLI path only smoke-checks that the launch is
    # well-formed when not exporting.
    compiled_fa2(
        q_dyn, k_dyn, v_dyn, o_dyn, cu_q_dyn, cu_k_dyn, block_begin_dyn,
        block_end_dyn, softmax_scale, h_kv, current_stream,
    )
    cp.cuda.Device().synchronize()

    def generate_tensors():
        q_w, _ = _create_bsnd_tensor(
            batch_size, seqlen_q, h_q, head_dim, dtype, fill_random=True
        )
        k_w, _ = _create_bsnd_tensor(
            batch_size, seqlen_k, h_kv, head_dim, dtype, fill_random=True
        )
        v_w, _ = _create_bsnd_tensor(
            batch_size, seqlen_k, h_kv, head_dim, dtype, fill_random=True
        )
        o_w, _ = _create_bsnd_tensor(
            batch_size, seqlen_q, h_q, head_dim, dtype, fill_random=False
        )
        cu_q_w, _ = _create_cu_seqlens_tensor(batch_size, seqlen_q)
        cu_k_w, _ = _create_cu_seqlens_tensor(batch_size, seqlen_k)
        block_begin_w = None
        block_end_w = None
        if vision_block:
            block_begin_w, _ = _create_block_range_tensor(batch_size, seqlen_q)
            block_end_w, _ = _create_block_range_tensor(batch_size, seqlen_q)
        return testing.JitArguments(
            q_w, k_w, v_w, o_w, cu_q_w, cu_k_w, block_begin_w, block_end_w,
            softmax_scale, h_kv, current_stream
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = sum(
            arr.nbytes for arr in (q_arr, k_arr, v_arr, o_arr)
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    avg_time_us = testing.benchmark(
        compiled_fa2,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    # FMHA forward FLOPs: 4 * B * H * Sq * Sk * D (Q@K^T + P@V, fp32 acc).
    causal_factor = 0.5 if is_causal else 1.0
    flops = (
        4.0 * batch_size * h_q * seqlen_q * seqlen_k * head_dim * causal_factor
    )
    tflops = flops / (avg_time_us * 1e-6) / 1e12
    print(f"{_tag} avg_time_us: {avg_time_us:.2f}  |  {tflops:.2f} TFLOPS ({dtype})")
    return avg_time_us


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CuTe DSL FFPA-style Ampere FMHA forward: test + AOT export."
    )
    p.add_argument("--dtype", type=cutlass.dtype, default=cutlass.BFloat16,
                   help="Input/output dtype: Float16 or BFloat16 (default: BFloat16).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seqlen_q", type=int, default=128)
    p.add_argument("--seqlen_k", type=int, default=128)
    p.add_argument("--num_head", type=int, default=4,
                   help="Number of Q heads (H_q). Must be divisible by kv_group_size.")
    p.add_argument("--head_dim", type=int, default=512)
    p.add_argument("--kv_group_size", type=int, default=1,
                   help="GQA group size H_q / H_kv (1 = MHA; default).")
    p.add_argument("--softmax_scale", type=float, default=0.0,
                   help="Softmax scale; 0 (default) => 1 / sqrt(head_dim).")
    # D=512 validated default on sm_86 (3090):
    #   Br=64, Bc=16, threads=128.  SMEM = (64 + 32) * 512 * 2 = 96 KB
    #   (within the 99 KB opt-in budget).  acc_O = 256 fp32 regs / thread →
    #   spills to local; correctness-first.  The no-spill alternative is
    #   Br=32 / Bc=32 / threads=64 (~33 % slower).
    p.add_argument("--m_block_size", type=int, default=64)
    p.add_argument("--n_block_size", type=int, default=16)
    p.add_argument("--num_threads", type=int, default=128)
    p.add_argument("--is_causal", action="store_true",
                   help="Compile-time enable causal mask.")
    p.add_argument("--vision_block", action="store_true",
                   help="Compile-time enable the Gemma4 vision-block overlay "
                        "(adds (B, S_q) Int32 mBlockBegin/mBlockEnd tensors to "
                        "the ABI; requires --is_causal).")
    p.add_argument("--skip_rescale", action="store_true",
                   help="Tier-1: clamp rescale factor to 1.0 when within 2^-8 of unity.")
    p.add_argument("--hybrid_exp2", action="store_true",
                   help="Scaffold for FA4-style 75%% MUFU + 25%% polynomial exp2 (not validated).")
    p.add_argument("--warmup_iterations", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--use_cold_l2", action="store_true",
                   help="Use circular buffer tensor sets to ensure L2 cold cache.")
    p.add_argument("--export_only", action="store_true",
                   help="Compile and export only; skip reference check and benchmark.")
    p.add_argument("--output_dir", type=str, default="./ffpa_aot_artifacts",
                   help="Output directory for AOT artifacts (<file_name>.{h,o}).")
    p.add_argument("--file_name", type=str, default="ffpa",
                   help="Base file name for exported artifacts.")
    p.add_argument("--function_prefix", type=str, default="ffpa",
                   help="Function prefix for exported C symbols.")
    return p.parse_known_args(args=argv)[0]


def main():
    args = _parsed_args
    run(
        dtype=args.dtype,
        batch_size=args.batch_size,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        num_head=args.num_head,
        head_dim=args.head_dim,
        softmax_scale=args.softmax_scale,
        m_block_size=args.m_block_size,
        n_block_size=args.n_block_size,
        num_threads=args.num_threads,
        is_causal=args.is_causal,
        kv_group_size=args.kv_group_size,
        vision_block=args.vision_block,
        skip_rescale=args.skip_rescale,
        hybrid_exp2=args.hybrid_exp2,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        use_cold_l2=args.use_cold_l2,
        export_only=args.export_only,
        output_dir=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
    print("PASS")
