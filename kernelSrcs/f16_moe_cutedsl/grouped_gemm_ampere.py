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
"""Ampere FP16 device-only grouped GEMM for the FP16 MoE plugin.

The kernel computes 128 independent row-major GEMMs using problem shapes,
strides, and addresses that remain resident on the device::

    D[M, N] = A[M, K] @ B[N, K].T

CUTLASS DSL's static persistent grouped scheduler assigns output tiles to a
runtime-sized persistent CTA grid. It searches the device shape array directly
and therefore does not require a host readback of expert row counts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

# Reuse the repository Ampere GEMM's copy/layout helpers.  Kernel scripts are
# executed directly, so add its directory before importing the module.
_GEMM_CUTEDSL_DIR = Path(__file__).resolve().parents[1] / "gemm_cutedsl"
if str(_GEMM_CUTEDSL_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMM_CUTEDSL_DIR))

from gemm_ampere import GemmAmpereFP16  # noqa: E402


class F16MoeGroupedGemmAmpere(GemmAmpereFP16):
    """Persistent grouped GEMM using Ampere ``cp.async`` and warp MMA.

    The CTA tile is intentionally short in M for decode/prefill MoE routing and
    uses a 64-element K tile so the three-stage pipeline fits every supported
    Ampere-family target, including devices with a 100-KiB block limit.
    """

    def __init__(self):
        super().__init__(
            ab_dtype=cutlass.Float16,
            c_dtype=cutlass.Float16,
            acc_dtype=cutlass.Float32,
            cta_tiler_mnk=(16, 128, 64),
            num_stages=3,
            atom_layout_mnk=(1, 4, 1),
        )

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_d: cute.Tensor,
        problem_shapes: cute.Tensor,
        strides: cute.Tensor,
        addresses: cute.Tensor,
        scratch: cute.Tensor,
        group_count: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Launch a fixed-grid grouped GEMM from device-resident descriptors.

        ``initial_a``, ``initial_b``, and ``initial_d`` provide the AOT compiler
        with tensor element types and majorness only.  Runtime matrix addresses
        come exclusively from ``addresses``.

        Descriptor layouts are ``problem_shapes[128,4]`` (M, N, K, L),
        ``strides[128,3,2]`` (A, B, D), and ``addresses[128,3]`` (A, B, D).
        ``scratch`` is UInt8 ``[128,3,128]`` backing storage retained for the
        cross-architecture AOT ABI; the Ampere implementation does not use it.
        L must be one.  A and B are K-major row-major tensors; D is N-major
        row-major.  All contiguous dimensions must be 16-byte aligned.
        """
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a)
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b)
        self.c_major_mode = utils.LayoutEnum.from_tensor(initial_d)

        if cutlass.const_expr(initial_a.element_type != cutlass.Float16):
            raise TypeError("Ampere grouped GEMM requires FP16 A")
        if cutlass.const_expr(initial_b.element_type != cutlass.Float16):
            raise TypeError("Ampere grouped GEMM requires FP16 B")
        if cutlass.const_expr(initial_d.element_type != cutlass.Float16):
            raise TypeError("Ampere grouped GEMM requires FP16 D")
        if cutlass.const_expr(problem_shapes.element_type != cutlass.Int32):
            raise TypeError(
                "Ampere grouped GEMM problem shapes must use Int32")
        if cutlass.const_expr(strides.element_type != cutlass.Int32):
            raise TypeError("Ampere grouped GEMM strides must use Int32")
        if cutlass.const_expr(addresses.element_type != cutlass.Int64):
            raise TypeError("Ampere grouped GEMM addresses must use Int64")
        if cutlass.const_expr(scratch.element_type not in
                              (cutlass.Int8, cutlass.Uint8)):
            raise TypeError("Ampere grouped GEMM ABI scratch must use bytes")
        if cutlass.const_expr(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("Ampere grouped GEMM requires row-major A")
        if cutlass.const_expr(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("Ampere grouped GEMM requires row-major B")
        if cutlass.const_expr(self.c_major_mode != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("Ampere grouped GEMM requires row-major D")

        copy_bits = 128
        s_a_layout = self._make_smem_layout_AB(
            cutlass.Float16,
            self.a_major_mode,
            copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        s_b_layout = self._make_smem_layout_AB(
            cutlass.Float16,
            self.b_major_mode,
            copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        s_d_layout = self._make_smem_layout_C(
            cutlass.Float16,
            self.c_major_mode,
            copy_bits,
            (self.bM, self.bN),
        )

        smem_size = max(
            cute.size_in_bytes(cutlass.Float16, s_d_layout),
            cute.size_in_bytes(cutlass.Float16, s_a_layout) +
            cute.size_in_bytes(cutlass.Float16, s_b_layout),
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_a = self._make_gmem_tiled_copy_AB(atom_async_copy,
                                                     cutlass.Float16,
                                                     self.a_major_mode,
                                                     copy_bits)
        tiled_copy_b = self._make_gmem_tiled_copy_AB(atom_async_copy,
                                                     cutlass.Float16,
                                                     self.b_major_mode,
                                                     copy_bits)

        atom_d_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_d = self._make_gmem_tiled_copy_C(atom_d_copy,
                                                    cutlass.Float16,
                                                    self.c_major_mode,
                                                    copy_bits)

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32,
                                              self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (1, 1, max_active_clusters), (1, 1, 1))

        self.kernel(
            problem_shapes,
            strides,
            addresses,
            group_count,
            tile_sched_params,
            s_a_layout,
            s_b_layout,
            s_d_layout,
            tiled_copy_a,
            tiled_copy_b,
            tiled_copy_d,
            tiled_mma,
        ).launch(
            # StaticPersistentGroupTileScheduler seeds work from blockIdx.z.
            grid=[1, 1, max_active_clusters],
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        problem_shapes: cute.Tensor,
        strides: cute.Tensor,
        addresses: cute.Tensor,
        group_count: cutlass.Int32,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        s_a_layout: cute.ComposedLayout,
        s_b_layout: cute.ComposedLayout,
        s_d_layout: cute.ComposedLayout,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
        tiled_copy_d: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_sched = utils.StaticPersistentGroupTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            self.cta_tiler,
            utils.create_initial_search_state(),
            group_count,
            problem_shapes,
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            tile_info = work_tile.group_search_result
            if tile_info.problem_shape_m > 0 and tile_info.cta_tile_count_k > 0:
                problem_shape_mnk = (
                    tile_info.problem_shape_m,
                    tile_info.problem_shape_n,
                    tile_info.problem_shape_k,
                )
                m_a = self._make_group_tensor(
                    tile_info.group_idx,
                    0,
                    cutlass.Float16,
                    problem_shape_mnk,
                    strides,
                    addresses,
                )
                m_b = self._make_group_tensor(
                    tile_info.group_idx,
                    1,
                    cutlass.Float16,
                    problem_shape_mnk,
                    strides,
                    addresses,
                )
                m_d = self._make_group_tensor(
                    tile_info.group_idx,
                    2,
                    cutlass.Float16,
                    problem_shape_mnk,
                    strides,
                    addresses,
                )
                self._compute_tile(
                    m_a,
                    m_b,
                    m_d,
                    tile_info.cta_tile_idx_m,
                    tile_info.cta_tile_idx_n,
                    s_a_layout,
                    s_b_layout,
                    s_d_layout,
                    tiled_copy_a,
                    tiled_copy_b,
                    tiled_copy_d,
                    tiled_mma,
                    tidx,
                )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    @cute.jit
    def _make_group_tensor(
        self,
        group_idx: cutlass.Int32,
        tensor_index: int,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides: cute.Tensor,
        addresses: cute.Tensor,
    ):
        pointer = cute.make_ptr(
            dtype,
            addresses[(group_idx, tensor_index)],
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        stride = cute.make_rmem_tensor(cute.make_layout(2), cutlass.Int32)
        cute.autovec_copy(strides[(group_idx, tensor_index, None)], stride)

        if cutlass.const_expr(tensor_index == 0):
            return cute.make_tensor(
                pointer,
                cute.make_layout(
                    (problem_shape_mnk[0], problem_shape_mnk[2]),
                    stride=(stride[0], 1),
                ),
            )
        if cutlass.const_expr(tensor_index == 1):
            return cute.make_tensor(
                pointer,
                cute.make_layout(
                    (problem_shape_mnk[1], problem_shape_mnk[2]),
                    stride=(stride[0], 1),
                ),
            )
        return cute.make_tensor(
            pointer,
            cute.make_layout(
                (problem_shape_mnk[0], problem_shape_mnk[1]),
                stride=(stride[0], 1),
            ),
        )

    @cute.jit
    def _compute_tile(
        self,
        m_a: cute.Tensor,
        m_b: cute.Tensor,
        m_d: cute.Tensor,
        tile_m: cutlass.Int32,
        tile_n: cutlass.Int32,
        s_a_layout: cute.ComposedLayout,
        s_b_layout: cute.ComposedLayout,
        s_d_layout: cute.ComposedLayout,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
        tiled_copy_d: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
    ):
        tile_coord = (tile_m, tile_n, None)
        g_a = cute.local_tile(m_a,
                              tiler=self.cta_tiler,
                              coord=tile_coord,
                              proj=(1, None, 1))
        g_b = cute.local_tile(m_b,
                              tiler=self.cta_tiler,
                              coord=tile_coord,
                              proj=(None, 1, 1))
        g_d = cute.local_tile(m_d,
                              tiler=self.cta_tiler,
                              coord=tile_coord,
                              proj=(1, 1, None))

        # The plugin contract makes every FC1/FC2 K dimension a multiple of the
        # 64-element CTA K tile. Avoiding the generic ragged-K domain offset
        # keeps the 128-bit cp.async alignment statically provable.
        g_a = cute.make_tensor(g_a.iterator.align(16), g_a.layout)
        g_b = cute.make_tensor(g_b.iterator.align(16), g_b.layout)

        identity_a = cute.make_identity_tensor(m_a.layout.shape)
        identity_b = cute.make_identity_tensor(m_b.layout.shape)
        coord_a = cute.local_tile(identity_a,
                                  tiler=self.cta_tiler,
                                  coord=tile_coord,
                                  proj=(1, None, 1))
        coord_b = cute.local_tile(identity_b,
                                  tiler=self.cta_tiler,
                                  coord=tile_coord,
                                  proj=(None, 1, 1))
        smem = cutlass.utils.SmemAllocator()
        s_a = smem.allocate_tensor(cutlass.Float16, s_a_layout, 16)
        s_b = smem.allocate_tensor(cutlass.Float16, s_b_layout, 16)
        s_d = cute.make_tensor(
            cute.recast_ptr(s_a.iterator, dtype=cutlass.Float16), s_d_layout)

        thread_copy_a = tiled_copy_a.get_slice(tidx)
        thread_copy_b = tiled_copy_b.get_slice(tidx)
        thread_copy_d = tiled_copy_d.get_slice(tidx)
        thread_g_a = thread_copy_a.partition_S(g_a)
        thread_s_a = thread_copy_a.partition_D(s_a)
        thread_g_b = thread_copy_b.partition_S(g_b)
        thread_s_b = thread_copy_b.partition_D(s_b)
        thread_s_d_epilogue = thread_copy_d.partition_S(s_d)
        thread_g_d_epilogue = thread_copy_d.partition_D(g_d)
        # Dynamic leading strides make partitioning conservatively drop the
        # ABI's 16-byte alignment. Each thread partition still starts on one
        # full 128-bit copy vector, so restore that fact at the copy boundary.
        thread_g_a = cute.make_tensor(
            thread_g_a.iterator.align(16), thread_g_a.layout)
        thread_g_b = cute.make_tensor(
            thread_g_b.iterator.align(16), thread_g_b.layout)
        thread_g_d_epilogue = cute.make_tensor(
            thread_g_d_epilogue.iterator.align(16),
            thread_g_d_epilogue.layout)
        thread_coord_a = thread_copy_a.partition_S(coord_a)
        thread_coord_b = thread_copy_b.partition_S(coord_b)

        predicate_a = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    thread_g_a.shape[0][1],
                    cute.size(thread_g_a, mode=[1]),
                    cute.size(thread_g_a, mode=[2]),
                ),
                stride=(cute.size(thread_g_a, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        predicate_b = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    thread_s_b.shape[0][1],
                    cute.size(thread_s_b, mode=[1]),
                    cute.size(thread_s_b, mode=[2]),
                ),
                stride=(cute.size(thread_s_b, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(predicate_a.shape[0]):
            for row in range(predicate_a.shape[1]):
                predicate_a[rest_v, row, 0] = cute.elem_less(
                    thread_coord_a[(0, rest_v), row, 0, 0][0], m_a.shape[0])
        for rest_v in range(predicate_b.shape[0]):
            for column in range(predicate_b.shape[1]):
                predicate_b[rest_v, column, 0] = cute.elem_less(
                    thread_coord_b[(0, rest_v), column, 0, 0][0], m_b.shape[0])

        thread_s_a.fill(0)
        thread_s_b.fill(0)
        cute.arch.sync_threads()
        num_smem_stages = cute.size(thread_s_a, mode=[3])
        k_tile_count = cute.size(thread_g_a, mode=[3])
        k_tile_index = cutlass.Int32(0)

        for k in range(predicate_a.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), thread_coord_a[0, 0, k,
                                                                0][1]):
                cute.copy(
                    tiled_copy_a,
                    thread_g_a[None, None, k, k_tile_index],
                    thread_s_a[None, None, k, 0],
                    pred=predicate_a[None, None, k],
                )
        for k in range(predicate_b.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), thread_coord_b[0, 0, k,
                                                                0][1]):
                cute.copy(
                    tiled_copy_b,
                    thread_g_b[None, None, k, k_tile_index],
                    thread_s_b[None, None, k, 0],
                    pred=predicate_b[None, None, k],
                )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            if k_tile == k_tile_count:
                predicate_a.fill(0)
                predicate_b.fill(0)
            cute.copy(
                tiled_copy_a,
                thread_g_a[None, None, None, k_tile_index],
                thread_s_a[None, None, None, k_tile],
                pred=predicate_a,
            )
            cute.copy(
                tiled_copy_b,
                thread_g_b[None, None, None, k_tile_index],
                thread_s_b[None, None, None, k_tile],
                pred=predicate_b,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        thread_mma = tiled_mma.get_slice(tidx)
        thread_mma_s_a = thread_mma.partition_A(s_a)
        thread_mma_s_b = thread_mma.partition_B(s_b)
        thread_mma_s_d = thread_mma.partition_C(s_d)
        thread_mma_g_d = thread_mma.partition_C(g_d)
        fragment_a = tiled_mma.make_fragment_A(thread_mma_s_a[None, None, None,
                                                              0])
        fragment_b = tiled_mma.make_fragment_B(thread_mma_s_b[None, None, None,
                                                              0])
        accumulator = tiled_mma.make_fragment_C(thread_mma_g_d)
        accumulator.fill(0.0)

        atom_s2r_a = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Float16)
        atom_s2r_b = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Float16)
        tiled_s2r_a = cute.make_tiled_copy_A(atom_s2r_a, tiled_mma)
        tiled_s2r_b = cute.make_tiled_copy_B(atom_s2r_b, tiled_mma)
        thread_s2r_a = tiled_s2r_a.get_slice(tidx)
        thread_s2r_b = tiled_s2r_b.get_slice(tidx)
        copy_view_s_a = thread_s2r_a.partition_S(s_a)
        copy_view_r_a = thread_s2r_a.retile(fragment_a)
        copy_view_s_b = thread_s2r_b.partition_S(s_b)
        copy_view_r_b = thread_s2r_b.retile(fragment_b)

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        piped_s_a = copy_view_s_a[None, None, None, smem_pipe_read]
        piped_s_b = copy_view_s_b[None, None, None, smem_pipe_read]
        num_k_blocks = cute.size(fragment_a, mode=[2])

        if num_k_blocks > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(
                tiled_s2r_a,
                piped_s_a[None, None, 0],
                copy_view_r_a[None, None, 0],
            )
            cute.copy(
                tiled_s2r_b,
                piped_s_b[None, None, 0],
                copy_view_r_b[None, None, 0],
            )

        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_blocks, unroll_full=True):
                if k_block == num_k_blocks - 1:
                    piped_s_a = copy_view_s_a[None, None, None, smem_pipe_read]
                    piped_s_b = copy_view_s_b[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                next_k_block = (k_block + 1) % num_k_blocks
                cute.copy(
                    tiled_s2r_a,
                    piped_s_a[None, None, next_k_block],
                    copy_view_r_a[None, None, next_k_block],
                )
                cute.copy(
                    tiled_s2r_b,
                    piped_s_b[None, None, next_k_block],
                    copy_view_r_b[None, None, next_k_block],
                )

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_a,
                            thread_g_a[None, None, None, k_tile_index],
                            thread_s_a[None, None, None, smem_pipe_write],
                            pred=predicate_a,
                        )

                cute.gemm(
                    tiled_mma,
                    accumulator,
                    fragment_a[None, None, k_block],
                    fragment_b[None, None, k_block],
                    accumulator,
                )

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_b,
                            thread_g_b[None, None, None, k_tile_index],
                            thread_s_b[None, None, None, smem_pipe_write],
                            pred=predicate_b,
                        )
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        fragment_d = cute.make_fragment_like(accumulator, cutlass.Float16)
        fragment_d[None] = accumulator.load().to(cutlass.Float16)
        cute.autovec_copy(fragment_d, thread_mma_s_d)

        register_d = cute.make_fragment_like(thread_s_d_epilogue)
        cute.arch.sync_threads()
        cute.autovec_copy(thread_s_d_epilogue, register_d)

        ceil_m, ceil_n = cute.ceil_div(m_d.shape, (self.bM, self.bN))
        identity_d = cute.make_identity_tensor(
            (cute.size(ceil_m) * self.bM, cute.size(ceil_n) * self.bN))
        coord_d = cute.local_tile(
            identity_d,
            tiler=self.cta_tiler,
            coord=(tile_m, tile_n, None),
            proj=(1, 1, None),
        )
        thread_coord_d = thread_copy_d.partition_S(coord_d)
        predicate_d = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    thread_g_d_epilogue.shape[0][1],
                    cute.size(thread_g_d_epilogue, mode=[1]),
                    cute.size(thread_g_d_epilogue, mode=[2]),
                ),
                stride=(cute.size(thread_g_d_epilogue, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(predicate_d.shape[0]):
            for row in range(predicate_d.shape[1]):
                predicate_d[rest_v, row, 0] = cute.elem_less(
                    thread_coord_d[(0, rest_v), row, 0][0], m_d.shape[0])

        for rest_v in range(predicate_d.shape[0]):
            for column in range(predicate_d.shape[2]):
                if cute.elem_less(thread_coord_d[(0, rest_v), 0, column][1],
                                  m_d.shape[1]):
                    cute.copy(
                        tiled_copy_d,
                        register_d[None, None, column],
                        thread_g_d_epilogue[None, None, column],
                        pred=predicate_d[None, None, column],
                    )

        cute.arch.sync_threads()
