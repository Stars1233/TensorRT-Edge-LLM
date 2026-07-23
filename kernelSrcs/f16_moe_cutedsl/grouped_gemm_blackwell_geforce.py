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
"""Raw FP16 grouped GEMM for Blackwell GeForce SM120/SM121.

The kernel computes ``D = A @ B.T`` independently for every expert. Problem
shapes, strides, and addresses stay device-resident. A fixed persistent grid
uses the CUTLASS grouped scheduler to skip zero-row experts and distribute the
remaining CTA tiles.

This module deliberately implements only GEMM. Activation and routed-output
finalization are common kernels outside the architecture-specific AOT module.
"""

from __future__ import annotations

import pathlib
import sys
from inspect import isclass
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

# Reuse the validated SM12x dense mainloop configuration and its layout/stage
# helpers. The grouped kernel below replaces only launch, scheduling, and TMA
# descriptor management.
_GEMM_SOURCE_DIR = pathlib.Path(__file__).resolve().parents[1] / "gemm_cutedsl"
if str(_GEMM_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_GEMM_SOURCE_DIR))

from gemm_blackwell_geforce import GemmBlackwellGeforceFP16  # noqa: E402


class F16MoeGroupedGemmBlackwellGeforce(GemmBlackwellGeforceFP16):
    """Persistent SM12x grouped GEMM with device-only problem metadata.

    Matrix layouts are fixed by the plugin ABI:

    * A: ``[M, K]`` row-major FP16.
    * B: ``[N, K]`` row-major FP16, consumed as ``B.T``.
    * D: ``[M, N]`` row-major FP16.
    * Accumulator: FP32.

    ``tensormap_scratch`` must contain at least
    ``max_active_clusters * 3 * 128`` UInt8 elements. Every persistent CTA owns
    three aligned 128-byte descriptors (A, B, D), which it updates directly in
    global memory whenever its scheduled expert changes.
    """

    bytes_per_tensormap = 128
    num_tensormaps = 3

    def __init__(self, tile_shape_mnk: tuple[int, int, int] = (16, 128, 64)):
        super().__init__(
            acc_dtype=cutlass.Float32,
            tile_shape_mnk=tile_shape_mnk,
        )

        # Keep the dense kernel's validated (2, 2, 1) MMA topology. Its tiled
        # copies and epilogue rely on that warp arrangement even when the CTA
        # uses the narrow M tile preferred by routed decode workloads.

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_d: cute.Tensor,
        problem_shapes: cute.Tensor,
        strides: cute.Tensor,
        addresses: cute.Tensor,
        tensormap_scratch: cute.Tensor,
        group_count: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Launch the grouped GEMM.

        ``initial_a``, ``initial_b``, and ``initial_d`` carry only the element
        type and static majorness needed to create TMA atoms. Real extents,
        strides, and addresses come from the device metadata tensors:

        * ``problem_shapes``: Int32 ``[group_count, 4]`` storing M, N, K, L.
        * ``strides``: Int32 ``[group_count, 3, 2]`` storing two strides for
          A, B, and D respectively.
        * ``addresses``: Int64 ``[group_count, 3]`` storing A, B, and D bases.
        * ``tensormap_scratch``: UInt8
          ``[max_active_clusters, 3, 128]`` descriptor backing storage.
        """
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.c_dtype = initial_d.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(initial_a)
        self.b_layout = utils.LayoutEnum.from_tensor(initial_b)
        self.c_layout = utils.LayoutEnum.from_tensor(initial_d)

        if cutlass.const_expr(self.a_dtype != cutlass.Float16
                              or self.b_dtype != cutlass.Float16
                              or self.c_dtype != cutlass.Float16):
            raise TypeError("SM12x FP16 MoE grouped GEMM requires FP16 A/B/D")
        if cutlass.const_expr(problem_shapes.element_type != cutlass.Int32):
            raise TypeError("problem_shapes must contain Int32 values")
        if cutlass.const_expr(strides.element_type != cutlass.Int32):
            raise TypeError("strides must contain Int32 values")
        if cutlass.const_expr(addresses.element_type != cutlass.Int64):
            raise TypeError("addresses must contain Int64 values")
        if cutlass.const_expr(tensormap_scratch.element_type not in
                              (cutlass.Int8, cutlass.Uint8)):
            raise TypeError("tensormap_scratch must contain byte values")
        if cutlass.const_expr(self.a_layout != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("SM12x grouped GEMM requires row-major A")
        if cutlass.const_expr(self.b_layout != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("SM12x grouped GEMM requires row-major B")
        if cutlass.const_expr(self.c_layout != utils.LayoutEnum.ROW_MAJOR):
            raise ValueError("SM12x grouped GEMM requires row-major D")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            initial_a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            initial_b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_d, tma_tensor_d = self._make_tma_store_atoms_and_tensors(
            initial_d,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            (1, 1, max_active_clusters),
            self.cluster_shape_mnk,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64,
                                                              self.ab_stage *
                                                              2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype,
                                     cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype,
                                     cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype,
                                     cute.cosize(self.epi_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            problem_shapes,
            strides,
            addresses,
            group_count,
            tensormap_scratch,
        ).launch(
            # The CUTLASS grouped scheduler uses blockIdx.z as its persistent
            # linear work index; x/y are reserved for cluster coordinates.
            grid=(1, 1, max_active_clusters),
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        initial_a: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        initial_b: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        initial_d: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        problem_shapes: cute.Tensor,
        strides: cute.Tensor,
        addresses: cute.Tensor,
        group_count: cutlass.Int32,
        tensormap_scratch: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk,
                                                   cluster_coord_mnk,
                                                   mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk,
                                                   cluster_coord_mnk,
                                                   mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout) + cute.size_in_bytes(
                self.b_dtype, b_smem_layout)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr(
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                     self.num_mma_warps),
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
        )

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer,
                                   swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer,
                                   swizzle=b_smem_layout_staged.inner)
        sD = storage.sD.get_tensor(epi_smem_layout_staged.outer,
                                   swizzle=epi_smem_layout_staged.inner)

        gA_mkl = cute.local_tile(
            initial_a,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            initial_b,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gD_mnl = cute.local_tile(
            initial_d,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(tidx)

        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_mnk[0],
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCgD = thr_mma.partition_C(gD_mnl)
        accumulators = cute.make_rmem_tensor(tCgD.shape[:3], self.acc_dtype)

        pipeline.sync(barrier_id=1)

        block_idx = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.GMEM,
            self.bytes_per_tensormap,
        )
        descriptor_slot = block_idx[2]
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_scratch[(descriptor_slot, 0, None)].iterator)
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_scratch[(descriptor_slot, 1, None)].iterator)
        tensormap_d_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_scratch[(descriptor_slot, 2, None)].iterator)

        tile_sched = utils.StaticPersistentGroupTileScheduler.create(
            tile_sched_params,
            block_idx,
            grid_dim,
            self.tile_shape_mnk,
            utils.create_initial_search_state(),
            group_count,
            problem_shapes,
        )
        initial_work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage)
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage)
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps * self.num_threads_per_warp,
            ),
        )

        if warp_idx < self.num_mma_warps and initial_work_tile.is_valid_tile:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            tensormap_manager.init_tensormap_from_atom(tma_atom_d,
                                                       tensormap_d_ptr, 0)
            tensormap_manager.fence_tensormap_initialization()

            atom_copy_ldmatrix_a = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(),
                                                  4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_b = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(),
                                                  4),
                self.b_dtype,
            )
            smem_tiled_copy_a = cute.make_tiled_copy_A(atom_copy_ldmatrix_a,
                                                       tiled_mma)
            smem_tiled_copy_b = cute.make_tiled_copy_B(atom_copy_ldmatrix_b,
                                                       tiled_mma)
            thr_copy_ldmatrix_a = smem_tiled_copy_a.get_slice(tidx)
            thr_copy_ldmatrix_b = smem_tiled_copy_b.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_a.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_a.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_b.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_b.retile(tCrB)
            num_k_blocks = cute.size(tCrA, mode=[2])

            work_tile = initial_work_tile
            last_group_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                tile_info = work_tile.group_search_result
                group_idx = tile_info.group_idx
                k_tile_count = tile_info.cta_tile_count_k
                group_changed = group_idx != last_group_idx

                if group_changed:
                    real_d = self.make_tensor_for_tensormap_update(
                        group_idx,
                        self.c_dtype,
                        (
                            tile_info.problem_shape_m,
                            tile_info.problem_shape_n,
                            tile_info.problem_shape_k,
                        ),
                        strides,
                        addresses,
                        2,
                    )
                    tensormap_manager.update_tensormap(
                        (real_d, ),
                        (tma_atom_d, ),
                        (tensormap_d_ptr, ),
                        0,
                        (None, ),
                    )
                    if warp_idx == 0:
                        tensormap_manager.fence_tensormap_update(
                            tensormap_d_ptr)

                tile_coord_mnl = (
                    tile_info.cta_tile_idx_m,
                    tile_info.cta_tile_idx_n,
                    0,
                )
                gD_mnl_slice = gD_mnl[(None, None, *tile_coord_mnl)]
                accumulators.fill(0.0)

                mainloop_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_count:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state)
                mainloop_pipeline.consumer_wait(mainloop_consumer_state,
                                                peek_ab_full_status)
                tCsA_p = tCsA_copy_view[None, None, None,
                                        mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None,
                                        mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_a,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_b,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                for _ in range(0, k_tile_count - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (0 if k_block_idx +
                                        1 == num_k_blocks else k_block_idx + 1)
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(
                                mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                            peek_ab_full_status = (
                                mainloop_pipeline.consumer_try_wait(
                                    mainloop_consumer_state))
                            tCsA_p = tCsA_copy_view[
                                None,
                                None,
                                None,
                                mainloop_consumer_state.index,
                            ]
                            tCsB_p = tCsB_copy_view[
                                None,
                                None,
                                None,
                                mainloop_consumer_state.index,
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state,
                                peek_ab_full_status,
                            )

                        cute.copy(
                            smem_tiled_copy_a,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_b,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )

                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (0 if k_block_idx +
                                    1 == num_k_blocks else k_block_idx + 1)
                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(
                            mainloop_consumer_state)
                        mainloop_consumer_state.advance()
                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_a,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_b,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA[None, None, k_block_idx],
                        tCrB[None, None, k_block_idx],
                        accumulators,
                    )

                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_d = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
                tiled_copy_d_atom = cute.make_tiled_copy_C_atom(
                    copy_atom_d, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_d_atom)
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sD)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sD))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape,
                                               self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                sD_for_tma = cute.group_modes(sD, 0, 2)
                gD_for_tma = cute.zipped_divide(gD_mnl_slice, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_d,
                    0,
                    cute.make_layout(1),
                    sD_for_tma,
                    gD_for_tma,
                )
                epi_tile_count = cute.size(gD_for_tma, mode=[1])
                epi_tile_shape = gD_for_tma.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape,
                                                   stride=(1,
                                                           epi_tile_shape[0]))

                for epi_idx in cutlass.range_constexpr(epi_tile_count):
                    for epi_value in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_value] = tRS_rAcc[epi_idx * size_tRS_rD +
                                                     epi_value]
                    tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape,
                                                       self.c_dtype)
                    tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                    epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_d,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_d_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = group_idx

            tma_store_pipeline.producer_tail()

        elif (warp_idx == self.num_mma_warps
              and initial_work_tile.is_valid_tile):
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            tensormap_manager.init_tensormap_from_atom(tma_atom_a,
                                                       tensormap_a_ptr,
                                                       self.num_mma_warps)
            tensormap_manager.init_tensormap_from_atom(tma_atom_b,
                                                       tensormap_b_ptr,
                                                       self.num_mma_warps)
            tensormap_manager.fence_tensormap_initialization()

            work_tile = initial_work_tile
            last_group_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                tile_info = work_tile.group_search_result
                group_idx = tile_info.group_idx
                k_tile_count = tile_info.cta_tile_count_k
                group_changed = group_idx != last_group_idx

                if group_changed:
                    problem_shape_mnk = (
                        tile_info.problem_shape_m,
                        tile_info.problem_shape_n,
                        tile_info.problem_shape_k,
                    )
                    real_a = self.make_tensor_for_tensormap_update(
                        group_idx,
                        self.a_dtype,
                        problem_shape_mnk,
                        strides,
                        addresses,
                        0,
                    )
                    real_b = self.make_tensor_for_tensormap_update(
                        group_idx,
                        self.b_dtype,
                        problem_shape_mnk,
                        strides,
                        addresses,
                        1,
                    )
                    tensormap_manager.update_tensormap(
                        (real_a, real_b),
                        (tma_atom_a, tma_atom_b),
                        (tensormap_a_ptr, tensormap_b_ptr),
                        self.num_mma_warps,
                        (None, None),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_ptr)

                tile_coord_mnl = (
                    tile_info.cta_tile_idx_m,
                    tile_info.cta_tile_idx_n,
                    0,
                )
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None,
                                 tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None,
                                 tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()
                for _ in range(0, k_tile_count, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state),
                        mcast_mask=a_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_a_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB[(None, mainloop_producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_b_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = group_idx

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides: cute.Tensor,
        addresses: cute.Tensor,
        tensor_index: int,
    ):
        """Construct one expert tensor from its device-resident metadata."""
        if cutlass.const_expr(not isclass(dtype)
                              or not issubclass(dtype, cutlass.Numeric)):
            raise TypeError(
                f"dtype must derive from cutlass.Numeric, got {dtype}")

        tensor_ptr = cute.make_ptr(
            dtype,
            addresses[(group_idx, tensor_index)],
            cute.AddressSpace.gmem,
            assumed_align=128,
        )
        stride_gmem = strides[(group_idx, tensor_index, None)]
        stride_reg = cute.make_rmem_tensor(cute.make_layout(2),
                                           strides.element_type)
        cute.autovec_copy(stride_gmem, stride_reg)
        leading_stride = stride_reg[0]
        contiguous_stride = stride_reg[1]
        one = cutlass.Int32(1)
        zero = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):
            return cute.make_tensor(
                tensor_ptr,
                cute.make_layout(
                    (problem_shape_mnk[0], problem_shape_mnk[2], one),
                    stride=(leading_stride, contiguous_stride, zero),
                ),
            )
        if cutlass.const_expr(tensor_index == 1):
            return cute.make_tensor(
                tensor_ptr,
                cute.make_layout(
                    (problem_shape_mnk[1], problem_shape_mnk[2], one),
                    stride=(leading_stride, contiguous_stride, zero),
                ),
            )
        return cute.make_tensor(
            tensor_ptr,
            cute.make_layout(
                (problem_shape_mnk[0], problem_shape_mnk[1], one),
                stride=(leading_stride, contiguous_stride, zero),
            ),
        )


# Exporters may use the architecture-neutral spelling across all three AOT
# family modules.
F16MoeGroupedGemm = F16MoeGroupedGemmBlackwellGeforce
