# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from utils import (
    fmax,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
    ldmatrix_m16n16_trans_b8,
    prmt_b32,
    silu_f32,
    st_shared_b16,
    st_shared_b32,
)

"""
Blackwell MoE FC1 kernel: N-major weights with in-flight SMEM nibble transpose.

This is a fork of ``blockscaled_contiguous_grouped_gemm.py``. The goal is to
accept FC1 weights stored in **N-major** layout on disk/device (the layout
consumed by the Marlin-style decode GEMV), so that the same weight buffer can
serve both the prefill GEMM and the decode GEMV with a single copy.

FP4 ``tcgen05.mma`` only supports K-major B (confirmed by
``tests/gpu/smoke_mn_major_b_fp4.py``), so the N-major layout has to be
transposed to K-major *inside* the kernel before the MMA consumes it. The
kernel grows from 7 to 8 warps:

  warp 0-3  epilogue                (unchanged)
  warp 4    MMA tcgen05.mma         (unchanged - reads K-major sB)
  warp 5    TMA loads A / SFA / SFB / B_raw
  warp 6    scheduler               (unchanged)
  warp 7    transpose               (new - repacks sB_raw -> sB nibbles)

Matrix layout:
  - A      : MxKx1, row-major("K"), contiguous expert-grouped
  - B      : (N, K, L) logical shape but **N-major** (N innermost) on device
  - C (out): MxN_outx1, row-major("N"), direct TMA store
  - SFB    : prefill-friendly atom-swizzle copy (M=N, K=K/16)

Other semantics (activation fusion, per-expert alpha, determinism,
StaticPersistentTileScheduler + runtime tile-info lookup) are identical to the
K-major kernel. See ``docs/nvfp4_prefill_n_major_weights_design.md`` for the
design and error budget.
"""


# The monkeypatch of PersistentTileSchedulerParams and StaticPersistentTileScheduler
# that was here has been removed — nvidia-cutlass-dsl >= 4.4 ships the required
# raster_along_m / FastDivmod support natively.


class BlockScaledContiguousGroupedGemmNMajorKernel:
    """Blackwell FC1 kernel: N-major B with in-flight SMEM nibble transpose.

    Implements C = activation((SFA * A) * (SFB * B)) for the gating/up-projection
    (FC1) layer of Mixture-of-Experts blocks. The epilogue applies fused activation
    (identity, relu2, or swiglu) and writes output via TMA store to a contiguous
    output tensor.

    Uses the FC2 kernel's scheduling infrastructure (7 warps with dedicated
    scheduler warp, StaticPersistentTileScheduler + runtime lookup tables) to
    eliminate compile-time group_count/total_num_clusters parameters.

    Only 1CTA mode (tile_size=128) and single B tensor are supported.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N).
        M must be 128.
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]
    :param activation: Fused activation function applied in the epilogue.
        "identity" (default), "relu2" (ReLU squared), or "swiglu".
    :type activation: str

    :note: In current version, A and B tensor must have the same data type

    :note: Supported combinations of A/B data types, SF data types and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Constraints:
        - MMA tiler M must be 128 (1CTA only)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Cluster shape M/N must be <= 4 for scale factor multicasts
        - For swiglu, MMA tiler N must be even; output C tile N is halved
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        activation: str = "identity",
        raster_along_m: bool = False,
    ):
        """Initializes the configuration for a Blackwell contiguous grouped GEMM kernel.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction. M must be 128.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param activation: Activation function to fuse in the epilogue.
            "identity" (default), "relu2" (ReLU squared), or "swiglu".
        :type activation: str
        :param raster_along_m: Boolean, True to use raster along M.
        :type raster_along_m: bool
        """
        assert activation in ("relu2", "swiglu", "identity"), (
            f"Unknown activation: {activation}"
        )
        self.activation = activation
        self.is_swiglu = activation == "swiglu"

        self.sf_vec_size = sf_vec_size
        self.acc_dtype = cutlass.Float32
        # Only 1CTA mode (tile_size=128) is supported
        self.use_2cta_instrs = False
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        # For SwiGLU, C has half the N dimension; tile C accordingly
        self.mma_tiler_c = (
            mma_tiler_mn[0],
            mma_tiler_mn[1] // 2 if self.is_swiglu else mma_tiler_mn[1],
            1,
        )

        self.cta_group = tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        # Transpose warps: N-major B_raw -> K-major sB nibble repack.
        # With 4 warps × 32 lanes = 128 writer threads, each warp owns a
        # contiguous 1/num_transpose_warps slice of the output bytes. This is
        # free occupancy-wise (kernel is SMEM-limited to 1 block/SM already)
        # and cuts the serial store latency proportionally.
        self.num_transpose_warps = 4
        self.transpose_warp_id_base = 7
        self.transpose_warp_ids = tuple(
            self.transpose_warp_id_base + i for i in range(self.num_transpose_warps)
        )
        # Back-compat alias (single-warp code reads self.transpose_warp_id).
        self.transpose_warp_id = self.transpose_warp_id_base
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                self.sched_warp_id,
                *self.transpose_warp_ids,
            )
        )
        # Threads excluding the scheduler warp (consumers of the tile-info pipeline).
        # Transpose warps also consume tile-info so they issue SMEM transposes per tile.
        self.threads_wo_sched = self.threads_per_warp * len(
            (
                *self.epilog_warp_id,
                self.mma_warp_id,
                self.tma_warp_id,
                *self.transpose_warp_ids,
            )
        )
        self.num_regs_uniform_warps = 64
        self.num_regs_sched_warps = 64
        self.num_regs_epilogue_warps = 216
        # Transpose warps read/write SMEM only (no TMEM), but each inner
        # iteration now keeps 8 u8 raw-byte loads + 8 u32 nibble temps + a
        # packed u32 output live simultaneously, so bump the per-thread
        # register budget from the TMA/sched default to reduce spills.
        self.num_regs_transpose_warps = 128

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        # Transpose-warp-group barrier. Used to fence reads of sB_raw across
        # all transpose warps before warp 0 signals b_raw_pipeline that TMA
        # can refill — otherwise TMA could overwrite sB_raw while later
        # transpose warps are still reading it.
        self.transpose_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.threads_per_warp * self.num_transpose_warps,
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        # TMEM offset for final accumulator
        self.tmem_final_offset = 384

        # Single B tensor mode
        self.num_b_tensors = 1

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0],
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Configure tiled mma
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # For SwiGLU, C output has half the N dimension (up*silu(gate) halves width)
        self.cta_tile_shape_mnk_c = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1] // 2 if self.is_swiglu else self.cta_tile_shape_mnk[1],
            self.cta_tile_shape_mnk[2],
        )

        # Compute epilogue subtile (using C-side dimensions for SwiGLU N/2)
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk_c,
            self.use_2cta_instrs,
            self.gemm_output_layout,
            self.out_dtype,
        )

        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.out_dtype,
            self.gemm_output_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.occupancy,
        )

        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        # N-major raw B buffer: TMA drops (N_TILE, K_TILE) tiles with N innermost.
        # This is a plain unswizzled FP4 layout — the transpose warp will later
        # repack its contents into ``b_smem_layout_staged`` (K-major, swizzled,
        # MMA-ready). Stage count matches the K-major sB so the two pipelines
        # stay symmetric.
        n_tile = self.mma_tiler[1]
        k_tile = self.mma_tiler[2]
        self.b_raw_smem_layout_staged = cute.make_ordered_layout(
            (n_tile, k_tile, self.num_ab_stage),
            order=(0, 1, 2),
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.out_dtype,
            self.gemm_output_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        self.overlapping_accum = self.num_acc_stage == 1

        sf_atom_mn = 32

        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )

        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        out: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with fused activation.

        :param a: Input tensor A, shape (m, K, 1), contiguous expert-grouped
        :type a: cute.Tensor
        :param b: Weight tensor B, shape (N, K, L), all experts stacked
        :type b: cute.Tensor
        :param out: Output tensor C, shape (m, N_out, 1), contiguous
        :type out: cute.Tensor
        :param sfa: Scale factor tensor A, atom layout
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B, atom layout
        :type sfb: cute.Tensor
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID
        :type tile_idx_to_expert_idx: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles, shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param tile_idx_to_mn_limit: Valid M limit per tile, shape (num_tiles,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param alpha: Per-expert alpha scaling tensor, shape (L,), float32
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function
        :type epilogue_op: cutlass.Constexpr
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.out_dtype: Type[cutlass.Numeric] = out.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.gemm_output_layout = utils.LayoutEnum.ROW_MAJOR

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()
        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B.
        #
        # N-major variant: the MMA-tied ``make_tiled_tma_atom_B`` helper insists
        # on a K-major SMEM layout (matches tcgen05 requirements). We instead
        # target our plain, unswizzled ``sB_raw`` buffer whose innermost axis is
        # N (same as the gmem layout). The transpose warp later repacks
        # ``sB_raw`` into the MMA-ready ``sB`` buffer.
        #
        # The MMA-tied ``b_smem_layout`` is still used downstream to build
        # fragments for tcgen05.mma but is no longer the TMA destination.
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        b_raw_smem_layout = cute.slice_(self.b_raw_smem_layout_staged, (None, None, 0))

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))

        # Helper to create TMA for B tensor
        def _make_tma_b(b_tensor, sfb_tensor):
            # N-major B -> unswizzled N-major sB_raw. Use the low-level
            # ``cpasync.make_tiled_tma_atom`` helper so we are not tied to an
            # MMA-specific SMEM layout. Non-multicast only for now (cluster
            # shape (1, 1) is enforced by ``is_valid_mma_tiler_and_cluster_shape``).
            atom_b, tensor_b = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                b_tensor,
                b_raw_smem_layout,
                (self.mma_tiler[1], self.mma_tiler[2]),
            )
            atom_sfb, tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb_tensor,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                x = tensor_sfb.stride[0][1]
                y = cute.ceil_div(tensor_sfb.shape[0][1], 4)

                new_shape = (
                    (tensor_sfb.shape[0][0], ((2, 2), y)),
                    tensor_sfb.shape[1],
                    tensor_sfb.shape[2],
                )
                # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
                x_times_3 = 3 * x
                new_stride = (
                    (tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                    tensor_sfb.stride[1],
                    tensor_sfb.stride[2],
                )
                tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
                tensor_sfb = cute.make_tensor(tensor_sfb.iterator, tensor_sfb_new_layout)
            return atom_b, tensor_b, atom_sfb, tensor_sfb

        # Create TMA for single B tensor
        tma_atom_b, tma_tensor_b, tma_atom_sfb, tma_tensor_sfb = _make_tma_b(b, sfb)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_raw_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        # ab_pipeline (TMA -> MMA) covers only A, SFA, SFB; its tx_count
        # excludes B which is on its own TMA -> transpose pipeline.
        self.num_tma_load_bytes_ab = (
            a_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size
        # b_raw_pipeline (TMA -> transpose) covers only B.
        self.num_tma_load_bytes_b = b_copy_size * atom_thr_size
        # Keep a combined count for any legacy callers; unused internally.
        self.num_tma_load_bytes = self.num_tma_load_bytes_ab + self.num_tma_load_bytes_b

        self.tile_sched_params, grid = self._compute_grid(
            (a.shape[0], b.shape[0], a.shape[2]),
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        # Setup TMA store for C (contiguous output — no tensormap updates needed)
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            out,
            epi_smem_layout,
            self.epi_tile,
        )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            # N-major TMA -> transpose warp (B_raw slot)
            b_raw_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            # transpose warp -> MMA (B_mma slot)
            b_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_tile_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE) — for TMA store C epilogue
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.out_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE) — K-major, MMA-ready. Filled by the
            # transpose warp from sB_raw, consumed by the MMA warp.
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (N_TILE, K_TILE, STAGE) — N-major, unswizzled. TMA-filled,
            # transpose-consumed.
            sB_raw: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_raw_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            out,
            tile_idx_to_expert_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.b_raw_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=False,
        )
        return

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and
        tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        out: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        alpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        b_raw_smem_layout_staged: cute.Layout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states.
        # In the N-major kernel the ab_pipeline covers only A, SFA, SFB; B flows
        # through its own TMA -> transpose -> MMA pair of pipelines (see below).
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes_ab,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # b_raw_pipeline: TMA warp fills sB_raw (N-major); transpose warp
        # consumes it. ``PipelineTmaAsync.consumer_release`` only arrives
        # from ``is_signalling_thread`` (one lane per CTA), so the consumer
        # cooperative group must have size 1 — otherwise the empty barrier's
        # arrive_count is set to 32 and each release contributes only 1/32,
        # meaning the barrier never flips and ``producer_tail`` deadlocks at
        # kernel exit.
        b_raw_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        b_raw_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        b_raw_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.b_raw_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=b_raw_producer_group,
            consumer_group=b_raw_consumer_group,
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # b_mma_pipeline: transpose warps produce sB (K-major, MMA-ready);
        # MMA warp consumes. Both are plain thread-level cooperative groups
        # so we use PipelineAsync — the MMA warp does an explicit
        # consumer_wait / consumer_release pair (UMMA-pipeline implicit
        # release semantics don't compose with our external transpose step).
        # Producer side now spans all transpose warps (4 × 32 = 128 threads
        # by default) so the barrier arrive count matches the number of
        # threads actually calling producer_commit.
        b_mma_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_transpose_warps,
        )
        b_mma_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        b_mma_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.b_mma_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=b_mma_producer_group,
            consumer_group=b_mma_consumer_group,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        #
        # Setup smem tensor A/B/C/Scale/ExpandedIdx
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE) — K-major, MMA-ready. Written by the
        # transpose warp.
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (N_TILE, K_TILE, STAGE) — N-major, TMA destination. Read by the
        # transpose warp.
        sB_raw = storage.sB_raw.get_tensor(b_raw_smem_layout_staged)
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # (EPI_TILE_M, EPI_TILE_N, STAGE) — for TMA store C
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        # (bidx, bidy, bidz, valid)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D — targets sB_raw (N-major, non-MMA-ready).
        # Unlike the K-major kernel we do NOT use ``thr_mma.partition_B``
        # because the MMA partitioning is tied to the K-major swizzled layout.
        # Here we tile gB_nkl directly and drop it into sB_raw verbatim; the
        # transpose warp later repacks its contents into sB.
        b_cta_layout = cute.make_layout(1)  # non-multicast (cluster (1,1))
        tBsB_raw, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            b_cta_layout,
            cute.group_modes(sB_raw, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)

        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )

        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Use mma_tiler_c for C partitioning (halved N for SwiGLU)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )

        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Cluster wait before tensor memory alloc
        #
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        griddepcontrol_wait()

        #
        # Specialized Schedule warp
        #
        if warp_idx == self.sched_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_sched_warps)
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_valid_tiles = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy("async.shared", space="cta")

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                    expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                    tile_idx = mma_tile_coord_m
                    if tile_idx < num_valid_tiles:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        mn_limit = tile_idx_to_mn_limit[tile_idx]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[0]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[1]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy("async.shared", space="cta")

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_uniform_warps)

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            # Separate state for the B (TMA -> transpose) pipeline. Kept in
            # lock-step with ab_producer_state — the same k_tile counter drives
            # both pipelines.
            b_raw_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, 0)]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, 0)]

                expert_idx = mma_tile_coord_mnl[2]
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                b_raw_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tAgSFA_k = tAgSFA_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tBsB_raw_pipe = tBsB_raw[(None, b_raw_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, ab_producer_state.index)]

                    tma_bar_ab = ab_pipeline.producer_get_barrier(ab_producer_state)

                    # Conditionally wait for AB buffer empty (A, SFA, SFB only)
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    # B_raw has its own pipeline; acquire independently.
                    b_raw_pipeline.producer_acquire(b_raw_producer_state)
                    tma_bar_b = b_raw_pipeline.producer_get_barrier(b_raw_producer_state)

                    # TMA load A
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=tma_bar_ab,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=tma_bar_ab,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    # TMA load SFB (still on ab_pipeline)
                    tBgSFB_slice = tBgSFB[(None, slice_n, None, expert_idx)]
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar_ab,
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    # TMA load B_raw (N-major, own pipeline -> transpose warp)
                    tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, expert_idx)]
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, b_raw_producer_state.count)],
                        tBsB_raw_pipe,
                        tma_bar_ptr=tma_bar_b,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    b_raw_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)
            b_raw_pipeline.producer_tail(b_raw_producer_state)

        #
        # Specialized Transpose warp
        # -------------------------
        # Consumes one stage of sB_raw (N-major, TMA-landed), repacks nibbles
        # into the K-major ``sB`` buffer, and signals b_mma_pipeline.
        # This is a first-pass, correctness-oriented implementation: each
        # thread reads one nibble at a time via CuteDSL's FP4 tensor interface.
        # The per-tile nibble count is distributed across the 32 threads of
        # the warp. See ``docs/nvfp4_prefill_n_major_weights_design.md`` §5 for
        # the optimized sub-block shuffle plan.
        #
        # CuteDSL doesn't like Python `and` on tracked boolean values; use two
        # nested compile-folded comparisons instead. `warp_idx` is runtime but
        # the pattern matches the existing `warp_idx == self.foo_warp_id`
        # dispatch style used by the other specialised warps.
        warp_in_transpose_runtime = warp_idx - self.transpose_warp_id_base
        if (warp_in_transpose_runtime >= 0) & (
            warp_in_transpose_runtime < self.num_transpose_warps
        ):
            cute.arch.warpgroup_reg_dealloc(self.num_regs_transpose_warps)

            b_raw_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            b_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            sB_raw_u8 = cute.recast_tensor(sB_raw, cutlass.Uint8)

            n_tile = self.mma_tiler[1]
            k_tile = self.mma_tiler[2]
            n_half = n_tile // 2
            k_half = k_tile // 2

            # Plain K-inner uint8 layout for sB (N rows × K_byte cols × STAGE).
            sB_u8_plain_layout = cute.make_layout(
                (n_tile, k_half, self.num_ab_stage),
                stride=(k_half, 1, n_tile * k_half),
            )
            # Compose with the same Swizzle object the FP4 sB uses, so accesses
            # XOR through the same bit pattern tcgen05 encodes in its matrix
            # descriptor.
            sB_u8_composed_layout = cute.make_composed_layout(
                b_smem_layout_staged.inner,
                0,
                sB_u8_plain_layout,
            )
            sB_u8_ptr = cute.recast_ptr(sB.iterator, dtype=cutlass.Uint8)
            sB_u8 = cute.make_tensor(sB_u8_ptr, sB_u8_composed_layout)

            # Work distribution: warp-level ``ldmatrix.m16n16.trans.b8`` loads.
            # Each call reads a 16 K-row × 16 N-byte tile (= 256 source bytes,
            # covering 16 K-values and all 32 N-FP4s of a 16-byte N-stripe)
            # with hardware transpose. The ``.b8`` source format picks 16-byte
            # rows, which matches ``sB_raw_u8``'s K-axis stride (64 B) and lets
            # us tile N at 16-byte granularity from N_base_byte ∈ {0, 16, 32,
            # 48} — the 16-byte row-alignment requirement is met naturally, in
            # contrast to ``.b4x16_p64`` (8-byte rows) which forces every-other
            # 16-FP4 stripe to be skipped.
            #
            # Warp split: each warp owns 16 N-bytes = 32 N-rows of sB output,
            # same as before. Per warp: k_tile // 16 ldmatrix calls (one per
            # K-tile), each producing 4 output u16s per lane at 4 distinct
            # N-FP4 rows. Total per warp per stage:
            #   (k_tile // 16) * 4 u16/lane = same 64 output bytes/lane as the
            # scalar path.
            #
            # Per lane TV (l = 16*g + 4*qq + w, g∈0..1, qq∈0..3, w∈0..3):
            #   Owns dst-rows (qq + 4g, qq + 4g + 8) = N-byte indices within
            #   the 16-byte tile; dst-cols 4w..4w+3 = K-values within tile.
            lane = tidx % self.threads_per_warp
            ldm_g = (lane >> 4) & 1
            ldm_qq = (lane >> 2) & 3
            ldm_w = lane & 3
            # Lane's N-byte offsets within the warp's 16-byte stripe (same for
            # every tile — only K_base varies across tiles).
            nb_lo = ldm_qq + 4 * ldm_g             # 0..7  (dst-row of R0)
            nb_hi = ldm_qq + 4 * ldm_g + 8          # 8..15 (dst-row of R1)
            # Absolute N-byte in sB_raw_u8 (and sB N-FP4 after *2). The tile
            # origin is the warp's stripe start, which is 16-aligned because
            # warp_in_transpose_runtime * 16 ∈ {0, 16, 32, 48}.
            warp_n_base_byte = warp_in_transpose_runtime * (n_half // self.num_transpose_warps)
            abs_nb_lo = warp_n_base_byte + nb_lo
            abs_nb_hi = warp_n_base_byte + nb_hi
            # Absolute N-FP4 row indices in sB (stride k_half along N):
            n_fp4_lo_even = 2 * abs_nb_lo
            n_fp4_lo_odd = n_fp4_lo_even + 1
            n_fp4_hi_even = 2 * abs_nb_hi
            n_fp4_hi_odd = n_fp4_hi_even + 1
            # K-byte base-offset within an N-row for this lane's stripe —
            # 2 consecutive K-bytes per call (= kb=2w..2w+1).
            kb_lane_off = 2 * ldm_w
            # Each lane supplies the pointer to one ldmatrix "row" = K=lane_row
            # of the current tile. lanes 0..15 carry the 16 row addresses;
            # lanes 16..31 duplicate (only lanes 0..15 matter for .x1).
            ldm_row_within_tile = lane & 0xF

            # Masks for the even/odd nibble-compression PTX pattern. See
            # comment at the pack-site below.
            MASK_LO8 = cutlass.Uint32(0x0F)
            MASK_HI8 = cutlass.Uint32(0xF0)
            MASK_LO16 = cutlass.Uint32(0xF00)
            MASK_HI16 = cutlass.Uint32(0xF000)

            # Get the first tile info from pipeline
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                for k_iter in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Wait for producer TMA to fill sB_raw[stage]
                    b_raw_pipeline.consumer_wait(b_raw_consumer_state)
                    # Wait for MMA to signal sB[stage] empty (ready to write)
                    b_mma_pipeline.producer_acquire(b_mma_producer_state)

                    raw_stage = b_raw_consumer_state.index
                    mma_stage = b_mma_producer_state.index

                    # One ldmatrix.m16n16.x1.trans.b8 per 16-K-value tile.
                    # At mma_tiler (K=128) this is k_tile // 16 = 8 calls per
                    # warp per stage. The instruction loads a 16 K × 16 N-byte
                    # slab of ``sB_raw_u8`` with byte-transposed destination:
                    #   dst[r'][c'] = src-u8 at (K=c', N-byte=r')
                    #                = { FP4(N=2*r', K=c'),
                    #                    FP4(N=2*r'+1, K=c') } packed.
                    #
                    # Per lane ownership (see the lane-TV summary above):
                    #   R0 = 4 dst bytes at (dst-row=nb_lo, dst-col=4w..4w+3)
                    #   R1 = 4 dst bytes at (dst-row=nb_hi, dst-col=4w..4w+3)
                    # i.e. 4 K-values × 2 N-bytes × 2 N-FP4 = 16 FP4s per lane
                    # per tile, matching the 4-u16-output contribution below.
                    for tile_idx in cutlass.range_constexpr(k_tile // 16):
                        k_base = tile_idx * 16
                        # Per-lane row pointer into sB_raw_u8 — lanes 0..15
                        # each cover one K-row of the tile; lanes 16..31
                        # duplicate (only lanes 0..15 matter for .x1).
                        src_row = cute.domain_offset(
                            (warp_n_base_byte, k_base + ldm_row_within_tile, raw_stage),
                            sB_raw_u8,
                        )
                        r0, r1 = ldmatrix_m16n16_trans_b8(
                            src_row.iterator.llvm_ptr
                        )

                        # Nibble-compress each ri into (even_ri, odd_ri) u16s.
                        # ri = [b0 b1 b2 b3] where bi is a u8 at one K value
                        # holding (N=2*nb even: low nib, N=2*nb+1 odd: high nib).
                        # For N-even u16 at (kb=2w..2w+1):
                        #   out[3:0]   = b0 low nib  = ri & 0x0F
                        #   out[7:4]   = b1 low nib  = (ri >> 4) & 0xF0
                        #   out[11:8]  = b2 low nib  = (ri >> 8) & 0xF00
                        #   out[15:12] = b3 low nib  = (ri >> 12) & 0xF000
                        # For N-odd u16: same with shifted source (+4 bits).
                        even_r0 = (
                            (r0 & MASK_LO8)
                            | ((r0 >> 4) & MASK_HI8)
                            | ((r0 >> 8) & MASK_LO16)
                            | ((r0 >> 12) & MASK_HI16)
                        )
                        odd_r0 = (
                            ((r0 >> 4) & MASK_LO8)
                            | ((r0 >> 8) & MASK_HI8)
                            | ((r0 >> 12) & MASK_LO16)
                            | ((r0 >> 16) & MASK_HI16)
                        )
                        even_r1 = (
                            (r1 & MASK_LO8)
                            | ((r1 >> 4) & MASK_HI8)
                            | ((r1 >> 8) & MASK_LO16)
                            | ((r1 >> 12) & MASK_HI16)
                        )
                        odd_r1 = (
                            ((r1 >> 4) & MASK_LO8)
                            | ((r1 >> 8) & MASK_HI8)
                            | ((r1 >> 12) & MASK_LO16)
                            | ((r1 >> 16) & MASK_HI16)
                        )

                        # Store 4 u16s at 4 distinct (N-FP4, kb) destinations.
                        # kb=k_base/2 + 2w..2w+1 (two K-bytes per u16).
                        kb_tile_off = (k_base >> 1) + kb_lane_off
                        dst_lo_even = cute.domain_offset(
                            (n_fp4_lo_even, kb_tile_off, mma_stage), sB_u8
                        )
                        dst_lo_odd = cute.domain_offset(
                            (n_fp4_lo_odd, kb_tile_off, mma_stage), sB_u8
                        )
                        dst_hi_even = cute.domain_offset(
                            (n_fp4_hi_even, kb_tile_off, mma_stage), sB_u8
                        )
                        dst_hi_odd = cute.domain_offset(
                            (n_fp4_hi_odd, kb_tile_off, mma_stage), sB_u8
                        )
                        st_shared_b16(dst_lo_even, even_r0)
                        st_shared_b16(dst_lo_odd, odd_r0)
                        st_shared_b16(dst_hi_even, even_r1)
                        st_shared_b16(dst_hi_odd, odd_r1)

                    # Fence so the MMA warp's async proxy view sees the
                    # transpose writes before it issues tcgen05.mma.
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    # Sync across all transpose warps so warp 0 can safely
                    # release sB_raw back to TMA (otherwise warps 1..N might
                    # still be reading it).
                    self.transpose_sync_barrier.arrive_and_wait()

                    # Release sB_raw (TMA-refill slot) — guarded to warp 0 of
                    # the transpose group because ``PipelineTmaAsync.consumer_release``
                    # elects ``lane < cluster_size`` (=1) per *warp* as the
                    # signalling thread; letting all 4 warps call it would
                    # over-arrive the empty barrier.
                    if warp_in_transpose_runtime == 0:
                        b_raw_pipeline.consumer_release(b_raw_consumer_state)

                    # producer_commit on b_mma uses PipelineAsync semantics:
                    # every thread arrives once, producer_group size matches
                    # num_transpose_warps * 32 so the barrier flips only after
                    # all transpose threads have written their share.
                    b_mma_pipeline.producer_commit(b_mma_producer_state)

                    b_raw_consumer_state.advance()
                    b_mma_producer_state.advance()

                # Advance to next tile
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            b_mma_pipeline.producer_tail(b_mma_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            # Consume the transpose warp's output sB before each MMA atom.
            b_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                b_mma_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for cta_tile_shape_n=192 case by two words
                    # (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in cutlass.range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full (A, SFA, SFB).
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        # Wait for the transpose warp to finish filling sB for
                        # this stage. The N-major -> K-major repack is on the
                        # critical path just like the TMA arrivals.
                        b_mma_pipeline.consumer_wait(b_mma_consumer_state)

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB_mma[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty + B_mma empty.
                        ab_pipeline.consumer_release(ab_consumer_state)
                        b_mma_pipeline.consumer_release(b_mma_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    b_mma_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()
                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc, tTR_rAcc_gate = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.out_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c_epi, bSG_sC, bSG_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Per-expert alpha scaling (computed once per tile)
                alpha_val = alpha[mma_tile_coord_mnl[2]]

                is_k_tile_cnt_zero = k_tile_cnt == 0

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                else:
                    acc_stage_index = acc_consumer_state.index

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]

                #
                # Wait for accumulator buffer full
                #
                if not is_k_tile_cnt_zero:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in subtiles,
                # applying fused activation (identity / relu2 / swiglu).
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                if cutlass.const_expr(self.is_swiglu):
                    # ── SwiGLU path: consume pairs of subtiles (up, gate) ──
                    for subtile_idx in cutlass.range(0, subtile_cnt, 2):
                        real_subtile_idx = subtile_idx // 2

                        if not is_k_tile_cnt_zero:
                            # Load up and gate subtiles from consecutive TMEM indices
                            tTR_tAcc_mn_up = tTR_tAcc[(None, None, None, real_subtile_idx * 2)]
                            tTR_tAcc_mn_gate = tTR_tAcc[(None, None, None, real_subtile_idx * 2 + 1)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc)
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                            # Apply per-expert alpha scaling before SwiGLU
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tTR_rAcc[i] = alpha_val * tTR_rAcc[i]
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_gate)):
                                tTR_rAcc_gate[i] = alpha_val * tTR_rAcc_gate[i]

                            # SwiGLU: up * silu(gate), store back into tTR_rAcc
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                up_val = tTR_rAcc[i]
                                gate_val = tTR_rAcc_gate[i]
                                tTR_rAcc[i] = up_val * silu_f32(
                                    gate_val, fastmath=True
                                )

                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            tRS_rC.store(acc_vec.to(self.out_dtype))
                        else:
                            tRS_rC.fill(0)

                        # Store C to shared memory
                        c_buffer = (num_prev_subtiles + real_subtile_idx) % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        # TMA store C to global memory
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c_epi,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, real_subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                else:
                    # ── Single-subtile path: ReLU² or identity ──
                    for subtile_idx in range(subtile_cnt):
                        if not is_k_tile_cnt_zero:
                            # Load accumulator from tensor memory buffer to register
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                            # Apply per-expert alpha scaling
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                tTR_rAcc[i] = alpha_val * tTR_rAcc[i]

                            if cutlass.const_expr(self.activation == "relu2"):
                                # ReLU²: max(acc, 0)²
                                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                                    acc_relu = fmax(tTR_rAcc[i], 0.0)
                                    tTR_rAcc[i] = acc_relu * acc_relu

                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            tRS_rC.store(acc_vec.to(self.out_dtype))
                        else:
                            tRS_rC.fill(0)

                        # Store C to shared memory
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()

                        # TMA store C to global memory
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c_epi,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()
                #
                # Async arrive accumulator buffer empty
                #
                if not is_k_tile_cnt_zero:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then partition tensor memory (source)
        and register arrays (destination). For SwiGLU, also allocates a gate register.

        :return: (tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc_gate)
            tTR_rAcc_gate is None for non-SwiGLU activations.
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.gemm_output_layout,
            self.out_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        frag_shape = tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape
        tTR_rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
        if cutlass.const_expr(self.is_swiglu):
            tTR_rAcc_gate = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc_gate
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, None

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store (R2S), then partition register (source)
        and shared memory (destination).
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.gemm_output_layout, self.out_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store (TMA store C),
        partition shared memory (source) and global memory (destination).
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        Uses the same algorithm as the FC1 grouped GEMM kernel, allocating C shared
        memory stages for TMA store and filling remaining capacity with A/B stages.

        :return: (num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage)
        """
        # ACC stages
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

        # Default C stages
        num_c_stage = 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB, C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1,
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, 1,
        )

        # Extra cost for the N-major B_raw staging buffer (same byte count as
        # the K-major sB, because both are (N_TILE, K_TILE) FP4 tiles).
        b_raw_bytes_per_stage = cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + b_raw_bytes_per_stage
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # 1024B alignment for mbar
        mbar_helpers_bytes = 1024

        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages, accounting for C shared memory
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Add remaining unused smem to epilogue C stages
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        gemm_shape: Tuple[int, int, int],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size based on GEMM shape.

        :param gemm_shape: The GEMM computation shape (M, N, L)
        :type gemm_shape: tuple[int, int, int]
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr
        :param raster_along_m: Boolean, True to use raster along M.
        :type raster_along_m: bool

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        (m, n, l) = gemm_shape  # noqa: E741

        num_ctas_m = cute.ceil_div(m, cta_tile_shape_mnk[0])
        num_ctas_n = cute.ceil_div(n, cta_tile_shape_mnk[1])
        num_ctas_l = l

        num_ctas_mnl = (num_ctas_m, num_ctas_n, num_ctas_l)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.
        Only 1CTA mode is supported.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        out_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        if out_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        out_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param out_major: The major dimension of the C tensor
        :type out_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # N-major kernel: A must be K-major, B must be N-major (the in-flight
        # SMEM transpose produces the K-major layout the MMA needs).
        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "n"):
            is_valid = False
        if out_dtype is cutlass.Float4E2M1FN and out_major == "m":
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid.
        Only 1CTA mode (mma_tiler_mn[0] == 128) is supported.

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Only tile_size=128 (1CTA) is supported
        if mma_tiler_mn[0] != 128:
            is_valid = False
        # Skip invalid mma tile n
        if mma_tiler_mn[1] not in (64, 128, 192, 256):
            is_valid = False

        # N-major variant: only the trivial cluster shape is supported in
        # this initial implementation. The B TMA atom is built with the
        # low-level ``CopyBulkTensorTileG2SOp`` (no multicast).
        if cluster_shape_mn != (1, 1):
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        out_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param out_major: The major axis of the C tensor
        :type out_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(out_dtype, out_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        out_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        out_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param out_major: The major axis of the C tensor
        :type out_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, out_dtype
        ):
            can_implement = False

        # Skip unsupported layouts
        if not cls.is_valid_layouts(ab_dtype, out_dtype, a_major, b_major, out_major):
            can_implement = False

        # Skip invalid mma tile shape and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(mma_tiler_mn, cluster_shape_mn):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, out_dtype, a_major, b_major, out_major
        ):
            can_implement = False
        # Skip unsupported A/B layout (this kernel REQUIRES b_major == "n").
        if not (a_major == "k" and b_major == "n"):
            can_implement = False
        return can_implement

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        n_out: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Wrapper for contiguous grouped GEMM with fused alpha + activation.

        L is the total number of experts (single B tensor contains all experts).
        n is the GEMM N dimension, n_out is the output N (halved for SwiGLU).
        alpha is per-expert scaling, shape (L,), float32.
        """
        scale_k = k // scaling_vector_size
        num_tiles = m // tile_size

        a = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, scale_k // 4, 1), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, n_out, 1), order=(1, 0, 2))
        )

        # N-major B layout: N is the innermost axis on device. Only ``order``
        # differs from the K-major kernel; the logical shape tuple stays
        # ``(n, k, l)``. Strides become ``(1, n, n*k)``.
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        alpha_tensor = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha_tensor,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )

    @cute.jit
    def aot_wrapper(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        a_sf: cute.Tensor,
        b_sf: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """AOT entry point for contiguous grouped GEMM that accepts pre-formed CuTe tensors.

        Tensor layout contract (callers must match these exactly):

        * ``a``       : ``(m, K, 1)``        order ``(1, 0, 2)``  FP4
        * ``b``       : ``(N, K, L)``         order ``(1, 0, 2)``  FP4
        * ``a_sf``    : ``(32, 4, m/128, 4, K/sfv/4, 1)`` order ``(2,1,4,0,3,5)``
        * ``b_sf``    : ``(32, 4, N/128, 4, K/sfv/4, L)`` order ``(2,1,4,0,3,5)``
        * ``c``       : ``(m, N_out, 1)`` order ``(1, 0, 2)``
        * ``alpha``   : ``(L,)`` float32
        * ``tile_idx_to_group_idx`` : ``(num_tiles,)``
        * ``tile_idx_to_mn_limit``  : ``(num_tiles,)``
        * ``num_non_exiting_tiles`` : ``(1,)``
        """
        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            tile_idx_to_mn_limit,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]
