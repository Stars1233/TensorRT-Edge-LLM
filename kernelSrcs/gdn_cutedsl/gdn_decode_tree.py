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

# CuTe DSL GDN DDTree decode kernel.
#
# The kernel evaluates a flattened DDTree verify batch in a single launch.  Each
# CTA handles one (batch, HV) recurrent state.  Nodes are processed in flattened
# tree order so a child can read its parent's checkpoint from intermediate_states.
#
# Test:  python gdn_decode_tree.py --n 2 --h 4 --hv 8 --seq_len 8
# AOT:   python gdn_decode_tree.py --export_only [--gpu_arch sm_110]

import argparse
import os
import sys
import time
from typing import Dict, Tuple

_parsed_args = None
_saved_argv = None
if __name__ == "__main__":
    _saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

TILE_K = 128
TILE_V = 32
TILE_V_PADDED = 36
NUM_THREADS = 256
NUM_WARPS = 8
V_PER_WARP = 4
ROWS_PER_ITER = 8
NUM_K_ITERS = TILE_K // ROWS_PER_ITER

T_MAX_AOT = 128

AOT_PLACEHOLDER_N = 1
AOT_PLACEHOLDER_H = 14
AOT_PLACEHOLDER_HV = 14
AOT_PLACEHOLDER_K = 128
AOT_PLACEHOLDER_V = 128
AOT_PLACEHOLDER_T = 64


def _define_tree_kernel(split_v=False):
    @cute.kernel
    def gdn_tree_kernel(
        h0_source: cute.Tensor,  # [n, hv, K, V] FP32, read-only in DDTree mode
        smem_h_layout: cute.Layout,
        num_v_tiles: Int32,
        q: cute.Tensor,  # [n, seq_len, h, K] FP16
        k: cute.Tensor,  # [n, seq_len, h, K] FP16
        v: cute.Tensor,  # [n, seq_len, hv, V] FP16
        a: cute.Tensor,  # [n, seq_len, hv] FP16
        b: cute.Tensor,  # [n, seq_len, hv] FP16
        A_log: cute.Tensor,  # [hv] FP32
        dt_bias: cute.Tensor,  # [hv] FP16
        o: cute.Tensor,  # [n, seq_len, hv, V] FP16
        intermediate_states: cute.Tensor,  # [n, seq_len, hv, K, V] FP32
        tree_parent_ids: cute.Tensor,  # [n, seq_len] INT32
        tree_depths: cute.Tensor,  # [n, seq_len] INT32
        seq_len: Int32,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        H = q.layout.shape[2]
        HV = v.layout.shape[2]
        state_idx = block_idx
        v_tile = 0
        if split_v:
            state_idx = block_idx // num_v_tiles
            v_tile = block_idx % num_v_tiles
        i_n = state_idx // HV
        i_hv = state_idx % HV
        i_h = i_hv // (HV // H)

        k_local = in_warp_tid // V_PER_WARP
        v_local = in_warp_tid % V_PER_WARP
        v_base = warp_idx * V_PER_WARP
        v_idx = v_base + v_local
        v_global = v_tile * TILE_V + v_idx

        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(cutlass.Float32, smem_h_layout, 128)
        smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
        smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
        sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
        sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
        smem_norm_layout = cute.make_layout((NUM_WARPS * 2,), stride=(1,))
        sNorm = smem.allocate_tensor(cutlass.Float32, smem_norm_layout, 128)

        for node_idx in range(seq_len):
            parent_idx = cutlass.Int32(tree_parent_ids[i_n, node_idx])

            # GDN state only needs the parent relation. Depth is still an ABI
            # input because attention/accept use it to build the verify tree.
            node_valid = False
            source_is_h0 = False
            if node_idx == 0:
                if parent_idx < 0:
                    node_valid = True
                    source_is_h0 = True
            else:
                if parent_idx >= 0:
                    if parent_idx < node_idx:
                        node_valid = True

            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, node_idx, i_h, tidx])
                sQ[tidx] = cutlass.Float32(q[i_n, node_idx, i_h, tidx])

            r_A_log = cutlass.Float32(A_log[i_hv])
            r_dt_bias = cutlass.Float32(dt_bias[i_hv])
            r_a = cutlass.Float32(a[i_n, node_idx, i_hv])
            r_b = cutlass.Float32(b[i_n, node_idx, i_hv])

            r_g = 0.0
            r_beta = 0.0
            if in_warp_tid == 0:
                x = r_a + r_dt_bias
                beta_x = softplus_beta * x
                softplus_x = x
                if beta_x <= softplus_threshold:
                    exp_bx = cute.exp(beta_x)
                    softplus_x = cutlass.Float32(
                        (cutlass.Float32(1.0) / softplus_beta)
                        * cutlass.Float32(cute.log(cutlass.Float32(1.0 + exp_bx)))
                    )
                r_g = cute.exp(-cute.exp(r_A_log) * softplus_x)
                r_beta = 1.0 / (1.0 + cute.exp(-r_b))

            r_g = cute.arch.shuffle_sync(r_g, 0)
            r_beta = cute.arch.shuffle_sync(r_beta, 0)
            cute.arch.barrier()

            if use_qk_l2norm:
                sum_q_partial = 0.0
                sum_k_partial = 0.0
                if tidx < TILE_K:
                    q_val = sQ[tidx]
                    k_val = sK[tidx]
                    sum_q_partial = q_val * q_val
                    sum_k_partial = k_val * k_val

                for offset in [16, 8, 4, 2, 1]:
                    sum_q_partial += cute.arch.shuffle_sync_bfly(
                        sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                    sum_k_partial += cute.arch.shuffle_sync_bfly(
                        sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)

                if in_warp_tid == 0:
                    sNorm[warp_idx] = sum_q_partial
                    sNorm[warp_idx + NUM_WARPS] = sum_k_partial
                cute.arch.barrier()

                if warp_idx == 0:
                    local_sum_q = 0.0
                    local_sum_k = 0.0
                    if in_warp_tid < NUM_WARPS:
                        local_sum_q = sNorm[in_warp_tid]
                        local_sum_k = sNorm[in_warp_tid + NUM_WARPS]
                    for offset in [4, 2, 1]:
                        local_sum_q += cute.arch.shuffle_sync_bfly(
                            local_sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                        local_sum_k += cute.arch.shuffle_sync_bfly(
                            local_sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                    if in_warp_tid == 0:
                        sNorm[0] = cute.rsqrt(local_sum_q + 1e-6)
                        sNorm[1] = cute.rsqrt(local_sum_k + 1e-6)
                cute.arch.barrier()

                inv_norm_q = sNorm[0]
                inv_norm_k = sNorm[1]
                if tidx < TILE_K:
                    sK[tidx] = sK[tidx] * inv_norm_k
                    sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                cute.arch.barrier()
            else:
                if tidx < TILE_K:
                    sQ[tidx] = sQ[tidx] * scale
                cute.arch.barrier()

            if split_v:
                if node_valid:
                    r_v = cutlass.Float32(v[i_n, node_idx, i_hv, v_global])

                    sum_hk = 0.0
                    for k_iter in range(NUM_K_ITERS):
                        k_idx = k_iter * ROWS_PER_ITER + k_local
                        h_src = 0.0
                        if source_is_h0:
                            h_src = h0_source[(i_n, i_hv, k_idx, v_global)]
                        else:
                            h_src = intermediate_states[(i_n, parent_idx, i_hv, k_idx, v_global)]
                        h_gated = h_src * r_g
                        sH[(k_idx, v_idx)] = h_gated
                        sum_hk += h_gated * sK[k_idx]

                    for offset in [4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = cute.arch.shuffle_sync(v_new, v_local)

                    sum_hq = 0.0
                    for k_iter in range(NUM_K_ITERS):
                        k_idx = k_iter * ROWS_PER_ITER + k_local
                        h_new = sH[(k_idx, v_idx)] + sK[k_idx] * v_new
                        intermediate_states[(i_n, node_idx, i_hv, k_idx, v_global)] = h_new
                        sum_hq += h_new * sQ[k_idx]

                    for offset in [4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                    if k_local == 0:
                        o[(i_n, node_idx, i_hv, v_global)] = cutlass.Float16(sum_hq)
                else:
                    if k_local == 0:
                        o[(i_n, node_idx, i_hv, v_global)] = cutlass.Float16(0.0)

                    for k_iter in range(NUM_K_ITERS):
                        k_idx = k_iter * ROWS_PER_ITER + k_local
                        h_val = h0_source[(i_n, i_hv, k_idx, v_global)]
                        intermediate_states[(i_n, node_idx, i_hv, k_idx, v_global)] = h_val

                cute.arch.barrier()
            else:
                for v_tile_loop in range(num_v_tiles):
                    v_global_loop = v_tile_loop * TILE_V + v_idx

                    if node_valid:
                        r_v = cutlass.Float32(v[i_n, node_idx, i_hv, v_global_loop])

                        sum_hk = 0.0
                        for k_iter in range(NUM_K_ITERS):
                            k_idx = k_iter * ROWS_PER_ITER + k_local
                            h_src = 0.0
                            if source_is_h0:
                                h_src = h0_source[(i_n, i_hv, k_idx, v_global_loop)]
                            else:
                                h_src = intermediate_states[(i_n, parent_idx, i_hv, k_idx, v_global_loop)]
                            h_gated = h_src * r_g
                            sH[(k_idx, v_idx)] = h_gated
                            sum_hk += h_gated * sK[k_idx]

                        for offset in [4, 2, 1]:
                            sum_hk += cute.arch.shuffle_sync_bfly(
                                sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                        v_new = (r_v - sum_hk) * r_beta
                        v_new = cute.arch.shuffle_sync(v_new, v_local)

                        sum_hq = 0.0
                        for k_iter in range(NUM_K_ITERS):
                            k_idx = k_iter * ROWS_PER_ITER + k_local
                            h_new = sH[(k_idx, v_idx)] + sK[k_idx] * v_new
                            intermediate_states[(i_n, node_idx, i_hv, k_idx, v_global_loop)] = h_new
                            sum_hq += h_new * sQ[k_idx]

                        for offset in [4, 2, 1]:
                            sum_hq += cute.arch.shuffle_sync_bfly(
                                sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                        if k_local == 0:
                            o[(i_n, node_idx, i_hv, v_global_loop)] = cutlass.Float16(sum_hq)
                    else:
                        if k_local == 0:
                            o[(i_n, node_idx, i_hv, v_global_loop)] = cutlass.Float16(0.0)

                        for k_iter in range(NUM_K_ITERS):
                            k_idx = k_iter * ROWS_PER_ITER + k_local
                            h_val = h0_source[(i_n, i_hv, k_idx, v_global_loop)]
                            intermediate_states[(i_n, node_idx, i_hv, k_idx, v_global_loop)] = h_val

                    cute.arch.barrier()

    return gdn_tree_kernel


_jit_tree = {}


def _get_jit_tree(split_v=False):
    global _jit_tree
    if split_v not in _jit_tree:
        _jit_tree[split_v] = _define_tree_kernel(split_v=split_v)
    return _jit_tree[split_v]


_jit_tree_wrapper = {}


def _create_jit_tree_wrapper(split_v=False):
    gdn_tree_kernel = _get_jit_tree(split_v=split_v)

    @cute.jit
    def run_tree(
        h0_source: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        intermediate_states: cute.Tensor,
        tree_parent_ids: cute.Tensor,
        tree_depths: cute.Tensor,
        seq_len: Int32,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        use_qk_l2norm: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        n_batch = h0_source.layout.shape[0]
        hv_dim = v.layout.shape[2]
        v_dim = v.layout.shape[3]
        batch_size = n_batch * hv_dim
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        smem_h_layout = cute.make_layout((TILE_K, TILE_V), stride=(TILE_V_PADDED, 1))
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED
            + 4 * TILE_K * 2
            + 4 * NUM_WARPS * 2
            + 128
        )

        gdn_tree_kernel(
            h0_source,
            smem_h_layout,
            num_v_tiles,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            o,
            intermediate_states,
            tree_parent_ids,
            tree_depths,
            seq_len,
            softplus_beta,
            softplus_threshold,
            scale,
            use_qk_l2norm,
        ).launch(
            grid=(batch_size * num_v_tiles if split_v else batch_size, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return run_tree


def _get_jit_tree_wrapper(split_v=False):
    global _jit_tree_wrapper
    if split_v not in _jit_tree_wrapper:
        _jit_tree_wrapper[split_v] = _create_jit_tree_wrapper(split_v=split_v)
    return _jit_tree_wrapper[split_v]


def _define_tree_split_v_precomputed_kernel():
    @cute.kernel
    def gdn_tree_split_v_precomputed_kernel(
        h0_source: cute.Tensor,  # [n, hv, K, V] FP32, read-only in DDTree mode
        smem_h_layout: cute.Layout,
        num_v_tiles: Int32,
        q: cute.Tensor,  # [n, seq_len, h, K] FP16
        k: cute.Tensor,  # [n, seq_len, h, K] FP16
        v: cute.Tensor,  # [n, seq_len, hv, V] FP16
        o: cute.Tensor,  # [n, seq_len, hv, V] FP16
        intermediate_states: cute.Tensor,  # [n, seq_len, hv, K, V] FP32
        tree_parent_ids: cute.Tensor,  # [n, seq_len] INT32
        tree_depths: cute.Tensor,  # [n, seq_len] INT32
        qk_scales: cute.Tensor,  # [n, seq_len, h, 2] FP32: q_scale, k_scale
        gate_values: cute.Tensor,  # [n, seq_len, hv, 2] FP32: g, beta
        seq_len: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        H = q.layout.shape[2]
        HV = v.layout.shape[2]
        state_idx = block_idx // num_v_tiles
        v_tile = block_idx % num_v_tiles
        i_n = state_idx // HV
        i_hv = state_idx % HV
        i_h = i_hv // (HV // H)

        k_local = in_warp_tid // V_PER_WARP
        v_local = in_warp_tid % V_PER_WARP
        v_base = warp_idx * V_PER_WARP
        v_idx = v_base + v_local
        v_global = v_tile * TILE_V + v_idx

        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(cutlass.Float32, smem_h_layout, 128)
        smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
        smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
        sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
        sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

        for node_idx in range(seq_len):
            parent_idx = cutlass.Int32(tree_parent_ids[i_n, node_idx])

            # GDN state only needs the parent relation. Depth is still an ABI
            # input because attention/accept use it to build the verify tree.
            node_valid = False
            source_is_h0 = False
            if node_idx == 0:
                if parent_idx < 0:
                    node_valid = True
                    source_is_h0 = True
            else:
                if parent_idx >= 0:
                    if parent_idx < node_idx:
                        node_valid = True

            q_scale = cutlass.Float32(qk_scales[(i_n, node_idx, i_h, 0)])
            k_scale = cutlass.Float32(qk_scales[(i_n, node_idx, i_h, 1)])
            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, node_idx, i_h, tidx]) * k_scale
                sQ[tidx] = cutlass.Float32(q[i_n, node_idx, i_h, tidx]) * q_scale

            r_g = cutlass.Float32(gate_values[(i_n, node_idx, i_hv, 0)])
            r_beta = cutlass.Float32(gate_values[(i_n, node_idx, i_hv, 1)])
            cute.arch.barrier()

            if node_valid:
                r_v = cutlass.Float32(v[i_n, node_idx, i_hv, v_global])

                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_idx = k_iter * ROWS_PER_ITER + k_local
                    h_src = 0.0
                    if source_is_h0:
                        h_src = h0_source[(i_n, i_hv, k_idx, v_global)]
                    else:
                        h_src = intermediate_states[(i_n, parent_idx, i_hv, k_idx, v_global)]
                    h_gated = h_src * r_g
                    sH[(k_idx, v_idx)] = h_gated
                    sum_hk += h_gated * sK[k_idx]

                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_idx = k_iter * ROWS_PER_ITER + k_local
                    h_new = sH[(k_idx, v_idx)] + sK[k_idx] * v_new
                    intermediate_states[(i_n, node_idx, i_hv, k_idx, v_global)] = h_new
                    sum_hq += h_new * sQ[k_idx]

                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                if k_local == 0:
                    o[(i_n, node_idx, i_hv, v_global)] = cutlass.Float16(sum_hq)
            else:
                if k_local == 0:
                    o[(i_n, node_idx, i_hv, v_global)] = cutlass.Float16(0.0)

                for k_iter in range(NUM_K_ITERS):
                    k_idx = k_iter * ROWS_PER_ITER + k_local
                    h_val = h0_source[(i_n, i_hv, k_idx, v_global)]
                    intermediate_states[(i_n, node_idx, i_hv, k_idx, v_global)] = h_val

            cute.arch.barrier()

    return gdn_tree_split_v_precomputed_kernel


_jit_tree_split_v_precomputed = None


def _get_jit_tree_split_v_precomputed():
    global _jit_tree_split_v_precomputed
    if _jit_tree_split_v_precomputed is None:
        _jit_tree_split_v_precomputed = _define_tree_split_v_precomputed_kernel()
    return _jit_tree_split_v_precomputed


_jit_tree_split_v_precomputed_wrapper = None


def _create_jit_tree_split_v_precomputed_wrapper():
    gdn_tree_kernel = _get_jit_tree_split_v_precomputed()

    @cute.jit
    def run_tree_split_v_precomputed(
        h0_source: cute.Tensor,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        intermediate_states: cute.Tensor,
        tree_parent_ids: cute.Tensor,
        tree_depths: cute.Tensor,
        qk_scales: cute.Tensor,
        gate_values: cute.Tensor,
        seq_len: Int32,
        stream: cuda.CUstream,
    ):
        n_batch = h0_source.layout.shape[0]
        hv_dim = v.layout.shape[2]
        v_dim = v.layout.shape[3]
        batch_size = n_batch * hv_dim
        num_v_tiles = cute.ceil_div(v_dim, TILE_V)
        smem_h_layout = cute.make_layout((TILE_K, TILE_V), stride=(TILE_V_PADDED, 1))
        smem_bytes = (
            4 * TILE_K * TILE_V_PADDED
            + 4 * TILE_K * 2
            + 128
        )

        gdn_tree_kernel(
            h0_source,
            smem_h_layout,
            num_v_tiles,
            q,
            k,
            v,
            o,
            intermediate_states,
            tree_parent_ids,
            tree_depths,
            qk_scales,
            gate_values,
            seq_len,
        ).launch(
            grid=(batch_size * num_v_tiles, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return run_tree_split_v_precomputed


def _get_jit_tree_split_v_precomputed_wrapper():
    global _jit_tree_split_v_precomputed_wrapper
    if _jit_tree_split_v_precomputed_wrapper is None:
        _jit_tree_split_v_precomputed_wrapper = _create_jit_tree_split_v_precomputed_wrapper()
    return _jit_tree_split_v_precomputed_wrapper


_compiled_tree: Dict[Tuple, object] = {}


def _make_tree_placeholder_tensors(n, h, hv, k, v, seq_len):
    dt = cp.float16
    return {
        "q": cp.zeros((n, seq_len, h, k), dtype=dt),
        "k": cp.zeros((n, seq_len, h, k), dtype=dt),
        "v": cp.zeros((n, seq_len, hv, v), dtype=dt),
        "a": cp.zeros((n, seq_len, hv), dtype=dt),
        "b": cp.zeros((n, seq_len, hv), dtype=dt),
        "A_log": cp.zeros(hv, dtype=cp.float32),
        "dt_bias": cp.zeros(hv, dtype=dt),
        "h0_source": cp.zeros((n, hv, k, v), dtype=cp.float32),
        "o": cp.zeros((n, seq_len, hv, v), dtype=dt),
        "intermediate_states": cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32),
        "tree_parent_ids": cp.zeros((n, seq_len), dtype=cp.int32),
        "tree_depths": cp.zeros((n, seq_len), dtype=cp.int32),
    }


def _make_tree_precomputed_placeholder_tensors(n, h, hv, k, v, seq_len):
    ph = _make_tree_placeholder_tensors(n, h, hv, k, v, seq_len)
    ph["qk_scales"] = cp.zeros((n, seq_len, h, 2), dtype=cp.float32)
    ph["gate_values"] = cp.zeros((n, seq_len, hv, 2), dtype=cp.float32)
    return ph


def _mark_dynamic_4d(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so)
            .mark_compact_shape_dynamic(mode=3, stride_order=so))


def _mark_dynamic_3d(tensor):
    so = (0, 1, 2)
    return (tensor.mark_layout_dynamic(leading_dim=2)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _mark_dynamic_2d(tensor):
    so = (0, 1)
    return (tensor.mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _mark_h0_source_dynamic(tensor):
    so = (0, 1, 2, 3)
    return (tensor.mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so))


def _mark_intermediate_states_dynamic(tensor):
    so = (0, 1, 2, 3, 4)
    return (tensor.mark_layout_dynamic(leading_dim=4)
            .mark_compact_shape_dynamic(mode=0, stride_order=so)
            .mark_compact_shape_dynamic(mode=1, stride_order=so)
            .mark_compact_shape_dynamic(mode=2, stride_order=so)
            .mark_compact_shape_dynamic(mode=3, stride_order=so))


def _to_tree_cute_tensors(ph):
    def wrap(arr, leading_dim=None, assumed_align=16):
        ct = from_dlpack(arr, assumed_align=assumed_align)
        ld = (len(arr.shape) - 1) if leading_dim is None else leading_dim
        return ct.mark_layout_dynamic(leading_dim=ld)

    return {
        "q": _mark_dynamic_4d(wrap(ph["q"])),
        "k": wrap(ph["k"]),
        "v": _mark_dynamic_4d(wrap(ph["v"])),
        "a": _mark_dynamic_3d(wrap(ph["a"])),
        "b": _mark_dynamic_3d(wrap(ph["b"])),
        "A_log": wrap(ph["A_log"], leading_dim=0),
        "dt_bias": wrap(ph["dt_bias"], leading_dim=0),
        "h0_source": _mark_h0_source_dynamic(from_dlpack(ph["h0_source"], assumed_align=32)),
        "o": wrap(ph["o"]),
        "intermediate_states": _mark_intermediate_states_dynamic(
            from_dlpack(ph["intermediate_states"], assumed_align=32)),
        "tree_parent_ids": _mark_dynamic_2d(wrap(ph["tree_parent_ids"])),
        "tree_depths": _mark_dynamic_2d(wrap(ph["tree_depths"])),
    }


def _to_tree_precomputed_cute_tensors(ph):
    t = _to_tree_cute_tensors(ph)

    def wrap(arr, leading_dim=None, assumed_align=16):
        ct = from_dlpack(arr, assumed_align=assumed_align)
        ld = (len(arr.shape) - 1) if leading_dim is None else leading_dim
        return ct.mark_layout_dynamic(leading_dim=ld)

    t["qk_scales"] = _mark_dynamic_4d(wrap(ph["qk_scales"], assumed_align=32))
    t["gate_values"] = _mark_dynamic_4d(wrap(ph["gate_values"], assumed_align=32))
    return t


def _compile_tree(n, h, hv, k, v, seq_len, stream, gpu_arch="", split_v=False):
    key = (split_v,)
    if key in _compiled_tree:
        return _compiled_tree[key]

    run_tree = _get_jit_tree_wrapper(split_v=split_v)
    ph = _make_tree_placeholder_tensors(n, h, hv, k, v, seq_len)
    t = _to_tree_cute_tensors(ph)
    compile_opts = ("--gpu-arch " + gpu_arch) if gpu_arch else None
    compiled = cute.compile(
        run_tree,
        t["h0_source"],
        t["q"],
        t["k"],
        t["v"],
        t["a"],
        t["b"],
        t["A_log"],
        t["dt_bias"],
        t["o"],
        t["intermediate_states"],
        t["tree_parent_ids"],
        t["tree_depths"],
        seq_len,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        scale=k ** -0.5,
        use_qk_l2norm=True,
        stream=stream,
        **(dict(options=compile_opts) if compile_opts else {}),
    )
    _compiled_tree[key] = compiled
    return compiled


def _compile_tree_split_v_precomputed(n, h, hv, k, v, seq_len, stream, gpu_arch=""):
    key = ("split_v_precomputed",)
    if key in _compiled_tree:
        return _compiled_tree[key]

    run_tree = _get_jit_tree_split_v_precomputed_wrapper()
    ph = _make_tree_precomputed_placeholder_tensors(n, h, hv, k, v, seq_len)
    t = _to_tree_precomputed_cute_tensors(ph)
    compile_opts = ("--gpu-arch " + gpu_arch) if gpu_arch else None
    compiled = cute.compile(
        run_tree,
        t["h0_source"],
        t["q"],
        t["k"],
        t["v"],
        t["o"],
        t["intermediate_states"],
        t["tree_parent_ids"],
        t["tree_depths"],
        t["qk_scales"],
        t["gate_values"],
        seq_len,
        stream=stream,
        **(dict(options=compile_opts) if compile_opts else {}),
    )
    _compiled_tree[key] = compiled
    return compiled


def export_gdn_decode_tree(
    n, h, hv, k, v, output_dir, file_name, function_prefix, gpu_arch="", split_v=False, precomputed=False):
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    os.makedirs(output_dir, exist_ok=True)
    print("[gdn_decode_tree] AOT compile gpu_arch=%r" % (gpu_arch or "default"))
    t0 = time.time()
    if precomputed:
        compiled = _compile_tree_split_v_precomputed(
            n,
            h,
            hv,
            k,
            v,
            seq_len=AOT_PLACEHOLDER_T,
            stream=stream,
            gpu_arch=gpu_arch,
        )
    else:
        compiled = _compile_tree(
            n,
            h,
            hv,
            k,
            v,
            seq_len=AOT_PLACEHOLDER_T,
            stream=stream,
            gpu_arch=gpu_arch,
            split_v=split_v,
        )
    print("[gdn_decode_tree] Compilation time: %.4fs" % (time.time() - t0))
    compiled.export_to_c(
        file_path=output_dir,
        file_name=file_name,
        function_prefix=function_prefix,
    )
    print("[gdn_decode_tree] Exported %s/%s.h and %s.o" % (output_dir, file_name, file_name))


def _softplus_np(x, beta=1.0, threshold=20.0):
    bx = beta * x
    if bx <= threshold:
        return (1.0 / beta) * np.log(1.0 + np.exp(bx))
    return float(x)


def _run_numpy_tree_reference(
    q_f32,
    k_f32,
    v_f32,
    a_f32,
    b_f32,
    A_log_f32,
    dt_bias_f32,
    h0_f32,
    tree_parent_ids,
    tree_depths,
    n,
    h,
    hv,
    k_dim,
    v_dim,
    seq_len,
    scale,
    use_qk_l2norm=True,
):
    o_ref = np.zeros((n, seq_len, hv, v_dim), dtype=np.float32)
    interm = np.zeros((n, seq_len, hv, k_dim, v_dim), dtype=np.float32)

    for i_n in range(n):
        for i_hv in range(hv):
            i_h = i_hv // (hv // h) if h else 0
            for node_idx in range(seq_len):
                parent_idx = int(tree_parent_ids[i_n, node_idx])
                depth = int(tree_depths[i_n, node_idx])
                is_root = node_idx == 0 and parent_idx < 0 and depth == 0
                is_child = node_idx > 0 and parent_idx >= 0 and parent_idx < node_idx and depth > 0
                if is_root:
                    H = h0_f32[i_n, i_hv].astype(np.float64).copy()
                elif is_child:
                    H = interm[i_n, parent_idx, i_hv].astype(np.float64).copy()
                else:
                    interm[i_n, node_idx, i_hv] = h0_f32[i_n, i_hv]
                    continue

                q_vec = q_f32[i_n, node_idx, i_h].astype(np.float64)
                k_vec = k_f32[i_n, node_idx, i_h].astype(np.float64)
                v_vec = v_f32[i_n, node_idx, i_hv].astype(np.float64)
                if use_qk_l2norm:
                    nq = np.sqrt(np.sum(q_vec ** 2) + 1e-6)
                    nk = np.sqrt(np.sum(k_vec ** 2) + 1e-6)
                    q_eff = q_vec / nq * scale
                    k_eff = k_vec / nk
                else:
                    q_eff = q_vec * scale
                    k_eff = k_vec

                sp = _softplus_np(float(a_f32[i_n, node_idx, i_hv]) + float(dt_bias_f32[i_hv]))
                g = np.exp(-np.exp(float(A_log_f32[i_hv])) * sp)
                beta = 1.0 / (1.0 + np.exp(-float(b_f32[i_n, node_idx, i_hv])))

                H_gated = H * g
                correction = (v_vec - H_gated.T @ k_eff) * beta
                H_new = H_gated + np.outer(k_eff, correction)
                o_ref[i_n, node_idx, i_hv, :] = (H_new.T @ q_eff).astype(np.float32)
                interm[i_n, node_idx, i_hv] = H_new.astype(np.float32)

    return o_ref, interm


def _run_tree_decode_jit(
    q_cp,
    k_cp,
    v_cp,
    a_cp,
    b_cp,
    A_log_cp,
    dt_bias_cp,
    h0_cp,
    o_cp,
    intermediate_states_cp,
    tree_parent_ids_cp,
    tree_depths_cp,
    seq_len,
    stream,
    gpu_arch="",
    split_v=False,
):
    n = h0_cp.shape[0]
    hv = h0_cp.shape[1]
    k = h0_cp.shape[2]
    v = h0_cp.shape[3]
    h = q_cp.shape[2]

    compiled = _compile_tree(n, h, hv, k, v, seq_len, stream, gpu_arch=gpu_arch, split_v=split_v)

    def wrap(arr, assumed_align=16):
        return from_dlpack(arr, assumed_align=assumed_align)

    compiled(
        wrap(h0_cp, 32),
        wrap(q_cp),
        wrap(k_cp),
        wrap(v_cp),
        wrap(a_cp),
        wrap(b_cp),
        wrap(A_log_cp),
        wrap(dt_bias_cp),
        wrap(o_cp),
        wrap(intermediate_states_cp, 32),
        wrap(tree_parent_ids_cp),
        wrap(tree_depths_cp),
        seq_len,
        stream,
    )


def _default_tree(n, seq_len):
    parent = -np.ones((n, seq_len), dtype=np.int32)
    depth = np.zeros((n, seq_len), dtype=np.int32)
    for i_n in range(n):
        if seq_len > 0:
            parent[i_n, 0] = -1
            depth[i_n, 0] = 0
        for node_idx in range(1, seq_len):
            if node_idx == seq_len - 1:
                parent[i_n, node_idx] = -1
                depth[i_n, node_idx] = 0
            else:
                parent_idx = max(0, (node_idx - 1) // 2)
                parent[i_n, node_idx] = parent_idx
                depth[i_n, node_idx] = depth[i_n, parent_idx] + 1
    return parent, depth


def run_test_tree(
    n=2,
    h=4,
    hv=8,
    k=128,
    v=128,
    seq_len=8,
    skip_ref_check=False,
    tolerance=0.12,
    warmup=3,
    iterations=100,
    gpu_arch="",
    split_v=False,
):
    dt = cp.float16
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    scale = k ** -0.5

    q_f32 = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    k_f32 = np.random.randn(n, seq_len, h, k).astype(np.float32) * 0.1
    v_f32 = np.random.randn(n, seq_len, hv, v).astype(np.float32) * 0.1
    a_f32 = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    b_f32 = np.random.randn(n, seq_len, hv).astype(np.float32) * 0.1
    A_log_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    dt_bias_f32 = np.random.randn(hv).astype(np.float32) * 0.1
    h0_f32 = np.random.randn(n, hv, k, v).astype(np.float32) * 0.01
    parent_np, depth_np = _default_tree(n, seq_len)

    q_cp = cp.asarray(q_f32.astype(np.float16))
    k_cp = cp.asarray(k_f32.astype(np.float16))
    v_cp = cp.asarray(v_f32.astype(np.float16))
    a_cp = cp.asarray(a_f32.astype(np.float16))
    b_cp = cp.asarray(b_f32.astype(np.float16))
    A_log_cp = cp.asarray(A_log_f32)
    dt_bias_cp = cp.asarray(dt_bias_f32.astype(np.float16))
    h0_cp = cp.asarray(h0_f32)
    parent_cp = cp.asarray(parent_np)
    depth_cp = cp.asarray(depth_np)
    o_cp = cp.zeros((n, seq_len, hv, v), dtype=dt)
    interm_cp = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)

    for _ in range(warmup):
        _run_tree_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp, A_log_cp, dt_bias_cp, h0_cp, o_cp, interm_cp,
            parent_cp, depth_cp, seq_len, stream, gpu_arch, split_v=split_v)
    cp.cuda.get_current_stream().synchronize()

    if not skip_ref_check:
        o_check = cp.zeros((n, seq_len, hv, v), dtype=dt)
        interm_check = cp.zeros((n, seq_len, hv, k, v), dtype=cp.float32)
        _run_tree_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp, A_log_cp, dt_bias_cp, h0_cp, o_check, interm_check,
            parent_cp, depth_cp, seq_len, stream, gpu_arch, split_v=split_v)
        cp.cuda.get_current_stream().synchronize()
        o_ref, interm_ref = _run_numpy_tree_reference(
            q_f32, k_f32, v_f32, a_f32, b_f32, A_log_f32, dt_bias_f32, h0_f32, parent_np, depth_np,
            n, h, hv, k, v, seq_len, scale)
        np.testing.assert_allclose(cp.asnumpy(o_check).astype(np.float32), o_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_decode_tree] Output reference check PASSED.")
        np.testing.assert_allclose(cp.asnumpy(interm_check), interm_ref, atol=tolerance, rtol=1e-2)
        print("[gdn_decode_tree] Intermediate states reference check PASSED.")

    cp.cuda.get_current_stream().synchronize()
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record(cp.cuda.get_current_stream())
    t0 = time.perf_counter()
    for _ in range(iterations):
        _run_tree_decode_jit(
            q_cp, k_cp, v_cp, a_cp, b_cp, A_log_cp, dt_bias_cp, h0_cp, o_cp, interm_cp,
            parent_cp, depth_cp, seq_len, stream, gpu_arch, split_v=split_v)
    end_event.record(cp.cuda.get_current_stream())
    end_event.synchronize()
    wall_us = (time.perf_counter() - t0) * 1e6 / iterations
    event_us = cp.cuda.get_elapsed_time(start_event, end_event) * 1000.0 / iterations
    print("[gdn_decode_tree] Latency: %.4f us cuda_event, %.4f us wall (n=%d h=%d hv=%d k=%d v=%d S=%d)" %
          (event_us, wall_us, n, h, hv, k, v, seq_len))
    return event_us


def _parse_int_list(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_sweep_tree(args):
    batches = _parse_int_list(args.sweep_batches) or [args.n]
    seq_lens = _parse_int_list(args.sweep_seq_lens) or [args.seq_len]
    print("batch,seq_len,h,hv,k,v,latency_us")
    for n in batches:
        for seq_len in seq_lens:
            latency_us = run_test_tree(
                n=n,
                h=args.h,
                hv=args.hv,
                k=args.k,
                v=args.v,
                seq_len=seq_len,
                skip_ref_check=not args.sweep_ref_check,
                tolerance=args.tolerance,
                warmup=args.warmup,
                iterations=args.iterations,
                gpu_arch=args.gpu_arch,
                split_v=args.split_v,
            )
            print("%d,%d,%d,%d,%d,%d,%.4f" % (n, seq_len, args.h, args.hv, args.k, args.v, latency_us))


def main():
    args = _parsed_args
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("GPU required.")
    cp.random.seed(42)
    np.random.seed(42)

    if args.export_only:
        export_gdn_decode_tree(
            n=AOT_PLACEHOLDER_N,
            h=AOT_PLACEHOLDER_H,
            hv=AOT_PLACEHOLDER_HV,
            k=AOT_PLACEHOLDER_K,
            v=AOT_PLACEHOLDER_V,
            output_dir=args.output_dir,
            file_name=args.file_name,
            function_prefix=args.function_prefix,
            gpu_arch=args.gpu_arch,
            split_v=args.split_v,
            precomputed=args.precomputed,
        )
        return

    if args.sweep:
        run_sweep_tree(args)
        return

    run_test_tree(
        n=args.n,
        h=args.h,
        hv=args.hv,
        k=args.k,
        v=args.v,
        seq_len=args.seq_len,
        skip_ref_check=args.skip_ref_check,
        tolerance=args.tolerance,
        warmup=args.warmup,
        iterations=args.iterations,
        gpu_arch=args.gpu_arch,
        split_v=args.split_v,
    )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="CuTe DSL GDN DDTree decode: AOT export and test (Ampere+).")
    p.add_argument("--export_only", action="store_true", help="AOT export only (no test run)")
    p.add_argument("--output_dir", type=str, default="./gdn_tree_artifacts")
    p.add_argument("--file_name", type=str, default="gdn_decode_tree")
    p.add_argument("--function_prefix", type=str, default="gdn_decode_tree")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--hv", type=int, default=8)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--v", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=8, help="DDTree verify nodes S (1 .. T_MAX=%d)" % T_MAX_AOT)
    p.add_argument("--sweep", action="store_true", help="Run a batch/seq_len latency sweep.")
    p.add_argument("--sweep_batches", type=str, default="", help="Comma-separated batch sizes for --sweep.")
    p.add_argument("--sweep_seq_lens", type=str, default="", help="Comma-separated DDTree verify sizes for --sweep.")
    p.add_argument("--sweep_ref_check", action="store_true", help="Run NumPy reference checks during --sweep.")
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--gpu_arch", type=str, default="", help="Target GPU arch for export (e.g. sm_110).")
    p.add_argument("--split_v", action="store_true", help="Use one CTA per (batch, hv, V tile).")
    p.add_argument("--precomputed", action="store_true",
                   help="Export split-v kernel variant that consumes precomputed q/k scales and gate values.")
    return p.parse_known_args(args=argv)[0]


if __name__ == "__main__":
    _parsed_args = _parse_args(_saved_argv)
    main()
