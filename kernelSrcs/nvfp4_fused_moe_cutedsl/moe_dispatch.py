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
"""SM120/SM121 MoE dispatch layer -- workspace, compilation, and launch.

Provides workspace allocation, kernel compilation, and launch helpers for
``MoEDecodeKernel`` / ``MoEPrefillKernel``. Supports both decode and prefill
backends with token-count-based selection (cutover at routed_rows == 640).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch

from cute_dsl_utils import (
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from moe_decode_kernel import MoEDecodeKernel
from moe_prefill_kernel import MoEPrefillKernel

# Constants
_NVFP4_BLOCK_SIZE = 16
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
SF_VEC_SIZE = 16
logger = logging.getLogger(__name__)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _torch_to_cutlass_io_dtype(dtype: torch.dtype):
    """Map a torch activation dtype to the matching cutlass dtype."""
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise ValueError(
        f"SM120 MoE input/output dtype must be bfloat16 or float16, got {dtype}."
    )


def _convert_sf_from_mma_layout(
    sf_6d: torch.Tensor,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    sf_k = (k + sf_vec_size - 1) // sf_vec_size
    m_tiles = (m + 127) // 128
    k_tiles = (sf_k + 3) // 4
    sf_storage = sf_6d.permute(5, 2, 4, 0, 1, 3).contiguous()
    padded_m = m_tiles * 128
    padded_sf_k = k_tiles * 4
    return sf_storage.reshape(num_groups * padded_m, padded_sf_k)


def _select_moe_mma_tiler_mn(routed_rows: int, n: int) -> Tuple[int, int]:
    """Select optimal MoE tile shape based on routed rows and N dimension."""
    sm_count = get_num_sm(torch.device("cuda"))
    coarse_tile = (128, 128)
    coarse_tiles = ((routed_rows + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    if routed_rows <= 128 and coarse_tiles < max(1, sm_count // 2):
        return (64, 128)
    return (128, 128)


# Decode workspace
@dataclass(kw_only=True)
class Sm120DecodeMoEWorkspace:
    """Scratch buffers for one SM120 decode MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device

    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor
    active_expert_count: torch.Tensor
    weight_expert_ids: torch.Tensor
    global_to_local_expert: torch.Tensor

    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None


def allocate_sm120_decode_workspace(
    *,
    state_E: int,
    weight_E: int,
    max_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
) -> Sm120DecodeMoEWorkspace:
    """Allocate workspace buffers for the SM120 decode MoE kernel."""
    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    workspace = Sm120DecodeMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(state_E, max_rows, dtype=torch.int32, device=device),
        token_weights=torch.zeros(
            state_E, max_rows, dtype=torch.float32, device=device
        ),
        packed_input=torch.empty(
            state_E, max_rows, k // 2, dtype=torch.uint8, device=device
        ),
        packed_input_scale=torch.empty(
            state_E, rows_pad_k, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        weight_expert_ids=torch.arange(state_E, dtype=torch.int32, device=device),
        global_to_local_expert=torch.empty(weight_E, dtype=torch.int32, device=device),
    )

    sf_dtype = cutlass.Float8E4M3FN
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return workspace


# Weight views
@dataclass
class _WeightViews:
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None
    w1_alpha: torch.Tensor | None = None
    w2_alpha: torch.Tensor | None = None
    _w13_sf_storage: torch.Tensor | None = None
    _down_sf_storage: torch.Tensor | None = None


_WEIGHT_CACHE: Dict[Tuple, _WeightViews] = {}


def _get_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    is_gated: bool = True,
) -> _WeightViews:
    """Create permuted weight views for the decode/prefill kernel.

    For gated activations (SwiGLU/GeGLU) the kernel expects concatenated w13
    data with shape [2*n, k//2, E]. For non-gated activations the kernel
    expects w1 data with shape [n, k//2, E].
    """
    if n % _LEVEL_TILE_N != 0:
        raise ValueError(
            f"intermediate_size ({n}) must be a multiple of {_LEVEL_TILE_N} "
            f"for the SM120 MoE kernel tile alignment."
        )

    key = (
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
        n,
        k,
        is_gated,
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    w13 = w1_fp4.permute(1, 2, 0)
    down = w2_fp4.permute(1, 2, 0)

    sf_dtype = cutlass.Float8E4M3FN
    w1_n_dim = 2 * n if is_gated else n
    num_experts_local = w1_fp4.shape[0]
    w13_sf_contiguous = _convert_sf_from_mma_layout(
        w1_blockscale,
        m=w1_n_dim,
        k=k,
        num_groups=num_experts_local,
    ).contiguous()
    down_sf_contiguous = _convert_sf_from_mma_layout(
        w2_blockscale,
        m=k,
        k=n,
        num_groups=w2_fp4.shape[0],
    ).contiguous()

    views = _WeightViews(
        w13_fp4=w13.view(torch.float4_e2m1fn_x2),
        down_fp4=down.view(torch.float4_e2m1fn_x2),
        sfb_w13_ptr=make_ptr(
            sf_dtype,
            w13_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        sfb_down_ptr=make_ptr(
            sf_dtype,
            down_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        w1_alpha=w1_alphas.contiguous().to(torch.float32),
        w2_alpha=w2_alphas.contiguous().to(torch.float32),
    )
    views._w13_sf_storage = w13_sf_contiguous
    views._down_sf_storage = down_sf_contiguous
    _WEIGHT_CACHE[key] = views
    return views


# Decode kernel compilation
_DECODE_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_decode_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    activation: str = "swiglu",
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    io_dtype=cutlass.BFloat16,
):
    """Compile (or retrieve cached) the SM120 decode MoE kernel."""
    sf_vec_size = 16
    sm_count = get_num_sm(torch.device("cuda"))
    mac = min(get_max_active_clusters(1), sm_count)

    routed_rows = m * num_topk
    mma_tiler_mn = (128, 128)
    if num_topk > 1:
        mma_tiler_mn = _select_moe_mma_tiler_mn(routed_rows, n)

    cache_key = (
        "decode",
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
        io_dtype,
    )
    cached = _DECODE_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = io_dtype
    alpha_dtype = cutlass.Float32

    kernel = MoEDecodeKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        activation=activation,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        io_dtype=io_dtype,
    )

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype, (m * num_topk,), assumed_align=topk_ids_align
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m * num_topk,), assumed_align=4
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (max_rows, k, state_E), stride_order=(1, 0, 2), assumed_align=16
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * max_rows * (k // 2),), assumed_align=16
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (state_E * rows_pad_k * cols_pad_k,), assumed_align=16
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    _is_gated = activation in ("swiglu", "geglu")
    w1_n_dim = 2 * n if _is_gated else n
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (w1_n_dim, k, weight_E), stride_order=(1, 0, 2), assumed_align=16
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (k, n, weight_E), stride_order=(1, 0, 2), assumed_align=16
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E,), assumed_align=4
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (weight_E,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (weight_E,), assumed_align=16
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype, (m, k), stride_order=(1, 0), assumed_align=16
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (state_E, max_rows), stride_order=(1, 0), assumed_align=4
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (state_E, max_rows), stride_order=(1, 0), assumed_align=16
    )
    compiled = cute.compile(
        kernel,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        active_expert_count_fake,
        weight_expert_ids_fake,
        global_to_local_expert_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        mac,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )

    result = (compiled, mac)
    _DECODE_KERNEL_CACHE[cache_key] = result
    return result


def _expand_to_experts(t: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Broadcast a scalar or [1] tensor to [num_experts]."""
    if t.numel() == 1:
        return t.expand(num_experts).contiguous()
    return t.contiguous().to(torch.float32)


def launch_sm120_decode_moe(
    *,
    workspace: Sm120DecodeMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    activation: str = "swiglu",
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
) -> torch.Tensor:
    """Launch the SM120 decode MoE kernel."""
    io_dtype = _torch_to_cutlass_io_dtype(a.dtype)
    if scatter_output.dtype != a.dtype:
        raise ValueError(
            "SM120 decode MoE requires scatter_output.dtype == a.dtype "
            f"(got a.dtype={a.dtype}, scatter_output.dtype={scatter_output.dtype})."
        )

    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)

    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    compiled, _ = _get_decode_kernel(
        workspace.state_E,
        num_experts,
        num_tokens,
        k,
        n,
        top_k,
        workspace.max_rows,
        activation=activation,
        topk_ids_dtype=torch.int32,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        io_dtype=io_dtype,
    )

    scatter_output.zero_()

    compiled(
        a,
        flat_ids,
        flat_weights,
        workspace.packed_a_view,
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat,
        workspace.scale_flat,
        workspace.barrier_count,
        workspace.barrier_epoch,
        weights.w13_fp4,
        weights._w13_sf_storage.data_ptr(),
        weights.down_fp4,
        weights._down_sf_storage.data_ptr(),
        workspace.row_counts,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output,
        workspace.token_map,
        workspace.token_weights,
    )

    return scatter_output


# Prefill backend

_DECODE_PREFILL_CUTOVER_PAIRS = 640
_PREFILL_SLICE_CHUNK = 2


def select_sm120_moe_backend(*, num_tokens: int, num_topk: int) -> str:
    """Pick decode or prefill backend based on routed-pair count."""
    routed_rows = num_tokens * num_topk
    if routed_rows <= _DECODE_PREFILL_CUTOVER_PAIRS:
        return "decode"
    return "prefill"


@dataclass(kw_only=True)
class Sm120PrefillMoEWorkspace:
    """Scratch buffers for one SM120 prefill MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device

    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor

    routed_rows_capacity: int
    physical_tiles_capacity: int
    task_capacity: int
    expert_write_rows: torch.Tensor
    expert_tile_base: torch.Tensor
    pair_head: torch.Tensor
    producers_done_count: torch.Tensor
    all_work_published: torch.Tensor
    task_head: torch.Tensor
    task_tail: torch.Tensor
    task_ready: torch.Tensor
    task_expert: torch.Tensor
    task_m_tile: torch.Tensor
    task_slice_begin: torch.Tensor
    task_slice_count: torch.Tensor
    task_valid_rows: torch.Tensor
    tile_write_count: torch.Tensor

    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None


def _prefill_task_geometry(state_E: int, n: int, routed_rows: int):
    """Compute task queue dimensions from problem geometry."""
    routed_rows = max(1, routed_rows)
    base_m_tiles = _align_up(routed_rows, _LEVEL_TILE_M) // _LEVEL_TILE_M
    active_expert_upper_bound = min(state_E, routed_rows)
    max_m_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_cnt = max(1, (n + _LEVEL_TILE_N - 1) // _LEVEL_TILE_N)
    slice_groups = max(
        1, (gate_tile_cnt + _PREFILL_SLICE_CHUNK - 1) // _PREFILL_SLICE_CHUNK
    )
    max_tasks = max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


def allocate_sm120_prefill_workspace(
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
) -> Sm120PrefillMoEWorkspace:
    """Allocate workspace buffers for the SM120 prefill MoE kernel."""
    physical_tiles, _, max_tasks = _prefill_task_geometry(state_E, n, routed_rows)
    rows_padded = physical_tiles * _LEVEL_TILE_M
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    workspace = Sm120PrefillMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=rows_padded,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        routed_rows_capacity=routed_rows,
        physical_tiles_capacity=physical_tiles,
        task_capacity=max_tasks,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(rows_padded, dtype=torch.int32, device=device),
        token_weights=torch.zeros(rows_padded, dtype=torch.float32, device=device),
        packed_input=torch.empty(
            1, rows_padded, k // 2, dtype=torch.uint8, device=device
        ),
        packed_input_scale=torch.empty(
            rows_padded, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        expert_write_rows=torch.zeros(state_E, dtype=torch.int32, device=device),
        expert_tile_base=torch.zeros(state_E + 1, dtype=torch.int32, device=device),
        pair_head=torch.zeros(1, dtype=torch.int32, device=device),
        producers_done_count=torch.zeros(1, dtype=torch.int32, device=device),
        all_work_published=torch.zeros(1, dtype=torch.int32, device=device),
        task_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_tail=torch.zeros(1, dtype=torch.int32, device=device),
        task_ready=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_expert=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_m_tile=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_begin=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_count=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_valid_rows=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),
    )

    sf_dtype = cutlass.Float8E4M3FN
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return workspace


# Prefill kernel compilation
class _PrefillMoELaunch:
    """Thin JIT wrapper that makes num_tokens and max_rows runtime Int32."""

    def __init__(self, kernel, k, num_topk):
        self._kernel = kernel
        self._k = k
        self._half_k = k // 2
        self._num_topk = num_topk
        self._cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready_ptr: cute.Pointer,
        task_expert_ptr: cute.Pointer,
        task_m_tile_ptr: cute.Pointer,
        task_slice_begin_ptr: cute.Pointer,
        task_slice_count_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        tile_write_count_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        max_phys_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1))
        )
        topk_ids = cute.make_tensor(
            topk_ids_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        topk_weights_t = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1)),
        )
        packed_a = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, self._k, 1), stride=(self._k, 1, rows_padded * self._k)
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout((rows_padded * self._half_k,), stride=(1,)),
        )
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((rows_padded * self._cols_pad_k,), stride=(1,)),
        )
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_ready = cute.make_tensor(
            task_ready_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_m_tile = cute.make_tensor(
            task_m_tile_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_begin = cute.make_tensor(
            task_slice_begin_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_count = cute.make_tensor(
            task_slice_count_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        tile_write_count = cute.make_tensor(
            tile_write_count_ptr,
            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),
        )
        self._kernel(
            a_input,
            topk_ids,
            topk_weights_t,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights_t,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )


_PREFILL_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_prefill_kernel(
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    activation: str = "swiglu",
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    io_dtype=cutlass.BFloat16,
):
    """Compile (or retrieve cached) the SM120 prefill MoE kernel."""
    sf_vec_size = 16
    sm_count = get_num_sm(torch.device("cuda"))
    mac = min(get_max_active_clusters(1), sm_count)

    cache_key = (
        "prefill",
        E,
        k,
        n,
        num_topk,
        mac,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
        io_dtype,
    )
    cached = _PREFILL_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = io_dtype
    alpha_dtype = cutlass.Float32

    kernel = MoEPrefillKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(_LEVEL_TILE_M, _LEVEL_TILE_N),
        activation=activation,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        io_dtype=io_dtype,
    )
    launch = _PrefillMoELaunch(kernel, k=k, num_topk=num_topk)

    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(
        topk_ids_cutlass_dtype,
        topk_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    packed_a_fake = make_ptr(ab_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    scale_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    producers_done_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    all_work_published_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )

    task_ready_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_expert_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_m_tile_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_begin_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_valid_rows_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    tile_write_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )

    _dyn_is_gated = activation in ("swiglu", "geglu")
    w1_n_dim = 2 * n if _dyn_is_gated else n
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (w1_n_dim, k, E), stride_order=(1, 0, 2), assumed_align=16
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype, (k, n, E), stride_order=(1, 0, 2), assumed_align=16
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E + 1,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    compiled = cute.compile(
        launch,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        pair_head_fake,
        producers_done_count_fake,
        all_work_published_fake,
        task_head_fake,
        task_tail_fake,
        task_ready_fake,
        task_expert_fake,
        task_m_tile_fake,
        task_slice_begin_fake,
        task_slice_count_fake,
        task_valid_rows_fake,
        tile_write_count_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        expert_write_rows_fake,
        expert_tile_base_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        1,
        1,
        1,
        1,
        1,  # runtime Int32 placeholders
        mac,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )

    result = (compiled, mac)
    _PREFILL_KERNEL_CACHE[cache_key] = result
    return result


def launch_sm120_prefill_moe(
    *,
    workspace: Sm120PrefillMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    activation: str = "swiglu",
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
) -> torch.Tensor:
    """Launch the SM120 prefill MoE kernel."""
    io_dtype = _torch_to_cutlass_io_dtype(a.dtype)
    if scatter_output.dtype != a.dtype:
        raise ValueError(
            "SM120 prefill MoE requires scatter_output.dtype == a.dtype "
            f"(got a.dtype={a.dtype}, scatter_output.dtype={scatter_output.dtype})."
        )

    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)

    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    compiled, _ = _get_prefill_kernel(
        num_experts,
        num_tokens,
        k,
        n,
        top_k,
        workspace.max_rows,
        activation=activation,
        topk_ids_dtype=torch.int32,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        io_dtype=io_dtype,
    )

    scatter_output.zero_()

    compiled(
        a.data_ptr(),
        flat_ids.data_ptr(),
        flat_weights.data_ptr(),
        workspace.packed_a_view.data_ptr(),
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat.data_ptr(),
        workspace.scale_flat.data_ptr(),
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.pair_head,
        workspace.producers_done_count,
        workspace.all_work_published,
        workspace.task_head,
        workspace.task_tail,
        workspace.task_ready.data_ptr(),
        workspace.task_expert.data_ptr(),
        workspace.task_m_tile.data_ptr(),
        workspace.task_slice_begin.data_ptr(),
        workspace.task_slice_count.data_ptr(),
        workspace.task_valid_rows.data_ptr(),
        workspace.tile_write_count.data_ptr(),
        weights.w13_fp4,
        weights._w13_sf_storage.data_ptr(),
        weights.down_fp4,
        weights._down_sf_storage.data_ptr(),
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output.data_ptr(),
        workspace.token_map.data_ptr(),
        workspace.token_weights.data_ptr(),
        num_tokens,
        workspace.max_rows,
        workspace.physical_tiles_capacity * _LEVEL_TILE_M,
        workspace.task_capacity,
        workspace.physical_tiles_capacity,
    )

    return scatter_output


# Unified dispatch

_Sm120Workspace = Union[Sm120DecodeMoEWorkspace, Sm120PrefillMoEWorkspace]


def launch_sm120_moe(
    *,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    scatter_output: torch.Tensor,
    activation: str = "swiglu",
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
) -> torch.Tensor:
    """Unified SM120 MoE dispatch -- selects decode or prefill by token count.

    This is a simplified, allocation-per-call version intended for use by
    the in-repo MoE CLI drivers; it does not use workspace caching.
    """
    num_tokens = topk_ids.size(0)
    k = a.size(1)
    is_gated = activation in ("swiglu", "geglu")
    w1_out_dim = w1_weight.size(1)
    if is_gated and w1_out_dim % 2 != 0:
        raise ValueError(
            f"w1_weight dim 1 ({w1_out_dim}) must be even for gated activations"
        )
    intermediate_size = w1_out_dim // 2 if is_gated else w1_out_dim
    n = intermediate_size
    routed_rows = num_tokens * top_k

    weights = _get_weight_views(
        w1_fp4=w1_weight,
        w1_blockscale=w1_weight_sf,
        w2_fp4=w2_weight,
        w2_blockscale=w2_weight_sf,
        w1_alphas=w1_alpha,
        w2_alphas=w2_alpha,
        n=n,
        k=k,
        is_gated=is_gated,
    )

    backend = select_sm120_moe_backend(num_tokens=num_tokens, num_topk=top_k)
    if backend == "prefill" and num_local_experts != num_experts:
        logger.warning(
            "Falling back from prefill to decode because prefill backend does "
            "not support expert-parallel layouts yet: num_local_experts=%d, "
            "num_experts=%d",
            num_local_experts,
            num_experts,
        )
        backend = "decode"

    if backend == "prefill":
        workspace = allocate_sm120_prefill_workspace(
            state_E=num_local_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=top_k,
            device=a.device,
        )
        return launch_sm120_prefill_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=w1_alpha,
            down_input_scale=fc2_input_scale,
            scatter_output=scatter_output,
            num_experts=num_experts,
            num_tokens=num_tokens,
            k=k,
            n=n,
            top_k=top_k,
            activation=activation,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
        )
    workspace = allocate_sm120_decode_workspace(
        state_E=num_local_experts,
        weight_E=num_experts,
        max_rows=max(1, routed_rows),
        k=k,
        n=n,
        num_topk=top_k,
        device=a.device,
    )
    return launch_sm120_decode_moe(
        workspace=workspace,
        weights=weights,
        a=a,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        input_gs=w1_alpha,
        down_input_scale=fc2_input_scale,
        scatter_output=scatter_output,
        num_experts=num_experts,
        num_tokens=num_tokens,
        k=k,
        n=n,
        top_k=top_k,
        activation=activation,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
    )


__all__ = [
    "Sm120DecodeMoEWorkspace",
    "Sm120PrefillMoEWorkspace",
    "allocate_sm120_decode_workspace",
    "allocate_sm120_prefill_workspace",
    "launch_sm120_decode_moe",
    "launch_sm120_prefill_moe",
    "launch_sm120_moe",
    "select_sm120_moe_backend",
]
