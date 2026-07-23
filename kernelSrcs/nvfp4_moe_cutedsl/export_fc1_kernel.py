#!/usr/bin/env python3
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

"""AOT export for split NVFP4 MoE FC1 gather grouped GEMM."""

from __future__ import annotations

import argparse
import os
import sys


def export_fc1(args: argparse.Namespace) -> tuple[str, str]:
    import cuda.bindings.driver as cuda
    import cupy as cp
    import cutlass
    import cutlass.cute as cute

    globals().update({"cuda": cuda, "cutlass": cutlass, "cute": cute})

    from export_common import (
        M_TILE_SIZE,
        SF_VEC_SIZE,
        allocate,
        atom_scale_bytes,
        get_max_active_clusters,
        make_ptr,
        resolve_activation_type,
        verify_export,
    )
    from blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        BlockScaledContiguousGatherGroupedGemmKernel,
    )

    cp.cuda.Device(0).use()

    activation_type = resolve_activation_type(args.activation)
    is_swiglu = args.activation == "swiglu"
    dummy_orig_m = args.dummy_tokens
    dummy_m = M_TILE_SIZE * args.dummy_experts
    dummy_k = args.dummy_hidden_size
    dummy_n = args.dummy_intermediate_size * 2 if is_swiglu else args.dummy_intermediate_size
    dummy_intermediate = dummy_n // 2 if is_swiglu else dummy_n
    cluster_shape_mn = (1, 1)
    mma_tiler_mn = (M_TILE_SIZE, args.mma_tiler_n)

    buffers: list = []
    a = allocate((dummy_orig_m, dummy_k // 2), cp.uint8, buffers)
    b = allocate((args.dummy_experts, dummy_n, dummy_k // 2), cp.uint8, buffers)
    a_sf = allocate((dummy_orig_m, dummy_k // SF_VEC_SIZE), cp.uint8, buffers)
    b_sf = allocate(atom_scale_bytes(dummy_n, dummy_k, args.dummy_experts), cp.uint8, buffers)
    c = allocate((dummy_m, dummy_intermediate // 2), cp.uint8, buffers)
    c_sf = allocate(atom_scale_bytes(dummy_m, dummy_intermediate), cp.uint8, buffers)
    alpha = allocate((args.dummy_experts,), cp.float32, buffers)
    tile_group = allocate((dummy_m // M_TILE_SIZE,), cp.int32, buffers)
    tile_limit = allocate((dummy_m // M_TILE_SIZE,), cp.int32, buffers)
    token_map = allocate((dummy_m,), cp.int32, buffers)
    num_tiles = allocate((1,), cp.int32, buffers)
    input_global_scale = allocate((args.dummy_experts,), cp.float32, buffers)
    down_input_scale = allocate((args.dummy_experts,), cp.float32, buffers)

    ptrs = (
        make_ptr(cutlass.Float4E2M1FN, a.data.ptr, assumed_align=32),
        make_ptr(cutlass.Float4E2M1FN, b.data.ptr, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, a_sf.data.ptr, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, b_sf.data.ptr, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, c.data.ptr, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, c_sf.data.ptr, assumed_align=16),
        make_ptr(cutlass.Float32, alpha.data.ptr, assumed_align=16),
        make_ptr(cutlass.Float32, input_global_scale.data.ptr, assumed_align=16),
        make_ptr(cutlass.Float32, down_input_scale.data.ptr, assumed_align=16),
        make_ptr(cutlass.Int32, tile_group.data.ptr),
        make_ptr(cutlass.Int32, tile_limit.data.ptr),
        make_ptr(cutlass.Int32, token_map.data.ptr),
        make_ptr(cutlass.Int32, num_tiles.data.ptr),
    )

    kernel = BlockScaledContiguousGatherGroupedGemmKernel(
        sf_vec_size=SF_VEC_SIZE,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=True,
        topk=args.dummy_top_k,
        raster_along_m=False,
        b_tensor_l_sizes=(args.dummy_experts,),
        activation_type=activation_type,
    )
    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    @cute.jit
    def single_b_wrapper(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        input_global_scale_ptr: cute.Pointer,
        down_input_scale_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        activation_type: cutlass.Constexpr,
    ):
        return kernel.wrapper(
            a_ptr,
            (b_ptr,),
            a_sf_ptr,
            (b_sf_ptr,),
            c_ptr,
            c_sf_ptr,
            (alpha_ptr,),
            input_global_scale_ptr,
            down_input_scale_ptr,
            tile_idx_to_group_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            token_id_mapping_ptr,
            num_non_exiting_tiles_ptr,
            orig_m,
            m,
            n,
            k,
            l,
            tile_size,
            scaling_vector_size,
            max_active_clusters,
            stream,
            activation_type=activation_type,
        )

    compiled = cute.compile(
        single_b_wrapper,
        *ptrs,
        dummy_orig_m,
        dummy_m,
        dummy_n,
        dummy_k,
        # Runtime L (num_experts); dummy value only drives AOT tracing, not the cubin.
        args.dummy_experts,
        tile_size=M_TILE_SIZE,
        scaling_vector_size=SF_VEC_SIZE,
        max_active_clusters=get_max_active_clusters(cluster_shape_mn),
        stream=stream,
        activation_type=activation_type,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    compiled.export_to_c(args.output_dir, args.file_name, args.function_prefix)
    return verify_export(args.output_dir, args.file_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", choices=["relu2", "swiglu", "geglu"], required=True)
    parser.add_argument("--mma_tiler_n", type=int, choices=[128, 256], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    parser.add_argument("--dummy-tokens", type=int, default=128)
    parser.add_argument("--dummy-experts", type=int, default=8)
    parser.add_argument("--dummy-top-k", type=int, default=2)
    parser.add_argument("--dummy-hidden-size", type=int, default=256)
    parser.add_argument("--dummy-intermediate-size", type=int, default=128)
    parser.add_argument("--export_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    header, obj = export_fc1(parse_args())
    print(f"Exported {header}")
    print(f"Exported {obj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
