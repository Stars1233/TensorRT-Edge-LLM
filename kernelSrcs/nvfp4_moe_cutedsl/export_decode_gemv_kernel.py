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
"""AOT export script for the NVFP4 MoE decode GEMV kernel.

Exports a pointer-based JIT wrapper around ``NvFP4MoeGemvKernel`` that
constructs CuTe tensors from raw pointers and runtime ``Int32`` dimensions.
The exported C entry point is shape-agnostic for K / N / topK.

Usage:
    python export_kernel.py \\
        --activation none \\
        --output_dir /tmp/staging \\
        --file_name nvfp4_moe_gemv_up \\
        --function_prefix nvfp4_moe_gemv_up

    python export_kernel.py \\
        --activation silu --gated --output_atomic \\
        --output_dir /tmp/staging \\
        --file_name nvfp4_moe_gemv_down_swiglu \\
        --function_prefix nvfp4_moe_gemv_down_swiglu
"""

import argparse
import os
import sys

import cupy as cp


def export_gemv_variant(args):
    """Export a single GEMV kernel variant."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from cute_dsl_utils import cute_compile_options, make_ptr

    cp.cuda.Device(0).use()

    swiglu_up = getattr(args, "swiglu_up", False)
    activation = args.activation
    gated = args.gated
    output_atomic = args.output_atomic
    verbose = getattr(args, "verbose", False)

    if swiglu_up:
        from moe_decode_gemv import NvFP4MoeGemvSwigluKernel
        print("GEMV variant: swiglu_up (fused gate+up with SwiGLU epilogue)")
        kernel = NvFP4MoeGemvSwigluKernel()
    else:
        from moe_decode_gemv import NvFP4MoeGemvKernel
        print(f"GEMV variant: activation={activation}, gated={gated}, output_atomic={output_atomic}")
        kernel = NvFP4MoeGemvKernel(activation=activation, gated=gated, output_atomic=output_atomic)

    # Dummy dims for tracing (runtime Int32 makes them shape-agnostic)
    _DUMMY_K = 2688
    _DUMMY_N = 1856
    _DUMMY_TOPK = 6
    _DUMMY_NUM_TOKENS = 1

    _DUMMY_K_HALF = _DUMMY_K // 2
    _DUMMY_SF_BLOCKS = (_DUMMY_K + 15) // 16
    _DUMMY_E = 8  # arbitrary for tracing

    class _GemvLaunch:
        """Thin JIT wrapper — accepts pointers + runtime Int32 scalars."""

        def __init__(self, kern):
            self._kernel = kern

        @cute.jit
        def __call__(
            self,
            input_ptr: cute.Pointer,           # [rows, K] fp16
            input2_ptr: cute.Pointer,          # [rows, K] fp16 (gated second input)
            weights_ptr: cute.Pointer,         # [E, K/2, N] uint8
            scales_ptr: cute.Pointer,          # [E, ceil(K/16), N] fp8 e4m3 (uint8)
            global_scales_ptr: cute.Pointer,   # [E] fp32 per-expert global scales
            expert_ids_ptr: cute.Pointer,      # [numTokens * TopK] int32
            gate_scores_ptr: cute.Pointer,     # [numTokens * TopK] fp32
            output_ptr: cute.Pointer,          # output fp16
            K: cutlass.Int32,
            N: cutlass.Int32,
            topK: cutlass.Int32,
            numTokens: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            _make = lambda p: cute.make_tensor(p, layout=cute.make_layout((1,), stride=(1,)))
            self._kernel(
                _make(input_ptr),
                _make(input2_ptr),
                _make(weights_ptr),
                _make(scales_ptr),
                _make(global_scales_ptr),
                _make(expert_ids_ptr),
                _make(gate_scores_ptr),
                _make(output_ptr),
                K,
                N,
                topK,
                numTokens,
                stream=stream,
            )

    launch = _GemvLaunch(kernel)

    # Create fake pointers for AOT tracing
    input_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    input2_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    weights_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    scales_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    global_scales_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    expert_ids_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    gate_scores_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    output_fake = make_ptr(
        cutlass.Float16, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    print("Compiling GEMV kernel via wrapper...")
    compiled = cute.compile(
        launch,
        input_fake,
        input2_fake,
        weights_fake,
        scales_fake,
        global_scales_fake,
        expert_ids_fake,
        gate_scores_fake,
        output_fake,
        _DUMMY_K,
        _DUMMY_N,
        _DUMMY_TOPK,
        _DUMMY_NUM_TOKENS,
        stream,
        options=cute_compile_options(),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Exporting to {args.output_dir}/{args.file_name}.[h|o]")
    compiled.export_to_c(
        file_path=args.output_dir,
        file_name=args.file_name,
        function_prefix=args.function_prefix,
    )

    header = os.path.join(args.output_dir, f"{args.file_name}.h")
    obj = os.path.join(args.output_dir, f"{args.file_name}.o")
    ok = True
    for path, label in [(header, "Header"), (obj, "Object")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            ok = False
        elif os.path.getsize(path) == 0:
            print(f"ERROR: {label} is empty: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)

    print(f"  header: {os.path.getsize(header)} bytes")
    print(f"  object: {os.path.getsize(obj)} bytes")

    if verbose:
        print("\n--- Generated Header (first 50 lines) ---")
        with open(header) as f:
            for i, line in enumerate(f):
                if i >= 50:
                    print("  ... (truncated)")
                    break
                print(f"  {line.rstrip()}")

    return header, obj


def main():
    parser = argparse.ArgumentParser(
        description="Export NVFP4 MoE decode GEMV kernel for AOT compilation"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="none",
        choices=["none", "relu2", "silu"],
        help="Activation function (default: none)",
    )
    parser.add_argument(
        "--gated",
        action="store_true",
        help="Enable gated mode (two input buffers: act(gate) * up)",
    )
    parser.add_argument(
        "--output_atomic",
        action="store_true",
        help="Use fp16 atomicAdd output (for down_proj)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write .h and .o files",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Base name for output files (without extension)",
    )
    parser.add_argument(
        "--function_prefix",
        type=str,
        required=True,
        help="C function name prefix",
    )
    parser.add_argument(
        "--swiglu_up",
        action="store_true",
        help="Export fused gate+up SwiGLU kernel (NvFP4MoeGemvSwigluKernel)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print generated header content"
    )
    args = parser.parse_args()

    export_gemv_variant(args)
    print("\nGEMV kernel export completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
