# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""AOT-compile CuTe DSL kernels into a static library for CMake linking.

Kernel groups:
  gdn              — Gated Delta Net decode/prefill
  fmha             — Fused Multi-Head Attention (Blackwell persistent)
  ssd              — Mamba2 SSM chunk-scan prefill
  gemm             — Talker MLP GEMM (Ampere / Blackwell / BW GeForce)
  nvfp4_moe        — NvFP4 MoE FC1+FC2 grouped GEMM (prefill)
  nvfp4_moe_decode — NvFP4 MoE decode GEMV (scalar K-parallel dot-product)
  nvfp4_fused_moe  — End-to-end NvFP4 fused MoE (Blackwell GeForce)

Usage (run from the repo root):
  python kernelSrcs/build_cutedsl.py                      # build all groups for this GPU
  python kernelSrcs/build_cutedsl.py --kernels gdn        # single group
  python kernelSrcs/build_cutedsl.py --kernels fmha,gdn   # multiple groups
  python kernelSrcs/build_cutedsl.py --gpu_arch sm_110    # override SM detection
  python kernelSrcs/build_cutedsl.py --clean --verbose    # clean rebuild

The GPU SM is auto-detected via cupy / nvidia-smi and only matching variants
are built.  See KERNEL_VARIANTS below for the full variant list.

Output (under {output_dir}/{arch}/{artifact_tag}/):
  libcutedsl_{arch}.a   — merged static archive (kernel objects + DSL runtime)
  include/cutedsl_all.h — umbrella header (#includes every variant header)
  metadata.json         — build provenance + group/variant list for CMake

Prebuilt tarballs:
  For supported targets (e.g. Thor SM110), prebuilt tarballs are committed
  under kernelSrcs/cuteDSLPrebuilt/.  CMake auto-extracts them when the
  artifact directory is absent — no manual build step needed.  To regenerate:
    python kernelSrcs/build_cutedsl.py --gpu_arch sm_110 --arch aarch64 --clean
"""

import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_OUTPUT_DIR = (_SCRIPT_DIR / "../cpp/kernels/cuteDSLArtifact").resolve()
_CUTLASS_DSL_VERSION = "4.5.1"
_CUPY_VERSIONS = {12: ("cupy-cuda12x", "12.3.0"), 13: ("cupy-cuda13x", "13.6.0")}

# Common flag sets for FMHA variants
_LLM = ["--is_causal", "--is_persistent", "--export_only", "--bottom_right_align"]
_LLM_FP8 = _LLM + ["--in_dtype", "Float8E4M3FN"]
_VIT = ["--is_persistent", "--export_only", "--vit_mode"]


@dataclass
class KernelVariant:
    """One compilable kernel variant in the CuTe DSL registry.

    Attributes:
        name:          Unique identifier — used as --file_name / --function_prefix.
        group:         Logical group ("gdn", "fmha", "nvfp4_moe", "nvfp4_fused_moe", "ssd", or "gemm"). cmake sets CUTE_DSL_<GROUP>_ENABLED.
        supported_sms: Explicit SM whitelist. With --kernels ALL, only variants whose
                       supported_sms contains the detected/requested SM are compiled.
        script:        Kernel script path relative to kernelSrcs/.
        script_args:   Args forwarded verbatim after --output_dir/--file_name/--function_prefix.
                       GDN variants MUST include "--export_only" here.

    """
    name: str
    group: str
    supported_sms: list[int]
    script: str
    script_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kernel registry — add new groups/variants here.
#
# Each KernelVariant has a supported_sms whitelist.  Only variants matching
# the target SM are compiled.  All kernel scripts compile device-native
# (no --gpu_arch forwarded), which works uniformly on Linux and QNX.
#
# Groups:
#   gdn              — Gated Delta Net decode/prefill
#   fmha             — Fused Multi-Head Attention (Blackwell persistent)
#   ssd              — Mamba2 SSM chunk-scan prefill
#   gemm             — Talker MLP cuBLAS replacement (Ampere/Blackwell/BW GeForce)
#   nvfp4_moe        — NvFP4 MoE FC1+FC2 grouped GEMM (prefill)
#   nvfp4_moe_decode — NvFP4 MoE decode GEMV (scalar K-parallel dot-product)
#   nvfp4_fused_moe  — End-to-end NvFP4 fused MoE (Blackwell GeForce)
# ---------------------------------------------------------------------------
KERNEL_VARIANTS = [
    # --- GDN group ---
    KernelVariant(
        name="gdn_decode",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_decode.py",
        script_args=["--export_only"],
    ),
    KernelVariant(
        name="gdn_prefill",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_prefill.py",
        script_args=["--export_only"],
    ),
    KernelVariant(
        name="gdn_prefill_blackwell",
        group="gdn",
        supported_sms=[100, 101, 110],
        script="gdn_cutedsl/gdn_prefill_blackwell.py",
        script_args=["--export_only"],
    ),
    # MTP decode: multi-token speculative-decoding verification (Ampere SM80+).
    # Only the cache variant is exported (per-step intermediate state checkpointing for rollback).
    KernelVariant(
        name="gdn_decode_mtp",
        group="gdn",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="gdn_cutedsl/gdn_decode_mtp.py",
        script_args=["--export_only", "--cache_only"],
    ),
    # --- SSD group (Mamba2 SSM chunk scan) ---
    # --- SSD SM80 variants (D×N combinations) ---
    KernelVariant(
        name="ssd_prefill_d128_n128",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "128", "--dstate", "128"],
    ),
    KernelVariant(
        name="ssd_prefill_d64_n128",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "128"],
    ),
    KernelVariant(
        name="ssd_prefill_d128_n64",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "128", "--dstate", "64"],
    ),
    KernelVariant(
        name="ssd_prefill_d64_n64",
        group="ssd",
        supported_sms=[80, 86, 87, 89, 90, 100, 101, 110, 120, 121],
        script="ssd_cutedsl/ssd_prefill.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "64"],
    ),
    # --- SSD Blackwell variants ---
    # Two has_init_states modes per (D, N): the default variant assumes the SSM state
    # at chunk 0 is zero (fast path; covers all current Nemotron-H prefill calls). The
    # `_init_states` variant accepts an optional user-provided initial hidden state at
    # chunk 0, used when prefill carries SSM state across calls (continuous batching,
    # multi-call prefill) or by the SsdCuteDslBlackwellChunkedPrefill unit test.
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n128",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "128"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n128_init_states",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "128",
                     "--has_init_states",
                     "--file_name", "ssd_prefill_blackwell_d64_n128_init_states",
                     "--function_prefix", "ssd_prefill_blackwell_d64_n128_init_states"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n64",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "64"],
    ),
    KernelVariant(
        name="ssd_prefill_blackwell_d64_n64_init_states",
        group="ssd",
        supported_sms=[100, 101, 110],
        script="ssd_cutedsl/ssd_prefill_blackwell.py",
        script_args=["--export_only", "--dim", "64", "--dstate", "64",
                     "--has_init_states",
                     "--file_name", "ssd_prefill_blackwell_d64_n64_init_states",
                     "--function_prefix", "ssd_prefill_blackwell_d64_n64_init_states"],
    ),
    # --- FMHA group ---
    KernelVariant(
        name="fmha_d64",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM,
    ),
    KernelVariant(
        name="fmha_d128",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM,
    ),
    KernelVariant(
        name="fmha_d64_sw",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM + ["--window_size", "4096,-1"],
    ),
    # LLM FP8 input → FP16 output
    KernelVariant(
        name="fmha_d64_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"] + _LLM_FP8,
    ),
    KernelVariant(
        name="fmha_d128_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"] + _LLM_FP8,
    ),
    KernelVariant(
        name="fmha_d64_sw_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,1,64"]
                    + _LLM_FP8 + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="fmha_d128_sw_fp8",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,1,128"]
                    + _LLM_FP8 + ["--window_size", "4096,-1"],
    ),
    KernelVariant(
        name="vit_fmha_d64",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,64", "--k_shape", "1,1024,14,64"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d72",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,72", "--k_shape", "1,1024,14,72"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d80",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,80", "--k_shape", "1,1024,14,80"] + _VIT,
    ),
    KernelVariant(
        name="vit_fmha_d128",
        group="fmha",
        supported_sms=[100, 101, 110],
        script="fmha_cutedsl_blackwell/fmha.py",
        script_args=["--q_shape", "1,1024,14,128", "--k_shape", "1,1024,14,128"] + _VIT,
    ),
    # --- NvFP4 MoE group ---
    # FC1 contiguous grouped GEMM: 2 activations x 2 N-tiles x 2 dtypes = 8 variants
    # (identity is not exported — no production path uses identity FC1 here.)
    KernelVariant(
        name="nvfp4_moe_fc1_relu2_n128_bf16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_n", "128",
                     "--output_dtype", "bf16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc1_relu2_n128_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_n", "128",
                     "--output_dtype", "fp16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc1_relu2_n256_bf16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_n", "256",
                     "--output_dtype", "bf16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc1_relu2_n256_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_n", "256",
                     "--output_dtype", "fp16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc1_swiglu_n128_bf16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_n", "128",
                     "--output_dtype", "bf16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc1_swiglu_n128_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_n", "128",
                     "--output_dtype", "fp16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc1_swiglu_n256_bf16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_n", "256",
                     "--output_dtype", "bf16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc1_swiglu_n256_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc1_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_n", "256",
                     "--output_dtype", "fp16", "--export_only"],
    ),
    # FC2 finalize (grouped GEMM + scatter-reduce): 2 N-tiles x 2 dtypes = 4 variants
    KernelVariant(
        name="nvfp4_moe_fc2_n128_bf16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc2_kernel.py",
        script_args=["--mma_tiler_n", "128",
                     "--output_dtype", "bf16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc2_n128_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc2_kernel.py",
        script_args=["--mma_tiler_n", "128",
                     "--output_dtype", "fp16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc2_n256_bf16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc2_kernel.py",
        script_args=["--mma_tiler_n", "256",
                     "--output_dtype", "bf16", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_moe_fc2_n256_fp16",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_fc2_kernel.py",
        script_args=["--mma_tiler_n", "256",
                     "--output_dtype", "fp16", "--export_only"],
    ),
    # --- NvFP4 MoE Decode GEMV group (SM100+) ---
    # Scalar GEMV for MoE decode: K-parallel vectorized dot-product with
    # atom-layout FP8 block scales (shared with prefill GEMM — no weight/scale
    # duplication). 5 kernel variants compose into pipelines:
    #   Nemotron (relu2):  up_none → dn_relu2
    #   Mixtral  (silu):   up_none → dn_silu
    #   LLaMA SwiGLU:      up_swiglu → dn_none     (fused interleaved FC1)
    KernelVariant(
        name="gemv_up_none",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_decode_gemv_kernel.py",
        script_args=["--activation", "none"],
    ),
    KernelVariant(
        name="gemv_up_swiglu",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_decode_gemv_kernel.py",
        script_args=["--swiglu_up"],
    ),
    KernelVariant(
        name="gemv_dn_relu2",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_decode_gemv_kernel.py",
        script_args=["--activation", "relu2", "--output_atomic"],
    ),
    KernelVariant(
        name="gemv_dn_silu",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_decode_gemv_kernel.py",
        script_args=["--activation", "silu", "--output_atomic"],
    ),
    KernelVariant(
        name="gemv_dn_none",
        group="nvfp4_moe",
        supported_sms=[100, 101, 110],
        script="nvfp4_moe_cutedsl/export_decode_gemv_kernel.py",
        script_args=["--activation", "none", "--output_atomic"],
    ),
    # --- NvFP4 Fused MoE group (SM120/SM121 — Blackwell GeForce) ---
    # Fused route/pack + FC1 + activation + quant + FC2 + scatter kernels.
    # Decode backend: resident-grid barrier between route/pack and compute
    #   phases; best for small routed working sets (num_tokens*top_k <= 640;
    #   see CuteDslNvfp4MoeRunner::kDecodePrefillCutoverRoutedRows).
    # Prefill backend: global task-queue driven producer/consumer overlap;
    #   best for large routed working sets.
    # NvFP4MoEPluginGeforce scope: FP16 io_dtype + {identity, silu, swiglu, gelu, relu2}
    # x {decode, prefill} x {n128 MMA N-tile} = 10 variants. Shape axes
    # N / E / top_k / hidden_size (K) are runtime (shape-polymorphic). The
    # MMA N-tile remains a compile-time variant axis. CuteDslNvfp4MoeRunner
    # currently dispatches n128 and accepts the bounded K set {1024, 2048}.
    # Decode backend, N-tile 128
    KernelVariant(
        name="nvfp4_fused_moe_decode_identity_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "identity", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_silu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "silu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_swiglu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_gelu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "gelu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_decode_relu2_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_decode_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_n", "128", "--export_only"],
    ),

    # Prefill backend, N-tile 128
    KernelVariant(
        name="nvfp4_fused_moe_prefill_identity_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "identity", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_silu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "silu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_swiglu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "swiglu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_gelu_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "gelu", "--mma_tiler_n", "128", "--export_only"],
    ),
    KernelVariant(
        name="nvfp4_fused_moe_prefill_relu2_n128",
        group="nvfp4_fused_moe",
        supported_sms=[120, 121],
        script="nvfp4_fused_moe_cutedsl/export_prefill_kernel.py",
        script_args=["--activation", "relu2", "--mma_tiler_n", "128", "--export_only"],
    ),
    # Prefill backend, N-tile 256 — DISABLED (same bug as decode n256).

    # =====================================================================
    # GEMM group — Talker MLP cuBLAS replacement
    #
    # Dispatch strategy per architecture:
    #   Ampere (SM80-89):
    #     M==1           → decode (16×128×128) + Split-K=4 for SM utilization
    #     M=2-95         → small_prefill (16×128×128)
    #     M=96-192       → medium_prefill (64×128×64) + Split-K=2
    #     M=193-383      → medium_prefill (64×128×64)
    #     M=384-640      → large_prefill (128×128×64) + unpredicated epilogue
    #     M>640          → medium_prefill (64×128×64)
    #
    #   Blackwell DC (SM100-110):  tcgen05 + TMA store
    #     M<=4*SMs                 → small  (64×128)  cluster=(1,2)
    #     low-SM GPU AND M>=256    → 2-CTA  (256×256) cluster=(2,1)
    #     else                     → default (128×128) cluster=(1,1)
    #     (e.g. B100 144 SMs → small cap M<=576, no 2-CTA;
    #      Thor 20 SMs → small cap M<=80, 2-CTA above M>=256;
    #      see cuteDslGemmRunner sBlackwellSmallTileMaxM /
    #          sBlackwell2ctaMinM)
    #
    #   BW GeForce (SM120-121):
    #     M<=64          → small (64×128×64) — persistent+warp-spec+TMA
    #     M>=128         → default (128×128×64)
    # =====================================================================
    # --- Ampere (SM80-89) ---
    KernelVariant(
        name="gemm_ampere_decode_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "1,2048,2048",
            "--cta_tiler_mnk", "16,128,128",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_small_prefill_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "64,2048,2048",
            "--cta_tiler_mnk", "16,128,128",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_medium_prefill_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_large_prefill_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "512,2048,2048",
            "--cta_tiler_mnk", "128,128,64",
            "--atom_layout_mnk", "2,4,1",
            "--num_stages", "3",

            "--export_only",
        ],
    ),
    # Split-K variant for small M (decode): split_k=4 gives 4x more CTAs.
    KernelVariant(
        name="gemm_ampere_splitk4_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere_streamk.py",
        script_args=[
            "--mnk", "1,2048,2048",
            "--cta_tiler_mnk", "16,128,128",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",
            "--split_k", "4",
            "--export_only",
        ],
    ),
    # Split-K=2 for medium M (M=128): doubles CTA count from 32 to 64.
    KernelVariant(
        name="gemm_ampere_splitk2_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere_streamk.py",
        script_args=[
            "--mnk", "128,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",
            "--split_k", "2",
            "--export_only",
        ],
    ),
    # =====================================================================
    # Fused MLP epilogue variants (Plan C: 4→2 kernel launches)
    #
    # FC1 path: GEMM + bias + SiLU fused in epilogue
    # FC2 path: GEMM + bias fused in epilogue
    #
    # Only medium_prefill tile (64×128×64) is fused — it covers the most
    # common prefill M range. Decode (M=1) uses separate bias kernels
    # since the 2us kernel launch is negligible at that scale.
    # =====================================================================
    KernelVariant(
        name="gemm_ampere_medium_bias_silu_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_ampere_medium_bias_fp16",
        group="gemm",
        supported_sms=[80, 86, 87, 89],
        script="gemm_cutedsl/gemm_ampere.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--cta_tiler_mnk", "64,128,64",
            "--atom_layout_mnk", "1,4,1",
            "--num_stages", "3",

            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    # Blackwell DC GEMM tile variants:
    #   "default"  tile=(128,128) cluster=(1,1) — wins for M >= 768 (large MM
    #              has enough M-tiles to keep the GPU busy without sub-tiling)
    #   "small"    tile=(64,128)  cluster=(1,2) — wins for M <= 512  (more
    #              CTAs per wave -> better SM utilization on small/medium M;
    #              cluster (1,2) multicasts B across 2 CTAs in N)
    # See cuteDslGemmRunner::run() for the M threshold dispatch.
    KernelVariant(
        name="gemm_blackwell_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "128,128",
            "--cluster_shape_mn", "1,1",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_bias_silu_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "128,128",
            "--cluster_shape_mn", "1,1",
            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_bias_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "128,128",
            "--cluster_shape_mn", "1,1",
            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_small_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--mma_tiler_mn", "64,128",
            "--cluster_shape_mn", "1,2",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_small_bias_silu_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--mma_tiler_mn", "64,128",
            "--cluster_shape_mn", "1,2",
            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_small_bias_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "256,2048,2048",
            "--mma_tiler_mn", "64,128",
            "--cluster_shape_mn", "1,2",
            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    # 2-CTA variants (tile=256x256, cluster=(2,1)): paired CTAs share a
    # 2x M tile per tcgen05.mma op. On low-SM-count GPUs (Thor 20 SMs)
    # this beats the single-CTA default for M >= 256 by ~15-25%; on
    # high-SM-count (B100 144 SMs) the single-CTA default already
    # saturates compute so 2-CTA isn't dispatched.
    KernelVariant(
        name="gemm_blackwell_2cta_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "256,256",
            "--cluster_shape_mn", "2,1",
            "--use_2cta",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_2cta_bias_silu_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "256,256",
            "--cluster_shape_mn", "2,1",
            "--use_2cta",
            "--fused_epilogue", "bias_silu",
            "--export_only",
        ],
    ),
    KernelVariant(
        name="gemm_blackwell_2cta_bias_fp16",
        group="gemm",
        supported_sms=[100, 101, 103, 110],
        script="gemm_cutedsl/gemm_blackwell.py",
        script_args=[
            "--mnk", "1024,2048,2048",
            "--mma_tiler_mn", "256,256",
            "--cluster_shape_mn", "2,1",
            "--use_2cta",
            "--fused_epilogue", "bias",
            "--export_only",
        ],
    ),
    # BW GeForce (SM120/121): warp-specialized, TMA, persistent tile scheduling.
    # Small tile for M<=64 — more CTAs on N1Auto's 20 SMs.
    KernelVariant(
        name="gemm_bw_geforce_small_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "64,2048,2048", "--tile_shape_mnk", "64,128,64", "--export_only"],
    ),
    # Default tile for M>=128.
    KernelVariant(
        name="gemm_bw_geforce_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "1024,2048,2048", "--tile_shape_mnk", "128,128,64", "--export_only"],
    ),
    KernelVariant(
        name="gemm_bw_geforce_bias_silu_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "1024,2048,2048", "--tile_shape_mnk", "128,128,64",
                     "--fused_epilogue", "bias_silu", "--export_only"],
    ),
    KernelVariant(
        name="gemm_bw_geforce_bias_fp16",
        group="gemm",
        supported_sms=[120, 121],
        script="gemm_cutedsl/gemm_blackwell_geforce.py",
        script_args=["--mnk", "1024,2048,2048", "--tile_shape_mnk", "128,128,64",
                     "--fused_epilogue", "bias", "--export_only"],
    ),
]

# All known group names (set for O(1) membership check — no manual maintenance needed).
_ALL_GROUPS: set[str] = {v.group for v in KERNEL_VARIANTS}


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

def _parse_sm(gpu_arch_str):
    """Parse SM number from "sm_87" → 87, or raise ValueError."""
    s = gpu_arch_str.strip().lower()
    if s.startswith("sm_"):
        s = s[3:]
    try:
        sm = int(s)
    except ValueError:
        raise ValueError(
            f"Invalid --gpu_arch {gpu_arch_str!r}. Expected format: sm_87, sm_100, etc."
        )
    if sm <= 0:
        raise ValueError(
            f"Invalid --gpu_arch {gpu_arch_str!r}: SM number must be positive (got {sm})."
        )
    return sm


def detect_gpu_sm() -> int:
    """Auto-detect the current GPU SM.

    Returns the SM as an integer, e.g. 87 for SM87, 100 for SM100, 110 for SM110.

    Detection order:
      1. cupy.cuda.Device — works on all platforms (Linux, QNX, etc.) since cupy is
         already a required dependency.  compute_capability returns e.g. "87", "100".
      2. nvidia-smi --query-gpu=compute_cap — fallback for environments where cupy
         is not yet importable at this point in the script (rare).

    Raises RuntimeError if both methods fail; caller should re-run with --gpu_arch.
    """
    # 1. Try cupy first — platform-agnostic, already a required dep.
    try:
        import cupy  # noqa: PLC0415
        cap = cupy.cuda.Device(0).compute_capability  # e.g. "87", "100", "110"
        sm = int(cap)
        if sm > 0:
            return sm
    except Exception:
        pass

    # 2. Fall back to nvidia-smi (Linux/x86; not available on QNX).
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Could not detect GPU SM: cupy unavailable and nvidia-smi not found. "
            "Pass --gpu_arch explicitly (e.g. --gpu_arch sm_87)."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed: {result.stderr.strip() or result.stdout.strip()}. "
            "Pass --gpu_arch explicitly to override."
        )
    # compute_cap format from nvidia-smi is "8.7" → 87, "10.0" → 100.
    line = result.stdout.strip().splitlines()[0].strip()
    parts = line.split(".")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise RuntimeError(
            f"Unexpected nvidia-smi compute_cap format: {line!r}. "
            "Pass --gpu_arch explicitly to override."
        )
    return int(parts[0]) * 10 + int(parts[1])


def select_variants(sm: int, kernels_arg: str):
    """Return the list of KernelVariants to compile for the given SM.

    sm:
      Integer SM number (e.g. 87, 100, 110) — used for supported_sms filtering.

    kernels_arg:
      "ALL"         — compile variants whose supported_sms contains the SM.
      "gdn"/"fmha"  — compile variants in that group whose supported_sms contains the SM.
      "gdn,fmha"    — same for the listed groups (unsupported variants are skipped).
    """
    groups_requested = kernels_arg.strip().upper()

    if groups_requested == "ALL":
        selected = [v for v in KERNEL_VARIANTS if sm in v.supported_sms]
        if not selected:
            print(f"WARNING: No CuTe DSL variants support SM{sm}. "
                  f"Check supported_sms in KERNEL_VARIANTS.")
        return selected

    # Parse explicit group list: "fmha,gdn" or "gdn".
    tokens = [t.strip().lower() for t in kernels_arg.split(",")]
    unknown = [t for t in tokens if t not in _ALL_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown kernel group(s): {unknown}. "
            f"Valid groups: {sorted(_ALL_GROUPS)}"
        )

    in_groups = [v for v in KERNEL_VARIANTS if v.group in tokens]
    skipped = [v for v in in_groups if sm not in v.supported_sms]
    selected = [v for v in in_groups if sm in v.supported_sms]

    if skipped:
        names = ", ".join(v.name for v in skipped)
        print(
            f"NOTE: Skipping {len(skipped)} variant(s) not supported on SM{sm}: {names}"
        )

    if selected:
        return selected

    # Explicit group list requested, but no variant in those groups supports the SM.
    requested_variants = [v for v in KERNEL_VARIANTS if v.group in tokens]
    names = ", ".join(v.name for v in requested_variants)
    raise ValueError(
        f"No variants in groups {tokens} support SM{sm}.\n"
        f"Requested variants: {names}\n"
        f"Use --kernels ALL to auto-filter across all groups, or check supported_sms in KERNEL_VARIANTS."
    )


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def detect_arch(override=None):
    if override:
        m = override.lower().replace("-", "_")
        if m in ("x86_64", "amd64"):
            return "x86_64"
        if m in ("aarch64", "arm64"):
            return "aarch64"
        raise ValueError(f"Unsupported --arch: {override!r}. Use 'x86_64' or 'aarch64'.")
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "aarch64"
    raise RuntimeError(
        f"Unsupported architecture: {platform.machine()!r}. Use --arch to override."
    )


def sm_to_artifact_tag(sm: int) -> str:
    return f"sm_{sm}"


def _nvcc_version():
    """Return CUDA version string (e.g. "12.6.0") or None.

    Detection order:
      1. nvcc on PATH
      2. /usr/local/cuda/bin/nvcc  (common on Jetson / embedded devices)
      3. cupy.cuda.runtime          (works on QNX and any platform with cupy)
    """
    # 1 & 2: try nvcc
    for nvcc in ("nvcc", "/usr/local/cuda/bin/nvcc"):
        try:
            out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
            for token in out.split():
                if token.startswith("V") and token[1:2].isdigit():
                    return token[1:].split(",")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # 3: cupy runtime API — runtimeGetVersion() returns e.g. 12060 for 12.6.0
    try:
        import cupy  # noqa: PLC0415
        v = cupy.cuda.runtime.runtimeGetVersion()   # e.g. 12060
        major, rest = divmod(v, 1000)
        minor, patch = divmod(rest, 10)
        return f"{major}.{minor}.{patch}"
    except Exception:
        pass

    return None


def check_dependencies():
    errors = []

    # nvidia-cutlass-dsl
    try:
        ver = importlib.metadata.version("nvidia-cutlass-dsl")
        if ver != _CUTLASS_DSL_VERSION:
            errors.append(
                f"nvidia-cutlass-dsl: found {ver}, need {_CUTLASS_DSL_VERSION}\n"
                f"  Fix: pip install nvidia-cutlass-dsl=={_CUTLASS_DSL_VERSION}"
            )
            lib_dir = None
        else:
            spec = importlib.util.find_spec("nvidia_cutlass_dsl")
            pkg_dir = (
                Path(next(iter(spec.submodule_search_locations)))
                if spec.submodule_search_locations
                else Path(spec.origin).parent
            )
            lib_dir = pkg_dir / "lib"
    except importlib.metadata.PackageNotFoundError:
        errors.append(
            f"nvidia-cutlass-dsl not found.\n"
            f"  Fix: pip install nvidia-cutlass-dsl=={_CUTLASS_DSL_VERSION}"
        )
        lib_dir, ver = None, "unknown"

    # cupy
    cuda_ver = _nvcc_version()
    if cuda_ver:
        major = int(cuda_ver.split(".")[0])
        if major in _CUPY_VERSIONS:
            cupy_pkg, cupy_req = _CUPY_VERSIONS[major]
            try:
                found = importlib.metadata.version(cupy_pkg)
                if found != cupy_req:
                    errors.append(
                        f"cupy: found {cupy_pkg}=={found}, need {cupy_req}\n"
                        f"  Fix: pip install {cupy_pkg}=={cupy_req}"
                    )
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"cupy not found.\n  Fix: pip install {cupy_pkg}=={cupy_req}")
        else:
            errors.append(f"Unsupported CUDA major version {major} for cupy.")
    else:
        errors.append("Could not detect CUDA version (is nvcc on PATH?).")

    if not shutil.which("ar"):
        errors.append("'ar' not found on PATH. Install binutils.")

    # cuda-python (provides `cuda.bindings.driver`, used by all GEMM/FMHA scripts)
    try:
        importlib.metadata.version("cuda-python")
    except importlib.metadata.PackageNotFoundError:
        errors.append("cuda-python not found.\n  Fix: pip install cuda-python")

    if errors:
        print("Dependency check failed:\n" + "\n".join(f"  • {e}" for e in errors))
        sys.exit(1)

    assert ver is not None and lib_dir is not None and cuda_ver is not None
    print(f"  nvidia-cutlass-dsl=={ver} ✓  CUDA {cuda_ver} ✓  ar ✓")
    return ver, lib_dir, cuda_ver


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _compile_one(variant, staging_dir, verbose, sm):
    """Invoke a kernel script to AOT-compile one variant into .o + .h.

    Returns (name, ok, elapsed_secs, error_msg).
    """
    cmd = [sys.executable, str(_SCRIPT_DIR / variant.script)]
    cmd += ["--output_dir", str(staging_dir),
            "--file_name", variant.name,
            "--function_prefix", variant.name]
    cmd += variant.script_args

    env = os.environ.copy()
    if sm == 121 and variant.group == "nvfp4_fused_moe":
        # Build a real SM121 image for DIGITS/GB10. SM120 cubins link, but
        # fail at runtime on SM121 with cudaErrorNoKernelImageForDevice.
        env.setdefault("CUTE_DSL_ARCH", "sm_121a")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, cwd=str(_SCRIPT_DIR), capture_output=not verbose, text=True, env=env
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        # Show the head (traceback / first error) rather than the tail, which is typically more diagnostic.
        return variant.name, False, elapsed, (result.stderr or result.stdout or "")[:4000]
    obj = staging_dir / f"{variant.name}.o"
    hdr = staging_dir / f"{variant.name}.h"
    if not obj.exists() or not hdr.exists():
        # Some variants (e.g. gdn_decode_mtp --cache_only) produce artifacts with
        # a suffix (e.g. gdn_decode_mtp_cache.o/.h).  Accept any .o + .h pair.
        any_objs = list(staging_dir.glob("*.o"))
        any_hdrs = list(staging_dir.glob("*.h"))
        if not any_objs or not any_hdrs:
            return variant.name, False, elapsed, f"{obj.name} / {hdr.name} not found after successful exit"
    return variant.name, True, elapsed, ""


def compile_variants(variants, staging_dirs, jobs, verbose, sm):
    """Compile all selected variants in parallel via a process pool.

    staging_dirs: dict mapping variant.name → Path of its dedicated staging dir.
    """
    print(f"\nCompiling {len(variants)} kernel variant(s) (jobs={jobs})...")
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_compile_one, v, staging_dirs[v.name], verbose, sm): v
            for v in variants
        }
        for future in concurrent.futures.as_completed(futures):
            name, ok, elapsed, msg = future.result()
            print(f"  {'✓' if ok else '✗'} {name:<25} ({elapsed:.1f}s)")
            if not ok:
                failures.append((name, msg))

    if failures:
        for name, msg in failures:
            print(f"\n  [{name}]\n{msg}")
        sys.exit(1)


def _check_obj_name_collision(kernel_objs, runtime_objs):
    kernel_names = {f.name for f in kernel_objs}
    runtime_names = {f.name for f in runtime_objs}
    collision = kernel_names & runtime_names
    if collision:
        raise RuntimeError(
            f"Object name collision between kernel and runtime archives: {collision}\n"
            "Rename the affected kernel variant(s) in KERNEL_VARIANTS to resolve."
        )


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def build(args):
    # Resolve SM: explicit override or auto-detect from the running GPU.
    if args.gpu_arch:
        sm = _parse_sm(args.gpu_arch)
        sm_source = "--gpu_arch override"
    else:
        sm = detect_gpu_sm()
        sm_source = "auto-detected"

    arch = detect_arch(args.arch)
    artifact_tag = sm_to_artifact_tag(sm)
    output_dir = Path(args.output_dir) / arch / artifact_tag

    print(f"Target arch : {arch}")
    print(f"GPU SM      : SM{sm} ({sm_source})")
    print(f"Artifact tag: {artifact_tag}")
    print(f"Output dir  : {output_dir}")

    variants = select_variants(sm, args.kernels)
    if not variants:
        print("No variants selected — nothing to build.")
        return

    # Check dependencies before cleaning — so a failed dep check doesn't
    # silently destroy a previously good build.
    print("\nChecking dependencies...")
    dsl_ver, lib_dir, cuda_ver = check_dependencies()

    groups_selected = sorted({v.group for v in variants})
    print(f"Groups      : {groups_selected}")
    print(f"Variants    : {[v.name for v in variants]}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    # Per-variant staging dirs prevent .o / .h filename collisions when multiple
    # variants share the same underlying script (e.g. all fmha.py invocations).
    root_staging = Path(tempfile.mkdtemp(prefix="cutedsl_build_"))
    try:
        staging_dirs = {}
        for v in variants:
            d = root_staging / v.name
            d.mkdir()
            staging_dirs[v.name] = d

        compile_variants(variants, staging_dirs, args.jobs, args.verbose, sm)

        # Collect all .o files from per-variant staging dirs.
        # Some variants (e.g. gdn_decode_mtp) produce multiple .o files.
        kernel_obj_files = []
        for v in variants:
            kernel_obj_files.extend(sorted(staging_dirs[v.name].glob("*.o")))

        # Extract the DSL runtime static lib into a separate dir to avoid collisions.
        runtime = lib_dir / "libcuda_dialect_runtime_static.a"
        if not runtime.exists():
            raise FileNotFoundError(f"{runtime} not found. Verify nvidia-cutlass-dsl installation.")
        runtime_obj_dir = root_staging / "runtime_objs"
        runtime_obj_dir.mkdir()
        subprocess.run(["ar", "x", str(runtime)], cwd=str(runtime_obj_dir), check=True)
        runtime_objs = sorted(runtime_obj_dir.glob("*.o"))

        _check_obj_name_collision(kernel_obj_files, runtime_objs)

        # Pack everything into one archive.
        output_dir.mkdir(parents=True, exist_ok=True)
        lib_path = output_dir / f"libcutedsl_{arch}.a"
        subprocess.run(
            ["ar", "rcs", str(lib_path)]
            + [str(o) for o in kernel_obj_files]
            + [str(o) for o in runtime_objs],
            check=True,
        )
        print(f"\n  Created {lib_path.name} ({lib_path.stat().st_size // 1024} KB)")

        # Copy per-variant headers and write umbrella header.
        # Some variants produce multiple .h files (e.g. gdn_decode_mtp + gdn_decode_mtp_cache).
        inc_dir = output_dir / "include"
        inc_dir.mkdir(exist_ok=True)
        all_headers = []
        for v in variants:
            for hdr in sorted(staging_dirs[v.name].glob("*.h")):
                shutil.copy2(hdr, inc_dir)
                all_headers.append(hdr.name)

        # Per-group umbrella headers (e.g. cutedsl_nvfp4_moe_all.h).
        for group in groups_selected:
            group_headers = []
            for v in [vv for vv in variants if vv.group == group]:
                for hdr in sorted(staging_dirs[v.name].glob("*.h")):
                    group_headers.append(hdr.name)
            group_umbrella = inc_dir / f"cutedsl_{group}_all.h"
            group_umbrella.write_text(
                "#pragma once\n"
                f"// Auto-generated by build_cutedsl.py -- do not edit\n"
                + "".join(f'#include "{h}"\n' for h in group_headers)
            )

        # Unified umbrella header (includes everything).
        umbrella = inc_dir / "cutedsl_all.h"
        umbrella.write_text(
            "#pragma once\n"
            "// Auto-generated by build_cutedsl.py -- do not edit\n"
            + "".join(f'#include "{h}"\n' for h in all_headers)
        )

        # Write build provenance + metadata for cmake consumption.
        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "arch": arch,
                    "artifact_tag": artifact_tag,
                    "gpu_arch": f"sm_{sm}",
                    "cuda_version": cuda_ver,
                    "cutlass_dsl_version": dsl_ver,
                    "build_date": datetime.now(timezone.utc).isoformat(),
                    "groups": groups_selected,
                    "variants": [o.stem for o in kernel_obj_files],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        shutil.rmtree(root_staging, ignore_errors=True)

    print(f"\nDone. Artifacts written to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gpu_arch",
        default=None,
        help="Override target GPU SM (e.g. sm_87, sm_100). "
             "Default: auto-detect via cupy / nvidia-smi. "
             "Used only for variant filtering — never forwarded to kernel scripts.",
    )
    p.add_argument(
        "--kernels",
        default="ALL",
        help="Which kernels to build: ALL (default), a group name "
             "(fmha | gdn | nvfp4_moe | nvfp4_fused_moe | ssd | gemm), "
             "or a comma-separated list of group names. "
             "Variants whose supported_sms does not include the target SM are skipped.",
    )
    p.add_argument(
        "--output_dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Root output dir (artifacts go into {{output_dir}}/{{arch}}/sm_<NN>/). "
             f"Default: {_DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument(
        "--arch",
        default=None,
        help="Host/target CPU architecture: x86_64 or aarch64 (default: auto-detected).",
    )
    p.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Parallel compile jobs (use -j 1 if GPU memory is limited). Default: 4.",
    )
    p.add_argument("--verbose", action="store_true", help="Show per-variant kernel script output.")
    p.add_argument("--clean", action="store_true", help="Remove output arch dir before building.")
    build(p.parse_args())


if __name__ == "__main__":
    main()
