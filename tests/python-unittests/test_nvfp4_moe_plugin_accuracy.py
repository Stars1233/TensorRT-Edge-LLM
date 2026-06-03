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
r"""
Accuracy tests for TensorRT ``Nvfp4MoePlugin`` (via new FE :class:`NemotronHMoEMLP`). Compares TRT output to a NumPy
Marlin-unpacked reference; HF dense refs (NumPy / torch) are cross-checks for the FP16-hidden decode path.
The plugin accepts FP16 activations only — any NVFP4 quantization needed by the prefill path is computed inside
the plugin via ``fp4Quantize``. Router logits are explicit engine inputs
(see :func:`serialize_nvfp4_moe_engine_with_explicit_router`), matching the C++ harness style.

**Needs:** CUDA, ``tensorrt`` ≥10.15, Nemotron-H in ``transformers``, ``torch.float8_e4m3fn`` or ``float8_e4m3``, built
``libNvInfer_edgellm_plugin.so`` (``EDGELLM_NVFP4_MOE_PLUGIN_SO`` or default ``build/`` paths).

**Run:** ``pytest tests/python-unittests/test_nvfp4_moe_plugin_accuracy.py -v``. Skips use :func:`check_requirements`
(importorskip / pytest.skip, no module ``pytestmark``). That function performs the same ``transformers`` imports as
``tensorrt_edgellm.quantization.llm_quantization`` before ``tensorrt_edgellm`` is loaded when ``_module_import_guard``
is enabled.

**Verbose stats:** set :data:`PRINT_TRT_VS_TORCH_DIST` and optional
:data:`PRINT_TRT_VS_TORCH_DIST_PRINT_CONFIG`; output via :meth:`NemotronHMoEReference.print_cross_check_output_distribution`
(flush to stderr; use ``pytest -s`` / ``--capture=no`` to see it live).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pytest
import test_attention_utils as attn_utils
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def resolve_edgellm_nvfp4_moe_plugin_so() -> Path:
    """``EDGELLM_NVFP4_MOE_PLUGIN_SO`` if set, else ``build/cpp/`` or ``build/`` ``libNvInfer_edgellm_plugin.so``."""
    env = os.environ.get("EDGELLM_NVFP4_MOE_PLUGIN_SO", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        _REPO_ROOT / "build" / "cpp" / "libNvInfer_edgellm_plugin.so",
        _REPO_ROOT / "build" / "libNvInfer_edgellm_plugin.so",
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return candidates[-1].resolve()


_PLUGIN_SO = resolve_edgellm_nvfp4_moe_plugin_so()

# Populated by :func:`check_requirements` before Nemotron-H / TensorRT test paths run.
NemotronHConfig: type | None = None
NemotronHMoE: type | None = None
trt: object | None = None


def check_requirements(*, _module_import_guard: bool = False) -> None:
    """Skip via ``pytest`` if deps are missing or out of range.

    Verifies ``transformers`` loads using the same imports as
    ``tensorrt_edgellm/quantization/llm_quantization.py`` (entry points that pull hub, tokenizers, and model classes).

    With ``_module_import_guard=True``, only that step runs and ``pytest.skip`` uses ``allow_module_level`` so
    collection skips instead of erroring before ``tensorrt_edgellm`` is imported.

    Otherwise: Nemotron-H, TensorRT 10.15+, plugin ``.so``, CUDA, FP8 dtypes.
    """
    try:
        from transformers import (  # noqa: F401  — mirror llm_quantization (validates HF stack)
            AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer)
    except ImportError as exc:
        pytest.skip(
            f"transformers imports failed (match tensorrt_edgellm.quantization.llm_quantization): {exc}",
            allow_module_level=_module_import_guard,
        )
    try:
        from modelopt.onnx.quantization.qdq_utils import \
            fp4qdq_to_2dq  # noqa: F401
    except ImportError as exc:
        pytest.skip(
            f"modelopt does not meet requirement (fp4qdq_to_2dq): {exc}",
            allow_module_level=_module_import_guard,
        )
    if _module_import_guard:
        return

    global NemotronHConfig, NemotronHMoE, trt
    if NemotronHConfig is not None and NemotronHMoE is not None and trt is not None:
        return
    pytest.importorskip(
        "transformers.models.nemotron_h.modeling_nemotron_h",
        reason="transformers Nemotron-H MoE not available",
    )
    from transformers.models.nemotron_h.configuration_nemotron_h import \
        NemotronHConfig as _cfg
    from transformers.models.nemotron_h.modeling_nemotron_h import \
        NemotronHMoE as _moe

    NemotronHConfig = _cfg
    NemotronHMoE = _moe

    if not (hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3")):
        pytest.skip(
            "NVFP4 Marlin scales need torch.float8_e4m3fn or float8_e4m3")

    trt = pytest.importorskip("tensorrt", reason="tensorrt not installed")
    version_ok, version_msg = attn_utils.check_tensorrt_version(trt, 10, 15)
    if not version_ok:
        pytest.skip(version_msg)
    if not _PLUGIN_SO.is_file():
        pytest.skip(f"missing plugin library: {_PLUGIN_SO}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


check_requirements(_module_import_guard=True)

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tensorrt_edgellm.config import NVFP4_MOE_BACKEND_THOR  # noqa: E402
from tensorrt_edgellm.config import QUANT_NVFP4, ModelConfig, QuantConfig
from tensorrt_edgellm.models.linear import NVFP4Linear  # noqa: E402
from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h import \
    NemotronHMoEMLP  # noqa: E402
from tensorrt_edgellm.models.qwen3_5_moe.modeling_qwen3_5_moe import \
    Qwen3_5SparseMoeBlock  # noqa: E402
from tensorrt_edgellm.models.qwen3_moe.modeling_qwen3_moe import \
    Qwen3SparseMoeBlock  # noqa: E402

# ---------------------------------------------------------------------------
# Standalone helpers previously on old FE classes
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0
_FP4_E2M1_POSITIVE_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                          dtype=np.float32)


def _moe_activation_numpy(z: np.ndarray, activation_type: int) -> np.ndarray:
    """FP32 expert nonlinearity: ``0`` = ReLU², ``1`` = SiLU (clipped to [-50, 50])."""
    z = np.asarray(z, dtype=np.float32)
    if int(activation_type) == 1:
        zc = np.clip(z, -50.0, 50.0)
        return (zc / (1.0 + np.exp(-zc))).astype(np.float32)
    t = np.maximum(z, 0.0)
    return (t * t).astype(np.float32)


def _f32_to_fp4_e2m1_nibble(x: float) -> int:
    """Nearest FP4 E2M1 nibble (sign + 3-bit magnitude)."""
    mag = abs(float(x))
    sign_bit = 8 if x < 0.0 else 0
    best_i = 0
    best_d = mag
    for i, lv in enumerate(_FP4_E2M1_POSITIVE_LEVELS):
        d = abs(mag - lv)
        if d < best_d:
            best_d = d
            best_i = i
    return int((sign_bit | best_i) & 0xF)


def _f16x2_in_u32_to_marlin_f8x4(out_u32: int) -> int:
    """Project one f16x2 uint32 onto Marlin dequant_fp8_scales manifold."""
    o = np.uint32(int(out_u32) & 0xFFFFFFFF)
    u = np.uint32((np.uint64(o) << np.uint64(1)) & np.uint64(0xFFFFFFFF))
    u = u & np.uint32(0xFF00FF00)
    return int(np.uint32(u >> np.uint32(1)))


def _f16x4_in_u32x2_to_marlin_f8x4_block_scale(out1: int, out2: int) -> int:
    """Merge two half2 words into one f8x4 block-scale int32."""
    o1 = np.uint32(int(out1) & 0xFFFFFFFF)
    o2 = np.uint32(int(out2) & 0xFFFFFFFF)
    u1 = np.uint32((np.uint64(o1) << np.uint64(1)) & np.uint64(0xFFFFFFFF))
    u2 = np.uint32((np.uint64(o2) << np.uint64(1)) & np.uint64(0xFFFFFFFF))
    b1 = int((np.uint64(u1) >> 8) & np.uint64(0xFF))
    b3 = int((np.uint64(u1) >> 24) & np.uint64(0xFF))
    b0 = int((np.uint64(u2) >> 8) & np.uint64(0xFF))
    b2 = int((np.uint64(u2) >> 24) & np.uint64(0xFF))
    return int(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))


def _fp32x4_to_marlin_f8x4_block_scale(s0: float, s1: float, s2: float,
                                       s3: float) -> int:
    """Pack four FP32 scales into one f8x4 block-scale int32."""
    h16 = np.array([s0, s1, s2, s3], dtype=np.float32).astype(np.float16)
    out2_raw = int(np.frombuffer(h16[0:2].tobytes(), dtype="<u4", count=1)[0])
    out1_raw = int(np.frombuffer(h16[2:4].tobytes(), dtype="<u4", count=1)[0])
    out1 = _f16x2_in_u32_to_marlin_f8x4(out1_raw)
    out2 = _f16x2_in_u32_to_marlin_f8x4(out2_raw)
    return _f16x4_in_u32x2_to_marlin_f8x4_block_scale(out1, out2)


def _atom_sf_offset(m_idx: int, k_idx: int, num_sf_cols: int) -> int:
    """Byte offset for atom-layout 128x4 swizzle (``get_sf_out_offset_128x4``)."""
    inner_k = k_idx % 4
    inner_m = (m_idx % 128) // 32
    outer_m = m_idx % 32
    k_tile = k_idx // 4
    num_k_tiles = (num_sf_cols + 3) // 4
    m_tile = m_idx // 128
    return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k


def _quantize_f32x64_to_fp4x64_with_f8x4_block_scale(
    vec64: np.ndarray,
    expert_block_scale_max_fp32: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Pack one 64-lane Marlin tile to ``(payload_int32x8, block_scale_packed_int32)``."""
    v = np.asarray(vec64, dtype=np.float32).reshape(64)
    s_max = max(float(expert_block_scale_max_fp32), 1e-12)
    scales = np.zeros(4, dtype=np.float32)
    for g in range(4):
        blk = v[g * 16:(g + 1) * 16]
        am = float(np.max(np.abs(blk)))
        scales[g] = max(am / 6.0, 1e-12)
    nib = np.empty(64, dtype=np.int32)
    for i in range(64):
        g = i // 16
        qn = float(v[i] / scales[g])
        if qn > 6.0:
            qn = 6.0
        elif qn < -6.0:
            qn = -6.0
        nib[i] = _f32_to_fp4_e2m1_nibble(qn)
    out = np.zeros(8, dtype=np.uint32)
    for lane in range(8):
        w = np.uint32(0)
        for j in range(4):
            base = lane * 8 + j * 2
            lo = int(nib[base]) & 0xF
            hi = int(nib[base + 1]) & 0xF
            b = np.uint32(lo | (hi << 4))
            w |= b << (8 * j)
        out[lane] = w
    scale_q = _fp32x4_to_marlin_f8x4_block_scale(
        float(scales[0]) / s_max * _FP8_MAX,
        float(scales[1]) / s_max * _FP8_MAX,
        float(scales[2]) / s_max * _FP8_MAX,
        float(scales[3]) / s_max * _FP8_MAX,
    )
    return out.astype(np.int32), int(scale_q)


# ---------------------------------------------------------------------------
# FP32 → NVFP4 quantization helpers (for populating new-FE NVFP4Linear from test weights)
# ---------------------------------------------------------------------------


def _quantize_fp32_to_nvfp4(
    weight_fp32: torch.Tensor,
    group_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``[out, in]`` FP32 weight to NVFP4 E2M1 packed format.

    Returns ``(weight_int8 [out, in//2], weight_scale_fp32 [out, in//group_size], weight_scale_2 [1])``.
    """
    w = weight_fp32.float()
    out_features, in_features = w.shape
    assert in_features % group_size == 0
    num_groups = in_features // group_size

    # Per-group amax and block scale
    w_grouped = w.reshape(out_features, num_groups, group_size)
    amax = w_grouped.abs().amax(dim=-1)  # [out, num_groups]
    block_scale = (amax / 6.0).clamp(min=1e-12)  # [out, num_groups]

    # Round block_scale to FP8 E4M3 and back to FP32
    dt8 = getattr(torch, "float8_e4m3fn", None)
    if dt8 is not None:
        block_scale_fp8 = block_scale.to(dt8).float()
        block_scale_fp8 = block_scale_fp8.clamp(min=1e-12)
    else:
        block_scale_fp8 = block_scale

    # Quantize each element to nearest FP4 E2M1 nibble
    levels = torch.tensor(_FP4_E2M1_POSITIVE_LEVELS, dtype=torch.float32)
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                             dtype=torch.float32)

    scaled = w_grouped / block_scale_fp8.unsqueeze(-1)
    sign = (scaled < 0).int()
    mag = scaled.abs()
    # Bucket via midpoints
    code = torch.zeros_like(mag, dtype=torch.int32)
    for i, mp in enumerate(midpoints):
        code = torch.where(mag > mp, i + 1, code)
    nibble = (sign * 8) | code  # [out, num_groups, group_size]
    nibble = nibble.reshape(out_features, in_features).to(torch.uint8)

    # Pack two nibbles per byte (lo=even, hi=odd along in-dim)
    lo = nibble[:, 0::2]
    hi = nibble[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.int8)  # [out, in//2]

    weight_scale_2 = torch.ones(1, dtype=torch.float32)
    return packed, block_scale_fp8, weight_scale_2


def _populate_nvfp4_linear(
    linear: NVFP4Linear,
    weight_fp32: torch.Tensor,
    input_scale_value: float = 1e-3,
) -> None:
    """Fill an NVFP4Linear's buffers from FP32 weights."""
    packed, block_scale, ws2 = _quantize_fp32_to_nvfp4(
        weight_fp32, group_size=linear.group_size)
    linear.weight.data.copy_(packed)
    linear.weight_scale.data.copy_(block_scale)
    linear.weight_scale_2.data.copy_(ws2)
    linear.input_scale.data.fill_(input_scale_value)


# ---------------------------------------------------------------------------
# New-FE MoE factory functions (replace old FE NemotronHMoEW4A4Plugin / Qwen3MoEW4A4Plugin)
# ---------------------------------------------------------------------------


def _create_toy_nemotron_h_moe_mlp(
    *,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
    top_k: int,
    seed: int,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[NemotronHMoEMLP, np.ndarray, np.ndarray]:
    """Create a toy NemotronHMoEMLP with NVFP4 experts and random weights.

    Returns ``(mlp, w_up_ehi_fp32, w_down_eih_fp32)`` — the MLP module with
    ``prepare_for_export()`` already called, and the original FP32 expert
    weights ``[E, H, I]`` / ``[E, I, H]`` for building reference outputs.
    """
    h = int(hidden_size)
    inter = int(moe_inter_size)
    e_ct = int(num_experts)
    tk = int(top_k)

    cfg = ModelConfig(
        model_type="nemotron_h",
        hidden_size=h,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=inter,
        head_dim=h,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        n_routed_experts=e_ct,
        num_experts_per_tok=tk,
        moe_intermediate_size=inter,
        moe_shared_expert_intermediate_size=inter,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=float(routed_scaling_factor),
        quant=QuantConfig(quant_type=QUANT_NVFP4, group_size=16),
    )
    mlp = NemotronHMoEMLP(cfg, module_prefix="backbone.layers.0.mixer")
    mlp.eval()

    # Deterministic random weights
    torch.manual_seed(seed)
    std_gate = WeightsGenerator.gate_test_weight_std(h)
    std_corr = std_gate * 0.3
    gen = torch.Generator()
    gen.manual_seed(seed)
    nn.init.normal_(mlp.gate.weight, mean=0.0, std=std_gate, generator=gen)
    nn.init.normal_(mlp.gate.e_score_correction_bias,
                    mean=0.0,
                    std=std_corr,
                    generator=gen)

    gen_w = torch.Generator()
    gen_w.manual_seed(seed + 101)
    std_up = WeightsGenerator.expert_linear_std(h, 0.08)
    std_dn = WeightsGenerator.expert_linear_std(inter, 0.08)
    w_up_ehi = torch.randn(e_ct, h, inter, generator=gen_w) * std_up
    w_down_eih = torch.randn(e_ct, inter, h, generator=gen_w) * std_dn

    # Populate each expert's NVFP4Linear with quantized weights
    for i, expert in enumerate(mlp.experts):
        # up_proj: [inter, h] (NVFP4Linear stores [out, in])
        _populate_nvfp4_linear(expert.up_proj, w_up_ehi[i].T.contiguous())
        # down_proj: [h, inter]
        _populate_nvfp4_linear(expert.down_proj, w_down_eih[i].T.contiguous())

    # Shared experts too (not used in tests but prepare_for_export expects them)
    gen_shared = torch.Generator()
    gen_shared.manual_seed(seed + 202)
    w_shared_up = torch.randn(h, inter, generator=gen_shared) * std_up
    w_shared_dn = torch.randn(inter, h, generator=gen_shared) * std_dn
    _populate_nvfp4_linear(mlp.shared_experts.up_proj,
                           w_shared_up.T.contiguous())
    _populate_nvfp4_linear(mlp.shared_experts.down_proj,
                           w_shared_dn.T.contiguous())

    mlp.prepare_for_export()

    return mlp, w_up_ehi.numpy(), w_down_eih.numpy()


def _create_toy_qwen3_moe_block(
    *,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
    top_k: int,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray]:
    """Create a toy gated-MoE wrapper with NVFP4 plugin buffers for Qwen3 tests.

    Returns ``(mod, w_gate_ehi, w_up_ehi, w_down_eih)`` — an ``nn.Module``
    with the plugin-compatible ``_stacked_*`` buffers, gate, and the scalar
    attributes ``build_nvfp4_moe_engine_trt_api`` reads. Original FP32 weights
    ``[E, H, I]`` / ``[E, I, H]`` are returned for building references.

    **Key difference from NemotronH**: the Nvfp4MoePlugin expects
    ``moe_inter_size = 2*I`` for gated (SiLU) models so that
    ``nOutFor(SiLU, 2*I) = I`` yields the correct FC2 contraction dim.
    FC1 (interleaved gate+up) has N-dim = 2*I, but FC2 (down) has contraction
    dim = I.  NemotronHMoEMLP forces both up and down to use the same
    ``moe_intermediate_size``, so we repack manually.
    """
    from tensorrt_edgellm.checkpoint.repacking import (
        repack_nvfp4_expert_down_prefill_raw,
        repack_nvfp4_expert_up_prefill_raw)

    h = int(hidden_size)
    inter = int(moe_inter_size)
    e_ct = int(num_experts)
    tk = int(top_k)
    plugin_inter = 2 * inter  # moe_inter_size for the plugin (gated: 2*I)

    torch.manual_seed(seed)
    gate_layer = nn.Linear(h, e_ct, bias=False, dtype=torch.float32)
    nn.init.normal_(gate_layer.weight, mean=0.0, std=0.1)

    gen = torch.Generator()
    gen.manual_seed(seed + 201)
    std_w = max(0.18, 3.0 / math.sqrt(float(h)))
    w_gate_ehi = torch.randn(e_ct, h, inter, generator=gen) * std_w
    w_up_ehi = torch.randn(e_ct, h, inter, generator=gen) * std_w
    w_down_eih = torch.randn(e_ct, inter, h, generator=gen) * std_w

    # Quantize each expert's weights to NVFP4, then repack to plugin layout.
    # FC1 SwiGLU: per ``kernelSrcs/nvfp4_moe_cutedsl/README.md`` the CuTeDSL
    # SwiGLU prefill kernel and the patched W4A16 decode GEMV both expect
    # (up, gate) **interleaved at 32-col granularity** along the N axis —
    # NOT plain ``[gate | up]`` halves. Per-chunk layout:
    #   B-N[chunk*64 +  0 : chunk*64 + 32) = up_proj   weights for output [chunk*32, (chunk+1)*32)
    #   B-N[chunk*64 + 32 : chunk*64 + 64) = gate_proj weights for output [chunk*32, (chunk+1)*32)
    # FC2: down [E, I, H] → NVFP4Linear-style [H, I] → N-major [I, H/2]
    _SWIGLU_GS = 32
    assert inter % _SWIGLU_GS == 0, (
        f"moe_inter_size ({inter}) must be a multiple of {_SWIGLU_GS} "
        f"for the SwiGLU prefill kernel's interleave layout")
    _n_chunks = inter // _SWIGLU_GS
    # [E, H, I] → [E, H, n_chunks, 32]
    _up_chunks = w_up_ehi.reshape(e_ct, h, _n_chunks, _SWIGLU_GS)
    _gate_chunks = w_gate_ehi.reshape(e_ct, h, _n_chunks, _SWIGLU_GS)
    # Stack: [E, H, n_chunks, 2, 32]  (axis 3: up=0, gate=1)
    _paired = torch.stack([_up_chunks, _gate_chunks], dim=3)
    w_interleaved = _paired.reshape(e_ct, h, 2 * inter)  # [E, H, 2*I]

    up_ws, up_scs, up_scs_dec, up_gs = [], [], [], []
    dn_ws, dn_scs, dn_scs_dec, dn_gs = [], [], [], []
    for i in range(e_ct):
        # FC1 (up): [2*I, H] as NVFP4Linear, repack with moe_inter_size=2*I
        fc1_packed, fc1_scale, fc1_ws2 = _quantize_fp32_to_nvfp4(
            w_interleaved[i].T.contiguous(), group_size=16)
        up_w, up_sc, up_sc_dec, up_gl = repack_nvfp4_expert_up_prefill_raw(
            fc1_packed, fc1_scale.to(torch.float8_e4m3fn), fc1_ws2, h,
            plugin_inter)
        up_ws.append(torch.as_tensor(up_w))
        up_scs.append(torch.as_tensor(up_sc))
        up_scs_dec.append(torch.as_tensor(up_sc_dec))
        up_gs.append(up_gl)

        # FC2 (down): [H, I] as NVFP4Linear, repack with moe_inter_size=I
        fc2_packed, fc2_scale, fc2_ws2 = _quantize_fp32_to_nvfp4(
            w_down_eih[i].T.contiguous(), group_size=16)
        dn_w, dn_sc, dn_sc_dec, dn_gl = repack_nvfp4_expert_down_prefill_raw(
            fc2_packed, fc2_scale.to(torch.float8_e4m3fn), fc2_ws2, h, inter)
        dn_ws.append(torch.as_tensor(dn_w))
        dn_scs.append(torch.as_tensor(dn_sc))
        dn_scs_dec.append(torch.as_tensor(dn_sc_dec))
        dn_gs.append(dn_gl)

    # Build a simple nn.Module wrapper with the attributes
    # build_nvfp4_moe_engine_trt_api reads.
    mod = nn.Module()
    mod.n_routed_experts = e_ct
    mod.num_experts_per_tok = tk
    mod.hidden_size = h
    mod.moe_intermediate_size = plugin_inter  # 2*I for plugin
    mod.gate = nn.Module()
    mod.gate.weight = nn.Parameter(gate_layer.weight.data.clone().to(
        torch.float16),
                                   requires_grad=False)
    mod.gate.n_group = 1
    mod.gate.topk_group = 1
    mod.gate.norm_topk_prob = True
    mod.gate.routed_scaling_factor = 1.0

    mod.register_buffer("_stacked_up_weights", torch.stack(up_ws))
    mod.register_buffer("_stacked_up_block_scale", torch.stack(up_scs))
    mod.register_buffer("_stacked_up_block_scale_decode",
                        torch.stack(up_scs_dec))
    mod.register_buffer("_stacked_up_global_scale",
                        torch.tensor(up_gs, dtype=torch.float32))
    mod.register_buffer("_stacked_down_weights", torch.stack(dn_ws))
    mod.register_buffer("_stacked_down_block_scale", torch.stack(dn_scs))
    mod.register_buffer("_stacked_down_block_scale_decode",
                        torch.stack(dn_scs_dec))
    mod.register_buffer("_stacked_down_global_scale",
                        torch.tensor(dn_gs, dtype=torch.float32))
    mod.register_buffer("_e_score_correction_bias_fp32",
                        torch.zeros(e_ct, dtype=torch.float32))

    return mod, w_gate_ehi.numpy(), w_up_ehi.numpy(), w_down_eih.numpy()


# If True, stderr cross-check stats for TRT vs refs (see :meth:`NemotronHMoEReference.print_cross_check_output_distribution`).
PRINT_TRT_VS_TORCH_DIST = False


@dataclass(frozen=True)
class CrossCheckPrintConfig:
    """Gates sections of :meth:`NemotronHMoEReference.print_cross_check_output_distribution`."""

    print_header: bool = True
    print_finite_summary: bool = True
    show_hf_dense_histograms: bool = True
    show_hf_dense_pair_norms: bool = True
    show_trt_numpy_marlin: bool = True
    show_torch_hf_fp16_block: bool = True
    show_marlin_torch_block: bool = True
    show_marlin_pair_norms: bool = True
    show_interpret_footer: bool = True
    log_tag: str = "PRINT_TRT_VS_TORCH_DIST"


# Default for ``print_config=None`` in :meth:`NemotronHMoEReference.print_cross_check_output_distribution`.
PRINT_TRT_VS_TORCH_DIST_PRINT_CONFIG: CrossCheckPrintConfig | None = None

# --- Nemotron-H fixtures ---


class WeightsGenerator:
    """Deterministic gate + expert weight init for test MoEs."""

    @staticmethod
    def gate_test_weight_std(fan_in: int) -> float:
        if fan_in <= 0:
            return 0.06
        s = 1.0 / math.sqrt(float(fan_in))
        return float(min(max(s, 0.06), 0.2))

    @staticmethod
    def expert_linear_std(fan_in: int, initializer_range: float) -> float:
        """Std dev for expert Linear: at least ``initializer_range`` and He-style width; floored for small dims."""
        fi = max(int(fan_in), 1)
        he = math.sqrt(2.0 / float(fi))
        base = float(initializer_range)
        floor = 1.0 / float(fi)
        width_min = 0.0
        if fi <= 512:
            width_min = 0.32
        elif fi <= 2048:
            width_min = max(0.18, 3.0 / math.sqrt(float(fi)))
        return float(max(base, he, floor, width_min))

    @staticmethod
    def random_dense_weights_ehi_eih(
        *,
        num_experts: int,
        expert_input_dim: int,
        moe_inter_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
        initializer_range: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random ``w_up_ehi`` ``[E,H,I]``, ``w_down_eih`` ``[E,I,H]`` (plugin layout)."""
        e, h_in, inter = int(num_experts), int(expert_input_dim), int(
            moe_inter_size)
        std_up = WeightsGenerator.expert_linear_std(h_in, initializer_range)
        std_dn = WeightsGenerator.expert_linear_std(inter, initializer_range)
        w_up_ehi = torch.randn(
            e, h_in, inter, device=device, dtype=dtype,
            generator=generator) * std_up
        w_down_eih = torch.randn(
            e, inter, h_in, device=device, dtype=dtype,
            generator=generator) * std_dn
        return w_up_ehi, w_down_eih

    @staticmethod
    def random_fill_nemotron_h_moe_params(
        moe: "NemotronHMoE",
        *,
        seed: int = 44,
        fill_hf_expert_params: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-init gate + experts; returns dense ``(w_up_ehi, w_down_eih)`` for packing."""
        wdev = moe.gate.weight.device
        gen = torch.Generator(
            device=wdev) if wdev.type == "cuda" else torch.Generator()
        gen.manual_seed(seed)
        cfg = moe.config
        std_gate = WeightsGenerator.gate_test_weight_std(int(cfg.hidden_size))
        std_corr = std_gate * 0.3
        nn.init.normal_(moe.gate.weight, mean=0.0, std=std_gate, generator=gen)
        nn.init.normal_(moe.gate.e_score_correction_bias,
                        mean=0.0,
                        std=std_corr,
                        generator=gen)

        gen_w = torch.Generator(
            device=wdev) if wdev.type == "cuda" else torch.Generator()
        gen_w.manual_seed(seed + 101)
        expert_in = int(
            cfg.moe_latent_size) if cfg.moe_latent_size is not None else int(
                cfg.hidden_size)
        w_up_ehi, w_down_eih = WeightsGenerator.random_dense_weights_ehi_eih(
            num_experts=int(moe.n_routed_experts),
            expert_input_dim=expert_in,
            moe_inter_size=int(cfg.moe_intermediate_size),
            device=wdev,
            dtype=torch.float32,
            generator=gen_w,
            initializer_range=float(cfg.initializer_range),
        )
        up_dt = moe.experts.up_proj.dtype
        dn_dt = moe.experts.down_proj.dtype
        if fill_hf_expert_params:
            i_sz = int(cfg.moe_intermediate_size)
            up = moe.experts.up_proj
            dn = moe.experts.down_proj
            up_hw = (int(up.shape[1]), int(up.shape[2]))
            dn_hw = (int(dn.shape[1]), int(dn.shape[2]))
            exp_tr = {(expert_in, i_sz), (i_sz, expert_in)}
            if up_hw not in exp_tr or dn_hw not in exp_tr:
                raise ValueError(
                    f"experts.up_proj / down_proj trailing shapes {up_hw} / {dn_hw} do not match "
                    f"expert_input_dim={expert_in} (moe_latent_size or hidden_size) and "
                    f"moe_intermediate_size={i_sz}")
            w_u = w_up_ehi.to(up_dt)
            w_d = w_down_eih.to(dn_dt)
            if up_hw == (expert_in, i_sz):
                up.data.copy_(w_u)
            else:
                up.data.copy_(w_u.transpose(1, 2))
            if dn_hw == (i_sz, expert_in):
                dn.data.copy_(w_d)
            else:
                dn.data.copy_(w_d.transpose(1, 2))
        else:
            nn.init.zeros_(moe.experts.up_proj)
            nn.init.zeros_(moe.experts.down_proj)
        return w_up_ehi, w_down_eih


def create_nemotron_h_moe_from_config(
    cfg: "NemotronHConfig",
    *,
    layer_idx: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
    eval_mode: bool = True,
    random_fill_weights: bool = False,
    random_fill_seed: int = 44,
    fill_hf_expert_params: bool = True,
) -> "NemotronHMoE":
    """Build ``NemotronHMoE``; optional :meth:`WeightsGenerator.random_fill_nemotron_h_moe_params`."""
    assert NemotronHMoE is not None
    moe = NemotronHMoE(cfg, layer_idx=int(layer_idx))
    if eval_mode:
        moe.eval()
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        moe = moe.to(device=dev, dtype=dtype)
    else:
        moe = moe.to(dtype=dtype)
    if random_fill_weights:
        WeightsGenerator.random_fill_nemotron_h_moe_params(
            moe,
            seed=random_fill_seed,
            fill_hf_expert_params=fill_hf_expert_params,
        )
    return moe


# --- Accuracy scenario dims + activations (see :func:`run_nvfp4_w4a16_moe_plugin_accuracy_case`) ---

_NVFP4_MOE_FAST_BATCH = 2
_NVFP4_MOE_FAST_HIDDEN = 384
_NVFP4_MOE_FAST_INTER = 768
_NVFP4_MOE_FAST_EXPERTS = 8

_NVFP4_MOE_TRT_REF_TOL_SCALE = 1.5

# FP16 prefill path: keep toy magnitudes moderate so intermediates stay finite (no FP32 inf).
_NVFP4_MOE_EXPERT_WEIGHT_SCALE = 0.1

_TOY_HIDDEN_SIN_AMPLITUDE = 1.0
_TOY_HIDDEN_SIN_FREQ = 0.031


def create_toy_moe(
    device: torch.device,
    *,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
    top_k: int,
    seed: int,
    routed_scaling_factor: float = 2.5,
) -> "NemotronHMoE":
    """Tiny routed Nemotron-H MoE for plugin accuracy."""
    assert NemotronHConfig is not None
    inter = int(moe_inter_size)
    cfg = NemotronHConfig(
        vocab_size=128,
        hidden_size=int(hidden_size),
        layers_block_type=["moe"],
        moe_intermediate_size=inter,
        n_routed_experts=int(num_experts),
        num_experts_per_tok=int(top_k),
        mlp_hidden_act="relu2",
        routed_scaling_factor=float(routed_scaling_factor),
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        moe_latent_size=None,
        initializer_range=0.08,
        moe_shared_expert_intermediate_size=inter,
    )
    torch.manual_seed(seed)
    return create_nemotron_h_moe_from_config(
        cfg,
        layer_idx=0,
        device=device,
        dtype=torch.float16,
        eval_mode=True,
        random_fill_weights=True,
        random_fill_seed=seed + 101,
    )


def structured_noise_hidden_states_bsh(
    batch: int,
    seq: int,
    hidden_size: int,
    *,
    seed: int,
    amp: float = 2.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """FP16 ``[B,S,H]`` structured + noise; optional ``device`` move after CPU RNG."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    bh = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
    sj = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    hh = torch.arange(hidden_size, dtype=torch.float32).view(1, 1, hidden_size)
    structural = float(amp) * torch.sin(_TOY_HIDDEN_SIN_FREQ * hh + 0.07 *
                                        (bh + sj))
    noise = torch.randn(
        batch, seq, hidden_size, generator=g,
        dtype=torch.float32) * (float(amp) * 0.12)
    out = (structural + noise).to(torch.float16)
    if device is not None:
        out = out.to(device)
    return out


# --- Nemotron-H MoE reference (NumPy + torch, TRT checks) ---


class NemotronHMoEReference:
    """NumPy + torch refs for ``Nvfp4MoePlugin``: routing, Marlin unpack, dense MoE, TRT tolerance."""

    @staticmethod
    def print_cross_check_output_distribution(
        trt_out: np.ndarray,
        torch_ref_fp32_bsh: np.ndarray,
        numpy_hf_ref_fp32_bsh: np.ndarray,
        case: str,
        *,
        numpy_marlin_unpack_ref_fp32: np.ndarray | None = None,
        torch_marlin_unpack_ref_fp32: np.ndarray | None = None,
        torch_hf_fp16_ref_fp32_bsh: np.ndarray | None = None,
        print_config: CrossCheckPrintConfig | None = None,
        file: TextIO | None = None,
    ) -> None:
        """Debug stats: TRT vs torch HF vs NumPy HF; optional Marlin blocks (see :class:`CrossCheckPrintConfig`)."""
        cfg = (print_config if print_config is not None else
               (PRINT_TRT_VS_TORCH_DIST_PRINT_CONFIG
                or CrossCheckPrintConfig()))
        out = file if file is not None else sys.stderr
        t = np.asarray(trt_out, dtype=np.float32).reshape(-1)
        r = np.asarray(torch_ref_fp32_bsh, dtype=np.float32).reshape(-1)
        n = np.asarray(numpy_hf_ref_fp32_bsh, dtype=np.float32).reshape(-1)
        m_arr = (None if numpy_marlin_unpack_ref_fp32 is None else np.asarray(
            numpy_marlin_unpack_ref_fp32, dtype=np.float32).reshape(-1))
        tm_arr = (None if torch_marlin_unpack_ref_fp32 is None else np.asarray(
            torch_marlin_unpack_ref_fp32, dtype=np.float32).reshape(-1))
        extra = ""
        if m_arr is not None or tm_arr is not None:
            extra = " + Marlin CPU ref"
            if m_arr is not None and tm_arr is not None:
                extra += " (NumPy decode + torch matmul)"
            elif m_arr is not None:
                extra += " (NumPy decode)"
            else:
                extra += " (torch matmul)"
        tag = cfg.log_tag
        if cfg.print_header:
            print(
                f"\n[{tag}] cross-check (TRT / torch / NumPy HF{extra})  case={case!r}\n"
                f"  shape_trt={tuple(np.asarray(trt_out).shape)} shape_torch={tuple(np.asarray(torch_ref_fp32_bsh).shape)} "
                f"shape_numpy_hf={tuple(np.asarray(numpy_hf_ref_fp32_bsh).shape)} nelems={t.size}",
                file=out,
                flush=True,
            )
        if not (t.size == r.size == n.size):
            print(
                f"  ERROR: size mismatch trt={t.size} torch={r.size} numpy_hf={n.size}",
                file=out,
                flush=True,
            )
            return
        if m_arr is not None and m_arr.size != t.size:
            print(f"  ERROR: marlin ref size {m_arr.size} != trt {t.size}",
                  file=out,
                  flush=True)
            return
        if tm_arr is not None and tm_arr.size != t.size:
            print(
                f"  ERROR: torch marlin ref size {tm_arr.size} != trt {t.size}",
                file=out,
                flush=True,
            )
            return

        t64 = t.astype(np.float64)
        r64 = r.astype(np.float64)
        n64 = n.astype(np.float64)
        fin = np.isfinite(t64) & np.isfinite(r64) & np.isfinite(n64)
        if m_arr is not None:
            m64 = m_arr.astype(np.float64)
            fin = fin & np.isfinite(m64)
        else:
            m64 = None
        if tm_arr is not None:
            tm64 = tm_arr.astype(np.float64)
            fin = fin & np.isfinite(tm64)
        else:
            tm64 = None

        n_fin = int(np.sum(fin))
        nf_m = int(np.sum(~np.isfinite(m64))) if m64 is not None else 0
        nf_tm = int(np.sum(~np.isfinite(tm64))) if tm64 is not None else 0
        nway = 3 + (1 if m64 is not None else 0) + (1
                                                    if tm64 is not None else 0)
        way = {3: "triples", 4: "quads", 5: "quints"}.get(nway, f"{nway}-way")
        if cfg.print_finite_summary:
            print(
                f"  finite={way}={n_fin}/{t.size}  nonfinite_trt={int(np.sum(~np.isfinite(t64)))} "
                f"nonfinite_torch={int(np.sum(~np.isfinite(r64)))} nonfinite_numpy_hf={int(np.sum(~np.isfinite(n64)))}"
                +
                (f" nonfinite_numpy_marlin={nf_m}" if m64 is not None else "")
                + (f" nonfinite_torch_marlin={nf_tm}"
                   if tm64 is not None else ""),
                file=out,
                flush=True,
            )
        if n_fin == 0:
            return

        tf = t64[fin]
        rf = r64[fin]
        nf = n64[fin]
        mf = m64[fin] if m64 is not None else None
        tmf = tm64[fin] if tm64 is not None else None

        def _ln(label: str, arr: np.ndarray) -> None:
            print(
                f"  {label}: min={float(np.min(arr)):.6g} max={float(np.max(arr)):.6g} "
                f"mean={float(np.mean(arr)):.6g} std={float(np.std(arr)):.6g} "
                f"p50={float(np.percentile(arr, 50)):.6g} p90={float(np.percentile(arr, 90)):.6g} "
                f"p99={float(np.percentile(arr, 99)):.6g}",
                file=out,
                flush=True,
            )

        def _pair_line(name: str, a: np.ndarray, b: np.ndarray) -> None:
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            nd = float(np.linalg.norm(a - b))
            cos = float(np.dot(a, b) / max(na * nb, 1e-20))
            print(
                f"  {name}: ||a||_2={na:.6g} ||b||_2={nb:.6g} ||a-b||_2={nd:.6g} "
                f"rel_l2_vs_b={nd / max(nb, 1e-20):.6g}  cosine={cos:.9f}",
                file=out,
                flush=True,
            )

        if cfg.show_hf_dense_histograms:
            _ln("trt_out       ", tf)
            _ln("torch_ref     ", rf)
            _ln("numpy_hf_ref  ", nf)
            _ln("|trt-torch|   ", np.abs(tf - rf))
            _ln("|trt-numpy_hf|", np.abs(tf - nf))
            _ln("|numpy_hf-torch|", np.abs(nf - rf))
            denom_t = np.maximum(np.abs(rf), 1e-12)
            _ln("|trt-torch|/|torch|", np.abs(tf - rf) / denom_t)
            denom_n = np.maximum(np.abs(nf), 1e-12)
            _ln("|trt-numpy_hf|/|numpy_hf|", np.abs(tf - nf) / denom_n)
            peak_t = float(np.max(np.abs(tf)))
            peak_r = float(np.max(np.abs(rf)))
            if peak_t < 0.01 and peak_r > 1.0:
                print(
                    "  NOTE: max|TRT| << max|ref| → per-element |trt-ref|/|ref| is ~1 everywhere (relative to ref), "
                    "not a small relative error; use pair norms below and Marlin torch block if enabled.",
                    file=out,
                    flush=True,
                )
        if cfg.show_hf_dense_pair_norms:
            _pair_line("pair trt ↔ torch", tf, rf)
            _pair_line("pair trt ↔ numpy_hf", tf, nf)
            _pair_line("pair numpy_hf ↔ torch", nf, rf)

        if mf is not None and cfg.show_trt_numpy_marlin:
            print(
                "  --- TRT vs NumPy Marlin unpack (plugin primary baseline) ---",
                file=out,
                flush=True,
            )
            _ln("|trt-numpy_marlin|", np.abs(tf - mf))
            _pair_line("pair trt ↔ numpy_marlin", tf, mf)

        if cfg.show_torch_hf_fp16_block and torch_hf_fp16_ref_fp32_bsh is not None:
            h16 = np.asarray(torch_hf_fp16_ref_fp32_bsh,
                             dtype=np.float32).reshape(-1)
            if h16.size == t.size:
                print(
                    "  --- torch HF FP16 expert matmul (diagnostic only; TRT assert uses Marlin unpack, not this ref) ---",
                    file=out,
                    flush=True,
                )
                h64 = h16.astype(np.float64)
                fin_h = np.isfinite(t64) & np.isfinite(h64)
                if int(np.sum(fin_h)) > 0:
                    th = t64[fin_h]
                    hh = h64[fin_h]
                    _ln("torch_hf_fp16_ref (as FP32 view)", hh)
                    _ln("|trt - torch_hf_fp16|", np.abs(th - hh))
                    _pair_line("pair trt ↔ torch_hf_fp16", th, hh)

        if mf is not None:
            if tmf is not None and cfg.show_marlin_torch_block:
                print(
                    "  --- torch Marlin unpack ref (same unpacked weights as NumPy Marlin; torch matmul on CPU FP32) ---",
                    file=out,
                    flush=True,
                )
                _ln("torch_marlin_ref ", tmf)
                _ln("|torch_marlin-numpy_marlin|", np.abs(tmf - mf))
                _ln("|trt-torch_marlin|", np.abs(tf - tmf))
                if cfg.show_marlin_pair_norms:
                    _pair_line("pair numpy_marlin ↔ torch_marlin", mf, tmf)
                    _pair_line("pair trt ↔ torch_marlin", tf, tmf)
            if cfg.show_interpret_footer:
                print(
                    "  Interpret:  TRT≈numpy_marlin & both≪HF/torch → NVFP4 decode+math agrees with CPU ref; gap vs HF is quantization.\n"
                    "              numpy_marlin≈HF but TRT≪numpy_marlin → TRT/plugin kernel or graph (not host pack/unpack).\n"
                    "              numpy_marlin≪HF with sane HF → suspect Marlin tile/FP8 block-scale decode vs packer (incl. non-OCP e4m3).",
                    file=out,
                    flush=True,
                )

    @staticmethod
    def moe_topk_softmax_renormalize_numpy(
        router_logits: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy softmax + top-k + renormalize aligned with CUDA ``moeTopkSoftmax``: iterative argmax with
        masking, **lower expert id wins on equal probability** (same as the fused kernel's warp reduce)."""
        logits = np.asarray(router_logits, dtype=np.float32)
        m = np.max(logits, axis=-1, keepdims=True)
        ex = np.exp(logits - m)
        probs = (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)
        num_tokens, num_experts = probs.shape
        k = min(int(top_k), int(num_experts))
        topw = np.zeros((num_tokens, k), dtype=np.float32)
        topi = np.zeros((num_tokens, k), dtype=np.int32)
        for t in range(int(num_tokens)):
            row = probs[t]
            used = np.zeros(num_experts, dtype=bool)
            for ki in range(k):
                best_e = -1
                best_p = np.float32(-1.0)
                for e in range(int(num_experts)):
                    if used[e]:
                        continue
                    p = row[e]
                    if best_e < 0 or p > best_p or (p == best_p
                                                    and e < best_e):
                        best_p = p
                        best_e = e
                used[best_e] = True
                topi[t, ki] = np.int32(best_e)
                topw[t, ki] = best_p
            denom = float(np.sum(topw[t])) + 1e-20
            topw[t] = (topw[t] / denom).astype(np.float32)
        return topw, topi

    @staticmethod
    def sort_topk_slots_descending(
        expert: np.ndarray,
        score: np.ndarray,
        *,
        num_tokens: int,
        top_k: int,
    ) -> None:
        """In-place bubble sort per token: descending score; ties → lower expert id first (matches CUDA kernel)."""
        e = np.asarray(expert, dtype=np.int32).reshape(num_tokens, top_k)
        s = np.asarray(score, dtype=np.float32).reshape(num_tokens, top_k)
        for t in range(num_tokens):
            row_e = e[t]
            row_s = s[t]
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    wi = float(row_s[i])
                    wj = float(row_s[j])
                    ei = int(row_e[i])
                    ej = int(row_e[j])
                    j_better = (wj > wi) or (wj == wi and ej < ei)
                    if j_better:
                        row_s[i], row_s[j] = row_s[j], row_s[i]
                        row_e[i], row_e[j] = row_e[j], row_e[i]

    @staticmethod
    def topk_weights_indices_from_sorted_desired_slots(
        expert: np.ndarray,
        score: np.ndarray,
        *,
        num_tokens: int,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Renormalized ``(topw, topi)`` from sorted slots (after :meth:`router_logits_from_desired_topk`)."""
        e = np.asarray(expert, dtype=np.int32).reshape(num_tokens, top_k)
        s = np.asarray(score, dtype=np.float32).reshape(num_tokens, top_k)
        denom = np.sum(s, axis=-1, keepdims=True) + np.float32(1e-20)
        topw = (s / denom).astype(np.float32)
        return topw, e

    @staticmethod
    def router_logits_from_desired_topk(
        expert: np.ndarray,
        score: np.ndarray,
        *,
        num_tokens: int,
        num_experts: int,
        top_k: int,
    ) -> np.ndarray:
        """FP32 logits matching C++ ``fillRouterLogitsAlignedWithTopkSoftmax``."""
        NemotronHMoEReference.sort_topk_slots_descending(expert,
                                                         score,
                                                         num_tokens=num_tokens,
                                                         top_k=top_k)
        e = np.asarray(expert, dtype=np.int32).reshape(num_tokens, top_k)
        s = np.asarray(score, dtype=np.float32).reshape(num_tokens, top_k)
        logits = np.full((num_tokens, num_experts), -120.0, dtype=np.float32)
        for t in range(num_tokens):
            for k in range(top_k):
                ex = int(e[t, k])
                w = float(s[t, k])
                logits[t, ex] = np.float32(np.log(max(w, 1e-12)) + 25.0)
        return logits

    @staticmethod
    def trt_output_elem_acceptable(got: float, ref_fp32: float) -> bool:
        """Match ``trtNvfp4MoeOutputElemAcceptable`` in ``experiment/test_nvfp4_moe_trt_plugin_accuracy.cu``."""
        if not math.isfinite(got) or not math.isfinite(ref_fp32):
            return False

        ref_h = float(np.float16(ref_fp32))
        err = min(abs(got - ref_fp32), abs(got - ref_h))
        scale = max(1.0, abs(ref_fp32))
        tol = max(1.4, 0.25 + 0.6 * scale)
        return err <= tol

    @staticmethod
    def assert_non_degenerate_output_magnitudes(
        trt_out: np.ndarray,
        ref_fp32: np.ndarray,
        case: str,
        *,
        min_peak_abs: float = 1e-4,
        min_p90_abs: float = 5e-6,
    ) -> None:
        """Assert TRT and ref outputs have non-trivial peak and p90 |x|."""
        t = np.asarray(trt_out, dtype=np.float32).reshape(-1)
        r = np.asarray(ref_fp32, dtype=np.float32).reshape(-1)
        assert t.size == r.size, f"{case}: magnitude check shape mismatch"
        assert np.all(
            np.isfinite(t)), f"{case}: TRT output has non-finite values"
        assert np.all(
            np.isfinite(r)), f"{case}: reference has non-finite values"
        at = np.abs(t.astype(np.float64))
        ar = np.abs(r.astype(np.float64))
        peak_t = float(np.max(at))
        peak_r = float(np.max(ar))
        p90_t = float(np.percentile(at, 90))
        p90_r = float(np.percentile(ar, 90))
        assert peak_t > min_peak_abs, (
            f"{case}: TRT max|out|={peak_t:.6g} <= {min_peak_abs} (expected non-trivial peak magnitude)"
        )
        assert peak_r > min_peak_abs, (
            f"{case}: ref max|out|={peak_r:.6g} <= {min_peak_abs} (expected non-trivial peak magnitude)"
        )
        assert p90_t > min_p90_abs, (
            f"{case}: TRT p90|out|={p90_t:.6g} <= {min_p90_abs} (bulk of elements near zero)"
        )
        assert p90_r > min_p90_abs, (
            f"{case}: ref p90|out|={p90_r:.6g} <= {min_p90_abs} (bulk of elements near zero)"
        )

    @staticmethod
    def assert_trt_matches_reference(
        trt_out: np.ndarray,
        ref_fp32: np.ndarray,
        case: str,
        *,
        tol_scale: float = 1.0,
        use_fp16_rounded_baseline: bool = True,
        max_outlier_frac: float = 0.0,
    ) -> None:
        """Per-element tolerance matching C++ ``trtNvfp4MoeOutputElemAcceptable`` (vectorized)."""
        t = np.asarray(trt_out, dtype=np.float32).reshape(-1)
        r = np.asarray(ref_fp32, dtype=np.float32).reshape(-1)
        assert t.size == r.size, f"{case}: shape mismatch"

        # Vectorized equivalent of :meth:`trt_output_elem_acceptable` (for reporting + pass/fail).
        # Use float64 for err/tol so results match the Python scalar reference (float32 intermediates can differ).
        t64 = t.astype(np.float64)
        r64 = r.astype(np.float64)
        if use_fp16_rounded_baseline:
            # Match :meth:`trt_output_elem_acceptable` / scalar path: min(|trt-fp32_ref|, |trt-fp16_roundtrip_ref|).
            ref_h = r.astype(np.float16).astype(np.float32)
            ref_h64 = ref_h.astype(np.float64)
            err_min = np.minimum(np.abs(t64 - r64), np.abs(t64 - ref_h64))
        else:
            ref_h = r
            err_min = np.abs(t64 - r64)
        raw_abs = np.abs(t - r)
        scale = np.maximum(1.0, np.abs(r64))
        ts = float(tol_scale)
        tol = np.maximum(1.4, 0.25 + 0.6 * scale) * ts
        finite = np.isfinite(t64) & np.isfinite(r64)
        ok = finite & (err_min <= tol)
        n_bad = int(np.sum(~ok))

        if n_bad == 0:
            return

        # Allow a small fraction of outliers (FP4 quantization noise).
        if max_outlier_frac > 0.0 and t.size > 0:
            bad_frac = n_bad / t.size
            if bad_frac <= max_outlier_frac:
                print(
                    f"{case}: {n_bad}/{t.size} elements ({bad_frac*100:.4f}%) exceed tolerance "
                    f"(within allowed {max_outlier_frac*100:.4f}% outlier budget, PASS)."
                )
                return

        max_abs = float(np.max(raw_abs)) if t.size else 0.0
        n_nonfinite = int(np.sum(~finite))
        ids = np.flatnonzero(~ok)
        err_bad = err_min[ids]
        tol_bad = tol[ids]
        raw_bad = raw_abs[ids]
        excess = err_bad - tol_bad
        order = np.argsort(-excess)
        topk = min(16, len(ids))

        lines: list[str] = [
            f"{case}: {n_bad}/{t.size} elements fail TRT vs ref "
            f"(experiment/test_nvfp4_moe_trt_plugin_accuracy.cu / trtNvfp4MoeOutputElemAcceptable, tol_scale={ts}"
            f"{'' if use_fp16_rounded_baseline else ', FP32 ref only (no FP16-rounded baseline)'}).",
            f"  max_raw_abs(trt - ref)={max_abs}",
        ]
        if n_nonfinite:
            lines.append(f"  non-finite (trt or ref): {n_nonfinite}")
        if use_fp16_rounded_baseline:
            closer_fp32 = np.abs(t64[ids] - r64[ids]) <= np.abs(t64[ids] -
                                                                ref_h64[ids])
            n_closer_fp32 = int(np.sum(closer_fp32))
            lines.extend([
                "  failing elements: err_min=min(|trt-fp32_ref|, |trt-fp16rounded_ref|) vs tol=max(1.4, 0.25+0.6*|ref|).",
                f"  err_min percentiles (bad only): p50={float(np.percentile(err_bad, 50)):.6g} "
                f"p90={float(np.percentile(err_bad, 90)):.6g} max={float(np.max(err_bad)):.6g}",
                f"  tol percentiles (bad only): p50={float(np.percentile(tol_bad, 50)):.6g} "
                f"p90={float(np.percentile(tol_bad, 90)):.6g}",
                f"  excess=err_min-tol (bad only): max={float(np.max(excess)):.6g} "
                f"p90={float(np.percentile(excess, 90)):.6g}",
                f"  bad elems closer to fp32 ref than fp16-rounded ref: {n_closer_fp32}/{len(ids)} "
                f"(remainder dominated by |trt-fp16_ref| arm)",
                "  worst indices (idx: trt, fp32_ref, fp16rt_ref, err_min, tol, excess, raw_abs):",
            ])
            for rank in range(topk):
                j = int(order[rank])
                i = int(ids[j])
                lines.append(
                    f"    {i}: trt={float(t[i]):.8g} fp32_ref={float(r[i]):.8g} fp16rt_ref={float(ref_h[i]):.8g} "
                    f"err_min={float(err_bad[j]):.6g} tol={float(tol_bad[j]):.6g} "
                    f"excess={float(excess[j]):.6g} raw_abs={float(raw_bad[j]):.6g}"
                )
        else:
            lines.extend([
                "  failing elements: err=|trt-ref| vs tol=max(1.4, 0.25+0.6*|ref|).",
                f"  err percentiles (bad only): p50={float(np.percentile(err_bad, 50)):.6g} "
                f"p90={float(np.percentile(err_bad, 90)):.6g} max={float(np.max(err_bad)):.6g}",
                f"  tol percentiles (bad only): p50={float(np.percentile(tol_bad, 50)):.6g} "
                f"p90={float(np.percentile(tol_bad, 90)):.6g}",
                f"  excess=err-tol (bad only): max={float(np.max(excess)):.6g} "
                f"p90={float(np.percentile(excess, 90)):.6g}",
                "  worst indices (idx: trt, ref, err, tol, excess, raw_abs):",
            ])
            for rank in range(topk):
                j = int(order[rank])
                i = int(ids[j])
                lines.append(
                    f"    {i}: trt={float(t[i]):.8g} ref={float(r[i]):.8g} "
                    f"err={float(err_bad[j]):.6g} tol={float(tol_bad[j]):.6g} "
                    f"excess={float(excess[j]):.6g} raw_abs={float(raw_bad[j]):.6g}"
                )

        msg = "\n".join(lines)
        print(msg, file=sys.stderr)
        raise AssertionError(msg)

    @staticmethod
    def assert_trt_output_matches_nvfp4_marlin_numpy_reference(
        trt_out: np.ndarray,
        numpy_marlin_dense_ref_fp32: np.ndarray,
        case: str,
        *,
        min_peak_abs: float = 1e-4,
        min_p90_abs: float = 5e-6,
        tol_scale: float = 1.0,
    ) -> None:
        """TRT vs NumPy Marlin-unpacked ref (:meth:`reference_from_packed_plugin_state`).

        Plugin tensors are FP16; per-element error uses ``min(|trt-fp32_ref|, |trt-fp16_ref|)``.
        ``tol_scale`` multiplies the C++ ``trtNvfp4MoeOutputElemAcceptable`` tolerance (see
        ``experiment/test_nvfp4_moe_trt_plugin_accuracy.cu``).
        """
        t = np.asarray(trt_out)
        r = np.asarray(numpy_marlin_dense_ref_fp32, dtype=np.float32)
        if r.shape != t.shape:
            if r.size != t.size:
                raise AssertionError(
                    f"{case}: TRT shape {tuple(t.shape)} vs numpy ref shape {tuple(r.shape)} (size mismatch)"
                )
            r = r.reshape(t.shape)
        NemotronHMoEReference.assert_non_degenerate_output_magnitudes(
            trt_out,
            r,
            case,
            min_peak_abs=min_peak_abs,
            min_p90_abs=min_p90_abs)
        NemotronHMoEReference.assert_trt_matches_reference(
            trt_out,
            r,
            case,
            use_fp16_rounded_baseline=True,
            tol_scale=tol_scale,
        )

    # Vectorized FP4 nibble → float LUT (indexed by 4-bit value 0..15).
    _FP4_LUT = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,  # positive (sign bit 0)
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,  # negative (sign bit 1)
        ],
        dtype=np.float32)

    @staticmethod
    def _fp4_nibble_to_float(nib: int) -> float:
        nib = int(nib) & 0xF
        mag_i = nib & 7
        sign = -1.0 if (nib & 8) else 1.0
        levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        return sign * levels[mag_i]

    @staticmethod
    def _unpack_fp8_e4m3_int32_word_to_float4(word: int) -> np.ndarray:
        """Four FP32 scales from one Marlin block-scale int32 (CUDA/host decode, not torch float8 cast)."""
        q = int(word) & 0xFFFFFFFF
        out1 = (q & 0xFF00FF00) >> 1
        qs = (q << 8) & 0xFFFFFFFF
        out2 = (qs & 0xFF00FF00) >> 1
        u1 = np.uint32(out1)
        u2 = np.uint32(out2)
        f1 = np.frombuffer(u1.tobytes(), dtype=np.float16).astype(np.float32)
        f2 = np.frombuffer(u2.tobytes(), dtype=np.float16).astype(np.float32)
        return np.array([f2[0], f2[1], f1[0], f1[1]], dtype=np.float32)

    @staticmethod
    def unpack_nvfp4_marlin_tile_64(
        payload_i32: np.ndarray,
        block_scale_i32: int,
    ) -> np.ndarray:
        """One 64-lane Marlin tile → ``float32[64]`` (8 int32 payload + block scales)."""
        pl = np.asarray(payload_i32, dtype=np.int32).reshape(8)
        scales4 = NemotronHMoEReference._unpack_fp8_e4m3_int32_word_to_float4(
            int(block_scale_i32))
        nib = np.empty(64, dtype=np.int32)
        for lane in range(8):
            w = int(pl[lane]) & 0xFFFFFFFF
            for i in range(4):
                b = (w >> (8 * i)) & 0xFF
                lo = int(b & 0xF)
                hi = int((b >> 4) & 0xF)
                base = lane * 8 + i * 2
                nib[base] = lo
                nib[base + 1] = hi
        out = np.empty(64, dtype=np.float32)
        for i in range(64):
            g = i // 16
            out[i] = NemotronHMoEReference._fp4_nibble_to_float(int(
                nib[i])) * float(scales4[g])
        return out

    @staticmethod
    def _read_atom_scale_word(buf_u8: np.ndarray, m_idx: int, k_tile: int,
                              num_sf_cols: int) -> int:
        """Read 4 raw FP8 bytes from atom-layout positions and return Marlin-packed int32 scale word."""
        raw = np.zeros(4, dtype=np.uint8)
        for g in range(4):
            off = _atom_sf_offset(m_idx, k_tile * 4 + g, num_sf_cols)
            raw[g] = buf_u8[off]
        # Atom stores {s0,s1,s2,s3}; Marlin expects {s0,s2,s1,s3}
        marlin = np.array([raw[0], raw[2], raw[1], raw[3]], dtype=np.uint8)
        return int(np.frombuffer(marlin.tobytes(), dtype=np.int32)[0])

    @staticmethod
    def _unpack_payload_to_nibbles_vectorized(payload_flat: np.ndarray,
                                              num_tiles: int) -> np.ndarray:
        """Vectorized: all tiles' int8 payload → uint8 nibble array [num_tiles, 64]."""
        # payload_flat is int8 with shape [num_tiles * 32]
        raw = payload_flat.view(np.uint8).reshape(num_tiles, 32)
        lo = (raw & 0x0F).astype(np.uint8)
        hi = ((raw >> 4) & 0x0F).astype(np.uint8)
        # Interleave lo/hi: for each byte → (lo, hi) in output
        nibs = np.empty((num_tiles, 64), dtype=np.uint8)
        nibs[:, 0::2] = lo
        nibs[:, 1::2] = hi
        return nibs

    @staticmethod
    def _unpack_scales_vectorized(bs_u8: np.ndarray, m_indices: np.ndarray,
                                  k_tiles: np.ndarray,
                                  num_sf_cols: int) -> np.ndarray:
        """Vectorized atom_sf_offset + FP8→FP16→FP32 scale decode for all tiles. Returns [num_tiles, 4]."""
        num_tiles = len(m_indices)
        # Compute atom_sf_offset for all 4 groups per tile
        # k_idx = k_tile * 4 + g, for g in [0,1,2,3]
        k_base = (k_tiles * 4).astype(np.int64)  # [num_tiles]
        g_offsets = np.arange(4, dtype=np.int64)  # [4]
        k_idx = k_base[:, None] + g_offsets[None, :]  # [num_tiles, 4]
        m_idx = m_indices[:, None].astype(np.int64)  # [num_tiles, 1] broadcast

        # atom_sf_offset vectorized
        inner_k = (k_idx % 4).astype(np.int64)
        inner_m = ((m_idx % 128) // 32).astype(np.int64)
        outer_m = (m_idx % 32).astype(np.int64)
        k_tile_div = (k_idx // 4).astype(np.int64)
        num_k_tiles = int((num_sf_cols + 3) // 4)
        m_tile = (m_idx // 128).astype(np.int64)
        offsets = m_tile * num_k_tiles * 512 + k_tile_div * 512 + outer_m * 16 + inner_m * 4 + inner_k
        # offsets: [num_tiles, 4]

        # Gather raw FP8 bytes
        raw = bs_u8[offsets.ravel()].reshape(num_tiles,
                                             4)  # [num_tiles, 4] uint8
        # Atom stores {s0,s1,s2,s3}; Marlin expects {s0,s2,s1,s3}
        marlin_bytes = np.stack([raw[:, 0], raw[:, 2], raw[:, 1], raw[:, 3]],
                                axis=1)  # [num_tiles, 4]

        # Decode marlin int32 → 4 FP32 scales per tile
        words = marlin_bytes.view(np.uint32).reshape(
            num_tiles)  # [num_tiles] uint32
        words64 = words.astype(np.uint64)
        out1 = ((words64 & 0xFF00FF00) >> 1).astype(np.uint32)
        qs = ((words64 << 8) & 0xFFFFFFFF).astype(np.uint64)
        out2 = ((qs & 0xFF00FF00) >> 1).astype(np.uint32)
        # Each uint32 → 2 float16 → 2 float32
        f1 = out1.view(np.uint16).reshape(num_tiles, 2).view(
            np.float16).astype(np.float32)
        f2 = out2.view(np.uint16).reshape(num_tiles, 2).view(
            np.float16).astype(np.float32)
        # Order: [f2[0], f2[1], f1[0], f1[1]] per tile (matches _unpack_fp8_e4m3_int32_word_to_float4)
        scales = np.empty((num_tiles, 4), dtype=np.float32)
        scales[:, 0] = f2[:, 0]
        scales[:, 1] = f2[:, 1]
        scales[:, 2] = f1[:, 0]
        scales[:, 3] = f1[:, 1]
        return scales

    @staticmethod
    def dense_weights_from_nvfp4_plugin_buffers(
        fc_up_qweights: np.ndarray,
        fc_up_blocks_scale: np.ndarray,
        fc_down_qweights: np.ndarray,
        fc_down_blocks_scale: np.ndarray,
        *,
        num_experts: int,
        hidden_size: int,
        moe_inter_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """INT8 plugin buffers → dense ``w_up_ehi`` / ``w_down_eih`` (no global scales; kernel applies those).

        Vectorized: processes all tiles per expert in bulk using numpy ops instead of per-tile Python loops.
        """
        e = int(num_experts)
        h = int(hidden_size)
        inter = int(moe_inter_size)
        nic = inter // 64
        n_h_chunks = h // 64
        assert nic * 64 == inter
        assert n_h_chunks * 64 == h

        up_q8 = np.ascontiguousarray(fc_up_qweights, dtype=np.int8)
        up_bs8 = np.ascontiguousarray(fc_up_blocks_scale, dtype=np.int8)
        dn_q8 = np.ascontiguousarray(fc_down_qweights, dtype=np.int8)
        dn_bs8 = np.ascontiguousarray(fc_down_blocks_scale, dtype=np.int8)
        assert up_q8.shape == (e, h // 2, inter)
        assert up_bs8.shape == (e, h // 16, inter)
        assert dn_q8.shape == (e, inter, h // 2)
        assert dn_bs8.shape == (e, inter, h // 16)

        num_sf_cols_up = inter // 16
        num_sf_cols_dn = h // 16
        lut = NemotronHMoEReference._FP4_LUT

        # Pre-compute tile indices for up-proj: tile[jj, c] = jj * nic + c
        # m_indices = jj for each tile, k_tiles = c for each tile
        up_num_tiles = h * nic
        up_m_idx = np.repeat(np.arange(h, dtype=np.int64), nic)
        up_k_tiles = np.tile(np.arange(nic, dtype=np.int64), h)

        dn_num_tiles = inter * n_h_chunks
        dn_m_idx = np.repeat(np.arange(inter, dtype=np.int64), n_h_chunks)
        dn_k_tiles = np.tile(np.arange(n_h_chunks, dtype=np.int64), inter)

        w_up = np.zeros((e, h, inter), dtype=np.float32)
        w_down = np.zeros((e, inter, h), dtype=np.float32)

        for ex in range(e):
            # --- Up projection ---
            up_flat = up_q8[ex].reshape(-1)  # [h//2 * inter] int8
            up_bs_u8 = up_bs8[ex].view(np.uint8).reshape(-1)

            # Unpack all payload nibbles: [up_num_tiles, 64]
            nibs = NemotronHMoEReference._unpack_payload_to_nibbles_vectorized(
                up_flat, up_num_tiles)
            # Decode nibbles → float via LUT: [up_num_tiles, 64]
            vals = lut[nibs]
            # Decode all scales: [up_num_tiles, 4]
            scales = NemotronHMoEReference._unpack_scales_vectorized(
                up_bs_u8, up_m_idx, up_k_tiles, num_sf_cols_up)
            # Broadcast scales to 64 lanes (each scale covers 16 lanes)
            scales_64 = np.repeat(scales, 16, axis=1)  # [up_num_tiles, 64]
            vals *= scales_64
            # Write to output: tile[jj, c] → w_up[ex, jj, c*64:(c+1)*64]
            w_up[ex] = vals.reshape(h, nic, 64).reshape(h, inter)

            # --- Down projection ---
            dn_flat = dn_q8[ex].reshape(-1)
            dn_bs_u8 = dn_bs8[ex].view(np.uint8).reshape(-1)

            nibs_dn = NemotronHMoEReference._unpack_payload_to_nibbles_vectorized(
                dn_flat, dn_num_tiles)
            vals_dn = lut[nibs_dn]
            scales_dn = NemotronHMoEReference._unpack_scales_vectorized(
                dn_bs_u8, dn_m_idx, dn_k_tiles, num_sf_cols_dn)
            scales_64_dn = np.repeat(scales_dn, 16, axis=1)
            vals_dn *= scales_64_dn
            w_down[ex] = vals_dn.reshape(inter, n_h_chunks,
                                         64).reshape(inter, h)

        return w_up, w_down

    @staticmethod
    def dense_weights_from_nvfp4_nmajor_buffers(
        fc_up_qweights: np.ndarray,
        fc_up_block_scale: np.ndarray,
        fc_down_qweights: np.ndarray,
        fc_down_block_scale: np.ndarray,
        *,
        num_experts: int,
        hidden_size: int,
        moe_inter_size: int,
        activation_type: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """N-major NVFP4 buffers → dense weights (no global scales applied).

        N-major payload: ``[K, N//2]`` int8 (2 nibbles packed per byte along N);
        prefill atom-layout SF: ``[padUp(N,128), padUp(K/16,4)]`` raw FP8 E4M3
        (M=N output dim, K_sf=K/16 input groups).
        Scale granularity: one per 16 K-elements × 16 N-elements.

        Returns ``w_up [E, H, moe_inter_size]`` and ``w_down [E, nOut, H]`` float32.
        """
        import torch as _torch

        def _pad_up(x, align):
            return ((x + align - 1) // align) * align

        def _atom_sf_byte_offset(m_idx, k_idx, num_sf_cols):
            inner_k = k_idx % 4
            inner_m = (m_idx % 128) // 32
            outer_m = m_idx % 32
            k_tile = k_idx // 4
            num_k_tiles = (num_sf_cols + 3) // 4
            m_tile = m_idx // 128
            return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k

        def _read_prefill_sf(sf_buf, k_gemv, n_gemv):
            """Read prefill atom-layout FP8 E4M3 scales → [K/16, N/16] float32.

            Prefill atom: M=N (output dim), K_sf=K/16 (input groups).
            """
            k_groups = k_gemv // 16
            n_blocks = n_gemv // 16
            padded_sf_cols = _pad_up(k_groups, 4)
            sf_flat = sf_buf.view(np.uint8).ravel()
            raw = np.empty((k_groups, n_blocks), dtype=np.uint8)
            for g in range(k_groups):
                for c in range(n_blocks):
                    off = _atom_sf_byte_offset(c * 16, g, padded_sf_cols)
                    raw[g, c] = sf_flat[off]
            # FP8 E4M3 → float32
            raw_f32 = _torch.from_numpy(raw.ravel()).view(
                _torch.float8_e4m3fn).float().numpy().reshape(
                    k_groups, n_blocks)
            return raw_f32

        e = int(num_experts)
        h = int(hidden_size)
        inter = int(moe_inter_size)
        nOut = inter // 2 if int(activation_type) == 1 else inter

        up_q = np.ascontiguousarray(fc_up_qweights, dtype=np.int8)
        dn_q = np.ascontiguousarray(fc_down_qweights, dtype=np.int8)

        assert up_q.shape == (e, h, inter // 2), (
            f"up payload {up_q.shape} != ({e}, {h}, {inter // 2})")
        assert dn_q.shape == (e, nOut, h // 2), (
            f"dn payload {dn_q.shape} != ({e}, {nOut}, {h // 2})")

        lut = NemotronHMoEReference._FP4_LUT

        w_up = np.zeros((e, h, inter), dtype=np.float32)
        w_down = np.zeros((e, nOut, h), dtype=np.float32)

        for ex in range(e):
            # --- FC1 up: N-major payload [H, inter/2] → [H, inter] ---
            up_u8 = up_q[ex].view(np.uint8)  # [H, inter/2]
            nibs_up = np.empty((h, inter), dtype=np.uint8)
            nibs_up[:, 0::2] = up_u8 & 0x0F
            nibs_up[:, 1::2] = (up_u8 >> 4) & 0x0F
            vals_up = lut[nibs_up]  # [H, inter] float32

            # Prefill atom-layout FP8 E4M3: [K_groups, N_blocks] float32
            sf_up = _read_prefill_sf(fc_up_block_scale[ex], h, inter)
            # Expand: [H/16, inter/16] → [H, inter] (16 K × 16 N share one scale)
            sf_expanded = np.repeat(np.repeat(sf_up, 16, axis=0), 16, axis=1)
            vals_up *= sf_expanded[:h, :inter]
            w_up[ex] = vals_up

            # --- FC2 down: N-major payload [nOut, H/2] → [nOut, H] ---
            dn_u8 = dn_q[ex].view(np.uint8)  # [nOut, H/2]
            nibs_dn = np.empty((nOut, h), dtype=np.uint8)
            nibs_dn[:, 0::2] = dn_u8 & 0x0F
            nibs_dn[:, 1::2] = (dn_u8 >> 4) & 0x0F
            vals_dn = lut[nibs_dn]

            sf_dn = _read_prefill_sf(fc_down_block_scale[ex], nOut, h)
            sf_dn_expanded = np.repeat(np.repeat(sf_dn, 16, axis=0),
                                       16,
                                       axis=1)
            vals_dn *= sf_dn_expanded[:nOut, :h]
            w_down[ex] = vals_dn

        return w_up, w_down

    @staticmethod
    def dense_forward_from_topk(
        x_bh: np.ndarray,
        topk_weights: np.ndarray,
        topk_indices: np.ndarray,
        w_up_ehi: np.ndarray,
        w_down_eih: np.ndarray,
        *,
        activation_type: int = 0,
    ) -> np.ndarray:
        """Dense MoE: weighted ``down(act(up @ x))`` per slot; ``activation_type`` 0=ReLU², 1=SiLU; out ``[B,1,H]``."""
        x_bh = np.asarray(x_bh, dtype=np.float32)
        topk_weights = np.asarray(topk_weights, dtype=np.float32)
        topk_indices = np.asarray(topk_indices, dtype=np.int32)
        w_up_ehi = np.asarray(w_up_ehi, dtype=np.float32)
        w_down_eih = np.asarray(w_down_eih, dtype=np.float32)

        b, h = x_bh.shape
        e, h2, inter = w_up_ehi.shape
        assert h2 == h
        assert w_down_eih.shape == (e, inter, h)
        bk, k = topk_weights.shape
        assert bk == b
        assert topk_indices.shape == (b, k)

        out = np.zeros((b, h), dtype=np.float32)
        for bb in range(b):
            for slot in range(k):
                ex = int(topk_indices[bb, slot])
                s = float(topk_weights[bb, slot])
                if s == 0.0 or ex < 0 or ex >= e:
                    continue
                z = x_bh[bb] @ w_up_ehi[ex]
                act = _moe_activation_numpy(z, activation_type)
                t = act * s
                out[bb] += t @ w_down_eih[ex]
        return out.reshape(b, 1, h)

    @staticmethod
    def _vectorized_atom_sf_offsets(num_sf_rows: int,
                                    num_sf_cols: int) -> np.ndarray:
        """Vectorized ``_atom_sf_offset`` over a full grid.

        Returns an ``[num_sf_rows, num_sf_cols]`` array of byte offsets into
        the atom-swizzled SF buffer.
        """
        m = np.arange(int(num_sf_rows), dtype=np.int64).reshape(-1, 1)
        k = np.arange(int(num_sf_cols), dtype=np.int64).reshape(1, -1)
        inner_k = k % 4
        inner_m = (m % 128) // 32
        outer_m = m % 32
        k_tile = k // 4
        num_k_tiles = (int(num_sf_cols) + 3) // 4
        m_tile = m // 128
        return (m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 +
                inner_m * 4 + inner_k)

    @staticmethod
    def dequant_fp4_prefill_weight_fc1(
        fc_up_qweights: np.ndarray,
        fc_up_blocks_scale: np.ndarray,
        fc_up_global_scale: np.ndarray,
        *,
        num_experts: int,
        hidden_size: int,
        moe_inter_size: int,
    ) -> np.ndarray:
        """Dequant packed v6 FC1 weights → dense ``[E, H, I]`` FP32.

        Bit-equivalent to what the CuteDSL prefill kernel reconstructs via its
        ``__nv_fp8_e4m3`` SF cast + block-scale multiply + per-expert global
        scale. Reads the **actually-packed** plugin buffers (not a re-roundtrip
        of the original BF16 weights), so any FP4/FP8 rounding the packer
        applied is preserved byte-for-byte — matching what the kernel sees.

        FC1 v6 layout:
        - ``fc_up_qweights`` shape ``[E, H, I/2]`` INT8, 2 FP4 nibbles per
          byte packed along I (N-major).
        - ``fc_up_blocks_scale`` shape ``[E, I, H/16]`` INT8, 128×4 atom
          swizzle M=I, K=H/16 holding raw IEEE FP8 E4M3 bytes.
        - ``fc_up_global_scale`` shape ``[E]`` FP32 = ``s_max_ex / 448``.
        """
        E = int(num_experts)
        H = int(hidden_size)
        I = int(moe_inter_size)
        up_q = np.ascontiguousarray(fc_up_qweights, dtype=np.int8)
        up_bs = np.ascontiguousarray(fc_up_blocks_scale, dtype=np.int8)
        assert up_q.shape == (E, H, I // 2), (
            f"FC1 qweights shape {up_q.shape} != ({E}, {H}, {I//2})")
        assert up_bs.shape == (E, I, H // 16), (
            f"FC1 blocks_scale shape {up_bs.shape} != ({E}, {I}, {H//16})")

        # Unpack FP4 nibbles → [E, H, I]
        up_q_u8 = up_q.view(np.uint8)
        lo = (up_q_u8 & 0x0F).astype(np.int32)
        hi = ((up_q_u8 >> 4) & 0x0F).astype(np.int32)
        nibbles = np.empty((E, H, I), dtype=np.int32)
        nibbles[:, :, 0::2] = lo
        nibbles[:, :, 1::2] = hi

        fp4_levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                              dtype=np.float32)
        mag = fp4_levels[nibbles & 0x7]
        sign = np.where((nibbles & 0x8) != 0, -1.0, 1.0).astype(np.float32)
        fp4_vals = sign * mag  # [E, H, I]

        # Gather IEEE FP8 E4M3 bytes from atom layout → [E, I, H/16] FP32
        num_sf_cols = H // 16
        offsets = (NemotronHMoEReference._vectorized_atom_sf_offsets(
            I, num_sf_cols)).reshape(-1)
        sf_u8 = up_bs.view(np.uint8).reshape(E, -1)
        sf_gathered = sf_u8[:, offsets].reshape(E, I,
                                                num_sf_cols).astype(np.uint8)
        sf_fp32 = (torch.as_tensor(sf_gathered, dtype=torch.uint8).view(
            torch.float8_e4m3fn).float().numpy())

        # Broadcast SF [E, I, H/16] → [E, H, I] (expand 16× along H axis)
        sf_expanded = np.ascontiguousarray(
            np.repeat(sf_fp32, 16, axis=2).transpose(0, 2, 1))

        gs = np.asarray(fc_up_global_scale, dtype=np.float32).reshape(E, 1, 1)
        return (fp4_vals * sf_expanded * gs).astype(np.float32)

    @staticmethod
    def dequant_fp4_prefill_weight_fc2(
        fc_down_qweights: np.ndarray,
        fc_down_blocks_scale: np.ndarray,
        fc_down_global_scale: np.ndarray,
        *,
        num_experts: int,
        hidden_size: int,
        moe_inter_size: int,
    ) -> np.ndarray:
        """Dequant packed v6 FC2 weights → dense ``[E, I, H]`` FP32.

        Symmetric to :meth:`dequant_fp4_prefill_weight_fc1`. FC2 v6 layout:
        - ``fc_down_qweights`` shape ``[E, I, H/2]`` INT8, 2 FP4 nibbles per
          byte packed along H (N-major).
        - ``fc_down_blocks_scale`` shape ``[E, H, I/16]`` INT8, atom swizzle
          M=H, K=I/16.
        - ``fc_down_global_scale`` shape ``[E]`` FP32.
        """
        E = int(num_experts)
        H = int(hidden_size)
        I = int(moe_inter_size)
        dn_q = np.ascontiguousarray(fc_down_qweights, dtype=np.int8)
        dn_bs = np.ascontiguousarray(fc_down_blocks_scale, dtype=np.int8)
        assert dn_q.shape == (E, I, H // 2)
        assert dn_bs.shape == (E, H, I // 16)

        dn_q_u8 = dn_q.view(np.uint8)
        lo = (dn_q_u8 & 0x0F).astype(np.int32)
        hi = ((dn_q_u8 >> 4) & 0x0F).astype(np.int32)
        nibbles = np.empty((E, I, H), dtype=np.int32)
        nibbles[:, :, 0::2] = lo
        nibbles[:, :, 1::2] = hi

        fp4_levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                              dtype=np.float32)
        mag = fp4_levels[nibbles & 0x7]
        sign = np.where((nibbles & 0x8) != 0, -1.0, 1.0).astype(np.float32)
        fp4_vals = sign * mag  # [E, I, H]

        num_sf_cols = I // 16
        offsets = (NemotronHMoEReference._vectorized_atom_sf_offsets(
            H, num_sf_cols)).reshape(-1)
        sf_u8 = dn_bs.view(np.uint8).reshape(E, -1)
        sf_gathered = sf_u8[:, offsets].reshape(E, H,
                                                num_sf_cols).astype(np.uint8)
        sf_fp32 = (torch.as_tensor(sf_gathered, dtype=torch.uint8).view(
            torch.float8_e4m3fn).float().numpy())

        # [E, H, I/16] → [E, I, H]
        sf_expanded = np.ascontiguousarray(
            np.repeat(sf_fp32, 16, axis=2).transpose(0, 2, 1))

        gs = np.asarray(fc_down_global_scale,
                        dtype=np.float32).reshape(E, 1, 1)
        return (fp4_vals * sf_expanded * gs).astype(np.float32)

    @staticmethod
    def _fp4_roundtrip_row_plugin_parity(x_nh: np.ndarray,
                                         sf_scale: float) -> np.ndarray:
        """Plugin-parity FP4 activation roundtrip: matches ``fp4Quantize.cu`` exactly.

        Takes the same user-provided ``sfScale`` the plugin consumes (one entry
        of ``hidden_global_scale``). For each 16-element block:
          - ``SFValue = (vecMax/6) / sfScale``
          - ``sf_fp8 = fp8_e4m3(SFValue)``     (FP8 E4M3 round + saturate)
          - ``effective_scale = sf_fp8 * sfScale``   (= 1/outScale)
          - ``nibble = round_nvfp4(v / effective_scale)``
          - ``v_recovered = nibble * effective_scale``
        Forward and inverse share the same FP8-rounded scale, so the round-trip
        is consistent the same way the plugin's ``fp4Quantize + tcgen05`` dequant
        pipeline is consistent.
        """
        x_arr = np.ascontiguousarray(x_nh, dtype=np.float32)
        shape = x_arr.shape
        if shape[-1] % 16 != 0:
            raise ValueError(
                f"last dim must be divisible by 16 for NVFP4 blocks; got {shape[-1]}"
            )
        sf_scale_f = max(float(sf_scale), 1e-12)
        fp4_levels_pos = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                  dtype=np.float32)
        flat = x_arr.reshape(-1, shape[-1])
        out = np.zeros_like(flat)
        n_rows, n_cols = flat.shape
        for r in range(n_rows):
            for i0 in range(0, n_cols, 16):
                blk = flat[r, i0:i0 + 16]
                vec_max = float(np.max(np.abs(blk)))
                if vec_max <= 0.0:
                    continue
                sf_value = (vec_max / 6.0) / sf_scale_f
                sf_fp8_fp32 = float(
                    torch.tensor(sf_value, dtype=torch.float32).to(
                        torch.float8_e4m3fn).to(torch.float32).item())
                if not np.isfinite(sf_fp8_fp32) or sf_fp8_fp32 <= 0.0:
                    continue
                effective_scale = sf_fp8_fp32 * sf_scale_f
                if not np.isfinite(effective_scale) or effective_scale <= 0.0:
                    continue
                inv = 1.0 / effective_scale
                for i in range(16):
                    v = float(blk[i])
                    qn = v * inv
                    if not np.isfinite(qn):
                        continue
                    if qn > 6.0:
                        qn = 6.0
                    elif qn < -6.0:
                        qn = -6.0
                    abs_qn = abs(qn)
                    idx = int(np.argmin(np.abs(fp4_levels_pos - abs_qn)))
                    sign = 1.0 if qn >= 0.0 else -1.0
                    bucket = sign * float(fp4_levels_pos[idx])
                    val = bucket * effective_scale
                    if np.isfinite(val):
                        out[r, i0 + i] = val
        return out.reshape(*shape)

    @staticmethod
    def _fp4_roundtrip_row_with_s_max(x_nh: np.ndarray,
                                      s_max: float) -> np.ndarray:
        """FP4-roundtrip along the last axis using ``s_max`` as the global scale.

        ``x_nh`` shape ``(..., H)`` with ``H`` divisible by 64. Each 64-element
        chunk is NVFP4-quantized (4 per-16-block FP8 scales, per-element FP4
        rounding) then dequantized — same pipeline the plugin's ``fp4Quantize``
        kernel applies on activations.
        """
        x_arr = np.ascontiguousarray(x_nh, dtype=np.float32)
        shape = x_arr.shape
        if shape[-1] % 64 != 0:
            raise ValueError(
                f"last dim must be divisible by 64 for NVFP4 tiling; got {shape[-1]}"
            )
        s_max_f = max(float(s_max), 1e-12)
        flat = x_arr.reshape(-1, shape[-1])
        out = np.zeros_like(flat)
        for r in range(flat.shape[0]):
            for c_start in range(0, shape[-1], 64):
                chunk = np.ascontiguousarray(flat[r, c_start:c_start + 64])
                pl, sw = _quantize_f32x64_to_fp4x64_with_f8x4_block_scale(
                    chunk, expert_block_scale_max_fp32=s_max_f)
                dequant_intermediate = (
                    NemotronHMoEReference.unpack_nvfp4_marlin_tile_64(pl, sw))
                out[r, c_start:c_start +
                    64] = dequant_intermediate * (s_max_f / 448.0)
        return out.reshape(*shape)

    @staticmethod
    def fp4_simulated_dense_forward(
        x_bh: np.ndarray,
        topk_weights: np.ndarray,
        topk_indices: np.ndarray,
        w_up_ehi_sim: np.ndarray,
        w_down_eih_sim: np.ndarray,
        *,
        activation_type: int,
        sf_scale_fc1: float,
        sf_scale_fc2: float | None = None,
    ) -> np.ndarray:
        """Dense MoE with plugin-parity FP4 activation roundtrip at both MatMul inputs.

        ``sf_scale_fc1`` is the FC1-input global scale (``hidden_global_scale[0]``,
        matches the plugin's calibrated scalar; also matches
        ``max|hidden|/(448·6)`` at runtime in the test, so this is effectively
        the cutedsl ``input_global_sf = None`` path too).

        ``sf_scale_fc2`` controls the FC2-input (intermediate) global scale:
        - Pass a float to mirror the plugin exactly (uses the pre-calibrated
          ``hidden_global_scale[1]``).
        - Pass ``None`` to recompute from the sim's **own** FC1 output max,
          mirroring cutedsl's ``fc1_output_global_sf = None`` path. This gives
          a tighter FP4 dynamic range for FC2 input quantization, closer to
          the ideal FP4 accuracy bound.

        FC1 outputs are not quantized in the real plugin; only the MatMul
        inputs pass through FP4.
        """
        x_bh_q = NemotronHMoEReference._fp4_roundtrip_row_plugin_parity(
            np.asarray(x_bh, dtype=np.float32), sf_scale_fc1)
        topk_weights = np.asarray(topk_weights, dtype=np.float32)
        topk_indices = np.asarray(topk_indices, dtype=np.int32)
        w_up_ehi_sim = np.asarray(w_up_ehi_sim, dtype=np.float32)
        w_down_eih_sim = np.asarray(w_down_eih_sim, dtype=np.float32)
        b, h = x_bh_q.shape
        e_ct, h_w, inter = w_up_ehi_sim.shape
        assert h_w == h
        assert w_down_eih_sim.shape == (e_ct, inter, h)
        k = topk_weights.shape[1]

        # ---- Pass 1: FC1 compute per active (token, top-k slot), collect
        # post-activation intermediates so we can either (a) use the passed
        # ``sf_scale_fc2`` or (b) recompute a runtime global scale from the
        # observed FC1 output distribution.
        per_slot: list[tuple[int, int, float, np.ndarray]] = []
        running_max = 0.0
        for bb in range(b):
            for slot in range(k):
                ex = int(topk_indices[bb, slot])
                s = float(topk_weights[bb, slot])
                if s == 0.0 or ex < 0 or ex >= e_ct:
                    continue
                z = x_bh_q[bb] @ w_up_ehi_sim[ex]
                act = _moe_activation_numpy(z,
                                            activation_type).astype(np.float32,
                                                                    copy=False)
                per_slot.append((bb, ex, s, act))
                m = float(np.max(np.abs(act))) if act.size > 0 else 0.0
                if m > running_max:
                    running_max = m

        if sf_scale_fc2 is None:
            # cutedsl-style: FC2 input global scale derived from the sim's own
            # FC1 output max (BF16-clamped, as the plugin's FP4 quantize would
            # see it post-cast).
            fc1_max_bf16 = float(
                torch.tensor(running_max, dtype=torch.float32).clamp(
                    -65504.0, 65504.0).to(torch.bfloat16).float().item())
            sf_scale_fc2_eff = max(fc1_max_bf16 / (448.0 * 6.0), 1e-12)
        else:
            sf_scale_fc2_eff = float(sf_scale_fc2)

        # ---- Pass 2: FP4-roundtrip each intermediate with the chosen
        # ``sf_scale_fc2_eff``, then FC2 MatMul + scatter-reduce with the
        # routing weight ``s`` applied after the MatMul.
        out = np.zeros((b, h), dtype=np.float32)
        for bb, ex, s, act in per_slot:
            act_q = NemotronHMoEReference._fp4_roundtrip_row_plugin_parity(
                act[None, :], sf_scale_fc2_eff)[0]
            out[bb] += s * (act_q @ w_down_eih_sim[ex])
        return out.reshape(b, 1, h)

    @staticmethod
    def fp4_weight_roundtrip_scheme_b(w_ekn: np.ndarray) -> np.ndarray:
        """Quantize-dequantize a ``(E, K, N)`` weight tensor through NVFP4 scheme B.

        For each expert, every ``(n, k)`` column of shape ``(K,)`` is chunked into 64-element
        tiles; each tile gets 4 per-16-block FP8 scales; every element is FP4-rounded.
        Uses the same per-expert global scale ``s_max = max|W_e| / 6`` and the same
        ``MarlinConverter`` primitives as :meth:`NemotronHMoEW4A4Plugin.populate_prefill_plugin_buffers`,
        so the returned tensor is bit-equivalent to what the TRT kernel sees after the
        packer's quantization + the kernel's dequant.

        Inputs: FC1 ``w_up_ehi`` is ``(E, H, I)`` = ``(E, K=H, N=I)`` → quant along axis 1 (H, the
        packer's K axis). FC2 ``w_down_eih`` is ``(E, I, H)`` = ``(E, K=I, N=H)`` → also axis 1.
        For both, the call signature is identical: caller passes the packer-input layout, this
        helper quantizes along axis 1.
        """
        w_np = np.ascontiguousarray(w_ekn, dtype=np.float32)
        if w_np.ndim != 3:
            raise ValueError(
                f"w_ekn must be 3D (E, K, N); got shape {w_np.shape}")
        e_ct, k_dim, n_dim = w_np.shape
        if k_dim % 16 != 0:
            raise ValueError(
                f"K dim must be divisible by 16 for NVFP4 blocks; got {k_dim}")
        out = np.zeros_like(w_np)
        # Plugin-parity weight roundtrip: per-expert sf_scale = s_max_ex / 448
        # (equivalent to the packer's ``fc_*_global_scale``). The plugin's
        # tcgen05.mma dequant reads the same FP8 byte and applies this scale,
        # so the simulation must use the same FP8-round-tripped per-block scale
        # for both forward FP4 rounding and inverse reconstruction.
        for ex in range(e_ct):
            s_max = max(float(np.max(np.abs(w_np[ex]))) / 6.0, 1e-12)
            sf_scale_w = s_max / 448.0
            # Reuse the plugin-parity row helper: treat each (K,) column as a
            # single "row" to roundtrip along its 16-block sub-blocks.
            for n_idx in range(n_dim):
                col = w_np[ex, :, n_idx].copy()
                col_q = NemotronHMoEReference._fp4_roundtrip_row_plugin_parity(
                    col[None, :], sf_scale_w)[0]
                out[ex, :, n_idx] = col_q
        return out

    @staticmethod
    def hf_expert_weights_numpy_ehi_eih(
            moe: "NemotronHMoE") -> tuple[np.ndarray, np.ndarray]:
        """HF expert weights as ``w_up_ehi`` / ``w_down_eih`` (CPU, via FP16 round-trip)."""
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers import failed).")
        up = moe.experts.up_proj.detach().cpu().half().float().numpy()
        dn = moe.experts.down_proj.detach().cpu().half().float().numpy()
        # HF: up (E, inter, hidden_in), down (E, hidden_in, inter)
        w_up_ehi = np.ascontiguousarray(np.transpose(up, (0, 2, 1)))
        w_down_eih = np.ascontiguousarray(np.transpose(dn, (0, 2, 1)))
        return w_up_ehi, w_down_eih

    @staticmethod
    def reference_dense_from_hf_moe(
        moe: "NemotronHMoE",
        hidden_states_fp16_bsh: np.ndarray,
        topk_weights: np.ndarray,
        topk_indices: np.ndarray,
        *,
        hidden_size: int,
        activation_type: int,
    ) -> np.ndarray:
        """NumPy dense MoE with HF weights (FP16 activations); aligns with torch HF ref for same top-k."""
        w_up, w_down = NemotronHMoEReference.hf_expert_weights_numpy_ehi_eih(
            moe)
        x = np.asarray(hidden_states_fp16_bsh, dtype=np.float16).reshape(
            -1, int(hidden_size)).astype(np.float32)
        return NemotronHMoEReference.dense_forward_from_topk(
            x,
            topk_weights,
            topk_indices,
            w_up,
            w_down,
            activation_type=int(activation_type),
        )

    @staticmethod
    def assert_numpy_dense_matches_torch_reference(
        ref_np_bsh: np.ndarray,
        torch_ref_fp32_bsh: np.ndarray,
        case: str,
        *,
        rtol: float = 1e-3,
        atol: float = 2.0,
    ) -> None:
        """NumPy HF dense vs torch dense FP32 ref (loose ``allclose``; CPU vs NumPy ordering)."""
        a = np.asarray(ref_np_bsh, dtype=np.float32).reshape(-1)
        b = np.asarray(torch_ref_fp32_bsh, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise AssertionError(
                f"{case}: numpy vs torch ref size mismatch {a.size} vs {b.size}"
            )
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError(
                f"{case}: non-finite numpy or torch reference")
        ok = np.allclose(a, b, rtol=float(rtol), atol=float(atol))
        if ok:
            return
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        worst = int(np.argmax(diff))
        raise AssertionError(
            f"{case}: NumPy HF dense ref diverges from torch FP32 ref (rtol={rtol}, atol={atol}); "
            f"max_abs_diff={float(np.max(diff)):.6g} worst_idx={worst} np={float(a[worst]):.8g} torch={float(b[worst]):.8g}"
        )

    @staticmethod
    def assert_marlin_unpacked_numpy_matches_torch(
        numpy_marlin_bsh: np.ndarray,
        torch_marlin_bsh: np.ndarray,
        case: str,
        *,
        rtol: float = 1e-3,
        atol: float = 2e-2,
    ) -> None:
        """Require Marlin-unpacked dense outputs from NumPy ``@`` vs torch CPU ``matmul`` to agree.

        Cross-check uses ``|a-b| <= atol + rtol * max(|a|,|b|)`` (float64 diff). These two CPU refs
        differ only in GEMM ordering / accumulation; a few hundred ULPs at mid-range magnitudes is
        expected, so ``rtol=1e-3`` (0.1%) is used here — **not** the same bar as TRT vs NumPy Marlin.
        """
        a = np.asarray(numpy_marlin_bsh, dtype=np.float32).reshape(-1)
        b = np.asarray(torch_marlin_bsh, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise AssertionError(
                f"{case}: numpy vs torch marlin ref size mismatch {a.size} vs {b.size}"
            )
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise AssertionError(
                f"{case}: non-finite numpy or torch Marlin unpack ref")
        af = a.astype(np.float64)
        bf = b.astype(np.float64)
        diff = np.abs(af - bf)
        scale = np.maximum(np.abs(af), np.abs(bf))
        bound = float(atol) + float(rtol) * scale
        if np.all(diff <= bound):
            return
        excess = diff - bound
        fail_idx = int(np.argmax(excess)) if diff.size else 0
        peak = float(np.max(diff)) if diff.size else 0.0
        raise AssertionError(
            f"{case}: NumPy Marlin unpack ref vs torch Marlin unpack ref diverges "
            f"(symmetric rtol={rtol:g}, atol={atol:g}); max_abs={peak:.6g}; "
            f"largest_violation_idx={fail_idx} diff={float(diff[fail_idx]):.6g} "
            f"bound={float(bound[fail_idx]):.6g} "
            f"np={float(a[fail_idx]):.8g} torch={float(b[fail_idx]):.8g}")

    @staticmethod
    def _scaled_dense_moe_tensors_from_packed_plugin_state(
        hidden_states_fp16_bsh: np.ndarray,
        fc_up_q: np.ndarray,
        fc_up_bs: np.ndarray,
        fc_dn_q: np.ndarray,
        fc_dn_bs: np.ndarray,
        fc_up_gs: np.ndarray,
        fc_dn_gs: np.ndarray,
        *,
        hidden_size: int,
        moe_inter_size: int,
        num_experts: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack NVFP4 buffers, apply per-expert global scales, FP32 activations ``x`` ``[num_tokens, H]``."""
        w_up, w_down = NemotronHMoEReference.dense_weights_from_nvfp4_plugin_buffers(
            fc_up_q,
            fc_up_bs,
            fc_dn_q,
            fc_dn_bs,
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_inter_size=moe_inter_size,
        )
        up_gs = np.asarray(fc_up_gs, dtype=np.float32).reshape(-1)
        dn_gs = np.asarray(fc_dn_gs, dtype=np.float32).reshape(-1)
        for ex in range(int(num_experts)):
            w_up[ex] *= float(up_gs[ex])
            w_down[ex] *= float(dn_gs[ex])
        x = np.asarray(hidden_states_fp16_bsh, dtype=np.float16).reshape(
            -1, hidden_size).astype(np.float32)
        return x, w_up, w_down, up_gs, dn_gs

    @staticmethod
    def reference_from_packed_plugin_state(
        hidden_states_fp16_bsh: np.ndarray,
        topk_weights_be: np.ndarray,
        topk_indices_be: np.ndarray,
        fc_up_q: np.ndarray,
        fc_up_bs: np.ndarray,
        fc_dn_q: np.ndarray,
        fc_dn_bs: np.ndarray,
        fc_up_gs: np.ndarray,
        fc_dn_gs: np.ndarray,
        *,
        hidden_size: int,
        moe_inter_size: int,
        num_experts: int,
        activation_type: int = 0,
    ) -> np.ndarray:
        """Packed plugin state → scaled dense MoE output (same top-k as engine)."""
        x, w_up, w_down, up_gs, dn_gs = (
            NemotronHMoEReference.
            _scaled_dense_moe_tensors_from_packed_plugin_state(
                hidden_states_fp16_bsh,
                fc_up_q,
                fc_up_bs,
                fc_dn_q,
                fc_dn_bs,
                fc_up_gs,
                fc_dn_gs,
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
            ))
        out = NemotronHMoEReference.dense_forward_from_topk(
            x,
            topk_weights_be,
            topk_indices_be,
            w_up,
            w_down,
            activation_type=int(activation_type),
        )
        return out

    @staticmethod
    def assert_topk_matches_numpy_reference(
        router_logits: torch.Tensor,
        top_k: int,
        topw_torch: torch.Tensor,
        topi_torch: torch.Tensor,
        case: str,
    ) -> None:
        """Require plugin-parity top-k tensors to match :meth:`NemotronHMoEReference.moe_topk_softmax_renormalize_numpy`."""
        L = np.ascontiguousarray(router_logits.detach().float().cpu().numpy())
        nw, ni = NemotronHMoEReference.moe_topk_softmax_renormalize_numpy(
            L, int(top_k))
        tw = topw_torch.detach().float().cpu().numpy()
        ti = topi_torch.detach().cpu().numpy().astype(np.int32)
        if not np.array_equal(ti, ni):
            raise AssertionError(
                f"{case}: top-k expert indices differ from NumPy moe_topk_softmax_renormalize"
            )
        if not np.allclose(tw, nw, rtol=1e-6, atol=1e-7):
            max_d = float(np.max(np.abs(tw - nw))) if tw.size else 0.0
            raise AssertionError(
                f"{case}: top-k weights differ from NumPy moe_topk_softmax_renormalize (max_abs_delta={max_d:.6g})"
            )

    @staticmethod
    def assert_moe_output_non_degenerate(
        out_fp32_bsh: np.ndarray,
        case: str,
        *,
        min_peak_abs: float = 1e-4,
        min_fraction_abs_gt: float = 0.01,
        abs_gt_eps: float = 1e-6,
    ) -> None:
        """Torch MoE ref: finite, peak magnitude, fraction of elems above ``abs_gt_eps``."""
        a = np.asarray(out_fp32_bsh, dtype=np.float32).reshape(-1)
        if a.size == 0:
            raise AssertionError(f"{case}: empty torch MoE reference output")
        if not np.all(np.isfinite(a)):
            raise AssertionError(
                f"{case}: torch MoE reference has non-finite values")
        am = np.abs(a.astype(np.float64))
        peak = float(np.max(am))
        if peak <= min_peak_abs:
            raise AssertionError(
                f"{case}: torch MoE ref max|out|={peak:.6g} <= {min_peak_abs} (expected non-trivial peak)"
            )
        frac = float(np.mean(am > float(abs_gt_eps)))
        if frac < float(min_fraction_abs_gt):
            raise AssertionError(
                f"{case}: torch MoE ref only {frac * 100:.2f}% of elems have |x| > {abs_gt_eps:g} "
                f"(require >= {min_fraction_abs_gt * 100:.2f}%)")

    @staticmethod
    def assert_fp16_ref_tracks_fp32_ref(
        ref_fp32_bsh: np.ndarray,
        ref_fp16_bsh: np.ndarray,
        case: str,
        *,
        rtol: float = 0.08,
        atol: float = 0.35,
    ) -> None:
        """FP16-matmul expert path should stay in the ballpark of the FP32 path (same routing, same HF weights)."""
        a = np.asarray(ref_fp32_bsh, dtype=np.float32).reshape(-1)
        b = np.asarray(ref_fp16_bsh, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise AssertionError(f"{case}: fp32 vs fp16 ref shape mismatch")
        ok = np.isfinite(a) & np.isfinite(b)
        if not np.all(ok):
            raise AssertionError(
                f"{case}: non-finite values in fp32/fp16 ref pair")
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        scale = np.maximum(np.abs(a.astype(np.float64)), 1.0)
        bound = float(atol) + float(rtol) * scale
        bad = diff > bound
        n_bad = int(np.sum(bad))
        if n_bad == 0:
            return
        worst = int(np.argmax(diff))
        raise AssertionError(
            f"{case}: fp16 torch ref diverges from fp32 ref on {n_bad}/{a.size} elems "
            f"(rtol={rtol}, atol={atol}); worst idx {worst}: fp32={float(a[worst]):.6g} fp16_as_f32={float(b[worst]):.6g} "
            f"diff={float(diff[worst]):.6g} bound={float(bound[worst]):.6g}")

    @staticmethod
    def moe_topk_softmax_renormalize_torch_plugin_parity(
        router_logits: torch.Tensor,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plugin-parity top-k on CPU float probs (tie-break: lower expert id)."""
        device = router_logits.device
        # Softmax on CPU float32 matches :meth:`NemotronHMoEReference.moe_topk_softmax_renormalize_numpy`
        # on the same logits array (GPU softmax can differ in ULPs and change top-k on ties).
        logits_cpu = router_logits.float().detach().cpu()
        probs = torch.nn.functional.softmax(logits_cpu, dim=-1)
        num_tokens, num_experts = probs.shape
        k = min(int(top_k), int(num_experts))
        topw = probs.new_zeros((num_tokens, k))
        topi = probs.new_empty((num_tokens, k), dtype=torch.long)
        for t in range(int(num_tokens)):
            row = probs[t]
            used = torch.zeros(num_experts, dtype=torch.bool, device="cpu")
            for _ki in range(k):
                best_e = -1
                best_p = -1.0
                for e in range(int(num_experts)):
                    if bool(used[e].item()):
                        continue
                    p = float(row[e].item())
                    if best_e < 0 or p > best_p or (p == best_p
                                                    and e < best_e):
                        best_e = e
                        best_p = p
                assert best_e >= 0
                used[best_e] = True
                topi[t, _ki] = best_e
                topw[t, _ki] = float(best_p)
            denom = float(topw[t].sum().item()) + 1e-20
            topw[t] = topw[t] / denom
        return topw.to(device=device), topi.to(device=device)

    @staticmethod
    def routed_experts_from_router_logits_via_hf_experts(
        moe: "NemotronHMoE",
        hidden_bsh: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        *,
        compute_dtype: torch.dtype,
        activation_type: int,
        case: str,
    ) -> np.ndarray:
        """HF expert weights + plugin-parity top-k; dense FP32 matmul matching :meth:`reference_dense_from_hf_moe`.

        Does not call ``moe.experts`` in FP16: that path can overflow for large activations, while the
        NumPy dense reference uses FP32 activations after FP16 storage round-trip.
        ``compute_dtype`` selects FP32 vs FP16 routing masses (same as the old HF forward).
        """
        if NemotronHMoE is None:
            raise RuntimeError(
                "NemotronHMoE is not available (transformers import failed).")
        if not isinstance(moe, NemotronHMoE):
            raise TypeError(
                f"moe must be NemotronHMoE, got {type(moe).__name__!r}")

        h_sz = int(moe.config.hidden_size)
        # Match ``NemotronHExperts`` / ``up_proj`` device — ``hidden_bsh`` may be CPU (e.g. host-only fixtures)
        # while the MoE module lives on CUDA.
        exp_dev = moe.experts.up_proj.device
        logits = router_logits.to(device=exp_dev, dtype=torch.float32)
        topw, topi = NemotronHMoEReference.moe_topk_softmax_renormalize_torch_plugin_parity(
            logits, int(top_k))
        NemotronHMoEReference.assert_topk_matches_numpy_reference(
            logits, int(top_k), topw, topi, case)
        # Match :meth:`reference_dense_from_hf_moe`: FP16-stored activations, then FP32 ``@`` with HF weights.
        x_np = np.asarray(
            hidden_bsh.detach().cpu().numpy(),
            dtype=np.float16,
        ).reshape(-1, h_sz).astype(np.float32)
        topw_cpu = topw.float().cpu()
        if compute_dtype == torch.float16:
            topw_cpu = topw_cpu.to(torch.float16).to(torch.float32)
        w_up, w_down = NemotronHMoEReference.hf_expert_weights_numpy_ehi_eih(
            moe)
        out_t = NemotronHMoEReference.dense_forward_from_topk_torch(
            torch.from_numpy(x_np),
            topw_cpu,
            topi.long().cpu(),
            torch.from_numpy(w_up),
            torch.from_numpy(w_down),
            activation_type=int(activation_type),
        )
        out_np = np.asarray(out_t.detach().numpy(),
                            dtype=np.float32).reshape(tuple(hidden_bsh.shape))
        NemotronHMoEReference.assert_moe_output_non_degenerate(out_np, case)
        return out_np

    @staticmethod
    def moe_activation_torch(z: torch.Tensor,
                             activation_type: int) -> torch.Tensor:
        """FP32 expert nonlinearity matching :meth:`_moe_activation_numpy`."""
        z = z.float()
        if int(activation_type) == 1:
            zc = z.clamp(-50.0, 50.0)
            return zc / (1.0 + torch.exp(-zc))
        t = torch.clamp(z, min=0.0)
        return t * t

    @staticmethod
    def dense_forward_from_topk_torch(
        x_bh: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        w_up_ehi: torch.Tensor,
        w_down_eih: torch.Tensor,
        *,
        activation_type: int = 0,
    ) -> torch.Tensor:
        """CPU torch ``matmul`` version of :meth:`dense_forward_from_topk`."""
        x_bh = x_bh.float().cpu()
        topk_weights = topk_weights.float().cpu()
        topk_indices = topk_indices.long().cpu()
        w_up_ehi = w_up_ehi.float().cpu()
        w_down_eih = w_down_eih.float().cpu()
        b, h = int(x_bh.shape[0]), int(x_bh.shape[1])
        e, h2, inter = int(w_up_ehi.shape[0]), int(w_up_ehi.shape[1]), int(
            w_up_ehi.shape[2])
        assert h2 == h
        assert tuple(w_down_eih.shape) == (e, inter, h)
        bk, k = int(topk_weights.shape[0]), int(topk_weights.shape[1])
        assert bk == b
        out = torch.zeros(b, h, dtype=torch.float32)
        for bb in range(b):
            for slot in range(k):
                ex = int(topk_indices[bb, slot].item())
                s = float(topk_weights[bb, slot].item())
                if s == 0.0 or ex < 0 or ex >= e:
                    continue
                z = x_bh[bb] @ w_up_ehi[ex]
                act = NemotronHMoEReference.moe_activation_torch(
                    z, activation_type)
                t = act * s
                out[bb] = out[bb] + t @ w_down_eih[ex]
        return out.reshape(b, 1, h)

    @staticmethod
    def routed_experts_from_router_logits_via_marlin_unpacked_weights(
        hidden_bsh: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        *,
        fc_up_q: np.ndarray,
        fc_up_bs: np.ndarray,
        fc_dn_q: np.ndarray,
        fc_dn_bs: np.ndarray,
        fc_up_gs: np.ndarray,
        fc_dn_gs: np.ndarray,
        hidden_size: int,
        moe_inter_size: int,
        num_experts: int,
        activation_type: int,
        case: str,
    ) -> np.ndarray:
        """Marlin-unpacked weights + CPU torch matmul; same routing checks as HF path."""
        hidden_np = np.ascontiguousarray(hidden_bsh.detach().cpu().numpy())
        x, w_up, w_down, _, _ = NemotronHMoEReference._scaled_dense_moe_tensors_from_packed_plugin_state(
            hidden_np,
            fc_up_q,
            fc_up_bs,
            fc_dn_q,
            fc_dn_bs,
            fc_up_gs,
            fc_dn_gs,
            hidden_size=int(hidden_size),
            moe_inter_size=int(moe_inter_size),
            num_experts=int(num_experts),
        )
        exp_dev = router_logits.device
        logits = router_logits.to(device=exp_dev, dtype=torch.float32)
        topw, topi = NemotronHMoEReference.moe_topk_softmax_renormalize_torch_plugin_parity(
            logits, int(top_k))
        NemotronHMoEReference.assert_topk_matches_numpy_reference(
            logits, int(top_k), topw, topi, case)
        out_t = NemotronHMoEReference.dense_forward_from_topk_torch(
            torch.from_numpy(x),
            topw,
            topi,
            torch.from_numpy(w_up),
            torch.from_numpy(w_down),
            activation_type=int(activation_type),
        )
        return np.asarray(out_t.detach().numpy(),
                          dtype=np.float32).reshape(tuple(hidden_bsh.shape))


# --- TensorRT: load plugin + run engine ---


def _require_trt_ok(ok: bool) -> None:
    if not ok:
        raise RuntimeError("TensorRT API call failed")


def _resolve_edgellm_trt_plugin_path(plugin_so: Path) -> Path:
    """Real path for plugin .so (symlink-safe); glob parent if basename missing."""
    p = plugin_so.expanduser()
    if p.is_file():
        return p.resolve()
    parent = p.parent
    if parent.is_dir():
        for cand in sorted(parent.glob("libNvInfer_edgellm_plugin.so*"),
                           key=lambda x: -len(x.name)):
            if cand.is_file():
                return cand.resolve()
    return p.resolve(strict=False)


def _dedupe_existing_lib_dirs(dirs: list[Path]) -> list[Path]:
    """Resolve, keep existing directories only, preserve order, drop duplicates."""
    out: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        try:
            r = d.resolve()
        except OSError:
            continue
        if r.is_dir() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _tensorrt_lib_dirs_from_trt_package_and_ld_path() -> list[Path]:
    """``TRT_PACKAGE_DIR/lib`` plus ``LD_LIBRARY_PATH`` entries (unordered, may contain duplicates)."""
    dirs: list[Path] = []
    trt_pkg = os.environ.get("TRT_PACKAGE_DIR", "").strip()
    if trt_pkg:
        dirs.append(Path(trt_pkg) / "lib")
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        e = entry.strip()
        if e:
            dirs.append(Path(e))
    return dirs


def _tensorrt_lib_dirs_env_ld_only() -> list[Path]:
    """``TRT_PACKAGE_DIR/lib`` and ``LD_LIBRARY_PATH`` entries (no Python tensorrt path)."""
    return _dedupe_existing_lib_dirs(
        _tensorrt_lib_dirs_from_trt_package_and_ld_path())


def _tensorrt_lib_search_dirs() -> list[Path]:
    """Directories that may contain ``libnvinfer.so*`` (for RTLD_GLOBAL preload)."""
    dirs = _tensorrt_lib_dirs_from_trt_package_and_ld_path()
    if trt is not None:
        dirs.append(Path(
            trt.__file__).resolve().parent)  # type: ignore[union-attr]
    else:
        try:
            import tensorrt as trt_mod

            dirs.append(Path(trt_mod.__file__).resolve().parent)
        except ImportError:
            pass
    return _dedupe_existing_lib_dirs(dirs)


_libnvinfer_rtld_global_path: Path | None = None


def _ctypes_preload_libnvinfer_from_dirs(dirs: list[Path], *,
                                         verbose: bool) -> Path | None:
    global _libnvinfer_rtld_global_path
    import ctypes

    if not hasattr(ctypes, "RTLD_GLOBAL"):
        return None
    for d in dirs:
        candidates = sorted(d.glob("libnvinfer.so*"),
                            key=lambda p: len(p.name),
                            reverse=True)
        for so in candidates:
            if not so.is_file():
                continue
            try:
                ctypes.CDLL(os.fspath(so), mode=ctypes.RTLD_GLOBAL)
                if _libnvinfer_rtld_global_path is None:
                    _libnvinfer_rtld_global_path = so.resolve()
                    if verbose:
                        print(
                            f"[TRT] Preloaded libnvinfer (RTLD_GLOBAL): {so}")
                return so
            except OSError:
                continue
    return None


def _preload_libnvinfer_rtld_global(verbose: bool) -> Path | None:
    p = _ctypes_preload_libnvinfer_from_dirs(_tensorrt_lib_dirs_env_ld_only(),
                                             verbose=verbose)
    if p is not None:
        return p
    p = _ctypes_preload_libnvinfer_from_dirs(_tensorrt_lib_search_dirs(),
                                             verbose=verbose)
    if p is None and verbose:
        print(
            "[TRT] Note: could not preload libnvinfer.so* from TRT_PACKAGE_DIR, LD_LIBRARY_PATH, or "
            "the tensorrt package directory. Set TRT_PACKAGE_DIR to the TensorRT tree used to build "
            "the plugin if load_library fails.")
    return p


def _get_nvfp4_moe_plugin_creator(registry,
                                  plugin_name: str = "Nvfp4MoePlugin"
                                  ) -> object | None:
    getters: list[tuple[str, object]] = []
    for gname in ("get_creator", "get_plugin_creator", "getPluginCreator"):
        g = getattr(registry, gname, None)
        if callable(g):
            getters.append((gname, g))
    if not getters:
        return None
    # Plugin version: "1" = FP16 hidden + NVFP4 weights, N-major FC1/FC2, scheme-B SF, decode
    # SF inputs at slots 9/10, e_score_correction_bias at slot 11, and routing_mode attribute
    # selecting between moeTopkSoftmax (mode 0, default) and moeSigmoidGroupTopk (mode 1).
    # Keep older versions in the search order so a stale engine raises a descriptive
    # not-compatible error rather than "plugin not found."
    versions = ("6", "6.0", "5", "5.0", "4", "4.0", "3", "3.0", "2", "2.0",
                "1", "1.0")
    namespaces = ("", "trt")
    for _gname, getter in getters:
        for ver in versions:
            for ns in namespaces:
                try:
                    c = getter(plugin_name, ver, ns)
                except TypeError:
                    c = None
                except Exception:
                    c = None
                if c is not None:
                    return c
        for ver in versions:
            try:
                c = getter(plugin_name, ver)
            except TypeError:
                c = None
            except Exception:
                c = None
            else:
                if c is not None:
                    return c
    return None


def _plugin_dlopen_diagnosis(plugin_path: Path) -> str:
    import ctypes

    lines: list[str] = []
    try:
        ctypes.CDLL(os.fspath(plugin_path),
                    mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        lines.append("ctypes.CDLL(plugin, RTLD_GLOBAL) succeeded.")
    except OSError as exc:
        lines.append(f"ctypes.CDLL(plugin): {exc}")

    if sys.platform.startswith("linux") and shutil.which("ldd"):
        try:
            proc = subprocess.run(
                ["ldd", os.fspath(plugin_path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            if out.strip():
                lines.append("ldd output:")
                lines.append(out.rstrip())
        except (subprocess.TimeoutExpired, OSError) as exc:
            lines.append(f"ldd failed: {exc}")

    return "\n".join(lines)


def load_nvfp4_moe_edge_llm_plugins(logger,
                                    plugin_so: Path,
                                    verbose: bool = True) -> None:
    import ctypes

    assert trt is not None
    lib_path = _resolve_edgellm_trt_plugin_path(plugin_so)
    path_str = os.fspath(lib_path)
    if verbose:
        print(f"[TRT] Loading Edge-LLM plugins from:\n      {lib_path}")

    _preload_libnvinfer_rtld_global(verbose=verbose)

    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()
    loaded_via_registry = bool(registry.load_library(path_str))

    def _try_ctypes_plugin() -> None:
        if not hasattr(ctypes, "RTLD_GLOBAL"):
            raise RuntimeError(
                "ctypes.RTLD_GLOBAL not available on this platform")
        ctypes.CDLL(path_str, mode=ctypes.RTLD_GLOBAL)

    if not loaded_via_registry:
        if verbose:
            print(
                "[TRT] load_library returned False; opening plugin with ctypes RTLD_GLOBAL "
                "(runs static ctors / plugin registration) …")
        try:
            _try_ctypes_plugin()
        except OSError as exc:
            diag = _plugin_dlopen_diagnosis(lib_path)
            raise RuntimeError("Could not load Edge-LLM plugin DSO.\n"
                               f"  Path: {lib_path}\n"
                               f"  ctypes: {exc}\n\n"
                               f"Diagnostics:\n{diag}") from exc

    creator = _get_nvfp4_moe_plugin_creator(registry)
    if creator is None and loaded_via_registry:
        if verbose:
            print(
                "[TRT] Nvfp4MoePlugin not visible after load_library; retrying ctypes RTLD_GLOBAL …"
            )
        try:
            _try_ctypes_plugin()
        except OSError:
            pass
        creator = _get_nvfp4_moe_plugin_creator(registry)

    if creator is None:
        diag = _plugin_dlopen_diagnosis(lib_path)
        raise RuntimeError(
            "Edge-LLM plugin library loaded but Nvfp4MoePlugin is not in the TensorRT plugin registry.\n"
            f"  Path: {lib_path}\n"
            "Rebuild the plugin against the same TensorRT as the Python ``tensorrt`` package "
            "(set TRT_PACKAGE_DIR when running CMake).\n\n"
            f"Diagnostics:\n{diag}")

    if verbose:
        if loaded_via_registry:
            print(
                "[TRT] Nvfp4MoePlugin is registered (load_library returned True)."
            )
        else:
            print(
                "[TRT] Nvfp4MoePlugin is registered via ctypes RTLD_GLOBAL "
                "(TensorRT load_library returned False; this is a known quirk on some 10.x builds)."
            )


def _trt_torch_dtype(trt_dtype):
    assert trt is not None
    if trt_dtype == trt.float16:
        return torch.float16
    if trt_dtype == trt.float32:
        return torch.float32
    if trt_dtype == trt.int32:
        return torch.int32
    if trt_dtype == trt.int8:
        return torch.int8
    raise ValueError(f"unsupported TensorRT dtype: {trt_dtype}")


def execute_trt_engine(
    serialized: bytes,
    inputs: dict[str, np.ndarray],
    stream=None,
    device=None,
    *,
    verbose: bool = True,
    label: str = "",
) -> np.ndarray:
    assert trt is not None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dev = device or torch.device("cuda", torch.cuda.current_device())
    if stream is None:
        stream = torch.cuda.Stream(device=dev)

    tag = f" {label}" if label else ""
    if verbose:
        print(
            f"[exec{tag}] CUDA device={dev}  engine blob={len(serialized)} bytes\n"
            f"[exec{tag}] Host inputs:")
        for name, arr in inputs.items():
            print(
                f"[exec{tag}]   {name}: shape={tuple(arr.shape)} dtype={arr.dtype}"
            )

    t0 = time.perf_counter()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context returned None")
    if verbose:
        print(
            f"[exec{tag}] deserialize + create_execution_context in {time.perf_counter() - t0:.3f}s"
        )

    for name, arr in inputs.items():
        _require_trt_ok(context.set_input_shape(name, tuple(arr.shape)))

    bindings: dict[str, torch.Tensor] = {}
    output_names: list[str] = []
    if verbose:
        print(f"[exec{tag}] I/O tensors ({engine.num_io_tensors}):")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(context.get_tensor_shape(name))
        trt_dtype = engine.get_tensor_dtype(name)
        torch_dtype = _trt_torch_dtype(trt_dtype)
        bindings[name] = torch.empty(shape, dtype=torch_dtype, device=dev)
        if verbose:
            kind = "OUTPUT" if mode == trt.TensorIOMode.OUTPUT else "INPUT"
            print(
                f"[exec{tag}]   [{i}] {kind} {name}: shape={shape} trt_dtype={trt_dtype}"
            )
        if mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    if not output_names:
        raise RuntimeError("TensorRT engine has no output tensors")
    # Prefer the Nvfp4MoePlugin output (``mark_output`` name in verify_nvfp4_moe_trt_engine.py); TensorRT may list
    # multiple OUTPUT tensors or order them such that the first is not the plugin dense MoE tensor — reading the
    # wrong binding yields uninitialized (NaN) FP16.
    output_name = "output" if "output" in output_names else output_names[0]
    for on in output_names:
        bindings[on].zero_()

    if verbose:
        print(f"[exec{tag}] H2D copy + execute_async_v3 …")
    t1 = time.perf_counter()
    with torch.cuda.stream(stream):
        for name, arr in inputs.items():
            host = torch.from_numpy(np.ascontiguousarray(arr))
            # Pageable CPU numpy → GPU: non_blocking requires pinned host memory; async copies can race and
            # corrupt INT8/FP32 plugin inputs (downstream MoE output all-NaN).
            bindings[name].copy_(
                host.to(device=dev,
                        dtype=bindings[name].dtype,
                        non_blocking=False),
                non_blocking=False,
            )
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            _require_trt_ok(
                context.set_tensor_address(name, bindings[name].data_ptr()))
        _require_trt_ok(context.execute_async_v3(stream.cuda_stream))
    stream.synchronize()
    if verbose:
        print(
            f"[exec{tag}] GPU kernel finished in {time.perf_counter() - t1:.3f}s "
            f"(wall incl. sync)\n[exec{tag}] D2H output '{output_name}' shape={tuple(bindings[output_name].shape)}"
        )

    return np.array(bindings[output_name].detach().cpu().numpy(), copy=True)


# --- Serialize engine (explicit router) + accuracy driver ---


def build_nvfp4_moe_engine_trt_api(
    module: object,
    dummy_hidden_fp16: Any,
    logger,
    *,
    verbose: bool = True,
    label: str = "",
    explicit_router_logits: bool = False,
    seq_max_for_profile: int | None = None,
) -> bytes:
    """
    Build a serialized TensorRT engine for ``Nvfp4MoePlugin`` using the network API (no ONNX).

    ``module`` must be a :class:`NemotronHMoEMLP` on CPU with NVFP4 buffers already filled
    (after ``prepare_for_export()``). ``dummy_hidden_fp16`` defines static ``(batch, seq, hidden)``
    layout (used for optimization profile bounds); it must match ``module.hidden_size``.
    ``hidden_states`` is always FP16; activation NVFP4 quantization (for the prefill path) is
    computed inside the plugin.

    When ``explicit_router_logits`` is True, the graph matches ``experiment/test_nvfp4_moe_trt_plugin_accuracy.cu``:
    ``router_logits`` is a dedicated FP32 input ``(batch * seq, num_experts)`` rather than
    ``cast(hidden_states) @ gate^T + bias``. That avoids routing ``hidden_states`` through an extra
    matmul subgraph (which can interact badly with STRONGLY_TYPED on some GPUs, e.g. spurious ~0 MoE
    output when ``top_k > 1`` despite correct logits at the plugin boundary).

    Accepts an optional ``activation_type`` override (default 0 = ReLU² for NemotronH; 1 = SiLU for
    Qwen3 gated MoE) and ``routing_mode`` override (default 0 = softmax; 1 = sigmoid group topk).
    """
    import tensorrt as trt

    if dummy_hidden_fp16.dim() != 3:
        raise ValueError(
            f"dummy_hidden_fp16 must be (B,S,H), got shape {tuple(dummy_hidden_fp16.shape)}"
        )
    b, s, h = (int(dummy_hidden_fp16.shape[i]) for i in range(3))
    if int(getattr(module, "hidden_size")) != h:
        raise ValueError(
            f"dummy hidden {h} != module.hidden_size {getattr(module, 'hidden_size')}"
        )

    tag = f" {label}" if label else ""
    if verbose:
        r_mode = "explicit FP32 input (T,E)" if explicit_router_logits else "from hidden_states linear"
        print(
            f"[TRT API{tag}] Building engine (STRONGLY_TYPED, add_plugin_v3)  "
            f"hidden_states=FP16 ({b},{s},{h})  router={r_mode}")

    registry = trt.get_plugin_registry()
    creator = _get_nvfp4_moe_plugin_creator(registry)
    if creator is None:
        raise RuntimeError(
            "Nvfp4MoePlugin creator not found (load Edge-LLM plugin DSO first)."
        )

    # Extract plugin attributes from NemotronHMoEMLP (new FE).
    e_ct = int(module.n_routed_experts)
    top_k = int(module.num_experts_per_tok)
    inter = int(module.moe_intermediate_size)
    f_act = int(getattr(module, "_plugin_activation_type", 0))
    f_qgs = 16  # NVFP4 always uses group_size=16
    f_n_group = int(module.gate.n_group)
    f_topk_group = int(module.gate.topk_group)
    f_norm_topk = int(module.gate.norm_topk_prob)
    f_routed_scaling = float(module.gate.routed_scaling_factor)
    f_routing_mode = int(getattr(module, "_plugin_routing_mode", 0))
    # TRT PluginField keeps a raw pointer into the numpy buffer; bind them to named locals so
    # the arrays outlive ``creator.create_plugin`` below (CPython can free a temporary in a
    # list-literal expression before the call returns).
    arr_num_experts = np.array([e_ct], dtype=np.int32)
    arr_top_k = np.array([top_k], dtype=np.int32)
    arr_hidden_size = np.array([h], dtype=np.int32)
    arr_moe_inter_size = np.array([inter], dtype=np.int32)
    arr_activation_type = np.array([f_act], dtype=np.int32)
    arr_qgs = np.array([f_qgs], dtype=np.int32)
    arr_n_group = np.array([f_n_group], dtype=np.int32)
    arr_topk_group = np.array([f_topk_group], dtype=np.int32)
    arr_norm_topk = np.array([f_norm_topk], dtype=np.int32)
    arr_routed_scaling = np.array([f_routed_scaling], dtype=np.float32)
    arr_routing_mode = np.array([f_routing_mode], dtype=np.int32)
    pfc = trt.PluginFieldCollection([
        trt.PluginField("num_experts", arr_num_experts,
                        trt.PluginFieldType.INT32),
        trt.PluginField("top_k", arr_top_k, trt.PluginFieldType.INT32),
        trt.PluginField("hidden_size", arr_hidden_size,
                        trt.PluginFieldType.INT32),
        trt.PluginField("moe_inter_size", arr_moe_inter_size,
                        trt.PluginFieldType.INT32),
        trt.PluginField("activation_type", arr_activation_type,
                        trt.PluginFieldType.INT32),
        trt.PluginField("quantization_group_size", arr_qgs,
                        trt.PluginFieldType.INT32),
        trt.PluginField("n_group", arr_n_group, trt.PluginFieldType.INT32),
        trt.PluginField("topk_group", arr_topk_group,
                        trt.PluginFieldType.INT32),
        trt.PluginField("norm_topk_prob", arr_norm_topk,
                        trt.PluginFieldType.INT32),
        trt.PluginField("routed_scaling_factor", arr_routed_scaling,
                        trt.PluginFieldType.FLOAT32),
        trt.PluginField("routing_mode", arr_routing_mode,
                        trt.PluginFieldType.INT32),
    ])
    try:
        plugin = creator.create_plugin("Nvfp4MoePlugin", pfc,
                                       trt.TensorRTPhase.BUILD)
    except TypeError:
        plugin = creator.create_plugin("Nvfp4MoePlugin", pfc)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    hidden_in = network.add_input("hidden_states", trt.float16, (b, s, h))
    # hidden_global_scale is a runtime input so test harnesses can feed calibrated FC1/FC2 scales
    # without rebuilding the engine. For the pure W4A16 decode path (numTokens<=16), values are
    # unused at runtime, but the shape contract must still be satisfied.
    hidden_gs_in = network.add_input("hidden_global_scale", trt.float32, (2, ))

    if explicit_router_logits:
        router_logits = network.add_input("router_logits", trt.float32,
                                          (b * s, e_ct))
    else:
        flat = network.add_shuffle(hidden_in)
        flat.reshape_dims = (b * s, h)

        router_f32 = network.add_cast(flat.get_output(0), trt.float32)

        # Match ``F.linear(x, W)`` / ``nn.Linear``: ``W`` is ``(E, H)``; logits are ``x @ W.T`` i.e. ``(T,H) @ (H,E)``.
        # Bake ``W.T`` as a contiguous ``(H, E)`` constant and use ``NONE`` on both operands so the GEMM does not rely
        # on ``MatrixOperation.TRANSPOSE`` for the weight tensor (avoids STRONGLY_TYPED / constant-layer quirks that can
        # corrupt logits and yield NaNs → plugin decode skips all experts, ~0 output).
        gate_w = module.gate.weight.detach().float().cpu().numpy()
        if gate_w.shape != (e_ct, h):
            raise ValueError(
                f"gate.weight shape {gate_w.shape} expected ({e_ct}, {h})")
        gate_w_T = np.ascontiguousarray(
            gate_w.astype(np.float32, copy=False).T)
        gate_const = network.add_constant((h, e_ct), gate_w_T)
        mm = network.add_matrix_multiply(
            router_f32.get_output(0),
            trt.MatrixOperation.NONE,
            gate_const.get_output(0),
            trt.MatrixOperation.NONE,
        )
        router_logits = mm.get_output(0)
        # NemotronHTopkRouter has no bias
        bias = getattr(module.gate, "bias", None)
        if bias is not None:
            b_np = np.ascontiguousarray(
                bias.detach().float().cpu().numpy().reshape(1, e_ct).astype(
                    np.float32))
            bias_const = network.add_constant((1, e_ct), b_np)
            router_logits = network.add_elementwise(
                router_logits, bias_const.get_output(0),
                trt.ElementWiseOperation.SUM).get_output(0)

    # New FE buffer names: _stacked_{up,down}_{weights,block_scale,block_scale_decode,global_scale}
    up_q = np.ascontiguousarray(
        module._stacked_up_weights.detach().cpu().numpy().astype(np.int8,
                                                                 copy=False))
    up_bs = np.ascontiguousarray(
        module._stacked_up_block_scale.detach().cpu().numpy().astype(
            np.int8, copy=False))
    up_gs = np.ascontiguousarray(
        module._stacked_up_global_scale.detach().cpu().numpy().astype(
            np.float32, copy=False))
    dn_q = np.ascontiguousarray(
        module._stacked_down_weights.detach().cpu().numpy().astype(np.int8,
                                                                   copy=False))
    dn_bs = np.ascontiguousarray(
        module._stacked_down_block_scale.detach().cpu().numpy().astype(
            np.int8, copy=False))
    dn_gs = np.ascontiguousarray(
        module._stacked_down_global_scale.detach().cpu().numpy().astype(
            np.float32, copy=False))
    up_bs_decode = np.ascontiguousarray(
        module._stacked_up_block_scale_decode.detach().cpu().numpy().astype(
            np.int8, copy=False))
    dn_bs_decode = np.ascontiguousarray(
        module._stacked_down_block_scale_decode.detach().cpu().numpy().astype(
            np.int8, copy=False))
    correction_bias = getattr(module, "_e_score_correction_bias_fp32", None)
    if correction_bias is None:
        correction_bias_np = np.zeros(e_ct, dtype=np.float32)
    else:
        correction_bias_np = np.ascontiguousarray(
            correction_bias.detach().cpu().numpy().astype(np.float32,
                                                          copy=False))

    c_up_q = network.add_constant(tuple(up_q.shape), up_q).get_output(0)
    c_up_bs = network.add_constant(tuple(up_bs.shape), up_bs).get_output(0)
    c_up_gs = network.add_constant(tuple(up_gs.shape), up_gs).get_output(0)
    c_dn_q = network.add_constant(tuple(dn_q.shape), dn_q).get_output(0)
    c_dn_bs = network.add_constant(tuple(dn_bs.shape), dn_bs).get_output(0)
    c_dn_gs = network.add_constant(tuple(dn_gs.shape), dn_gs).get_output(0)
    c_up_bs_decode = network.add_constant(tuple(up_bs_decode.shape),
                                          up_bs_decode).get_output(0)
    c_dn_bs_decode = network.add_constant(tuple(dn_bs_decode.shape),
                                          dn_bs_decode).get_output(0)
    c_score_bias = network.add_constant(tuple(correction_bias_np.shape),
                                        correction_bias_np).get_output(0)

    plugin_inputs = [
        router_logits,
        hidden_in,
        hidden_gs_in,
        c_up_q,
        c_up_bs,
        c_up_gs,
        c_dn_q,
        c_dn_bs,
        c_dn_gs,
        c_up_bs_decode,
        c_dn_bs_decode,
        c_score_bias,
    ]
    # TensorRT 10.x: ``add_plugin_v3(inputs, shape_inputs, plugin)`` — no shape tensors for Nvfp4MoePlugin.
    moe_layer = network.add_plugin_v3(plugin_inputs, [], plugin)

    out_t = moe_layer.get_output(0)
    out_t.name = "output"
    network.mark_output(out_t)

    profile = builder.create_optimization_profile()
    # ``seq_max_for_profile`` widens the profile's max bound past the runtime
    # ``s``; default keeps min == opt == max.
    s_max = int(seq_max_for_profile) if seq_max_for_profile is not None else s
    if s_max < s:
        raise ValueError(f"seq_max_for_profile ({s_max}) must be >= seq ({s})")
    profile.set_shape("hidden_states", (b, s, h), (b, s, h), (b, s_max, h))
    profile.set_shape("hidden_global_scale", (2, ), (2, ), (2, ))
    if explicit_router_logits:
        rt_shape = (b * s, e_ct)
        rt_shape_max = (b * s_max, e_ct)
        profile.set_shape("router_logits", rt_shape, rt_shape, rt_shape_max)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.add_optimization_profile(profile)

    if verbose:
        print(
            f"[TRT API{tag}] build_serialized_network (workspace limit 1 GiB) …"
        )
    t1 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    if not isinstance(serialized, (bytes, bytearray)):
        serialized = bytes(serialized)
    if verbose:
        mib = len(serialized) / (1024 * 1024)
        print(
            f"[TRT API{tag}] Engine ready: {len(serialized)} bytes ({mib:.2f} MiB) "
            f"in {time.perf_counter() - t1:.2f}s")
    return serialized


def serialize_nvfp4_moe_engine_with_explicit_router(
    mod_cpu: NemotronHMoEMLP,
    *,
    batch: int,
    seq: int,
    logger,
    router_logits_np: np.ndarray,
    verbose: bool = False,
    seq_max_for_profile: int | None = None,
) -> bytes:
    """Engine blob with router as runtime input (``explicit_router_logits=True``)."""
    num_tokens = int(batch * seq)
    e_ct = int(mod_cpu.n_routed_experts)
    r_logits = np.ascontiguousarray(
        np.asarray(router_logits_np, dtype=np.float32))
    if tuple(r_logits.shape) != (num_tokens, e_ct):
        raise ValueError(
            f"router_logits_np shape {tuple(r_logits.shape)} != (num_tokens, num_experts)=({num_tokens}, {e_ct})"
        )

    h = int(mod_cpu.hidden_size)
    dummy_hidden_fp16 = torch.zeros(batch,
                                    seq,
                                    h,
                                    dtype=torch.float16,
                                    device="cpu")
    return build_nvfp4_moe_engine_trt_api(
        mod_cpu,
        dummy_hidden_fp16,
        logger,
        verbose=verbose,
        label="nvfp4-moe-explicit-router",
        explicit_router_logits=True,
        seq_max_for_profile=seq_max_for_profile,
    )


def run_nvfp4_w4a16_moe_plugin_accuracy_case(
    *,
    device: torch.device,
    top_k: int,
    moe_seed: int,
    expert: np.ndarray,
    score: np.ndarray,
    hidden_seed: int,
    hidden_on_cuda: bool,
    trt_execute_label: str,
    case: str,
) -> None:
    """One accuracy scenario: refs + TRT vs NumPy Marlin; optional :data:`PRINT_TRT_VS_TORCH_DIST`."""
    check_requirements()
    assert trt is not None
    batch = _NVFP4_MOE_FAST_BATCH
    expert = np.asarray(expert, dtype=np.int32)
    score = np.asarray(score, dtype=np.float32)
    assert expert.shape == score.shape
    num_tokens, rk = expert.shape
    tk = int(top_k)
    assert rk == tk, f"expert columns {rk} must equal top_k {tk}"
    assert num_tokens % batch == 0, (
        f"expert row count {num_tokens} must be divisible by batch={batch}")
    seq = num_tokens // batch
    h, inter, e_ct = _NVFP4_MOE_FAST_HIDDEN, _NVFP4_MOE_FAST_INTER, _NVFP4_MOE_FAST_EXPERTS

    # HF reference model — single source of truth for FP32 weights
    moe = create_toy_moe(
        device,
        hidden_size=h,
        moe_inter_size=inter,
        num_experts=e_ct,
        top_k=tk,
        seed=moe_seed,
    )

    # New FE: build NemotronHMoEMLP populated from the HF model's actual weights
    cfg = ModelConfig(
        model_type="nemotron_h",
        hidden_size=h,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=inter,
        head_dim=h,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        n_routed_experts=e_ct,
        num_experts_per_tok=tk,
        moe_intermediate_size=inter,
        moe_shared_expert_intermediate_size=inter,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        quant=QuantConfig(quant_type=QUANT_NVFP4, group_size=16),
    )
    mod = NemotronHMoEMLP(cfg, module_prefix="backbone.layers.0.mixer")
    mod.eval()
    # Extract HF expert weights: up_proj [E, I, H], down_proj [E, H, I]
    w_up_eih = moe.experts.up_proj.data.detach().cpu().float()  # [E, I, H]
    w_down_ehi = moe.experts.down_proj.data.detach().cpu().float()  # [E, H, I]
    for i, exp_mod in enumerate(mod.experts):
        _populate_nvfp4_linear(exp_mod.up_proj, w_up_eih[i].contiguous())
        _populate_nvfp4_linear(exp_mod.down_proj, w_down_ehi[i].contiguous())
    # Gate weights — NemotronHTopkRouter stores as fp16
    std_gate = WeightsGenerator.gate_test_weight_std(h)
    gen_gate = torch.Generator()
    gen_gate.manual_seed(moe_seed)
    nn.init.normal_(mod.gate.weight,
                    mean=0.0,
                    std=std_gate,
                    generator=gen_gate)
    nn.init.normal_(mod.gate.e_score_correction_bias,
                    mean=0.0,
                    std=std_gate * 0.3,
                    generator=gen_gate)
    # Shared experts (not used in tests but prepare_for_export expects them)
    gen_shared = torch.Generator()
    gen_shared.manual_seed(moe_seed + 202)
    std_up = WeightsGenerator.expert_linear_std(h, 0.08)
    std_dn = WeightsGenerator.expert_linear_std(inter, 0.08)
    w_shared_up = torch.randn(h, inter, generator=gen_shared) * std_up
    w_shared_dn = torch.randn(inter, h, generator=gen_shared) * std_dn
    _populate_nvfp4_linear(mod.shared_experts.up_proj,
                           w_shared_up.T.contiguous())
    _populate_nvfp4_linear(mod.shared_experts.down_proj,
                           w_shared_dn.T.contiguous())
    mod.prepare_for_export()
    mod._plugin_activation_type = 0
    mod._plugin_routing_mode = 0

    logits_np = NemotronHMoEReference.router_logits_from_desired_topk(
        expert, score, num_tokens=num_tokens, num_experts=e_ct, top_k=tk)

    hidden_dev = device if hidden_on_cuda else None
    hidden_bsh = structured_noise_hidden_states_bsh(batch,
                                                    seq,
                                                    h,
                                                    seed=hidden_seed,
                                                    amp=2.5,
                                                    device=hidden_dev)

    mod_cpu = mod.cpu()
    topw_np, topi_np = NemotronHMoEReference.moe_topk_softmax_renormalize_numpy(
        logits_np, tk)
    hidden_np = np.ascontiguousarray(hidden_bsh.cpu().numpy())

    # Build Q/DQ reference from N-major buffers with prefill atom-layout scales
    # (quantize→dequantize eliminates FP4 rounding from the comparison).
    act_type = 0  # ReLU² for NemotronH
    plugin_inter = int(mod_cpu.moe_intermediate_size)
    w_up_qdq, w_down_qdq = NemotronHMoEReference.dense_weights_from_nvfp4_nmajor_buffers(
        mod_cpu._stacked_up_weights.numpy(),
        mod_cpu._stacked_up_block_scale.numpy(),
        mod_cpu._stacked_down_weights.numpy(),
        mod_cpu._stacked_down_block_scale.numpy(),
        num_experts=e_ct,
        hidden_size=h,
        moe_inter_size=plugin_inter,
        activation_type=act_type,
    )
    up_gs_np = mod_cpu._stacked_up_global_scale.numpy().reshape(-1)
    dn_gs_np = mod_cpu._stacked_down_global_scale.numpy().reshape(-1)
    for ex in range(e_ct):
        w_up_qdq[ex] *= float(up_gs_np[ex])
        w_down_qdq[ex] *= float(dn_gs_np[ex])

    # Patch Q/DQ weights into the HF experts module and call its forward().
    # up_proj: (E, I, H), down_proj: (E, H, I) — transposed from Q/DQ layout.
    with torch.no_grad():
        moe.experts.up_proj.data.copy_(
            torch.from_numpy(np.ascontiguousarray(w_up_qdq.transpose(
                0, 2, 1))).to(moe.experts.up_proj.data))
        moe.experts.down_proj.data.copy_(
            torch.from_numpy(
                np.ascontiguousarray(w_down_qdq.transpose(0, 2, 1))).to(
                    moe.experts.down_proj.data))

    expert_dtype = moe.experts.up_proj.dtype  # fp16
    x_fp16 = torch.from_numpy(
        np.asarray(hidden_np, dtype=np.float16).reshape(-1, h)).to(
            device=moe.experts.up_proj.device, dtype=expert_dtype)
    topi_t = torch.from_numpy(topi_np.astype(np.int64)).to(
        moe.experts.up_proj.device)
    topw_t = torch.from_numpy(topw_np).to(device=moe.experts.up_proj.device,
                                          dtype=torch.float32)
    with torch.no_grad():
        ref_torch = moe.experts(x_fp16, topi_t, topw_t)
    ref_fp32 = ref_torch.detach().cpu().numpy().astype(np.float32).reshape(
        batch, seq, h)

    logger = trt.Logger(trt.Logger.WARNING)
    load_nvfp4_moe_edge_llm_plugins(logger, _PLUGIN_SO, verbose=False)
    eng = serialize_nvfp4_moe_engine_with_explicit_router(
        mod_cpu,
        batch=batch,
        seq=seq,
        logger=logger,
        router_logits_np=logits_np,
        verbose=False,
    )

    stream = torch.cuda.Stream(device=device)
    # Plugin v2 expects hidden_global_scale at the engine I/O. Decode (this test's regime) does
    # not consume it, so `1.0` is a safe no-op placeholder; prefill tests would pass calibrated
    # FC1/FC2 activation forward scales here.  Plugin v3 unified weight layout: all token
    # counts route through the prefill (CuteDSL) path, so the hidden_global_scale values
    # are consumed for activation fp4Quantize inside the plugin.  Calibrate from hidden.
    hidden_max_abs = float(np.max(np.abs(hidden_np)))
    act_gs_fc1 = max(1e-6, hidden_max_abs / (448.0 * 6.0))
    # Rough FC2 calibration: fc1 output bounded by ~topK × max|w_up| × hidden_max.
    # For these small toy cases using ones(2) would under-scale; a loose upper bound keeps
    # FP8 block scales in range.
    w_up_amax = float(moe.experts.up_proj.data.abs().amax().item())
    fc1_bound = max(1e-6, hidden_max_abs * w_up_amax * float(h))
    act_gs_fc2 = max(1e-6, fc1_bound / (448.0 * 6.0))
    hidden_gs_np = np.array([act_gs_fc1, act_gs_fc2], dtype=np.float32)
    trt_out = execute_trt_engine(
        eng,
        {
            "router_logits": np.ascontiguousarray(logits_np.astype(
                np.float32)),
            "hidden_states": np.ascontiguousarray(hidden_np.astype(
                np.float16)),
            "hidden_global_scale": np.ascontiguousarray(hidden_gs_np),
        },
        stream,
        device=device,
        verbose=False,
        label=trt_execute_label,
    )

    # Compare TRT output against Q/DQ reference (FP4 quantize→dequantize weights).
    trt_np = np.asarray(trt_out, dtype=np.float32).reshape(*ref_fp32.shape)
    med_cos, mean_cos, min_cos = _cosine_sim_per_row(trt_np, ref_fp32)
    mag = _magnitude_ratio(trt_np, ref_fp32)
    print(
        f"[w4a16 {case}] numTokens={num_tokens} topK={tk} median_cos={med_cos:.4f} "
        f"mean_cos={mean_cos:.4f} min_cos={min_cos:.4f} mag_ratio={mag:.3f} "
        f"|trt|={float(np.linalg.norm(trt_np)):.4g} |ref|={float(np.linalg.norm(ref_fp32)):.4g}"
    )
    if med_cos < 0.99:
        raise AssertionError(
            f"{case}: median cosine vs Q/DQ ref {med_cos:.4f} < 0.99 "
            f"(min_cos={min_cos:.4f}, mag={mag:.3f})")
    if not (0.5 <= mag <= 2.0):
        raise AssertionError(
            f"{case}: magnitude ratio {mag:.3f} outside [0.5, 2.0]")


@pytest.mark.parametrize(
    (
        "top_k",
        "moe_seed",
        "expert_rows",
        "score_rows",
        "hidden_seed",
        "hidden_on_cuda",
        "trt_execute_label",
        "case",
    ),
    [
        pytest.param(
            1,
            2026,
            [[0], [5]],
            [[1.0], [1.0]],
            91021,
            True,
            "nvfp4-moe-realistic-b2s1-tk1",
            "realistic_b2s1_h384_e8_topk1",
            id="topk1",
        ),
        pytest.param(
            2,
            2027,
            [[0, 1], [6, 3]],
            [[0.62, 0.38], [0.55, 0.45]],
            91022,
            False,
            "nvfp4-moe-realistic-b2s1-tk2",
            "realistic_b2s1_h384_e8_topk2",
            id="topk2",
        ),
        pytest.param(
            4,
            2028,
            [[0, 1, 2, 3], [7, 5, 4, 6]],
            [[0.40, 0.28, 0.20, 0.12], [0.36, 0.30, 0.22, 0.12]],
            91023,
            True,
            "nvfp4-moe-realistic-b2s1-tk4",
            "realistic_b2s1_h384_e8_topk4",
            id="topk4",
        ),
        pytest.param(
            2,
            2029,
            [[0, 1], [6, 3], [4, 5], [2, 0]],
            [[0.62, 0.38], [0.55, 0.45], [0.50, 0.50], [0.70, 0.30]],
            91024,
            False,
            "nvfp4-moe-realistic-b2s2-tk2",
            "realistic_b2s2_h384_e8_topk2",
            id="b2s2_topk2",
        ),
    ],
)
def test_nvfp4_w4a16_moe_plugin_accuracy(
    top_k: int,
    moe_seed: int,
    expert_rows: list[list[int]],
    score_rows: list[list[float]],
    hidden_seed: int,
    hidden_on_cuda: bool,
    trt_execute_label: str,
    case: str,
) -> None:
    """Parametrized plugin accuracy (B=2 fixed; S=len(expert_rows)/B; H=384,I=768,E=8)."""
    check_requirements()
    dev = torch.device("cuda", torch.cuda.current_device())
    expert = np.asarray(expert_rows, dtype=np.int32)
    score = np.asarray(score_rows, dtype=np.float32)
    run_nvfp4_w4a16_moe_plugin_accuracy_case(
        device=dev,
        top_k=int(top_k),
        moe_seed=int(moe_seed),
        expert=expert,
        score=score,
        hidden_seed=int(hidden_seed),
        hidden_on_cuda=bool(hidden_on_cuda),
        trt_execute_label=trt_execute_label,
        case=case,
    )


def _test_nvfp4_w4a16_moe_plugin_accuracy_sf_padding_non128_aligned() -> None:
    """Regression: hidden_size=1856 (not 128-aligned) exercises Cutlass Atom SF padding.

    Disabled on this branch: the reference helper ``dense_weights_from_nvfp4_plugin_buffers``
    asserts K-major weight shape ``[E, H/2, I]`` (main's layout), but this branch stores
    weights N-major ``[E, H, I/2]``. Re-enable once layouts are reconciled at merge.
    """
    check_requirements()
    assert trt is not None
    dev = torch.device("cuda", torch.cuda.current_device())
    batch = 1
    seq = 1
    h = 1856  # NOT a multiple of 128 (1856 = 128*14 + 64)
    inter = 2688  # multiple of 128
    e_ct = 2  # minimal experts to keep pure-Python packing loops fast
    tk = 1
    moe_seed = 5050
    hidden_seed = 91050

    expert = np.array([[0]], dtype=np.int32)
    score = np.array([[1.0]], dtype=np.float32)

    mod, _, _ = _create_toy_nemotron_h_moe_mlp(
        hidden_size=h,
        moe_inter_size=inter,
        num_experts=e_ct,
        top_k=tk,
        seed=moe_seed,
    )
    mod._plugin_activation_type = 0
    mod._plugin_routing_mode = 0

    logits_np = NemotronHMoEReference.router_logits_from_desired_topk(
        expert, score, num_tokens=batch * seq, num_experts=e_ct, top_k=tk)

    hidden_bsh = structured_noise_hidden_states_bsh(batch,
                                                    seq,
                                                    h,
                                                    seed=hidden_seed,
                                                    amp=2.5,
                                                    device=dev)

    mod_cpu = mod.cpu()
    topw_np, topi_np = NemotronHMoEReference.moe_topk_softmax_renormalize_numpy(
        logits_np, tk)
    hidden_np = np.ascontiguousarray(hidden_bsh.cpu().numpy())

    ref_marlin_np = NemotronHMoEReference.reference_from_packed_plugin_state(
        hidden_np,
        topw_np,
        topi_np,
        mod_cpu._stacked_up_weights.numpy(),
        mod_cpu._stacked_up_block_scale.numpy(),
        mod_cpu._stacked_down_weights.numpy(),
        mod_cpu._stacked_down_block_scale.numpy(),
        mod_cpu._stacked_up_global_scale.numpy(),
        mod_cpu._stacked_down_global_scale.numpy(),
        hidden_size=h,
        moe_inter_size=inter,
        num_experts=e_ct,
        activation_type=0,
    )

    logger = trt.Logger(trt.Logger.WARNING)
    load_nvfp4_moe_edge_llm_plugins(logger, _PLUGIN_SO, verbose=False)
    eng = serialize_nvfp4_moe_engine_with_explicit_router(
        mod_cpu,
        batch=batch,
        seq=seq,
        logger=logger,
        router_logits_np=logits_np,
        verbose=False,
    )

    stream = torch.cuda.Stream(device=dev)
    trt_out = execute_trt_engine(
        eng,
        {
            "router_logits": np.ascontiguousarray(logits_np.astype(
                np.float32)),
            "hidden_states": np.ascontiguousarray(hidden_np.astype(
                np.float16)),
        },
        stream,
        device=dev,
        verbose=False,
        label="nvfp4-moe-sf-padding-non128",
    )

    NemotronHMoEReference.assert_trt_output_matches_nvfp4_marlin_numpy_reference(
        trt_out,
        ref_marlin_np.astype(np.float32),
        "sf_padding_non128_h1856_i2688",
        tol_scale=_NVFP4_MOE_TRT_REF_TOL_SCALE,
        max_outlier_frac=0.003,
    )


# ============================================================================
# Decode-path routing-mode-1 (sigmoid group top-k) smoke test.
# The existing ``test_nvfp4_w4a16_moe_plugin_accuracy`` parametrizations all
# run with ``routing_mode=0`` (``moeTopkSoftmax``). This smoke test forces
# ``routing_mode=1`` (``moeSigmoidGroupTopk``) on the decode path and asserts
# the TRT output is finite with non-zero magnitude, validating that the dual-
# mode routing selector reaches the decode dispatch without crashing and that
# both decode SF slots are plumbed through the sigmoid-group-topk branch.
# Full numerical reference would require re-implementing sigmoid-group-topk +
# renormalize + routed_scaling here; the C++ unit tests cover the decode
# kernel accuracy independent of routing mode.
# ============================================================================


@pytest.mark.parametrize(
    ("batch", "seq", "top_k", "n_group", "topk_group"),
    [
        pytest.param(1, 1, 1, 1, 1, id="decode_rm1_b1s1_tk1"),
        pytest.param(2, 1, 2, 2, 1, id="decode_rm1_b2s1_tk2"),
        pytest.param(2, 8, 2, 2, 1, id="decode_rm1_b2s8_tk2"),
    ],
)
def test_nvfp4_moe_plugin_decode_routing_mode_1_smoke(
    batch: int,
    seq: int,
    top_k: int,
    n_group: int,
    topk_group: int,
) -> None:
    """Decode path with ``routing_mode=1`` (``moeSigmoidGroupTopk``): smoke test that
    output is finite and has non-trivial magnitude. Numerical accuracy of the sigmoid-
    group-topk router is covered by ``moeSigmoidGroupTopkKernelTests``; this test
    validates the **plugin wiring** of the dual-mode selector on the decode dispatch."""
    check_requirements()
    assert trt is not None
    dev = torch.device("cuda", torch.cuda.current_device())

    num_tokens = int(batch) * int(seq)
    if num_tokens > 16:
        raise ValueError(
            f"decode-case expects numTokens <= 16 to exercise the dispatch; got {num_tokens}"
        )
    h, inter, e_ct = _NVFP4_MOE_FAST_HIDDEN, _NVFP4_MOE_FAST_INTER, _NVFP4_MOE_FAST_EXPERTS

    mod, _, _ = _create_toy_nemotron_h_moe_mlp(
        hidden_size=h,
        moe_inter_size=inter,
        num_experts=e_ct,
        top_k=top_k,
        seed=20262,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    mod._plugin_activation_type = 0
    mod._plugin_routing_mode = 1

    hidden_bsh = structured_noise_hidden_states_bsh(batch,
                                                    seq,
                                                    h,
                                                    seed=7777,
                                                    amp=1.5,
                                                    device=dev)
    hidden_np = np.ascontiguousarray(hidden_bsh.cpu().numpy())
    mod_cpu = mod.cpu()

    # Sigmoid-group-topk uses hidden @ gate^T + e_score_correction_bias (optional).
    # Compute router logits explicitly so we can feed them as a runtime input.
    gate_w = getattr(mod_cpu, "gate").weight.detach().cpu().float().numpy()
    logits_np = hidden_np.reshape(num_tokens, h).astype(
        np.float32) @ gate_w.T.astype(np.float32)

    logger = trt.Logger(trt.Logger.WARNING)
    load_nvfp4_moe_edge_llm_plugins(logger, _PLUGIN_SO, verbose=False)
    eng = serialize_nvfp4_moe_engine_with_explicit_router(
        mod_cpu,
        batch=batch,
        seq=seq,
        logger=logger,
        router_logits_np=logits_np,
        verbose=False,
    )

    stream = torch.cuda.Stream(device=dev)
    hidden_gs_np = np.array([1.0, 1.0], dtype=np.float32)
    trt_out = execute_trt_engine(
        eng,
        {
            "router_logits": np.ascontiguousarray(logits_np),
            "hidden_states": np.ascontiguousarray(hidden_np.astype(
                np.float16)),
            "hidden_global_scale": hidden_gs_np,
        },
        stream,
        device=dev,
        verbose=False,
        label=f"rm1-decode-b{batch}s{seq}-tk{top_k}",
    )
    trt_np = np.asarray(trt_out, dtype=np.float32).reshape(batch, seq, h)
    assert np.all(np.isfinite(trt_np)), (
        f"routing_mode=1 decode: TRT output contains non-finite values")
    l2 = float(np.linalg.norm(trt_np))
    assert l2 > 1e-6, (
        f"routing_mode=1 decode: TRT output is (near-)zero (|out|={l2:.3e}); "
        "likely the sigmoid-group-topk path short-circuited every expert")
    print(
        f"[decode rm=1 b{batch}s{seq}tk{top_k}] |out|={l2:.4g} (smoke: finite + non-zero)"
    )


# ============================================================================
# Prefill-path end-to-end accuracy tests (plugin v2 dispatches here when
# ``numTokens = B*S > 16``).  These exercise the CuteDSL grouped-GEMM pipeline
# (K0..K5) rather than the per-token decode GEMVs.
# ============================================================================


def _cosine_sim_per_row(a: np.ndarray,
                        b: np.ndarray) -> tuple[float, float, float]:
    """Per-row cosine similarity → (median, mean, min).  Matches the reference
    NvFP4-MoE repo's Tier-1/Tier-2 metric (tests/gpu/test_fp16_e2e_nvfp4_moe.py).
    NaN rows (zero vector on either side) are excluded from the aggregate."""
    a = a.astype(np.float64).reshape(a.shape[0] * a.shape[1], -1)
    b = b.astype(np.float64).reshape(b.shape[0] * b.shape[1], -1)
    num = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(denom > 0.0, num / np.maximum(denom, 1e-30), 1.0)
    cos = cos[np.isfinite(cos)]
    if cos.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.median(cos)), float(np.mean(cos)), float(np.min(cos))


def _magnitude_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """|trt|_F / |ref|_F (reference repo's ``mag_ratio``)."""
    na = float(np.linalg.norm(a.astype(np.float64)))
    nb = float(np.linalg.norm(b.astype(np.float64)))
    return na / max(nb, 1e-30)


def _run_nvfp4_moe_plugin_prefill_accuracy_case(
    *,
    device: torch.device,
    batch: int,
    seq: int,
    top_k: int,
    num_experts: int,
    hidden_size: int,
    moe_inter_size: int,
    moe_seed: int,
    hidden_seed: int,
    hidden_amp: float,
    weight_scale: float,
    cos_threshold: float,
    mag_ratio_lo: float,
    mag_ratio_hi: float,
    case: str,
    seq_max_for_profile: int | None = None,
) -> None:
    """Build a toy NVFP4 MoE plugin engine, run it with calibrated ``hidden_global_scale``
    on a ``numTokens = B*S`` input that forces the prefill dispatch, and compare the output
    against an FP32 reference built from the HF (non-quantized) experts.

    Accuracy metric matches the reference NvFP4-MoE repo
    (``tests/gpu/test_fp16_e2e_nvfp4_moe.py``): per-row cosine similarity vs a Tier-1 FP32
    reference, plus a magnitude-ratio bound on ``|trt|_F / |ref|_F``.  Tier-1 threshold is
    cos > 0.95 (multi-stage FP4 quantization makes tighter bounds unrealistic); mag_ratio
    [0.5, 2.0] follows the reference's Tier-2 range since our toy inputs stay within the
    FP4 quantization regime's normal-magnitude band.
    """
    check_requirements()
    assert trt is not None

    num_tokens = int(batch) * int(seq)
    if num_tokens <= 16:
        raise ValueError(
            f"prefill-case expects numTokens > 16 to exercise the dispatch; got {num_tokens}"
        )

    moe = create_toy_moe(
        device,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
        num_experts=num_experts,
        top_k=top_k,
        seed=moe_seed,
    )
    # Scale down the expert weights — the prefill path quantizes FC1 output to NVFP4 and
    # unbounded toy weights push FC1 outputs past the E4M3 block-scale range we calibrate for.
    with torch.no_grad():
        moe.experts.up_proj.data.mul_(weight_scale)
        moe.experts.down_proj.data.mul_(weight_scale)

    # New FE: build NemotronHMoEMLP populated from the HF moe's (scaled) weights
    h_sz = int(hidden_size)
    cfg = ModelConfig(
        model_type="nemotron_h",
        hidden_size=h_sz,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=moe_inter_size,
        head_dim=h_sz,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        n_routed_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter_size,
        moe_shared_expert_intermediate_size=moe_inter_size,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        quant=QuantConfig(quant_type=QUANT_NVFP4, group_size=16),
    )
    mod = NemotronHMoEMLP(cfg, module_prefix="backbone.layers.0.mixer")
    mod.eval()
    # Extract HF expert weights: up_proj [E, I, H], down_proj [E, H, I]
    w_up_eih = moe.experts.up_proj.data.detach().cpu().float()  # [E, I, H]
    w_down_ehi = moe.experts.down_proj.data.detach().cpu().float()  # [E, H, I]
    for i, expert in enumerate(mod.experts):
        _populate_nvfp4_linear(expert.up_proj, w_up_eih[i].contiguous())
        _populate_nvfp4_linear(expert.down_proj, w_down_ehi[i].contiguous())
    # Gate weights — NemotronHTopkRouter stores as fp16
    torch.manual_seed(moe_seed)
    std_gate = WeightsGenerator.gate_test_weight_std(h_sz)
    gen_gate = torch.Generator()
    gen_gate.manual_seed(moe_seed)
    nn.init.normal_(mod.gate.weight,
                    mean=0.0,
                    std=std_gate,
                    generator=gen_gate)
    nn.init.normal_(mod.gate.e_score_correction_bias,
                    mean=0.0,
                    std=std_gate * 0.3,
                    generator=gen_gate)
    # Shared experts (not used in tests but prepare_for_export expects them)
    gen_shared = torch.Generator()
    gen_shared.manual_seed(moe_seed + 202)
    std_up = WeightsGenerator.expert_linear_std(h_sz, 0.08)
    std_dn = WeightsGenerator.expert_linear_std(moe_inter_size, 0.08)
    w_shared_up = torch.randn(h_sz, moe_inter_size,
                              generator=gen_shared) * std_up
    w_shared_dn = torch.randn(moe_inter_size, h_sz,
                              generator=gen_shared) * std_dn
    _populate_nvfp4_linear(mod.shared_experts.up_proj,
                           w_shared_up.T.contiguous())
    _populate_nvfp4_linear(mod.shared_experts.down_proj,
                           w_shared_dn.T.contiguous())
    mod.prepare_for_export()
    mod._plugin_activation_type = 0
    mod._plugin_routing_mode = 0

    # Router logits feed one token per row.  We pick uniformly-random distinct experts per
    # token so the layout builder exercises multiple groups, including some zero-hit ones
    # when num_tokens is small relative to num_experts.
    rng = np.random.default_rng(12345 + moe_seed)
    expert_rows = rng.integers(low=0,
                               high=num_experts,
                               size=(num_tokens, top_k),
                               dtype=np.int64)
    # Deduplicate within each row: moe_topk_softmax_renormalize_numpy assumes unique indices
    # per token; overlap would just be redundant entries.
    for t in range(num_tokens):
        used: set[int] = set()
        for k in range(top_k):
            e = int(expert_rows[t, k])
            while e in used:
                e = (e + 1) % num_experts
            used.add(e)
            expert_rows[t, k] = e
    score_rows = np.ones((num_tokens, top_k), dtype=np.float32) / float(top_k)
    logits_np = NemotronHMoEReference.router_logits_from_desired_topk(
        expert_rows.astype(np.int32),
        score_rows,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k)

    hidden_bsh = structured_noise_hidden_states_bsh(batch,
                                                    seq,
                                                    hidden_size,
                                                    seed=hidden_seed,
                                                    amp=hidden_amp,
                                                    device=device)
    hidden_np = np.ascontiguousarray(hidden_bsh.cpu().numpy())

    # Calibrate ``hidden_global_scale``:
    #   [0] forward FC1 activation GS = max|hidden| / (448 * 6).
    #   [1] forward FC2 activation GS = max|fc1_out_estimate| / (448 * 6).
    # For the toy model we pre-run the FP32 reference FC1 on CPU to bound fc1 output magnitude.
    fc1_max = 0.0
    x_flat = hidden_np.reshape(num_tokens, hidden_size).astype(np.float32)
    w_up_e, _ = NemotronHMoEReference.hf_expert_weights_numpy_ehi_eih(moe)
    for t in range(num_tokens):
        for slot in range(top_k):
            ex = int(expert_rows[t, slot])
            z = x_flat[t] @ w_up_e[ex]
            activation_type = 0  # ReLU² for NemotronH
            a = _moe_activation_numpy(z, activation_type)
            fc1_max = max(fc1_max, float(np.max(np.abs(a))))
    act_gs_fc1 = max(1e-6, float(np.max(np.abs(hidden_np))) / (448.0 * 6.0))
    act_gs_fc2 = max(1e-6, fc1_max / (448.0 * 6.0))
    hidden_gs_np = np.asarray([act_gs_fc1, act_gs_fc2], dtype=np.float32)

    mod_cpu = mod.cpu()
    topw_np, topi_np = NemotronHMoEReference.moe_topk_softmax_renormalize_numpy(
        logits_np, top_k)
    ref_np = NemotronHMoEReference.reference_dense_from_hf_moe(
        moe,
        hidden_np,
        topw_np,
        topi_np,
        hidden_size=hidden_size,
        activation_type=0)

    logger = trt.Logger(trt.Logger.WARNING)
    load_nvfp4_moe_edge_llm_plugins(logger, _PLUGIN_SO, verbose=False)
    eng = serialize_nvfp4_moe_engine_with_explicit_router(
        mod_cpu,
        batch=batch,
        seq=seq,
        logger=logger,
        router_logits_np=logits_np,
        verbose=False,
        seq_max_for_profile=seq_max_for_profile)

    stream = torch.cuda.Stream(device=device)
    exec_inputs = {
        "router_logits": np.ascontiguousarray(logits_np.astype(np.float32)),
        "hidden_states": np.ascontiguousarray(hidden_np.astype(np.float16)),
        "hidden_global_scale": np.ascontiguousarray(hidden_gs_np),
    }
    trt_out = execute_trt_engine(
        eng,
        exec_inputs,
        stream,
        device=device,
        verbose=False,
        label=f"prefill-{case}",
    )
    trt_np = np.asarray(trt_out,
                        dtype=np.float32).reshape(batch, seq, hidden_size)

    # Plugin v1 allocates per-expert α buffers in ``attachToContext`` and
    # initializes them once on the first prefill enqueue; subsequent calls
    # must reuse the already-populated buffer. Re-run the same engine twice
    # more with identical inputs and assert the repeat outputs stay **very**
    # close to the first call — the FP4·FP4 MMA + atomic scatter-reduce
    # doesn't provide bit-level determinism (float accumulator ordering is
    # warp-schedule-dependent), but the drift is ULP-scale. A real α-buffer
    # corruption, re-init with wrong data, or aliasing would produce a much
    # larger delta than this tolerance.
    repeat_median_cos_floor = 0.99999
    repeat_rel_max_abs_tol = 1e-3  # |Δ|_∞ / |ref|_F
    trt_norm_abs = float(np.linalg.norm(trt_np.astype(np.float64)))
    for repeat_idx in range(2):
        trt_out_repeat = execute_trt_engine(
            eng,
            exec_inputs,
            stream,
            device=device,
            verbose=False,
            label=f"prefill-{case}-repeat{repeat_idx + 1}",
        )
        trt_np_repeat = np.asarray(trt_out_repeat, dtype=np.float32).reshape(
            batch, seq, hidden_size)
        if not np.all(np.isfinite(trt_np_repeat)):
            raise AssertionError(
                f"{case}: enqueue repeat {repeat_idx + 1} produced non-finite output"
            )
        repeat_max_abs = float(np.max(np.abs(trt_np - trt_np_repeat)))
        rel = repeat_max_abs / max(trt_norm_abs, 1e-30)
        repeat_median_cos, _, repeat_min_cos = _cosine_sim_per_row(
            trt_np, trt_np_repeat)
        if repeat_median_cos < repeat_median_cos_floor or rel > repeat_rel_max_abs_tol:
            raise AssertionError(
                f"{case}: enqueue repeat {repeat_idx + 1} diverged from first "
                f"call beyond tolerance (repeat_median_cos={repeat_median_cos:.6f} "
                f"< {repeat_median_cos_floor}, or |Δ|_∞/|ref|_F={rel:.3e} "
                f"> {repeat_rel_max_abs_tol}, |Δ|_∞={repeat_max_abs:.3e}). "
                "Likely a regression in the persistent α buffer / first-enqueue "
                "init logic — kernel non-determinism should stay well below "
                "this bound.")
    ref_np = ref_np.reshape(batch, seq, hidden_size).astype(np.float32)

    if not np.all(np.isfinite(trt_np)):
        raise AssertionError(
            f"{case}: TRT prefill output contains non-finite values")

    median_cos, mean_cos, min_cos = _cosine_sim_per_row(trt_np, ref_np)
    mag_ratio = _magnitude_ratio(trt_np, ref_np)
    trt_norm = float(np.linalg.norm(trt_np.astype(np.float64)))
    ref_norm = float(np.linalg.norm(ref_np.astype(np.float64)))
    max_abs_diff = float(np.max(np.abs(trt_np - ref_np)))

    # FP4-simulated reference (cutedsl-style Tier-2 parity): the weights are
    # dequantized from the **actually-packed plugin buffers** so the sim sees
    # bit-identical FP4 bucket values to what the CuteDSL kernel reads, and
    # the FC2-input global scale is recomputed from the sim's own FC1 output
    # max (mirrors cutedsl's ``fc1_output_global_sf = None`` path, not the
    # plugin's pre-calibrated ``hidden_global_scale[1]``). The remaining cos
    # gap vs TRT is only MMA accumulator ordering + hardware ``rcpApproxFtz``
    # error, so a correctly computing kernel sits above ~0.99 here —
    # analogous to ``test_nemotron_tier2`` in cutedsl-nvfp4-moe.
    w_up_ehi_sim = NemotronHMoEReference.dequant_fp4_prefill_weight_fc1(
        mod_cpu._stacked_up_weights.numpy(),
        mod_cpu._stacked_up_block_scale.numpy(),
        mod_cpu._stacked_up_global_scale.numpy(),
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
    )
    w_down_eih_sim = NemotronHMoEReference.dequant_fp4_prefill_weight_fc2(
        mod_cpu._stacked_down_weights.numpy(),
        mod_cpu._stacked_down_block_scale.numpy(),
        mod_cpu._stacked_down_global_scale.numpy(),
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
    )
    ref_fp4_np = NemotronHMoEReference.fp4_simulated_dense_forward(
        x_flat,
        topw_np,
        topi_np,
        w_up_ehi_sim,
        w_down_eih_sim,
        activation_type=0,
        sf_scale_fc1=float(hidden_gs_np[0]),
        sf_scale_fc2=None,  # runtime-max from sim's FC1 output (cutedsl-style)
    )
    ref_fp4_np = ref_fp4_np.reshape(batch, seq, hidden_size).astype(np.float32)
    median_cos_fp4, mean_cos_fp4, min_cos_fp4 = _cosine_sim_per_row(
        trt_np, ref_fp4_np)
    mag_ratio_fp4 = _magnitude_ratio(trt_np, ref_fp4_np)

    print(
        f"[prefill {case}] numTokens={num_tokens} (B={batch},S={seq},topK={top_k}) "
        f"vs FP32 HF ref:  median_cos={median_cos:.4f} mean_cos={mean_cos:.4f} "
        f"min_cos={min_cos:.4f} mag_ratio={mag_ratio:.3f} max_abs_diff={max_abs_diff:.4g} "
        f"|trt|={trt_norm:.4g} |ref|={ref_norm:.4g}  "
        f"vs FP4-weights-sim ref: median_cos={median_cos_fp4:.4f} "
        f"min_cos={min_cos_fp4:.4f} mag_ratio={mag_ratio_fp4:.3f}  "
        f"hidden_gs=[{hidden_gs_np[0]:.3e}, {hidden_gs_np[1]:.3e}]")

    if median_cos < cos_threshold:
        raise AssertionError(
            f"{case}: median cosine similarity {median_cos:.4f} < threshold {cos_threshold}. "
            f"(min_cos={min_cos:.4f}, mean_cos={mean_cos:.4f}, mag_ratio={mag_ratio:.3f})"
        )
    # Soft sanity check on the FP4-simulated reference: TRT vs a dense forward
    # with the same FP4 weight buckets and the same per-16-block activation
    # quantization should be substantially tighter than the FP32 HF reference
    # because the weight-rounding error is shared. The CPU simulation here does
    # not exactly reproduce the plugin's internal FC2-input quantization (the
    # plugin quantizes post-scatter intermediates differently than the per-slot
    # roundtrip this reference does), so we assert a loose ``cos_threshold``
    # rather than a tight 0.999 bound — the primary use of this metric is as a
    # diagnostic when ``median_cos`` vs FP32 regresses, to distinguish kernel
    # arithmetic drift from natural FP4 quantization noise.
    if median_cos_fp4 < cos_threshold:
        raise AssertionError(
            f"{case}: median cosine vs FP4-sim ref {median_cos_fp4:.4f} < "
            f"threshold {cos_threshold}. (min_cos_fp4={min_cos_fp4:.4f}, "
            f"mag_ratio_fp4={mag_ratio_fp4:.3f})")
    if not (mag_ratio_lo <= mag_ratio <= mag_ratio_hi):
        raise AssertionError(
            f"{case}: |trt|_F / |ref|_F = {mag_ratio:.3f} is outside "
            f"[{mag_ratio_lo}, {mag_ratio_hi}] — suggests TRT output is saturated, zero, "
            f"or scaled incorrectly. (|trt|={trt_norm:.4g}, |ref|={ref_norm:.4g})"
        )


@pytest.mark.parametrize(
    ("batch", "seq", "top_k", "seq_max_for_profile", "case"),
    [
        # Just above the dispatch threshold — non-128-aligned numTokens (17*2=34).
        pytest.param(
            2, 17, 2, None, "prefill_b2_s17_tk2", id="prefill_b2_s17_tk2"),
        # Non-multiple of 128 to stress the row-padding path.
        pytest.param(
            2, 9, 2, None, "prefill_b2_s9_tk2", id="prefill_b2_s9_tk2"),
        # Single-batch large seq — exactly one 128-aligned tile boundary.
        pytest.param(
            1, 128, 2, None, "prefill_b1_s128_tk2", id="prefill_b1_s128_tk2"),
        # Mid-size non-aligned: T = 130, topK = 4 — multiple active groups, most experts hit.
        pytest.param(
            2, 65, 4, None, "prefill_b2_s65_tk4", id="prefill_b2_s65_tk4"),
        # Large 2048-token prefill (single batch) — stresses the grouped-GEMM with many
        # tile boundaries and all 8 experts saturated.
        pytest.param(1,
                     2048,
                     2,
                     None,
                     "prefill_b1_s2048_tk2",
                     id="prefill_b1_s2048_tk2"),
    ],
)
def test_nvfp4_moe_plugin_prefill_accuracy(
    batch: int,
    seq: int,
    top_k: int,
    seq_max_for_profile: int | None,
    case: str,
) -> None:
    """Plugin v2 prefill path (``numTokens > 16``): grouped-GEMM pipeline vs FP32 reference."""
    check_requirements()
    dev = torch.device("cuda", torch.cuda.current_device())
    _run_nvfp4_moe_plugin_prefill_accuracy_case(
        device=dev,
        batch=int(batch),
        seq=int(seq),
        top_k=int(top_k),
        seq_max_for_profile=seq_max_for_profile,
        num_experts=_NVFP4_MOE_FAST_EXPERTS,
        hidden_size=_NVFP4_MOE_FAST_HIDDEN,
        moe_inter_size=_NVFP4_MOE_FAST_INTER,
        moe_seed=3030,
        hidden_seed=40404,
        hidden_amp=0.5,
        weight_scale=_NVFP4_MOE_EXPERT_WEIGHT_SCALE,
        # Tier-1 accuracy threshold from the reference repo (cos > 0.95) —
        # compares the TRT plugin output (FP4 A × FP4 W, atomic scatter-reduce)
        # against an FP32 HF-experts reference with no FP4 quantization.
        cos_threshold=0.95,
        mag_ratio_lo=0.5,
        mag_ratio_hi=2.0,
        case=case,
    )


# ============================================================================
# Qwen3 Gated MoE tests: output = down( SiLU(gate(x)) * up(x) )
# ============================================================================


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"),
                    reason="needs torch.float8_e4m3fn")
def test_qwen3_sparse_moe_block_thor_prepare_contract() -> None:
    """Real Qwen3SparseMoeBlock Thor path prepares plugin buffers and traces the prefill-sized shape."""
    hidden_size, moe_inter_size, num_experts, top_k = 64, 64, 4, 2
    cfg = ModelConfig(
        model_type="qwen3_moe",
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=moe_inter_size,
        head_dim=hidden_size,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter_size,
        decoder_sparse_step=1,
        quant=QuantConfig(quant_type=QUANT_NVFP4,
                          group_size=16,
                          nvfp4_moe_backend=NVFP4_MOE_BACKEND_THOR),
    )
    block = Qwen3SparseMoeBlock(cfg).eval()

    gen = torch.Generator()
    gen.manual_seed(91531)
    block.gate.weight.data.copy_(
        torch.randn(num_experts,
                    hidden_size,
                    generator=gen,
                    dtype=torch.float16))
    for expert in block.experts:
        _populate_nvfp4_linear(
            expert.gate_proj,
            torch.randn(moe_inter_size, hidden_size, generator=gen) * 0.25,
            input_scale_value=2e-3)
        _populate_nvfp4_linear(
            expert.up_proj,
            torch.randn(moe_inter_size, hidden_size, generator=gen) * 0.25,
            input_scale_value=3e-3)
        _populate_nvfp4_linear(
            expert.down_proj,
            torch.randn(hidden_size, moe_inter_size, generator=gen) * 0.25,
            input_scale_value=4e-3)

    block._prepare_moe_weights()

    assert tuple(block.fc_up_qweights.shape) == (num_experts, hidden_size,
                                                 moe_inter_size)
    assert tuple(block.fc_up_blocks_scale.shape) == (num_experts, 128, 4)
    assert tuple(block.fc_down_qweights.shape) == (num_experts, moe_inter_size,
                                                   hidden_size // 2)
    assert tuple(block.fc_down_blocks_scale.shape) == (num_experts, 128, 4)
    assert tuple(block.hidden_global_scale.shape) == (2, )
    assert tuple(block.e_score_correction_bias.shape) == (num_experts, )
    assert len(block.experts) == 0

    # The custom-op eager stub returns zeros_like, but this still verifies that
    # the real module path wires all Thor plugin inputs for a prefill-sized
    # token count (B*S > 16).
    hidden = torch.randn(1,
                         17,
                         hidden_size,
                         generator=gen,
                         dtype=torch.float16)
    out = block(hidden)
    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"),
                    reason="needs torch.float8_e4m3fn")
def test_qwen3_5_moe_sparse_block_thor_prepare_contract() -> None:
    """Qwen3_5SparseMoeBlock Thor path: inherited NVFP4 repack + shared-expert addition.

    Verifies the subclass (used by Qwen3.5 / Qwen3.6 35B-A3B) correctly composes
    on top of Qwen3SparseMoeBlock: routed-expert NVFP4 buffers come from the
    inherited ``_prepare_moe_weights``, while shared_expert + shared_expert_gate
    survive the repack and participate in forward.

    Uses E=8 / top_k=4 (parent test uses E=4 / top_k=2) so the per-expert
    loop, top-k tile, and shared-expert path are all exercised on a different
    profile, while keeping H / I small for fast CI. The shape contract is
    dim-parametric — verified to hold at full Qwen3.6 35B-A3B production dims
    (H=2048, I=512, E=256, top_k=8) during initial bring-up.
    """
    hidden_size, moe_inter_size, num_experts, top_k = 64, 64, 8, 4
    cfg = ModelConfig(
        model_type="qwen3_5_moe_text",
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=moe_inter_size,
        head_dim=hidden_size,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter_size,
        moe_shared_expert_intermediate_size=moe_inter_size,
        decoder_sparse_step=1,
        quant=QuantConfig(
            quant_type=QUANT_NVFP4,
            group_size=16,
            nvfp4_moe_backend=NVFP4_MOE_BACKEND_THOR,
            # Mirror NVIDIA's NVFP4 recipe: ``shared_expert_gate`` is in the
            # ignore list, so ``make_linear`` returns an FP16Linear for it.
            excluded={"model.layers.0.mlp.shared_expert_gate"},
        ),
    )
    block = Qwen3_5SparseMoeBlock(cfg, layer_idx=0).eval()

    gen = torch.Generator()
    gen.manual_seed(91532)
    block.gate.weight.data.copy_(
        torch.randn(num_experts,
                    hidden_size,
                    generator=gen,
                    dtype=torch.float16))
    for expert in block.experts:
        _populate_nvfp4_linear(
            expert.gate_proj,
            torch.randn(moe_inter_size, hidden_size, generator=gen) * 0.25,
            input_scale_value=2e-3)
        _populate_nvfp4_linear(
            expert.up_proj,
            torch.randn(moe_inter_size, hidden_size, generator=gen) * 0.25,
            input_scale_value=3e-3)
        _populate_nvfp4_linear(
            expert.down_proj,
            torch.randn(hidden_size, moe_inter_size, generator=gen) * 0.25,
            input_scale_value=4e-3)
    # Shared expert: same NVFP4Linear shape contract as routed experts.
    _populate_nvfp4_linear(
        block.shared_expert.gate_proj,
        torch.randn(moe_inter_size, hidden_size, generator=gen) * 0.25,
        input_scale_value=2e-3)
    _populate_nvfp4_linear(
        block.shared_expert.up_proj,
        torch.randn(moe_inter_size, hidden_size, generator=gen) * 0.25,
        input_scale_value=3e-3)
    _populate_nvfp4_linear(
        block.shared_expert.down_proj,
        torch.randn(hidden_size, moe_inter_size, generator=gen) * 0.25,
        input_scale_value=4e-3)
    # Shared-expert gate is excluded → FP16Linear with a [1, H] weight.
    block.shared_expert_gate.weight.data.copy_(
        torch.randn(1, hidden_size, generator=gen, dtype=torch.float16))

    block._prepare_moe_weights()

    # Inherited Thor plugin buffers — shapes computed dim-parametrically per
    # the schema in ``tensorrt_edgellm/llm_models/layers/nvfp4_moe_plugin.py``:
    # FC1 SwiGLU has 2*I along the N axis; SF tensors use the 128x4 atom
    # padding (padUp(M, 128), padUp(K/16, 4)).
    def pad_up(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    assert tuple(block.fc_up_qweights.shape) == (num_experts, hidden_size,
                                                 moe_inter_size)
    assert tuple(
        block.fc_up_blocks_scale.shape) == (num_experts,
                                            pad_up(2 * moe_inter_size, 128),
                                            pad_up(hidden_size // 16, 4))
    assert tuple(block.fc_down_qweights.shape) == (num_experts, moe_inter_size,
                                                   hidden_size // 2)
    assert tuple(
        block.fc_down_blocks_scale.shape) == (num_experts,
                                              pad_up(hidden_size, 128),
                                              pad_up(moe_inter_size // 16, 4))
    assert tuple(block.hidden_global_scale.shape) == (2, )
    assert tuple(block.e_score_correction_bias.shape) == (num_experts, )
    assert len(block.experts) == 0  # parent clears per-expert ModuleList

    # Shared-expert sub-modules survive the repack.
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        assert hasattr(block.shared_expert, proj_name), \
            f"shared_expert.{proj_name} lost during repack"
    assert hasattr(block, "shared_expert_gate")

    # Forward: ``routed + shared * sigmoid(shared_expert_gate)`` returns
    # FP16 hidden states with the input shape. The custom-op stub yields zeros,
    # so this just verifies all op signatures wire correctly and dtypes match.
    hidden = torch.randn(1,
                         17,
                         hidden_size,
                         generator=gen,
                         dtype=torch.float16)
    out = block(hidden)
    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype


def _qwen3_dense_forward_from_topk_torch(
    x_fp32: torch.Tensor,
    topw: torch.Tensor,
    topi: torch.Tensor,
    w_gate_ehi: torch.Tensor,
    w_up_ehi: torch.Tensor,
    w_down_eih: torch.Tensor,
    activation_type: int = 1,
) -> torch.Tensor:
    """CUDA torch reference for gated MoE: output = sum_k[ w_k * down(SiLU(gate(x)) * up(x)) ].

    Args:
        x_fp32: (num_tokens, H) FP32
        topw: (num_tokens, top_k) FP32 routing weights
        topi: (num_tokens, top_k) INT64 expert indices
        w_gate_ehi: (E, H, I) FP32 gate weights
        w_up_ehi: (E, H, I) FP32 up weights
        w_down_eih: (E, I, H) FP32 down weights
        activation_type: 1 = SiLU (only SiLU supported for gated)
    """
    assert activation_type == 1, "Gated MoE only supports SiLU activation"
    num_tokens, h = x_fp32.shape
    top_k = topw.shape[1]
    out = torch.zeros(num_tokens, h, dtype=torch.float32)
    x_dev = x_fp32.cuda()
    for t in range(num_tokens):
        for ki in range(top_k):
            expert_idx = int(topi[t, ki].item())
            weight = float(topw[t, ki].item())
            # gate_out = x @ gate_proj^T -> (H,) @ (H, I)^T -> (I,)
            gate_out = torch.mm(x_dev[t:t + 1],
                                w_gate_ehi[expert_idx].cuda())  # (1, I)
            up_out = torch.mm(x_dev[t:t + 1],
                              w_up_ehi[expert_idx].cuda())  # (1, I)
            # SiLU activation on gate output
            gate_act = torch.nn.functional.silu(gate_out)
            # Element-wise multiply
            inter = gate_act * up_out  # (1, I)
            # down projection
            down_out = torch.mm(inter, w_down_eih[expert_idx].cuda())  # (1, H)
            out[t] += weight * down_out.cpu().squeeze(0)
    return out


def _build_qwen3_gated_moe_engine_trt_api(
    module: nn.Module,
    *,
    batch: int,
    seq: int,
    logger,
    verbose: bool = False,
) -> bytes:
    """Build a serialized TensorRT engine for gated MoE using the unified ``Nvfp4MoePlugin``."""
    dummy_hidden = torch.zeros(batch,
                               seq,
                               int(module.hidden_size),
                               dtype=torch.float16)
    return build_nvfp4_moe_engine_trt_api(
        module,
        dummy_hidden,
        logger,
        verbose=verbose,
        label="qwen3-gated",
        explicit_router_logits=True,
    )


def _create_qwen3_moe_plugin(
    *,
    hidden_size: int,
    moe_inter_size: int,
    num_experts: int,
    top_k: int,
    seed: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray]:
    """Create a Qwen3-style gated MoE wrapper with activation_type=1 (SiLU).

    Returns ``(mod, w_gate_ehi, w_up_ehi, w_down_eih)`` — the module with
    interleaved gate+up as FC1 (``moe_inter_size=2*I``), and the original FP32
    weights for building reference outputs.
    """
    mod, w_gate_ehi, w_up_ehi, w_down_eih = _create_toy_qwen3_moe_block(
        hidden_size=hidden_size,
        moe_inter_size=moe_inter_size,
        num_experts=num_experts,
        top_k=top_k,
        seed=seed,
    )
    mod._plugin_activation_type = 1  # SiLU for gated MoE
    mod._plugin_routing_mode = 0
    return mod, w_gate_ehi, w_up_ehi, w_down_eih


def _run_qwen3_w4a16_case(
    *,
    device: torch.device,
    top_k: int,
    moe_seed: int,
    expert: np.ndarray,
    score: np.ndarray,
    hidden_seed: int,
    case: str,
) -> None:
    """One Qwen3 gated MoE accuracy scenario: TRT plugin vs torch reference with Q/DQ'd weights."""
    check_requirements()
    assert trt is not None
    batch = 2
    expert = np.asarray(expert, dtype=np.int32)
    score = np.asarray(score, dtype=np.float32)
    assert expert.shape == score.shape
    num_tokens, rk = expert.shape
    tk = int(top_k)
    assert rk == tk
    assert num_tokens % batch == 0
    seq = num_tokens // batch
    h, inter, e_ct = 384, 768, 8

    mod, _, _, _ = _create_qwen3_moe_plugin(hidden_size=h,
                                            moe_inter_size=inter,
                                            num_experts=e_ct,
                                            top_k=tk,
                                            seed=moe_seed)

    logits_np = NemotronHMoEReference.router_logits_from_desired_topk(
        expert, score, num_tokens=num_tokens, num_experts=e_ct, top_k=tk)

    torch.manual_seed(hidden_seed)
    hidden_bsh = torch.randn(batch, seq, h) * 2.5
    hidden_np = hidden_bsh.numpy().astype(np.float16)

    mod_cpu = mod.cpu()
    topw_np, topi_np = NemotronHMoEReference.moe_topk_softmax_renormalize_numpy(
        logits_np, tk)

    logger = trt.Logger(trt.Logger.WARNING)
    load_nvfp4_moe_edge_llm_plugins(logger, _PLUGIN_SO, verbose=False)
    eng = _build_qwen3_gated_moe_engine_trt_api(mod_cpu,
                                                batch=batch,
                                                seq=seq,
                                                logger=logger,
                                                verbose=False)

    stream = torch.cuda.Stream(device=device)
    # hidden_global_scale [2] FP32: unused by W4A16 decode path, but the engine input must be bound.
    hidden_gs_np = np.ones(2, dtype=np.float32)
    trt_out = execute_trt_engine(
        eng,
        {
            "router_logits": np.ascontiguousarray(logits_np.astype(
                np.float32)),
            "hidden_states": np.ascontiguousarray(hidden_np),
            "hidden_global_scale": np.ascontiguousarray(hidden_gs_np),
        },
        stream,
        device=device,
        verbose=False,
        label=f"qwen3-gated-{case}",
    )

    # Build torch reference with Q/DQ'd weights (quantize then dequantize to match NVFP4 rounding).
    # The unified plugin stores interleaved gate+up in fc_up_qweights [E, H, 2*I/2] with
    # moe_inter_size = 2*I. Extract the full interleaved FC1 weight, then split gate and up.
    # Use N-major buffers with prefill atom-layout scales.
    plugin_inter = int(mod_cpu.moe_intermediate_size)  # 2*I
    w_fc1_qdq, w_down_qdq = NemotronHMoEReference.dense_weights_from_nvfp4_nmajor_buffers(
        mod_cpu._stacked_up_weights.numpy(),
        mod_cpu._stacked_up_block_scale.numpy(),
        mod_cpu._stacked_down_weights.numpy(),
        mod_cpu._stacked_down_block_scale.numpy(),
        num_experts=e_ct,
        hidden_size=h,
        moe_inter_size=plugin_inter,
        activation_type=1,
    )
    up_gs_np = mod_cpu._stacked_up_global_scale.numpy().reshape(-1)
    dn_gs_np = mod_cpu._stacked_down_global_scale.numpy().reshape(-1)
    for ex in range(e_ct):
        w_fc1_qdq[ex] *= float(up_gs_np[ex])
        w_down_qdq[ex] *= float(dn_gs_np[ex])
    # De-interleave FC1: (up, gate) alternate at 32-col granularity along N.
    # Reshape [E, H, 2*I] → [E, H, n_chunks, 2, 32] then split into up / gate.
    _SWIGLU_GS = 32
    _n_chunks = inter // _SWIGLU_GS
    _fc1_paired = w_fc1_qdq.reshape(e_ct, h, _n_chunks, 2, _SWIGLU_GS)
    w_up_qdq = _fc1_paired[:, :, :, 0, :].reshape(e_ct, h, inter)
    w_gate_qdq = _fc1_paired[:, :, :, 1, :].reshape(e_ct, h, inter)

    x_fp32 = hidden_np.reshape(-1, h).astype(np.float32)
    ref_torch = _qwen3_dense_forward_from_topk_torch(
        torch.from_numpy(x_fp32),
        torch.from_numpy(topw_np),
        torch.from_numpy(topi_np.astype(np.int64)),
        torch.from_numpy(w_gate_qdq),
        torch.from_numpy(w_up_qdq),
        torch.from_numpy(w_down_qdq),
        activation_type=1,
    )
    ref_fp32 = ref_torch.detach().numpy().astype(np.float32).reshape(
        trt_out.shape)

    # Compare TRT output against Q/DQ reference (FP4 quantize→dequantize weights).
    trt_np = np.asarray(trt_out, dtype=np.float32)
    med_cos, mean_cos, min_cos = _cosine_sim_per_row(trt_np, ref_fp32)
    mag = _magnitude_ratio(trt_np, ref_fp32)
    print(
        f"[qwen3 {case}] numTokens={num_tokens} topK={tk} median_cos={med_cos:.4f} "
        f"mean_cos={mean_cos:.4f} min_cos={min_cos:.4f} mag_ratio={mag:.3f} "
        f"|trt|={float(np.linalg.norm(trt_np)):.4g} |ref|={float(np.linalg.norm(ref_fp32)):.4g}"
    )
    if med_cos < 0.99:
        raise AssertionError(
            f"{case}: median cosine vs Q/DQ ref {med_cos:.4f} < 0.99 "
            f"(min_cos={min_cos:.4f}, mag={mag:.3f})")
    if not (0.5 <= mag <= 2.0):
        raise AssertionError(
            f"{case}: magnitude ratio {mag:.3f} outside [0.5, 2.0]")


@pytest.mark.parametrize(
    ("top_k", "moe_seed", "expert_rows", "score_rows", "hidden_seed", "case"),
    [
        pytest.param(1,
                     3026, [[0], [5]], [[1.0], [1.0]],
                     92021,
                     "qwen3_w4a16_topk1",
                     id="topk1"),
        pytest.param(2,
                     3027, [[0, 1], [6, 3]], [[0.62, 0.38], [0.55, 0.45]],
                     92022,
                     "qwen3_w4a16_topk2",
                     id="topk2"),
        pytest.param(4,
                     3028, [[0, 1, 2, 3], [7, 5, 4, 6]],
                     [[0.40, 0.28, 0.20, 0.12], [0.36, 0.30, 0.22, 0.12]],
                     92023,
                     "qwen3_w4a16_topk4",
                     id="topk4"),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qwen3_gated_moe_w4a16_decode_accuracy(
    top_k,
    moe_seed,
    expert_rows,
    score_rows,
    hidden_seed,
    case,
):
    """Nvfp4MoePlugin gated (SwiGLU) accuracy: W4A16 decode vs torch Q/DQ reference."""
    check_requirements()
    dev = torch.device("cuda", torch.cuda.current_device())
    _run_qwen3_w4a16_case(
        device=dev,
        top_k=top_k,
        moe_seed=moe_seed,
        expert=np.array(expert_rows, dtype=np.int32),
        score=np.array(score_rows, dtype=np.float32),
        hidden_seed=hidden_seed,
        case=case,
    )
