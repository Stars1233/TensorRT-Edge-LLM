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
"""Accuracy tests for the GeForce NVFP4 fused MoE CuTeDSL backends."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

torch = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KERNEL_SRC_DIR = _REPO_ROOT / "kernelSrcs" / "nvfp4_fused_moe_cutedsl"


@dataclass(frozen=True)
class _MoeCase:
    name: str
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    seed: int

    @property
    def routed_rows(self) -> int:
        return self.num_tokens * self.top_k


@dataclass(frozen=True)
class _CutedslMoeHelpers:
    moe_dispatch: object
    nvfp4_quant: object


@pytest.fixture(scope="module")
def torch_cuda():
    global torch
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for NVFP4 fused MoE accuracy tests")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip(
            "torch.float4_e2m1fn_x2 is required for NVFP4 fused MoE tests")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip(
            "torch.float8_e4m3fn is required for NVFP4 fused MoE tests")
    return torch


@pytest.fixture(scope="module")
def cutedsl_moe_helpers(torch_cuda) -> _CutedslMoeHelpers:
    del torch_cuda
    if not _KERNEL_SRC_DIR.is_dir():
        pytest.skip(
            f"missing CuTeDSL MoE kernel source directory: {_KERNEL_SRC_DIR}")
    kernel_src_dir = str(_KERNEL_SRC_DIR)
    if kernel_src_dir not in sys.path:
        sys.path.insert(0, kernel_src_dir)
    try:
        moe_dispatch = importlib.import_module("moe_dispatch")
        nvfp4_quant = importlib.import_module("nvfp4_quant")
    except (ImportError, OSError, RuntimeError) as exc:
        pytest.skip(f"CuTeDSL MoE launch helpers are unavailable: {exc}")
    return _CutedslMoeHelpers(moe_dispatch=moe_dispatch,
                              nvfp4_quant=nvfp4_quant)


def _silu(x):
    return x * torch.sigmoid(x)


def _gelu_reference(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 *
                                       (x + 0.044715 * x * x * x)))


def _apply_activation(x, activation: str):
    if activation == "identity":
        return x
    if activation == "silu":
        return _silu(x)
    if activation == "gelu":
        return _gelu_reference(x)
    if activation == "relu":
        return torch.relu(x)
    if activation == "relu2":
        relu_out = torch.relu(x)
        return relu_out * relu_out
    raise ValueError(f"Unknown activation: {activation}")


def _quant_dequant_fp4_reference(nvfp4_quant,
                                 tensor,
                                 global_scale,
                                 sf_vec_size: int = 16):
    tensor_bf16 = tensor.to(torch.bfloat16)
    fp4_packed, sf_linear = nvfp4_quant.fp4_quantize(
        tensor_bf16,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    dequantized = nvfp4_quant.e2m1_and_ufp8sf_scale_to_float(
        fp4_packed,
        sf_linear,
        (1.0 / global_scale),
        sf_vec_size=sf_vec_size,
        ufp8_type=1,
        is_sf_swizzled_layout=False,
    )
    return dequantized.to(tensor.device).float()


def _compute_reference_moe_fp4(
    nvfp4_quant,
    *,
    hidden_states,
    gemm1_weights,
    gemm2_weights,
    token_selected_experts,
    token_final_scales,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    fc2_input_scale=None,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
):
    if num_local_experts is None:
        num_local_experts = num_experts

    device = hidden_states.device
    hidden_states = hidden_states.float()
    gemm1_weights = gemm1_weights.float()
    gemm2_weights = gemm2_weights.float()

    output = torch.zeros((num_tokens, hidden_size),
                         dtype=torch.float32,
                         device=device)

    for token_idx in range(num_tokens):
        token_input = hidden_states[token_idx:token_idx + 1]
        for topk_idx in range(top_k):
            expert_idx = token_selected_experts[token_idx, topk_idx].item()
            scale = token_final_scales[token_idx, topk_idx].item()
            if expert_idx < 0 or expert_idx >= num_experts:
                continue
            local_idx = expert_idx - local_expert_offset
            if local_idx < 0 or local_idx >= num_local_experts:
                continue

            w1 = gemm1_weights[local_idx]
            gemm1_out = token_input @ w1.T

            linear = gemm1_out[:, :intermediate_size]
            gate = gemm1_out[:, intermediate_size:]
            swiglu_out = _silu(gate) * linear

            if fc2_input_scale is not None:
                swiglu_out = _quant_dequant_fp4_reference(nvfp4_quant,
                                                          swiglu_out,
                                                          fc2_input_scale,
                                                          sf_vec_size=16)

            w2 = gemm2_weights[local_idx]
            gemm2_out = swiglu_out @ w2.T
            output[token_idx] += scale * gemm2_out.squeeze(0)

    return output


def _compute_reference_moe_nongated(
    nvfp4_quant,
    *,
    hidden_states,
    gemm1_weights,
    gemm2_weights,
    token_selected_experts,
    token_final_scales,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    activation: str = "identity",
    fc2_input_scale=None,
):
    device = hidden_states.device
    hidden_states = hidden_states.float()
    gemm1_weights = gemm1_weights.float()
    gemm2_weights = gemm2_weights.float()

    output = torch.zeros((num_tokens, hidden_size),
                         dtype=torch.float32,
                         device=device)

    for token_idx in range(num_tokens):
        token_input = hidden_states[token_idx:token_idx + 1]
        for topk_idx in range(top_k):
            expert_idx = token_selected_experts[token_idx, topk_idx].item()
            scale = token_final_scales[token_idx, topk_idx].item()
            if expert_idx < 0 or expert_idx >= num_experts:
                continue

            w1 = gemm1_weights[expert_idx]
            fc1_out = token_input @ w1.T
            act_out = _apply_activation(fc1_out, activation)

            if fc2_input_scale is not None:
                act_out = _quant_dequant_fp4_reference(nvfp4_quant,
                                                       act_out,
                                                       fc2_input_scale,
                                                       sf_vec_size=16)

            w2 = gemm2_weights[expert_idx]
            fc2_out = act_out @ w2.T
            output[token_idx] += scale * fc2_out.squeeze(0)

    return output


def _convert_sf_to_mma_layout(
    sf,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
):
    sf_k = (k + sf_vec_size - 1) // sf_vec_size
    m_tiles = (m + 127) // 128
    k_tiles = (sf_k + 3) // 4
    expected_elements = num_groups * m_tiles * k_tiles * 32 * 4 * 4
    actual_elements = sf.numel()
    if actual_elements != expected_elements:
        raise ValueError(
            f"Scale factor tensor has {actual_elements} elements, "
            f"expected {expected_elements} for m={m}, k={k}, num_groups={num_groups}"
        )
    return sf.view(num_groups, m_tiles, k_tiles, 32, 4,
                   4).permute(3, 4, 1, 5, 2, 0)


def _quantize_weights_for_moe(
    nvfp4_quant,
    w_bf16,
    *,
    num_experts: int,
    rows_per_expert: int,
    cols: int,
):
    sf_vec_size = 16
    global_scale = torch.tensor([1.0],
                                device=w_bf16.device,
                                dtype=torch.float32)

    w_flat = w_bf16.reshape(num_experts * rows_per_expert, cols)
    w_q_flat, w_sf_flat = nvfp4_quant.fp4_quantize(
        w_flat,
        global_scale=global_scale,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=True,
    )
    w_q = w_q_flat.view(num_experts, rows_per_expert, cols // 2)
    w_sf_mma = _convert_sf_to_mma_layout(
        w_sf_flat,
        m=rows_per_expert,
        k=cols,
        num_groups=num_experts,
        sf_vec_size=sf_vec_size,
    )
    return w_q, w_sf_mma


def _create_moe_tensors(
    nvfp4_quant,
    *,
    case: _MoeCase,
    activation: str,
    device: str = "cuda",
    dtype=None,
):
    if dtype is None:
        dtype = torch.float16

    torch.manual_seed(case.seed)
    torch.cuda.manual_seed_all(case.seed)

    x_bf16 = torch.randn(
        case.num_tokens, case.hidden_size, dtype=dtype, device=device) / 10

    router_logits = torch.randn(case.num_tokens,
                                case.num_experts,
                                device=device)
    routing_weights = torch.nn.functional.softmax(router_logits,
                                                  dim=1,
                                                  dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights,
                                                   case.top_k,
                                                   dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1,
                                                            keepdim=True)
    routing_weights = routing_weights.float()
    selected_experts = selected_experts.to(torch.int32)

    is_gated = activation in ("swiglu", "geglu")
    w1_rows = 2 * case.intermediate_size if is_gated else case.intermediate_size
    w1_bf16 = (torch.randn(
        case.num_experts,
        w1_rows,
        case.hidden_size,
        dtype=dtype,
        device=device,
    ) / 10)
    w1_q, w1_weight_sf = _quantize_weights_for_moe(
        nvfp4_quant,
        w1_bf16,
        num_experts=case.num_experts,
        rows_per_expert=w1_rows,
        cols=case.hidden_size,
    )
    w1_alpha = torch.ones(case.num_experts, device=device, dtype=torch.float32)

    w2_bf16 = (torch.randn(
        case.num_experts,
        case.hidden_size,
        case.intermediate_size,
        dtype=dtype,
        device=device,
    ) / 10)
    w2_q, w2_weight_sf = _quantize_weights_for_moe(
        nvfp4_quant,
        w2_bf16,
        num_experts=case.num_experts,
        rows_per_expert=case.hidden_size,
        cols=case.intermediate_size,
    )
    w2_alpha = torch.ones(case.num_experts, device=device, dtype=torch.float32)

    return {
        "x_bf16": x_bf16,
        "token_selected_experts": selected_experts,
        "token_final_scales": routing_weights,
        "w1_weight": w1_q,
        "w1_weight_sf": w1_weight_sf,
        "w1_weight_bf16": w1_bf16,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": torch.tensor([1.0],
                                        device=device,
                                        dtype=torch.float32),
        "w2_weight": w2_q,
        "w2_weight_sf": w2_weight_sf,
        "w2_weight_bf16": w2_bf16,
        "w2_alpha": w2_alpha,
    }


def _check_accuracy(actual, expected, percent_threshold: float = 0.97):
    actual = actual.float()
    expected = expected.float()

    output_scale = max(expected.std().item(), 0.01)
    atol = max(0.05, 1.5 * output_scale)
    rtol = 0.5

    abs_diff = torch.abs(actual - expected)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-8)
    within_tolerance = (abs_diff < atol) | (rel_diff < rtol)
    percent_within = within_tolerance.float().mean().item()
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()

    return percent_within >= percent_threshold, percent_within, atol, max_abs_error, mean_abs_error


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _MoeCase(
                name="decode",
                num_tokens=16,
                hidden_size=128,
                intermediate_size=640,
                num_experts=8,
                top_k=2,
                seed=42,
            ),
            id="decode_routed_rows_le_640",
        ),
        pytest.param(
            _MoeCase(
                name="prefill",
                num_tokens=321,
                hidden_size=128,
                intermediate_size=640,
                num_experts=8,
                top_k=2,
                seed=43,
            ),
            id="prefill_routed_rows_gt_640",
        ),
    ],
)
def test_nvfp4_fused_moe_geforce_matches_pytorch_reference(
        torch_cuda, cutedsl_moe_helpers, case: _MoeCase):
    del torch_cuda
    activation = "swiglu"
    assert case.intermediate_size > 512
    if case.name == "decode":
        assert case.routed_rows <= 640
    else:
        assert case.routed_rows > 640

    tensors = _create_moe_tensors(
        cutedsl_moe_helpers.nvfp4_quant,
        case=case,
        activation=activation,
        dtype=torch.float16,
    )

    scatter_output = torch.zeros(
        case.num_tokens,
        case.hidden_size,
        dtype=torch.float16,
        device="cuda",
    )
    result = cutedsl_moe_helpers.moe_dispatch.launch_sm120_moe(
        a=tensors["x_bf16"],
        topk_ids=tensors["token_selected_experts"],
        topk_weights=tensors["token_final_scales"],
        w1_weight=tensors["w1_weight"],
        w1_weight_sf=tensors["w1_weight_sf"],
        w1_alpha=tensors["w1_alpha"],
        fc2_input_scale=tensors["fc2_input_scale"],
        w2_weight=tensors["w2_weight"],
        w2_weight_sf=tensors["w2_weight_sf"],
        w2_alpha=tensors["w2_alpha"],
        num_experts=case.num_experts,
        top_k=case.top_k,
        num_local_experts=case.num_experts,
        scatter_output=scatter_output,
        activation=activation,
        input_scales_are_reciprocal=False,
        fast_math=True,
    )

    assert result.shape == (case.num_tokens, case.hidden_size)
    assert result.dtype == torch.float16
    assert torch.isfinite(
        result).all(), f"non-finite values in {case.name} output"

    ref_output = _compute_reference_moe_fp4(
        cutedsl_moe_helpers.nvfp4_quant,
        hidden_states=tensors["x_bf16"].float().cuda(),
        gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
        gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
        token_selected_experts=tensors["token_selected_experts"],
        token_final_scales=tensors["token_final_scales"],
        num_tokens=case.num_tokens,
        num_experts=case.num_experts,
        top_k=case.top_k,
        hidden_size=case.hidden_size,
        intermediate_size=case.intermediate_size,
        fc2_input_scale=tensors["fc2_input_scale"],
    )
    assert torch.isfinite(
        ref_output).all(), f"non-finite values in {case.name} reference output"

    passed, percent_within, atol, max_abs_error, mean_abs_error = _check_accuracy(
        result, ref_output)
    assert passed, (
        f"{case.name} routed_rows={case.routed_rows}: {percent_within * 100:.2f}% within "
        f"tolerance (atol={atol:.4f}, max_abs_error={max_abs_error:.4f}, "
        f"mean_abs_error={mean_abs_error:.4f})")
