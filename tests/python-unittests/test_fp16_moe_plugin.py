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
"""TensorRT lifecycle and accuracy tests for the FP16 MoE plugin.

Exercises ``Fp16MoePlugin`` on unquantized FP16 experts for both routing
modes: softmax top-k (Qwen, SwiGLU) and sigmoid group-topk (Nemotron-H,
ReLU²). Each positive test builds one dynamic-profile engine, round-trips its
serialization, and runs decode (S=1) and prefill (S=128) against a pure-torch
reference. The experts are packed through the production repack helper
(``repack_fp16_moe_experts``) so the plugin's FC1/FC2 layout contract is
covered end to end. Tests skip when the ``f16_moe`` CuTeDSL artifact is not
linked into the plugin build.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, PluginUnsupportedError,
                              assert_close, pf_float32, pf_int32)

if DEPENDENCIES_AVAILABLE:
    import torch

    from tensorrt_edgellm.checkpoint.repacking import repack_fp16_moe_experts

_PLUGIN_NAME = "Fp16MoePlugin"
_PLUGIN_VERSION = "1"
_NUM_EXPERTS = 128
_HIDDEN_SIZE = 128
_MAX_SEQUENCE_LENGTH = 128
_ACT_SWIGLU = 2
_ACT_RELU2 = 4


@pytest.fixture(autouse=True)
def require_plugin_test_dependencies(request):
    """Prevent the GPU-only L0 suite from passing through dependency skips."""
    if DEPENDENCIES_AVAILABLE:
        return
    reason = f"TensorRT/torch CUDA not available: {IMPORT_ERROR}"
    if request.config.getoption("--priority") == "l0_python_ut":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


@dataclass(frozen=True)
class MoeCase:
    name: str
    top_k: int
    moe_inter_size: int
    activation_type: int
    routing_mode: int
    n_group: int
    topk_group: int
    norm_topk_prob: int
    routed_scaling_factor: float
    hidden_size: int = _HIDDEN_SIZE
    num_experts: int = _NUM_EXPERTS

    @property
    def fc1_out_dim(self) -> int:
        return (2 * self.moe_inter_size if self.activation_type == _ACT_SWIGLU
                else self.moe_inter_size)

    @property
    def routed_row_capacity(self) -> int:
        return _MAX_SEQUENCE_LENGTH * self.top_k


_QWEN_CASE = MoeCase(name="qwen_swiglu_softmax",
                     top_k=8,
                     moe_inter_size=128,
                     activation_type=_ACT_SWIGLU,
                     routing_mode=0,
                     n_group=1,
                     topk_group=1,
                     norm_topk_prob=1,
                     routed_scaling_factor=1.0)

_NEMOTRON_CASE = MoeCase(name="nemotron_relu2_sigmoid_group",
                         top_k=6,
                         moe_inter_size=128,
                         activation_type=_ACT_RELU2,
                         routing_mode=1,
                         n_group=8,
                         topk_group=4,
                         norm_topk_prob=1,
                         routed_scaling_factor=2.5)


def _make_expert(case: MoeCase, generator: "torch.Generator"):
    """A minimal expert exposing the ``.<proj>.weight`` tensors the repack
    helper reads. ``SimpleNamespace`` avoids referencing ``torch.nn`` at module
    import time (the deps guard imports torch lazily)."""
    h, i = case.hidden_size, case.moe_inter_size

    def _weight(out_f, in_f):
        w = (torch.randn(
            (out_f, in_f), generator=generator, dtype=torch.float32) * 0.1).to(
                torch.float16)
        return SimpleNamespace(weight=w)

    expert = SimpleNamespace(up_proj=_weight(i, h), down_proj=_weight(h, i))
    if case.activation_type == _ACT_SWIGLU:
        expert.gate_proj = _weight(i, h)
    return expert


@dataclass
class MoeFixture:
    case: MoeCase
    experts: List["SimpleNamespace"]
    fc1_weights: "torch.Tensor"
    fc2_weights: "torch.Tensor"


def _make_fixture(case: MoeCase) -> MoeFixture:
    generator = torch.Generator().manual_seed(hash(case.name) & 0xFFFF)
    experts = [_make_expert(case, generator) for _ in range(case.num_experts)]
    fc1, fc2, padded_inter = repack_fp16_moe_experts(experts, case.hidden_size,
                                                     case.moe_inter_size,
                                                     case.activation_type)
    assert padded_inter == case.moe_inter_size  # test dims are pre-aligned
    return MoeFixture(case=case,
                      experts=experts,
                      fc1_weights=fc1.contiguous(),
                      fc2_weights=fc2.contiguous())


def _routing_reference(case: MoeCase, router_logits: "torch.Tensor",
                       expert_score_bias: "torch.Tensor"):
    if case.routing_mode == 0:
        scores = torch.softmax(router_logits, dim=-1)
        weights, indices = torch.topk(scores, case.top_k, dim=-1)
        if case.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices

    scores = torch.sigmoid(router_logits)
    biased = scores + expert_score_bias
    experts_per_group = case.num_experts // case.n_group
    grouped = biased.view(-1, case.n_group, experts_per_group)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(case.topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = group_mask[:, :, None].expand_as(grouped).reshape_as(biased)
    selected_scores = biased.masked_fill(~expert_mask, float("-inf"))
    indices = selected_scores.topk(case.top_k, dim=-1).indices
    weights = scores.gather(1, indices)
    if case.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * case.routed_scaling_factor, indices


def _moe_reference(fixture: MoeFixture, hidden_states: "torch.Tensor",
                   router_logits: "torch.Tensor",
                   expert_score_bias: "torch.Tensor"):
    case = fixture.case
    hidden_2d = hidden_states.reshape(-1, case.hidden_size)
    route_weights, route_indices = _routing_reference(case, router_logits,
                                                      expert_score_bias)
    out = torch.zeros((hidden_2d.shape[0], case.hidden_size),
                      dtype=torch.float32,
                      device=hidden_states.device)

    for expert_id in range(case.num_experts):
        token_slot = (route_indices == expert_id).nonzero(as_tuple=False)
        if token_slot.numel() == 0:
            continue
        token_ids = token_slot[:, 0]
        slot_ids = token_slot[:, 1]
        x = hidden_2d[token_ids].to(torch.float32)
        expert = fixture.experts[expert_id]
        up = (x @ expert.up_proj.weight.float().to(x.device).T)
        if case.activation_type == _ACT_SWIGLU:
            gate = (x @ expert.gate_proj.weight.float().to(x.device).T)
            activated = torch.nn.functional.silu(gate) * up
        else:
            activated = torch.relu(up).square()
        down = (activated @ expert.down_proj.weight.float().to(x.device).T)
        weighted = down * route_weights[token_ids, slot_ids, None].float()
        out[token_ids] += weighted

    return out.to(torch.float16).reshape_as(hidden_states)


def _plugin_fields(case: MoeCase):
    fields = [
        pf_int32("num_experts", case.num_experts),
        pf_int32("top_k", case.top_k),
        pf_int32("hidden_size", case.hidden_size),
        pf_int32("moe_inter_size", case.moe_inter_size),
        pf_int32("activation_type", case.activation_type),
        pf_int32("norm_topk_prob", case.norm_topk_prob),
        pf_int32("max_routed_rows", case.routed_row_capacity),
        pf_int32("routing_mode", case.routing_mode),
    ]
    if case.routing_mode == 1:
        fields += [
            pf_int32("n_group", case.n_group),
            pf_int32("topk_group", case.topk_group),
            pf_float32("routed_scaling_factor", case.routed_scaling_factor),
        ]
    return fields


def _io_specs(case: MoeCase):
    import tensorrt as trt
    specs = [
        ("router_logits", trt.float32, (-1, case.num_experts)),
        ("hidden_states", trt.float16, (-1, -1, case.hidden_size)),
        ("fc1_weights", trt.float16, (case.num_experts, case.fc1_out_dim,
                                      case.hidden_size)),
        ("fc2_weights", trt.float16, (case.num_experts, case.hidden_size,
                                      case.moe_inter_size)),
    ]
    if case.routing_mode == 1:
        specs.append(("expert_score_bias", trt.float32, (case.num_experts, )))
    return specs


def _profiles(case: MoeCase, input_specs):
    profiles = {}
    for name, _, shape in input_specs:
        if name == "router_logits":
            profiles[name] = ((1, case.num_experts), (1, case.num_experts),
                              (_MAX_SEQUENCE_LENGTH, case.num_experts))
        elif name == "hidden_states":
            profiles[name] = ((1, 1, case.hidden_size), (1, 1,
                                                         case.hidden_size),
                              (1, _MAX_SEQUENCE_LENGTH, case.hidden_size))
        else:
            profiles[name] = (shape, shape, shape)
    return profiles


def _execute_decode_and_prefill(fixture: MoeFixture) -> None:
    case = fixture.case
    runner = PluginRunner()
    input_specs = _io_specs(case)
    try:
        runner.build(input_specs=input_specs,
                     output_names=["output"],
                     plugin_name=_PLUGIN_NAME,
                     plugin_version=_PLUGIN_VERSION,
                     plugin_fields=_plugin_fields(case),
                     profiles=_profiles(case, input_specs),
                     expect_unsupported=True)
    except PluginUnsupportedError:
        pytest.skip(
            "Fp16MoePlugin f16_moe CuTeDSL artifact not linked in this build")

    static_inputs = {
        "fc1_weights": fixture.fc1_weights.to("cuda").contiguous(),
        "fc2_weights": fixture.fc2_weights.to("cuda").contiguous(),
    }

    for sequence_length in (1, _MAX_SEQUENCE_LENGTH):
        generator = torch.Generator().manual_seed(30000 + sequence_length +
                                                  case.routing_mode)
        hidden_states = (torch.randn(
            (1, sequence_length, case.hidden_size),
            generator=generator,
            dtype=torch.float32) * 0.25).to(torch.float16).to("cuda")
        router_logits = torch.randn((sequence_length, case.num_experts),
                                    generator=generator,
                                    dtype=torch.float32).to("cuda")
        expert_score_bias = (torch.randn(
            (case.num_experts, ), generator=generator, dtype=torch.float32) *
                             0.01).to("cuda")
        expected = _moe_reference(fixture, hidden_states, router_logits,
                                  expert_score_bias)
        actual = torch.empty_like(hidden_states)
        tensors = {
            "router_logits": router_logits,
            "hidden_states": hidden_states,
            "output": actual,
            **static_inputs,
        }
        if case.routing_mode == 1:
            tensors["expert_score_bias"] = expert_score_bias

        runner.execute(tensors)

        assert bool(torch.isfinite(actual.to(torch.float32)).all())
        assert_close(f"{case.name}[S={sequence_length}]",
                     expected,
                     actual,
                     atol=0.05,
                     rtol=0.02,
                     cos_threshold=0.99)


@pytest.mark.parametrize("case", [_QWEN_CASE, _NEMOTRON_CASE],
                         ids=lambda c: c.name)
def test_fp16_moe_decode_and_prefill_dynamic_engine(case):
    _execute_decode_and_prefill(_make_fixture(case))
