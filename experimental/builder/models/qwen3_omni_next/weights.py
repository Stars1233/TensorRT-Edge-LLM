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
"""Qwen3-Omni-Next checkpoint conversion and expert packing."""

import os
import tempfile
import threading
from typing import Dict

import numpy as np

from ...core.weights import ParameterSpec
from ...weight_packing import nvfp4 as nvfp4_pack

_PREFIXES = {
    "llm": ("thinker.", "language_model.", "model.language_model.",
            "vlm.model.language_model.", "vlm."),
    "visual": ("thinker.visual.", "visual.", "vision_tower.", "model.visual."),
    "audio":
    ("thinker.audio_tower.", "audio_tower.", "audio.", "model.audio."),
    "talker": ("talker.", ),
    "code-predictor": ("talker.code_predictor.", "code_predictor."),
}
_WRAPPERS = (
    "model.language_model.",
    "thinker.model.",
    "talker.code_predictor.",
    "talker.model.",
    "language_model.",
    "text_model.",
    "llm.",
    "thinker.",
    "talker.",
)
_CODEC_STAGES: Dict[str, tempfile.TemporaryDirectory] = {}
_CODEC_STAGE_LOCK = threading.Lock()


def _codec_source(model_dir: str) -> str:
    nested = os.path.join(model_dir, "codec_decode_online")
    if os.path.isfile(os.path.join(nested, "model_weights.pt")):
        return nested
    if os.path.isfile(os.path.join(model_dir, "model_weights.pt")):
        return model_dir
    return ""


def _stage_codec_checkpoint(source_dir: str) -> str:
    """Flatten the provider codec archive for the standard read-only store."""
    source_dir = os.path.realpath(source_dir)
    with _CODEC_STAGE_LOCK:
        cached = _CODEC_STAGES.get(source_dir)
        if cached is not None:
            return cached.name

        import torch

        source = os.path.join(source_dir, "model_weights.pt")
        state = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        if isinstance(state, dict) and isinstance(state.get("generator"),
                                                  dict):
            state = state["generator"]
        if not isinstance(state, dict):
            raise TypeError(
                f"Qwen3-Omni-Next codec expected a tensor dict in {source}")

        generator = any(key.startswith("generator.") for key in state)
        if generator:
            state = {
                key[len("generator."):]: value
                for key, value in state.items() if key.startswith("generator.")
            }
        non_tensors = [
            key for key, value in state.items()
            if not isinstance(value, torch.Tensor)
        ]
        if non_tensors:
            raise TypeError(
                "Qwen3-Omni-Next codec archive contains non-tensor entries: " +
                ", ".join(non_tensors[:8]))

        stage = tempfile.TemporaryDirectory(
            prefix="edgellm-qwen3-omni-next-codec-")
        torch.save(state, os.path.join(stage.name, "pytorch_model.bin"))
        _CODEC_STAGES[source_dir] = stage
        return stage.name


def checkpoint_dir(model_dir: str, component: str) -> str:
    """Select the checkpoint subtree that owns one Next component."""
    if component == "code2wav":
        source = _codec_source(model_dir)
        if source:
            return _stage_codec_checkpoint(source)
    return model_dir


def writes_runtime_embedding(args) -> bool:
    """Native MTP drafts consume the base thinker's embedding."""
    return not (args.resolved_spec_role.value == "draft"
                and args.spec_type == "mtp")


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend names to full-root and standalone Next checkpoints."""
    prefixes = _PREFIXES.get(component, ())
    candidates = []
    if spec_role == "draft" and spec_type == "mtp":
        candidates.extend((f"thinker.mtp.{name}", f"mtp.{name}"))
    candidates.extend(prefix + name for prefix in prefixes)
    if component == "llm" and name.startswith("model."):
        nested = name[len("model."):]
        candidates.extend(prefix + nested for prefix in prefixes)
    if component == "talker" and name == "model.embed_tokens.weight":
        candidates.extend(
            ("talker.model.embed_tokens.weight", "model.embed_tokens.weight"))
    if component == "talker" and name == "model.codec_embedding.weight":
        candidates.extend(("talker.model.codec_embedding.weight",
                           "model.codec_embedding.weight"))
    if component == "talker" and name.startswith("codec_head."):
        suffix = name[len("codec_head."):]
        candidates.extend(
            (f"talker.codec_head.{suffix}", f"codec_head.{suffix}"))
    if component == "code-predictor" and name.startswith(
            "small_to_mtp_projection."):
        suffix = name[len("small_to_mtp_projection."):]
        candidates.extend(
            (f"talker.code_predictor.model.talker_projection.{suffix}",
             f"code_predictor.model.talker_projection.{suffix}"))
    if name == "lm_head.weight" and quant_type == "fp16":
        candidates.extend(
            ("thinker.model.embed_tokens.weight", "model.embed_tokens.weight",
             "model.language_model.embed_tokens.weight"))
    return tuple(candidates)


def normalize_checkpoint_name(name: str) -> str:
    """Remove only wrappers used by Qwen3-Omni-Next checkpoints."""
    for prefix in _WRAPPERS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


_GDN_INPUT_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def expand_quantized_module(name: str):
    """Map provider-fused projections to frontend linear modules."""
    if name.endswith(".self_attn.qkv_proj"):
        prefix = name[:-len("qkv_proj")]
        return tuple(prefix + projection
                     for projection in ("q_proj", "k_proj", "v_proj"))
    if name.endswith(".mlp.gate_up_proj"):
        prefix = name[:-len("gate_up_proj")]
        return (prefix + "gate_proj", prefix + "up_proj")
    return (name, )


def finalize_exclusions(modules):
    """Exclude a fused GDN input only when all source projections are plain."""
    result = set(modules)
    grouped = {}
    for module in result:
        for projection in _GDN_INPUT_PROJECTIONS:
            suffix = "." + projection
            if module.endswith(suffix):
                grouped.setdefault(module[:-len(suffix)],
                                   set()).add(projection)
                break
    for prefix, projections in grouped.items():
        if all(projection in projections
               for projection in _GDN_INPUT_PROJECTIONS):
            result.add(prefix + ".in_proj_fused")
    return tuple(result)


def convert_linear_fp16(weights, prefix: str):
    """Split fused QKV or gate/up tensors without changing numeric values."""
    parent, _, projection = prefix.rpartition(".")
    if projection in ("q_proj", "k_proj", "v_proj"):
        fused_prefix = parent + ".qkv_proj"
        if not weights.has(fused_prefix + ".weight"):
            return None
        weight = weights.f16(fused_prefix + ".weight")
        query_size = int(weight.shape[1])
        remaining = int(weight.shape[0]) - query_size
        if remaining <= 0 or remaining % 2:
            raise ValueError(
                f"invalid fused QKV shape {weight.shape} for {prefix!r}")
        key_value_size = remaining // 2
        selected = {
            "q_proj":
            slice(0, query_size),
            "k_proj":
            slice(query_size, query_size + key_value_size),
            "v_proj":
            slice(query_size + key_value_size,
                  query_size + 2 * key_value_size),
        }[projection]
    elif projection in ("gate_proj", "up_proj"):
        fused_prefix = parent + ".gate_up_proj"
        if not weights.has(fused_prefix + ".weight"):
            return None
        weight = weights.f16(fused_prefix + ".weight")
        if int(weight.shape[0]) % 2:
            raise ValueError(
                f"invalid fused gate/up shape {weight.shape} for {prefix!r}")
        half = int(weight.shape[0]) // 2
        selected = (slice(0, half) if projection == "gate_proj" else slice(
            half, 2 * half))
    else:
        return None
    bias = weights.opt_f16(fused_prefix + ".bias")
    return (np.ascontiguousarray(weight[selected]),
            None if bias is None else np.ascontiguousarray(bias[selected]))


def repack_nvfp4_experts(load_expert, num_experts: int, hidden_size: int,
                         intermediate_size: int, group_size: int,
                         fc1_layout: str):
    """Arrange provider-packed experts without requantization."""
    return nvfp4_pack.pack_gated_nvfp4_experts(
        load_expert,
        num_experts,
        hidden_size,
        intermediate_size,
        group_size,
        fc1_layout,
    )


def load_gptq_expert_projection(weights, experts_prefix: str,
                                expert_index: int, projection: str):
    """Load one GPTQ expert while rejecting unsupported act-order layouts."""
    prefix = f"{experts_prefix}.{expert_index}.{projection}"
    if weights.has(prefix + ".g_idx"):
        group_index = weights.array(prefix + ".g_idx").reshape(-1)
        expected = np.arange(group_index.size) // weights.group_size
        if not np.array_equal(group_index, expected):
            raise ValueError(
                "Qwen3-Omni-Next does not support act-order GPTQ: "
                f"{prefix}")
    qzeros = (weights.array(prefix + ".qzeros")
              if weights.has(prefix + ".qzeros") else np.empty(
                  (1, 0), dtype=np.int32))
    return (weights.array(prefix + ".qweight"), qzeros,
            weights.f16(prefix + ".scales"))


def prepare_fp16_experts(weights, experts_prefix: str, num_experts: int,
                         hidden_size: int, intermediate_size: int) -> dict:
    """Pack BF16/FP16 routed experts for the FP16 MoE plugin."""
    chunk_rows = 64
    if intermediate_size % chunk_rows:
        raise ValueError(
            "Qwen3-Omni-Next FP16 MoE intermediate size must be a multiple "
            f"of {chunk_rows}, got {intermediate_size}")
    chunks = intermediate_size // chunk_rows
    fc1_weights = []
    fc2_weights = []
    for expert in range(num_experts):
        prefix = f"{experts_prefix}.{expert}"
        up = weights.f16(prefix + ".up_proj.weight")
        gate = weights.f16(prefix + ".gate_proj.weight")
        down = weights.f16(prefix + ".down_proj.weight")
        expected_fc1 = (intermediate_size, hidden_size)
        expected_fc2 = (hidden_size, intermediate_size)
        if up.shape != expected_fc1 or gate.shape != expected_fc1:
            raise ValueError(f"{prefix} gate/up must be {expected_fc1}, got "
                             f"{gate.shape} and {up.shape}")
        if down.shape != expected_fc2:
            raise ValueError(
                f"{prefix}.down_proj must be {expected_fc2}, got {down.shape}")
        interleaved = np.stack(
            (up.reshape(chunks, chunk_rows, hidden_size),
             gate.reshape(chunks, chunk_rows, hidden_size)),
            axis=1,
        ).reshape(2 * intermediate_size, hidden_size)
        fc1_weights.append(np.ascontiguousarray(interleaved))
        fc2_weights.append(np.ascontiguousarray(down))
    return {
        "fc1_weights": np.stack(fc1_weights),
        "fc2_weights": np.stack(fc2_weights),
    }


def fp16_expert_specs(weights, experts_prefix: str, num_experts: int) -> dict:
    """Describe final FP16 expert buffers without loading their payloads."""
    up = weights.parameter_spec(f"{experts_prefix}.0.up_proj.weight",
                                np.float16)
    down = weights.parameter_spec(f"{experts_prefix}.0.down_proj.weight",
                                  np.float16)
    return {
        "fc1_weights":
        ParameterSpec((num_experts, 2 * up.shape[0], up.shape[1]), np.float16),
        "fc2_weights":
        ParameterSpec((num_experts, *down.shape), np.float16),
    }


def fp16_expert_bindings(weights, experts_prefix: str,
                         num_experts: int) -> dict:
    """Map provider expert tensors to the final FP16 plugin buffers."""
    fc1_names = []
    fc2_names = []
    for expert in range(num_experts):
        prefix = f"{experts_prefix}.{expert}"
        fc1_names.extend(
            (prefix + ".up_proj.weight", prefix + ".gate_proj.weight"))
        fc2_names.append(prefix + ".down_proj.weight")
    return {
        "fc1_weights":
        weights.checkpoint_binding(fc1_names,
                                   "fp16",
                                   "fp16_moe_fc1",
                                   num_experts=num_experts),
        "fc2_weights":
        weights.checkpoint_binding(fc2_names,
                                   "fp16",
                                   "fp16_moe_fc2",
                                   num_experts=num_experts),
    }


def int4_expert_bindings(weights, experts_prefix: str, num_experts: int,
                         group_size: int, zero_point_offset: int) -> dict:
    """Map per-expert GPTQ tensors to Int4MoePlugin inputs."""

    def names(projections, leaves):
        return [
            f"{experts_prefix}.{expert}.{projection}.{leaf}"
            for expert in range(num_experts) for projection in projections
            for leaf in leaves
        ]

    common = {
        "num_experts": num_experts,
        "group_size": group_size,
        "zero_point_offset": zero_point_offset,
    }
    return {
        "fc_gate_up_qweights":
        weights.checkpoint_binding(
            names(("gate_proj", "up_proj"), ("qweight", "qzeros", "g_idx")),
            "plugin", "int4_moe_gate_up", **common),
        "fc_gate_up_scales":
        weights.checkpoint_binding(
            names(("gate_proj", "up_proj"), ("scales", )), "plugin",
            "int4_moe_gate_up_scales", **common),
        "fc_down_qweights":
        weights.checkpoint_binding(
            names(("down_proj", ), ("qweight", "qzeros", "g_idx")), "plugin",
            "int4_moe_down", **common),
        "fc_down_scales":
        weights.checkpoint_binding(names(("down_proj", ), ("scales", )),
                                   "plugin", "int4_moe_down_scales", **common),
    }


def int4_expert_specs(weights, experts_prefix: str, num_experts: int) -> dict:
    """Describe final INT4 MoE buffers from checkpoint metadata."""
    gate_qweight = weights.parameter_spec(
        f"{experts_prefix}.0.gate_proj.qweight", np.int32)
    gate_scales = weights.parameter_spec(
        f"{experts_prefix}.0.gate_proj.scales", np.float16)
    down_qweight = weights.parameter_spec(
        f"{experts_prefix}.0.down_proj.qweight", np.int32)
    down_scales = weights.parameter_spec(
        f"{experts_prefix}.0.down_proj.scales", np.float16)
    hidden_size = gate_qweight.shape[0] * 8
    intermediate_size = gate_qweight.shape[1]
    down_input_size = down_qweight.shape[0] * 8
    down_output_size = down_qweight.shape[1]
    return {
        "fc_gate_up_qweights":
        ParameterSpec((num_experts, hidden_size // 16, 16 * intermediate_size),
                      np.int8),
        "fc_gate_up_scales":
        ParameterSpec(
            (num_experts, gate_scales.shape[0], 2 * intermediate_size),
            np.float16),
        "fc_down_qweights":
        ParameterSpec(
            (num_experts, down_input_size // 16, 8 * down_output_size),
            np.int8),
        "fc_down_scales":
        ParameterSpec((num_experts, down_scales.shape[0], down_output_size),
                      np.float16),
    }


def nvfp4_expert_bindings(weights, experts_prefix: str, num_experts: int,
                          sm12x: bool) -> dict:
    """Map provider-packed NVFP4 experts to architecture-specific inputs."""
    records = {
        (expert, projection):
        weights.nvfp4_checkpoint_names(
            f"{experts_prefix}.{expert}.{projection}")
        for expert in range(num_experts)
        for projection in ("up_proj", "gate_proj", "down_proj")
    }

    def fields(projections, order):
        return [
            records[expert, projection][field] for expert in range(num_experts)
            for field in order for projection in projections
        ]

    fc1_reciprocal = {
        records[expert, projection][3]
        for expert in range(num_experts)
        for projection in ("up_proj", "gate_proj")
    }
    fc2_reciprocal = {
        records[expert, "down_proj"][3]
        for expert in range(num_experts)
    }
    if len(fc1_reciprocal) != 1 or len(fc2_reciprocal) != 1:
        raise ValueError(
            "Qwen3-Omni-Next NVFP4 experts use inconsistent alpha formats")
    fc1_reciprocal = fc1_reciprocal.pop()
    fc2_reciprocal = fc2_reciprocal.pop()
    common = {
        "num_experts": num_experts,
        "fc1_layout": "concat" if sm12x else "interleave",
    }
    gate_up = ("up_proj", "gate_proj")
    down = ("down_proj", )
    return {
        "fc1_qweights":
        weights.checkpoint_binding(fields(gate_up, (0, 1, 2)),
                                   "nvfp4_qweight",
                                   "nvfp4_gated_fc1_qweight",
                                   reciprocal_alpha=fc1_reciprocal,
                                   **common),
        "fc1_blocks_scale":
        weights.checkpoint_binding(fields(gate_up, (0, 1, 2)),
                                   "nvfp4_scale_linear",
                                   "nvfp4_gated_fc1_scale",
                                   reciprocal_alpha=fc1_reciprocal,
                                   **common),
        "fc1_alpha":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "fc2_qweights":
        weights.checkpoint_binding(fields(down, (0, 1, 2)),
                                   "nvfp4_qweight",
                                   "nvfp4_gated_fc2_qweight",
                                   reciprocal_alpha=fc2_reciprocal,
                                   **common),
        "fc2_blocks_scale":
        weights.checkpoint_binding(fields(down, (0, 1, 2)),
                                   "nvfp4_scale_linear",
                                   "nvfp4_gated_fc2_scale",
                                   reciprocal_alpha=fc2_reciprocal,
                                   **common),
        "fc2_alpha":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "input_global_scale":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "down_input_scale":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "e_score_correction_bias":
        weights.checkpoint_binding([], "generated", "fill", fill_value=0.0),
    }


def nvfp4_expert_specs(weights, experts_prefix: str, num_experts: int) -> dict:
    """Describe final NVFP4 MoE buffers from checkpoint headers."""
    up_names = weights.nvfp4_checkpoint_names(f"{experts_prefix}.0.up_proj")
    down_names = weights.nvfp4_checkpoint_names(
        f"{experts_prefix}.0.down_proj")
    up_weight = weights.store.shape(up_names[0])
    up_scale = weights.store.shape(up_names[1])
    down_weight = weights.store.shape(down_names[0])
    down_scale = weights.store.shape(down_names[1])
    return {
        "fc1_qweights":
        ParameterSpec((num_experts, 2 * up_weight[0], up_weight[1]), np.int8),
        "fc1_blocks_scale":
        ParameterSpec((num_experts, (2 * up_scale[0] + 127) // 128,
                       (up_scale[1] + 3) // 4, 32, 4, 4), np.int8),
        "fc1_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "fc2_qweights":
        ParameterSpec((num_experts, *down_weight), np.int8),
        "fc2_blocks_scale":
        ParameterSpec((num_experts, (down_scale[0] + 127) // 128,
                       (down_scale[1] + 3) // 4, 32, 4, 4), np.int8),
        "fc2_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "input_global_scale":
        ParameterSpec((num_experts, ), np.float32),
        "down_input_scale":
        ParameterSpec((num_experts, ), np.float32),
        "e_score_correction_bias":
        ParameterSpec((num_experts, ), np.float32),
    }
