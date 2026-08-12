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
"""Gemma4 text decoder implementation for checkpoint-based export."""

from __future__ import annotations

import itertools
import re
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from ...checkpoint import checkpoint_utils
from ...config import (QUANT_INT4_AWQ, QUANT_INT4_AWQ_MODELOPT,
                       QUANT_INT4_GPTQ, QUANT_NVFP4, ModelConfig)
from ..default.modeling_default import (MLP, Attention, CausalLM, DecoderLayer,
                                        OnnxSpec, RMSNorm)
from ..linear import TPMode, make_linear
from ..ops import (KV_PAGE_SIZE, attention_plugin, int4_moe_plugin,
                   nvfp4_moe_plugin, nvfp4_moe_plugin_geforce,
                   use_geforce_nvfp4_moe)

__all__ = [
    "Gemma4Attention",
    "Gemma4ForCausalLM",
    "Gemma4DecoderLayer",
    "Gemma4DenseMoEBlock",
    "Gemma4Int4MoEBlock",
    "Gemma4NvFP4MoEBlock",
    "Gemma4NvFP4MoEExperts",
    "Gemma4Transformer",
    "Gemma4ValueRMSNorm",
    "GEMMA4_NVFP4_KEY_REMAP",
    "_gemma4_dense_moe_routing",
    "GEMMA4_FUSED_BF16_KEY_REMAP",
]

# Plugin constants for ``Nvfp4MoePlugin`` (same as Qwen3 MoE).
_NVFP4_ROUTING_MODE_SOFTMAX_TOPK_POST_SCALE = 2
_NVFP4_ACTIVATION_GEGLU = 5
_NVFP4_MOE_BACKEND_AUTO = 0
_NVFP4_MOE_IO_DTYPE_FP16 = 1
_NVFP4_MOE_MAX_ROUTED_ROWS_AUTO = 0
_NVFP4_MOE_N_GROUP_FLAT = 1
_NVFP4_MOE_TOPK_GROUP_FLAT = 1

# Int4MoePlugin activation type for GeGLU (matches C++ kACTIVATION_GEGLU = 5).
_INT4_ACTIVATION_GEGLU = 5

# These are dummy tensor extents used only to seed torch.export/ONNX export.
# Runtime limits are controlled by dynamic_shapes and the builder profiles.
_DUMMY_BATCH_SIZE = 1
_DUMMY_SEQ_LEN = 1
_DUMMY_PAST_LEN = 1
_DUMMY_ROPE_CACHE_LEN = 4096


def _attention_type_for_layer(config: ModelConfig, layer_idx: int) -> str:
    """Return Gemma4's per-layer attention type."""
    if layer_idx >= len(config.attention_layer_types):
        raise ValueError(
            "Gemma4 attention_layer_types must have one entry per layer; "
            f"missing layer {layer_idx}.")
    return config.attention_layer_types[layer_idx]


def _uses_attention_k_eq_v(config: ModelConfig, attention_type: str) -> bool:
    """Return whether a Gemma4 attention layer reuses K as the V source."""
    return bool(config.attention_k_eq_v and attention_type == "full_attention")


def _head_dim_for_attention_type(config: ModelConfig,
                                 attention_type: str) -> int:
    """Return Gemma4's per-layer attention head dimension."""
    if attention_type == "full_attention" and config.global_head_dim:
        return int(config.global_head_dim)
    return int(config.head_dim)


def _num_kv_heads_for_attention_type(config: ModelConfig,
                                     attention_type: str) -> int:
    """Return Gemma4's per-layer KV head count."""
    if (_uses_attention_k_eq_v(config, attention_type)
            and config.num_global_key_value_heads):
        return int(config.num_global_key_value_heads)
    return int(config.num_key_value_heads)


def _gemma4_dense_moe_routing(
    router_logits: torch.Tensor,
    top_k: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gemma4 dense-MoE top-k expert weights and expert IDs."""
    expert_ids = torch.topk(router_logits, k=top_k, dim=-1).indices
    routing_weights = torch.softmax(router_logits.float(),
                                    dim=-1).gather(1, expert_ids)
    routing_weights = routing_weights / routing_weights.sum(dim=-1,
                                                            keepdim=True)
    routing_weights = routing_weights * per_expert_scale.to(
        dtype=routing_weights.dtype, device=routing_weights.device)[expert_ids]
    return routing_weights, expert_ids


def _kv_cache_dims_for_layer(config: ModelConfig,
                             layer_idx: int) -> tuple[int, int]:
    """Return (num_kv_heads, head_dim) for one Gemma4 KV-cache input."""
    attention_type = _attention_type_for_layer(config, layer_idx)
    return (_num_kv_heads_for_attention_type(config, attention_type),
            _head_dim_for_attention_type(config, attention_type))


def _mlp_intermediate_size_for_layer(config: ModelConfig,
                                     layer_idx: int) -> int:
    """Return Gemma4's per-layer MLP width."""
    intermediate_size = int(config.intermediate_size)
    if not config.use_double_wide_mlp:
        return intermediate_size
    first_shared_layer = int(config.num_hidden_layers -
                             config.num_kv_shared_layers)
    if layer_idx >= first_shared_layer:
        return intermediate_size * 2
    return intermediate_size


def _compute_kv_donor_indices(config: ModelConfig) -> dict[int, int]:
    """Return shared-layer -> donor-layer KV indices for Gemma4."""
    num_shared = int(getattr(config, "num_kv_shared_layers", 0) or 0)
    if num_shared <= 0:
        return {}

    num_layers = int(config.num_hidden_layers)
    if num_shared > num_layers:
        raise ValueError(
            "Gemma4 num_kv_shared_layers cannot exceed num_hidden_layers: "
            f"{num_shared} > {num_layers}.")

    first_shared = num_layers - num_shared
    donors_by_type: dict[str, int] = {}
    for layer_idx in range(first_shared):
        donors_by_type[_attention_type_for_layer(config,
                                                 layer_idx)] = layer_idx

    donor_map: dict[int, int] = {}
    for layer_idx in range(first_shared, num_layers):
        attention_type = _attention_type_for_layer(config, layer_idx)
        if attention_type not in donors_by_type:
            raise ValueError(
                "Gemma4 shared KV layer has no compatible donor before the "
                f"shared range: layer={layer_idx}, type={attention_type}.")
        donor_map[layer_idx] = donors_by_type[attention_type]
    return donor_map


def _rotary_dim_from_rope_config(config: ModelConfig,
                                 rope_config: dict | None,
                                 head_dim: int | None = None) -> int:
    """Return the RoPE table width for one Gemma4 runtime RoPE config."""
    effective_head_dim = head_dim if head_dim is not None else int(
        config.head_dim)
    if rope_config is None:
        rope_config = {
            "rope_scaling": config.rope_scaling,
            "partial_rotary_factor": config.partial_rotary_factor,
        }
    return checkpoint_utils.rotary_dim_for_runtime(
        rope_config, effective_head_dim, config.partial_rotary_factor)


def _select_rope_for_layer(
    layer: nn.Module,
    rope_rotary_cos_sin: torch.Tensor | None,
    rope_rotary_cos_sin_sliding: torch.Tensor | None,
    rope_rotary_cos_sin_full: torch.Tensor | None,
) -> torch.Tensor:
    """Select the Gemma4 RoPE table matching ``layer`` attention type."""
    if (rope_rotary_cos_sin_sliding is None
            and rope_rotary_cos_sin_full is None):
        if rope_rotary_cos_sin is None:
            raise ValueError(
                "rope_rotary_cos_sin is required for single-RoPE Gemma4 export."
            )
        return rope_rotary_cos_sin

    attention_type = getattr(layer.self_attn, "attention_type",
                             "full_attention")
    if attention_type == "sliding_attention":
        if rope_rotary_cos_sin_sliding is None:
            raise ValueError(
                "rope_rotary_cos_sin_sliding is required for Gemma4 sliding attention layers."
            )
        return rope_rotary_cos_sin_sliding

    if rope_rotary_cos_sin_full is None:
        raise ValueError(
            "rope_rotary_cos_sin_full is required for Gemma4 full attention layers."
        )
    return rope_rotary_cos_sin_full


def _make_gemma4_flat_wrapper(model: nn.Module,
                              Na: int,
                              num_ple_inputs: int,
                              use_dual_rope: bool = False,
                              eagle_base: bool = False,
                              vision_block_attention: bool = False,
                              emit_hidden_states: bool = False) -> nn.Module:
    """Build a Gemma4 export wrapper with explicit PLE/RoPE tensor inputs."""
    has_hidden_output = eagle_base or emit_hidden_states

    param_names: List[str] = (
        ["inputs_embeds"] +
        [f"ple_token_embeds_{i}" for i in range(num_ple_inputs)] +
        [f"past_key_values_{i}" for i in range(Na)])
    if use_dual_rope:
        param_names += [
            "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full"
        ]
    else:
        param_names += ["rope_rotary_cos_sin"]
    param_names += [
        "context_lengths", "kvcache_start_index", "kv_page_table",
        "last_token_ids"
    ]
    if vision_block_attention:
        param_names += ["vision_block_ids"]
    if eagle_base:
        param_names += ["attention_pos_id", "attention_mask"]

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    ple_tuple = "({},)".format(", ".join(
        f"ple_token_embeds_{i}"
        for i in range(num_ple_inputs))) if num_ple_inputs else "()"
    ple_kwarg = f", ple_token_embeds={ple_tuple}" if num_ple_inputs > 0 else ""
    eagle_kwargs = (", attention_mask=attention_mask"
                    ", attention_pos_id=attention_pos_id"
                    if eagle_base else "")
    vision_kwargs = (", vision_block_ids=vision_block_ids"
                     if vision_block_attention else "")
    if use_dual_rope:
        rope_arg = "None"
        rope_kwargs = (
            ", rope_rotary_cos_sin_sliding=rope_rotary_cos_sin_sliding"
            ", rope_rotary_cos_sin_full=rope_rotary_cos_sin_full")
    else:
        rope_arg = "rope_rotary_cos_sin"
        rope_kwargs = ""

    if has_hidden_output:
        body = (
            f"    logits, hidden_states, present_key_values = self._model(\n"
            f"        inputs_embeds, {past_kv_tuple}, {rope_arg}, "
            f"context_lengths, kvcache_start_index, kv_page_table, "
            f"last_token_ids"
            f"{eagle_kwargs}{vision_kwargs}{ple_kwarg}{rope_kwargs})\n"
            f"    return (logits, hidden_states) + tuple(present_key_values)\n"
        )
    else:
        body = (f"    logits, present_key_values = self._model(\n"
                f"        inputs_embeds, {past_kv_tuple}, {rope_arg}, "
                f"context_lengths, kvcache_start_index, kv_page_table, "
                f"last_token_ids"
                f"{eagle_kwargs}{vision_kwargs}{ple_kwarg}{rope_kwargs})\n"
                f"    return (logits,) + tuple(present_key_values)\n")

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


def _resolve_hidden_activation(
        activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the Gemma4 PLE gate activation."""
    if activation_name in ACT2FN:
        return ACT2FN[activation_name]
    raise ValueError(
        f"Unsupported hidden_activation for Gemma4 PLE gate: {activation_name!r}"
    )


def _compute_kv_donor_indices(config: ModelConfig) -> dict:
    """Compute the KV donor layer index for each KV-shared layer.

    Returns a dict mapping shared layer_idx -> donor layer_idx.
    Donor is the last non-shared layer of the same type (sliding/full).
    """
    num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
    if num_kv_shared <= 0:
        return {}
    n = config.num_hidden_layers
    first_shared = n - num_kv_shared
    layer_types = (list(config.attention_layer_types)
                   if config.attention_layer_types else [])

    # Find last non-shared layer of each type
    prev_layers = layer_types[:first_shared]
    donors: dict = {}
    for lt in set(prev_layers):
        donors[lt] = first_shared - 1 - prev_layers[::-1].index(lt)

    result: dict = {}
    for i in range(first_shared, n):
        if i >= len(layer_types):
            raise ValueError(
                f"Layer index {i} exceeds attention_layer_types length "
                f"({len(layer_types)}). Check num_hidden_layers vs "
                f"attention_layer_types in config.")
        lt = layer_types[i]
        if lt not in donors:
            raise ValueError(f"KV-shared layer {i} has type '{lt}' with no "
                             f"non-shared donor layer of the same type.")
        result[i] = donors[lt]
    return result


class Gemma4RMSNorm(RMSNorm):
    """RMSNorm with f32 weight storage for Gemma4.

    Gemma4 31B has norm weights up to 1248 which overflow fp16 when multiplied
    with normalized values. Weight is stored as f32 and multiplication is done
    in f32 before casting back to input dtype. This also ensures both operands
    have matching type in the ONNX graph (required by TRT --stronglyTyped).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        nn.Module.__init__(self)
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size,
                                              dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        hidden_states = hidden_states * self.weight
        return hidden_states.to(input_dtype)


class Gemma4ValueRMSNorm(nn.Module):
    """Weightless per-head RMSNorm used by Gemma4 attention values."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return hidden_states.to(input_dtype)


class Gemma4Attention(Attention):
    """Gemma4 attention with HF-compatible value norm, K=V, and QK scaling.

    KV-shared layers (index >= num_hidden_layers - num_kv_shared_layers) do not
    compute their own K/V.  They reuse the KV cache of the donor layer (the last
    preceding layer of the same attention type before the shared range).

    Full-attention layers use global_head_dim (512) instead of head_dim (256).
    """

    def __init__(self,
                 config: ModelConfig,
                 layer_idx: int,
                 in_features: int = 0) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.attention_type = _attention_type_for_layer(config, layer_idx)
        self.attention_k_eq_v = _uses_attention_k_eq_v(config,
                                                       self.attention_type)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = _num_kv_heads_for_attention_type(
            config, self.attention_type)
        self.head_dim = _head_dim_for_attention_type(config,
                                                     self.attention_type)
        self.enable_fp8_kv_cache = config.quant.kv_cache_quant == "fp8"
        hidden_size = int(config.hidden_size)
        qkv_in_features = int(in_features or hidden_size)
        module_prefix = f"layers.{layer_idx}.self_attn"

        self.q_proj = make_linear(config,
                                  qkv_in_features,
                                  self.num_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.q_proj")
        self.k_proj = make_linear(config,
                                  qkv_in_features,
                                  self.num_kv_heads * self.head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.k_proj")
        if self.attention_k_eq_v:
            # K=V: forward uses key_states as value_states, but we still
            # instantiate v_proj so checkpoint loading can assign its weight.
            self.v_proj = make_linear(config,
                                      qkv_in_features,
                                      self.num_kv_heads * self.head_dim,
                                      bias=config.attention_bias,
                                      module_name=f"{module_prefix}.v_proj")
        else:
            self.v_proj = make_linear(config,
                                      qkv_in_features,
                                      self.num_kv_heads * self.head_dim,
                                      bias=config.attention_bias,
                                      module_name=f"{module_prefix}.v_proj")

        if self.enable_fp8_kv_cache:
            self.q_proj.register_buffer("q_scale", torch.ones(1))
            self.k_proj.register_buffer("k_scale", torch.ones(1))
            if self.v_proj is not None:
                self.v_proj.register_buffer("v_scale", torch.ones(1))

        self.o_proj = make_linear(config,
                                  self.num_heads * self.head_dim,
                                  hidden_size,
                                  module_name=f"{module_prefix}.o_proj")
        if config.has_qk_norm:
            self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # KV-sharing: layers in the shared range reuse a donor's KV cache.
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_shared = (config.num_hidden_layers - num_kv_shared
                        if num_kv_shared > 0 else config.num_hidden_layers)
        self.is_kv_shared = layer_idx >= first_shared

        # KV-shared layers don't use k_proj/v_proj/k_norm — remove them
        # so their weights are not loaded from the checkpoint.
        if self.is_kv_shared:
            del self.k_proj
            if self.v_proj is not None:
                del self.v_proj
            if hasattr(self, "k_norm") and self.k_norm is not None:
                del self.k_norm

        self.attention_scale = config.attention_scaling
        if not self.is_kv_shared:
            self.v_norm = (Gemma4ValueRMSNorm(self.head_dim,
                                              config.rms_norm_eps)
                           if config.has_value_norm else None)
        else:
            self.v_norm = None
        self.sliding_window_size = (config.sliding_window_size
                                    if self.attention_type
                                    == "sliding_attention" else -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
        vision_block_ids: torch.Tensor | None = None,
        context_mask_selector: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)

        if self.is_kv_shared:
            # Shared-KV layer (enable_kv_shared=1): qkv carries Q only; K/V come
            # from the donor layer's cache (past_key_value).
            key_states = None
            value_states = None
        else:
            key_states = self.k_proj(hidden_states)
            if self.attention_k_eq_v:
                value_states = key_states
            else:
                value_states = self.v_proj(hidden_states)

        if self.q_norm is not None:
            query_states = self.q_norm(
                query_states.reshape(batch_size, seq_len, self.num_heads,
                                     self.head_dim)).reshape(
                                         batch_size, seq_len,
                                         self.num_heads * self.head_dim)
        if not self.is_kv_shared:
            if self.k_norm is not None:
                key_states = self.k_norm(
                    key_states.reshape(batch_size, seq_len, self.num_kv_heads,
                                       self.head_dim)).reshape(
                                           batch_size, seq_len,
                                           self.num_kv_heads * self.head_dim)

            if self.v_norm is not None:
                value_states = self.v_norm(
                    value_states.reshape(batch_size, seq_len,
                                         self.num_kv_heads,
                                         self.head_dim)).reshape(
                                             batch_size, seq_len,
                                             self.num_kv_heads * self.head_dim)

        enable_tree = attention_mask is not None and attention_pos_id is not None
        enable_vision_block = vision_block_ids is not None
        if enable_tree and enable_vision_block:
            raise ValueError(
                "Gemma4 vision block attention and tree attention are mutually exclusive."
            )
        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": enable_tree,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
            "attention_scale": self.attention_scale,
            "enable_context_mask_selector": context_mask_selector is not None,
            "enable_vision_block_attention": enable_vision_block,
            "skip_softmax_scale_factor": 0.0,
        }
        if context_mask_selector is not None:
            kwargs["context_mask_selector"] = context_mask_selector
        if enable_tree:
            kwargs["attention_mask"] = attention_mask
            kwargs["attention_pos_id"] = attention_pos_id
        elif enable_vision_block:
            # AttentionPlugin input slot 7 is shared with the tree mask.  The
            # static plugin attribute selects its [B,S] block-ID semantics.
            kwargs["attention_mask"] = vision_block_ids
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])

        # Packed QKV input: Q-only for shared-KV layers, Q+K+V otherwise.
        if key_states is None:
            qkv = query_states
            kwargs["enable_kv_shared"] = 1
        else:
            qkv = torch.cat([query_states, key_states, value_states], dim=-1)
        attn_output, present_key_value = attention_plugin(
            qkv,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            kv_page_table,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


class Gemma4MLP(MLP):
    """Gemma4 MLP using the checkpoint-configured activation."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        nn.Module.__init__(self)
        intermediate_size = _mlp_intermediate_size_for_layer(config, layer_idx)
        module_prefix = f"layers.{layer_idx}.mlp"
        self.gate_proj = make_linear(
            config,
            config.hidden_size,
            intermediate_size,
            module_name=f"{module_prefix}.gate_proj",
            tp_mode=TPMode.COL,
        )
        self.up_proj = make_linear(
            config,
            config.hidden_size,
            intermediate_size,
            module_name=f"{module_prefix}.up_proj",
            tp_mode=TPMode.COL,
        )
        self.down_proj = make_linear(
            config,
            intermediate_size,
            config.hidden_size,
            module_name=f"{module_prefix}.down_proj",
            tp_mode=TPMode.ROW,
        )
        self.act_fn = _resolve_hidden_activation(config.hidden_activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compute gate*up in f32 to prevent fp16 overflow.
        # Clamp intermediate to +/-2048 before casting to fp16 for down_proj.
        # Without this clamp, the fp16 down_proj MatMul can produce Inf
        # (dot product of 21504 elements at +/-65504 overflows fp16 output range),
        # which then causes NaN in the subsequent RMSNorm (Inf*0=NaN).
        gate = self.act_fn(self.gate_proj(hidden_states).to(torch.float32))
        up = self.up_proj(hidden_states).to(torch.float32)
        intermediate = (gate * up).clamp(-2048.0, 2048.0)
        return self.down_proj(intermediate.to(hidden_states.dtype))


class Gemma4Router(nn.Module):
    """Gemma4 MoE router weights for checkpoint loading.

    Holds norm, scale, proj, and per_expert_scale parameters that get repacked
    by Gemma4NvFP4MoEBlock._prepare_moe_weights(). The TRT plugin handles
    softmax + topk internally.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps

        # Weightless RMSNorm (no learnable scale parameter)
        self.norm = Gemma4ValueRMSNorm(self.hidden_size, eps=self.eps)
        self.proj = make_linear(config,
                                self.hidden_size,
                                self.num_experts,
                                bias=False,
                                module_name=f"layers.{layer_idx}.router.proj")
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))


class Gemma4NvFP4MoEExperts(nn.Module):
    """Per-expert NVFP4 linear modules for Gemma4 MoE checkpoint loading.

    Mirrors :class:`Qwen3MoEExperts`: each expert has gate_proj, up_proj,
    down_proj created via ``make_linear()`` which returns ``NVFP4Linear``
    when ``config.quant.quant_type == QUANT_NVFP4``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.moe_intermediate_size
        experts = []
        for _ in range(config.num_experts):
            expert = nn.Module()
            expert.gate_proj = make_linear(config, hidden, inter)
            expert.up_proj = make_linear(config, hidden, inter)
            expert.down_proj = make_linear(config, inter, hidden)
            experts.append(expert)
        self._experts = nn.ModuleList(experts)

    def __getitem__(self, idx: int) -> nn.Module:
        return self._experts[idx]

    def __len__(self) -> int:
        return len(self._experts)

    def __iter__(self):
        return iter(self._experts)


class Gemma4DenseMoEBlock(nn.Module):
    """Dense Gemma4 MoE fallback for non-NVFP4 checkpoints."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.router = Gemma4Router(config, layer_idx)
        self.experts = Gemma4NvFP4MoEExperts(config)
        self.act_fn = _resolve_hidden_activation(config.hidden_activation)

    def forward(self, expert_input: torch.Tensor,
                residual: torch.Tensor) -> torch.Tensor:
        hidden_flat = residual.reshape(-1, self.hidden_size)
        normed = self.router.norm(hidden_flat)
        scaled = normed * (self.router.scale *
                           self.router.scalar_root_size).to(normed.dtype)
        router_logits = self.router.proj(scaled).float()
        routing_weights, expert_ids = _gemma4_dense_moe_routing(
            router_logits, self.top_k, self.router.per_expert_scale)

        output = torch.zeros_like(expert_input)
        for expert_idx, expert in enumerate(self.experts):
            gate = self.act_fn(
                expert.gate_proj(expert_input).to(torch.float32))
            up = expert.up_proj(expert_input).to(torch.float32)
            expert_output = expert.down_proj(
                (gate * up).to(expert_input.dtype))
            expert_weight = torch.sum(
                torch.where(
                    expert_ids == expert_idx,
                    routing_weights,
                    torch.zeros_like(routing_weights),
                ),
                dim=-1,
                keepdim=True,
            )
            output = output + expert_output * expert_weight.to(output.dtype)
        return output


class Gemma4NvFP4MoEBlock(nn.Module):
    """NVFP4 MoE block for Gemma4 26B-A4B using ``Nvfp4MoePlugin``.

    Wraps the router + NVFP4 per-expert weights for checkpoint loading,
    then repacks into plugin-compatible layout via ``_prepare_moe_weights()``.

    Forward path: Router RMSNorm + scale + proj produces raw logits; the
    plugin handles softmax + topk + expert GEMMs internally.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self._padded_moe_intermediate_size = self.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.group_size = config.quant.group_size
        self.activation_type = _NVFP4_ACTIVATION_GEGLU
        self.backend = _NVFP4_MOE_BACKEND_AUTO
        self.io_dtype = _NVFP4_MOE_IO_DTYPE_FP16
        self.max_routed_rows = _NVFP4_MOE_MAX_ROUTED_ROWS_AUTO

        self.router = Gemma4Router(config, layer_idx)
        self.experts = Gemma4NvFP4MoEExperts(config)

    def _prepare_moe_weights(self) -> None:
        """Repack NVFP4 experts for Nvfp4MoePlugin.

        Called by :func:`~checkpoint.repacking._stack_moe_experts`.
        """
        from ...checkpoint.repacking import (
            NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT,
            NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT,
            repack_nvfp4_gated_moe_experts)

        use_geforce_plugin = use_geforce_nvfp4_moe()
        fc1_layout = "concat" if use_geforce_plugin else "interleave"
        moe_inter_size_alignment = (NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT
                                    if use_geforce_plugin else
                                    NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT)
        fc1_qweights, fc1_blocks_scale, fc2_qweights, fc2_blocks_scale = (
            repack_nvfp4_gated_moe_experts(
                self.experts,
                self.hidden_size,
                self.moe_intermediate_size,
                self.group_size,
                fc1_layout=fc1_layout,
                moe_inter_size_alignment=moe_inter_size_alignment))
        self._padded_moe_intermediate_size = int(fc2_qweights.shape[-1]) * 2

        device = self.router.proj.weight.device
        self.register_buffer("fc1_qweights",
                             fc1_qweights.to(device).contiguous())
        self.register_buffer("fc1_blocks_scale",
                             fc1_blocks_scale.to(device).contiguous())
        self.register_buffer("fc2_qweights",
                             fc2_qweights.to(device).contiguous())
        self.register_buffer("fc2_blocks_scale",
                             fc2_blocks_scale.to(device).contiguous())

        # w4a16: weights are NVFP4, activations stay FP16.
        # repack_nvfp4_gated_moe_experts decodes weights to dense (folding
        # weight_scale_2 in) then re-quantizes → alpha must be 1.0.
        # No activation quantization → input scales are also 1.0.
        self.register_buffer(
            "fc1_alpha",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer(
            "fc2_alpha",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer(
            "input_global_scale",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))
        self.register_buffer(
            "down_input_scale",
            torch.ones(self.num_experts, dtype=torch.float32, device=device))

        # per_expert_scale → raw scale applied post-renorm by plugin
        # (routing_mode=2 triggers multiplicative post-topk application).
        self.register_buffer(
            "e_score_correction_bias",
            self.router.per_expert_scale.data.float().to(device))

        # Discard per-expert modules after repacking.
        self.experts = nn.ModuleList()

    def forward(self, expert_input: torch.Tensor,
                residual: torch.Tensor) -> torch.Tensor:
        """Route via plugin: router_logits → Nvfp4MoePlugin.

        Args:
            expert_input: [num_tokens, H] — pre-normed expert input (2D).
            residual: [B, S, H] — pre-MLP residual used for routing.
        """
        hidden_flat = residual.reshape(-1, self.hidden_size)
        # Router: RMSNorm + scale + proj → raw logits (softmax done by plugin)
        normed = self.router.norm(hidden_flat)
        scaled = normed * (self.router.scale *
                           self.router.scalar_root_size).to(normed.dtype)
        router_logits = self.router.proj(scaled).float()

        moe_op = (nvfp4_moe_plugin_geforce
                  if use_geforce_nvfp4_moe() else nvfp4_moe_plugin)
        return moe_op(
            router_logits,
            expert_input.unsqueeze(0),  # Plugin expects 3D [B, T, H]
            self.fc1_qweights,
            self.fc1_blocks_scale,
            self.fc1_alpha,
            self.fc2_qweights,
            self.fc2_blocks_scale,
            self.fc2_alpha,
            self.input_global_scale,
            self.down_input_scale,
            self.e_score_correction_bias,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self._padded_moe_intermediate_size,
            self.activation_type,
            _NVFP4_MOE_N_GROUP_FLAT,
            _NVFP4_MOE_TOPK_GROUP_FLAT,
            1,
            1.0,
            _NVFP4_ROUTING_MODE_SOFTMAX_TOPK_POST_SCALE,
            self.backend,
            self.io_dtype,
            self.max_routed_rows,
        )


class Gemma4FusedBF16MoEExperts(nn.Module):
    """Fused BF16 expert weights for QAT-unquantized Gemma4 MoE checkpoint.

    Stores gate_up_proj [E, 2*inter, hidden] and down_proj [E, hidden, inter]
    as plain parameters, matching the checkpoint's fused tensor layout.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        E = config.num_experts
        H = config.hidden_size
        I = config.moe_intermediate_size
        # Register as parameters so state_dict loading can populate them.
        self.gate_up_proj = nn.Parameter(
            torch.empty(E, 2 * I, H, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(
            torch.empty(E, H, I, dtype=torch.bfloat16))


class Gemma4Int4MoEBlock(nn.Module):
    """INT4 AWQ MoE block for Gemma4 26B-A4B using ``Int4MoePlugin``.

    For BF16 QAT-unquantized checkpoints: loads fused BF16 expert weights,
    then applies per-group INT4 RTN quantization and packs to Marlin format
    during _prepare_moe_weights().

    For pre-quantized INT4 checkpoints (GPTQ/AWQ): loads per-expert quantized
    weights and repacks to Marlin format.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        _gs = getattr(config.quant, 'group_size', 128)
        # Default group_size=1 means "unset" for BF16 QAT → pick largest
        # power-of-2 that divides both hidden_size and moe_intermediate_size.
        if _gs <= 1:
            _gs = 128
            while _gs > 1 and (config.hidden_size % _gs != 0
                               or config.moe_intermediate_size % _gs != 0):
                _gs //= 2
        self.group_size = _gs
        self.zero_point_offset = getattr(config.quant,
                                         'gptq_zero_point_offset', 1)
        self.activation_type = _INT4_ACTIVATION_GEGLU
        self.quantize_experts_from_bf16 = getattr(config,
                                                  '_needs_moe_quantization',
                                                  False)

        self.router = Gemma4Router(config, layer_idx)
        if self.quantize_experts_from_bf16:
            self.experts = Gemma4FusedBF16MoEExperts(config)
        else:
            self.experts = Gemma4NvFP4MoEExperts(config)

    def _prepare_moe_weights(self) -> None:
        """Quantize (if BF16) and repack expert weights to Marlin INT4 format.

        Called by :func:`~checkpoint.repacking._stack_moe_experts`.
        """

        # Promote router projection to Linear for standard MatMul trace.
        self.gate_linear = make_linear(self.config,
                                       self.hidden_size,
                                       self.num_experts,
                                       bias=False,
                                       module_name="moe_block.gate_linear")
        self.gate_linear.weight.data = self.router.proj.weight.data

        if self.quantize_experts_from_bf16:
            self._prepare_from_fused_bf16()
        else:
            self._prepare_from_gptq()

        # per_expert_scale → raw scale applied post-renorm by plugin
        self.register_buffer("e_score_correction_bias",
                             self.router.per_expert_scale.data.float())

        # Discard expert modules after repacking.
        self.experts = nn.ModuleList()

    def _quantize_int4_rtn(self, weight: torch.Tensor) -> tuple:
        """Apply per-group INT4 RTN (Round-To-Nearest) quantization.

        Args:
            weight: [N, K] float tensor (already transposed for Marlin: N=out, K=in)

        Returns:
            (qweight_uint [N, K] int16, scales [N, K//group] fp16)
            qweight values are unsigned [0, 15] with zero_point=8
            (Marlin dequant: (q - 8) * scale)
        """
        N, K = weight.shape
        G = self.group_size
        assert K % G == 0, f"K={K} must be divisible by group_size={G}"

        # Reshape to [N, K//G, G] for per-group quantization.
        w = weight.float().reshape(N, K // G, G)

        # Symmetric quantization: scale = max(abs(group)) / 7
        # Signed range: [-8, 7], unsigned = signed + 8 → [0, 15]
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        scales = absmax / 7.0  # [N, K//G, 1]

        # Quantize to signed [-8, 7], then offset to unsigned [0, 15]
        qw_signed = (w / scales).round().clamp(-8, 7)
        qw = (qw_signed + 8).to(torch.int16)  # [N, K//G, G] unsigned [0, 15]
        qw = qw.reshape(N, K)  # [N, K]
        scales = scales.squeeze(-1).half()  # [N, K//G]

        return qw, scales

    def _prepare_from_fused_bf16(self) -> None:
        """Quantize fused BF16 expert weights to INT4 and pack to Marlin."""
        from ...checkpoint.repacking import pack_int4_awq_marlin

        gate_up = self.experts.gate_up_proj.data  # [E, 2*I, H]
        down = self.experts.down_proj.data  # [E, H, I]

        E = self.num_experts
        gate_up_w_list = []
        gate_up_s_list = []
        down_w_list = []
        down_s_list = []

        for e in range(E):
            # gate_up_proj: [2*I, H] — already in [N, K] form for Marlin
            gu_qw, gu_s = self._quantize_int4_rtn(gate_up[e])
            gate_up_w_list.append(gu_qw)
            gate_up_s_list.append(gu_s)

            # down_proj: [H, I] — already in [N, K] form for Marlin
            d_qw, d_s = self._quantize_int4_rtn(down[e])
            down_w_list.append(d_qw)
            down_s_list.append(d_s)

        # Stack: [E, N, K] for weights, [E, N, K//G] for scales
        gate_up_w = torch.stack(gate_up_w_list, dim=0)
        gate_up_s = torch.stack(gate_up_s_list, dim=0)
        down_w = torch.stack(down_w_list, dim=0)
        down_s = torch.stack(down_s_list, dim=0)

        gu_marlin_w, gu_marlin_s = pack_int4_awq_marlin(
            gate_up_w, gate_up_s, self.group_size)
        dn_marlin_w, dn_marlin_s = pack_int4_awq_marlin(
            down_w, down_s, self.group_size)

        device = self.router.proj.weight.device
        self.register_buffer(
            "fc_gate_up_qweights",
            gu_marlin_w.view(torch.int8).to(device).contiguous())
        self.register_buffer("fc_gate_up_scales",
                             gu_marlin_s.to(device).contiguous())
        self.register_buffer(
            "fc_down_qweights",
            dn_marlin_w.view(torch.int8).to(device).contiguous())
        self.register_buffer("fc_down_scales",
                             dn_marlin_s.to(device).contiguous())

    def _prepare_from_gptq(self) -> None:
        """Extract pre-quantized GPTQ weights and repack to Marlin."""
        from ...checkpoint.repacking import (_extract_gptq_for_marlin,
                                             pack_int4_awq_marlin)

        gate_up_weights_list = []
        gate_up_scales_list = []
        down_weights_list = []
        down_scales_list = []

        for expert in self.experts:
            gw, gs = _extract_gptq_for_marlin(expert.gate_proj,
                                              self.group_size,
                                              self.zero_point_offset)
            uw, us = _extract_gptq_for_marlin(expert.up_proj, self.group_size,
                                              self.zero_point_offset)
            gate_up_weights_list.append(torch.cat([gw, uw], dim=0))
            gate_up_scales_list.append(torch.cat([gs, us], dim=0))

            dw, ds = _extract_gptq_for_marlin(expert.down_proj,
                                              self.group_size,
                                              self.zero_point_offset)
            down_weights_list.append(dw)
            down_scales_list.append(ds)

        gate_up_w = torch.stack(gate_up_weights_list, dim=0)
        gate_up_s = torch.stack(gate_up_scales_list, dim=0)
        down_w = torch.stack(down_weights_list, dim=0)
        down_s = torch.stack(down_scales_list, dim=0)

        gu_marlin_w, gu_marlin_s = pack_int4_awq_marlin(
            gate_up_w, gate_up_s, self.group_size)
        dn_marlin_w, dn_marlin_s = pack_int4_awq_marlin(
            down_w, down_s, self.group_size)

        self.register_buffer("fc_gate_up_qweights",
                             gu_marlin_w.view(torch.int8).contiguous())
        self.register_buffer("fc_gate_up_scales", gu_marlin_s.contiguous())
        self.register_buffer("fc_down_qweights",
                             dn_marlin_w.view(torch.int8).contiguous())
        self.register_buffer("fc_down_scales", dn_marlin_s.contiguous())

    def forward(self, expert_input: torch.Tensor,
                residual: torch.Tensor) -> torch.Tensor:
        """Route via Int4MoePlugin: router_logits → INT4 expert GEMMs.

        Args:
            expert_input: [num_tokens, H] — pre-normed expert input (2D).
            residual: [B, S, H] — pre-MLP residual used for routing.
        """
        hidden_flat = residual.reshape(-1, self.hidden_size)
        # Router: RMSNorm + scale + proj → raw logits (softmax done by plugin)
        normed = self.router.norm(hidden_flat)
        scaled = normed * (self.router.scale *
                           self.router.scalar_root_size).to(normed.dtype)
        router_logits = self.gate_linear(scaled).float()

        return int4_moe_plugin(
            router_logits,
            expert_input.unsqueeze(0),  # Plugin expects 3D [B, T, H]
            self.fc_gate_up_qweights,
            self.fc_gate_up_scales,
            self.fc_down_qweights,
            self.fc_down_scales,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.moe_intermediate_size,
            self.activation_type,
            self.group_size,
        )


class Gemma4DecoderLayer(DecoderLayer):
    """Gemma4 decoder layer with per-layer input injection."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.self_attn = Gemma4Attention(config, layer_idx=layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx=layer_idx)
        self.hidden_size_per_layer_input = int(
            config.hidden_size_per_layer_input)
        self.act_fn = _resolve_hidden_activation(config.hidden_activation)

        # Gemma4 uses 4 distinct RMSNorm layers per decoder block:
        #   input_layernorm            -> pre-attention (inherited from super)
        #   post_attention_layernorm   -> post-attention, before residual add (inherited)
        #   pre_feedforward_layernorm  -> pre-MLP
        #   post_feedforward_layernorm -> post-MLP, before residual add
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size,
                                                       config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps)

        # layer_scalar is applied unconditionally in HF Gemma4 - it scales the
        # residual stream per-layer (early layers have very small values ~0.06-0.09).
        self.register_buffer("layer_scalar", torch.ones(1))

        _INT4_QUANT_TYPES = (QUANT_INT4_AWQ, QUANT_INT4_AWQ_MODELOPT,
                             QUANT_INT4_GPTQ)

        # MoE block: parallel routed experts alongside dense MLP (Gemma4 26B).
        # NVFP4 checkpoints use the TRT plugin; dense weights use a reference
        # fallback for export smoke tests and non-quantized checkpoints.
        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            if config.quant.quant_type == QUANT_NVFP4:
                self.moe_block = Gemma4NvFP4MoEBlock(config, layer_idx)
            elif (config.quant.quant_type in _INT4_QUANT_TYPES
                  or getattr(config, '_use_int4_moe_plugin', False)):
                self.moe_block = Gemma4Int4MoEBlock(config, layer_idx)
            else:
                self.moe_block = Gemma4DenseMoEBlock(config, layer_idx)
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size,
                                                       config.rms_norm_eps)

        if self.hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = make_linear(
                config,
                config.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                module_name=f"layers.{layer_idx}.per_layer_input_gate",
                tp_mode=TPMode.REPLICATED,
            )
            self.per_layer_projection = make_linear(
                config,
                self.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
                module_name=f"layers.{layer_idx}.per_layer_projection",
                tp_mode=TPMode.REPLICATED,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps)

    def _apply_per_layer_input(
            self, hidden_states: torch.Tensor,
            per_layer_input: torch.Tensor | None) -> torch.Tensor:
        """Apply Gemma4 PLE gate/projection/post-norm/residual injection."""
        if per_layer_input is None:
            return hidden_states
        if self.hidden_size_per_layer_input <= 0:
            raise ValueError(
                "per_layer_input was provided but Gemma4 PLE is disabled.")
        if per_layer_input.ndim != 3:
            raise ValueError(
                "Gemma4DecoderLayer._apply_per_layer_input expects "
                "per_layer_input to be rank-3, got shape "
                f"{tuple(per_layer_input.shape)}.")
        if per_layer_input.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "Gemma4DecoderLayer._apply_per_layer_input expects "
                "per_layer_input batch/sequence dimensions to match "
                f"hidden_states. Got {tuple(per_layer_input.shape)} and "
                f"{tuple(hidden_states.shape)}.")
        if per_layer_input.shape[-1] != self.hidden_size_per_layer_input:
            raise ValueError(
                "Gemma4DecoderLayer._apply_per_layer_input expected final "
                f"dimension {self.hidden_size_per_layer_input}, got "
                f"{per_layer_input.shape[-1]}.")

        gated = self.per_layer_input_gate(hidden_states)
        gated = self.act_fn(gated)
        gated = gated * per_layer_input.to(dtype=gated.dtype)
        gated = self.per_layer_projection(gated)
        gated = self.post_per_layer_input_norm(gated)
        return hidden_states + gated

    def _layer_scalar(self, phase_is_encoder: torch.Tensor | None,
                      hidden_states: torch.Tensor) -> torch.Tensor:
        del phase_is_encoder
        return self.layer_scalar.to(dtype=hidden_states.dtype,
                                    device=hidden_states.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
        vision_block_ids: torch.Tensor | None = None,
        context_mask_selector: torch.Tensor | None = None,
        per_layer_input: torch.Tensor | None = None,
        phase_is_encoder: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            vision_block_ids=vision_block_ids,
            context_mask_selector=context_mask_selector,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            # MoE branch: norm MLP output, route residual through experts,
            # combine both normalized outputs.
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # NVFP4 path: plugin handles routing + expert compute.
            # pre_feedforward_layernorm_2 normalizes expert input.
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            expert_input = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.moe_block(expert_input, residual)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(
                hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self._apply_per_layer_input(hidden_states,
                                                    per_layer_input)
        hidden_states = hidden_states * self._layer_scalar(
            phase_is_encoder, hidden_states)

        return hidden_states, present_key_value


class Gemma4Transformer(nn.Module):
    """Gemma4 attention decoder with optional per-layer input injection."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size_per_layer_input = int(
            config.hidden_size_per_layer_input)
        self.vocab_size_per_layer_input = int(
            config.vocab_size_per_layer_input)
        self.ple_enabled = self.hidden_size_per_layer_input > 0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.ple_enabled:
            if self.vocab_size_per_layer_input <= 0:
                raise ValueError(
                    "Gemma4 PLE requires vocab_size_per_layer_input > 0.")
            # Weight holder for the token-identity PLE table. This module is
            # not called by the ONNX forward path; checkpoint export writes its
            # weight to ple_embedding.safetensors for runtime-side gather.
            self.embed_tokens_per_layer = nn.Embedding(
                self.vocab_size_per_layer_input,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
            )
            self.per_layer_model_projection = make_linear(
                config,
                config.hidden_size,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False,
                module_name="per_layer_model_projection",
                tp_mode=TPMode.REPLICATED,
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input,
                config.rms_norm_eps,
            )
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0, dtype=torch.float32)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_model_projection_scale",
                torch.tensor(float(config.hidden_size)**-0.5,
                             dtype=torch.float32),
                persistent=False,
            )

        self.last_pre_norm_hidden_states: torch.Tensor | None = None
        self.target_hidden_concat: torch.Tensor | None = None
        self.dflash_hidden_concat: torch.Tensor | None = None

    def _project_per_layer_inputs(
            self, inputs_embeds: torch.Tensor) -> torch.Tensor | None:
        """Project runtime ``inputs_embeds`` into Gemma4 per-layer inputs."""
        if not self.ple_enabled:
            return None

        projected = self.per_layer_model_projection(inputs_embeds)
        scale = self.per_layer_model_projection_scale.to(
            dtype=projected.dtype, device=projected.device)
        projected = projected * scale
        projected = projected.reshape(
            *inputs_embeds.shape[:-1],
            len(self.layers),
            self.hidden_size_per_layer_input,
        )
        return self.per_layer_projection_norm(projected)

    def _combine_per_layer_input(self, projected_per_layer_inputs: torch.Tensor
                                 | None, ple_token_embeds: Tuple[torch.Tensor,
                                                                 ...],
                                 layer_index: int) -> torch.Tensor | None:
        """Combine context projection and runtime token-identity PLE input."""
        if not self.ple_enabled:
            return None
        if projected_per_layer_inputs is None:
            raise ValueError("Gemma4 PLE projection unexpectedly missing.")
        if len(ple_token_embeds) != len(self.layers):
            raise ValueError(
                "Gemma4 PLE expects one ple_token_embeds input per layer; "
                f"got {len(ple_token_embeds)} for {len(self.layers)} layers.")

        combined = (projected_per_layer_inputs[:, :, layer_index, :] +
                    ple_token_embeds[layer_index].to(
                        dtype=projected_per_layer_inputs.dtype))
        scale = self.per_layer_input_scale.to(dtype=combined.dtype,
                                              device=combined.device)
        return combined * scale

    def _select_rope_for_layer(
        self,
        layer: nn.Module,
        rope_rotary_cos_sin: torch.Tensor | None,
        rope_rotary_cos_sin_sliding: torch.Tensor | None,
        rope_rotary_cos_sin_full: torch.Tensor | None,
    ) -> torch.Tensor:
        """Select the Gemma4 RoPE tensor for one decoder layer."""
        return _select_rope_for_layer(
            layer,
            rope_rotary_cos_sin,
            rope_rotary_cos_sin_sliding,
            rope_rotary_cos_sin_full,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor | None,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
        vision_block_ids: torch.Tensor | None = None,
        context_mask_selector: torch.Tensor | None = None,
        phase_is_encoder: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        target_layer_ids: List[int] | None = None,
        ple_token_embeds: Tuple[torch.Tensor, ...] = (),
        rope_rotary_cos_sin_sliding: torch.Tensor | None = None,
        rope_rotary_cos_sin_full: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Tuple, Tuple | None]:
        hidden_states = inputs_embeds
        projected_per_layer_inputs = self._project_per_layer_inputs(
            inputs_embeds)
        present_key_values_list: List[torch.Tensor] = []
        all_hidden_states: list = []
        target_hidden_list: list = []
        target_layer_set = set(target_layer_ids or [])

        for layer_index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_rope_rotary_cos_sin = _select_rope_for_layer(
                layer,
                rope_rotary_cos_sin,
                rope_rotary_cos_sin_sliding,
                rope_rotary_cos_sin_full,
            )
            per_layer_input = self._combine_per_layer_input(
                projected_per_layer_inputs, ple_token_embeds, layer_index)
            hidden_states, next_key_value = layer(
                hidden_states,
                past_key_values[layer_index],
                layer_rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                attention_mask=attention_mask,
                attention_pos_id=attention_pos_id,
                vision_block_ids=vision_block_ids,
                context_mask_selector=context_mask_selector,
                per_layer_input=per_layer_input,
                phase_is_encoder=phase_is_encoder,
            )
            present_key_values_list.append(next_key_value)

            if layer_index in target_layer_set:
                target_hidden_list.append(hidden_states)

        self.last_pre_norm_hidden_states = hidden_states
        self.target_hidden_concat = (torch.cat(target_hidden_list, dim=-1)
                                     if target_hidden_list else None)
        self.dflash_hidden_concat = self.target_hidden_concat
        normed = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(normed)

        return (normed, tuple(present_key_values_list),
                tuple(all_hidden_states) if output_hidden_states else None)


class Gemma4ForCausalLM(CausalLM):
    """Gemma4 CausalLM wrapper for the checkpoint exporter."""

    # RMSNorm weights are f32 initializers that feed element-wise Mul with
    # f32 normalized hidden states.  Without this flag, _fix_initializer_dtypes
    # downgrades them to f16, creating a type mismatch with --stronglyTyped.
    match_fp32_elementwise_initializers = True

    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = Gemma4Transformer(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    @property
    def ple_enabled(self) -> bool:
        """Whether this Gemma4 export uses per-layer embedding inputs."""
        return self.model.ple_enabled

    def onnx_export_spec(self) -> OnnxSpec:
        """Return Gemma4-specific ONNX export parameters."""
        config = self.config
        dflash_base = getattr(config, "dflash_base", False)
        dspark_base = getattr(config, "dspark_base", False)
        target_hidden_base = dflash_base or dspark_base
        eagle_base = config.eagle_base
        tree_attention_base = (eagle_base or config.gemma4_mtp_base
                               or target_hidden_base)
        vision_block_attention = bool(config.use_vision_bidirectional_attention
                                      ) and not tree_attention_base
        if (not self.ple_enabled and not config.use_dual_rope
                and not vision_block_attention and not tree_attention_base):
            return super().onnx_export_spec()

        Na = config.num_hidden_layers
        num_ple_inputs = Na if self.ple_enabled else 0
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        batch_size, seq_len, past_len, max_pos = (_DUMMY_BATCH_SIZE,
                                                  _DUMMY_SEQ_LEN,
                                                  _DUMMY_PAST_LEN,
                                                  _DUMMY_ROPE_CACHE_LEN)

        inputs_embeds = torch.zeros(batch_size,
                                    seq_len,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        ple_token_embeds_list: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        seq_len,
                        config.hidden_size_per_layer_input,
                        dtype=dtype16,
                        device=device) for _ in range(num_ple_inputs)
        ]
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        # Paged KV pool binding: [2, num_pages, KV_PAGE_SIZE, num_kv_heads, head_dim].
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(
                2,
                1,
                KV_PAGE_SIZE,
                num_kv_heads,
                layer_head_dim,
                dtype=kv_dtype,
                device=device,
            ) for num_kv_heads, layer_head_dim in (
                _kv_cache_dims_for_layer(config, layer_idx)
                for layer_idx in range(Na))
        ]

        args = (inputs_embeds, *ple_token_embeds_list, *past_key_values_list)
        input_names = (
            ["inputs_embeds"] +
            [f"ple_token_embeds_{i}" for i in range(num_ple_inputs)] +
            [f"past_key_values_{i}" for i in range(Na)])

        if config.use_dual_rope:
            sliding_head_dim = _head_dim_for_attention_type(
                config, "sliding_attention")
            full_head_dim = _head_dim_for_attention_type(
                config, "full_attention")
            sliding_rotary_dim = _rotary_dim_from_rope_config(
                config, config.sliding_rope_config, sliding_head_dim)
            full_rotary_dim = _rotary_dim_from_rope_config(
                config, config.full_rope_config, full_head_dim)
            rope_rotary_cos_sin_sliding = torch.zeros(batch_size,
                                                      max_pos,
                                                      sliding_rotary_dim,
                                                      dtype=torch.float32,
                                                      device=device)
            rope_rotary_cos_sin_full = torch.zeros(batch_size,
                                                   max_pos,
                                                   full_rotary_dim,
                                                   dtype=torch.float32,
                                                   device=device)
            args = args + (rope_rotary_cos_sin_sliding,
                           rope_rotary_cos_sin_full)
            input_names = input_names + [
                "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full"
            ]
        else:
            rotary_head_dim = _head_dim_for_attention_type(
                config, "full_attention")
            rotary_dim = _rotary_dim_from_rope_config(config, None,
                                                      rotary_head_dim)
            rope_rotary_cos_sin = torch.zeros(batch_size,
                                              max_pos,
                                              rotary_dim,
                                              dtype=torch.float32,
                                              device=device)
            args = args + (rope_rotary_cos_sin, )
            input_names = input_names + ["rope_rotary_cos_sin"]

        context_lengths = torch.zeros(batch_size,
                                      dtype=torch.int32,
                                      device=device)
        kvcache_start_index = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)
        kv_page_table = torch.zeros(batch_size,
                                    2,
                                    1,
                                    dtype=torch.int32,
                                    device=device)
        last_token_ids = torch.zeros(batch_size,
                                     1,
                                     dtype=torch.int64,
                                     device=device)

        args = args + (context_lengths, kvcache_start_index, kv_page_table,
                       last_token_ids)
        input_names = input_names + [
            "context_lengths", "kvcache_start_index", "kv_page_table",
            "last_token_ids"
        ]
        if vision_block_attention:
            vision_block_ids = torch.full((batch_size, seq_len),
                                          -1,
                                          dtype=torch.int32,
                                          device=device)
            args = args + (vision_block_ids, )
            input_names = input_names + ["vision_block_ids"]
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)])
        if self.emit_hidden_states and not eagle_base:
            output_names = (["logits", "hidden_states"] +
                            [f"present_key_values_{i}" for i in range(Na)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)

        num_selected = torch.export.Dim(
            "num_selected", min=1, max=256) if tree_attention_base else None
        all_shapes: list = [{0: batch, 1: seq}]
        for _ in range(num_ple_inputs):
            all_shapes.append({0: batch, 1: seq})
        for _ in range(Na):
            all_shapes.append({1:
                               num_pages})  # past_key_values_i (pool-shaped)
        all_shapes.append({0: rope_batch, 1: pos})
        if config.use_dual_rope:
            all_shapes.append({0: rope_batch, 1: pos})
        all_shapes.append({0: batch})
        all_shapes.append({0: kv_batch})
        all_shapes.append({0: page_batch, 2: max_pages})  # kv_page_table
        if tree_attention_base:
            all_shapes.append({0: batch, 1: num_selected})
        else:
            all_shapes.append({0: batch})
        if vision_block_attention:
            all_shapes.append({0: batch, 1: seq})
        if tree_attention_base:
            attention_pos_id = torch.zeros(batch_size,
                                           seq_len,
                                           dtype=torch.int32,
                                           device=device)
            attention_mask = torch.zeros(batch_size,
                                         seq_len,
                                         seq_len + past_len,
                                         dtype=torch.int32,
                                         device=device)
            args = args + (attention_pos_id, attention_mask)
            input_names = input_names + ["attention_pos_id", "attention_mask"]
            output_names = ["logits", "hidden_states"
                            ] + [f"present_key_values_{i}" for i in range(Na)]

            eagle_seq = torch.export.Dim("eagle_seq_len", min=1, max=32768)
            mask_kv_len = torch.export.Dim("mask_kv_len", min=1, max=65536)
            all_shapes.append({0: batch, 1: eagle_seq})
            all_shapes.append({0: batch, 1: eagle_seq, 2: mask_kv_len})

        wrapped = _make_gemma4_flat_wrapper(
            self,
            Na,
            num_ple_inputs=num_ple_inputs,
            use_dual_rope=config.use_dual_rope,
            eagle_base=tree_attention_base,
            vision_block_attention=vision_block_attention,
            emit_hidden_states=(self.emit_hidden_states
                                or config.gemma4_mtp_base))
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor | None,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_pos_id: torch.Tensor | None = None,
        vision_block_ids: torch.Tensor | None = None,
        ple_token_embeds: Tuple[torch.Tensor, ...] = (),
        rope_rotary_cos_sin_sliding: torch.Tensor | None = None,
        rope_rotary_cos_sin_full: torch.Tensor | None = None,
    ) -> Tuple:
        eagle_base = self.config.eagle_base
        gemma4_mtp_base = self.config.gemma4_mtp_base
        dflash_base = getattr(self.config, "dflash_base", False)
        dspark_base = getattr(self.config, "dspark_base", False)
        target_hidden_base = dflash_base or dspark_base
        eagle_target_layer_ids = getattr(self.config,
                                         "eagle3_target_layer_ids", [])
        eagle_target_hidden_base = eagle_base and bool(eagle_target_layer_ids)
        target_layer_ids = None
        if dspark_base:
            target_layer_ids = getattr(self.config, "dspark_target_layer_ids",
                                       None)
        elif dflash_base:
            target_layer_ids = getattr(self.config, "dflash_target_layer_ids",
                                       None)
        elif eagle_target_hidden_base:
            target_layer_ids = eagle_target_layer_ids

        hidden_states, present_key_values, all_hidden_states = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id,
            vision_block_ids=vision_block_ids,
            output_hidden_states=eagle_base and not eagle_target_hidden_base
            and not target_hidden_base,
            target_layer_ids=target_layer_ids,
            ple_token_embeds=ple_token_embeds,
            rope_rotary_cos_sin_sliding=rope_rotary_cos_sin_sliding,
            rope_rotary_cos_sin_full=rope_rotary_cos_sin_full,
        )
        target_hidden_concat = getattr(self.model, "target_hidden_concat",
                                       None)

        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, last_token_ids)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)

        final_logit_softcapping = getattr(self.config,
                                          "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = torch.tanh(
                logits / final_logit_softcapping) * final_logit_softcapping

        if ((target_hidden_base or eagle_target_hidden_base)
                and target_hidden_concat is not None):
            return logits, target_hidden_concat.to(
                torch.float16), present_key_values

        if eagle_base and all_hidden_states is not None:
            n_layers = len(all_hidden_states) - 1
            idx = [2, n_layers // 2, n_layers - 4]
            eagle_hidden = torch.cat([
                all_hidden_states[idx[0]],
                all_hidden_states[idx[1]],
                all_hidden_states[idx[2]],
            ],
                                     dim=-1).to(torch.float16)
            return logits, eagle_hidden, present_key_values

        if self.emit_hidden_states:
            return logits, self.model.last_pre_norm_hidden_states, \
                present_key_values

        if gemma4_mtp_base:
            return logits, hidden_states, present_key_values

        return logits, present_key_values


# ---------------------------------------------------------------------------
# Checkpoint key remap for NVFP4 Gemma4 MoE
# ---------------------------------------------------------------------------
# Checkpoint: layers.{i}.router.* → model tree: layers.{i}.moe_block.router.*
_ROUTER_RE = re.compile(r"(layers\.\d+\.)router\.")
# Checkpoint: layers.{i}.experts.{j}.* → model tree: layers.{i}.moe_block.experts._experts.{j}.*
_EXPERTS_RE = re.compile(r"(layers\.\d+\.)experts\.(\d+)\.")


def GEMMA4_NVFP4_KEY_REMAP(key: str) -> "str | None":
    """Remap Gemma4 NVFP4 checkpoint keys to the internal module tree.

    Checkpoint layout (nvidia/Gemma-4-26B-A4B-NVFP4):
        model.layers.{i}.router.proj.weight
        model.layers.{i}.router.scale
        model.layers.{i}.router.per_expert_scale
        model.layers.{i}.experts.{j}.gate_proj.weight
        model.layers.{i}.experts.{j}.gate_proj.weight_scale
        ...

    Model tree (with Gemma4NvFP4MoEBlock):
        model.layers.{i}.moe_block.router.proj.weight
        model.layers.{i}.moe_block.router.scale
        model.layers.{i}.moe_block.router.per_expert_scale
        model.layers.{i}.moe_block.experts._experts.{j}.gate_proj.weight
        ...
    """
    key = _ROUTER_RE.sub(r"\1moe_block.router.", key)
    key = _EXPERTS_RE.sub(r"\1moe_block.experts._experts.\2.", key)
    return key


# Fused BF16 expert keys (QAT-unquantized Gemma4):
#   layers.{i}.experts.gate_up_proj → layers.{i}.moe_block.experts.gate_up_proj
#   layers.{i}.experts.down_proj    → layers.{i}.moe_block.experts.down_proj
# Note: after prefix stripping, key may have "model." prefix.
_FUSED_EXPERTS_RE = re.compile(
    r"((?:model\.)?layers\.\d+\.)experts\.(gate_up_proj|down_proj)")


def GEMMA4_FUSED_BF16_KEY_REMAP(key: str) -> "str | None":
    """Remap BF16 QAT-unquantized Gemma4 MoE checkpoint keys.

    Checkpoint layout (google/gemma-4-26B-A4B-it-qat-q4_0-unquantized):
        model.language_model.layers.{i}.router.proj.weight
        model.language_model.layers.{i}.router.scale
        model.language_model.layers.{i}.router.per_expert_scale
        model.language_model.layers.{i}.experts.gate_up_proj
        model.language_model.layers.{i}.experts.down_proj

    Model tree (with Gemma4Int4MoEBlock + Gemma4FusedBF16MoEExperts):
        model.layers.{i}.moe_block.router.proj.weight
        model.layers.{i}.moe_block.router.scale
        model.layers.{i}.moe_block.router.per_expert_scale
        model.layers.{i}.moe_block.experts.gate_up_proj
        model.layers.{i}.moe_block.experts.down_proj
    """
    key = _ROUTER_RE.sub(r"\1moe_block.router.", key)
    key = _FUSED_EXPERTS_RE.sub(r"\1moe_block.experts.\2", key)
    return key
