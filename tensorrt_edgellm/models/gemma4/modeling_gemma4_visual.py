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
"""Gemma4 vision encoder implementation for checkpoint-based export."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ... import config as config_module
from ..linear import make_linear
from ..ops import (is_trt_native_attention_enabled, trt_ragged_attention,
                   vit_attention_plugin)

if TYPE_CHECKING:
    from ...config import ModelConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Gemma4RMSNorm(nn.Module):
    """Gemma4 RMSNorm.

    Vision Q/K norms use a learned scale. Vision V norm and multimodal
    pre-projection norm are weightless.
    """

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 with_scale: bool = True) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size,
                                                  dtype=torch.float16),
                                       requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        if self.with_scale:
            hidden_states = hidden_states * self.weight.float()
        return hidden_states.to(input_dtype)


class Gemma4ClippableLinear(nn.Module):
    """Gemma4 bias-free linear with optional input/output clipping."""

    def __init__(self,
                 config: dict,
                 in_features: int,
                 out_features: int,
                 model_config: "ModelConfig",
                 module_name: str = "",
                 compute_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.compute_dtype = compute_dtype
        self.use_clipped_linears = bool(
            config.get("use_clipped_linears", False))
        self.linear = make_linear(model_config,
                                  in_features,
                                  out_features,
                                  bias=False,
                                  module_name=module_name)
        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    @staticmethod
    def _clip(hidden_states: torch.Tensor, minimum: torch.Tensor,
              maximum: torch.Tensor) -> torch.Tensor:
        minimum = minimum.to(dtype=hidden_states.dtype,
                             device=hidden_states.device)
        maximum = maximum.to(dtype=hidden_states.dtype,
                             device=hidden_states.device)
        return torch.where(
            hidden_states < minimum, minimum,
            torch.where(hidden_states > maximum, maximum, hidden_states))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = self._clip(hidden_states, self.input_min,
                                       self.input_max)
        if self.compute_dtype == torch.float32:
            weight = getattr(self.linear, "weight", None)
            if weight is None:
                raise TypeError("Gemma4 FP32 linear export requires a "
                                "floating-point weight tensor.")
            bias = getattr(self.linear, "bias", None)
            hidden_states = F.linear(hidden_states.float(), weight.float(),
                                     None if bias is None else bias.float())
        else:
            hidden_states = self.linear(hidden_states.to(torch.float16))
        if self.use_clipped_linears:
            hidden_states = self._clip(hidden_states, self.output_min,
                                       self.output_max)
        return hidden_states


class Gemma4VisionPatchEmbedder(nn.Module):
    """Gemma4 linear patch embedder with learned x/y position embeddings."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.patch_size = int(config["patch_size"])
        self.position_embedding_size = int(config["position_embedding_size"])
        self.input_proj = make_linear(
            model_config,
            3 * self.patch_size * self.patch_size,
            self.hidden_size,
            bias=False,
            module_name="vision_tower.patch_embedder.input_proj")
        self.position_embedding_table = nn.Parameter(torch.ones(
            2,
            self.position_embedding_size,
            self.hidden_size,
            dtype=torch.float16),
                                                     requires_grad=False)

    def forward(self, pixel_values: torch.Tensor,
                pixel_position_ids: torch.Tensor) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - pixel_values.new_tensor(0.5))
        hidden_states = self.input_proj(pixel_values.to(torch.float16))
        clamped_positions = pixel_position_ids.clamp(min=0)
        x_emb = nn.functional.embedding(clamped_positions[..., 0],
                                        self.position_embedding_table[0])
        y_emb = nn.functional.embedding(clamped_positions[..., 1],
                                        self.position_embedding_table[1])
        return hidden_states + x_emb + y_emb


def _apply_gemma4_2d_rope(x: torch.Tensor, cos: torch.Tensor,
                          sin: torch.Tensor) -> torch.Tensor:
    """Apply Gemma4's independent x/y 2-D RoPE to [T, heads, head_dim]."""
    input_dtype = x.dtype
    axis_dim = x.shape[-1] // 2
    x_parts = torch.split(x.float(), [axis_dim, axis_dim], dim=-1)
    cos_parts = torch.split(cos.float(), [axis_dim, axis_dim], dim=-1)
    sin_parts = torch.split(sin.float(), [axis_dim, axis_dim], dim=-1)
    y_parts = []
    for x_part, cos_part, sin_part in zip(x_parts, cos_parts, sin_parts):
        cos_part = cos_part.unsqueeze(1)
        sin_part = sin_part.unsqueeze(1)
        y_parts.append((x_part * cos_part) + (_rotate_half(x_part) * sin_part))
    return torch.cat(y_parts, dim=-1).to(input_dtype)


class Gemma4VisionRotaryEmbedding(nn.Module):
    """Gemma4 vision 2-D RoPE from runtime-provided angle embeddings."""

    def __init__(self, config: dict) -> None:
        super().__init__()

    def forward(
            self, hidden_states: torch.Tensor,
            rotary_pos_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = rotary_pos_emb.float().cos().to(hidden_states.dtype)
        sin = rotary_pos_emb.float().sin().to(hidden_states.dtype)
        return cos, sin


class Gemma4VisionAttention(nn.Module):
    """Gemma4 vision attention implemented with the EdgeLLM ViT attention op."""

    def __init__(self, config: dict, layer_idx: int,
                 model_config: "ModelConfig") -> None:
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(
            config.get("num_key_value_heads", self.num_heads))
        if self.num_kv_heads != self.num_heads:
            raise ValueError("Gemma4 visual export currently requires "
                             "num_key_value_heads == num_attention_heads")
        self.head_dim = int(
            config.get("head_dim", self.hidden_size // self.num_heads))
        self.attention_scale = config_module._get_attention_scaling(
            config, self.head_dim, 1.0)
        self._use_trt_attn = is_trt_native_attention_enabled()
        prefix = f"vision_tower.encoder.layers.{layer_idx}.self_attn"
        self.q_proj = Gemma4ClippableLinear(config,
                                            self.hidden_size,
                                            self.num_heads * self.head_dim,
                                            model_config,
                                            module_name=f"{prefix}.q_proj."
                                            "linear")
        self.k_proj = Gemma4ClippableLinear(config,
                                            self.hidden_size,
                                            self.num_heads * self.head_dim,
                                            model_config,
                                            module_name=f"{prefix}.k_proj."
                                            "linear")
        self.v_proj = Gemma4ClippableLinear(config,
                                            self.hidden_size,
                                            self.num_heads * self.head_dim,
                                            model_config,
                                            module_name=f"{prefix}.v_proj."
                                            "linear")
        self.o_proj = Gemma4ClippableLinear(config,
                                            self.num_heads * self.head_dim,
                                            self.hidden_size,
                                            model_config,
                                            module_name=f"{prefix}.o_proj."
                                            "linear")
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_len, self.num_heads,
                                            self.head_dim)
        k = self.k_proj(hidden_states).view(seq_len, self.num_heads,
                                            self.head_dim)
        v = self.v_proj(hidden_states).view(seq_len, self.num_heads,
                                            self.head_dim)

        cos, sin = position_embeddings
        q = _apply_gemma4_2d_rope(self.q_norm(q), cos, sin)
        k = _apply_gemma4_2d_rope(self.k_norm(k), cos, sin)
        v = self.v_norm(v)

        if self._use_trt_attn:
            attn_output = trt_ragged_attention(
                q.to(torch.float16),
                k.to(torch.float16),
                v.to(torch.float16),
                cu_seqlens,
                kv_lengths,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale)
        else:
            attn_output = vit_attention_plugin(
                q.to(torch.float16),
                k.to(torch.float16),
                v.to(torch.float16),
                cu_seqlens,
                max_seqlen_carrier,
                num_heads=self.num_heads,
                head_size=self.head_dim,
                attention_scale=self.attention_scale)
        attn_output = attn_output.reshape(seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Gemma4VisionMLP(nn.Module):
    """Gemma4 gated vision MLP."""

    def __init__(self, config: dict, layer_idx: int,
                 model_config: "ModelConfig") -> None:
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.intermediate_size = int(config["intermediate_size"])
        # Standardized Gemma4 visual checkpoints are sensitive to FP16 MLP
        # accumulation in TensorRT; keep only that MLP path in FP32.
        mlp_compute_dtype = (torch.float32 if bool(
            config.get("standardize", False)) else torch.float16)
        prefix = f"vision_tower.encoder.layers.{layer_idx}.mlp"
        self.gate_proj = Gemma4ClippableLinear(
            config,
            self.hidden_size,
            self.intermediate_size,
            model_config,
            module_name=f"{prefix}.gate_proj.linear",
            compute_dtype=mlp_compute_dtype)
        self.up_proj = Gemma4ClippableLinear(
            config,
            self.hidden_size,
            self.intermediate_size,
            model_config,
            module_name=f"{prefix}.up_proj.linear",
            compute_dtype=mlp_compute_dtype)
        self.down_proj = Gemma4ClippableLinear(
            config,
            self.intermediate_size,
            self.hidden_size,
            model_config,
            module_name=f"{prefix}.down_proj.linear",
            compute_dtype=mlp_compute_dtype)
        activation_name = config.get("hidden_activation",
                                     config.get("hidden_act", "gelu"))
        if activation_name not in ACT2FN:
            raise ValueError(
                f"Unsupported Gemma4 vision activation {activation_name!r}")
        self.act_fn = ACT2FN[activation_name]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))


class Gemma4VisionEncoderLayer(nn.Module):
    """Gemma4 vision transformer block."""

    def __init__(self, config: dict, layer_idx: int,
                 model_config: "ModelConfig") -> None:
        super().__init__()
        hidden_size = int(config["hidden_size"])
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.self_attn = Gemma4VisionAttention(config, layer_idx, model_config)
        self.mlp = Gemma4VisionMLP(config, layer_idx, model_config)
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states,
                                       position_embeddings,
                                       cu_seqlens,
                                       max_seqlen_carrier,
                                       kv_lengths=kv_lengths)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4VisionEncoder(nn.Module):
    """Gemma4 vision encoder stack."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([
            Gemma4VisionEncoderLayer(config, layer_idx, model_config)
            for layer_idx in range(int(config["num_hidden_layers"]))
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(hidden_states, rotary_pos_emb)
        for layer in self.layers:
            hidden_states = layer(hidden_states,
                                  position_embeddings,
                                  cu_seqlens,
                                  max_seqlen_carrier,
                                  kv_lengths=kv_lengths)
        return hidden_states


class Gemma4VisionPooler(nn.Module):
    """Position-aware spatial average pooler.

    Runtime provides a sparse block-diagonal pooling matrix as a dense input:
    [total_soft_tokens, total_patches]. This matches the reference
    position-bin average while avoiding dynamic one-hot construction in ONNX.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.root_hidden_size = float(hidden_size)**0.5

    def forward(self, hidden_states: torch.Tensor,
                pooling_weights: torch.Tensor) -> torch.Tensor:
        hidden_states = pooling_weights.float() @ hidden_states.float()
        return hidden_states * hidden_states.new_tensor(self.root_hidden_size)


class Gemma4VisionTower(nn.Module):
    """Gemma4 vision tower through pooled visual soft tokens."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        self.config = config
        hidden_size = int(config["hidden_size"])
        self.patch_embedder = Gemma4VisionPatchEmbedder(config, model_config)
        self.encoder = Gemma4VisionEncoder(config, model_config)
        self.pooler = Gemma4VisionPooler(hidden_size)
        self.standardize = bool(config.get("standardize", False))
        if self.standardize:
            self.register_buffer("std_bias", torch.empty(hidden_size))
            self.register_buffer("std_scale", torch.empty(hidden_size))

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        pooling_weights: torch.Tensor,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.patch_embedder(pixel_values, pixel_position_ids)
        hidden_states = self.encoder(hidden_states,
                                     rotary_pos_emb,
                                     cu_seqlens,
                                     max_seqlen_carrier,
                                     kv_lengths=kv_lengths)
        hidden_states = self.pooler(hidden_states, pooling_weights)
        if self.standardize:
            hidden_states = (hidden_states -
                             self.std_bias.float()) * self.std_scale.float()
        return hidden_states.to(pixel_values.dtype)


class Gemma4MultimodalEmbedder(nn.Module):
    """Project Gemma4 visual soft tokens to text hidden size."""

    def __init__(self, vision_config: dict, text_config: dict,
                 model_config: "ModelConfig") -> None:
        super().__init__()
        multimodal_hidden_size = int(
            vision_config.get("output_proj_dims",
                              vision_config["hidden_size"]))
        text_hidden_size = int(text_config["hidden_size"])
        eps = float(vision_config.get("rms_norm_eps", 1e-6))
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            multimodal_hidden_size, eps=eps, with_scale=False)
        self.embedding_projection = make_linear(
            model_config,
            multimodal_hidden_size,
            text_hidden_size,
            bias=False,
            module_name="embed_vision.embedding_projection")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding_pre_projection_norm(hidden_states)
        return self.embedding_projection(hidden_states.to(torch.float16))


class Gemma4VisualModel(nn.Module):
    """Gemma4 visual encoder plus multimodal projection."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        vision_config = config.get("vision_config", config)
        text_config = config.get("text_config") or {}
        if "hidden_size" not in text_config:
            raise ValueError("Gemma4 visual export requires text_config")
        self.vision_tower = Gemma4VisionTower(vision_config, model_config)
        self.embed_vision = Gemma4MultimodalEmbedder(vision_config,
                                                     text_config, model_config)
        self._use_trt_attn = is_trt_native_attention_enabled()

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen_carrier: torch.Tensor,
        pooling_weights: torch.Tensor,
        kv_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.vision_tower(pixel_values,
                                          pixel_position_ids,
                                          rotary_pos_emb,
                                          cu_seqlens,
                                          max_seqlen_carrier,
                                          pooling_weights,
                                          kv_lengths=kv_lengths)
        return self.embed_vision(hidden_states)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return ONNX export example inputs and dynamic-shape metadata."""
        vision_config = config.get("vision_config", config)
        patch_size = int(vision_config["patch_size"])
        pooling_kernel_size = int(vision_config.get("pooling_kernel_size", 3))
        head_dim = int(
            vision_config.get(
                "head_dim", vision_config["hidden_size"] //
                vision_config["num_attention_heads"]))
        total_soft_tokens = 16
        total_patches = total_soft_tokens * pooling_kernel_size * pooling_kernel_size
        in_channels = 3 * patch_size * patch_size

        pixel_values = torch.zeros(total_patches,
                                   in_channels,
                                   dtype=torch.float16,
                                   device=device)
        pixel_position_ids = torch.zeros(total_patches,
                                         2,
                                         dtype=torch.int64,
                                         device=device)
        rotary_pos_emb = torch.zeros(total_patches,
                                     head_dim,
                                     dtype=torch.float32,
                                     device=device)
        cu_seqlens = torch.tensor([0, total_patches],
                                  dtype=torch.int32,
                                  device=device)
        max_seqlen_carrier = torch.zeros(total_patches,
                                         dtype=torch.int32,
                                         device=device)
        pooling_weights = torch.zeros(total_soft_tokens,
                                      total_patches,
                                      dtype=torch.float16,
                                      device=device)

        args = (pixel_values, pixel_position_ids, rotary_pos_emb, cu_seqlens,
                max_seqlen_carrier, pooling_weights)
        input_names = [
            "input",
            "pixel_position_ids",
            "rotary_pos_emb",
            "cu_seqlens",
            "max_seqlen_carrier",
            "pooling_weights",
        ]
        if self._use_trt_attn:
            kv_lengths = torch.tensor([0, total_patches],
                                      dtype=torch.int32,
                                      device=device)
            args = args + (kv_lengths, )
            input_names.append("kv_lengths")
        output_names = ["output"]

        total_tokens = torch.export.Dim("total_tokens")
        total_soft = torch.export.Dim("total_soft_tokens")
        max_seqlen = torch.export.Dim("max_seqlen", min=1)
        dynamic_shapes = {
            "pixel_values": {
                0: total_tokens
            },
            "pixel_position_ids": {
                0: total_tokens
            },
            "rotary_pos_emb": {
                0: total_tokens
            },
            "cu_seqlens": {
                0: torch.export.Dim("batch_p1")
            },
            "max_seqlen_carrier": {
                0: max_seqlen
            },
            "pooling_weights": {
                0: total_soft,
                1: total_tokens
            },
        }
        if self._use_trt_attn:
            dynamic_shapes["kv_lengths"] = {0: torch.export.Dim("kv_batch_p1")}
        return args, input_names, output_names, dynamic_shapes


def _load_weights(model: nn.Module, weights: dict) -> None:
    """Load Gemma4 visual weights from a flat safetensors dict."""
    from ...checkpoint.loader import load_submodule_weights

    def _remap(key: str) -> "str | None":
        for prefix in ("model.", ""):
            candidate = key[len(prefix):] if prefix and key.startswith(
                prefix) else key
            if (candidate.startswith("vision_tower.")
                    or candidate.startswith("embed_vision.")):
                return candidate
        return None

    load_submodule_weights(model, weights, _remap, label="gemma4 visual")


def build_gemma4_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Gemma4VisualModel:
    """Instantiate and load Gemma4 visual encoder weights."""
    model = Gemma4VisualModel(config, model_config=model_config)
    model.to(dtype)
    _load_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "Gemma4VisualModel",
    "Gemma4VisionAttention",
    "Gemma4VisionEncoder",
    "Gemma4VisionEncoderLayer",
    "Gemma4VisionMLP",
    "Gemma4VisionPatchEmbedder",
    "Gemma4VisionPooler",
    "Gemma4VisionRotaryEmbedding",
    "Gemma4VisionTower",
    "build_gemma4_visual",
]
