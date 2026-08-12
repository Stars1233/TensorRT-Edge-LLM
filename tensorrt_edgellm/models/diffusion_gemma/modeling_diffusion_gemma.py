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
"""DiffusionGemma text backbone export."""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..default.modeling_default import OnnxSpec, RMSNorm
from ..gemma4 import modeling_gemma4_text as gemma4_text
from ..linear import TPMode, make_linear

__all__ = [
    "DiffusionGemmaBackbone",
    "DiffusionGemmaDecoderLayer",
    "DiffusionGemmaMLP",
    "DiffusionGemmaSelfConditioning",
    "DiffusionGemmaTransformer",
    "make_diffusion_gemma_key_remap",
]

_DUMMY_BATCH_SIZE = 1
_DUMMY_SEQ_LEN = 4
_DUMMY_PAST_LEN = 2
_DUMMY_ROPE_CACHE_LEN = 4096
_DUMMY_CONTEXT_SELECTOR_LEN = 2


def make_diffusion_gemma_key_remap(
    include_backbone: bool = True,
    include_self_conditioning: bool = False,
    nvfp4_moe: bool = False,
) -> Callable[[str], Optional[str]]:
    """Return a checkpoint-key remapper for DiffusionGemma backbone exports.

    DiffusionGemma checkpoints carry the text backbone under
    ``model.decoder``.  The encoder tree holds vision weights plus per-layer
    encoder phase scalars under ``model.encoder.language_model.layers.*``.
    The Edge-LLM backbone graph is intentionally one large phase-aware
    transformer, so decoder tensors load into ``model.*`` while encoder and
    decoder phase scalars load into separate buffers.  Current production
    exports build unified self-conditioning in ``DiffusionGemmaBackbone`` and
    load ``model.decoder.self_conditioning.*`` into the same DLLM backbone
    pass as the quantized decoder weights.
    """
    seen_backbone_keys: set[str] = set()

    def _map_backbone_suffix(suffix: str) -> Optional[str]:
        mapped = f"model.{suffix}"
        if nvfp4_moe:
            mapped = gemma4_text.GEMMA4_NVFP4_KEY_REMAP(mapped)
        if mapped in seen_backbone_keys:
            return None
        seen_backbone_keys.add(mapped)
        return mapped

    def _remap(key: str) -> Optional[str]:
        if key.startswith("model.decoder.self_conditioning."):
            if not include_self_conditioning:
                return None
            return "self_conditioning." + key.split(
                "model.decoder.self_conditioning.", 1)[1]

        if key == "model.decoder.embed_tokens.weight":
            if not include_backbone:
                return None
            return _map_backbone_suffix(key.split("model.decoder.", 1)[1])

        if (key.startswith("model.encoder.language_model.layers.")
                and key.endswith(".layer_scalar")):
            if not include_backbone:
                return None
            suffix = key.split("model.encoder.language_model.", 1)[1]
            return "model." + suffix.replace(".layer_scalar",
                                             ".encoder_layer_scalar")

        if key.startswith("model.decoder.layers.") and key.endswith(
                ".layer_scalar"):
            if not include_backbone:
                return None
            suffix = key.split("model.decoder.", 1)[1]
            return "model." + suffix.replace(".layer_scalar",
                                             ".decoder_layer_scalar")

        if key.startswith("model.decoder."):
            if not include_backbone:
                return None
            return _map_backbone_suffix(key.split("model.decoder.", 1)[1])

        if key.startswith("model.encoder."):
            return None

        return key if include_backbone else None

    return _remap


def _make_diffusion_gemma_flat_wrapper(
        model: nn.Module, Na: int, num_ple_inputs: int, use_dual_rope: bool,
        unified_conditioning: bool) -> nn.Module:
    """Build a flat-signature wrapper for the phase-aware backbone export."""
    param_names: List[str] = ["inputs_embeds", "phase_is_encoder"]
    if unified_conditioning:
        param_names += [
            "canvas_ids",
            "prev_self_conditioning_embeds",
            "self_conditioning_temperature",
        ]
    param_names += ([f"ple_token_embeds_{i}" for i in range(num_ple_inputs)] +
                    [f"past_key_values_{i}" for i in range(Na)])
    if use_dual_rope:
        param_names += [
            "rope_rotary_cos_sin_sliding", "rope_rotary_cos_sin_full"
        ]
    else:
        param_names += ["rope_rotary_cos_sin"]
    param_names += [
        "context_lengths",
        "kvcache_start_index",
        "kv_page_table",
        "select_token_indices",
        "context_mask_selector",
    ]

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    ple_tuple = "({},)".format(", ".join(
        f"ple_token_embeds_{i}"
        for i in range(num_ple_inputs))) if num_ple_inputs else "()"
    ple_kwarg = f", ple_token_embeds={ple_tuple}" if num_ple_inputs else ""
    if use_dual_rope:
        rope_arg = "None"
        rope_kwargs = (
            ", rope_rotary_cos_sin_sliding=rope_rotary_cos_sin_sliding"
            ", rope_rotary_cos_sin_full=rope_rotary_cos_sin_full")
    else:
        rope_arg = "rope_rotary_cos_sin"
        rope_kwargs = ""

    if unified_conditioning:
        conditioning_kwarg = (
            ", canvas_ids=canvas_ids"
            ", prev_self_conditioning_embeds=prev_self_conditioning_embeds"
            ", self_conditioning_temperature=self_conditioning_temperature")
        body = (
            f"    logits, next_self_conditioning_embeds, present_key_values = self._model(\n"
            f"        inputs_embeds, phase_is_encoder, {past_kv_tuple}, "
            f"{rope_arg}, context_lengths, kvcache_start_index, kv_page_table, "
            f"select_token_indices, context_mask_selector"
            f"{ple_kwarg}{rope_kwargs}{conditioning_kwarg})\n"
            f"    return (logits, next_self_conditioning_embeds) + tuple(present_key_values)\n"
        )
    else:
        body = (
            f"    logits, present_key_values = self._model(\n"
            f"        inputs_embeds, phase_is_encoder, {past_kv_tuple}, "
            f"{rope_arg}, context_lengths, kvcache_start_index, kv_page_table, "
            f"select_token_indices, context_mask_selector"
            f"{ple_kwarg}{rope_kwargs})\n"
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


class DiffusionGemmaMLP(gemma4_text.Gemma4MLP):
    """DiffusionGemma dense MLP matching the HF fp16 gate/up product path."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) *
            self.up_proj(hidden_states))


class DiffusionGemmaDecoderLayer(gemma4_text.Gemma4DecoderLayer):
    """Gemma4 decoder layer with DiffusionGemma encoder/decoder scalars."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.mlp = DiffusionGemmaMLP(config, layer_idx)
        encoder_scalars = list(config.encoder_layer_scalars or [])
        decoder_scalars = list(config.decoder_layer_scalars or [])
        encoder_value = (float(encoder_scalars[layer_idx])
                         if layer_idx < len(encoder_scalars) else 1.0)
        decoder_value = (float(decoder_scalars[layer_idx])
                         if layer_idx < len(decoder_scalars) else 1.0)
        self.register_buffer(
            "encoder_layer_scalar",
            torch.tensor([encoder_value], dtype=torch.float16))
        self.register_buffer(
            "decoder_layer_scalar",
            torch.tensor([decoder_value], dtype=torch.float16))

    def _layer_scalar(self, phase_is_encoder: torch.Tensor | None,
                      hidden_states: torch.Tensor) -> torch.Tensor:
        if phase_is_encoder is None:
            return self.decoder_layer_scalar.to(dtype=hidden_states.dtype,
                                                device=hidden_states.device)

        encoder_scalar = self.encoder_layer_scalar.to(
            dtype=hidden_states.dtype, device=hidden_states.device)
        decoder_scalar = self.decoder_layer_scalar.to(
            dtype=hidden_states.dtype, device=hidden_states.device)
        phase_mask = phase_is_encoder.reshape(-1, 1, 1).to(torch.bool)
        return torch.where(phase_mask, encoder_scalar, decoder_scalar)


class DiffusionGemmaTransformer(gemma4_text.Gemma4Transformer):
    """Gemma4-family transformer using DiffusionGemma decoder layers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList([
            DiffusionGemmaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])


class DiffusionGemmaBackbone(gemma4_text.Gemma4ForCausalLM):
    """Single phase-aware DiffusionGemma transformer backbone engine."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model = DiffusionGemmaTransformer(config)
        self.diffusion_engine_role = "dllm"
        self.diffusion_unified_conditioning = False
        self.enable_unified_conditioning()

    def enable_unified_conditioning(self) -> None:
        """Embed DiffusionGemma self-conditioning into the backbone graph."""
        if self.diffusion_unified_conditioning:
            return
        if self.config.reduced_vocab_size:
            raise ValueError(
                "Unified DiffusionGemma conditioning does not support "
                "reduced vocabulary exports because self-conditioning "
                "feedback must align with the full embedding table.")
        self.self_conditioning = DiffusionGemmaSelfConditioning(self.config)
        self.register_buffer(
            "diffusion_embedding_scale",
            torch.tensor(float(self.config.embedding_scale),
                         dtype=torch.float32),
            persistent=False,
        )
        self.diffusion_unified_conditioning = True

    def _scaled_embedding_weight(self) -> torch.Tensor:
        weight = self.model.embed_tokens.weight
        scale = self.diffusion_embedding_scale.to(dtype=weight.dtype,
                                                  device=weight.device)
        return (weight * scale).to(torch.float16)

    def _unified_conditioned_inputs(
        self,
        canvas_ids: torch.Tensor,
        prev_self_conditioning_embeds: torch.Tensor,
    ) -> torch.Tensor:
        embedding_weight = self.model.embed_tokens.weight.to(torch.float16)
        token_embeds = F.embedding(canvas_ids.to(torch.long), embedding_weight)
        scale = self.diffusion_embedding_scale.to(dtype=token_embeds.dtype,
                                                  device=token_embeds.device)
        token_embeds = token_embeds * scale
        soft_embeds = prev_self_conditioning_embeds.to(token_embeds.dtype)
        return self.self_conditioning(token_embeds,
                                      soft_embeds).to(token_embeds.dtype)

    def _next_self_conditioning_embeds(
        self,
        logits: torch.Tensor,
        self_conditioning_temperature: torch.Tensor,
    ) -> torch.Tensor:
        embedding_weight = self.model.embed_tokens.weight.to(torch.float16)
        safe_temperature = torch.clamp(self_conditioning_temperature.reshape(
            ()).to(torch.float32),
                                       min=1.0e-6)
        probs = torch.softmax(logits.to(torch.float32) / safe_temperature,
                              dim=-1).to(embedding_weight.dtype)
        soft_embeds = torch.matmul(probs, embedding_weight)
        scale = self.diffusion_embedding_scale.to(dtype=soft_embeds.dtype,
                                                  device=soft_embeds.device)
        return (soft_embeds * scale).to(embedding_weight.dtype)

    def onnx_export_spec(self) -> OnnxSpec:
        config = self.config
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
        phase_is_encoder = torch.ones(batch_size,
                                      dtype=torch.int32,
                                      device=device)
        args = (inputs_embeds, phase_is_encoder)
        input_names = ["inputs_embeds", "phase_is_encoder"]

        if self.diffusion_unified_conditioning:
            canvas_ids = torch.zeros(batch_size,
                                     seq_len,
                                     dtype=torch.int32,
                                     device=device)
            prev_self_conditioning_embeds = torch.zeros(batch_size,
                                                        seq_len,
                                                        config.hidden_size,
                                                        dtype=dtype16,
                                                        device=device)
            self_conditioning_temperature = torch.ones(1,
                                                       dtype=torch.float32,
                                                       device=device)
            args = args + (canvas_ids, prev_self_conditioning_embeds,
                           self_conditioning_temperature)
            input_names = input_names + [
                "canvas_ids",
                "prev_self_conditioning_embeds",
                "self_conditioning_temperature",
            ]

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
                gemma4_text.KV_PAGE_SIZE,
                num_kv_heads,
                layer_head_dim,
                dtype=kv_dtype,
                device=device,
            ) for num_kv_heads, layer_head_dim in (
                gemma4_text._kv_cache_dims_for_layer(config, layer_idx)
                for layer_idx in range(Na))
        ]

        args = args + (*ple_token_embeds_list, *past_key_values_list)
        input_names = (
            input_names +
            [f"ple_token_embeds_{i}" for i in range(num_ple_inputs)] +
            [f"past_key_values_{i}" for i in range(Na)])

        if config.use_dual_rope:
            sliding_head_dim = gemma4_text._head_dim_for_attention_type(
                config, "sliding_attention")
            full_head_dim = gemma4_text._head_dim_for_attention_type(
                config, "full_attention")
            sliding_rotary_dim = gemma4_text._rotary_dim_from_rope_config(
                config, config.sliding_rope_config, sliding_head_dim)
            full_rotary_dim = gemma4_text._rotary_dim_from_rope_config(
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
            rotary_head_dim = gemma4_text._head_dim_for_attention_type(
                config, "full_attention")
            rotary_dim = gemma4_text._rotary_dim_from_rope_config(
                config, None, rotary_head_dim)
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
        select_token_indices = torch.arange(seq_len,
                                            dtype=torch.int64,
                                            device=device).reshape(
                                                batch_size, seq_len)
        context_mask_selector = torch.zeros(_DUMMY_CONTEXT_SELECTOR_LEN,
                                            dtype=torch.int32,
                                            device=device)

        args = args + (context_lengths, kvcache_start_index, kv_page_table,
                       select_token_indices, context_mask_selector)
        input_names = input_names + [
            "context_lengths",
            "kvcache_start_index",
            "kv_page_table",
            "select_token_indices",
            "context_mask_selector",
        ]
        output_names = ["logits"]
        if self.diffusion_unified_conditioning:
            output_names.append("next_self_conditioning_embeds")
        output_names += [f"present_key_values_{i}" for i in range(Na)]

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        max_selected = (int(config.diffusion.canvas_length) if
                        (self.diffusion_unified_conditioning
                         and config.diffusion is not None) else 32768)
        selected = torch.export.Dim("num_selected", min=1, max=max_selected)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        selector_len = torch.export.Dim("context_selector_len", min=0, max=256)

        all_shapes: list = [{0: batch, 1: seq}, {0: batch}]
        if self.diffusion_unified_conditioning:
            # Runtime binds these tensors to the same denoise canvas sequence shape as inputs_embeds.
            all_shapes += [
                {
                    0: batch,
                    1: seq
                },
                {
                    0: batch,
                    1: seq
                },
                {},
            ]
        for _ in range(num_ple_inputs):
            all_shapes.append({0: batch, 1: seq})
        for _ in range(Na):
            all_shapes.append({1: num_pages})
        all_shapes.append({0: rope_batch, 1: pos})
        if config.use_dual_rope:
            all_shapes.append({0: rope_batch, 1: pos})
        all_shapes.append({0: batch})
        all_shapes.append({0: kv_batch})
        all_shapes.append({0: page_batch, 2: max_pages})
        all_shapes.append({0: batch, 1: selected})
        all_shapes.append({0: selector_len})

        wrapped = _make_diffusion_gemma_flat_wrapper(
            self,
            Na,
            num_ple_inputs=num_ple_inputs,
            use_dual_rope=config.use_dual_rope,
            unified_conditioning=self.diffusion_unified_conditioning)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        phase_is_encoder: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor | None,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        select_token_indices: torch.Tensor,
        context_mask_selector: torch.Tensor,
        ple_token_embeds: Tuple[torch.Tensor, ...] = (),
        rope_rotary_cos_sin_sliding: torch.Tensor | None = None,
        rope_rotary_cos_sin_full: torch.Tensor | None = None,
        canvas_ids: torch.Tensor | None = None,
        prev_self_conditioning_embeds: torch.Tensor | None = None,
        self_conditioning_temperature: torch.Tensor | None = None,
    ) -> Tuple:
        if self.diffusion_unified_conditioning:
            if (canvas_ids is None or prev_self_conditioning_embeds is None
                    or self_conditioning_temperature is None):
                raise ValueError(
                    "Unified DiffusionGemma conditioning requires "
                    "canvas_ids, prev_self_conditioning_embeds, and "
                    "self_conditioning_temperature inputs.")
            conditioned_inputs = self._unified_conditioned_inputs(
                canvas_ids, prev_self_conditioning_embeds)
            phase_is_encoder_mask = phase_is_encoder.reshape(-1, 1,
                                                             1).to(torch.bool)
            inputs_embeds = torch.where(phase_is_encoder_mask, inputs_embeds,
                                        conditioned_inputs)

        hidden_states, present_key_values, _ = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            phase_is_encoder=phase_is_encoder,
            context_mask_selector=context_mask_selector,
            output_hidden_states=False,
            ple_token_embeds=ple_token_embeds,
            rope_rotary_cos_sin_sliding=rope_rotary_cos_sin_sliding,
            rope_rotary_cos_sin_full=rope_rotary_cos_sin_full,
        )
        selected_hidden_states = torch.ops.trt.gather_nd(
            hidden_states, select_token_indices)
        logits = self.lm_head(selected_hidden_states).to(torch.float32)
        if (self.config.final_logit_softcapping is not None
                and self.config.final_logit_softcapping > 0.0):
            scale = logits.new_tensor(self.config.final_logit_softcapping)
            logits = torch.tanh(logits / scale) * scale
        if self.diffusion_unified_conditioning:
            next_self_conditioning_embeds = self._next_self_conditioning_embeds(
                logits, self_conditioning_temperature)
            return logits, next_self_conditioning_embeds, present_key_values
        return logits, present_key_values


class DiffusionGemmaSelfConditioning(nn.Module):
    """Small DiffusionGemma self-conditioning MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        sc_size = int(config.self_conditioning_size
                      or config.intermediate_size)
        self.pre_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = make_linear(config,
                                     hidden_size,
                                     sc_size,
                                     bias=False,
                                     module_name="self_conditioning.gate_proj",
                                     tp_mode=TPMode.COL)
        self.up_proj = make_linear(config,
                                   hidden_size,
                                   sc_size,
                                   bias=False,
                                   module_name="self_conditioning.up_proj",
                                   tp_mode=TPMode.COL)
        self.down_proj = make_linear(config,
                                     sc_size,
                                     hidden_size,
                                     bias=False,
                                     module_name="self_conditioning.down_proj",
                                     tp_mode=TPMode.ROW)
        # DiffusionGemma checkpoints do not carry a post-norm weight, so this
        # default unit-weight RMSNorm is mathematically equivalent to a
        # weightless RMSNorm. Keeping the explicit weight preserves the ONNX
        # graph form that TensorRT/Myelin compiles reliably on Thor.
        self.post_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor,
                soft_embeds: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(soft_embeds)
        sc = self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        return self.post_norm(inputs_embeds + sc)
