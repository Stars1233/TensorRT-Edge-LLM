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
"""Nemotron-H MTP (multi-token-prediction) draft model for ONNX export.

Structure follows the checkpoint's ``NemotronHMultiTokenPredictor`` (whose HF
``forward`` is a stub) plus the DeepSeek-V3 MTP forward it is derived from:

    mtp.layers.0  (is_first, attention): enorm, hnorm, eh_proj, norm, mixer(attn)
    mtp.layers.1  (is_last,  MoE):       norm, mixer(MoE), final_layernorm

The one MTP prediction module (``num_nextn_predict_layers == 1``) is a hybrid
stack whose layer types come from ``mtp_hybrid_override_pattern`` ("*E"). The
fusion ``eh_proj(concat(enorm(embed), hnorm(hidden)))`` merges the current
token embedding with the previous hidden state (DeepSeek-V3 / Eagle order:
embedding first). The draft's experts are unquantized FP16 (routed through the
sigmoid-group-topk ``Fp16MoePlugin``); the lm_head is borrowed from the
quantized base model.

Module attribute paths equal the checkpoint keys with the ``mtp.`` prefix
stripped (the MTP key-remap), so weights load without a bespoke remap.
"""

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...checkpoint.repacking import repack_fp16_moe_experts
from ...config import LAYER_ATTN, LAYER_MOE, ModelConfig
from ..default.modeling_default import OnnxSpec
from ..linear import make_linear
from ..ops import KV_PAGE_SIZE, fp16_moe_plugin_sigmoid
from .modeling_nemotron_h import (NemotronHAttentionMixer, NemotronHTopkRouter,
                                  RMSNorm)

__all__ = ["NemotronHMtpDraftModel"]

_BATCH_SIZE = 2
_SEQ_LEN = 2
_PAST_LEN = 1
_MAX_POS = 4096

_ACTIVATION_RELU2 = 4
_ROUTING_SIGMOID_GROUP_TOPK = 1
_MOE_MAX_ROUTED_ROWS_AUTO = 0


class _MtpLayerNorm(nn.Module):
    """LayerNorm without bias (``nn.LayerNorm(bias=False)`` in the checkpoint).

    Unlike RMSNorm this subtracts the mean. Decomposed into primitive ops (like
    the base model's RMSNorm) so the ONNX graph stays TRT-parseable and the
    FP32->FP16 cast forms a clean partition boundary.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size,
                                              dtype=torch.float16))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states = hidden_states - mean
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight.to(input_dtype) * hidden_states


class _MtpReLU2MLP(nn.Module):
    """Ungated ReLU² MLP (up_proj -> relu² -> down_proj), FP16.

    Used for the routed experts (repack reads ``.weight``) and the shared
    expert (needs ``forward``).
    """

    def __init__(self, config: ModelConfig, hidden_size: int,
                 intermediate_size: int, module_prefix: str) -> None:
        super().__init__()
        self.up_proj = make_linear(config,
                                   hidden_size,
                                   intermediate_size,
                                   module_name=f"{module_prefix}.up_proj")
        self.down_proj = make_linear(config,
                                     intermediate_size,
                                     hidden_size,
                                     module_name=f"{module_prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r = F.relu(self.up_proj(hidden_states))
        return self.down_proj(r * r)


class NemotronHMtpFp16MoE(nn.Module):
    """Unquantized FP16 MoE mixer for the MTP draft.

    Routed experts run through the sigmoid-group-topk ``Fp16MoePlugin`` (via
    :func:`fp16_moe_plugin_sigmoid`); the shared expert is a dense ReLU² MLP.
    Same routing math as the base ``NemotronHMoEMLP`` but plain FP16 weights.
    """

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.activation_type = _ACTIVATION_RELU2
        self.max_routed_rows = _MOE_MAX_ROUTED_ROWS_AUTO
        # FC1_N is padded to a multiple of 128 by the repack helper.
        self._padded_moe_inter = (
            (self.moe_intermediate_size + 127) // 128) * 128

        self.gate = NemotronHTopkRouter(config)
        self.experts = nn.ModuleList([
            _MtpReLU2MLP(config, self.hidden_size, self.moe_intermediate_size,
                         f"{module_prefix}.experts.{j}")
            for j in range(self.n_routed_experts)
        ])
        self.shared_experts = _MtpReLU2MLP(
            config, self.hidden_size,
            config.moe_shared_expert_intermediate_size,
            f"{module_prefix}.shared_experts")
        self._export_ready = False

    def prepare_for_export(self) -> None:
        if self._export_ready:
            return
        fc1_weights, fc2_weights, padded_inter = repack_fp16_moe_experts(
            self.experts, self.hidden_size, self.moe_intermediate_size,
            self.activation_type)
        self._padded_moe_inter = padded_inter
        device = self.gate.weight.device
        self.register_buffer("fc1_weights",
                             fc1_weights.to(device).contiguous())
        self.register_buffer("fc2_weights",
                             fc2_weights.to(device).contiguous())
        self.register_buffer(
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.clone().to(
                device=device, dtype=torch.float32))
        # Raw per-expert buffers are now stacked into fc1/fc2; drop them.
        self.experts = nn.ModuleList()
        self._export_ready = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = F.linear(hidden_states.view(-1, self.hidden_size),
                                 self.gate.weight).float()
        moe_out = fp16_moe_plugin_sigmoid(
            router_logits,
            hidden_states,
            self.fc1_weights,
            self.fc2_weights,
            self._e_score_correction_bias_fp32,
            self.n_routed_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self._padded_moe_inter,
            self.activation_type,
            self.gate.n_group,
            self.gate.topk_group,
            int(bool(self.gate.norm_topk_prob)),
            float(self.gate.routed_scaling_factor),
            self.max_routed_rows,
        )
        return moe_out + self.shared_experts(hidden_states)


class NemotronHMtpLayer(nn.Module):
    """One draft layer: pre-norm + mixer (+ Eagle fusion / final norm).

    ``is_first`` layers additionally own the ``eh_proj`` fusion (enorm/hnorm);
    ``is_last`` layers own ``final_layernorm`` (applied by the model after the
    last-token gather). Attribute names match the ``mtp.``-stripped keys.
    """

    def __init__(self, config: ModelConfig, layer_idx: int, layer_type: str,
                 is_first: bool, is_last: bool) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.is_first = is_first
        self.is_last = is_last
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        prefix = f"layers.{layer_idx}"

        if is_first:
            self.enorm = RMSNorm(hidden, eps)
            self.hnorm = RMSNorm(hidden, eps)
            self.eh_proj = make_linear(config,
                                       2 * hidden,
                                       hidden,
                                       bias=False,
                                       module_name=f"{prefix}.eh_proj")

        self.norm = RMSNorm(hidden, eps)
        if layer_type == LAYER_ATTN:
            self.mixer = NemotronHAttentionMixer(
                config,
                layer_idx,
                module_prefix=f"{prefix}.mixer",
                enable_tree_attention=True)
        elif layer_type == LAYER_MOE:
            self.mixer = NemotronHMtpFp16MoE(config,
                                             module_prefix=f"{prefix}.mixer")
        else:
            raise ValueError(
                f"Unsupported MTP draft layer type: {layer_type!r}")

        if is_last:
            self.final_layernorm = _MtpLayerNorm(hidden, config.rms_norm_eps)

    def _fuse(self, inputs_embeds: torch.Tensor,
              hidden_states: torch.Tensor) -> torch.Tensor:
        # DeepSeek-V3 / Eagle order: [enorm(embedding), hnorm(hidden)].
        return self.eh_proj(
            torch.cat((self.enorm(inputs_embeds), self.hnorm(hidden_states)),
                      dim=-1))

    def forward_attention(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pos_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_first:
            hidden_states = self._fuse(inputs_embeds, hidden_states)
        residual = hidden_states
        attn_out, present_kv = self.mixer(
            self.norm(hidden_states),
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            attention_mask,
            attention_pos_id,
        )
        return residual + attn_out, present_kv

    def forward_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mixer(self.norm(hidden_states))


def _make_flat_wrapper_nemotron_h_mtp(model: nn.Module,
                                      num_attn_layers: int) -> nn.Module:
    """Flat-signature wrapper for Nemotron-H MTP draft ONNX export."""
    param_names: List[str] = (
        ["inputs_embeds"] +
        [f"past_key_values_{i}" for i in range(num_attn_layers)] + [
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "kv_page_table",
            "last_token_ids",
            "hidden_states_input",
            "hidden_states_from_draft",
            "attention_pos_id",
            "attention_mask",
        ])
    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}"
        for i in range(num_attn_layers))) if num_attn_layers else "()"
    body = (
        f"    logits, hidden_states, present_key_values = self._model(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin,\n"
        f"        context_lengths, kvcache_start_index, kv_page_table,\n"
        f"        last_token_ids,\n"
        f"        hidden_states_input, hidden_states_from_draft,\n"
        f"        attention_pos_id, attention_mask)\n"
        f"    return (logits, hidden_states) + tuple(present_key_values)\n")
    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


class NemotronHMtpDraftModel(nn.Module):
    """Nemotron-H MTP draft model (hybrid attention + FP16 MoE)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        layer_types = config.layer_types
        n = len(layer_types)
        self.layers = nn.ModuleList([
            NemotronHMtpLayer(config,
                              layer_idx=i,
                              layer_type=layer_types[i],
                              is_first=(i == 0),
                              is_last=(i == n - 1)) for i in range(n)
        ])
        self.num_attn_layers = sum(1 for t in layer_types if t == LAYER_ATTN)
        # lm_head is borrowed from the (quantized) base model.
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        kv_page_table: torch.Tensor,
        last_token_ids: torch.Tensor,
        hidden_states_from_base: torch.Tensor,
        hidden_states_from_draft: torch.Tensor,
        attention_pos_id: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        # Add-mux: the runtime zeroes exactly one input (base on accept/prefill,
        # draft on proposal steps), so Add selects the live hidden state.
        hidden_states = hidden_states_from_base + hidden_states_from_draft

        present_key_values: List[torch.Tensor] = []
        attn_idx = 0
        last_layer = self.layers[-1]
        for layer in self.layers:
            if layer.layer_type == LAYER_ATTN:
                hidden_states, present_kv = layer.forward_attention(
                    inputs_embeds,
                    hidden_states,
                    past_key_values[attn_idx],
                    rope_rotary_cos_sin,
                    context_lengths,
                    kvcache_start_index,
                    kv_page_table,
                    attention_mask,
                    attention_pos_id,
                )
                present_key_values.append(present_kv)
                attn_idx += 1
            else:
                hidden_states = layer.forward_moe(hidden_states)

        # Gather the predicted-token positions, then the draft's final LayerNorm.
        hidden_states = torch.ops.trt.gather_nd(hidden_states, last_token_ids)
        hidden_states = last_layer.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states).to(torch.float32)
        logits = F.log_softmax(logits, dim=-1)
        return logits, hidden_states, tuple(present_key_values)

    def onnx_export_spec(self) -> OnnxSpec:
        for layer in self.layers:
            if hasattr(layer.mixer, "prepare_for_export"):
                layer.mixer.prepare_for_export()

        config = self.config
        na = self.num_attn_layers
        device = next(itertools.chain(self.parameters(),
                                      self.buffers())).device
        dtype16 = torch.float16
        batch_size, seq_len, past_len, max_pos = (_BATCH_SIZE, _SEQ_LEN,
                                                  _PAST_LEN, _MAX_POS)

        inputs_embeds = torch.zeros(batch_size,
                                    seq_len,
                                    config.hidden_size,
                                    dtype=dtype16,
                                    device=device)
        kv_dtype = (torch.float8_e4m3fn
                    if config.quant.kv_cache_quant == "fp8" else dtype16)
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(2,
                        1,
                        KV_PAGE_SIZE,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(na)
        ]
        rope_rotary_cos_sin = torch.zeros(batch_size,
                                          max_pos,
                                          config.head_dim,
                                          dtype=torch.float32,
                                          device=device)
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
        hidden_states_input = torch.zeros(batch_size,
                                          seq_len,
                                          config.hidden_size,
                                          dtype=dtype16,
                                          device=device)
        hidden_states_from_draft = torch.zeros(batch_size,
                                               seq_len,
                                               config.hidden_size,
                                               dtype=dtype16,
                                               device=device)
        attention_pos_id = torch.zeros(batch_size,
                                       seq_len,
                                       dtype=torch.int32,
                                       device=device)
        attention_mask = torch.zeros(batch_size,
                                     seq_len,
                                     seq_len + past_len,
                                     dtype=torch.int32,
                                     device=device)

        args = (
            inputs_embeds,
            *past_key_values_list,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            last_token_ids,
            hidden_states_input,
            hidden_states_from_draft,
            attention_pos_id,
            attention_mask,
        )
        input_names = (["inputs_embeds"] +
                       [f"past_key_values_{i}" for i in range(na)] + [
                           "rope_rotary_cos_sin",
                           "context_lengths",
                           "kvcache_start_index",
                           "kv_page_table",
                           "last_token_ids",
                           "hidden_states_input",
                           "hidden_states_from_draft",
                           "attention_pos_id",
                           "attention_mask",
                       ])
        output_names = (["logits", "hidden_states"] +
                        [f"present_key_values_{i}" for i in range(na)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)
        page_batch = torch.export.Dim("page_batch", min=1, max=256)
        max_pages = torch.export.Dim("max_pages_per_seq", min=1, max=32768)
        num_pages = torch.export.Dim("num_pages", min=1, max=1048576)
        attn_seq = torch.export.Dim("attn_seq_len", min=1, max=32768)
        num_selected = torch.export.Dim("num_selected", min=1, max=256)
        mask_kv_len = torch.export.Dim("mask_kv_len", min=1, max=65536)

        dynamic_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(na):
            dynamic_shapes.append({1: num_pages})  # past_key_values_i
        dynamic_shapes.append({0: rope_batch, 1: pos})  # rope_rotary_cos_sin
        dynamic_shapes.append({0: batch})  # context_lengths
        dynamic_shapes.append({0: kv_batch})  # kvcache_start_index
        dynamic_shapes.append({0: page_batch, 2: max_pages})  # kv_page_table
        dynamic_shapes.append({0: batch, 1: num_selected})  # last_token_ids
        dynamic_shapes.append({0: batch, 1: seq})  # hidden_states_input
        dynamic_shapes.append({0: batch, 1: seq})  # hidden_states_from_draft
        dynamic_shapes.append({0: batch, 1: attn_seq})  # attention_pos_id
        dynamic_shapes.append({0: batch, 1: attn_seq, 2: mask_kv_len})  # mask

        wrapped = _make_flat_wrapper_nemotron_h_mtp(self, na)
        wrapped.eval()
        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=dynamic_shapes)
