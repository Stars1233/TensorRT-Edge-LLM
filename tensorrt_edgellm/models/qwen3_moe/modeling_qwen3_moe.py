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
"""
Qwen3 MoE causal LM modeling for checkpoint export.

Architecture
------------
Standard decoder transformer with sparse MoE layers interleaved at a
frequency controlled by ``decoder_sparse_step`` (and optionally overridden
per-layer by ``mlp_only_layers``).  Attention is identical to Qwen3 dense
(GQA with per-head QK-norm, optional sliding window).

This module is a standalone reimplementation — it shares only leaf
building blocks (``RMSNorm``, ``Attention``, ``MLP``, ``OnnxSpec``,
``_make_flat_wrapper``) with the default modeling, following the same
decoupling convention used by ``nemotron_h``.

MoE layer design
----------------
The entire Qwen3SparseMoeBlock — router (Linear + Softmax + TopK) and all
expert GEMMs (gate/up/down projections with SiLU gating) plus weighted
combine — is captured as a single custom MoE op. GPTQ/AWQ exports keep using
``trt_edgellm::Int4MoePlugin``. NVFP4 exports use the CuTeDSL GeForce
``trt_edgellm::NvFP4MoEPluginGeforce`` contract.

Per-expert weights are loaded into an ``nn.ModuleList`` for checkpoint
compatibility, then stacked into 3-D tensors ``[E, ...]`` during the
post-load repacking pass.

Checkpoint key structure (GPTQ per-expert layout)
--------------------------------------------------
model.embed_tokens.weight
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,...}
model.layers.{i}.self_attn.{q,k}_norm.weight
model.layers.{i}.mlp.gate_proj.weight          (dense-MLP layers)
model.layers.{i}.mlp.up_proj.weight            (dense-MLP layers)
model.layers.{i}.mlp.down_proj.weight          (dense-MLP layers)
model.layers.{i}.mlp.gate.weight               (MoE layers — router)
model.layers.{i}.mlp.experts.{j}.gate_proj.*   (MoE layers — per-expert)
model.layers.{i}.mlp.experts.{j}.up_proj.*     (MoE layers — per-expert)
model.layers.{i}.mlp.experts.{j}.down_proj.*   (MoE layers — per-expert)
model.norm.weight
lm_head.weight
"""

import itertools
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

from ...config import NVFP4_MOE_BACKEND_GEFORCE, QUANT_NVFP4, ModelConfig
from ..default.modeling_default import (MLP, Attention, OnnxSpec, RMSNorm,
                                        _make_flat_wrapper)
from ..linear import FP16Linear, make_linear
from ..ops import int4_moe_plugin, nvfp4_moe_plugin, nvfp4_moe_plugin_geforce

logger = logging.getLogger(__name__)

# SiLU activation type expected by the C++ Int4MoePlugin.
_INT4_ACTIVATION_SILU = 0

# Activation_type for the Marlin/CuTeDSL ``Nvfp4MoePlugin`` (Thor SM110
# backend). The plugin folds gated SwiGLU internally via the 32-column
# interleaved (up, gate) FC1 weight layout and ``nOutFor(SiLU, 2*I) = I``; see
# ``cpp/plugins/nvfp4MoePlugin/nvfp4MoePlugin.cpp::mapActivation``.
_NVFP4_THOR_ACTIVATION_SILU = 1
# Plugin slot 0 — softmax + flat top-k + renormalize routing kernel
# (``Nvfp4MoeRoutingMode::kSOFTMAX_TOPK``); Qwen3 does not use the
# sigmoid-group-topk path.
_NVFP4_ROUTING_MODE_SOFTMAX_TOPK = 0

# SwiGLU activation_type for the GeForce SM120/121 ``NvFP4MoEPluginGeforce``
# (different integer mapping than the Thor plugin: kSwiGLU=2 there).
_NVFP4_GEFORCE_ACTIVATION_SWIGLU = 2
_GEFORCE_BACKEND_AUTO = 0
_GEFORCE_IO_DTYPE_FP16 = 1
_GEFORCE_MAX_ROUTED_ROWS_AUTO = 0

# ONNX export dummy-input dims.
_BATCH_SIZE = 1
_SEQ_LEN = 1
_PAST_LEN = 1
_MAX_POS = 4096

__all__ = [
    "Qwen3MoERouter",
    "Qwen3MoEExperts",
    "Qwen3SparseMoeBlock",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeTransformer",
    "Qwen3MoeCausalLM",
]

# ---------------------------------------------------------------------------
# MoE layer predicate
# ---------------------------------------------------------------------------


def _is_moe_layer(config: ModelConfig, layer_idx: int) -> bool:
    """Return True if *layer_idx* should use a Qwen3SparseMoeBlock.

    Mirrors HuggingFace Qwen3MoeDecoderLayer logic:
    - Any layer listed in ``mlp_only_layers`` is always dense.
    - Otherwise, layer is MoE when ``(layer_idx + 1) % decoder_sparse_step == 0``
      and ``num_experts > 0``.
    """
    if layer_idx in config.mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_idx +
                                       1) % config.decoder_sparse_step == 0


# ---------------------------------------------------------------------------
# MoE router
# ---------------------------------------------------------------------------


class Qwen3MoERouter(nn.Module):
    """Router weight holder.

    Checkpoint key: ``mlp.gate.weight`` [num_experts, hidden_size].
    The actual routing computation (linear + softmax + topk) is fused
    inside the ``trt_edgellm::Int4MoePlugin`` custom op.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_experts,
                        config.hidden_size,
                        dtype=torch.float16),
            requires_grad=False,
        )


# ---------------------------------------------------------------------------
# MoE experts (weight holder)
# ---------------------------------------------------------------------------


class Qwen3MoEExperts(nn.Module):
    """Per-expert MLP modules stored as nn.ModuleList.

    Supports all checkpoint formats including per-expert quantized layouts
    (GPTQ, AWQ) where each expert has independent quantized weight tensors.

    Checkpoint keys (under ``mlp.experts``):
        {i}.gate_proj.*  -- i-th expert gate projection
        {i}.up_proj.*    -- i-th expert up projection
        {i}.down_proj.*  -- i-th expert down projection

    Non-quantized (FP16) checkpoints that use stacked 3-D parameters
    (``gate_up_proj`` / ``down_proj``) are handled by weight-loading
    remapping in the checkpoint loader, not here.
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


# ---------------------------------------------------------------------------
# Sparse MoE block
# ---------------------------------------------------------------------------


class Qwen3SparseMoeBlock(nn.Module):
    """Sparse MoE block exported via a fused TRT MoE plugin.

    In the ONNX graph this produces:
    - One ``MatMul`` for the gate (router) linear -- traced from ``gate_linear``
    - One fused MoE plugin node for softmax + topk + expert GEMMs

    Weight loading: ``gate`` (Qwen3MoERouter) and ``experts`` (Qwen3MoEExperts as
    nn.ModuleList) hold per-expert parameters for checkpoint loading.
    After loading, :meth:`_prepare_moe_weights` (called by the repacking
    pass) extracts GPTQ weights, repacks to Marlin format, fuses gate+up
    projections, and registers the result as buffers on this module.

    Checkpoint sub-keys under ``mlp``:
        gate.*        -> Qwen3MoERouter (weight holder)
        experts.*     -> Qwen3MoEExperts (per-expert modules for loading)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.group_size = config.quant.group_size
        self.zero_point_offset = config.quant.gptq_zero_point_offset
        self._use_nvfp4_moe = config.quant.quant_type == QUANT_NVFP4
        # NVFP4 backend dispatch: Thor (CuTeDSL ``Nvfp4MoePlugin``,
        # SM100/101/110) or GeForce (``NvFP4MoEPluginGeforce``, SM120/121).
        # See ``QuantConfig.nvfp4_moe_backend`` and the
        # ``NVFP4_MOE_BACKEND_*`` constants in ``tensorrt_edgellm/config.py``.
        self._nvfp4_moe_backend = config.quant.nvfp4_moe_backend
        self._use_nvfp4_geforce = (self._use_nvfp4_moe
                                   and self._nvfp4_moe_backend
                                   == NVFP4_MOE_BACKEND_GEFORCE)
        # ``activation_type`` integer is plugin-specific — ints carry
        # different meanings between Int4MoePlugin (SiLU=0), Nvfp4MoePlugin
        # (SiLU=1) and NvFP4MoEPluginGeforce (SwiGLU=2). The per-path
        # constants above document each mapping; see the corresponding C++
        # plugin headers for the source of truth.
        if not self._use_nvfp4_moe:
            self.activation_type = _INT4_ACTIVATION_SILU
        elif self._use_nvfp4_geforce:
            self.activation_type = _NVFP4_GEFORCE_ACTIVATION_SWIGLU
        else:
            self.activation_type = _NVFP4_THOR_ACTIVATION_SILU
        # GeForce-only attributes; values track the defaults consumed by
        # ``NvFP4MoEPluginGeforce``.
        self.backend = _GEFORCE_BACKEND_AUTO
        self.io_dtype = _GEFORCE_IO_DTYPE_FP16
        self.max_routed_rows = _GEFORCE_MAX_ROUTED_ROWS_AUTO
        self.gate = Qwen3MoERouter(config)
        self.experts = Qwen3MoEExperts(config)

    def _prepare_moe_weights(self) -> None:
        """Extract expert weights and prepare plugin buffers.

        Called by :func:`~checkpoint.repacking._stack_moe_experts` BEFORE
        regular GPTQ repacking.  After extraction, per-expert ``qweight``
        buffers are set to ``None`` so ``_repack_gptq_weights`` skips them.
        """
        if self._use_nvfp4_moe:
            self._prepare_nvfp4_moe_weights()
            return

        from ...checkpoint.repacking import (_extract_gptq_for_marlin,
                                             pack_int4_awq_marlin)

        # Promote Qwen3MoERouter weight -> nn.Linear for standard MatMul trace.
        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        # Extract per-expert GPTQ weights -> [N, K] int16 + [N, groups] fp16.
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

        # Stack [E, N, K] and Marlin-pack.
        gate_up_w = torch.stack(gate_up_weights_list, dim=0)
        gate_up_s = torch.stack(gate_up_scales_list, dim=0)
        down_w = torch.stack(down_weights_list, dim=0)
        down_s = torch.stack(down_scales_list, dim=0)

        gu_marlin_w, gu_marlin_s = pack_int4_awq_marlin(
            gate_up_w, gate_up_s, self.group_size)
        dn_marlin_w, dn_marlin_s = pack_int4_awq_marlin(
            down_w, down_s, self.group_size)

        # Store as int8 (Marlin int32 viewed as int8).
        self.register_buffer("fc_gate_up_qweights",
                             gu_marlin_w.view(torch.int8).contiguous())
        self.register_buffer("fc_gate_up_scales", gu_marlin_s.contiguous())
        self.register_buffer("fc_down_qweights",
                             dn_marlin_w.view(torch.int8).contiguous())
        self.register_buffer("fc_down_scales", dn_marlin_s.contiguous())

        logger.info(
            "Marlin-packed %d experts: gate_up_qw %s, down_qw %s",
            self.num_experts,
            list(self.fc_gate_up_qweights.shape),
            list(self.fc_down_qweights.shape),
        )

        # Discard per-expert modules — weights are now in the stacked Marlin
        # buffers above.  This also prevents _repack_gptq_weights from seeing
        # the (now-consumed) per-expert qweight buffers.
        self.experts = nn.ModuleList()

    def _prepare_nvfp4_moe_weights(self) -> None:
        """Backend-agnostic dispatcher: pick Thor or GeForce repack."""
        if self._use_nvfp4_geforce:
            self._prepare_nvfp4_moe_weights_geforce()
        else:
            self._prepare_nvfp4_moe_weights_thor()

    def _prepare_nvfp4_moe_weights_thor(self) -> None:
        """Repack Qwen3 NVFP4 experts for ``Nvfp4MoePlugin`` (Thor SM110, gated SwiGLU).

        Both the prefill grouped GEMM and the decode GEMV are CuTeDSL AOT
        cubins (``cute_dsl_nvfp4_moe_fc{1,2}_*`` and ``gemv_{up,dn}_*``;
        target SM100/101/110); they share a single N-major weight buffer.

        Per-expert ``gate_proj`` / ``up_proj`` are interleaved into a single
        FC1 weight ``[H, 2*I]`` as 32-column ``(up, gate)`` chunks and packed
        N-major; ``down_proj`` is packed N-major as ``[I, H/2]``.
        The plugin sees ``moe_inter_size = 2*I`` (the FC1 N-axis) and
        ``activation_type = 1`` (SiLU); ``nOutFor(SiLU, 2*I) = I`` recovers
        the post-SwiGLU intermediate dim. Activation NVFP4 quantization is
        done inside the plugin via ``fp4Quantize`` — only the per-tensor
        global scales need to be threaded through (slot 2:
        ``hidden_global_scale [2]``).
        """
        from ...checkpoint.repacking import repack_nvfp4_qwen3_moe_experts_thor

        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        (
            fc_up_qw,
            fc_up_bs,
            fc_up_gs,
            fc_down_qw,
            fc_down_bs,
            fc_down_gs,
            fc_up_bs_dec,
            fc_down_bs_dec,
        ) = repack_nvfp4_qwen3_moe_experts_thor(self.experts, self.hidden_size,
                                                self.moe_intermediate_size,
                                                self.group_size)

        device = self.gate.weight.device
        self.register_buffer("fc_up_qweights",
                             fc_up_qw.to(device).contiguous())
        self.register_buffer("fc_up_blocks_scale",
                             fc_up_bs.to(device).contiguous())
        self.register_buffer("fc_up_global_scale",
                             fc_up_gs.to(device).contiguous())
        self.register_buffer("fc_down_qweights",
                             fc_down_qw.to(device).contiguous())
        self.register_buffer("fc_down_blocks_scale",
                             fc_down_bs.to(device).contiguous())
        self.register_buffer("fc_down_global_scale",
                             fc_down_gs.to(device).contiguous())
        # Slots 9 / 10 are decode-scale slots; decode now reads the prefill SF
        # at slots 4 / 7, but the plugin still validates them as INT8 tensors.
        self.register_buffer("fc_up_blocks_scale_decode",
                             fc_up_bs_dec.to(device).contiguous())
        self.register_buffer("fc_down_blocks_scale_decode",
                             fc_down_bs_dec.to(device).contiguous())

        # Length-2 [FC1, FC2] activation global scales — max across experts of
        # ``NVFP4Linear.input_scale`` (already ``amax / (6 * 448)`` per linear
        # forward; see ``models/linear.py:NVFP4Linear``). Gate and up share
        # the same activation, so FC1's scale folds in both.
        fc1_act_gs = max(
            max(float(e.gate_proj.input_scale.detach().float().max().item()),
                float(e.up_proj.input_scale.detach().float().max().item()))
            for e in self.experts)
        fc2_act_gs = max(
            float(e.down_proj.input_scale.detach().float().max().item())
            for e in self.experts)
        self.register_buffer(
            "hidden_global_scale",
            torch.tensor([fc1_act_gs, fc2_act_gs],
                         dtype=torch.float32,
                         device=device))

        # Qwen3 routes via softmax-topk and does not consume an
        # e_score_correction_bias; pass zeros so the bias is a no-op.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts, dtype=torch.float32, device=device))

        logger.info(
            "Nvfp4MoePlugin (Thor gated)-packed %d Qwen3 experts: "
            "fc_up_qw %s, fc_down_qw %s",
            self.num_experts,
            list(self.fc_up_qweights.shape),
            list(self.fc_down_qweights.shape),
        )

        self.experts = nn.ModuleList()

    def _prepare_nvfp4_moe_weights_geforce(self) -> None:
        """Repack Qwen3 NVFP4 experts for ``NvFP4MoEPluginGeforce`` (SM120/121).

        Decodes ModelOpt NVFP4 to dense, rounds through BF16, and emits the
        CuTeDSL 6D MMA scale layout the GeForce fused kernel expects. FC1
        is ``[up, gate]`` along the M axis (the SwiGLU order the GeForce
        kernel hard-codes; see ``kernelSrcs/nvfp4_fused_moe_cutedsl/`` and
        :func:`repack_nvfp4_qwen3_moe_experts_geforce`).
        """
        from ...checkpoint.repacking import \
            repack_nvfp4_qwen3_moe_experts_geforce

        self.gate_linear = nn.Linear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     dtype=torch.float16)
        self.gate_linear.weight.data = self.gate.weight.data

        fc1_qweights, fc1_blocks_scale, fc2_qweights, fc2_blocks_scale = (
            repack_nvfp4_qwen3_moe_experts_geforce(self.experts,
                                                   self.hidden_size,
                                                   self.moe_intermediate_size,
                                                   self.group_size))

        device = self.gate.weight.device
        self.register_buffer("fc1_qweights",
                             fc1_qweights.to(device).contiguous())
        self.register_buffer("fc1_blocks_scale",
                             fc1_blocks_scale.to(device).contiguous())
        self.register_buffer("fc2_qweights",
                             fc2_qweights.to(device).contiguous())
        self.register_buffer("fc2_blocks_scale",
                             fc2_blocks_scale.to(device).contiguous())
        # FP32 alpha + activation scale buffers: one per expert. Populated
        # by the calibration pipeline; default to ones so an uncalibrated
        # checkpoint still produces numerically reasonable outputs.
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

        logger.info(
            "NvFP4MoEPluginGeforce (CuTeDSL fused)-packed %d Qwen3 experts: "
            "fc1_qw %s, fc2_qw %s",
            self.num_experts,
            list(self.fc1_qweights.shape),
            list(self.fc2_qweights.shape),
        )

        self.experts = nn.ModuleList()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate_linear(hidden_flat).float()
        if self._use_nvfp4_geforce:
            return nvfp4_moe_plugin_geforce(
                router_logits,
                hidden_states,
                self.fc1_qweights,
                self.fc1_blocks_scale,
                self.fc1_alpha,
                self.fc2_qweights,
                self.fc2_blocks_scale,
                self.fc2_alpha,
                self.input_global_scale,
                self.down_input_scale,
                self.num_experts,
                self.top_k,
                self.hidden_size,
                self.moe_intermediate_size,
                self.activation_type,
                self.backend,
                self.io_dtype,
                self.max_routed_rows,
            )
        if self._use_nvfp4_moe:
            # Thor path. FC1 width passed to the plugin is 2*I; the plugin
            # reads 32-column interleaved (up, gate) weights and folds SwiGLU
            # in the FC1 epilogue (``nOutFor(SiLU, 2*I) = I``).
            return nvfp4_moe_plugin(
                router_logits,
                hidden_states,
                self.hidden_global_scale,
                self.fc_up_qweights,
                self.fc_up_blocks_scale,
                self.fc_up_global_scale,
                self.fc_down_qweights,
                self.fc_down_blocks_scale,
                self.fc_down_global_scale,
                self.fc_up_blocks_scale_decode,
                self.fc_down_blocks_scale_decode,
                self.e_score_correction_bias,
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                moe_inter_size=2 * self.moe_intermediate_size,
                activation_type=self.activation_type,
                n_group=1,
                topk_group=1,
                norm_topk_prob=1,
                routed_scaling_factor=1.0,
                routing_mode=_NVFP4_ROUTING_MODE_SOFTMAX_TOPK,
            )
        return int4_moe_plugin(
            router_logits,
            hidden_states,
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


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Qwen3MoeDecoderLayer(nn.Module):
    """Single Qwen3 MoE decoder layer.

    Submodule names match checkpoint keys:
        self_attn, mlp, input_layernorm, post_attention_layernorm

    ``mlp`` is a :class:`Qwen3SparseMoeBlock` for MoE-designated layers
    (per :func:`_is_moe_layer`) and a dense :class:`MLP` otherwise.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx=layer_idx)
        if _is_moe_layer(config, layer_idx):
            self.mlp = Qwen3SparseMoeBlock(config)
        else:
            self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            past_key_value,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class Qwen3MoeTransformer(nn.Module):
    """Full Qwen3 MoE decoder stack.

    Stored as ``model`` inside :class:`Qwen3MoeCausalLM` so parameter keys
    carry the ``model.`` prefix matching safetensors checkpoint keys.

    Submodules: ``embed_tokens``, ``layers``, ``norm``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []

        for layer_index, layer in enumerate(self.layers):
            hidden_states, next_key_value = layer(
                hidden_states,
                past_key_values[layer_index],
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
            )
            present_key_values_list.append(next_key_value)

        return self.norm(hidden_states), tuple(present_key_values_list)


# ---------------------------------------------------------------------------
# CausalLM
# ---------------------------------------------------------------------------


class Qwen3MoeCausalLM(nn.Module):
    """Qwen3 MoE causal LM: Transformer + lm_head.

    The inner transformer is stored as attribute ``model`` so parameter keys
    carry the ``model.`` prefix matching checkpoint key prefixes.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoeTransformer(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        """Clone embed_tokens.weight into lm_head.weight when tie_word_embeddings=True."""
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        embed_weight = self.model.embed_tokens.weight
        self.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                           requires_grad=False)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        config = self.config
        Na = config.num_hidden_layers
        Nd = 0  # Qwen3-MoE has no deepstack visual embeddings
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
            torch.zeros(batch_size,
                        2,
                        config.num_key_value_heads,
                        past_len,
                        config.head_dim,
                        dtype=kv_dtype,
                        device=device) for _ in range(Na)
        ]
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        rope_rotary_cos_sin = torch.zeros(batch_size,
                                          max_pos,
                                          rotary_dim,
                                          dtype=torch.float32,
                                          device=device)
        context_lengths = torch.zeros(batch_size,
                                      dtype=torch.int32,
                                      device=device)
        kvcache_start_index = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)
        last_token_ids = torch.zeros(batch_size,
                                     1,
                                     dtype=torch.int64,
                                     device=device)

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, last_token_ids)

        input_names = (["inputs_embeds"] +
                       [f"past_key_values_{i}" for i in range(Na)] + [
                           "rope_rotary_cos_sin", "context_lengths",
                           "kvcache_start_index", "last_token_ids"
                       ])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)])

        batch = torch.export.Dim("batch", min=1, max=256)
        seq = torch.export.Dim("seq_len", min=1, max=32768)
        pos = torch.export.Dim("max_pos", min=1, max=32768)
        past = torch.export.Dim("past_len", min=1, max=32768)
        rope_batch = torch.export.Dim("rope_batch", min=1, max=256)
        kv_batch = torch.export.Dim("kv_batch", min=1, max=256)

        all_shapes: list = [{0: batch, 1: seq}]  # inputs_embeds
        for _ in range(Na):
            all_shapes.append({0: batch, 3: past})  # past_key_values_i
        all_shapes.append({0: rope_batch, 1: pos})  # rope_rotary_cos_sin
        all_shapes.append({0: batch})  # context_lengths
        all_shapes.append({0: kv_batch})  # kvcache_start_index
        all_shapes.append({0: batch})  # last_token_ids

        wrapped = _make_flat_wrapper(self, Na, Nd)
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
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
    ) -> Tuple:
        hidden_states, present_key_values = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
        )
        # Select hidden states for specified token positions before lm_head.
        hidden_states = torch.ops.trt.gather_nd(hidden_states, last_token_ids)

        logits = self.lm_head(hidden_states).to(torch.float32)
        return logits, present_key_values
