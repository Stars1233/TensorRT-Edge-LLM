# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
NemotronH hybrid causal LM (Mamba2 SSM + MLP + Attention + MoE).

Checkpoint key structure
------------------------
backbone.embeddings.weight                     - token embedding
backbone.layers.{i}.norm.weight                - pre-mixer RMSNorm (all layer types)
backbone.layers.{i}.mixer.*                    - mixer (type depends on layer)
  Attention  : q_proj, k_proj, v_proj, o_proj
  Mamba2 SSM : in_proj, out_proj, conv1d.{weight,bias}, A_log, D, dt_bias, norm.weight
  MLP        : up_proj, down_proj
  MoE        : gate.{weight,e_score_correction_bias},
               experts.{j}.{up_proj,down_proj}, shared_experts.{up_proj,down_proj}
backbone.norm_f.weight                         - final RMSNorm
lm_head.weight                                 - output projection (FP16, tied or standalone)

Layer type pattern is read from ``hybrid_override_pattern`` in config.json:
  'M' -> LAYER_MAMBA   '*' -> LAYER_ATTN   '-' -> LAYER_MLP   'E' -> LAYER_MOE

Forward-pass conventions
------------------------
``NemotronHCausalLM.forward``:

    inputs_embeds        [batch, seq_len, hidden_size]              float16
    past_key_values      tuple of [batch, 2, num_kv_heads, past, head_dim] per attn-layer
    rope_rotary_cos_sin  [batch, max_pos, rotary_dim]               float32
    context_lengths      [batch]                                    int32
    kvcache_start_index  [batch]                                    int32
    last_token_ids       [batch, num_tokens]                        int64
    conv_states          tuple of [batch, conv_dim, conv_kernel-1] per mamba-layer
    ssm_states           tuple of [batch, num_heads, head_dim, ssm_state] per mamba-layer
    ──────────────────────────────────────────────────────────────────────
    -> logits             [batch, num_tokens, vocab_size]            float32
    -> present_key_values tuple of updated KV caches per attn-layer
    -> present_conv_states tuple of updated conv states per mamba-layer
    -> present_ssm_states  tuple of updated SSM states per mamba-layer
"""

import itertools
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import (LAYER_ATTN, LAYER_MAMBA, LAYER_MLP, LAYER_MOE,
                       MambaConfig, ModelConfig)
from ..default.modeling_default import OnnxSpec
from ..linear import FP16Linear, NVFP4Linear, make_linear
from ..ops import (attention_plugin, causal_conv1d, nvfp4_moe_plugin,
                   update_ssm_state)


class RMSNorm(nn.Module):
    """RMSNorm for hybrid Mamba models — primitive decomposed ops.

    Uses explicit Pow+ReduceMean+Rsqrt+Mul ops (same as the default/Qwen
    RMSNorm) so the ONNX graph contains only standard ops that every TRT
    version can parse.  The explicit FP32→FP16 cast at the end creates an
    ONNX partition boundary so TRT splits the ForeignNode cleanly at the
    Mamba plugin inputs (same technique used by ``_gated_rmsnorm`` below).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size,
                                              dtype=torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight.to(input_dtype) * hidden_states


# ---------------------------------------------------------------------------
# Mamba-aware flat wrapper (NemotronH-specific)
# ---------------------------------------------------------------------------


def _make_flat_wrapper_mamba(model: nn.Module, Na: int, Nm: int) -> nn.Module:
    """Build an explicit flat forward wrapper for hybrid Mamba+Attention models.

    Extends the transformer wrapper with ``conv_state_i`` and ``ssm_state_i``
    inputs and ``present_conv_state_i`` / ``present_recurrent_state_i`` outputs.

    Using ``*flat_args`` in ``forward`` triggers a PyTorch 2.10 bug where the
    guard code generated by the non-strict exporter accesses the variadic tuple
    by index during decomposition, causing an ``IndexError``.  An explicit,
    named parameter list avoids this entirely.

    ``Na`` = number of attention layers, ``Nm`` = number of Mamba layers.
    """
    param_names: List[str] = (["inputs_embeds"] +
                              [f"past_key_values_{i}" for i in range(Na)] + [
                                  "rope_rotary_cos_sin", "context_lengths",
                                  "kvcache_start_index", "last_token_ids"
                              ] + [f"conv_state_{i}" for i in range(Nm)] +
                              [f"recurrent_state_{i}" for i in range(Nm)])

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"
    conv_tuple = "({},)".format(", ".join(f"conv_state_{i}"
                                          for i in range(Nm))) if Nm else "()"
    ssm_tuple = "({},)".format(", ".join(f"recurrent_state_{i}"
                                         for i in range(Nm))) if Nm else "()"

    body = (
        f"    logits, present_key_values, present_conv_states, "
        f"present_ssm_states = self._model(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
        f"context_lengths, kvcache_start_index, last_token_ids,\n"
        f"        {conv_tuple}, {ssm_tuple})\n"
        f"    return ((logits,) + tuple(present_key_values)\n"
        f"            + tuple(present_conv_states) + tuple(present_ssm_states))\n"
    )

    src = "def _forward(self, {}):\n{}".format(", ".join(param_names), body)
    globs: dict = {}
    exec(src, globs)  # noqa: S102

    class _Wrapper(nn.Module):

        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self._model = m

    _Wrapper.forward = globs["_forward"]
    return _Wrapper(model)


__all__ = [
    "Conv1dBuffers",
    "MambaMixer",
    "NemotronHMLP",
    "NemotronHMoEMLP",
    "NemotronHTopkRouter",
    "NemotronHAttentionMixer",
    "NemotronHDecoderLayer",
    "NemotronHBackbone",
    "NemotronHCausalLM",
]

# ---------------------------------------------------------------------------
# Conv1dBuffers
# ---------------------------------------------------------------------------


class Conv1dBuffers(nn.Module):
    """Holds conv1d weight and bias as plain buffers (not quantized).

    Named ``conv1d`` inside :class:`MambaMixer` so checkpoint keys like
    ``backbone.layers.N.mixer.conv1d.weight`` resolve correctly.
    """

    def __init__(self, conv_dim: int, conv_kernel: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(conv_dim, 1, conv_kernel))
        self.register_buffer("bias", torch.zeros(conv_dim))


# ---------------------------------------------------------------------------
# MambaMixer
# ---------------------------------------------------------------------------


class MambaMixer(nn.Module):
    """Mamba2 SSM computation module.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.

    Buffers / parameters (names match checkpoint keys exactly):
        in_proj.weight           - input projection (quantised)
        out_proj.weight          - output projection (quantised)
        conv1d.weight            - [conv_dim, 1, conv_kernel]
        conv1d.bias              - [conv_dim]
        A_log                    - [num_heads]  float32
        D                        - [num_heads]  float32
        dt_bias                  - [num_heads]  float32
        norm.weight              - [intermediate_size] (gated RMSNorm)
    """

    def __init__(self, config: ModelConfig, mc: MambaConfig,
                 module_prefix: str) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        d_inner = mc.intermediate_size  # num_heads * head_dim

        # in_proj output: d_inner (gate) + conv_dim + num_heads (dt)
        in_proj_out = d_inner + mc.conv_dim + mc.num_heads
        self.in_proj = make_linear(config,
                                   hidden_size,
                                   in_proj_out,
                                   bias=False,
                                   module_name=f"{module_prefix}.in_proj")
        self.out_proj = make_linear(config,
                                    d_inner,
                                    hidden_size,
                                    bias=False,
                                    module_name=f"{module_prefix}.out_proj")

        self.conv1d = Conv1dBuffers(mc.conv_dim, mc.conv_kernel)

        # A_log must be FP16 so that .to(torch.float32) in forward() produces
        # an explicit Cast node — the Mamba plugin requires ssm_A as FP32.
        # (D and dt_bias are cast to FP16 in forward; register as FP16 directly.)
        self.register_buffer("A_log",
                             torch.zeros(mc.num_heads, dtype=torch.float16))
        self.register_buffer("D", torch.zeros(mc.num_heads,
                                              dtype=torch.float16))
        self.register_buffer("dt_bias",
                             torch.zeros(mc.num_heads, dtype=torch.float16))

        # Gated RMSNorm weight (stored as submodule named "norm")
        self.norm = RMSNorm(d_inner, eps=config.rms_norm_eps)

        self.num_heads = mc.num_heads
        self.head_dim = mc.head_dim
        self.n_groups = mc.n_groups
        self.ssm_state_size = mc.ssm_state_size
        self.conv_dim = mc.conv_dim
        self.conv_kernel = mc.conv_kernel
        self._group_size = (mc.num_heads * mc.head_dim) // mc.n_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        context_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        d_inner = self.num_heads * self.head_dim
        d_state = self.n_groups * self.ssm_state_size

        projected_states = self.in_proj(hidden_states)
        # Split: [gate, BC conv path, dt]
        gate, hidden_states_for_conv, dt = projected_states.split(
            [d_inner, self.conv_dim, self.num_heads], dim=-1)

        hidden_states_for_conv, conv_state_out = causal_conv1d(
            hidden_states_for_conv,
            self.conv1d.weight,
            self.conv1d.bias,
            conv_state,
            context_lengths,
            stride=1,
            padding=self.conv_kernel - 1,
            dilation=1,
            groups=self.conv_dim,
        )
        hidden_states_for_conv = F.silu(hidden_states_for_conv)

        ssm_input, ssm_b_flat, ssm_c_flat = hidden_states_for_conv.split(
            [d_inner, d_state, d_state], dim=-1)

        ssm_input_states = ssm_input.view(batch_size, seq_len, self.num_heads,
                                          self.head_dim)
        ssm_b_states = ssm_b_flat.view(batch_size, seq_len, self.n_groups,
                                       self.ssm_state_size)
        ssm_c_states = ssm_c_flat.view(batch_size, seq_len, self.n_groups,
                                       self.ssm_state_size)

        ssm_A = -torch.exp(self.A_log.to(torch.float32))

        ssm_output, ssm_state_out = update_ssm_state(
            ssm_input_states,
            ssm_A,
            ssm_b_states,
            ssm_c_states,
            self.D,
            dt,
            self.dt_bias,
            ssm_state,
            context_lengths,
            dt_softplus=1,
            ngroups=self.n_groups,
        )

        ssm_output = ssm_output.view(batch_size, seq_len, d_inner)
        normed = self._gated_rmsnorm(ssm_output, gate)
        return self.out_proj(normed), conv_state_out, ssm_state_out

    def _gated_rmsnorm(self, hidden_states: torch.Tensor,
                       gate: torch.Tensor) -> torch.Tensor:
        group_size = self._group_size
        gated = hidden_states * F.silu(gate)
        # Compute RMSNorm in FP32 explicitly, matching what TRT would do
        # internally (it forces ReduceMean to FP32).  The explicit
        # FP32→FP16 Cast at the end creates an ONNX partition boundary
        # so TRT can split the ForeignNode at the plugin inputs.
        gated_f32 = gated.to(torch.float32)
        gated_grouped = gated_f32.view(*gated_f32.shape[:-1], -1, group_size)
        variance = (gated_grouped * gated_grouped).mean(-1, keepdim=True)
        normed = gated_grouped * torch.rsqrt(variance +
                                             self.norm.variance_epsilon)
        normed = normed.view(*hidden_states.shape)
        return normed.to(torch.float16) * self.norm.weight


# ---------------------------------------------------------------------------
# NemotronHMLP  (relu² gated MLP)
# ---------------------------------------------------------------------------


class NemotronHMLP(nn.Module):
    """NemotronH MLP block: up_proj + relu² + down_proj.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.
    """

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        self.up_proj = make_linear(config,
                                   config.hidden_size,
                                   config.intermediate_size,
                                   module_name=f"{module_prefix}.up_proj")
        self.down_proj = make_linear(config,
                                     config.intermediate_size,
                                     config.hidden_size,
                                     module_name=f"{module_prefix}.down_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(hidden_states)
        r = F.relu(h)
        return self.down_proj(r * r)


# ---------------------------------------------------------------------------
# NemotronHTopkRouter
# ---------------------------------------------------------------------------


class NemotronHTopkRouter(nn.Module):
    """Sigmoid-based grouped top-k router for MoE layers.

    Submodule names match checkpoint keys:
        weight                  - [n_routed_experts, hidden_size] (FP32)
        e_score_correction_bias - [n_routed_experts] (FP32)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts,
                        config.hidden_size,
                        dtype=torch.float16))
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float16))

    def forward(
            self,
            hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(hidden_states, self.weight).float()
        scores = router_logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.float(
        ).unsqueeze(0)
        group_scores = (scores_for_choice.view(
            -1, self.n_group,
            self.n_routed_experts // self.n_group).topk(2,
                                                        dim=-1)[0].sum(dim=-1))
        group_idx = torch.topk(group_scores,
                               k=self.topk_group,
                               dim=-1,
                               sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (group_mask.unsqueeze(-1).expand(
            -1, self.n_group, self.n_routed_experts // self.n_group).reshape(
                -1, self.n_routed_experts))
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice,
                                  k=self.top_k,
                                  dim=-1,
                                  sorted=False)[1]

        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


# ---------------------------------------------------------------------------
# NemotronHMoEMLP
# ---------------------------------------------------------------------------


class NemotronHMoEMLP(nn.Module):
    """Mixture-of-Experts MLP for NemotronH using Nvfp4MoePlugin.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.

    The routed experts are handled by the ``trt_edgellm::Nvfp4MoePlugin``
    custom op which takes stacked NVFP4 weights + router logits and performs
    top-k routing + W4A16 GEMV internally.  The shared expert runs as a
    separate FP8/FP16 forward pass added to the plugin output.

    Submodule names match checkpoint keys:
        gate                     - NemotronHTopkRouter (weight + bias)
        experts.{j}.up_proj     - per-expert up projection (NVFP4)
        experts.{j}.down_proj   - per-expert down projection (NVFP4)
        shared_experts.up_proj  - shared expert up projection (FP8)
        shared_experts.down_proj - shared expert down projection (FP8)
    """

    def __init__(self, config: ModelConfig, module_prefix: str) -> None:
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.gate = NemotronHTopkRouter(config)

        self.experts = nn.ModuleList([
            self._make_expert(config, config.moe_intermediate_size,
                              f"{module_prefix}.experts.{j}")
            for j in range(config.n_routed_experts)
        ])

        self.shared_experts = self._make_expert(
            config, config.moe_shared_expert_intermediate_size,
            f"{module_prefix}.shared_experts")

        self._export_ready = False

    @staticmethod
    def _make_expert(config: ModelConfig, inter_size: int,
                     prefix: str) -> nn.Module:
        expert = nn.Module()
        expert.up_proj = make_linear(config,
                                     config.hidden_size,
                                     inter_size,
                                     module_name=f"{prefix}.up_proj")
        expert.down_proj = make_linear(config,
                                       inter_size,
                                       config.hidden_size,
                                       module_name=f"{prefix}.down_proj")
        return expert

    @staticmethod
    def _expert_forward(expert: nn.Module,
                        hidden_states: torch.Tensor) -> torch.Tensor:
        h = expert.up_proj(hidden_states)
        r = F.relu(h)
        return expert.down_proj(r * r)

    def prepare_for_export(self) -> None:
        """Pack ModelOpt NVFP4 expert tensors to plugin layout.

        ``Nvfp4MoePlugin`` expects N-major byte payloads + an atom-swizzled
        prefill SFB + a Marlin-FP16-top-8-bit-projected decode SFB. Stacked
        per-expert so TRT sees direct initializer→plugin edges.

        **All buffers are raw-aligned with vLLM**: the checkpoint's FP4 weight
        nibbles, FP8-E4M3 block scales, and scalar ``weight_scale_2`` are
        preserved bit-exact — only the on-device layouts change. Specifically:

        * ``_stacked_*_weights`` — FP4 nibble transpose (K-major→N-major).
        * ``_stacked_*_block_scale`` — checkpoint FP8 bytes atom-swizzled (128×4).
        * ``_stacked_*_block_scale_decode`` — checkpoint FP8 bytes Marlin-
          projected (top-8-bits of FP16 representation).
        * ``_stacked_*_global_scale`` — per-expert raw ``weight_scale_2``.

        With this, both paths use the **same** per-expert global scale and the
        FP4 × SFB × global_scale product recovers the checkpoint dense
        bit-exact in prefill and approximately (Marlin-projection loss only)
        in decode.

        **Activation global scale** (``_hidden_act_input_scale``) follows the
        plugin's forward-direction convention (``fp4Quantize.cu:212``: *callers
        pass the forward global SF (e.g. max|x|/(448*6)); the kernel computes
        its reciprocal internally*). The checkpoint's ``NVFP4Linear.input_scale``
        already equals ``amax / (6 * 448)`` (see ``models/linear.py:185``), so we
        pass the per-layer max raw — no ``/6`` adjustment. This matches both
        :meth:`Nvfp4MoePlugin.populate_hidden_global_scales` and vLLM's
        ``input_global_scale`` handling.
        """
        from ...checkpoint.repacking import (
            repack_nvfp4_expert_down_prefill_raw,
            repack_nvfp4_expert_up_prefill_raw)

        H = self.hidden_size
        I = self.moe_intermediate_size

        up_ws, up_scs, up_scs_dec, up_gs = [], [], [], []
        dn_ws, dn_scs, dn_scs_dec, dn_gs = [], [], [], []
        for expert in self.experts:
            up = expert.up_proj
            dn = expert.down_proj
            assert isinstance(up, NVFP4Linear) and isinstance(
                dn, NVFP4Linear), "MoE experts must be NVFP4Linear"

            up_w, up_sc, up_sc_dec, up_gl = repack_nvfp4_expert_up_prefill_raw(
                up.weight, up.weight_scale, up.weight_scale_2, H, I)
            if up_w.shape != (H, I // 2):
                raise ValueError(
                    f"up weight shape {up_w.shape} != ({H}, {I // 2})")
            dn_w, dn_sc, dn_sc_dec, dn_gl = (
                repack_nvfp4_expert_down_prefill_raw(dn.weight,
                                                     dn.weight_scale,
                                                     dn.weight_scale_2, H, I))
            if dn_w.shape != (I, H // 2):
                raise ValueError(
                    f"down weight shape {dn_w.shape} != ({I}, {H // 2})")

            up_ws.append(torch.as_tensor(up_w))
            up_scs.append(torch.as_tensor(up_sc))
            up_scs_dec.append(torch.as_tensor(up_sc_dec))
            up_gs.append(up_gl)
            dn_ws.append(torch.as_tensor(dn_w))
            dn_scs.append(torch.as_tensor(dn_sc))
            dn_scs_dec.append(torch.as_tensor(dn_sc_dec))
            dn_gs.append(dn_gl)

        self.register_buffer("_stacked_up_weights", torch.stack(up_ws))
        self.register_buffer("_stacked_up_block_scale", torch.stack(up_scs))
        self.register_buffer("_stacked_up_block_scale_decode",
                             torch.stack(up_scs_dec))
        self.register_buffer("_stacked_up_global_scale",
                             torch.tensor(up_gs, dtype=torch.float32))
        self.register_buffer("_stacked_down_weights", torch.stack(dn_ws))
        self.register_buffer("_stacked_down_block_scale", torch.stack(dn_scs))
        self.register_buffer("_stacked_down_block_scale_decode",
                             torch.stack(dn_scs_dec))
        self.register_buffer("_stacked_down_global_scale",
                             torch.tensor(dn_gs, dtype=torch.float32))
        self.register_buffer(
            "_e_score_correction_bias_fp32",
            self.gate.e_score_correction_bias.data.clone().to(torch.float32))
        # Forward-direction convention — see method docstring.
        fc1_scale = max(
            float(e.up_proj.input_scale.detach().float().max().item())
            for e in self.experts)
        fc2_scale = max(
            float(e.down_proj.input_scale.detach().float().max().item())
            for e in self.experts)
        self.register_buffer(
            "_hidden_act_input_scale",
            torch.tensor([fc1_scale, fc2_scale], dtype=torch.float32))
        self._export_ready = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        # Router logits: [batch*seq, E] FP32 (plugin expects FP32 input)
        router_logits = F.linear(hidden_states.view(-1, self.hidden_size),
                                 self.gate.weight).float()

        moe_out = nvfp4_moe_plugin(
            router_logits,
            hidden_states,
            self._hidden_act_input_scale,
            self._stacked_up_weights,
            self._stacked_up_block_scale,
            self._stacked_up_global_scale,
            self._stacked_down_weights,
            self._stacked_down_block_scale,
            self._stacked_down_global_scale,
            self._stacked_up_block_scale_decode,
            self._stacked_down_block_scale_decode,
            self._e_score_correction_bias_fp32,
            num_experts=self.n_routed_experts,
            top_k=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            moe_inter_size=self.moe_intermediate_size,
            activation_type=0,  # 0 = ReLU²
            n_group=self.gate.n_group,
            topk_group=self.gate.topk_group,
            norm_topk_prob=int(bool(self.gate.norm_topk_prob)),
            routed_scaling_factor=float(self.gate.routed_scaling_factor),
            # NemotronH always drives the sigmoid + grouped top-k routing kernel (routing_mode=1).
            routing_mode=1,
        )

        return moe_out + self._expert_forward(self.shared_experts,
                                              hidden_states)


# ---------------------------------------------------------------------------
# NemotronHAttentionMixer
# ---------------------------------------------------------------------------


class NemotronHAttentionMixer(nn.Module):
    """GQA attention for NemotronH.

    Named ``mixer`` inside :class:`NemotronHDecoderLayer` to match checkpoint
    key prefix ``backbone.layers.N.mixer.*``.

    Submodule names match checkpoint keys:
        q_proj, k_proj, v_proj, o_proj
    """

    def __init__(self, config: ModelConfig, layer_idx: int,
                 module_prefix: str) -> None:
        super().__init__()
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden_size = config.hidden_size

        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.enable_fp8_kv_cache = config.quant.kv_cache_quant == "fp8"
        self.sliding_window_size = -1

        self.q_proj = make_linear(config,
                                  hidden_size,
                                  num_attention_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.q_proj")
        self.k_proj = make_linear(config,
                                  hidden_size,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.k_proj")
        self.v_proj = make_linear(config,
                                  hidden_size,
                                  num_key_value_heads * head_dim,
                                  bias=config.attention_bias,
                                  module_name=f"{module_prefix}.v_proj")
        if self.enable_fp8_kv_cache:
            self.k_proj.register_buffer("k_scale", torch.ones(1))
            self.v_proj.register_buffer("v_scale", torch.ones(1))

        self.o_proj = make_linear(config,
                                  num_attention_heads * head_dim,
                                  hidden_size,
                                  module_name=f"{module_prefix}.o_proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        kwargs: dict = {
            "num_q_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "sliding_window_size": self.sliding_window_size,
            "enable_tree_attention": False,
            "enable_fp8_kv_cache": self.enable_fp8_kv_cache,
        }
        # Always pass qkv_scales so torch.export includes a valid FLOATS
        # value in the FX graph for the unified ONNX translation.
        kwargs["qkv_scales"] = getattr(self, "_qkv_scales_float",
                                       [1.0, 1.0, 1.0])
        attn_output, present_key_value = attention_plugin(
            query_states, key_states, value_states, past_key_value,
            context_lengths, rope_rotary_cos_sin, kvcache_start_index,
            **kwargs)

        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_key_value


# ---------------------------------------------------------------------------
# NemotronHDecoderLayer
# ---------------------------------------------------------------------------


class NemotronHDecoderLayer(nn.Module):
    """Single NemotronH decoder layer: pre-norm + mixer.

    Submodule names match checkpoint keys:
        norm    - RMSNorm (pre-mixer)
        mixer   - MambaMixer | NemotronHMLP | NemotronHAttentionMixer
    """

    def __init__(self, config: ModelConfig, layer_idx: int,
                 layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        module_prefix = f"backbone.layers.{layer_idx}.mixer"

        if layer_type == LAYER_MAMBA:
            assert config.mamba_cfg is not None
            self.mixer = MambaMixer(config, config.mamba_cfg, module_prefix)
        elif layer_type == LAYER_MLP:
            self.mixer = NemotronHMLP(config, module_prefix)
        elif layer_type == LAYER_MOE:
            self.mixer = NemotronHMoEMLP(config, module_prefix)
        elif layer_type == LAYER_ATTN:
            self.mixer = NemotronHAttentionMixer(config, layer_idx,
                                                 module_prefix)
        else:
            raise ValueError(f"Unknown layer type: {layer_type!r}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        # Attention-specific (ignored by Mamba/MLP/MoE layers)
        past_key_value: Optional[torch.Tensor] = None,
        rope_rotary_cos_sin: Optional[torch.Tensor] = None,
        context_lengths: Optional[torch.Tensor] = None,
        kvcache_start_index: Optional[torch.Tensor] = None,
        # Mamba-specific (ignored by Attention/MLP/MoE layers)
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        normed = self.norm(hidden_states)
        if self.layer_type == LAYER_MAMBA:
            mixer_out, conv_state_out, ssm_state_out = self.mixer(
                normed, conv_state, ssm_state, context_lengths)
            return residual + mixer_out, conv_state_out, ssm_state_out
        elif self.layer_type in (LAYER_MLP, LAYER_MOE):
            return residual + self.mixer(normed)
        else:
            attn_out, present_kv = self.mixer(normed, past_key_value,
                                              rope_rotary_cos_sin,
                                              context_lengths,
                                              kvcache_start_index)
            return residual + attn_out, present_kv


# ---------------------------------------------------------------------------
# NemotronHBackbone
# ---------------------------------------------------------------------------


class NemotronHBackbone(nn.Module):
    """NemotronH transformer backbone.

    Submodule names match checkpoint keys:
        embeddings   - token embedding (backbone.embeddings.weight)
        layers       - decoder layer list
        norm_f       - final RMSNorm (backbone.norm_f.weight)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            NemotronHDecoderLayer(config, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])
        self.norm_f = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_types: List[str] = config.layer_types

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        conv_states: Tuple[torch.Tensor, ...] = (),
        ssm_states: Tuple[torch.Tensor, ...] = (),
    ) -> Tuple[torch.Tensor, Tuple, Tuple, Tuple]:
        hidden_states = inputs_embeds
        present_key_values_list: List[torch.Tensor] = []
        present_conv_states_list: List[torch.Tensor] = []
        present_ssm_states_list: List[torch.Tensor] = []
        attn_idx = 0
        mamba_idx = 0

        for layer_idx, (layer,
                        lt) in enumerate(zip(self.layers, self.layer_types)):
            if lt == LAYER_MAMBA:
                hidden_states, conv_out, ssm_out = layer(
                    hidden_states,
                    context_lengths=context_lengths,
                    conv_state=conv_states[mamba_idx],
                    ssm_state=ssm_states[mamba_idx],
                )
                present_conv_states_list.append(conv_out)
                present_ssm_states_list.append(ssm_out)
                mamba_idx += 1
            elif lt in (LAYER_MLP, LAYER_MOE):
                hidden_states = layer(hidden_states)
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    past_key_value=past_key_values[attn_idx],
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    context_lengths=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                )
                present_key_values_list.append(present_kv)
                attn_idx += 1

        return (self.norm_f(hidden_states), tuple(present_key_values_list),
                tuple(present_conv_states_list),
                tuple(present_ssm_states_list))


# ---------------------------------------------------------------------------
# NemotronHCausalLM
# ---------------------------------------------------------------------------

_BATCH_SIZE = 1
_SEQ_LEN = 1
_PAST_LEN = 1
_MAX_POS = 4096


class NemotronHCausalLM(nn.Module):
    """NemotronH causal LM: backbone + lm_head.

    The inner backbone is stored as attribute ``backbone`` so parameter keys
    carry the ``backbone.`` prefix matching checkpoint key prefixes.
    ``lm_head`` maps directly to ``lm_head.weight``.
    """

    # Dtypes of the Mamba state tensors this model feeds the ONNX graph.
    # These drive (a) the dummy tensor dtypes in ``export_onnx`` and
    # (b) the ``recurrent_state_dtype`` / ``conv_state_dtype`` strings written
    # into ``config.json`` (see checkpoint_utils). They must stay in sync, so
    # the single source of truth is this class attribute, not a separate table.
    # The dtype is dictated by the ``trt_edgellm::update_ssm_state`` plugin
    # schema: ``state`` has type ``T`` where we pick float16.
    RECURRENT_STATE_DTYPE = torch.float16
    CONV_STATE_DTYPE = torch.float16

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = NemotronHBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")

    def tie_weights(self) -> None:
        """Clone embeddings.weight into lm_head.weight when tie_word_embeddings=True."""
        if not self.config.tie_word_embeddings:
            return
        if not isinstance(self.lm_head, FP16Linear):
            return
        embed_weight = self.backbone.embeddings.weight
        self.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                           requires_grad=False)

    def onnx_export_spec(self) -> OnnxSpec:
        """Return all model-specific parameters needed for ONNX export."""
        # Pre-process MoE layers: reinterpret FP8 scales as INT8
        for layer in self.backbone.layers:
            if hasattr(layer.mixer, 'prepare_for_export'):
                layer.mixer.prepare_for_export()
        config = self.config
        mc = config.mamba_cfg
        Na = config.num_attn_layers
        Nm = config.num_mamba_layers
        Nd = 0  # NemotronH has no deepstack visual embeddings
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
        last_token_ids = torch.zeros(batch_size,
                                     1,
                                     dtype=torch.int64,
                                     device=device)

        assert mc is not None, "NemotronHCausalLM requires mamba_cfg"
        conv_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        mc.conv_dim,
                        mc.conv_kernel,
                        dtype=self.CONV_STATE_DTYPE,
                        device=device) for _ in range(Nm)
        ]
        ssm_states: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        mc.num_heads,
                        mc.head_dim,
                        mc.ssm_state_size,
                        dtype=self.RECURRENT_STATE_DTYPE,
                        device=device) for _ in range(Nm)
        ]

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, last_token_ids,
                *conv_states, *ssm_states)

        input_names = (["inputs_embeds"] +
                       [f"past_key_values_{i}" for i in range(Na)] + [
                           "rope_rotary_cos_sin", "context_lengths",
                           "kvcache_start_index", "last_token_ids"
                       ] + [f"conv_state_{i}" for i in range(Nm)] +
                       [f"recurrent_state_{i}" for i in range(Nm)])
        output_names = (["logits"] +
                        [f"present_key_values_{i}" for i in range(Na)] +
                        [f"present_conv_state_{i}" for i in range(Nm)] +
                        [f"present_recurrent_state_{i}" for i in range(Nm)])

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
        for _ in range(Nm):
            all_shapes.append({0: batch})  # conv_state_i
        for _ in range(Nm):
            all_shapes.append({0: batch})  # recurrent_state_i

        wrapped = _make_flat_wrapper_mamba(self, Na, Nm)
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
            conv_states: Tuple[torch.Tensor, ...] = (),
            ssm_states: Tuple[torch.Tensor, ...] = (),
    ) -> Tuple:
        (hidden_states, present_key_values, present_conv_states,
         present_ssm_states) = self.backbone(
             inputs_embeds,
             past_key_values,
             rope_rotary_cos_sin,
             context_lengths,
             kvcache_start_index,
             conv_states,
             ssm_states,
         )
        # Select hidden states for specified token positions before lm_head.
        hidden_states = torch.ops.trt.gather_nd(hidden_states, last_token_ids)

        logits = self.lm_head(hidden_states).to(torch.float32)
        return logits, present_key_values, present_conv_states, present_ssm_states
