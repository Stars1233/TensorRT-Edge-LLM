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
Qwen3-TTS CodePredictor — residual codec token predictor.

The CodePredictor is a small 5-layer Qwen3 decoder that generates residual
audio codes to complement the coarse code from the Talker.  It has 15 separate
lm_heads (one per residual codebook layer) and 15 codec embedding tables.

Key differences from a standard CausalLM:
- ``lm_head_weight`` is an ONNX **input** tensor (not a fixed weight) to enable
  dynamic lm_head selection at runtime via 15 CUDA Graphs.
- The forward pass returns both ``logits`` and ``hidden_states`` (for residual
  connection in the multi-token prediction loop).
- ``embed_tokens`` is a ModuleList of 15 codec embeddings (embedding lookup
  happens at runtime, not in the ONNX graph — the model takes ``inputs_embeds``).

Extra weight files extracted alongside the ONNX:
- ``codec_embeddings.safetensors`` — 15 embedding tables [codebookSize, hiddenSize]
- ``lm_heads.safetensors`` — 15 lm_head weights [codebookSize, hiddenSize]
- ``small_to_mtp_projection.safetensors`` — Linear [cpHiddenSize, talkerHiddenSize]

Checkpoint prefix: ``talker.code_predictor.*``
"""

import itertools
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..default.modeling_default import (_BATCH_SIZE, _MAX_POS, _PAST_LEN,
                                        _SEQ_LEN, CausalLM, OnnxSpec)

logger = logging.getLogger(__name__)

__all__ = [
    "CodePredictorCausalLM",
    "apply_code_predictor_mlp_war",
]

# ---------------------------------------------------------------------------
# MLP FP16 overflow WAR
# ---------------------------------------------------------------------------


class _DownProjFP32(nn.Module):
    """FP32-weight linear for CodePredictor's ``down_proj``.

    The reference ``tensorrt_edgellm`` CP ONNX stores the five down_proj
    weights as **FP32** initializers, so that the full matmul happens in
    FP32 — necessary because CP's silu*up intermediate reaches [-39, 72]
    which loses precision when cast back to FP16.

    Replicate that here by pre-casting the FP16 weight to an FP32
    ``nn.Parameter`` at WAR-apply time, so ONNX export bakes the FP32
    values directly into the initializer (no in-graph Cast, no lossy
    FP16 roundtrip).
    """

    def __init__(self, original_down_proj: nn.Module) -> None:
        super().__init__()
        w = original_down_proj.weight.detach().to(torch.float32)
        self.weight = nn.Parameter(w, requires_grad=False)
        b = getattr(original_down_proj, "bias", None)
        if b is not None:
            self.bias = nn.Parameter(b.detach().to(torch.float32),
                                     requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, intermediate_fp32: torch.Tensor) -> torch.Tensor:
        return F.linear(intermediate_fp32, self.weight, self.bias)


class _MLPFloat32WAR(nn.Module):
    """MLP wrapper that runs the full ``silu(gate) * up → down_proj`` path
    in FP32 to prevent FP16 overflow.

    The CodePredictor's intermediate activations can reach [-39.5, 72.4]
    in later layers, which overflows FP16 in the ``silu(gate) * up``
    multiply and loses precision in the subsequent ``down_proj``.  We
    cast gate/up outputs to FP32 for the multiply and keep the down_proj
    matmul in FP32 (see :class:`_DownProjFP32`).  The final result is
    cast back to the input dtype for the layer residual.
    """

    def __init__(self, original_mlp: nn.Module) -> None:
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = _DownProjFP32(original_mlp.down_proj)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_proj(hidden_states).to(torch.float32)
        up_output = self.up_proj(hidden_states).to(torch.float32)
        intermediate = F.silu(gate_output) * up_output
        result = self.down_proj(intermediate)
        return result.to(hidden_states.dtype)


def apply_code_predictor_mlp_war(model: nn.Module) -> None:
    """Replace MLP modules in all layers with FP32 overflow WAR."""
    transformer = getattr(model, "model", model)
    layers = getattr(transformer, "layers", [])
    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp"):
            layer.mlp = _MLPFloat32WAR(layer.mlp)
    logger.info("Applied MLP FP32 WAR to %d CodePredictor layers", len(layers))


# ---------------------------------------------------------------------------
# CodePredictor flat wrapper for ONNX export
# ---------------------------------------------------------------------------


def _make_code_predictor_flat_wrapper(model: nn.Module, Na: int) -> nn.Module:
    """Build a flat-signature wrapper for CodePredictor ONNX export.

    Unlike the standard CausalLM wrapper, this includes:
    - ``lm_head_weight`` as an input (dynamic lm_head for 15 CUDA graphs)
    - ``hidden_states`` as an output (for residual connection)
    """
    param_names: List[str] = (
        ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
            "rope_rotary_cos_sin", "context_lengths", "kvcache_start_index",
            "last_token_ids", "lm_head_weight"
        ])

    past_kv_tuple = "({},)".format(", ".join(
        f"past_key_values_{i}" for i in range(Na))) if Na else "()"

    body = (
        f"    logits, hidden_states, present_key_values = self._model(\n"
        f"        inputs_embeds, {past_kv_tuple}, rope_rotary_cos_sin, "
        f"context_lengths, kvcache_start_index, last_token_ids, "
        f"lm_head_weight)\n"
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


# ---------------------------------------------------------------------------
# CodePredictorCausalLM
# ---------------------------------------------------------------------------


class CodePredictorCausalLM(CausalLM):
    """CodePredictor variant of CausalLM with dynamic lm_head and hidden_states output.

    The lm_head is NOT a fixed weight — it is passed as an input tensor at
    each inference step (one of 15 different heads selected by the runtime).
    The forward pass also returns hidden_states for residual connections.
    """

    # CodePredictor's ``down_proj`` is wired as FP32 (see :class:`_DownProjFP32`)
    # because silu*up intermediates reach ~[-39, 72] and lose precision if
    # cast back to FP16.  Opt the exported initializer names out of the
    # generic FP32→FP16 downgrade in ``_fix_initializer_dtypes`` — without
    # this, the FP16 round-trip silently re-appears and breaks codec-EOS.
    preserve_fp32_initializer_patterns = ("down_proj.weight", )
    match_fp32_matmul_initializers = True

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Tuple:
        hidden_states, present_key_values, _ = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
        )
        # Select last token hidden states via GatherND
        last_hidden = torch.ops.trt.gather_nd(hidden_states, last_token_ids)
        # Dynamic lm_head: logits = last_hidden @ lm_head_weight.T
        logits = torch.matmul(last_hidden, lm_head_weight.T).to(torch.float32)
        return logits, hidden_states, present_key_values

    def onnx_export_spec(self) -> OnnxSpec:
        """ONNX export spec with lm_head_weight input and hidden_states output."""
        config = self.config
        Na = config.num_hidden_layers
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
        past_key_values_list: List[torch.Tensor] = [
            torch.zeros(batch_size,
                        2,
                        config.num_key_value_heads,
                        past_len,
                        config.head_dim,
                        dtype=dtype16,
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
        lm_head_weight = torch.zeros(config.vocab_size,
                                     config.hidden_size,
                                     dtype=dtype16,
                                     device=device)

        args = (inputs_embeds, *past_key_values_list, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, last_token_ids,
                lm_head_weight)

        input_names = (
            ["inputs_embeds"] + [f"past_key_values_{i}" for i in range(Na)] + [
                "rope_rotary_cos_sin", "context_lengths",
                "kvcache_start_index", "last_token_ids", "lm_head_weight"
            ])
        output_names = (["logits", "hidden_states"] +
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
        all_shapes.append({})  # lm_head_weight (fixed shape)

        wrapped = _make_code_predictor_flat_wrapper(self, Na)
        wrapped.eval()

        return OnnxSpec(wrapped=wrapped,
                        args=args,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_shapes=all_shapes)
