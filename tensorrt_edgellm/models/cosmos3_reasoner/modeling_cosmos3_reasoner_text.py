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
"""Cosmos3-Edge reasoner language model.

The reasoner is a standard mRoPE dense VLM decoder — the SAME architecture as
Edge-LLM's shared ``default.CausalLM`` (GQA attention, RMSNorm, mRoPE supplied
as a graph input, KV-cache autoregressive decode, separate ``lm_head``) — with
ONE departure: the per-layer feed-forward is Nemotron-H **squared-ReLU,
non-gated** ``down_proj(relu(up_proj(x))**2)`` instead of SwiGLU. Its config sets
``qk_norm_for_text=False`` (no q/k norm on text self-attention), ``head_dim=128``
GQA 16/8, ``rope_theta=1e8`` and interleaved ``mrope_section=[24, 20, 20]``.

The Nemotron-H checkpoint stores the tower as 56 alternating attention/MLP
blocks; the 28-layer (attn+MLP per layer) view here is numerically identical
(each ``DecoderLayer`` = one attention block + one MLP block, one norm each).
Key remapping is handled in :mod:`.weights`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..default.modeling_default import (CausalLM, DecoderLayer, ModelConfig,
                                        Transformer)
from ..linear import TPMode, make_linear


class Relu2MLP(nn.Module):
    """Nemotron-H non-gated squared-ReLU MLP: ``down(relu(up(x))**2)``.

    Submodule names (``up_proj`` / ``down_proj``) match the checkpoint keys
    after remapping, and the linears use the same ``make_linear`` machinery as
    the rest of the stack so tensor-parallel and quantized builds keep working.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = -1) -> None:
        super().__init__()
        prefix = f"layers.{layer_idx}.mlp" if layer_idx >= 0 else ""
        self.up_proj = make_linear(
            config,
            config.hidden_size,
            config.intermediate_size,
            module_name=f"{prefix}.up_proj" if prefix else "",
            tp_mode=TPMode.COL)
        self.down_proj = make_linear(
            config,
            config.intermediate_size,
            config.hidden_size,
            module_name=f"{prefix}.down_proj" if prefix else "",
            tp_mode=TPMode.ROW)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.up_proj(hidden_states))
        return self.down_proj(h * h)


class Cosmos3ReasonerDecoderLayer(DecoderLayer):
    """Shared decoder layer with the Nemotron-H relu² MLP."""

    mlp_cls = Relu2MLP


class Cosmos3ReasonerTransformer(Transformer):
    """Shared transformer stack built from :class:`Cosmos3ReasonerDecoderLayer`."""

    decoder_layer_cls = Cosmos3ReasonerDecoderLayer


class Cosmos3ReasonerCausalLM(CausalLM):
    """Cosmos3-Edge reasoner LM = the shared dense decoder with a relu² MLP."""

    transformer_cls = Cosmos3ReasonerTransformer


def build_cosmos3_reasoner_text(
        config: ModelConfig) -> Cosmos3ReasonerCausalLM:
    """Construct the Cosmos3-Edge reasoner language model."""
    model = Cosmos3ReasonerCausalLM(config)
    model.eval()
    return model
