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
"""
Qwen3.5 MoE hybrid causal LM.

This model combines the Qwen3.5 mixer stack (GatedDeltaNet linear attention
and gated full attention) with a sparse MoE feed-forward block in every layer.
The FFN (routed experts via the MoE plugin path plus the FP16 shared expert
gated by ``mlp.shared_expert_gate``) is handled entirely by the shared
``Qwen3SparseMoeBlock``.
"""

import torch.nn as nn

from ...config import GdnConfig, ModelConfig
from ..linear import make_linear
from ..qwen3_5.modeling_qwen3_5_text import (Qwen3_5Backbone, Qwen3_5CausalLM,
                                             Qwen3_5DecoderLayer,
                                             Qwen3_5RMSNorm)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3SparseMoeBlock

__all__ = ["Qwen3_5MoeCausalLM"]


class Qwen3_5MoeDecoderLayer(Qwen3_5DecoderLayer):
    """Qwen3.5 decoder layer with GDN/full-attention mixer and MoE FFN."""

    def __init__(self, config: ModelConfig, gc: GdnConfig, layer_idx: int,
                 layer_type: str) -> None:
        super().__init__(config, gc, layer_idx, layer_type)
        self.mlp = Qwen3SparseMoeBlock(config,
                                       module_prefix=f"layers.{layer_idx}.mlp")


class Qwen3_5MoeBackbone(Qwen3_5Backbone):
    """Qwen3.5 hybrid decoder backbone with sparse MoE FFNs."""

    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        gc = config.gdn_cfg
        assert gc is not None, "Qwen3.5 MoE requires gdn_cfg"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoderLayer(config, gc, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_types = config.layer_types


class Qwen3_5MoeCausalLM(Qwen3_5CausalLM):
    """Qwen3.5 MoE causal LM: hybrid backbone + lm_head."""

    def __init__(self, config: ModelConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen3_5MoeBackbone(config)
        self.lm_head = make_linear(config,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False,
                                   module_name="lm_head")
