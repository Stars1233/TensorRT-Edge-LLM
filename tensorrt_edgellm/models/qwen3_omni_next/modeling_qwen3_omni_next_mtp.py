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
"""Qwen3-Omni Next MoE MTP draft model.

The Omni-Next thinker ships a single-layer MTP head (``thinker.mtp.*``,
BF16, in the quantization ignore list) whose decoder block is a
full-attention layer with the thinker's 256-expert sparse-MoE FFN
(routed experts + shared expert).  Everything except the FFN matches the
dense Qwen3.5 MTP draft, so this module only swaps the MLP for the
sparse-MoE block; the FP16 MoE plugin path handles the experts.
"""

import torch.nn as nn

from ...config import ModelConfig
from ..qwen3_5.modeling_qwen3_5_mtp import (Qwen3_5MtpDecoderLayer,
                                            Qwen3_5MtpDraftModel)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3SparseMoeBlock

__all__ = ["Qwen3OmniNextMoeMtpDraftModel", "Qwen3OmniNextMoeMtpDecoderLayer"]


class Qwen3OmniNextMoeMtpDecoderLayer(Qwen3_5MtpDecoderLayer):
    """MTP draft decoder block with the Omni-Next sparse-MoE FFN."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        # Replace the dense MLP with the sparse MoE block (routed experts +
        # shared expert), matching ``thinker.mtp.layers.0.mlp.*``.
        self.mlp = Qwen3SparseMoeBlock(config,
                                       module_prefix=f"layers.{layer_idx}.mlp")


class Qwen3OmniNextMoeMtpDraftModel(Qwen3_5MtpDraftModel):
    """Qwen3-Omni Next MoE MTP draft model (one full-attention MoE layer)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3OmniNextMoeMtpDecoderLayer(config, layer_idx=0)])
