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
"""Qwen3.5 MoE MTP draft model (ONNX export)."""

import torch.nn as nn

from ...config import ModelConfig
from ..qwen3_5.modeling_qwen3_5_mtp import (Qwen3_5MtpDecoderLayer,
                                            Qwen3_5MtpDraftModel)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3SparseMoeBlock

__all__ = ["Qwen3_5MoeMtpDraftModel", "Qwen3_5MoeMtpDecoderLayer"]


class Qwen3_5MoeMtpDecoderLayer(Qwen3_5MtpDecoderLayer):
    """Qwen3.5 MoE MTP decoder layer: GatedAttention + sparse MoE FFN."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        # Replace dense MLP with sparse MoE block.
        self.mlp = Qwen3SparseMoeBlock(config,
                                       module_prefix=f"layers.{layer_idx}.mlp")


class Qwen3_5MoeMtpDraftModel(Qwen3_5MtpDraftModel):
    """Qwen3.5 MoE MTP draft model."""

    def _make_decoder_layer(self, config: ModelConfig) -> nn.Module:
        return Qwen3_5MoeMtpDecoderLayer(config, layer_idx=0)

    # forward() and onnx_export_spec() are inherited from Qwen3_5MtpDraftModel
    # unchanged — the MoE block is transparent to the outer control flow.
