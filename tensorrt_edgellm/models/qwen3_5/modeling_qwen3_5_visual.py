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
Qwen3.5 visual encoder.

Same architecture as Qwen3-VL (from the ``qwen3_vl`` module) but without
deepstack intermediate features. Reuses Qwen3-VL classes directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..qwen3_vl.modeling_qwen3_vl_visual import (Qwen3VLVisualModel,
                                                 _load_weights)

if TYPE_CHECKING:
    from ...config import ModelConfig


class Qwen3_5VLVisualModel(Qwen3VLVisualModel):
    """Qwen3.5 visual encoder — same arch as Qwen3-VL, no deepstack features."""

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        # Force deepstack disabled for Qwen3.5
        config = {**config, "deepstack_visual_indexes": []}
        super().__init__(config, model_config=model_config)


def build_qwen3_5_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> Qwen3_5VLVisualModel:
    """Instantiate and load weights for a Qwen3.5 visual encoder.

    Qwen3.5 shares the Qwen3-VL architecture but without deepstack features.

    Args:
        config:       Parsed ``vision_config`` dict from ``config.json``.
        weights:      Safetensors weight dict (all model weights).
        model_config: Top-level ``ModelConfig`` for quantized Linear dispatch.
        dtype:        Target dtype (default float16).
    """
    config = {**config, "deepstack_visual_indexes": []}
    model = Qwen3_5VLVisualModel(config, model_config=model_config)
    model.to(dtype)
    _load_weights(model, weights, prefix="model.visual")
    model.eval()
    return model


__all__ = [
    "Qwen3_5VLVisualModel",
    "build_qwen3_5_visual",
]
