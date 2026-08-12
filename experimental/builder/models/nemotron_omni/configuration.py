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
"""Nemotron-Omni checkpoint configuration."""

from ...core import contracts

_PATTERN_TYPES = {
    "M": "mamba",
    "-": "mlp",
    "*": "attention",
    "E": "moe",
}


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or root
    if component == contracts.Component.AUDIO:
        return root.get("audio_config") or root.get("sound_config") or root
    raise ValueError(f"Nemotron-Omni has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    """Normalize the embedded Nemotron-H decoder configuration."""
    config = dict(config)
    config["rotary_dim_override"] = int(
        config.get("head_dim",
                   config["hidden_size"] // config["num_attention_heads"]))
    config["hybrid_uses_rope"] = False
    raw_types = config.get("layers_block_type") or config.get("layer_types")
    if raw_types:
        config["num_hidden_layers"] = len(raw_types)
        config["layer_types"] = [
            "mamba"
            if str(layer_type).lower() == "linear_attention" else layer_type
            for layer_type in raw_types
        ]
    elif config.get("hybrid_override_pattern"):
        config["layer_types"] = [
            _PATTERN_TYPES[token]
            for token in config["hybrid_override_pattern"]
            if token in _PATTERN_TYPES
        ]
    else:
        raise ValueError("Nemotron Omni requires explicit decoder layer types")
    return config
