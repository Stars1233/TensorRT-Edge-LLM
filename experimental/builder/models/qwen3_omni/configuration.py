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
"""Qwen3-Omni checkpoint and component configuration."""

import json
import os
from dataclasses import replace

from ...core import contracts, quantization, weight_policy

_TALKER_TYPES = frozenset((
    "qwen3_omni_talker",
    "qwen3_omni_moe_talker",
))
_PREDICTOR_TYPES = frozenset((
    "qwen3_omni_talker_code_predictor",
    "qwen3_omni_moe_talker_code_predictor",
))


def prepare_root(model_dir: str, root: dict) -> dict:
    root = dict(root)
    speech_config = os.path.join(model_dir, "speech_tokenizer", "config.json")
    if os.path.isfile(speech_config):
        with open(speech_config) as config_file:
            root["_speech_tokenizer_config"] = json.load(config_file)
    return root


def component_config(root: dict, component: contracts.Component) -> dict:
    root_type = str(root.get("model_type", ""))
    thinker = root.get("thinker_config") or {}
    if component == contracts.Component.LLM:
        return thinker or root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or thinker.get(
            "vision_config") or root
    if component == contracts.Component.AUDIO:
        return root.get("audio_config") or thinker.get("audio_config") or root
    if component == contracts.Component.TALKER:
        talker = root.get("talker_config")
        if isinstance(talker, dict):
            return talker
        if root_type in _TALKER_TYPES:
            return root
        raise ValueError("Qwen3-Omni checkpoint has no talker_config")
    if component == contracts.Component.CODE_PREDICTOR:
        talker = root.get("talker_config") or root
        predictor = talker.get("code_predictor_config")
        if isinstance(predictor, dict):
            return predictor
        if root_type in _PREDICTOR_TYPES:
            return root
        raise ValueError("Qwen3-Omni checkpoint has no code_predictor_config")
    if component == contracts.Component.CODE2WAV:
        talker = root.get("talker_config") or {}
        speech = root.get("_speech_tokenizer_config") or {}
        return (root.get("code2wav_config") or talker.get("code2wav_config")
                or speech.get("decoder_config") or root)
    raise ValueError(f"Qwen3-Omni has no {component.value} configuration")


def component_weight_policy(args, policy):
    """Keep dynamic Code2Wav dense weights out of TensorRT's Myelin path."""
    if args.resolved_component != contracts.Component.CODE2WAV:
        return policy
    return policy.without((weight_policy.EXTERNAL_WEIGHT_FP16, ),
                          strict=bool(args.externalize_weights))


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    if component == contracts.Component.LLM:
        thinker = root.get("thinker_config") or {}
        visual = thinker.get("vision_config") or root.get(
            "vision_config") or {}
        deepstack = visual.get("deepstack_visual_indexes")
        default_features = (len(deepstack) if isinstance(deepstack, list) else
                            (3 if visual else 0))
        config.setdefault("num_deepstack_features", default_features)
    return config


def configure_base(config, **kwargs) -> None:
    """Enable the native MTP feedback contract on the thinker."""
    config.mtp_base = True


def configure_draft(config, **kwargs) -> None:
    """Select the checkpoint's unquantized full-attention MTP layers."""
    if config.mtp_num_hidden_layers is None:
        raise ValueError("Qwen3-Omni MTP draft requires mtp_num_hidden_layers")
    config.num_hidden_layers = config.mtp_num_hidden_layers
    config.layer_types = ["attention"] * config.num_hidden_layers
    config.attention_layer_types = ["full_attention"
                                    ] * config.num_hidden_layers
    config.gdn_cfg = None
    config.mtp_base = False
    config.tie_word_embeddings = False
    config.quant = replace(config.quant,
                           quant_type=quantization.QUANT_FP16,
                           excluded=(),
                           layer_overrides={},
                           is_mixed_precision=False)
