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
"""Cosmos3 checkpoint loading and weight splitting."""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from safetensors.torch import load_file as _load_safetensors
except ImportError:  # pragma: no cover
    _load_safetensors = None

_UND_ATTN_RENAME = {
    ".self_attn.to_q.": ".self_attn.q_proj.",
    ".self_attn.to_k.": ".self_attn.k_proj.",
    ".self_attn.to_v.": ".self_attn.v_proj.",
    ".self_attn.to_out.": ".self_attn.o_proj.",
    ".self_attn.norm_q.": ".self_attn.q_norm.",
    ".self_attn.norm_k.": ".self_attn.k_norm.",
}

_GEN_LAYER_MARKERS = (
    ".self_attn.add_q_proj.",
    ".self_attn.add_k_proj.",
    ".self_attn.add_v_proj.",
    ".self_attn.to_add_out.",
    ".self_attn.norm_added_q.",
    ".self_attn.norm_added_k.",
    ".mlp_moe_gen.",
    ".input_layernorm_moe_gen.",
    ".post_attention_layernorm_moe_gen.",
)

_GEN_TOPLEVEL = (
    "proj_in.",
    "proj_out.",
    "time_embedder.",
    "action_proj_in.",
    "action_proj_out.",
    "action_modality_embed",
    "norm_moe_gen.",
)

_GEN_LAYER_RENAME = {
    ".self_attn.add_q_proj.": ".cross_attention.to_q.",
    ".self_attn.add_k_proj.": ".cross_attention.to_k.",
    ".self_attn.add_v_proj.": ".cross_attention.to_v.",
    ".self_attn.to_add_out.": ".cross_attention.to_out.",
    ".self_attn.norm_added_q.": ".cross_attention.norm_q.",
    ".self_attn.norm_added_k.": ".cross_attention.norm_k.",
    ".mlp_moe_gen.": ".mlp.",
    ".input_layernorm_moe_gen.": ".input_layernorm.",
    ".post_attention_layernorm_moe_gen.": ".post_attention_layernorm.",
}


def _load_dir_safetensors(
        directory: str,
        pattern: str = "*.safetensors") -> Dict[str, torch.Tensor]:
    """Load and merge every safetensors shard in ``directory``."""
    if _load_safetensors is None:
        raise RuntimeError(
            "safetensors is required to load Cosmos3 checkpoints")
    shards = sorted(glob.glob(os.path.join(directory, pattern)))
    if not shards:
        raise FileNotFoundError(
            f"No safetensors found in {directory} ({pattern})")
    merged: Dict[str, torch.Tensor] = {}
    for shard in shards:
        merged.update(_load_safetensors(shard))
    logger.info("Loaded %d tensors from %d shard(s) in %s", len(merged),
                len(shards), directory)
    return merged


def _rename(key: str, table: Dict[str, str]) -> str:
    for src, dst in table.items():
        if src in key:
            return key.replace(src, dst)
    return key


def split_transformer_weights(
    transformer_dir: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return ``(und_weights, gen_weights)`` from the unified transformer dir."""
    raw = _load_dir_safetensors(transformer_dir)
    und: Dict[str, torch.Tensor] = {}
    gen: Dict[str, torch.Tensor] = {}

    for key, tensor in raw.items():
        if any(key.startswith(p) for p in _GEN_TOPLEVEL):
            if key.startswith("action_proj_in.") or key.startswith(
                    "action_proj_out."):
                gen[key.replace(".fc.weight",
                                ".fc").replace(".bias.weight",
                                               ".bias")] = tensor
            else:
                gen[key] = tensor
            continue

        if any(marker in key for marker in _GEN_LAYER_MARKERS):
            gen[_rename(key, _GEN_LAYER_RENAME)] = tensor
            continue

        und[_rename(key, _UND_ATTN_RENAME)] = tensor

    logger.info("Split transformer: %d UND tensors, %d GEN tensors", len(und),
                len(gen))
    return und, gen


def load_vae_weights(vae_dir: str) -> Dict[str, torch.Tensor]:
    """Load the AutoencoderKLWan weights."""
    return _load_dir_safetensors(vae_dir)


def load_config_json(checkpoint_dir: str, sub: str = "transformer") -> dict:
    """Load ``config.json`` from a Cosmos3 checkpoint subdirectory."""
    with open(os.path.join(checkpoint_dir, sub, "config.json")) as f:
        return json.load(f)
