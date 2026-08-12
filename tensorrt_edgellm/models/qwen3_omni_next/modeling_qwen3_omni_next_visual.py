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
"""Qwen3-Next Omni visual encoder adapter.

Delegates the computation graph to :class:`Qwen3VLVisualModel`; only the
checkpoint key naming differs and is translated below before load.
"""
import re
from typing import TYPE_CHECKING

from ..qwen3_vl.modeling_qwen3_vl_visual import \
    Qwen3VLVisualModel as Qwen3OmniNextVisualModel
from ..qwen3_vl.modeling_qwen3_vl_visual import build_qwen3_vl_visual

if TYPE_CHECKING:
    import torch

    from ...config import ModelConfig

__all__ = ["Qwen3OmniNextVisualModel", "build_qwen3_omni_next_visual"]

# Prefix translation: HF Qwen3-Next Omni -> canonical Qwen3-VL.
_HF_OMNI_VISUAL_PREFIX = "thinker.visual."
_TARGET_VISUAL_PREFIX = "model.visual."

# Sub-module suffix translations. Replacements are disjoint, but we keep
# ln_q first for readability.
_SUBMODULE_RENAMES = (
    (".ln_q.", ".norm."),
    (".mlp.0.", ".linear_fc1."),
    (".mlp.2.", ".linear_fc2."),
)

# Compiled once. Matches ``model.visual.merger_list.`` exactly so we
# don't accidentally rewrite an unrelated future submodule named
# ``merger_list``.
_MERGER_LIST_RE = re.compile(r"\b" + re.escape(_TARGET_VISUAL_PREFIX) +
                             r"merger_list\.")
_DEEPSTACK_MERGER_LIST = _TARGET_VISUAL_PREFIX + "deepstack_merger_list."


def _translate_qwen3_omni_next_visual_key(key: str) -> str:
    """Return *key* translated from HF Qwen3-Next Omni to Qwen3-VL naming.

    Non-visual keys (``thinker.audio_tower.*``, ``thinker.model.*``,
    ``talker.*``, ...) are returned unchanged so downstream filters
    (e.g. ``build_qwen3_vl_visual``'s ``model.visual.`` prefix filter)
    drop them naturally.
    """
    if not key.startswith(_HF_OMNI_VISUAL_PREFIX):
        return key
    out = _TARGET_VISUAL_PREFIX + key[len(_HF_OMNI_VISUAL_PREFIX):]
    out = _MERGER_LIST_RE.sub(_DEEPSTACK_MERGER_LIST, out)
    for old, new in _SUBMODULE_RENAMES:
        out = out.replace(old, new)
    return out


def _validate_visual_coverage(model,
                              remapped_weights: dict,
                              prefix: str = "model.visual") -> None:
    """Assert every model parameter is populated by *remapped_weights*
    and every visual-prefixed key matches a model parameter.

    Without this check, ``load_submodule_weights`` warns on extra-source
    keys but silently leaves *model-side* parameters uninitialised
    (zero-init from ``nn.Parameter(torch.empty(...))``). An uninitialised
    visual encoder exports to ONNX without raising — the bug only surfaces
    at inference time as garbled image features.
    """
    expected = set(model.state_dict().keys())
    strip = prefix + "." if prefix else ""
    provided = {
        k[len(strip):] if strip else k
        for k in remapped_weights.keys() if not strip or k.startswith(strip)
    }
    missing = sorted(expected - provided)
    extra_visual = sorted(provided - expected)
    if missing or extra_visual:

        def _fmt(items):
            head = items[:10]
            return f"{head}{' ...' if len(items) > 10 else ''}"

        raise RuntimeError(
            "Qwen3-Next Omni visual encoder weight coverage mismatch.\n"
            f"  Missing in checkpoint ({len(missing)}): {_fmt(missing)}\n"
            f"  Extra under '{prefix}.' ({len(extra_visual)}): "
            f"{_fmt(extra_visual)}\n"
            "Check the HF-Omni -> Qwen3-VL key translation in "
            "_translate_qwen3_omni_next_visual_key().")


def build_qwen3_omni_next_visual(config: dict, weights: dict,
                                 model_config: "ModelConfig",
                                 dtype: "torch.dtype"):
    """Build a Qwen3-Next Omni visual encoder.

    Translates HF Qwen3-Next Omni checkpoint keys to the Qwen3-VL naming
    expected by :func:`build_qwen3_vl_visual`, then delegates. After
    construction, validates that every model parameter was covered by the
    remapped checkpoint (see :func:`_validate_visual_coverage`).
    """
    remapped = {
        _translate_qwen3_omni_next_visual_key(k): v
        for k, v in weights.items()
    }
    # If HF Qwen3-Next Omni vision_config omits the explicit
    # ``num_position_embeddings`` field that the Qwen3-VL builder demands
    # but ships ``image_size`` + ``patch_size``, inject the computed value
    # so build_qwen3_vl_visual can run without forking its config schema.
    if "num_position_embeddings" not in config:
        image_size = config.get("image_size")
        patch_size = config.get("patch_size")
        if image_size is not None and patch_size:
            grid = int(image_size) // int(patch_size)
            config = {**config, "num_position_embeddings": grid * grid}
    # The quant ``excluded`` list arrives in HF Qwen3-Omni naming
    # (``visual.merger.mlp.0`` / ``mlp.2``) — short names derived from
    # checkpoint keys via ``_normalize_module_name``. But the Qwen3-VL builder
    # queries ``make_linear`` with the renamed form
    # (``visual.merger.linear_fc1`` / ``linear_fc2``), so the exclusion never
    # matches and an FP16 merger would be promoted to AWQLinear/NVFP4Linear,
    # demanding scale tensors the FP16 ckpt does not provide.  Mirror the
    # rename so excluded entries cover both name spaces.
    extra_excluded = []
    for entry in model_config.quant.excluded:
        for old, new in _SUBMODULE_RENAMES:
            old_no_dot = old.rstrip(".")
            new_no_dot = new.rstrip(".")
            if entry.endswith(old_no_dot):
                extra_excluded.append(entry[:-len(old_no_dot)] + new_no_dot)
            elif old in entry:
                extra_excluded.append(entry.replace(old, new))
    if extra_excluded:
        existing = set(model_config.quant.excluded)
        model_config.quant.excluded = list(model_config.quant.excluded) + [
            r for r in extra_excluded if r not in existing
        ]
    model = build_qwen3_vl_visual(config,
                                  remapped,
                                  model_config=model_config,
                                  dtype=dtype)
    _validate_visual_coverage(model, remapped)
    return model
