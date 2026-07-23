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
"""Quantization recipe configurations for ModelOpt."""

import copy
from typing import Any, Dict, List, Optional

import modelopt.torch.quantization as mtq

# LM-head overrides layer on top of the backbone recipe. Each entry must
# enable only the lm_head quantizers — never a global ``"default": {"enable":
# False}`` here. Such a wildcard merged into the list as ``{"quantizer_name":
# "*", "enable": False}`` lands AFTER the backbone's ``*weight_quantizer``
# enables, silently disabling every body Linear's weight_quantizer. The
# backbone CFG (``mtq.FP8_DEFAULT_CFG`` / ``NVFP4_DEFAULT_CFG`` / ...) already
# disables everything by default; only that one should — re-disabling here
# is the producer-side bug that left 100+ checkpoints with fp16 bodies
# despite ``-LMFP8`` / ``-LMNVFP4`` names.

FP8_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
    }
}

INT4_AWQ_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {
                -1: 128,
                "type": "static"
            },
            "enable": True,
        },
    }
}

NVFP4_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True,
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True,
        },
    }
}

MXFP8_LM_HEAD = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
    }
}

FP8_ATTN = {
    "quant_cfg": {
        "*q_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
        "*k_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
        "*v_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
    }
}

# Visual submodule prefixes as they appear in HuggingFace VLM module paths.
# Different families pick different names — Qwen-VL uses ``visual.*``, InternVL
# 3.5 uses ``vision_tower.*`` + ``multi_modal_projector.*``, InternVL3 (custom)
# uses ``vision_model.*`` + ``mlp1.*``, Phi-4-multimodal nests the visual
# encoder under ``model.embed_tokens_extend.image_embed.*``, and Gemma4 uses
# ``embed_vision.*``.  This single tuple is the source of truth; everything
# below derives from it.
_VISUAL_PREFIXES = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "mlp1",
    "image_embed",
    "embed_vision",
)


def _visual_quant_cfg(input_cfg: Optional[Dict[str, Any]],
                      weight_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Expand a single ``input_cfg`` / ``weight_cfg`` pair into wildcard
    patterns covering every visual prefix.

    Modelopt matches patterns with ``fnmatch``; ``*visual*input_quantizer``
    only catches paths that contain the literal substring ``visual``, so a
    single-prefix pattern would silently leave ``vision_tower.*``,
    ``vision_model.*``, ``multi_modal_projector.*``, ``mlp1.*`` modules
    unquantized.  Emit one entry per prefix so all VLM families are covered.

    ``input_cfg=None`` skips the input-quantizer entries (used for INT4 AWQ
    which is W4A16, weight-only).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for prefix in _VISUAL_PREFIXES:
        out[f"*{prefix}*weight_quantizer"] = weight_cfg
        if input_cfg is not None:
            out[f"*{prefix}*input_quantizer"] = input_cfg
    return out


# Per-visual-submodule overrides.  Used when the user asks for a visual
# precision that differs from the backbone (e.g. backbone NVFP4 + visual FP8),
# mirroring the *_LM_HEAD pattern.  When visual matches an FP8 backbone,
# this is still required to override the default disable on visual prefixes.
# Only FP8 visual is exposed today; lower-bit recipes (NVFP4 / MXFP8 / INT4)
# are deferred until each is validated end-to-end on a VLM eval.
FP8_VISUAL = {
    "quant_cfg":
    _visual_quant_cfg(
        input_cfg={
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        weight_cfg={
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
    )
}

# Wildcards grouped so ``build_quant_config`` can selectively disable them.
# Derived from ``_VISUAL_PREFIXES`` so adding a prefix flows through to both
# the ``disable`` patterns here and the per-recipe overrides above.
_VISUAL_PATTERNS = tuple(f"*{p}.*" for p in _VISUAL_PREFIXES)

# Audio submodule prefixes. Mirror of ``_VISUAL_PREFIXES`` for the audio
# tower (Qwen3-ASR / Qwen3-Omni audio encoder, Phi-4mm audio_embed, Gemma4
# embed_audio).
_AUDIO_PREFIXES = (
    "audio_tower",
    "audio_embed",
    "embed_audio",
)
_AUDIO_PATTERNS = tuple(f"*{p}.*" for p in _AUDIO_PREFIXES)


def _audio_quant_cfg(input_cfg: Optional[Dict[str, Any]],
                     weight_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Expand a single ``input_cfg`` / ``weight_cfg`` pair into wildcard
    patterns covering every audio prefix.

    Mirrors :func:`_visual_quant_cfg`. ``input_cfg=None`` skips the
    input-quantizer entries (used for weight-only recipes).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for prefix in _AUDIO_PREFIXES:
        out[f"*{prefix}*weight_quantizer"] = weight_cfg
        if input_cfg is not None:
            out[f"*{prefix}*input_quantizer"] = input_cfg
    return out


# Per-audio-submodule overrides. Used when the user asks for an audio
# precision that differs from the backbone (e.g. backbone NVFP4 + audio
# FP8), mirroring the *_VISUAL pattern. Only FP8 audio is exposed today;
# lower-bit recipes (NVFP4 / MXFP8) are deferred until validated end-to-end
# on an ASR eval (WER on LibriSpeech).
FP8_AUDIO = {
    "quant_cfg":
    _audio_quant_cfg(
        input_cfg={
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        weight_cfg={
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
    )
}

# Qwen3-Omni Talker CodePredictor — 5-layer Qwen3 decoder under
# ``talker.code_predictor.*`` emitting residual codec tokens.
_CP_PREFIXES = ("code_predictor", )
_CP_PATTERNS = tuple(f"*{p}.*" for p in _CP_PREFIXES)

# Linear submodules where FP8 loses precision: down_proj (silu*up ∈
# [-39, 72] → per-tensor amax quantizes to garbage) and lm_head[0..14]
# (each codebook sees 1/15 of calib signal, amax undertrained).
_CP_LINEAR_EXCLUDES = ("lm_head", "down_proj")

# Attention KV-cache BMM quantizers (mixed-precision KV rejected by ONNX
# export). Named differently from Linear quantizers — a single
# ``q_bmm_quantizer`` attribute, not ``*_quantizer/input`` sub-selectors.
_CP_BMM_EXCLUDES = ("q_bmm", "k_bmm", "v_bmm")

# Qwen3-Omni Code2Wav — codec→waveform decoder. Always disabled today.
_CODE2WAV_PATTERNS = ("*code2wav.*", )


def _cp_quant_cfg(input_cfg: Optional[Dict[str, Any]],
                  weight_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Expand ``input_cfg`` / ``weight_cfg`` into CP wildcard patterns and
    layer disables for :data:`_CP_LINEAR_EXCLUDES` / :data:`_CP_BMM_EXCLUDES`.

    Mirrors :func:`_visual_quant_cfg` / :func:`_audio_quant_cfg`; disables are
    emitted after generic enables so specific patterns win (ModelOpt matches
    in insertion order — last-writer wins).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for prefix in _CP_PREFIXES:
        out[f"*{prefix}*weight_quantizer"] = weight_cfg
        if input_cfg is not None:
            out[f"*{prefix}*input_quantizer"] = input_cfg
    for prefix in _CP_PREFIXES:
        for sub in _CP_LINEAR_EXCLUDES:
            out[f"*{prefix}*{sub}*weight_quantizer"] = {"enable": False}
            out[f"*{prefix}*{sub}*input_quantizer"] = {"enable": False}
        for sub in _CP_BMM_EXCLUDES:
            out[f"*{prefix}*{sub}_quantizer"] = {"enable": False}
    return out


# CP FP8 recipe. Per-channel weight (axis=0) + per-tensor static input
# (axis=None); see :data:`_CP_SUBMODULE_EXCLUDES` for disabled submodules.
FP8_CP = {
    "quant_cfg":
    _cp_quant_cfg(
        input_cfg={
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        weight_cfg={
            "num_bits": (4, 3),
            "axis": 0,
            "enable": True,
        },
    )
}

_BACKBONE_CFG_MAP = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
}

_LM_HEAD_CFG_MAP = {
    "fp8": FP8_LM_HEAD,
    "int4_awq": INT4_AWQ_LM_HEAD,
    "nvfp4": NVFP4_LM_HEAD,
    "mxfp8": MXFP8_LM_HEAD,
}

_VISUAL_CFG_MAP = {
    "fp8": FP8_VISUAL,
}

_AUDIO_CFG_MAP = {
    "fp8": FP8_AUDIO,
}

_CP_CFG_MAP = {
    "fp8": FP8_CP,
}


def _disable_groups(*pattern_groups) -> Dict[str, Dict[str, bool]]:
    """Flatten wildcard pattern groups into a ``{pattern: {enable: False}}`` dict."""
    out: Dict[str, Dict[str, bool]] = {}
    for group in pattern_groups:
        for k in group:
            out[k] = {"enable": False}
    return out


def _dict_entry_to_list_entry(pattern: str,
                              value: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy dict-style quant cfg entry to ModelOpt list-style entry."""
    entry: Dict[str, Any] = {
        "quantizer_name": "*" if pattern == "default" else pattern
    }
    if set(value.keys()) == {"enable"}:
        entry["enable"] = value["enable"]
        return entry

    cfg_payload = {k: v for k, v in value.items() if k != "enable"}
    if cfg_payload:
        entry["cfg"] = cfg_payload
    if "enable" in value:
        entry["enable"] = value["enable"]
    return entry


def _merge_quant_cfg(target: Any, extra: Any) -> Any:
    """Merge quant cfg data while supporting both dict and list structures."""
    if isinstance(target, dict):
        if isinstance(extra, dict):
            target.update(extra)
            return target
        if isinstance(extra, list):
            for item in extra:
                if not isinstance(item, dict):
                    continue
                quantizer_name = item.get("quantizer_name")
                if not quantizer_name:
                    continue
                pattern = "default" if quantizer_name == "*" else quantizer_name
                merged_value: Dict[str, Any] = {}
                if isinstance(item.get("cfg"), dict):
                    cfg_payload = item["cfg"]
                    if "enable" in cfg_payload:
                        raise ValueError(
                            "Cannot losslessly convert ModelOpt list-style "
                            "quant_cfg to dict form when cfg contains "
                            "'enable'.")
                    merged_value.update(cfg_payload)
                if "enable" in item:
                    merged_value["enable"] = item["enable"]
                target[pattern] = merged_value
            return target
        raise TypeError(
            f"Unsupported quant_cfg source type for dict target: {type(extra)}"
        )

    if isinstance(target, list):
        pending_entries: List[Dict[str, Any]] = []
        if isinstance(extra, dict):
            for pattern, value in extra.items():
                if isinstance(value, dict):
                    pending_entries.append(
                        _dict_entry_to_list_entry(pattern, value))
        elif isinstance(extra, list):
            for item in extra:
                if isinstance(item, dict) and item.get("quantizer_name"):
                    pending_entries.append(copy.deepcopy(item))
        else:
            raise TypeError(
                f"Unsupported quant_cfg source type for list target: {type(extra)}"
            )

        index_by_name = {}
        for idx, item in enumerate(target):
            if isinstance(item, dict) and item.get("quantizer_name"):
                index_by_name[item["quantizer_name"]] = idx

        for entry in pending_entries:
            quantizer_name = entry["quantizer_name"]
            if quantizer_name in index_by_name:
                target[index_by_name[quantizer_name]] = entry
            else:
                index_by_name[quantizer_name] = len(target)
                target.append(entry)
        return target

    raise TypeError(f"Unsupported quant_cfg target type: {type(target)}")


def _remove_lm_head_quantizers(quant_cfg: Any) -> Any:
    """Remove lm_head-related quantizers from dict/list quant cfg."""
    if isinstance(quant_cfg, dict):
        return {k: v for k, v in quant_cfg.items() if "lm_head" not in k}
    if isinstance(quant_cfg, list):
        return [
            item for item in quant_cfg
            if not ("lm_head" in str(item.get("quantizer_name", "")))
        ]
    raise TypeError(
        f"Unsupported quant_cfg type when removing lm_head: {type(quant_cfg)}")


def build_quant_config(
    quantization: Optional[str] = None,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    visual_quantization: Optional[str] = None,
    audio_quantization: Optional[str] = None,
    cp_quantization: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a composite ModelOpt quantization config from method names.

    Args:
        quantization:          Backbone precision (``fp8`` / ``nvfp4`` / ...).
        lm_head_quantization:  Optional LM-head precision (typically matches
                               backbone; can also be lower for head sensitivity).
        kv_cache_quantization: Optional KV-cache precision (``fp8`` only).
        visual_quantization:   Optional visual-tower precision.  When ``None``
                               the visual tower is left untouched (``enable=False``
                               on all ``*visual.*`` / ``*vision_tower.*`` /
                               ``*multi_modal_projector.*`` / ``*mlp1.*``
                               patterns) — this preserves the existing behaviour.
                               When set to a method name the visual tower is
                               quantized via that method; if it matches the
                               backbone we just stop disabling, otherwise we
                               layer an explicit per-visual override on top.
        audio_quantization:    Optional audio-tower precision. Mirror of
                               ``visual_quantization`` for ``audio_tower.*`` /
                               ``audio_embed.*`` paths. Only ``fp8`` is exposed
                               today.
        cp_quantization:       Optional Qwen3-Omni CodePredictor precision
                               (``fp8`` only today).  When ``None`` the CP is
                               left untouched (``*code_predictor.*`` disabled);
                               when set, the CP backbone is quantized but
                               ``*code_predictor*down_proj*`` is explicitly
                               kept unquantized to preserve the FP32 MLP WAR.
    """
    if quantization is None:
        cfg = {"quant_cfg": {"default": {"enable": False}}, "algorithm": "max"}
    elif quantization in _BACKBONE_CFG_MAP:
        # Deep-copy so subsequent quant cfg merges do not mutate the shared
        # module-level dict/list that ModelOpt reuses for
        # every quant_algo (``mtq.NVFP4_DEFAULT_CFG`` and friends are
        # singletons; ``.copy()`` only copies the outer mapping, leaving
        # the inner ``quant_cfg`` aliased to the global).
        cfg = copy.deepcopy(_BACKBONE_CFG_MAP[quantization])
    else:
        raise ValueError(f"Unsupported quantization: {quantization}. "
                         f"Choose from: {list(_BACKBONE_CFG_MAP)}")

    if lm_head_quantization is not None:
        if lm_head_quantization not in _LM_HEAD_CFG_MAP:
            raise ValueError(
                f"Unsupported lm_head_quantization: {lm_head_quantization}. "
                f"Choose from: {list(_LM_HEAD_CFG_MAP)}")
        cfg["quant_cfg"] = _remove_lm_head_quantizers(cfg["quant_cfg"])
        cfg["quant_cfg"] = _merge_quant_cfg(
            cfg["quant_cfg"],
            _LM_HEAD_CFG_MAP[lm_head_quantization]["quant_cfg"],
        )

    if kv_cache_quantization == "fp8":
        cfg["quant_cfg"] = _merge_quant_cfg(cfg["quant_cfg"],
                                            mtq.FP8_KV_CFG["quant_cfg"])
        cfg["quant_cfg"] = _merge_quant_cfg(cfg["quant_cfg"],
                                            FP8_ATTN["quant_cfg"])

    # Disable every non-LLM group by default. Re-enable the ones the user
    # explicitly asked to quantize.  ``_CODE2WAV_PATTERNS`` is always disabled
    # today; ``_CP_PATTERNS`` only when ``cp_quantization is None``.
    groups_to_disable = [_CODE2WAV_PATTERNS]
    if cp_quantization is None:
        groups_to_disable.append(_CP_PATTERNS)
    if visual_quantization is None:
        groups_to_disable.append(_VISUAL_PATTERNS)
    if audio_quantization is None:
        groups_to_disable.append(_AUDIO_PATTERNS)
    cfg["quant_cfg"] = _merge_quant_cfg(
        cfg["quant_cfg"],
        _disable_groups(*groups_to_disable),
    )

    # CP override layered AFTER the generic disables so specific patterns win.
    if cp_quantization is not None:
        if cp_quantization not in _CP_CFG_MAP:
            raise ValueError(
                f"Unsupported cp_quantization: {cp_quantization}. "
                f"Choose from: {list(_CP_CFG_MAP)}")
        cfg["quant_cfg"] = _merge_quant_cfg(
            cfg["quant_cfg"],
            _CP_CFG_MAP[cp_quantization]["quant_cfg"],
        )

    # When visual != backbone, layer an explicit override. When visual ==
    # backbone we don't need overrides — the backbone's generic wildcards
    # already cover visual submodules now that we've stopped disabling them.
    if (visual_quantization is not None
            and visual_quantization != quantization):
        if visual_quantization not in _VISUAL_CFG_MAP:
            raise ValueError(
                f"Unsupported visual_quantization: {visual_quantization}. "
                f"Choose from: {list(_VISUAL_CFG_MAP)}")
        cfg["quant_cfg"] = _merge_quant_cfg(
            cfg["quant_cfg"],
            _VISUAL_CFG_MAP[visual_quantization]["quant_cfg"],
        )

    if (audio_quantization is not None and audio_quantization != quantization):
        if audio_quantization not in _AUDIO_CFG_MAP:
            raise ValueError(
                f"Unsupported audio_quantization: {audio_quantization}. "
                f"Choose from: {list(_AUDIO_CFG_MAP)}")
        cfg["quant_cfg"] = _merge_quant_cfg(
            cfg["quant_cfg"],
            _AUDIO_CFG_MAP[audio_quantization]["quant_cfg"],
        )

    return cfg
