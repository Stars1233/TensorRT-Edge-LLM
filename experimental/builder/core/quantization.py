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
"""Checkpoint quantization contracts used by direct TensorRT graph builders."""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Optional, Tuple

from . import safetensors_np

QUANT_FP16 = "fp16"
QUANT_FP8 = "fp8"
QUANT_FP8_BLOCK = "fp8_block"
QUANT_MXFP8 = "mxfp8"
QUANT_NVFP4 = "nvfp4"
QUANT_INT4_AWQ = "int4_awq"
QUANT_INT4_AWQ_MODELOPT = "int4_awq_modelopt"
QUANT_INT4_GPTQ = "int4_gptq"
QUANT_INT8_SQ = "int8_sq"
QUANT_MIXED = "mixed_precision"

QUANT_TYPES = frozenset((
    QUANT_FP16,
    QUANT_FP8,
    QUANT_FP8_BLOCK,
    QUANT_MXFP8,
    QUANT_NVFP4,
    QUANT_INT4_AWQ,
    QUANT_INT4_AWQ_MODELOPT,
    QUANT_INT4_GPTQ,
    QUANT_INT8_SQ,
))


@dataclass(frozen=True)
class QuantConfig:
    """Effective quantization for checkpoint-backed linear layers."""

    quant_type: str = QUANT_FP16
    group_size: int = 1
    gptq_zero_point_offset: int = 1
    kv_cache_quant: Optional[str] = None
    excluded: Tuple[str, ...] = ()
    layer_overrides: Dict[str, str] = field(default_factory=dict)
    is_mixed_precision: bool = False

    def module_type(self,
                    module_name: str,
                    tie_word_embeddings: bool = False) -> str:
        """Return the concrete precision used by one linear module."""
        normalized = normalize_module_name(module_name)
        if normalized in self.excluded:
            return QUANT_FP16
        if (normalized == "lm_head" and tie_word_embeddings
                and normalized not in self.layer_overrides
                and self.quant_type == QUANT_FP16):
            return QUANT_FP16
        if self.layer_overrides:
            fallback = QUANT_FP16 if self.is_mixed_precision else self.quant_type
            return self.layer_overrides.get(normalized, fallback)
        return self.quant_type

    @property
    def is_quantized(self) -> bool:
        """Return whether at least one module uses quantized weights."""
        return self.quant_type != QUANT_FP16 or bool(self.layer_overrides)


def normalize_module_name(name: str) -> str:
    """Normalize the frontend's ordinary ``model.`` graph namespace."""
    if name.startswith("model."):
        name = name[len("model."):]
    return name


def parse_quantization(model_dir: str,
                       root: dict,
                       component: dict,
                       conversion: Optional[ModuleType] = None) -> QuantConfig:
    """Parse ModelOpt, AWQ, GPTQ, SmoothQuant, and mixed checkpoints."""
    hf_path = os.path.join(model_dir, "hf_quant_config.json")
    if os.path.isfile(hf_path):
        with open(hf_path) as quant_file:
            quantization = json.load(quant_file).get("quantization", {})
        algorithm = str(quantization.get("quant_algo") or "").upper()
        if algorithm == "MIXED_PRECISION":
            dominant, group_size, overrides = _parse_mixed_precision(
                quantization.get("quantized_layers", {}), conversion)
            return QuantConfig(
                quant_type=dominant,
                group_size=group_size,
                kv_cache_quant=_normalize_kv(
                    quantization.get("kv_cache_quant_algo")),
                excluded=tuple(
                    _effective_exclusions(
                        model_dir,
                        list(quantization.get("exclude_modules",
                                              [])), conversion)),
                layer_overrides=overrides,
                is_mixed_precision=True,
            )

        quant_type = algorithm_to_type(algorithm)
        group_size = int(quantization.get("group_size", 1))
        if quant_type == QUANT_MXFP8 and group_size == 1:
            group_size = 32
        if quant_type == QUANT_NVFP4 and group_size == 1:
            group_size = 16
        if quant_type in (QUANT_INT4_AWQ,
                          QUANT_INT4_AWQ_MODELOPT) and group_size == 1:
            group_size = 128
        excluded = _effective_exclusions(
            model_dir, list(quantization.get("exclude_modules", [])),
            conversion)
        excluded.extend(
            name for name in _detect_plain_weights(model_dir, conversion)
            if name not in excluded)
        return QuantConfig(
            quant_type=quant_type,
            group_size=group_size,
            kv_cache_quant=_normalize_kv(
                quantization.get("kv_cache_quant_algo")),
            excluded=tuple(sorted(set(excluded))),
        )

    embedded = (component.get("quantization_config")
                or root.get("quantization_config"))
    if not isinstance(embedded, dict):
        return QuantConfig()

    method = str(embedded.get("quant_method", "")).lower()
    if method == "awq":
        return QuantConfig(
            quant_type=QUANT_INT4_AWQ,
            group_size=int(embedded.get("group_size", 128)),
            excluded=tuple(
                _detect_unquantized_int4_modules(model_dir, conversion)),
        )
    if method == "gptq":
        return QuantConfig(
            quant_type=QUANT_INT4_GPTQ,
            group_size=int(embedded.get("group_size", 128)),
            gptq_zero_point_offset=_detect_gptq_zero_point_offset(
                model_dir, embedded),
            excluded=tuple(
                _detect_unquantized_int4_modules(model_dir, conversion)),
        )
    if method == "compressed-tensors":
        config_groups = embedded.get("config_groups") or {}
        formats = [str(embedded.get("format", "")).lower()]
        formats.extend(
            str(group.get("format", "")).lower()
            for group in config_groups.values())
        if any("nvfp4" in value for value in formats):
            first_group = next(iter(config_groups.values()), {})
            group_size = int(
                first_group.get("weights", {}).get("group_size", 16))
            excluded = _effective_exclusions(model_dir,
                                             list(embedded.get("ignore", [])),
                                             conversion)
            excluded.extend(
                name for name in _detect_plain_weights(model_dir, conversion)
                if name not in excluded)
            return QuantConfig(
                quant_type=QUANT_NVFP4,
                group_size=group_size,
                kv_cache_quant=("fp8"
                                if embedded.get("kv_cache_scheme") else None),
                excluded=tuple(sorted(set(excluded))),
            )
        raise ValueError(
            "unsupported compressed-tensors checkpoint format: "
            f"{', '.join(value or '<missing>' for value in formats)}")

    algorithm = str(embedded.get("quant_algo") or "").upper()
    if not algorithm:
        if method:
            raise ValueError(
                f"unsupported checkpoint quantization method {method!r}")
        return QuantConfig()
    quant_type = algorithm_to_type(algorithm)
    group_size = int(embedded.get("group_size", 1))
    config_groups = embedded.get("config_groups") or {}
    if config_groups:
        first_group = next(iter(config_groups.values()), {})
        group_size = int(
            first_group.get("weights", {}).get("group_size", group_size))
    if quant_type == QUANT_NVFP4 and group_size == 1:
        group_size = 16
    if quant_type == QUANT_MXFP8 and group_size == 1:
        group_size = 32
    if quant_type in (QUANT_INT4_AWQ,
                      QUANT_INT4_AWQ_MODELOPT) and group_size == 1:
        group_size = 128
    return QuantConfig(
        quant_type=quant_type,
        group_size=group_size,
        kv_cache_quant=("fp8" if embedded.get("kv_cache_scheme") else None),
        excluded=tuple(
            _effective_exclusions(model_dir, list(embedded.get("ignore", [])),
                                  conversion)),
    )


def algorithm_to_type(algorithm: str) -> str:
    """Map checkpoint algorithm names to direct-builder precision names."""
    algorithm = str(algorithm).upper()
    if "FP8_PB" in algorithm:
        return QUANT_FP8_BLOCK
    if "MXFP4" in algorithm:
        raise ValueError(
            f"unsupported checkpoint quantization algorithm {algorithm!r}: "
            "MXFP4 weight layouts are not implemented")
    if "NVFP4" in algorithm:
        return QUANT_NVFP4
    if "MXFP8" in algorithm:
        return QUANT_MXFP8
    if "FP8" in algorithm:
        return QUANT_FP8
    if "FP4" in algorithm:
        raise ValueError(
            f"unsupported checkpoint quantization algorithm {algorithm!r}: "
            "only NVFP4 weight layouts are implemented")
    if "W4A16" in algorithm and "AWQ" in algorithm:
        return QUANT_INT4_AWQ_MODELOPT
    if "AWQ" in algorithm or "INT4_AWQ" in algorithm:
        return QUANT_INT4_AWQ
    if "GPTQ" in algorithm:
        return QUANT_INT4_GPTQ
    if "W8A8" in algorithm or "INT8" in algorithm:
        return QUANT_INT8_SQ
    if not algorithm:
        return QUANT_FP16
    raise ValueError(
        f"unsupported checkpoint quantization algorithm {algorithm!r}")


def _parse_mixed_precision(
        quantized_layers: dict,
        conversion: Optional[ModuleType]) -> Tuple[str, int, Dict[str, str]]:
    algorithm_counts: Counter = Counter()
    group_sizes: Dict[str, int] = {}
    for layer_config in quantized_layers.values():
        algorithm = str(layer_config.get("quant_algo", "")).upper()
        algorithm_counts[algorithm] += 1
        group_sizes.setdefault(algorithm,
                               int(layer_config.get("group_size", 1)))
    if not algorithm_counts:
        return QUANT_FP16, 1, {}

    dominant_algorithm = algorithm_counts.most_common(1)[0][0]
    overrides: Dict[str, str] = {}
    for name, layer_config in quantized_layers.items():
        quant_type = algorithm_to_type(layer_config.get("quant_algo", ""))
        short_name = _normalize_checkpoint_name(name, conversion)
        for module_name in _expand_quantized_module(short_name, conversion):
            overrides[module_name] = quant_type
    return (algorithm_to_type(dominant_algorithm),
            group_sizes.get(dominant_algorithm, 1), overrides)


def _checkpoint_keys(model_dir: str) -> List[str]:
    try:
        with safetensors_np.SafetensorsStore(model_dir) as store:
            return store.keys()
    except (FileNotFoundError, OSError, ValueError):
        return []


def _normalize_checkpoint_name(name: str,
                               conversion: Optional[ModuleType]) -> str:
    normalize = getattr(conversion, "normalize_checkpoint_name", None)
    if normalize is not None:
        name = normalize(name)
    return normalize_module_name(name)


def _expand_quantized_module(
        name: str, conversion: Optional[ModuleType]) -> Tuple[str, ...]:
    expand = getattr(conversion, "expand_quantized_module", None)
    return tuple(expand(name)) if expand is not None else (name, )


def _finalize_exclusions(modules: List[str],
                         conversion: Optional[ModuleType]) -> List[str]:
    finalize = getattr(conversion, "finalize_exclusions", None)
    if finalize is not None:
        modules = list(finalize(modules))
    return sorted(set(modules))


def _detect_plain_weights(model_dir: str,
                          conversion: Optional[ModuleType]) -> List[str]:
    keys = _checkpoint_keys(model_dir)
    weights = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith(".weight")
    }
    scales = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith(".weight_scale")
    }
    excluded = set()
    for name in weights - scales:
        normalized = _normalize_checkpoint_name(name, conversion)
        excluded.update(_expand_quantized_module(normalized, conversion))
    return _finalize_exclusions(list(excluded), conversion)


def _detect_unquantized_int4_modules(
        model_dir: str, conversion: Optional[ModuleType]) -> List[str]:
    keys = _checkpoint_keys(model_dir)
    quantized = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith(".qweight")
    }
    weights = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith(".weight")
    }
    excluded = [
        module_name for name in weights - quantized
        for module_name in _expand_quantized_module(
            _normalize_checkpoint_name(name, conversion), conversion)
    ]
    known = {
        _normalize_checkpoint_name(name, conversion)
        for name in weights | quantized
    }
    if "lm_head" not in known:
        excluded.append("lm_head")
    return _finalize_exclusions(excluded, conversion)


def _effective_exclusions(model_dir: str, configured: List[str],
                          conversion: Optional[ModuleType]) -> List[str]:
    keys = _checkpoint_keys(model_dir)
    sidecar_suffixes = (".qweight", ".weight_scale", ".weight_scale_2",
                        ".input_scale", ".scales")
    quantized = {
        _normalize_checkpoint_name(key.rsplit(".", 1)[0], conversion)
        for key in keys if key.endswith(sidecar_suffixes)
    }
    normalized = []
    for name in configured:
        short_name = _normalize_checkpoint_name(name, conversion)
        if short_name not in quantized:
            normalized.extend(_expand_quantized_module(short_name, conversion))
    return _finalize_exclusions(normalized, conversion)


def _detect_gptq_zero_point_offset(model_dir: str, config: dict) -> int:
    if not bool(config.get("sym", False)):
        return 1
    try:
        with safetensors_np.SafetensorsStore(model_dir) as store:
            key = next(name for name in store.keys()
                       if name.endswith(".qzeros"))
            packed = store.get_numpy(key).reshape(-1)[:1024]
    except (FileNotFoundError, KeyError, OSError, StopIteration, ValueError):
        return 1
    if packed.size == 0:
        return 1
    nibbles = []
    for value in packed:
        unsigned = int(value) & 0xFFFFFFFF
        nibbles.extend((unsigned >> (4 * index)) & 0xF for index in range(8))
    return 0 if nibbles and all(value == 8 for value in nibbles) else 1


def _normalize_kv(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    return normalized or None
