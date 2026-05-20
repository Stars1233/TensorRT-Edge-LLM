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
"""Export-time model mutation for reduced-vocabulary ONNX export."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from .constants import VOCAB_INFO_NAME, VOCAB_MAP_NAME

logger = logging.getLogger(__name__)


def load_reduced_vocab_map(reduced_vocab_dir: str,
                           vocab_size: Optional[int] = None,
                           device: str = "cpu") -> torch.Tensor:
    """Load and validate ``vocab_map.safetensors`` from *reduced_vocab_dir*."""
    vocab_map_file = Path(reduced_vocab_dir) / VOCAB_MAP_NAME
    if not vocab_map_file.exists():
        raise FileNotFoundError(f"{VOCAB_MAP_NAME} not found in "
                                f"{reduced_vocab_dir!r}")

    with safe_open(vocab_map_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if "vocab_map" not in keys:
            raise KeyError(f"'vocab_map' key not found in {vocab_map_file}. "
                           f"Available keys: {keys}")
        vocab_map = f.get_tensor("vocab_map")

    if vocab_map.dim() != 1:
        raise ValueError(f"vocab_map must be 1D, got shape "
                         f"{tuple(vocab_map.shape)}")
    if vocab_map.numel() == 0:
        raise ValueError("vocab_map must not be empty")
    if torch.is_floating_point(vocab_map):
        raise TypeError(f"vocab_map must contain integer token IDs, got "
                        f"{vocab_map.dtype}")

    vocab_map = vocab_map.to(torch.long)
    if torch.any(vocab_map < 0):
        raise ValueError("vocab_map contains negative token IDs")
    if torch.unique(vocab_map).numel() != vocab_map.numel():
        raise ValueError("vocab_map contains duplicate token IDs")
    if vocab_size is not None and torch.max(vocab_map).item() >= vocab_size:
        raise ValueError(f"vocab_map token ID {torch.max(vocab_map).item()} "
                         f"is outside vocab_size={vocab_size}")

    return vocab_map.to(device=device, dtype=torch.long)


def apply_reduced_vocab_from_dir(model: nn.Module,
                                 reduced_vocab_dir: str) -> None:
    """Load *reduced_vocab_dir* and apply vocabulary reduction to *model*."""
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    device = _module_device(model)
    vocab_map = load_reduced_vocab_map(reduced_vocab_dir,
                                       vocab_size=vocab_size,
                                       device=str(device))
    apply_reduced_vocab(model, vocab_map)
    model._reduced_vocab_dir = reduced_vocab_dir


def apply_reduced_vocab(model: nn.Module, vocab_map: torch.Tensor) -> None:
    """Reduce the model LM head in place for reduced-vocabulary export."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model has no config; cannot apply reduced vocab")
    if getattr(config, "is_eagle3_draft", False):
        raise ValueError("Reduced vocabulary is not supported for EAGLE3 "
                         "draft models")

    vocab_map = vocab_map.to(device=_module_device(model), dtype=torch.long)
    reduced_vocab_size = int(vocab_map.numel())
    config.reduced_vocab_size = reduced_vocab_size
    model._reduced_vocab_map_for_runtime = vocab_map.detach().cpu().to(
        torch.int32)

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Model has no lm_head; cannot apply reduced vocab")

    _reduce_lm_head_in_place(lm_head, vocab_map)
    logger.info("Reduced lm_head output dimension to %d", reduced_vocab_size)


def copy_reduced_vocab_artifacts(model: nn.Module,
                                 out_dir: str,
                                 reduced_vocab_dir: str = "") -> None:
    """Write/copy reduced-vocabulary sidecars to *out_dir* when enabled."""
    config = getattr(model, "config", None)
    if not getattr(config, "reduced_vocab_size", None):
        return

    os.makedirs(out_dir, exist_ok=True)
    src_dir = reduced_vocab_dir or getattr(model, "_reduced_vocab_dir", "")
    vocab_map = getattr(model, "_reduced_vocab_map_for_runtime", None)
    if src_dir:
        src = os.path.join(src_dir, VOCAB_MAP_NAME)
        if not os.path.exists(src):
            raise FileNotFoundError(f"{VOCAB_MAP_NAME} not found in "
                                    f"{src_dir!r}")
        if vocab_map is None:
            shutil.copy2(src, os.path.join(out_dir, VOCAB_MAP_NAME))
        else:
            save_file({"vocab_map": vocab_map.cpu().to(torch.int32)},
                      os.path.join(out_dir, VOCAB_MAP_NAME))
        info_src = os.path.join(src_dir, VOCAB_INFO_NAME)
        if os.path.exists(info_src):
            shutil.copy2(info_src, os.path.join(out_dir, VOCAB_INFO_NAME))
        return

    if vocab_map is None:
        raise ValueError("Reduced vocab is enabled but no vocab_map is "
                         "available to write")
    save_file({"vocab_map": vocab_map.cpu().to(torch.int32)},
              os.path.join(out_dir, VOCAB_MAP_NAME))


def should_apply_reduced_vocab_before_repacking(model: nn.Module) -> bool:
    """Return True when lm_head reduction must run on checkpoint-layout tensors."""
    from ..models.linear import (AWQLinear, GPTQLinear,
                                 ModelOptAWQPrepackedLinear)
    return isinstance(getattr(model, "lm_head", None),
                      (AWQLinear, GPTQLinear, ModelOptAWQPrepackedLinear))


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def _select_output_rows(tensor: Optional[torch.Tensor],
                        vocab_map: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.index_select(
        0, vocab_map.to(device=tensor.device)).clone().contiguous()


def _set_buffer(module: nn.Module, name: str,
                value: Optional[torch.Tensor]) -> None:
    if value is None:
        module._buffers[name] = None
    else:
        module._buffers[name] = value


def _validate_int4_checkpoint_reduction(lm_head: nn.Module,
                                        vocab_map: torch.Tensor) -> None:
    reduced_vocab_size = int(vocab_map.numel())
    if reduced_vocab_size % 128 != 0:
        raise ValueError(
            "Reduced vocabulary with a packed INT4 lm_head requires "
            f"reduced_vocab_size to be a multiple of 128, got "
            f"{reduced_vocab_size}.")
    if getattr(lm_head, "group_size", None) != 128:
        raise ValueError(
            "Reduced vocabulary with a packed INT4 lm_head requires "
            f"group_size=128, got {getattr(lm_head, 'group_size', None)}.")
    if lm_head.out_features % 8 != 0:
        raise ValueError(
            "Packed INT4 lm_head reduction requires original out_features to "
            f"be divisible by 8, got {lm_head.out_features}.")
    if lm_head.in_features % 64 != 0:
        raise ValueError(
            "Packed INT4 lm_head reduction requires in_features to be "
            f"divisible by 64, got {lm_head.in_features}.")


def _select_column_packed_int4(
        tensor: torch.Tensor, vocab_map: torch.Tensor, out_features: int,
        channel_to_bit: tuple[int, ...]) -> torch.Tensor:
    if tuple(tensor.shape)[1] != out_features // 8:
        raise ValueError(
            f"Expected output-packed INT4 tensor shape (*, {out_features // 8}), "
            f"got {tuple(tensor.shape)}.")

    rows = vocab_map.detach().cpu().to(torch.long)
    if rows.numel() == 0 or torch.min(rows).item() < 0 or torch.max(
            rows).item() >= out_features:
        raise ValueError("vocab_map contains token IDs outside lm_head "
                         f"out_features={out_features}")
    src = tensor.detach().cpu().to(torch.int32)
    reduced_vocab_size = int(rows.numel())
    packed = torch.zeros(src.shape[0],
                         reduced_vocab_size // 8,
                         dtype=torch.int32)

    for target_offset in range(8):
        selected_rows = rows[target_offset::8]
        source_columns = torch.div(selected_rows, 8, rounding_mode="floor")
        source_offsets = selected_rows % 8
        source_bits = torch.tensor(
            [channel_to_bit[int(offset)] for offset in source_offsets],
            dtype=torch.int32)
        values = src.index_select(1, source_columns)
        values = (values >> (4 * source_bits).unsqueeze(0)) & 0xF
        target_bit = channel_to_bit[target_offset]
        packed |= values << (4 * target_bit)

    return packed.to(device=tensor.device, dtype=tensor.dtype)


def _select_pair_packed_int4_rows(tensor: torch.Tensor,
                                  vocab_map: torch.Tensor,
                                  out_features: int) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise ValueError(
            "Reduced vocabulary for ModelOpt INT4 lm_head must run before "
            f"loader repacking, expected uint8 checkpoint weights, got "
            f"{tensor.dtype}.")
    if tuple(tensor.shape)[0] != out_features // 2:
        raise ValueError(
            "Expected pair-packed INT4 tensor shape "
            f"({out_features // 2}, *), got {tuple(tensor.shape)}.")

    rows = vocab_map.detach().cpu().to(torch.long)
    if rows.numel() == 0 or torch.min(rows).item() < 0 or torch.max(
            rows).item() >= out_features:
        raise ValueError("vocab_map contains token IDs outside lm_head "
                         f"out_features={out_features}")
    src = tensor.detach().cpu().to(torch.int16)
    reduced_vocab_size = int(rows.numel())
    packed = torch.zeros(reduced_vocab_size // 2,
                         src.shape[1],
                         dtype=torch.int16)

    for target_offset in range(2):
        selected_rows = rows[target_offset::2]
        source_rows = torch.div(selected_rows, 2, rounding_mode="floor")
        source_shifts = (selected_rows % 2) * 4
        values = src.index_select(0, source_rows)
        values = (values >> source_shifts.unsqueeze(1)) & 0xF
        packed |= values << (4 * target_offset)

    return packed.to(device=tensor.device, dtype=tensor.dtype)


def _select_scale_columns(scales: torch.Tensor,
                          vocab_map: torch.Tensor) -> torch.Tensor:
    return scales.index_select(
        1, vocab_map.to(device=scales.device)).clone().contiguous()


def _reduce_awq_lm_head_before_repacking(lm_head: nn.Module,
                                         vocab_map: torch.Tensor) -> None:
    _validate_int4_checkpoint_reduction(lm_head, vocab_map)
    if lm_head.qweight.dtype != torch.int32:
        raise ValueError(
            "Reduced vocabulary for AWQ lm_head must run before loader "
            f"repacking, expected int32 checkpoint qweight, got "
            f"{lm_head.qweight.dtype}.")
    channel_to_bit = (0, 4, 1, 5, 2, 6, 3, 7)
    _set_buffer(
        lm_head, "qweight",
        _select_column_packed_int4(lm_head.qweight, vocab_map,
                                   lm_head.out_features, channel_to_bit))
    _set_buffer(
        lm_head, "qzeros",
        _select_column_packed_int4(lm_head.qzeros, vocab_map,
                                   lm_head.out_features, channel_to_bit))
    _set_buffer(lm_head, "scales",
                _select_scale_columns(lm_head.scales, vocab_map))
    _select_optional_output_buffer(lm_head, "bias", vocab_map)
    lm_head.out_features = int(vocab_map.numel())


def _reduce_gptq_lm_head_before_repacking(lm_head: nn.Module,
                                          vocab_map: torch.Tensor) -> None:
    _validate_int4_checkpoint_reduction(lm_head, vocab_map)
    if lm_head.qweight.dtype != torch.int32:
        raise ValueError(
            "Reduced vocabulary for GPTQ lm_head must run before loader "
            f"repacking, expected int32 checkpoint qweight, got "
            f"{lm_head.qweight.dtype}.")
    _set_buffer(
        lm_head, "qweight",
        lm_head.qweight.index_select(
            1,
            vocab_map.to(device=lm_head.qweight.device)).clone().contiguous())
    _set_buffer(
        lm_head, "qzeros",
        _select_column_packed_int4(lm_head.qzeros, vocab_map,
                                   lm_head.out_features, tuple(range(8))))
    _set_buffer(lm_head, "scales",
                _select_scale_columns(lm_head.scales, vocab_map))
    _select_optional_output_buffer(lm_head, "bias", vocab_map)
    lm_head.out_features = int(vocab_map.numel())


def _reduce_modelopt_awq_lm_head_before_repacking(
        lm_head: nn.Module, vocab_map: torch.Tensor) -> None:
    _validate_int4_checkpoint_reduction(lm_head, vocab_map)
    weight = lm_head._buffers.get("weight")
    scales = lm_head._buffers.get("weight_scale")
    if weight is None or scales is None:
        raise ValueError("Packed INT4 lm_head is missing 'weight' or "
                         "'weight_scale'.")

    _set_buffer(
        lm_head, "weight",
        _select_pair_packed_int4_rows(weight, vocab_map, lm_head.out_features))
    if tuple(scales.shape) != (lm_head.out_features,
                               lm_head.in_features // lm_head.group_size):
        raise ValueError(f"Packed INT4 lm_head scales must have shape "
                         f"({lm_head.out_features}, "
                         f"{lm_head.in_features // lm_head.group_size}), got "
                         f"{tuple(scales.shape)}.")
    _set_buffer(
        lm_head, "weight_scale",
        scales.index_select(
            0, vocab_map.to(device=scales.device)).clone().contiguous())
    _select_optional_output_buffer(lm_head, "bias", vocab_map)
    lm_head.out_features = int(vocab_map.numel())


def _reduce_row_sliced_lm_head(lm_head: nn.Module,
                               vocab_map: torch.Tensor) -> None:
    _set_buffer(lm_head, "weight",
                _select_output_rows(lm_head.weight, vocab_map))
    _select_optional_output_buffer(lm_head, "weight_scale", vocab_map)
    _select_optional_output_buffer(lm_head, "bias", vocab_map)
    lm_head.out_features = int(vocab_map.numel())


def _reduce_lm_head_in_place(lm_head: nn.Module,
                             vocab_map: torch.Tensor) -> None:
    from ..models.linear import (AWQLinear, FP8Linear, FP16Linear, GPTQLinear,
                                 INT8SQLinear, ModelOptAWQPrepackedLinear,
                                 MXFP8Linear, NVFP4Linear)

    reduced_vocab_size = int(vocab_map.numel())
    if isinstance(lm_head, FP16Linear):
        lm_head.weight = nn.Parameter(_select_output_rows(
            lm_head.weight.data, vocab_map),
                                      requires_grad=False)
        if lm_head.bias is not None:
            lm_head.bias = nn.Parameter(_select_output_rows(
                lm_head.bias.data, vocab_map),
                                        requires_grad=False)
        lm_head.out_features = reduced_vocab_size
        return

    if isinstance(lm_head,
                  (FP8Linear, MXFP8Linear, NVFP4Linear, INT8SQLinear)):
        _reduce_row_sliced_lm_head(lm_head, vocab_map)
        return

    if isinstance(lm_head, AWQLinear):
        _reduce_awq_lm_head_before_repacking(lm_head, vocab_map)
        return

    if isinstance(lm_head, GPTQLinear):
        _reduce_gptq_lm_head_before_repacking(lm_head, vocab_map)
        return

    if isinstance(lm_head, ModelOptAWQPrepackedLinear):
        _reduce_modelopt_awq_lm_head_before_repacking(lm_head, vocab_map)
        return

    raise ValueError("Reduced vocabulary requires in-place lm_head reduction, "
                     f"but {type(lm_head).__name__} is unsupported.")


def _select_optional_output_buffer(module: nn.Module, name: str,
                                   vocab_map: torch.Tensor) -> None:
    value = module._buffers.get(name)
    if value is None:
        return
    if value.dim() > 0 and value.shape[0] == getattr(module, "out_features",
                                                     value.shape[0]):
        _set_buffer(module, name, _select_output_rows(value, vocab_map))
