# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Weight loader for safetensors checkpoints.

Supports both single-file (``model.safetensors``) and multi-shard
(``model.safetensors.index.json`` -> multiple shard files) layouts.

Key design
----------
Rather than ``module.load_state_dict()``, weights are assigned directly to
module buffers/parameters via :func:`_set_tensor`.  This preserves the
original tensor dtype (fp8, uint8, bfloat16, ...) without any silent cast that
PyTorch's state-dict mechanism might introduce, and handles keys not present
in the module (logged as warnings rather than hard errors).

Usage
-----
    model = AutoModel.from_pretrained(model_dir)   # from model.py
    load_weights(model, model_dir)                 # fills all buffers in-place
"""

import json
import logging
import os
from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open

from .repacking import apply_all_repacking

logger = logging.getLogger(__name__)

__all__ = ["load_weights", "load_submodule_weights"]

_FUSED_INPUT_CHANNEL_ATTRS = {"pre_quant_scale"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_weights(
        model: nn.Module,
        model_dir: str,
        device: str = "cpu",
        key_remap: "Optional[Callable[[str], Optional[str]]]" = None,
        key_prefix: Optional[str] = None,
        pre_repack_hook: Optional[Callable[[nn.Module], None]] = None) -> None:
    """Load all safetensors weights from *model_dir* into *model* in-place.

    Args:
        model:      Module built by :meth:`~tensorrt_edgellm.model.AutoModel.from_pretrained`.
        model_dir:  Checkpoint directory that contains safetensors files.
        device:     Target device (e.g. ``"cpu"``, ``"cuda:0"``).
                    Tensors are moved here after loading.
        key_remap:  Optional callable ``(key: str) -> Optional[str]``.
                    Called on each checkpoint key *after* stripping the prefix.
                    Return a new key to remap, the original key unchanged, or
                    ``None`` to skip the tensor entirely.
        key_prefix: Explicit checkpoint key prefix to strip (e.g.
                    ``"talker."`` or ``"talker.code_predictor."``).
                    When provided, only keys starting with this prefix are
                    loaded and auto-detection via :func:`_detect_key_prefix`
                    is skipped.
        pre_repack_hook:
                    Optional callback invoked after raw checkpoint tensors are
                    loaded and before quantized weights are repacked.
    """
    shard_map = _build_shard_map(model_dir)
    # Group keys by shard path to open each shard only once
    path_to_keys: Dict[str, list] = {}
    for key, path in shard_map.items():
        path_to_keys.setdefault(path, []).append(key)

    if key_prefix is not None:
        strip_prefix, insert_prefix = key_prefix, ""
    else:
        # Auto-detect a common VL wrapper prefix (e.g. "language_model." for
        # InternVL) and strip it so the LLM weights map to our module tree.
        all_keys = list(shard_map.keys())
        strip_prefix, insert_prefix = _detect_key_prefix(all_keys)
    if strip_prefix:
        logger.info(
            "Stripping key prefix %r from checkpoint keys (inserting %r)",
            strip_prefix, insert_prefix)

    def _apply_prefix(key: str) -> Optional[str]:
        if strip_prefix and key.startswith(strip_prefix):
            return insert_prefix + key[len(strip_prefix):]
        # When an explicit key_prefix was given, skip keys outside the prefix.
        if key_prefix is not None:
            return None
        return key

    loaded = skipped = 0
    for shard_path, keys in path_to_keys.items():
        if shard_path.endswith(".bin"):
            # PyTorch pickle shard -- load all at once, then iterate keys.
            bin_state = torch.load(shard_path,
                                   map_location=device,
                                   weights_only=True)
            for key in keys:
                tensor = bin_state.get(key)
                if tensor is None:
                    logger.debug("Key not found in .bin shard: %s", key)
                    skipped += 1
                    continue
                mapped_key = _apply_prefix(key)
                if mapped_key is None:
                    skipped += 1
                    continue
                if key_remap is not None:
                    mapped_key = key_remap(mapped_key)
                    if mapped_key is None:
                        skipped += 1
                        continue
                if _set_tensor(model, mapped_key, tensor):
                    loaded += 1
                elif _try_split_fused_tensor(model, mapped_key, tensor):
                    loaded += 1
                else:
                    logger.debug("Key not found in model: %s", key)
                    skipped += 1
        else:
            with safe_open(shard_path, framework="pt", device=device) as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    mapped_key = _apply_prefix(key)
                    if mapped_key is None:
                        skipped += 1
                        continue
                    if key_remap is not None:
                        mapped_key = key_remap(mapped_key)
                        if mapped_key is None:
                            skipped += 1
                            continue
                    if _set_tensor(model, mapped_key, tensor):
                        loaded += 1
                    elif _try_split_fused_tensor(model, mapped_key, tensor):
                        loaded += 1
                    else:
                        logger.debug("Key not found in model: %s", key)
                        skipped += 1

    logger.info("Loaded %d tensors, skipped %d from %s", loaded, skipped,
                model_dir)

    if pre_repack_hook is not None:
        pre_repack_hook(model)

    apply_all_repacking(model)
    # Post-process: apply tied embeddings (HF tie_word_embeddings=True models
    # omit lm_head.weight from the checkpoint; tie_weights() restores the share).
    config = getattr(model, "config", None)
    if (hasattr(model, "tie_weights") and config is not None
            and getattr(config, "tie_word_embeddings", False)):
        from ..models.linear import FP16Linear
        if isinstance(getattr(model, "lm_head", None), FP16Linear):
            model.tie_weights()
            logger.info("Tied lm_head.weight to embed_tokens.weight")
        else:
            logger.debug(
                "Skipping tied lm_head.weight for non-FP16 lm_head type %s",
                type(getattr(model, "lm_head", None)).__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_key_prefix(keys: list) -> Tuple[str, str]:
    """Return ``(strip_prefix, insert_prefix)`` for remapping checkpoint keys.

    Detects VL wrapper prefixes so that LLM weights map into our module tree.
    After stripping ``strip_prefix`` from each key, ``insert_prefix`` is
    prepended before setting the tensor.

    Examples:
      InternVL3 keys: ``language_model.model.embed_tokens.weight``
        → strip ``"language_model."`` → ``"model.embed_tokens.weight"`` ✓

      Qwen3-VL-2B keys: ``model.language_model.embed_tokens.weight``
        → strip ``"model.language_model."`` + insert ``"model."``
        → ``"model.embed_tokens.weight"`` ✓
    """
    key_set = set(keys)
    for prefix in ("language_model.", "text_model.", "llm."):
        if (f"{prefix}model.embed_tokens.weight" in key_set
                or any(k.startswith(f"{prefix}model.layers.0.") for k in keys)
                or any(
                    k.startswith(f"{prefix}backbone.layers.0.")
                    for k in keys)):
            return prefix, ""
    # Qwen3-VL-2B: text decoder is under "model.language_model.*".
    # Strip the outer "model.language_model." and prepend "model." so that
    # the stripped key (e.g. "embed_tokens.weight") becomes
    # "model.embed_tokens.weight" matching the CausalLM parameter tree.
    if (any(k.startswith("model.language_model.") for k in keys)
            and "model.embed_tokens.weight" not in key_set):
        return "model.language_model.", "model."
    # Alpamayo-R1: LLM text decoder is under "vlm.model.language_model.*".
    # lm_head at "vlm.lm_head.*" falls through and requires key_remap.
    if any(k.startswith("vlm.model.language_model.") for k in keys):
        return "vlm.model.language_model.", "model."
    # Qwen3-ASR / Qwen3-Omni: LLM weights are under thinker.model.*
    if (any(k.startswith("thinker.model.layers.0.") for k in keys)
            and not any(k.startswith("model.layers.0.") for k in keys)):
        return "thinker.", ""
    # Qwen3-TTS: all weights are under talker.* (talker is the LLM backbone)
    if (any(k.startswith("talker.model.layers.0.") for k in keys)
            and not any(k.startswith("model.layers.0.") for k in keys)):
        return "talker.", ""
    return "", ""


def _build_shard_map(model_dir: str) -> Dict[str, str]:
    """Return a mapping of weight-key -> absolute shard file path.

    Handles:
      • ``model.safetensors``             (single file, no index)
      • ``model.safetensors.index.json``  (multi-shard, weight_map inside)
      • ``pytorch_model.bin``             (single PyTorch pickle file)
      • ``pytorch_model.bin.index.json``  (multi-shard PyTorch pickle)
    """
    # ---- safetensors (preferred) ----------------------------------------
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map: Dict[str, str] = index["weight_map"]
        return {
            key: os.path.join(model_dir, shard)
            for key, shard in weight_map.items()
        }

    if os.path.exists(single_path):
        keys: Dict[str, str] = {}
        with safe_open(single_path, framework="pt") as f:
            for key in f.keys():
                keys[key] = single_path
        return keys

    # ---- PyTorch pickle (.bin) fallback ---------------------------------
    bin_index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
    bin_single_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(bin_index_path):
        with open(bin_index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        return {
            key: os.path.join(model_dir, shard)
            for key, shard in weight_map.items()
        }

    if os.path.exists(bin_single_path):
        # Load with map_location="meta" to get key names without loading data.
        meta = torch.load(bin_single_path,
                          map_location="meta",
                          weights_only=True)
        return {key: bin_single_path for key in meta.keys()}

    raise FileNotFoundError(
        f"No checkpoint files found in {model_dir!r}. "
        "Expected 'model.safetensors', 'model.safetensors.index.json', "
        "'pytorch_model.bin', or 'pytorch_model.bin.index.json'.")


def _navigate(model: nn.Module, parts: list) -> Tuple[nn.Module, str]:
    """Walk the module tree following *parts* and return (leaf_module, attr).

    Integer parts are interpreted as ``nn.ModuleList`` indices.

    Raises:
        AttributeError: if any part of the path is not found.
    """
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module, parts[-1]


def _set_tensor(model: nn.Module, key: str, tensor: torch.Tensor) -> bool:
    """Assign *tensor* to the buffer or parameter at *key* inside *model*.

    Bfloat16 tensors are cast to float16 on the fly — the export pipeline
    assumes FP16 activations and the C++ runtime requires FP16 (or FP8)
    weight files.  Doing the cast here avoids a separate post-loading sweep.

    Returns True on success, False if the key does not resolve to a known
    buffer or parameter.
    """
    parts = key.split(".")
    try:
        module, attr = _navigate(model, parts)
    except (AttributeError, IndexError, TypeError):
        return False

    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)

    if attr in module._buffers:
        module._buffers[attr] = tensor
        return True
    if attr in module._parameters:
        module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
        return True

    # Key resolves to a module-level attribute that is neither buffer nor
    # parameter (e.g. 'weight' on nn.Embedding before load).
    if hasattr(module, attr):
        setattr(module, attr, tensor)
        return True

    return False


def _try_split_fused_tensor(model: nn.Module, key: str,
                            tensor: torch.Tensor) -> bool:
    """Handle fused-weight checkpoint patterns not matched by _set_tensor.

    Supports two transformations applied in order:
    1. ``.base_layer.`` removal — PEFT/LoRA checkpoints nest the base weight
       under ``module.base_layer.weight``. Strip ``.base_layer`` to recover the
       plain ``module.weight`` name before trying again.
    2. Fused QKV split — ``self_attn.qkv_proj.weight`` (shape
       ``[q+k+v, hidden]``) is split into the three separate projection weights
       ``q_proj``, ``k_proj``, ``v_proj`` using the model's attention head
       configuration.
    3. Fused gate+up split — ``mlp.gate_up_proj.weight`` (shape
       ``[2*intermediate, hidden]``) is split into ``gate_proj`` and ``up_proj``
       (each half of the first dimension).

    Returns True if at least one split sub-tensor was set successfully.
    """
    # --- 1. Strip PEFT base_layer prefix ------------------------------------
    if ".base_layer." in key:
        stripped = key.replace(".base_layer.", ".")
        if _set_tensor(model, stripped, tensor):
            return True
        # Still failed: fall through to fused-split checks with the stripped key
        key = stripped

    # --- 2. Fused QKV split -------------------------------------------------
    if ".self_attn.qkv_proj." in key:
        config = getattr(model, "config", None)
        if config is None:
            return False
        # Extract layer prefix and attribute suffix (e.g. "weight", "weight_scale")
        qkv_idx = key.index(".self_attn.qkv_proj.")
        prefix = key[:qkv_idx]
        attr_suffix = key[qkv_idx + len(".self_attn.qkv_proj."):]

        num_q = config.num_attention_heads * config.head_dim
        num_kv = config.num_key_value_heads * config.head_dim

        # Scalar, per-tensor, or per-input-channel attributes:
        # copy the same value to all three projections.
        if (tensor.dim() == 0 or (tensor.dim() == 1 and tensor.shape[0] <= 1)
                or attr_suffix in _FUSED_INPUT_CHANNEL_ATTRS):
            ok = _set_tensor(model, f"{prefix}.self_attn.q_proj.{attr_suffix}",
                             tensor)
            ok |= _set_tensor(model,
                              f"{prefix}.self_attn.k_proj.{attr_suffix}",
                              tensor)
            ok |= _set_tensor(model,
                              f"{prefix}.self_attn.v_proj.{attr_suffix}",
                              tensor)
            if ok:
                logger.debug("Broadcast qkv_proj.%s → q/k/v for prefix %r",
                             attr_suffix, prefix)
            return ok

        # Per-output-channel attributes (weight, weight_scale): split dim 0.
        split_sizes = [num_q, num_kv, num_kv]
        expected = num_q + 2 * num_kv
        if attr_suffix == "weight" and tensor.dim(
        ) >= 2 and tensor.shape[0] * 2 == expected:
            # ModelOpt W4A16 stores packed int4 weights as [out/2, in].
            split_sizes = [s // 2 for s in split_sizes]
            expected //= 2
        if tensor.shape[0] != expected:
            logger.warning(
                "qkv_proj.%s shape %s doesn't match expected (%d, %d, %d) "
                "— skipping split", attr_suffix, tensor.shape, *split_sizes)
            return False
        q, k, v = tensor.split(split_sizes, dim=0)
        ok = _set_tensor(model, f"{prefix}.self_attn.q_proj.{attr_suffix}", q)
        ok |= _set_tensor(model, f"{prefix}.self_attn.k_proj.{attr_suffix}", k)
        ok |= _set_tensor(model, f"{prefix}.self_attn.v_proj.{attr_suffix}", v)
        if ok:
            logger.debug("Split qkv_proj.%s → q/k/v for prefix %r",
                         attr_suffix, prefix)
        return ok

    # --- 3. Fused gate+up split --------------------------------------------
    if ".mlp.gate_up_proj." in key:
        gate_up_idx = key.index(".mlp.gate_up_proj.")
        prefix = key[:gate_up_idx]
        attr_suffix = key[gate_up_idx + len(".mlp.gate_up_proj."):]

        # Scalar, per-tensor, or per-input-channel attributes: copy to both.
        if (tensor.dim() == 0 or (tensor.dim() == 1 and tensor.shape[0] <= 1)
                or attr_suffix in _FUSED_INPUT_CHANNEL_ATTRS):
            ok = _set_tensor(model, f"{prefix}.mlp.gate_proj.{attr_suffix}",
                             tensor)
            ok |= _set_tensor(model, f"{prefix}.mlp.up_proj.{attr_suffix}",
                              tensor)
            if ok:
                logger.debug(
                    "Broadcast gate_up_proj.%s → gate/up for prefix %r",
                    attr_suffix, prefix)
            return ok

        # Per-output-channel attributes: split in half on dim 0.
        half = tensor.shape[0] // 2
        gate, up = tensor[:half], tensor[half:]
        ok = _set_tensor(model, f"{prefix}.mlp.gate_proj.{attr_suffix}", gate)
        ok |= _set_tensor(model, f"{prefix}.mlp.up_proj.{attr_suffix}", up)
        if ok:
            logger.debug("Split gate_up_proj.%s → gate/up for prefix %r",
                         attr_suffix, prefix)
        return ok

    return False


def iter_checkpoint_keys(model_dir: str) -> Iterator[str]:
    """Yield all weight keys present in a checkpoint directory (no data)."""
    shard_map = _build_shard_map(model_dir)
    yield from shard_map.keys()


def load_submodule_weights(
    model: nn.Module,
    weights: Dict[str, torch.Tensor],
    key_remap: Callable[[str], Optional[str]],
    *,
    transform: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None,
    label: str = "model",
    log: Optional[logging.Logger] = None,
    do_repack: bool = True,
) -> None:
    """Load a sliced ``{key: tensor}`` dict into a sub-encoder via ``_set_tensor``.

    Used by visual / audio modeling files that receive a flat weights dict
    (already loaded from safetensors by the orchestrator) and need to filter
    and rename keys before assignment.  Sharing this helper avoids duplicating
    the iterate-remap-set-tensor-track-missing-log pattern across families.

    ``_set_tensor`` is used (not ``load_state_dict``) so that:
      - ``bfloat16 -> float16`` cast happens automatically;
      - quantized weights (``float8_e4m3fn`` / packed int8 / ...) keep their
        original dtype rather than being silently cast.

    :param model: Target sub-encoder module.
    :param weights: Flat ``{key: tensor}`` dict from safetensors.
    :param key_remap: Called per checkpoint key. Return the new path inside
        ``model`` (e.g. ``"encoder.blocks.0.attn.weight"``), or ``None`` to
        skip the key. Use this to strip / rewrite checkpoint prefixes.
    :param transform: Optional ``(remapped_key, tensor) -> tensor`` hook for
        per-tensor reshaping (e.g. flat → conv2d) or interpolation.
    :param label: Used in the missing-keys warning ("<label>: keys not loaded").
    :param log: Logger used for the missing-keys warning. Defaults to this
        module's logger.
    :param do_repack: When ``True`` (default), call ``apply_all_repacking``
        after assignment so type-aware fixups (FP8 scale cast, NVFP4 view-cast,
        AWQ/GPTQ swizzle, ...) run.  Safe to leave on for non-quantized
        sub-encoders — each fixup is gated by an ``isinstance`` check.
    """
    if log is None:
        log = logger
    missing: list[str] = []
    for k, v in weights.items():
        new_key = key_remap(k)
        if new_key is None:
            continue
        if transform is not None:
            v = transform(new_key, v)
        if not _set_tensor(model, new_key, v):
            missing.append(new_key)
    if do_repack:
        apply_all_repacking(model)
    if missing:
        log.warning(
            "%s: keys not loaded: %s%s",
            label,
            missing[:10],
            " ..." if len(missing) > 10 else "",
        )
