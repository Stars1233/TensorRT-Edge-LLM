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
Auto-dispatch model factory and parameter utilities.

``AutoModel.from_pretrained`` reads a checkpoint config, picks the right model
class, constructs it, and loads weights — the primary entry point for callers.

Custom model classes can be registered via :func:`register_model` to override
the default :class:`~models.default.modeling_default.CausalLM` for a given
``model_type`` string.
"""

from typing import Dict, Type

import torch.nn as nn

from .checkpoint.loader import load_weights
from .config import ModelConfig, make_mtp_draft_config

__all__ = ["AutoModel", "register_model", "dtype_summary", "param_count"]

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str, model_class: Type[nn.Module]) -> None:
    """Register *model_class* as the handler for *model_type*.

    When :meth:`AutoModel.from_pretrained` encounters a checkpoint whose
    ``model_type`` field equals *model_type*, it instantiates *model_class*
    instead of the built-in :class:`~models.default.modeling_default.CausalLM`.

    Args:
        model_type:  Value of ``model_type`` in the checkpoint ``config.json``.
        model_class: ``nn.Module`` subclass; must accept a single
                     :class:`~config.ModelConfig` as its constructor argument.
    """
    _MODEL_REGISTRY[model_type] = model_class


class AutoModel:
    """HuggingFace-style factory that dispatches on ``model_type``."""

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        device: str = "cpu",
                        key_remap=None,
                        key_prefix: "str | None" = None,
                        eagle_base: bool = False,
                        reduced_vocab_dir: "str | None" = None,
                        nvfp4_moe_backend: "str | None" = None,
                        mtp_base: bool = False,
                        mtp_draft: bool = False) -> nn.Module:
        """Construct and load a model from *model_dir*.

        Reads ``config.json`` via :class:`~config.ModelConfig`, looks up the
        model class in the registry (falling back to the built-in
        :class:`~models.default.modeling_default.CausalLM`), instantiates it,
        moves it to *device*, and loads safetensors weights.

        Args:
            model_dir:      Local HF checkpoint directory.
            device:         Target device (e.g. ``"cpu"``, ``"cuda:0"``).
            key_remap:      Optional callable ``(key: str) -> Optional[str]``.
                            Passed through to :func:`load_weights` for checkpoint
                            key remapping (e.g. TTS talker ``codec_embedding``
                            → ``embed_tokens``).
            key_prefix:     Explicit checkpoint key prefix to strip (e.g.
                            ``"talker."``).  Passed through to :func:`load_weights`.
            eagle_base:     When True, export as EAGLE3 base model with extra
                            tree-attention inputs and hidden_states output.
            reduced_vocab_dir:
                            Optional directory containing ``vocab_map.safetensors``.
            nvfp4_moe_backend:
                            Optional export-time override for Qwen3 NVFP4 MoE
                            backend selection (``"thor"`` or ``"geforce"``).
            mtp_base:       When True, export the standard Qwen3.5 text model as
                            the dense MTP base variant.
            mtp_draft:      When True, build the dedicated Qwen3.5 dense MTP
                            draft model from the base checkpoint config.

        Returns:
            Loaded ``nn.Module`` in eval mode.
        """
        from .models.default.modeling_default import CausalLM

        config = ModelConfig.from_pretrained(model_dir)
        if nvfp4_moe_backend is not None:
            config.quant.nvfp4_moe_backend = nvfp4_moe_backend
            config.quant.__post_init__()
        if eagle_base:
            config.eagle_base = True
        if mtp_base or config.mtp_base:
            config.mtp_base = True

        variant = _resolve_model_variant(config,
                                         eagle_base=eagle_base,
                                         mtp_base=config.mtp_base,
                                         mtp_draft=mtp_draft)

        # EAGLE3 draft: auto-detect from draft_vocab_size
        if variant == "eagle3_draft":
            from .models.eagle3.modeling_eagle3_draft import Eagle3DraftModel
            model_class = Eagle3DraftModel
            # Set up key remapping: midlayer -> layers.0, skip t2d
            if key_remap is None:
                key_remap = _eagle3_key_remap
        elif variant == "mtp_draft":
            # TODO: support other model types
            if config.model_type != "qwen3_5_text":
                raise NotImplementedError(
                    "MTP draft is only supported for qwen3_5_text checkpoints."
                )
            from .models.qwen3_5 import Qwen3_5MtpDraftModel
            tie_word_embeddings = config.tie_word_embeddings
            config = make_mtp_draft_config(config)
            model_class = Qwen3_5MtpDraftModel
            if key_remap is None:
                key_remap = lambda key: _mtp_key_remap(
                    key, tie_word_embeddings=tie_word_embeddings)
        else:
            if variant == "mtp_base" and config.model_type != "qwen3_5_text":
                raise NotImplementedError(
                    "Qwen3.5 dense MTP base is only supported for qwen3_5_text checkpoints."
                )
            model_class = _MODEL_REGISTRY.get(config.model_type, CausalLM)

        model = model_class(config)
        model.to(device)

        pre_repack_hook = None
        if reduced_vocab_dir is not None:
            from .vocab_reduction.onnx_export import (
                apply_reduced_vocab, load_reduced_vocab_map,
                should_apply_reduced_vocab_before_repacking)
            vocab_map = load_reduced_vocab_map(reduced_vocab_dir,
                                               vocab_size=config.vocab_size,
                                               device=device)
            if should_apply_reduced_vocab_before_repacking(model):

                def _apply_pre_repack_reduced_vocab(loaded_model: nn.Module):
                    apply_reduced_vocab(loaded_model, vocab_map)
                    loaded_model._reduced_vocab_dir = reduced_vocab_dir

                pre_repack_hook = _apply_pre_repack_reduced_vocab

        load_weights(model,
                     model_dir,
                     device=device,
                     key_remap=key_remap,
                     key_prefix=key_prefix,
                     pre_repack_hook=pre_repack_hook)
        if reduced_vocab_dir is not None and pre_repack_hook is None:
            from .vocab_reduction.onnx_export import \
                apply_reduced_vocab_from_dir
            apply_reduced_vocab_from_dir(model, reduced_vocab_dir)

        # Post-load optimisation: fuse GDN input projections for Qwen3.5 / Qwen3.5-MoE.
        if (config.model_type in ("qwen3_5_text", "qwen3_5_moe_text")
                and not mtp_draft):
            from .models.qwen3_5 import fuse_gdn_input_projections
            fuse_gdn_input_projections(model)

        return model


def param_count(model: nn.Module) -> int:
    """Return total parameter element count (trainable and frozen)."""
    return sum(p.numel() for p in model.parameters())


def dtype_summary(model: nn.Module) -> Dict[str, int]:
    """Map dtype name -> number of parameter elements."""
    out: Dict[str, int] = {}
    for p in model.parameters():
        name = str(p.dtype).replace("torch.", "")
        out[name] = out.get(name, 0) + p.numel()
    return dict(sorted(out.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# EAGLE3 helpers
# ---------------------------------------------------------------------------


def _resolve_model_variant(config: ModelConfig, *, eagle_base: bool,
                           mtp_base: bool, mtp_draft: bool) -> str:
    """Resolve the requested model variant while keeping EAGLE3 behavior intact."""
    if eagle_base and mtp_base:
        raise ValueError("eagle_base and mtp_base cannot both be enabled.")
    if eagle_base and mtp_draft:
        raise ValueError("eagle_base and mtp_draft cannot both be enabled.")
    if mtp_base and mtp_draft:
        raise ValueError("mtp_base and mtp_draft cannot both be enabled.")
    if config.is_eagle3_draft:
        if mtp_base or mtp_draft:
            raise ValueError(
                "EAGLE3 draft checkpoints cannot be loaded as Qwen3.5 MTP variants."
            )
        return "eagle3_draft"
    if mtp_draft:
        return "mtp_draft"
    if mtp_base:
        return "mtp_base"
    if eagle_base:
        return "eagle_base"
    return "llm"


def _eagle3_key_remap(key: str) -> "str | None":
    """Remap EAGLE3 draft checkpoint keys.

    Handles all known EAGLE3 draft checkpoint variations:
    - ``t2d`` keys are skipped (but ``d2t`` is kept).
    - ``target_model.*`` keys are skipped (multi-target training artifact).
    - ``midlayer.*`` -> ``layers.0.*``
    - ``qkv_proj.{q,k,v}_proj`` -> ``{q,k,v}_proj`` (flatten old pipeline
      ``EdgeLLMAttention`` wrapper nesting, used by quantized checkpoints).
    - ``._pre_quant_scale`` -> ``.pre_quant_scale`` (modelopt internal naming;
      normally stripped by ``postprocess_state_dict()`` but not by per-module
      export via ``_export_quantized_weight()``).
    """
    if "t2d" in key and "d2t" not in key:
        return None  # skip t2d
    if key.startswith("target_model."):
        return None  # skip multi-target training artifact
    key = key.replace("midlayer.", "layers.0.")
    key = key.replace("qkv_proj.q_proj", "q_proj")
    key = key.replace("qkv_proj.k_proj", "k_proj")
    key = key.replace("qkv_proj.v_proj", "v_proj")
    key = key.replace("._pre_quant_scale", ".pre_quant_scale")
    return key


def _mtp_key_remap(key: str, *, tie_word_embeddings: bool) -> "str | None":
    """Remap MTP checkpoint keys for the draft model.

    The embedding table is only a valid LM-head fallback when the source
    checkpoint ties word embeddings.
    """
    if key.startswith("mtp."):
        return key[len("mtp."):]
    if key == "lm_head.weight":
        return key
    if tie_word_embeddings and key in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight"):
        return "lm_head.weight"
    return None
