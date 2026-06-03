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
"""Joint Qwen3-ASR calibration model factory.

Qwen3-ASR's HF checkpoint declares ``model_type="qwen3_asr"`` and ships
no in-tree modeling code, so the vanilla ``AutoModel`` factories cannot
load it directly. The text decoder under ``thinker_config.text_config``
is a standard Qwen3 transformer though (q/k norm, GQA, SwiGLU, tied
word embeddings; the MRoPE bit is multimodal-only and irrelevant for
text-or-audio-spliced calibration), and the audio encoder is the
pure-PyTorch :class:`Qwen3ASRAudioEncoder` in the sibling module.

:meth:`Qwen3ASRForConditionalGeneration.from_pretrained` builds a
joint calibration model: a vanilla :class:`transformers.Qwen3ForCausalLM`
with the audio encoder attached as an ``audio_tower`` submodule plus a
custom forward that runs the audio tower, splices its output into the
text input-embedding stream at ``<|audio_pad|>`` positions, and forwards
through the LLM. ModelOpt's ``mtq.quantize`` walks the resulting module
tree, so a single :func:`_calibrate_asr_multimodal` loop drives both
halves under realistic ASR-shaped activations -- the equivalent of
``_calibrate_multimodal`` for VLMs.

The "class" is intentionally a factory namespace, not an ``nn.Module``
subclass: ``from_pretrained`` returns the augmented
``Qwen3ForCausalLM`` instance directly. Keeping the saved state_dict
layout identical to a vanilla ``Qwen3ForCausalLM`` (``model.embed_tokens.*``,
``lm_head.*``, plus ``audio_tower.*`` as a sibling submodule) means
:func:`postprocess_qwen3_asr_checkpoint` can do the ``thinker.``
re-prefix step with no special-casing. A follow-up may flip this to a
proper composition wrapper once the new modeling has soaked.
"""

from __future__ import annotations

import json
import logging
import os
import types
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from .modeling_qwen3_asr_audio import build_qwen3_asr_audio

logger = logging.getLogger(__name__)


def _load_safetensors_maybe_sharded(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load all weights from a Hugging Face ckpt dir into a single dict.

    Handles both single-file (``model.safetensors``) and multi-shard
    (``model.safetensors.index.json`` + ``model-*-of-*.safetensors``)
    layouts.
    """
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return load_file(single, device="cpu")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Expected {single!r} or {index_path!r} in {model_dir!r}.")
    with open(index_path) as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]
    merged: Dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        shard_path = os.path.join(model_dir, shard)
        merged.update(load_file(shard_path, device="cpu"))
    return merged


def _sanitise_text_config(text_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert qwen3_asr's ``thinker_config.text_config`` into kwargs the
    HF ``Qwen3Config`` accepts.

    Drops MRoPE-specific fields HF Qwen3 does not know about, and translates
    the flat ``rope_theta`` / ``rope_scaling`` representation into the
    nested ``rope_parameters`` dict ``transformers >= 5`` expects. For
    text-only or audio-spliced calibration, the standard 1-D RoPE is what
    gets exercised -- MRoPE positional bookkeeping is irrelevant.
    """
    out = dict(text_cfg)
    out["model_type"] = "qwen3"
    out.pop("architectures", None)

    rope_theta = out.pop("rope_theta", 10000.0)
    raw_rope = out.pop("rope_scaling", None) or {}
    cleaned = {
        k: v
        for k, v in raw_rope.items() if k not in (
            "mrope_section",
            "mrope_interleaved",
            "interleaved",
            "type",
        )
    }
    rope_type = cleaned.get("rope_type", "default")
    if rope_type == "default":
        out["rope_parameters"] = {
            "rope_theta": rope_theta,
            "rope_type": "default",
        }
    else:
        cleaned.setdefault("rope_theta", rope_theta)
        out["rope_parameters"] = cleaned
    return out


def _asr_joint_forward(self,
                       input_ids: torch.LongTensor = None,
                       audio_padded_feature: Optional[torch.Tensor] = None,
                       audio_indices: Optional[torch.Tensor] = None,
                       audio_attn_mask: Optional[torch.Tensor] = None,
                       **kwargs):
    """Forward used during joint ASR calibration.

    Runs the audio encoder, splices its output into the text token
    embedding stream at ``<|audio_pad|>`` positions, and forwards through
    the underlying Qwen3 text decoder. A single ``<|audio_pad|>`` per row
    in the tokenized prompt expands to the full ``T_audio`` audio-embedding
    sequence -- matches the runtime semantics.

    When audio inputs are ``None`` (e.g. ModelOpt's ``export_hf_checkpoint``
    runs internal text-only dummy forwards during the export flow), falls
    back to the underlying Qwen3 text-only forward unchanged.
    """
    if (audio_padded_feature is None or audio_indices is None
            or audio_attn_mask is None):
        return self._asr_original_forward(input_ids=input_ids, **kwargs)

    audio_embeds = self.audio_tower(audio_padded_feature, audio_indices,
                                    audio_attn_mask)
    inputs_embeds = self.model.embed_tokens(input_ids)

    audio_token_id = self._asr_audio_token_id
    rows: List[torch.Tensor] = []
    for b in range(input_ids.shape[0]):
        pad_positions = (input_ids[b] == audio_token_id).nonzero(
            as_tuple=True)[0]
        if len(pad_positions) == 0:
            rows.append(inputs_embeds[b])
            continue
        pos = int(pad_positions[0].item())
        rows.append(
            torch.cat([
                inputs_embeds[b, :pos],
                audio_embeds.to(inputs_embeds.dtype),
                inputs_embeds[b, pos + 1:],
            ],
                      dim=0))

    max_len = max(r.shape[0] for r in rows)
    hidden = inputs_embeds.shape[-1]
    packed = inputs_embeds.new_zeros((input_ids.shape[0], max_len, hidden))
    attn_mask = inputs_embeds.new_zeros((input_ids.shape[0], max_len),
                                        dtype=torch.long)
    for b, r in enumerate(rows):
        packed[b, :r.shape[0]] = r
        attn_mask[b, :r.shape[0]] = 1

    kwargs.pop("attention_mask", None)
    return self._asr_original_forward(inputs_embeds=packed,
                                      attention_mask=attn_mask,
                                      use_cache=False,
                                      **kwargs)


class Qwen3ASRForConditionalGeneration:
    """Factory namespace for the joint Qwen3-ASR calibration model.

    The factory does NOT define ``__init__`` -- :meth:`from_pretrained`
    builds a real :class:`transformers.Qwen3ForCausalLM` and returns it
    (augmented with an ``audio_tower`` submodule and an ASR-spliced
    forward). Keeping the returned object's class as ``Qwen3ForCausalLM``
    means HF's ``save_pretrained`` / ``_tied_weights_keys`` / ModelOpt's
    ``export_hf_checkpoint`` see exactly what they would for a plain
    text-only Qwen3.
    """

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        torch_dtype: torch.dtype,
                        device: str = "cuda") -> Qwen3ForCausalLM:
        """Build the joint calibration model.

        Args:
            model_dir:  Path to a directory containing ``config.json``
                and either ``model.safetensors`` (single-file) or a
                ``model.safetensors.index.json`` + shards.
            torch_dtype: dtype for the text decoder and the audio encoder
                FP16 parameters.
            device:     Target device for the assembled model.

        Returns:
            A ``Qwen3ForCausalLM`` whose ``forward`` has been replaced by
            :func:`_asr_joint_forward`. The instance carries:

              * ``audio_tower`` -- :class:`Qwen3ASRAudioEncoder` submodule.
              * ``_asr_audio_token_id`` -- the ``<|audio_pad|>`` token id.
              * ``_asr_original_forward`` -- the unpatched Qwen3 forward,
                used by the splice path as the text-decoder fallback.
        """
        with open(os.path.join(model_dir, "config.json")) as f:
            full_cfg = json.load(f)
        text_cfg_raw = full_cfg.get("thinker_config", {}).get("text_config")
        if text_cfg_raw is None:
            raise ValueError(f"{model_dir}/config.json does not contain "
                             "thinker_config.text_config")
        sanitised = _sanitise_text_config(text_cfg_raw)
        qwen_cfg = Qwen3Config(**sanitised)
        qwen_cfg.architectures = ["Qwen3ForCausalLM"]

        model = Qwen3ForCausalLM(qwen_cfg).to(torch_dtype).to(device)

        weights = _load_safetensors_maybe_sharded(model_dir)

        text_state: Dict[str, torch.Tensor] = {}
        skipped_audio = 0
        for k, v in weights.items():
            if k.startswith("thinker.audio_tower."):
                skipped_audio += 1
                continue
            if k.startswith("thinker."):
                text_state[k[len("thinker."):]] = v.to(torch_dtype)

        missing, unexpected = model.load_state_dict(text_state, strict=False)
        if missing:
            logger.warning(
                "Qwen3-ASR text load: %d missing keys (first 5: %s)",
                len(missing), missing[:5])
        if unexpected:
            logger.warning(
                "Qwen3-ASR text load: %d unexpected keys (first 5: %s)",
                len(unexpected), unexpected[:5])
        logger.info(
            "Qwen3-ASR text load: %d text tensors loaded, %d audio tower "
            "tensors deferred to audio_tower load below", len(text_state),
            skipped_audio)

        audio_tower = build_qwen3_asr_audio(
            config=full_cfg,
            weights=weights,
            dtype=torch_dtype,
            prefix="thinker.audio_tower.",
        ).to(device)

        # Attach as a submodule of the Qwen3ForCausalLM so mtq.quantize
        # walks it as part of the unified module tree, and the resulting
        # state_dict carries audio_tower.* as a top-level sibling of
        # model.* / lm_head.* (no extra prefix to strip later).
        model.audio_tower = audio_tower

        # Cache the audio_pad token id for the splice. Qwen3-ASR-0.6B uses
        # 151676 (matches ``<|audio_pad|>`` in the tokenizer).
        audio_token_id = full_cfg.get("thinker_config",
                                      {}).get("audio_token_id", 151676)
        model._asr_audio_token_id = int(audio_token_id)

        # Monkey-patch forward in place; stash the original for the splice
        # path to delegate into.
        model._asr_original_forward = model.forward
        model.forward = types.MethodType(_asr_joint_forward, model)

        return model


__all__ = ["Qwen3ASRForConditionalGeneration"]
