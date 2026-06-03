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
"""Qwen3-ASR quantization helpers.

Qwen3-ASR's HF checkpoint declares ``model_type="qwen3_asr"`` but ships
no in-tree HF modeling code, so the vanilla ``AutoModel`` factories in
:func:`tensorrt_edgellm.quantization.quantize._load_model` raise
``ValueError: Unrecognized configuration class``. This module wraps the
standalone joint calibration model in
:mod:`tensorrt_edgellm.quantization.models.qwen3_asr` with the bookkeeping
``_load_model`` expects, plus a LibriSpeech audio+transcript calibration
dataloader and a post-process step to re-prefix the saved unified
checkpoint back to the qwen3_asr namespace.

The actual modeling (audio encoder, joint factory) lives under
:mod:`tensorrt_edgellm.quantization.models.qwen3_asr` -- pure ``nn.Linear``
throughout, no imports from :mod:`tensorrt_edgellm`. The runtime
side has its own quant-aware modeling tree under
:mod:`tensorrt_edgellm.models.qwen3_asr` for ONNX export.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
from typing import Dict, Iterator

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .models.qwen3_asr import Qwen3ASRForConditionalGeneration
from .models.qwen3_asr.modeling_qwen3_asr_audio import prepare_audio_inputs

logger = logging.getLogger(__name__)

# Default config knobs for the LibriSpeech calibration loader so
# ``--dataset openslr/librispeech_asr`` works for ASR calibration.
_DEFAULT_LIBRISPEECH_NAME = "openslr/librispeech_asr"
_DEFAULT_LIBRISPEECH_CONFIG = "clean"
_DEFAULT_LIBRISPEECH_SPLIT = "train.100"


def is_qwen3_asr_model(model_dir: str) -> bool:
    """Return True if *model_dir* holds a qwen3_asr HF checkpoint."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            return json.load(f).get("model_type") == "qwen3_asr"
    except (OSError, ValueError):
        return False


def load_qwen3_asr_joint_for_calibration(model_dir: str, torch_dtype, device):
    """Build a joint Qwen3-ASR calibration model.

    Thin wrapper around
    :meth:`Qwen3ASRForConditionalGeneration.from_pretrained` that also
    builds the tokenizer, matching the ``(model, tokenizer, processor)``
    contract of :func:`tensorrt_edgellm.quantization.quantize._load_model`.

    ``processor`` is always ``None`` -- the ASR calibration loader builds
    its own tokenized inputs from raw audio + transcripts, so no HF
    processor is required.
    """
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    return model, tokenizer, None


def asr_calibration_dataloader(
    tokenizer,
    audio_token_id: int,
    audio_n_window: int,
    num_mel_bins: int = 128,
    dataset_name: str = _DEFAULT_LIBRISPEECH_NAME,
    dataset_config: str = _DEFAULT_LIBRISPEECH_CONFIG,
    dataset_split: str = _DEFAULT_LIBRISPEECH_SPLIT,
    num_samples: int = 128,
    max_audio_seconds: float = 20.0,
    sample_rate: int = 16000,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Stream LibriSpeech (audio, transcript) pairs as joint-forward batches.

    Yields one batch dict per sample; each dict is consumable directly by
    the joint forward in
    :meth:`Qwen3ASRForConditionalGeneration.from_pretrained` (and by
    :func:`tensorrt_edgellm.quantization.quantize._calibrate_asr_multimodal`).

    The chat-template prompt mirrors the runtime layout (system / user
    with a single ``<|audio_pad|>`` between ``<|audio_start|>`` and
    ``<|audio_end|>`` / assistant prefix), with the assistant content
    filled in with the ground-truth transcript so the LLM's calibration
    sees realistic generation activations as well as prompt-side ones.
    """
    import soundfile as sf
    from datasets import Audio, load_dataset

    # Reuse the shared mel extractor instead of redoing the Whisper
    # feature-extractor wiring here; the runtime preprocess (Step 4 of
    # the ASR example) uses the same helper, so calibration and runtime
    # match by construction. ``num_mel_bins`` is the Whisper default
    # already (128) so we pass nothing.
    from tensorrt_edgellm.scripts.preprocess_audio import \
        extract_mel_spectrogram

    # ``decode=False`` avoids torchcodec; we decode bytes with soundfile.
    stream = load_dataset(dataset_name,
                          dataset_config,
                          split=dataset_split,
                          streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))

    def _build_prompt(transcript: str) -> str:
        # Match the runtime chat_template.json. The single ``<|audio_pad|>``
        # is what the joint forward splices the audio embeddings into.
        return ("<|im_start|>system\n<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
                "<|im_start|>assistant\n"
                f"language English{transcript}<|im_end|>")

    yielded = 0
    for example in stream:
        if yielded >= num_samples:
            break
        try:
            audio_bytes = example["audio"]["bytes"]
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                continue
            duration = audio.shape[0] / sr
            if duration > max_audio_seconds:
                continue
            transcript = example.get("text", "").strip()
            if not transcript:
                continue
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            # ``RuntimeError`` covers ``soundfile.LibsndfileError`` for
            # corrupt audio bytes; ``KeyError`` covers a HF dataset schema
            # drift. Real bugs (``TypeError``, ``AttributeError``, ...)
            # propagate. The ``yielded == 0`` check below turns the
            # all-samples-skipped case into a loud failure.
            logger.debug("Skipping calib sample (read error): %s", exc)
            continue

        mel, mel_len = extract_mel_spectrogram(
            audio, sample_rate=sr, feature_extractor_type="whisper")
        # ``extract_mel_spectrogram`` returns ``[mel_bins, T_padded]``
        # numpy; the encoder ``forward`` expects ``[B, mel_bins, T]``
        # float16, so unsqueeze the batch dim and convert.
        input_features = torch.from_numpy(mel).unsqueeze(0).to(torch.float32)
        feature_lens = torch.tensor([mel_len], dtype=torch.int64)

        ragged = prepare_audio_inputs(
            input_features,
            feature_lens,
            n_window=audio_n_window,
            dtype=input_features.dtype,
        )

        prompt = _build_prompt(transcript)
        token_ids = tokenizer(prompt,
                              return_tensors="pt",
                              add_special_tokens=False)["input_ids"]
        if (token_ids == audio_token_id).sum().item() != 1:
            # Defensive: tokenizer changes that re-split <|audio_pad|>
            # would break the joint forward's splice.
            logger.debug("Skipping sample without exactly one audio_pad token")
            continue

        yield {
            "input_ids": token_ids,
            "audio_padded_feature": ragged["padded_feature"],
            "audio_indices": ragged["padded_mask_after_cnn_indices"],
            "audio_attn_mask": ragged["attention_mask"],
        }
        yielded += 1

    if yielded == 0:
        raise RuntimeError(
            f"ASR calibration dataloader produced 0 samples from "
            f"{dataset_name!r}; check dataset access / streaming.")
    logger.info("ASR calibration: %d samples streamed from %s/%s/%s", yielded,
                dataset_name, dataset_config, dataset_split)


def postprocess_qwen3_asr_checkpoint(model_dir: str, output_dir: str) -> None:
    """Convert eo1's vanilla-Qwen3-shaped output back into qwen3_asr layout.

    Mutates *output_dir* in place:

      * ``model.safetensors`` -- prepends ``thinker.`` to every saved key
        (so ``model.embed_tokens.weight`` becomes
        ``thinker.model.embed_tokens.weight``,
        ``audio_tower.layers.0.self_attn.q_proj.weight`` becomes
        ``thinker.audio_tower.layers.0.self_attn.q_proj.weight``, etc.).
      * ``config.json`` -- replaced with the original qwen3_asr config
        (preserves ``thinker_config.audio_config`` /
        ``thinker_config.text_config`` and ``model_type="qwen3_asr"``
        which eo2 routes on).
      * ``chat_template.json`` / ``preprocessor_config.json`` -- copied
        from *model_dir* if present (Qwen3 tokenizer save does not
        produce these).

    ``hf_quant_config.json`` is untouched: ModelOpt writes short module
    names (``model.layers.0.self_attn.q_proj`` /
    ``audio_tower.layers.0.self_attn.q_proj`` / ``lm_head``), and
    ``tensorrt_edgellm.config._normalize_module_name`` strips the ``model.``
    prefix during lookup, so keys remain valid against the re-prefixed
    safetensors.
    """
    saved_path = os.path.join(output_dir, "model.safetensors")
    if not os.path.exists(saved_path):
        raise FileNotFoundError(
            f"Expected eo1 output at {saved_path}; nothing to post-process.")

    saved_weights = load_file(saved_path, device="cpu")
    merged: Dict[str, torch.Tensor] = {
        f"thinker.{k}": v
        for k, v in saved_weights.items()
    }

    # The joint calibration model always carries audio_tower.* in its
    # state_dict (whether quantized or kept FP16). If those keys are
    # missing here, something upstream dropped them -- fail loud rather
    # than silently merge from the source ckpt.
    has_audio = any(
        k.startswith("thinker.audio_tower.") for k in merged.keys())
    if not has_audio:
        raise RuntimeError(
            f"No thinker.audio_tower.* keys in {saved_path} after "
            "export_hf_checkpoint. The joint calibration model should "
            "always include audio_tower.* in its state_dict -- "
            "investigate whether mtq.quantize / export_hf_checkpoint "
            "dropped them.")

    logger.info("Re-prefixed %d tensors with 'thinker.' in %s",
                len(saved_weights), saved_path)
    save_file(merged, saved_path)

    shutil.copy2(os.path.join(model_dir, "config.json"),
                 os.path.join(output_dir, "config.json"))
    logger.info("Replaced config.json with original qwen3_asr config")

    for name in ("chat_template.json", "preprocessor_config.json"):
        src = os.path.join(model_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info("Copied %s from original checkpoint", name)


__all__ = [
    "is_qwen3_asr_model",
    "load_qwen3_asr_joint_for_calibration",
    "asr_calibration_dataloader",
    "postprocess_qwen3_asr_checkpoint",
]
