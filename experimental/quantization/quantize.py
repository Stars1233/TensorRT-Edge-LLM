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
"""Quantize a HuggingFace LLM and export a unified checkpoint.

Loads a model via ``AutoModelForCausalLM`` (with ``AutoModelForImageTextToText``
fallback for VLMs), runs ModelOpt quantization, and writes a unified safetensors
checkpoint consumable by ``llm_loader``.  No ``tensorrt_edgellm`` dependency.
"""

import json
import os
import shutil
import time
from contextlib import contextmanager
from typing import Any, Optional

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.utils import is_quantized
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer)

from .quantization_configs import build_quant_config


def _text_calib_dataloader(tokenizer,
                           dataset_name="cnn_dailymail",
                           batch_size=1,
                           num_samples=512,
                           max_length=512):
    """Return a DataLoader of tokenised ``input_ids`` for calibration."""
    if "cnn_dailymail" in dataset_name:
        ds = load_dataset(dataset_name, name="3.0.0", split="train")
        texts = ds["article"][:num_samples]
    elif os.path.isdir(dataset_name):
        ds = load_dataset(dataset_name, split="train")
        if "text" in ds.column_names:
            col = "text"
        elif "article" in ds.column_names:
            col = "article"
        else:
            raise ValueError(
                f"Local dataset {dataset_name!r} has no 'text' or 'article' column: "
                f"{ds.column_names}")
        texts = ds[col][:num_samples]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    enc = tokenizer(texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length)
    return DataLoader(enc["input_ids"], batch_size=batch_size, shuffle=False)


def _is_nemotron_h_model(model_dir: str) -> bool:
    """True if ``<model_dir>/config.json`` declares ``model_type == "nemotron_h"``."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            return json.load(f).get("model_type") == "nemotron_h"
    except (OSError, ValueError):
        return False


def _is_phi4mm_model(model_dir: str) -> bool:
    """True if ``<model_dir>/config.json`` declares a Phi-4MM checkpoint."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            model_type = json.load(f).get("model_type")
        return model_type in ("phi4mm", "phi4_multimodal")
    except (OSError, ValueError):
        return False


def _copy_phi4mm_processor_files(model_dir: str, output_dir: str) -> None:
    for name in ("preprocessor_config.json", "processor_config.json",
                 "processing_phi4mm.py"):
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, name))


def _iter_image_question_pairs(dataset_name: str):
    """Yield ``(image, question)`` pairs from a HuggingFace calibration dataset.

    Tolerant of two common schemas:
      * ScienceQA-style: single ``image`` column.
      * MMMU-style: numbered ``image_1`` / ``image_2`` / ... columns, no
        single ``image`` column (one row may carry several images; we take
        the first non-empty one for calibration).

    Splits are tried in the order ``dev`` → ``validation`` → ``train`` —
    matches the legacy ``el`` pipeline (``tensorrt_edgellm.quantization.
    omni_quantization.get_omni_multimodal_calib_dataset``) which uses
    ``split="dev"`` for ``lmms-lab/MMMU`` and ``split="train"`` for ScienceQA.
    """
    last_err: Optional[Exception] = None
    ds = None
    for split in ("dev", "validation", "train"):
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            break
        except Exception as e:  # pylint: disable=broad-except
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load {dataset_name!r} via any of "
                           f"split=dev/validation/train") from last_err

    for example in ds:
        image = example.get("image")
        if image is None:
            for i in range(1, 8):
                image = example.get(f"image_{i}")
                if image is not None:
                    break
        question = example.get("question") or ""
        if image is not None and question:
            yield image, question


def _multimodal_calib_dataloader(processor,
                                 dataset_name: str = "lmms-lab/MMMU",
                                 num_samples: int = 128,
                                 max_length: int = 512,
                                 is_phi4mm: bool = False):
    """Yield ``BatchFeature`` dicts with ``input_ids`` + ``pixel_values``.

    Streams image-question pairs through the model's own ``AutoProcessor``
    chat template so the visual tower receives real activations.  Used when
    ``visual_quantization`` is set — text-only calibration would leave visual
    quantizers with uninitialised scales.

    Default is ``lmms-lab/MMMU`` (the single-config HF re-pack of MMMU's
    multi-config original), matching the legacy ``el`` pipeline default.
    Pass any other HF name to switch (e.g. ``derek-thomas/ScienceQA``).
    Drops down to 128 samples at batch_size=1 — VLM calibration is
    GPU-memory bound; small batches are safest.
    """
    batches: list[dict[str, Any]] = []
    for image, question in _iter_image_question_pairs(dataset_name):
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": question
                },
            ],
        }]

        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            if not is_phi4mm:
                raise
            # Phi-4MM's remote processor has a legacy chat-template signature
            # that rejects ``tokenize`` / ``return_dict`` / ``return_tensors``.
            # It also uses textual image placeholders, so keep this fallback
            # Phi-4MM-only instead of applying ``<|image_1|>`` to other VLMs.
            fallback_messages = [{
                "role": "user",
                "content": f"<|image_1|>{question}",
            }]
            template_owner = processor if hasattr(
                processor, "apply_chat_template") else processor.tokenizer
            text = template_owner.apply_chat_template(
                fallback_messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text,
                               images=[image],
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=max_length)

        batches.append({
            k: v
            for k, v in inputs.items() if v is not None
            and not (isinstance(v, torch.Tensor) and v.numel() == 0)
        })
        if len(batches) >= num_samples:
            break

    if not batches:
        raise RuntimeError(
            f"No usable multimodal samples from {dataset_name!r}. "
            "Check dataset access / processor chat template.")
    return batches


def _load_model(model_dir, dtype="fp16", device="cuda"):
    """Load model + tokenizer + optional processor via Auto* classes."""
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True,
                                                  min_pixels=128 * 28 * 28,
                                                  max_pixels=2048 * 32 * 32)
    except Exception:
        processor = None

    # NemotronH (hybrid Mamba+Attention): the custom modeling code imports
    # ``mamba_ssm.ops.triton.layernorm_gated`` and (when available)
    # ``causal_conv1d``. Apply the in-package patch BEFORE
    # AutoModelForCausalLM.from_pretrained so the modeling import resolves
    # against pure-PyTorch substitutes. This mirrors the legacy
    # tensorrt-edgellm-quantize-llm flow which applies the same patch via
    # tensorrt_edgellm.llm_models.models.nemotron_h_patch.
    if _is_phi4mm_model(model_dir):
        from experimental.llm_loader.lora import load_phi4mm_model
        model = load_phi4mm_model(model_dir, torch_dtype)
        model.to(device)
    else:
        if _is_nemotron_h_model(model_dir):
            from .nemotron_h_patch import apply as _apply_nemotron_h_patch
            _apply_nemotron_h_patch()

        # Try ImageTextToText, then CausalLM, then the generic AutoModel.
        # ImageTextToText goes first because Qwen3.5 / Qwen3-VL register both
        # a CausalLM (text-only) and an ImageTextToText (multimodal)
        # architecture for the same checkpoint; AutoModelForCausalLM happily
        # resolves to the text-only entry and silently drops the visual tower
        # from the loaded model, breaking visual quantization downstream.
        # Some VLMs (e.g. InternVL3) are custom architectures registered only
        # under ``AutoModel``; the more specific factories raise
        # ``ValueError: Unrecognized configuration class``.  We only fall back
        # for *recognition* failures — not for ImportError or other runtime
        # errors, which would otherwise be silently masked by a misleading
        # "Unrecognized configuration class" exception.
        last_err: Optional[Exception] = None
        for factory in (AutoModelForImageTextToText, AutoModelForCausalLM,
                        AutoModel):
            try:
                model = factory.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                ).to(device)
                break
            except (ValueError, KeyError) as e:
                last_err = e
        else:
            raise RuntimeError(
                f"Could not load {model_dir} via any AutoModel factory"
            ) from last_err

    model.to(torch_dtype)

    # modelopt export_hf_checkpoint crashes when architectures is None
    # (e.g. Qwen3.5 resolves to text_config with architectures=None).
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [type(model).__name__]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, processor


def _calibrate(model, dataloader):
    """Forward-loop calibration pass."""
    for data in tqdm(dataloader, desc="Calibrating"):
        data = data.to(model.device)
        if getattr(getattr(model, "config", None), "model_type",
                   "") in ("phi4mm", "phi4_multimodal"):
            model(input_ids=data, input_mode=0, use_cache=False)
        else:
            model(data)


def _normalize_tied_weights_keys(model) -> None:
    """WAR for transformers >= 5.x ``_tied_weights_keys`` format change.

    Newer transformers expects each submodule's ``_tied_weights_keys``
    attribute to be a *dict-like* (so ``modeling_utils._get_tied_weight_keys``
    can call ``.keys()``). Older custom modeling code (e.g. NemotronH's
    ``modeling_nemotron_h.py``) still declares it as a *list*, which
    crashes ``model.save_pretrained`` with::

        AttributeError: 'list' object has no attribute 'keys'

    Convert list-shaped attributes to ``{key: key}`` dicts in place. The
    dict's keys exactly match the original list, preserving behavior for
    downstream tied-weight tracking. No-op for modules that already use
    the dict format or have no ``_tied_weights_keys`` set.
    """
    for module in model.modules():
        attr = getattr(module, "_tied_weights_keys", None)
        if isinstance(attr, list):
            module._tied_weights_keys = {k: k for k in attr}


def _fix_generation_config_for_strict_validate(model) -> None:
    """WAR for transformers >= 5.x ``GenerationConfig.validate(strict=True)``.

    ModelOpt's ``export_hf_checkpoint`` -> ``model.save_pretrained`` ->
    ``generation_config.save_pretrained`` runs ``validate(strict=True)``
    which rejects HF model checkpoints whose ``generation_config.json``
    sets sampling-only kwargs (``top_p`` / ``top_k`` / ``temperature``)
    without setting ``do_sample = True``. NVIDIA-Nemotron-3-Nano-* and
    similar checkpoints ship that exact mismatch.

    Force ``do_sample = True`` when any sampling kwarg is present. This
    only changes the saved ``generation_config.json``; the C++ runtime
    (llm_inference / llm_bench) reads its own runtime params and does
    not depend on this file.
    """
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    sampling_set = (getattr(gc, "top_p", None) not in (None, 1.0)
                    or getattr(gc, "top_k", None) not in (None, 0, 50)
                    or getattr(gc, "temperature", None) not in (None, 1.0))
    if sampling_set and not getattr(gc, "do_sample", False):
        gc.do_sample = True


def _is_hybrid_model(model):
    """Return True if the model has hybrid Mamba+Attention layers.

    Checks multiple signals: ``layers_block_type`` in config (NemotronH),
    ``mamba_ssm_dtype`` in config (Qwen3.5), or ``linear_attn`` submodules.
    """
    config = model.config
    if hasattr(config, "text_config"):
        config = config.text_config
    if getattr(config, "layers_block_type", None) is not None:
        return True
    if getattr(config, "mamba_ssm_dtype", None) is not None:
        return True
    if any("linear_attn" in n for n, _ in model.named_modules()):
        return True
    return False


@contextmanager
def _skip_resmooth_for_hybrid(model, quantization: str = ""):
    """WAR for ModelOpt resmoothing bugs on selected custom models.

    ``export_hf_checkpoint`` calls ``requantize_resmooth_fused_llm_layers``
    which averages AWQ pre_quant_scales across all linear modules that share
    the same input and re-quantizes their weights.  For hybrid models the
    dummy forward used to detect shared inputs does not propagate through
    Mamba layers correctly, and the Mamba projections (qkv, z, a, b) get
    incorrectly fused, corrupting the int4 weights.  For Phi-4 multimodal,
    the dummy forward is incompatible with the required ``input_mode``.

    NVFP4 is exempt: resmoothing works correctly for NVFP4 and is required
    to equalise per-tensor scales across GDN input projections that share
    the same input activation, enabling fusion into a single GEMM.

    This context manager patches the resmoothing function to a no-op when the
    model needs it.  Standard transformer models are unaffected.

    TODO: Remove once ModelOpt fixes these model paths upstream.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    # NVFP4 on hybrid models: resmoothing is safe and required for GDN
    # input projection fusion — do NOT skip.
    is_nvfp4 = quantization.lower() in ("nvfp4", "fp4")
    should_skip = ((_is_hybrid_model(model) and not is_nvfp4)
                   or model_type in ("phi4mm", "phi4_multimodal"))
    if not should_skip:
        yield
        return

    import modelopt.torch.export.unified_export_hf as _ueh
    _orig = _ueh.requantize_resmooth_fused_llm_layers

    def _noop(m):
        print("[WAR] Skipping requantize_resmooth_fused_llm_layers "
              "for this model (ModelOpt bug workaround)")

    _ueh.requantize_resmooth_fused_llm_layers = _noop
    try:
        yield
    finally:
        _ueh.requantize_resmooth_fused_llm_layers = _orig


def _calibrate_multimodal(model, batches):
    """Forward-loop calibration pass for multimodal ``BatchFeature`` dicts."""
    device = model.device
    for batch in tqdm(batches, desc="Calibrating (multimodal)"):
        kwargs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                # Float inputs (pixel_values) inherit the model's dtype;
                # int inputs (input_ids, attention_mask) stay untouched.
                if v.dtype.is_floating_point:
                    v = v.to(next(model.parameters()).dtype)
            kwargs[k] = v
        kwargs.setdefault("use_cache", False)
        with torch.no_grad():
            model(**kwargs)


def quantize_and_export(
    model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    lm_head_quantization: Optional[str] = None,
    visual_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    dataset: str = "cnn_dailymail",
    num_samples: int = 512,
) -> str:
    """Load a HuggingFace model, quantize it, and export a unified checkpoint.

    ``visual_quantization`` turns on quantization of the visual tower
    (``visual.*`` / ``vision_tower.*`` / ``multi_modal_projector.*``); when
    ``None`` (default) the visual tower stays in fp16.  Quantizing the visual
    tower with text-only calibration produces uninitialised activation scales
    on the visual path — a multimodal calibration loader is required for
    accurate visual stats (see ``A3``).
    """
    t0 = time.time()
    model, tokenizer, processor = _load_model(model_dir, dtype, device)

    # --- MTP draft: detect and quantize BEFORE base quantization ----------
    text_config = getattr(model.config, "text_config", model.config)
    mtp_layers = getattr(text_config, "mtp_num_hidden_layers", 0) or 0
    quantized_mtp_draft = None

    if (mtp_layers > 0 and quantization is not None
            and not is_quantized(model)):
        from .models.mtp_draft import quantize_mtp_from_base
        print(f"Detected {mtp_layers} MTP layer(s) — quantizing MTP draft "
              f"before base model.")
        quantized_mtp_draft = quantize_mtp_from_base(
            base_model=model,
            tokenizer=tokenizer,
            model_dir=model_dir,
            quantization=quantization,
            lm_head_quantization=lm_head_quantization,
            kv_cache_quantization=kv_cache_quantization,
            dtype=dtype,
            device=device,
            dataset=dataset,
            num_samples=num_samples,
        )

    # --- Quantize base model ----------------------------------------------
    if is_quantized(model):
        print("Model already quantized — skipping.")
    else:
        quant_cfg = build_quant_config(
            quantization,
            lm_head_quantization,
            kv_cache_quantization,
            visual_quantization=visual_quantization,
        )
        if visual_quantization is not None:
            # Multimodal calibration: feed (image, text) pairs through the
            # whole VLM so visual + LLM quantizers both see real activations.
            processor = AutoProcessor.from_pretrained(model_dir,
                                                      trust_remote_code=True)
            mm_samples = min(num_samples, 128)
            # Use the user's --dataset when it looks like an image+text dataset;
            # fall back to lmms-lab/MMMU when --dataset is the text-only default
            # (cnn_dailymail) since that has no images.  MMMU mirrors the
            # legacy ``el`` pipeline default
            # (tensorrt_edgellm.quantization.llm_quantization::quantize_llm,
            # ``visual_dataset_dir="lmms-lab/MMMU"``).
            mm_dataset = (dataset
                          if dataset != "cnn_dailymail" else "lmms-lab/MMMU")
            batches = _multimodal_calib_dataloader(
                processor,
                dataset_name=mm_dataset,
                num_samples=mm_samples,
                is_phi4mm=_is_phi4mm_model(model_dir))
            mtq.quantize(
                model,
                quant_cfg,
                forward_loop=lambda m: _calibrate_multimodal(m, batches),
            )
        else:
            batch_size = 16 if quantization in (None, "int4_awq") else 1
            loader = _text_calib_dataloader(tokenizer,
                                            dataset,
                                            batch_size=batch_size,
                                            num_samples=num_samples)
            mtq.quantize(model,
                         quant_cfg,
                         forward_loop=lambda m: _calibrate(m, loader))
        mtq.print_quant_summary(model)

    print(f"Quantization: {time.time() - t0:.1f}s")

    _fix_generation_config_for_strict_validate(model)
    _normalize_tied_weights_keys(model)

    os.makedirs(output_dir, exist_ok=True)
    with torch.inference_mode(), _skip_resmooth_for_hybrid(
            model, quantization or ""):
        export_hf_checkpoint(model, export_dir=output_dir)
    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        if _is_phi4mm_model(model_dir):
            _copy_phi4mm_processor_files(model_dir, output_dir)
        else:
            processor.save_pretrained(output_dir)

    # --- MTP draft: merge quantized weights or copy unquantized -----------
    if mtp_layers > 0:
        from .models.mtp_draft import copy_unquantized_mtp, save_quantized_mtp
        if quantized_mtp_draft is not None:
            save_quantized_mtp(quantized_mtp_draft, output_dir, dtype)
        else:
            copy_unquantized_mtp(model_dir, output_dir)

    # Copy preprocessor / processor configs so downstream tools (llm_loader's
    # export_all_cli, the C++ visual builder) can find image preprocessing
    # parameters (patch_size, image_mean, image_std, ...).  ``export_hf_checkpoint``
    # only writes the model + hf_quant_config; processor metadata is part of the
    # source HF directory and must be carried over explicitly.
    for fname in ("preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json", "chat_template.jinja"):
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    print(f"Saved to {output_dir} (total {time.time() - t0:.1f}s)")
    return output_dir
