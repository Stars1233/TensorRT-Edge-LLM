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
"""
LLM Quantization Module for TensorRT Edge-LLM.

This module provides quantization utilities for large language models using NVIDIA ModelOpt.
It supports various quantization schemes including FP8, INT4 AWQ, and NVFP4.

For Qwen3-Omni, multimodal calibration is handled by ``omni_quantization.py``.
"""

import contextlib
import glob
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Union

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.utils import is_quantized
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoTokenizer)

from ..llm_models.model_utils import (_is_qwen3_asr_model,
                                      _is_qwen3_omni_model,
                                      load_eagle3_draft_model, load_hf_model,
                                      load_mtp_draft_model)
from ..llm_models.models.eagle3_draft import Eagle3DraftModel
from ..llm_models.models.mtp_draft import MtpDraftModel
from .calib_dataloaders import (get_audio_llm_calib_dataloader,
                                get_text_calib_dataloader)
from .quantization_utils import (enable_huggingface_checkpointing_patch,
                                 quantize_draft_model, quantize_model)

enable_huggingface_checkpointing_patch()


@contextlib.contextmanager
def _patch_get_tied_weight_keys():
    """Temporarily patch ``_get_tied_weight_keys`` to accept list-valued ``_tied_weights_keys``.

    transformers 5.x calls ``tied.keys()`` which fails when a model class
    (e.g. Nemotron) sets ``_tied_weights_keys`` as a plain list.
    """
    import transformers.modeling_utils as _mu
    _orig = _mu._get_tied_weight_keys

    def _patched(module):
        tied_weight_keys = []
        for name, submodule in module.named_modules():
            tied = getattr(submodule, "_tied_weights_keys", None) or []
            keys = tied.keys() if isinstance(tied, dict) else tied
            tied_weight_keys.extend(
                [f"{name}.{k}" if name else k for k in keys])
        return tied_weight_keys

    _mu._get_tied_weight_keys = _patched
    try:
        yield
    finally:
        _mu._get_tied_weight_keys = _orig


# Quantization configuration constants
# FP8 quantization configuration for language model head.
FP8_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "default": {
            "enable": False
        }
    }
}

# INT4 AWQ quantization configuration for language model head.
INT4_AWQ_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {
                -1: 128,
                "type": "static"
            },
            "enable": True
        },
        "default": {
            "enable": False
        }
    }
}

# NVFP4 quantization configuration for language model head.
NVFP4_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True
        },
        "default": {
            "enable": False
        }
    }
}

# MXFP8 quantization configuration for language model head.
MXFP8_LM_HEAD_CONFIG: Dict[str, Any] = {
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
        "default": {
            "enable": False
        }
    }
}

# FP8 attention configuration: enables FP8 Q/K/V BMM quantizers + attention output quantizer.
# ModelOpt's QuantAttention creates q/k/v_bmm_quantizer on attention modules.
# FP8_KV_CFG sets "default": {"enable": false}, so we must explicitly enable each.
FP8_ATTN_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*q_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "*k_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "*v_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
    }
}

# Disable non-LLM submodule quantization during LLM quantization.
# Visual and audio encoders are quantized separately (FP8) via export_visual / export_audio.
DISABLE_NON_LLM_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        k: {
            "enable": False
        }
        for k in (
            "*visual.*",  # Qwen VLM / Qwen3-Omni visual encoder
            "*vision_tower.*",  # LLaVA / InternVL-hf vision encoder
            "*multi_modal_projector.*",  # InternVL-hf projector MLP
            "*mlp1.*",  # InternVL (original) projector MLP
            "*audio_tower.*",  # Qwen3-Omni audio encoder
            "*audio_embed.*",  # Phi-4MM audio embedding
            "*image_embed.*",  # Phi-4MM image embedding
            "*code_predictor.*",  # Qwen3-Omni CodePredictor (stays FP16)
            "*code2wav.*",  # Qwen3-Omni Code2Wav vocoder (stays FP16)
        )
    }
}


def get_llm_quant_config(
        quantization: Optional[str], lm_head_quantization: Optional[str],
        kv_cache_quantization: Optional[str]) -> Dict[str, Any]:
    """
    Get quantization configuration for LLM models.
    
    Args:
        quantization: Optional quantization method
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional attention quantization method
            (enables FP8 KV cache + FP8 FMHA compute)
        
    Returns:
        Dict containing quantization configuration
        
    Raises:
        ValueError: If quantization method is not supported
    """
    # Get base config
    if quantization is None:
        quant_cfg = {"quant_cfg": {}, "algorithm": "max"}
    elif quantization == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    elif quantization == "int4_awq":
        quant_cfg = mtq.INT4_AWQ_CFG.copy()
    elif quantization == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG.copy()
    elif quantization == "mxfp8":
        quant_cfg = mtq.MXFP8_DEFAULT_CFG.copy()
    elif quantization == "int8_sq":
        quant_cfg = mtq.INT8_SMOOTHQUANT_CFG.copy()
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    # Add LM head quantization if specified
    if lm_head_quantization is not None:
        # Remove any existing lm_head configuration
        quant_cfg["quant_cfg"] = {
            k: v
            for k, v in quant_cfg["quant_cfg"].items() if "*lm_head" not in k
        }

        if lm_head_quantization == "fp8":
            quant_cfg["quant_cfg"].update(FP8_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "nvfp4":
            quant_cfg["quant_cfg"].update(NVFP4_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "mxfp8":
            quant_cfg["quant_cfg"].update(MXFP8_LM_HEAD_CONFIG["quant_cfg"])

    # Add attention/KV-cache quantization if specified (FP8 KV cache + FP8 FMHA compute)
    if kv_cache_quantization is not None:
        if kv_cache_quantization == "fp8":
            quant_cfg["quant_cfg"].update(mtq.FP8_KV_CFG["quant_cfg"])
            quant_cfg["quant_cfg"].update(FP8_ATTN_CONFIG["quant_cfg"])

    # Disable non-LLM submodules (visual/audio encoders, Phi-4MM embeds, etc.)
    quant_cfg["quant_cfg"].update(DISABLE_NON_LLM_CONFIG["quant_cfg"])
    return quant_cfg


def quantize_llm(
    model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    quantization: Optional[str],
    lm_head_quantization: Optional[str],
    kv_cache_quantization: Optional[str],
    is_omni: bool = False,
    processor=None,
    model_dir: Optional[str] = None,
    audio_dataset_dir: str = "openslr/librispeech_asr",
    visual_dataset_dir: str = "lmms-lab/MMMU",
) -> Union[AutoModelForCausalLM, AutoModelForImageTextToText]:
    """Quantize a language model using the specified quantization method.

    Qwen3-ASR uses audio-backed calibration for the LLM backbone.
    Qwen3-Omni uses multimodal calibration when a processor is available,
    and falls back to text-only calibration otherwise.

    Args:
        model: The model to quantize.
        tokenizer: Tokenizer for text processing.
        dataset_dir: Calibration dataset. Text by default; ASR may switch to
            audio-backed calibration automatically.
        quantization: Quantization method.
        lm_head_quantization: Optional LM head quantization method.
        kv_cache_quantization: Optional KV cache quantization method.
        is_omni: Use multimodal Omni calibration pipeline.
        processor: HuggingFace processor (Omni multimodal calib).
        model_dir: Original model directory.
        audio_dataset_dir: Audio calibration dataset (Omni).
        visual_dataset_dir: Image calibration dataset (Omni).
    """
    assert (quantization is not None) or (lm_head_quantization is not None) or (kv_cache_quantization is not None), \
        "At least one of 'quantization', 'lm_head_quantization', or 'kv_cache_quantization' must be set (not all None)."
    assert quantization in [
        None, "fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"
    ]
    assert lm_head_quantization in [None, "fp8", "nvfp4", "mxfp8"]
    assert kv_cache_quantization in [None, "fp8"]

    quant_config = get_llm_quant_config(quantization, lm_head_quantization,
                                        kv_cache_quantization)

    if is_omni and processor is not None:
        from .omni_quantization import (get_omni_multimodal_calib_dataset,
                                        omni_multimodal_calib_loop)

        accept_layer = getattr(getattr(model.config, "talker_config", None),
                               "accept_hidden_layer", 14)

        calib_dataset = get_omni_multimodal_calib_dataset(
            processor,
            audio_dataset_dir=audio_dataset_dir,
            visual_dataset_dir=visual_dataset_dir,
            text_dataset_dir=dataset_dir,
        )
        has_talker = hasattr(model, "has_talker") and model.has_talker
        print(f"Omni multimodal calibration: {len(calib_dataset)} samples "
              f"(accept_hidden_layer={accept_layer}, "
              f"talker={'yes' if has_talker else 'no'})")

        def _omni_forward_loop(m):
            omni_multimodal_calib_loop(m, calib_dataset, accept_layer)

        mtq.quantize(model, quant_config, forward_loop=_omni_forward_loop)
        mtq.print_quant_summary(model)

    elif is_omni:
        from tqdm import tqdm

        print(
            "Warning: No processor — falling back to text-only Omni calibration."
        )
        if quantization is None or "int4" in quantization:
            batch_size = 16
        else:
            batch_size = 1
        text_loader = get_text_calib_dataloader(tokenizer=tokenizer,
                                                dataset_dir=dataset_dir,
                                                batch_size=batch_size,
                                                num_samples=512,
                                                max_length=512)
        has_talker = hasattr(model, "has_talker") and model.has_talker

        def _omni_text_loop(m):
            device = next(m.parameters()).device
            for data in tqdm(text_loader,
                             desc="Calibrating Thinker (text-only)"):
                m.thinker(data.to(device))
            if has_talker:
                tc = m.talker.config.text_config
                for i in tqdm(range(64),
                              desc="Calibrating Talker (synthetic)"):
                    seq = 16 + i % 48
                    talker_dtype = next(m.talker.parameters()).dtype
                    m.talker(inputs_embeds=torch.randn(1,
                                                       seq,
                                                       tc.hidden_size,
                                                       dtype=talker_dtype,
                                                       device=device),
                             attention_mask=torch.ones(1,
                                                       seq,
                                                       dtype=torch.long,
                                                       device=device),
                             talker_input_ids=torch.randint(0,
                                                            tc.vocab_size,
                                                            (1, seq),
                                                            device=device))

        mtq.quantize(model, quant_config, forward_loop=_omni_text_loop)
        mtq.print_quant_summary(model)

    else:
        use_audio_calib = model_dir is not None and _is_qwen3_asr_model(
            model_dir)

        if use_audio_calib:
            if dataset_dir == "cnn_dailymail":
                dataset_dir = "openslr/librispeech_asr"
                print("ASR model detected; switching calibration dataset to "
                      f"'{dataset_dir}' (override with --dataset_dir).")
            data_loader = get_audio_llm_calib_dataloader(
                model_dir=model_dir,
                dataset_dir=dataset_dir,
                num_samples=512,
            )
        else:
            if quantization is None or "int4" in quantization:
                batch_size = 16
            else:
                batch_size = 1
            data_loader = get_text_calib_dataloader(tokenizer=tokenizer,
                                                    dataset_dir=dataset_dir,
                                                    batch_size=batch_size,
                                                    num_samples=512,
                                                    max_length=512)
        model = quantize_model(model, quant_config, data_loader)

    return model


def quantize_draft(
    base_model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    draft_model: Union[Eagle3DraftModel],
    tokenizer: AutoTokenizer,
    quantization: str,
    dataset_dir: str,
    lm_head_quantization: Optional[str],
    kv_cache_quantization: Optional[str],
) -> Union[Eagle3DraftModel]:
    """
    Quantize a language model using the specified quantization method.
    
    Args:
        base_model: Based model which is used to generate inputs for the draft model.
        draft_model: The draft model to quantize
        tokenizer: Tokenizer for text processing
        quantization: Quantization method ("fp8", "int4_awq", "nvfp4", "int8_sq")
        dataset_dir: Dataset for calibration
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional attention quantization method
            (enables FP8 KV cache + FP8 FMHA compute)

    Returns:
        Quantized draft model
        
    Raises:
        AssertionError: If quantization method is not supported
    """
    assert quantization in ["fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8"]
    assert lm_head_quantization in [None, "fp8", "nvfp4", "mxfp8"]
    assert kv_cache_quantization in [None, "fp8"]

    # Get calibration dataloader
    if "int4" in quantization:
        batch_size = 16
    else:
        batch_size = 1
    data_loader = get_text_calib_dataloader(tokenizer=tokenizer,
                                            dataset_dir=dataset_dir,
                                            batch_size=batch_size,
                                            num_samples=512,
                                            max_length=512)
    quant_config = get_llm_quant_config(quantization, lm_head_quantization,
                                        kv_cache_quantization)
    model = quantize_draft_model(base_model, draft_model, quant_config,
                                 data_loader)

    return model


def _sanitize_generation_config(model: Any) -> None:
    """Reset sampling-only params to neutral defaults when do_sample=False.

    transformers >= 4.39 validates GenerationConfig strictly during
    save_pretrained(), raising ValueError if sampling-only params are set
    while do_sample=False. Affected params: temperature, top_p, min_p,
    typical_p, top_k, epsilon_cutoff, eta_cutoff.
    """
    if not (hasattr(model, "generation_config")
            and model.generation_config is not None):
        return
    gc = model.generation_config
    if gc.do_sample:
        return
    if gc.temperature != 1.0:
        gc.temperature = 1.0
    if gc.top_p != 1.0:
        gc.top_p = 1.0
    if getattr(gc, "min_p", None) is not None:
        gc.min_p = None
    if getattr(gc, "typical_p", 1.0) != 1.0:
        gc.typical_p = 1.0
    if gc.top_k != 50 and not getattr(gc, "penalty_alpha", None):
        gc.top_k = 50
    if getattr(gc, "epsilon_cutoff", 0.0) != 0.0:
        gc.epsilon_cutoff = 0.0
    if getattr(gc, "eta_cutoff", 0.0) != 0.0:
        gc.eta_cutoff = 0.0


def _graft_mtp_weights(model_dir: str, output_dir: str) -> None:
    """Copy unquantized MTP tensors from the original checkpoint into the
    quantized output, and mark them as excluded from quantization in configs.
    """
    mtp_tensors: Dict[str, torch.Tensor] = {}
    for sf_path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(sf_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    mtp_tensors[key] = f.get_tensor(key)

    if not mtp_tensors:
        return

    _merge_tensors_into_safetensors(output_dir, mtp_tensors)

    # Mark MTP linear layers as excluded from quantization so downstream
    # loaders (e.g. llm_loader) build FP16 linears for them.
    mtp_linear_names = sorted(
        {k.rsplit(".", 1)[0]
         for k, v in mtp_tensors.items() if v.dim() >= 2})
    _patch_quant_exclude_list(output_dir, mtp_linear_names)

    print(f"Grafted {len(mtp_tensors)} unquantized MTP weight(s) into "
          f"{output_dir}.")


def _quantize_mtp_draft_from_base(
    model_dir: str,
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    quantization: str,
    lm_head_quantization: Optional[str],
    dataset_dir: str,
    dtype: str,
    device: str,
) -> MtpDraftModel:
    """Load and quantize the MTP draft model using the unquantized base model.

    Called *before* base model quantization so the unquantized base model can
    generate calibration hidden states without a redundant reload.
    """
    mtp_draft = load_mtp_draft_model(model_dir, dtype, device)

    if is_quantized(mtp_draft):
        print("MTP draft model is already quantized, skipping quantization.")
    else:
        _share_embed_tokens(base_model, mtp_draft)
        mtp_draft = quantize_mtp_draft(base_model, mtp_draft, tokenizer,
                                       quantization, dataset_dir,
                                       lm_head_quantization)
    return mtp_draft


def _save_mtp_draft(mtp_draft: MtpDraftModel, output_dir: str, dtype: str,
                    unified_checkpoint: bool) -> None:
    """Extract, pack, and merge quantized MTP weights into *output_dir*."""
    mtp_tensors = _extract_mtp_state_dict(mtp_draft, dtype, unified_checkpoint)
    _merge_tensors_into_safetensors(output_dir, mtp_tensors)
    print(f"Merged {len(mtp_tensors)} quantized MTP weight(s) into "
          f"{output_dir}.")


def quantize_and_save_llm(model_dir: str,
                          output_dir: str,
                          quantization: Optional[str] = None,
                          dtype: str = "fp16",
                          dataset_dir: str = "cnn_dailymail",
                          lm_head_quantization: Optional[str] = None,
                          kv_cache_quantization: Optional[str] = None,
                          device: str = "cuda",
                          unified_checkpoint: bool = False,
                          audio_dataset_dir: str = "openslr/librispeech_asr",
                          visual_dataset_dir: str = "lmms-lab/MMMU") -> None:
    """Load a model, quantize it if specified, and save the result.

    For Qwen3-Omni models, multimodal calibration data (audio + images + text)
    is used automatically when a processor is available.

    Args:
        model_dir: Directory containing the input HuggingFace model
        output_dir: Directory to save the quantized model
        quantization: Quantization method to apply
        dtype: Model data type for loading ("fp16")
        dataset_dir: Dataset name or path for text calibration data
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional KV cache quantization method
        device: Device to use for model loading and quantization
        unified_checkpoint: Whether to export unified checkpoint
        audio_dataset_dir: HuggingFace dataset for audio calibration (Omni)
        visual_dataset_dir: HuggingFace dataset for image calibration (Omni)
    """
    start_time = time.time()
    is_omni = _is_qwen3_omni_model(model_dir)

    # Load model and tokenizer
    model, tokenizer, processor = load_hf_model(model_dir, dtype, device)

    # Qwen3ASRForConditionalGeneration has no forward(); add one that
    # delegates to the thinker so the calibration loop can call
    # model(input_ids).
    if _is_qwen3_asr_model(model_dir):
        type(model).forward = lambda self, *args, **kwargs: self.thinker(
            *args, **kwargs)
    # TODO: unify base and MTP draft quantization into a single pass
    # to avoid redundant calibration and speed up the process.
    # --- MTP draft: detect and quantize BEFORE base quantization ----------
    # TODO: unify base and MTP draft quantization into a single pass to avoid redundant calibration and speed up the process.
    text_config = getattr(model.config, "text_config", model.config)
    mtp_layers = getattr(text_config, "mtp_num_hidden_layers", 0) or 0
    model_type = getattr(text_config, "model_type", "")
    quantized_mtp_draft: Optional[MtpDraftModel] = None

    if (mtp_layers > 0 and quantization is not None
            and model_type in ("qwen3_5", "qwen3_5_text")
            and not is_quantized(model)):
        quantized_mtp_draft = _quantize_mtp_draft_from_base(
            model_dir=model_dir,
            base_model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            lm_head_quantization=lm_head_quantization,
            dataset_dir=dataset_dir,
            dtype=dtype,
            device=device,
        )

    # --- Quantize base model ----------------------------------------------
    if is_quantized(model):
        print(f"Model is already quantized, skipping quantization.")
    else:
        model = quantize_llm(model,
                             tokenizer,
                             dataset_dir,
                             quantization,
                             lm_head_quantization,
                             kv_cache_quantization,
                             is_omni=is_omni,
                             processor=processor,
                             model_dir=model_dir,
                             audio_dataset_dir=audio_dataset_dir,
                             visual_dataset_dir=visual_dataset_dir)

    quant_end_time = time.time()
    print(f"Quantization finished in {quant_end_time - start_time}s.")

    # Save the quantized model
    os.makedirs(output_dir, exist_ok=True)

    _sanitize_generation_config(model)

    if unified_checkpoint:  # Original checkpoint read by ModelOpt
        from modelopt.torch.export import export_hf_checkpoint

        with torch.inference_mode():
            # WAR: transformers 5.x _get_tied_weight_keys() calls .keys() on
            # _tied_weights_keys, but some models set it as a list, not a dict.
            with _patch_get_tied_weight_keys():
                export_hf_checkpoint(model, export_dir=output_dir)
    else:  # Unified checkpoint read by AutoDeploy
        with torch.inference_mode():
            model.save_pretrained(output_dir)
        # Save the quant config
        quant_config = get_quant_config(model)
        with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
            json.dump(quant_config, f)

    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)

    # --- MTP draft: merge quantized weights or graft unquantized ----------
    if mtp_layers > 0:
        if quantized_mtp_draft is not None:
            _save_mtp_draft(quantized_mtp_draft, output_dir, dtype,
                            unified_checkpoint)
        else:
            if quantization is not None:
                print(f"Warning: MTP quantization is not supported for "
                      f"model_type '{model_type}'. "
                      f"Grafting unquantized MTP weights.")
            _graft_mtp_weights(model_dir, output_dir)

    end_time = time.time()
    print(
        f"Quantized model saved to {output_dir} in {end_time - quant_end_time}s."
    )
    print(f"Total time: {end_time - start_time}s.")


def quantize_and_save_draft(
    base_model_dir: str,
    draft_model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "fp16",
    dataset_dir: str = "cnn_dailymail",
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    unified_checkpoint: bool = False,
) -> None:
    """
    Load an EAGLE draft model, quantize it if specified, and save the result.

    This is the main entry point for quantizing EAGLE draft models. It requires
    both a base model and draft model directory.

    Args:
        base_model_dir: Directory containing the base HuggingFace model
        draft_model_dir: Directory containing the EAGLE draft model
        output_dir: Directory to save the quantized model
        quantization: Quantization method to apply (None, "fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8")
        device: Device to use for model loading and quantization ("cuda", "cpu")
        dtype: Model data type for loading ("fp16")
        dataset_dir: Dataset name or path for calibration data
        lm_head_quantization: Optional separate quantization for language model head (only "fp8", "nvfp4", and "mxfp8" are currently supported)
        kv_cache_quantization: Optional attention quantization (enables FP8 KV cache + FP8 FMHA compute)
        unified_checkpoint: Whether to export as a unified HF checkpoint (compressed safetensors)

    Raises:
        ValueError: If model loading fails or quantization parameters are invalid
    """
    start_time = time.time()

    draft_model = load_eagle3_draft_model(draft_model_dir, base_model_dir,
                                          dtype, device)

    if is_quantized(draft_model):
        print(f"Draft Model is already quantized, skipping quantization.")
    else:
        base_model, tokenizer, _ = load_hf_model(base_model_dir, dtype, device)
        draft_model = quantize_draft(base_model, draft_model, tokenizer,
                                     quantization, dataset_dir,
                                     lm_head_quantization,
                                     kv_cache_quantization)
    quant_end_time = time.time()
    print(f"Quantization finished in {quant_end_time - start_time}s.")

    # Save the quantized model
    os.makedirs(output_dir, exist_ok=True)

    if unified_checkpoint:
        # Save as a unified HF checkpoint (compressed safetensors).
        # We cannot use ``export_hf_checkpoint()`` because it runs
        # ``requantize_resmooth_fused_llm_layers(model)`` which triggers a
        # forward pass, but ``Eagle3DraftModel.forward()`` requires
        # non-standard args.  Instead, manually compress quantized linear
        # modules and build the state dict.
        from modelopt.torch.export.unified_export_hf import (
            QUANTIZATION_NONE, _export_quantized_weight,
            get_quantization_format, is_quantlinear, postprocess_state_dict)
        from safetensors.torch import save_file

        model_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        with torch.inference_mode():
            for _name, sub_module in draft_model.named_modules():
                if get_quantization_format(sub_module) != QUANTIZATION_NONE:
                    if is_quantlinear(sub_module):
                        _export_quantized_weight(sub_module, model_dtype)

        quant_config = get_quant_config(draft_model)
        kv_format = quant_config["quantization"]["kv_cache_quant_algo"]
        sd = draft_model.state_dict()
        sd = postprocess_state_dict(sd, 0, kv_format)

        save_file(sd, os.path.join(output_dir, "model.safetensors"))

        # Copy config.json from the original draft model directory
        src_config = os.path.join(draft_model_dir, "config.json")
        if os.path.isfile(src_config):
            shutil.copy2(src_config, os.path.join(output_dir, "config.json"))
    else:
        _sanitize_generation_config(draft_model)
        with torch.inference_mode():
            draft_model.save_pretrained(output_dir)

    # Save the quant config
    quant_config = get_quant_config(draft_model)
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(quant_config, f)

    end_time = time.time()
    print(
        f"Quantized model saved to {output_dir} in {end_time - quant_end_time}s."
    )
    print(f"Total time: {end_time - start_time}s.")


# ---------------------------------------------------------------------------
# MTP draft quantization
# ---------------------------------------------------------------------------


def quantize_mtp_draft(
    base_model: AutoModelForCausalLM,
    mtp_draft: MtpDraftModel,
    tokenizer: AutoTokenizer,
    quantization: str,
    dataset_dir: str,
    lm_head_quantization: Optional[str],
) -> MtpDraftModel:
    """Quantize the MTP draft model.

    Same structure as ``quantize_draft`` but uses MTP-specific calibration
    (last hidden state from base model) and no KV-cache quantization.
    """
    assert quantization in ["fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8"]
    assert lm_head_quantization in [None, "fp8", "nvfp4", "mxfp8"]

    batch_size = 16 if "int4" in quantization else 1
    data_loader = get_text_calib_dataloader(tokenizer=tokenizer,
                                            dataset_dir=dataset_dir,
                                            batch_size=batch_size,
                                            num_samples=512,
                                            max_length=512)
    quant_config = get_llm_quant_config(quantization, lm_head_quantization,
                                        None)
    return quantize_mtp_draft_model(base_model, mtp_draft, quant_config,
                                    data_loader)


def quantize_mtp_draft_model(
    base_model: torch.nn.Module,
    mtp_draft: MtpDraftModel,
    quant_config: Dict[str, Any],
    calib_dataloader,
) -> MtpDraftModel:
    """Calibrate and quantize the MTP draft model.

    Mirrors ``quantize_draft_model`` for EAGLE: defines a calibration loop
    that runs the base model to produce last-layer hidden states, feeds them
    through the MTP draft's ``quant_forward``, and calls ``mtq.quantize``.

    Args:
        base_model: Unquantized base model for generating calibration hidden states.
        mtp_draft: MTP draft model to quantize.
        quant_config: Quantization configuration dictionary.
        calib_dataloader: DataLoader for calibration data.

    Returns:
        Quantized MTP draft model.
    """
    from tqdm import tqdm

    def calibrate_loop(draft: MtpDraftModel) -> None:
        print(f"Calibrating MTP draft on {len(calib_dataloader)} samples...")
        assert base_model.device == draft.device, \
            "Base model and MTP draft must be on the same device"
        for data in tqdm(calib_dataloader,
                         desc="Calibrating MTP draft",
                         unit="num_samples"):
            data = data.to(base_model.device)
            with torch.no_grad():
                outputs = base_model(data, output_hidden_states=True)
            last_hidden = outputs["hidden_states"][-1]
            draft.quant_forward(data, last_hidden)

    mtq.quantize(mtp_draft, quant_config, forward_loop=calibrate_loop)
    mtq.print_quant_summary(mtp_draft)
    return mtp_draft


# ---------------------------------------------------------------------------
# MTP draft save/merge helpers
# ---------------------------------------------------------------------------


def _share_embed_tokens(base_model: AutoModelForCausalLM,
                        mtp_draft: MtpDraftModel) -> None:
    """Share the base model's embedding table with the MTP draft model."""
    if hasattr(base_model, "model"):
        base_inner = base_model.model
        if hasattr(base_inner, "language_model"):
            mtp_draft.embed_tokens = base_inner.language_model.embed_tokens
        elif hasattr(base_inner, "embed_tokens"):
            mtp_draft.embed_tokens = base_inner.embed_tokens
    if mtp_draft.embed_tokens is None:
        raise ValueError("Could not find embed_tokens in the base model")


def _extract_mtp_state_dict(
        mtp_draft: MtpDraftModel, dtype: str,
        unified_checkpoint: bool) -> Dict[str, torch.Tensor]:
    """Extract the MTP state dict with ``mtp.*`` prefix and HF key names.

    When *unified_checkpoint* is True, quantized weights are packed into
    compressed format before extraction.
    """
    if unified_checkpoint:
        from modelopt.torch.export.unified_export_hf import (
            QUANTIZATION_NONE, _export_quantized_weight,
            get_quantization_format, is_quantlinear, postprocess_state_dict)

        model_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        with torch.inference_mode():
            for _name, sub_module in mtp_draft.named_modules():
                if get_quantization_format(sub_module) != QUANTIZATION_NONE:
                    if is_quantlinear(sub_module):
                        _export_quantized_weight(sub_module, model_dtype)

        mtp_state = postprocess_state_dict(mtp_draft.state_dict(), 0, None)
    else:
        mtp_state = mtp_draft.state_dict()

    mtp_tensors: Dict[str, torch.Tensor] = {}
    for key, tensor in mtp_state.items():
        if key.startswith("embed_tokens") or key.startswith("rotary_emb"):
            continue
        mtp_tensors[f"mtp.{_unremap_attn_key(key)}"] = tensor
    return mtp_tensors


def _unremap_attn_key(key: str) -> str:
    """Reverse EdgeLLMAttention key remapping back to HF checkpoint format."""
    for proj in ("q_proj", "k_proj", "v_proj"):
        old = f"self_attn.qkv_proj.{proj}"
        new = f"self_attn.{proj}"
        if old in key:
            return key.replace(old, new)
    for norm in ("q_norm", "k_norm"):
        old = f"self_attn.qk_norm.{norm}"
        new = f"self_attn.{norm}"
        if old in key:
            return key.replace(old, new)
    return key


# ---------------------------------------------------------------------------
# Shared safetensors / config helpers
# ---------------------------------------------------------------------------


def _merge_tensors_into_safetensors(output_dir: str,
                                    tensors: Dict[str, torch.Tensor]) -> None:
    """Merge *tensors* into the safetensors file(s) in *output_dir*."""
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    single_path = os.path.join(output_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        first_shard = sorted(set(index["weight_map"].values()))[0]
        shard_path = os.path.join(output_dir, first_shard)
        existing = load_file(shard_path)
        existing.update(tensors)
        save_file(existing, shard_path)
        for key in tensors:
            index["weight_map"][key] = first_shard
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    elif os.path.exists(single_path):
        existing = load_file(single_path)
        existing.update(tensors)
        save_file(existing, single_path)
    else:
        save_file(tensors, os.path.join(output_dir, "mtp.safetensors"))


def _patch_quant_exclude_list(output_dir: str,
                              module_names: List[str]) -> None:
    """Add *module_names* to quantization exclude lists in the output dir."""
    for cfg_file, path_to_list in [
        ("config.json", ("quantization_config", )),
        ("hf_quant_config.json", ("quantization", )),
    ]:
        cfg_path = os.path.join(output_dir, cfg_file)
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        section = cfg
        for key in path_to_list:
            section = section.get(key, {})
        if not section:
            continue
        list_key = "ignore" if "ignore" in section else "exclude_modules"
        exclude = list(section.get(list_key, []))
        for name in module_names:
            if name not in exclude:
                exclude.append(name)
        section[list_key] = exclude
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
