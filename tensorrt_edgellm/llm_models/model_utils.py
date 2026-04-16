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
Model utility functions for loading and setting up LLM models for models quantization and ONNX export.

This module contains functions for loading Hugging Face models,
checking model types, and setting up quantization.
"""

import gc
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from modelopt.torch.quantization.utils import is_quantized_linear
from safetensors.torch import safe_open
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer, PreTrainedModel)

from .models.eagle3_draft import Eagle3DraftModel


def is_nvfp4_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with NVFP4 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (4, 3)
    return False


def is_mxfp8_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with MXFP8 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (8, 0)
    return False


def set_dynamic_quant(model: nn.Module, dtype: str) -> None:
    """Set quantization for nvfp4 and mxfp8 quantization."""
    for module in model.modules():
        if is_nvfp4_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half" if dtype == "fp16" else "BFloat16"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"
        elif is_mxfp8_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"


def is_vlm(model_dir: str) -> bool:
    """
    Check if the model is a Vision-Language Model (VLM).
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        True if the model is a VLM, False otherwise
    """
    try:
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        cfg_dict = cfg.to_dict()
        has_vision = "vision_config" in cfg_dict
        has_phi4_vision = "image_embd_layer" in cfg_dict.get("embd_layer", {})
        return has_vision or has_phi4_vision
    except Exception:
        return False


def is_gptq_model(model: PreTrainedModel) -> bool:
    """Check if the model is a GPTQ model by config."""
    config = model.config.to_dict()
    quant_config = config.get("quantization_config", None)
    return quant_config and quant_config.get("quant_method") == "gptq"


def _is_gptq_moe_model(model_dir: str) -> bool:
    """Check if a model directory contains a GPTQ MoE model (before loading)."""
    try:
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        cfg_dict = cfg.to_dict()
        quant_config = cfg_dict.get("quantization_config", None)
        is_gptq = quant_config and quant_config.get("quant_method") == "gptq"
        model_type = getattr(cfg, "model_type", "")
        is_moe = "moe" in model_type.lower()
        return is_gptq and is_moe
    except Exception:
        return False


def _resolve_model_path(model_dir: str) -> Path:
    """Resolve model_dir to actual path, handling HuggingFace model IDs."""
    model_path = Path(model_dir)
    if model_path.exists():
        return model_path

    try:
        from huggingface_hub import snapshot_download
        cache_path = snapshot_download(model_dir, local_files_only=True)
        return Path(cache_path)
    except Exception:
        return model_path


def _fix_gptq_moe_gate_weights(model: PreTrainedModel, model_dir: str) -> None:
    """
    Fix MoE gate weights for GPTQ models.

    In GPTQ quantization, the MoE gate/router layer is typically not quantized.
    However, gptqmodel incorrectly converts gate layers to TorchFusedQuantLinear,
    which expects quantized weights (qweight, qzeros, scales, g_idx).

    This function replaces the TorchFusedQuantLinear gate with a regular nn.Linear
    and loads the FP16 weights from the checkpoint.
    """
    model_path = _resolve_model_path(model_dir)
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        print(f"Warning: No safetensor files found at {model_path}")
        return

    gate_weights = {}
    for shard_path in safetensor_files:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".mlp.gate.weight" in key:
                    layer_idx = int(key.split(".")[2])
                    gate_weights[layer_idx] = f.get_tensor(key)

    if not gate_weights:
        print("Warning: No gate weights found in checkpoint")
        return

    gate_layers_fixed = 0
    for layer_idx, gate_weight in gate_weights.items():
        mlp = model.model.layers[layer_idx].mlp
        old_gate = mlp.gate

        out_features, in_features = gate_weight.shape
        new_gate = nn.Linear(in_features, out_features, bias=False)
        new_gate.weight.data.copy_(gate_weight)

        if hasattr(old_gate, 'qweight'):
            device = old_gate.qweight.device
        else:
            device = next(model.parameters()).device
        dtype = gate_weight.dtype
        new_gate = new_gate.to(device=device, dtype=dtype)

        mlp.gate = new_gate
        gate_layers_fixed += 1

    print(
        f'Replaced {gate_layers_fixed} MoE gate layers (TorchFusedQuantLinear -> nn.Linear) in the model'
    )


def _check_model_type(model_dir: str, model_identifier: str) -> bool:
    """
    Check if a model matches a given identifier by checking model_type and architectures.
    
    Args:
        model_dir: Path to the model directory
        model_identifier: String to match against model_type or architectures (case-insensitive)
        
    Returns:
        True if model matches the identifier
    """
    try:
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        return False

    model_type = str(getattr(cfg, "model_type", "")).lower()
    if model_identifier in model_type:
        return True

    archs = getattr(cfg, "architectures", []) or []
    return any(model_identifier in str(a).lower() for a in archs)


def _is_phi4mm_model(dir_path: str) -> bool:
    """Check if the model is a Phi4MM model."""
    return _check_model_type(dir_path, "phi4mm")


def _is_nemotron_h_model(model_dir: str) -> bool:
    """Check if the model is a NemotronH (hybrid Mamba+Attention) model."""
    return _check_model_type(model_dir, "nemotron_h")


def _is_qwen3_omni_model(model_dir: str) -> bool:
    """Check if the model is a Qwen3 Omni model by checking config.json for model_type."""
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    return getattr(cfg, "model_type", None) == "qwen3_omni"


def _is_qwen3_tts_model(model_dir: str) -> bool:
    """Qwen3-TTS is not integrated into transformers yet."""
    return "Qwen3-TTS" in model_dir


def _is_qwen3_asr_model(model_dir: str) -> bool:
    """Qwen3-ASR is not integrated into transformers yet."""
    return "Qwen3-ASR" in model_dir


# Models that require explicit chat template because auto-extraction fails
INCOMPATIBLE_CHAT_TEMPLATE_MODELS = [
    "phi4mm",  # Phi-4-multimodal: tokenizer lacks proper chat template
    "Qwen3-TTS",  # Qwen3-TTS: Special model that does not come with chat template
]


def is_incompatible_chat_template_model(model_dir: str) -> Tuple[bool, str]:
    """
    Check if the model requires an explicit chat template file.
    
    Some models have tokenizers that don't contain proper chat templates
    or have incompatible formats that cannot be auto-extracted.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Tuple of (is_incompatible, model_identifier):
            - is_incompatible: True if model requires explicit chat template
            - model_identifier: String identifying the incompatible model type (empty if compatible)
    """
    for model_identifier in INCOMPATIBLE_CHAT_TEMPLATE_MODELS:
        if _check_model_type(model_dir, model_identifier):
            return True, model_identifier

    return False, ""


def _load_phi4mm_war(model_dir: str):
    """
    Dynamically import local modeling_phi4mm.py as a synthetic package so that
    relative imports work, then inject a no-op prepare_inputs_for_generation
    on Phi4MMModel to satisfy PEFT checks during initialization.
    """
    package_name = "local_phi4mm"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [model_dir]
        sys.modules[package_name] = pkg

    # Preload configuration module if present (support both relative and absolute imports)
    cfg_path = os.path.join(model_dir, "configuration_phi4mm.py")
    if os.path.exists(cfg_path):
        cfg_name_local = f"{package_name}.configuration_phi4mm"
        if cfg_name_local not in sys.modules:
            cfg_spec = importlib.util.spec_from_file_location(
                cfg_name_local, cfg_path)
            cfg_mod = importlib.util.module_from_spec(cfg_spec)
            sys.modules[cfg_name_local] = cfg_mod
            sys.modules["configuration_phi4mm"] = cfg_mod
            cfg_mod.__package__ = package_name
            assert cfg_spec is not None and cfg_spec.loader is not None
            cfg_spec.loader.exec_module(cfg_mod)

    module_name = f"{package_name}.modeling_phi4mm"
    mdl_path = os.path.join(model_dir, "modeling_phi4mm.py")
    spec = importlib.util.spec_from_file_location(module_name, mdl_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[module_name] = module
    sys.modules["modeling_phi4mm"] = module
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    lora_dir = os.path.join(model_dir, "vision-lora")
    if os.path.exists(lora_dir):
        print(f"Loading LoRA models into the PEFT framework.")
        if hasattr(module, "Phi4MMModel"):

            def _fake_prepare_inputs_for_generation(self, *args, **kwargs):
                pass

            module.Phi4MMModel.prepare_inputs_for_generation = _fake_prepare_inputs_for_generation
    else:
        # WAR: Override Phi4MMForCausalLM.__init__ to prevent the model from being
        # converted into a PEFT model, which modelopt and transformers cannot handle correctly.
        # The LoRA weights have already been merged into the base model.
        if (hasattr(module, "Phi4MMForCausalLM")
                and hasattr(module, "Phi4MMModel")
                and hasattr(module, "Phi4MMPreTrainedModel")):

            def _phi4mm_init_war(self, config):
                module.Phi4MMPreTrainedModel.__init__(self, config)
                self.model = module.Phi4MMModel(config)
                self.vocab_size = config.vocab_size
                self.lm_head = nn.Linear(config.hidden_size,
                                         config.vocab_size,
                                         bias=False)

            module.Phi4MMForCausalLM.__init__ = _phi4mm_init_war

        if hasattr(module, "Phi4MMImageAudioEmbedding"):

            def _phi4mm_image_audio_embedding_init_text_only(
                    self, config, **kwargs):
                nn.Module.__init__(self)
                self.vocab_size = config.vocab_size

                # Keep token ids consistent for assertions/BC.
                self.image_input_id = kwargs.get("image_input_id", -1)
                self.audio_input_id = kwargs.get("audio_input_id", -10000)
                assert self.image_input_id != self.audio_input_id, (
                    "image_input_id and audio_input_id should be different")
                self.image_embed = None
                self.audio_embed = None
                self.input_image_embeds = None
                self.image_sizes = None
                self.image_attention_mask = None
                self.input_audio_embeds = None
                self.audio_embed_sizes = None

            # Override Phi4MMImageAudioEmbedding.__init__ to set `image_embed` and `audio_embed` to None.
            # This avoids creating the image/audio towers in the LLM export/quantization pipeline, which
            # is not compatible with ModelOpt currently. We export the visual encoder with a
            # dedicated script (`tensorrt-edgellm-export-visual`) that handles the visual model separately.
            module.Phi4MMImageAudioEmbedding.__init__ = _phi4mm_image_audio_embedding_init_text_only

    return module


def load_hf_model(
    model_dir: str, dtype: str, device: str
) -> Tuple[Union[AutoModelForCausalLM, AutoModelForImageTextToText],
           AutoTokenizer, Optional[AutoProcessor]]:
    """
    Load a HuggingFace model, tokenizer, and optional processor with automatic model type detection.
    
    Args:
        model_dir: Directory containing the model files
        dtype: Model data type ("fp16")
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        
    Returns:
        Tuple of (model, tokenizer, processor)
        processor will be None if AutoProcessor cannot be loaded from the model directory
        
    Raises:
        ValueError: If dtype is not supported or model loading fails
    """
    # Convert dtype string to torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)

    # NemotronH: apply mamba_ssm stub before import to avoid ABI-broken CUDA extension errors.
    # The model runs on the pure-PyTorch slow path for ONNX export.
    if _is_nemotron_h_model(model_dir):
        from .models.nemotron_h_patch import apply as _apply_nemotron_h_patch
        _apply_nemotron_h_patch()
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            trust_remote_code=True).to(device)

    # Due to a known loading issue with Phi4MM on recent transformers, special handling is required.
    # See: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/75.
    elif _is_phi4mm_model(model_dir):
        # Avoid converting the model into a PEFT-wrapped model, which ModelOpt and
        # Transformers cannot currently handle correctly. LoRA weights will instead
        # be merged directly into the base model.
        module = _load_phi4mm_war(model_dir)
        model = module.Phi4MMForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager").to(device)
    elif _is_qwen3_asr_model(model_dir):
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            trust_remote_code=True).model.to(device)
    elif _is_qwen3_tts_model(model_dir):
        from qwen_tts.core.models import (Qwen3TTSConfig,
                                          Qwen3TTSForConditionalGeneration)
        from transformers import AutoConfig, AutoModel, PreTrainedModel
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        # Use PreTrainedModel.from_pretrained to skip speech_tokenizer loading
        # (Qwen3TTSForConditionalGeneration.from_pretrained tries to load
        # speech_tokenizer/feature_extractor which we don't need for LLM export)
        model = PreTrainedModel.from_pretrained.__func__(
            Qwen3TTSForConditionalGeneration,
            model_dir,
            torch_dtype=torch_dtype)
        model = model.to(device)
    elif _is_qwen3_omni_model(model_dir):
        from transformers import Qwen3OmniForConditionalGeneration
        model = Qwen3OmniForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            trust_remote_code=True).to(device)
    elif _is_gptq_moe_model(model_dir):
        print(
            f"Loading GPTQ MoE model from {model_dir}. You might see warnings saying 'Some weights of the model checkpoint at Qwen/Qwen3-30B-A3B-GPTQ-Int4 were not used when initializing Qwen3MoeForCausalLM', which is expected. The weights will be fixed automatically afterwards."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            trust_remote_code=True).to(device)
        _fix_gptq_moe_gate_weights(model, model_dir)
    else:
        # Try loading as AutoModelForCausalLM first
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch_dtype,
                trust_remote_code=True).to(device)
        except Exception as e_causal:
            print(f"AutoModelForCausalLM failed: {e_causal}")
            # If that fails, try AutoModelForImageTextToText
            try:
                # TODO: Need a WAR to quantize only the language model.
                # In VLMs, the model has both model.language_model and model.vision_model.
                model = AutoModelForImageTextToText.from_pretrained(
                    model_dir, torch_dtype=torch_dtype,
                    trust_remote_code=True).to(device)
            except Exception as e:
                raise ValueError(
                    f"Could not load model from {model_dir}. Error: {e}")
    if not is_gptq_model(model):
        model.to(torch_dtype)

    # Set tokenizer padding token if needed
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load processor if available
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            # The fields are required because during quantization it may OOM due to large images in the dataset.
            min_pixels=128 * 28 * 28,
            max_pixels=2048 * 32 * 32)
        print(
            f"Warning: Loaded processor from {model_dir}. The processor will skip image processing for images smaller than 128x28x28 or bigger than 2048x32x32 due to excessive memory usage during image quntization."
        )
    except Exception:
        # Processor not available for this model
        pass

    return model, tokenizer, processor


def load_llm_model(
    model_dir: str,
    dtype: str,
    device: str,
    is_eagle_base: bool,
    reduced_vocab_size: Optional[int] = None,
    vocab_map: Optional[torch.Tensor] = None,
    trt_native_ops: bool = False
) -> tuple[nn.Module, AutoTokenizer, Optional[AutoProcessor]]:
    """
    Load a language model (standard or EAGLE base).
    
    Args:
        model_dir: Directory containing the torch model
        dtype: Model dtype
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)
        is_eagle_base: Whether this is an EAGLE3 base model
        reduced_vocab_size: Size of the reduced vocabulary (optional)
        vocab_map: Tensor of shape (reduced_vocab_size,) with int32 indices for vocabulary reduction (optional)
        trt_native_ops: Whether to use TensorRT native operations instead of plugin
        
    Returns:
        tuple: (model, tokenizer, processor)
    """
    from .models.llm_model import (EdgeLLMHybridModelForCausalLM,
                                   EdgeLLMModelForCausalLM)
    from .models.llm_model_trtnative import EdgeLLMModelTRTNative

    # Determine model type and print message
    if is_eagle_base:
        print(f"Loading eagle3 base model from {model_dir}")
    else:
        print(f"Loading standard model from {model_dir}")

    model, tokenizer, processor = load_hf_model(model_dir, dtype, device)
    set_dynamic_quant(model, dtype)

    # Create EdgeLLMModel wrappers based on model type
    if _is_qwen3_tts_model(model_dir):
        # Qwen3-TTS: Talker + CodePredictor only (no Thinker)
        from .models.qwen3_omni_talker import (Qwen3OmniCodePredictorPatch,
                                               Qwen3OmniTalkerPatch)

        edge_model = {}
        edge_model["talker"] = Qwen3OmniTalkerPatch._from_pretrained_tts(
            model.talker).eval().to(device)
        edge_model[
            "code_predictor"] = Qwen3OmniCodePredictorPatch._from_pretrained_tts(
                model.talker.code_predictor).eval().to(device)

    elif _is_qwen3_omni_model(model_dir) or _is_qwen3_asr_model(model_dir):
        # Qwen3-Omni / ASR: Thinker + optional Talker + CodePredictor
        hf_model = model.thinker

        if not trt_native_ops:
            edge_model = {}
            edge_model["thinker"] = EdgeLLMModelForCausalLM(
                hf_model, is_eagle_base, reduced_vocab_size, vocab_map)

            if hasattr(model, 'has_talker') and model.has_talker:
                from .models.qwen3_omni_talker import (
                    Qwen3OmniCodePredictorPatch, Qwen3OmniTalkerPatch)

                edge_model["talker"] = Qwen3OmniTalkerPatch._from_pretrained(
                    model.talker).eval().to(device)
                edge_model[
                    "code_predictor"] = Qwen3OmniCodePredictorPatch._from_pretrained(
                        model.talker.code_predictor).eval().to(device)
        else:
            edge_model = EdgeLLMModelTRTNative(hf_model, is_eagle_base,
                                               reduced_vocab_size, vocab_map)

    else:
        # Standard LLM / EAGLE
        hf_model = model

        is_hybrid = hasattr(hf_model.config, 'layers_block_type')

        if is_hybrid and not trt_native_ops:
            print("Detected hybrid (Mamba+Attention) architecture")
            edge_model = EdgeLLMHybridModelForCausalLM(hf_model,
                                                       reduced_vocab_size,
                                                       vocab_map)
        elif not trt_native_ops:
            edge_model = {}
            edge_model["model"] = EdgeLLMModelForCausalLM(
                hf_model, is_eagle_base, reduced_vocab_size, vocab_map)
        else:
            edge_model = EdgeLLMModelTRTNative(hf_model, is_eagle_base,
                                               reduced_vocab_size, vocab_map)

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return edge_model, tokenizer, processor


def load_eagle3_draft_model(draft_model_dir: str,
                            base_model_dir: str,
                            dtype: str,
                            device: str,
                            trt_native_ops: bool = False) -> nn.Module:
    """
    Load an EAGLE draft model with base model for weight copying.
    
    Args:
        draft_model_dir: Directory containing the draft model
        base_model_dir: Directory containing the base model 
        dtype: Model data type ("fp16")
        device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.")
        trt_native_ops: Whether to use TensorRT native operations instead of plugin
        
    Returns:
        nn.Module: Draft model (Eagle3DraftModel or Eagle3DraftModelTRTNative)
    """
    print(f"Loading eagle3 draft model from {draft_model_dir}")
    # Convert dtype string to torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Load draft model using from_pretrained. Draft model only support fp16.
    if trt_native_ops:
        from .models.llm_model_trtnative import Eagle3DraftModelTRTNative
        draft_model = Eagle3DraftModelTRTNative.from_pretrained(
            draft_model_dir=draft_model_dir,
            base_model_dir=base_model_dir,
            device=device,
            torch_dtype=torch_dtype).eval().to(device)
    else:
        draft_model = Eagle3DraftModel.from_pretrained(
            draft_model_dir=draft_model_dir,
            base_model_dir=base_model_dir,
            device=device).eval().to(device)

    if not is_gptq_model(draft_model):
        draft_model.to(torch_dtype)

    set_dynamic_quant(draft_model, dtype)

    return draft_model


def load_tensor_by_candidate_keys(model_dir: str, keys_candidate: List[str],
                                  device: str) -> Optional[torch.Tensor]:
    """
    Search all .safetensors shards in `model_dir` and lazily load
    the first matching tensor in `candidate_keys`.

    Returns
    -------
    tensor : Optional[torch.Tensor]
        The requested tensor moved to `device`.
    """
    model_dir = Path(model_dir)
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    for shard_path in safetensor_files:
        # Lazy/MMAP open – only metadata is read
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for key in keys_candidate:
                # try candidates in order
                if key in keys:
                    print(f"Using {key} from {model_dir}/{shard_path.name}")
                    tensor = f.get_tensor(key)  # actually loads data
                    return tensor.to(device)  # move to desired device

    return None


def load_reduced_vocab_map(reduced_vocab_dir: str,
                           device: str) -> Tuple[int, torch.Tensor]:
    """
    Load the reduced vocabulary map from a directory.
    
    The directory should contain a vocab_map.safetensors file with a 'vocab_map' tensor.
    
    Args:
        reduced_vocab_dir: Directory containing vocab_map.safetensors
        device: Device to load the tensor on
        
    Returns:
        Tuple of (reduced_vocab_size, vocab_map)
        
    Raises:
        FileNotFoundError: If vocab_map.safetensors is not found
        KeyError: If 'vocab_map' key is not found in the file
    """
    reduced_vocab_dir = Path(reduced_vocab_dir)
    vocab_map_file = reduced_vocab_dir / "vocab_map.safetensors"

    if not vocab_map_file.exists():
        raise FileNotFoundError(
            f"vocab_map.safetensors not found in {reduced_vocab_dir}")

    print(f"Loading vocab_map from {vocab_map_file}")

    with safe_open(vocab_map_file, framework="pt", device="cpu") as f:
        if "vocab_map" not in f.keys():
            raise KeyError(
                f"'vocab_map' key not found in {vocab_map_file}. Available keys: {list(f.keys())}"
            )
        vocab_map = f.get_tensor("vocab_map")

    vocab_map = vocab_map.to(device)
    reduced_vocab_size = vocab_map.shape[0]

    print(f"Loaded vocab_map with reduced_vocab_size={reduced_vocab_size}")

    return reduced_vocab_size, vocab_map


def prepare_language_model_and_config(hf_model: nn.Module):
    """
    Prepare the language model and config from the HuggingFace model.
    
    Args:
        hf_model: HuggingFace model
        
    Returns:
        Tuple of (model, config)
    """

    language_model = None
    config = None

    # Use language_model if available, otherwise use model.model.
    if hasattr(hf_model, 'language_model'):
        language_model = hf_model.language_model
        config = hf_model.config.text_config
    elif getattr(hf_model.config, 'model_type', '') == 'qwen3_omni_thinker':
        language_model = hf_model.model
        config = hf_model.config.text_config
    elif hasattr(hf_model, 'backbone'):
        language_model = hf_model.backbone
        config = hf_model.config
    else:
        language_model = hf_model.model
        config = hf_model.config

    if hasattr(hf_model.config, "quantization_config"):
        config.quantization_config = hf_model.config.quantization_config

    return language_model, config


def get_eagle3_draft_config(draft_model_dir: str):
    """
    Load and prepare configuration for EAGLE3 draft model.
    
    This function:
    - Loads configuration from the draft model directory
    - Auto-detects VLM models and extracts text config
    
    Args:
        draft_model_dir: Path to the draft model directory
        
    Returns:
        Model configuration object
    """
    config = AutoConfig.from_pretrained(draft_model_dir,
                                        trust_remote_code=True)

    # Auto-detect VLM models and extract text config
    if hasattr(config, 'text_config'):
        config = config.text_config

    return config


def _load_eagle3_draft_weights(draft_model_dir: str,
                               device: str) -> Tuple[Optional[dict], str]:
    """
    Load EAGLE3 draft model weights from available formats.
    
    Checks for weights in the following priority:
    1. pytorch_model.bin
    2. model.safetensors
    
    Args:
        draft_model_dir: Path to the draft model directory
        device: Device to load weights on
        
    Returns:
        state_dict: Loaded weights dictionary
        
    Raises:
        AssertionError: If no model file is found
    """
    from safetensors.torch import load_file

    pytorch_bin_path = os.path.join(draft_model_dir, "pytorch_model.bin")
    safetensors_path = os.path.join(draft_model_dir, "model.safetensors")

    if os.path.exists(pytorch_bin_path):
        print(f"Loading model from {pytorch_bin_path}")
        state_dict = torch.load(pytorch_bin_path,
                                weights_only=True,
                                map_location=device)
        return state_dict
    elif os.path.exists(safetensors_path):
        print(f"Loading model from {safetensors_path}")
        state_dict = load_file(safetensors_path, device=device)
        return state_dict

    raise FileNotFoundError(
        f"Model file not found at {pytorch_bin_path} or {safetensors_path}")


def _process_eagle3_draft_state_dict(state_dict: dict) -> dict:
    """
    Process EAGLE3 draft model state dict with specific key mapping.
    
    This function handles EAGLE3 specific transformations:
    - Keeps 'd2t' key as-is
    - Renames 'midlayer' to 'layers.0'
    - Skips 't2d' key
    - Remaps self_attn.{q,k,v}_proj to self_attn.qkv_proj.{q,k,v}_proj
      to match the EdgeLLMQKVProj wrapper introduced in the attention refactor
    - Remaps self_attn.{q,k,qk}_norm to self_attn.qk_norm.{q,k,qk}_norm
      to match the EdgeLLMQKNorm wrapper
    - Keeps all other keys unchanged
    
    Args:
        state_dict: Raw state dictionary from loaded weights
        
    Returns:
        Processed state dictionary with renamed keys
    """
    processed_state_dict = {}
    for key, value in state_dict.items():
        if 'd2t' in key:
            processed_state_dict[key] = state_dict[key]
        elif 'midlayer' in key:
            new_key = key.replace('midlayer', 'layers.0')
            processed_state_dict[new_key] = value
        elif 't2d' in key:
            continue
        else:
            processed_state_dict[key] = value

    # Remap QKV projection keys to match EdgeLLMQKVProj wrapper nesting:
    # self_attn.q_proj -> self_attn.qkv_proj.q_proj (same for k_proj, v_proj)
    remapped_dict = {}
    for key, value in processed_state_dict.items():
        new_key = key
        for proj in ('q_proj', 'k_proj', 'v_proj'):
            old_pattern = f'self_attn.{proj}'
            new_pattern = f'self_attn.qkv_proj.{proj}'
            if old_pattern in new_key and 'qkv_proj' not in new_key:
                new_key = new_key.replace(old_pattern, new_pattern)
                break
        # Remap QK norm keys to match EdgeLLMQKNorm wrapper nesting:
        # self_attn.q_norm -> self_attn.qk_norm.q_norm (same for k_norm, qk_norm)
        for norm in ('q_norm', 'k_norm'):
            old_pattern = f'self_attn.{norm}'
            new_pattern = f'self_attn.qk_norm.{norm}'
            if old_pattern in new_key and 'qk_norm.' not in new_key:
                new_key = new_key.replace(old_pattern, new_pattern)
                break
        remapped_dict[new_key] = value

    return remapped_dict


def _load_eagle3_draft_embedding_weights(processed_state_dict: dict,
                                         base_model_dir: Optional[str],
                                         device: str) -> None:
    """
    Load embedding weights for EAGLE3 draft model.
    
    If embedding weights are not present in the processed state dict,
    attempts to load them from the base model directory.
    
    Args:
        processed_state_dict: Processed state dictionary (modified in-place)
        base_model_dir: Path to the base model directory (optional)
        device: Device to load weights on
        
    Raises:
        ValueError: If embedding weights are not found and base_model_dir is not provided
    """
    if "embed_tokens.weight" not in processed_state_dict:
        assert base_model_dir is not None, "Base model directory is required to load embedding weights"
        key_candidates = [
            "embed_tokens.weight", "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight"
        ]
        embed_tokens_weight = load_tensor_by_candidate_keys(
            base_model_dir, key_candidates, device)
        if embed_tokens_weight is not None:
            processed_state_dict["embed_tokens.weight"] = embed_tokens_weight
        else:
            raise ValueError(
                "embed_tokens.weight not found in base or draft model")


def load_and_prepare_eagle3_draft_weights(
        draft_model_dir: str, base_model_dir: Optional[str],
        device: str) -> Tuple[Optional[dict], str]:
    """
    Combined helper to load, process, and prepare EAGLE3 draft model weights.
    
    This function combines three operations:
    1. Load weights from disk (handles quantized, pytorch_bin, safetensors)
    2. Process state dict with EAGLE3 specific key mapping
    3. Load embedding weights from base model if needed
    
    Args:
        draft_model_dir: Path to the draft model directory
        base_model_dir: Path to the base model directory (optional, needed for embeddings)
        device: Device to load weights on
        
    Returns:
        Tuple of (processed_state_dict, weight_format) where:
        - processed_state_dict: Fully processed state dictionary ready for model.load_state_dict()
          (None if quantized format)
        - weight_format: One of "quantized", "pytorch_bin", or "safetensors"
        
    Raises:
        AssertionError: If no model file is found
        ValueError: If embedding weights are not found when needed
    """
    # Step 1: Load weights from disk
    state_dict = _load_eagle3_draft_weights(draft_model_dir, device)

    # Step 2: Process EAGLE3 specific key mapping
    processed_state_dict = _process_eagle3_draft_state_dict(state_dict)

    # Step 3: Load embedding weights from base model if needed
    _load_eagle3_draft_embedding_weights(processed_state_dict, base_model_dir,
                                         device)

    return processed_state_dict
