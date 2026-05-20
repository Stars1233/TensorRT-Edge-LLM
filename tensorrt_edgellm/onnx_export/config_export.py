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

from typing import Any, Dict, List

from ..version import __version__

_NEMOTRON_H_CHAR_TO_BLOCK_TYPE: Dict[str, str] = {
    "M": "mamba",
    "*": "attention",
    "-": "mlp",
    "E": "moe",
}


def _resolve_hybrid_block_types(config_dict: Dict[str, Any]) -> list[str]:
    """Resolve hybrid block sequence from config, including Nemotron-H MoE ('E')."""
    layers_block_type = config_dict.get("layers_block_type", [])
    if layers_block_type:
        return [str(block) for block in layers_block_type]

    pattern = config_dict.get("hybrid_override_pattern", "")
    if not pattern:
        return []

    return [_NEMOTRON_H_CHAR_TO_BLOCK_TYPE.get(ch, ch) for ch in pattern]


def _nemotron_h_config_layers_block_type(self) -> List[str]:
    """Replacement ``layers_block_type`` property for :class:`NemotronHConfig`.

    Extends the original implementation to support ``'E'`` (MoE expert block)
    in ``hybrid_override_pattern`` in addition to the existing ``'M'``,
    ``'*'``, and ``'-'`` characters.
    """
    pattern = getattr(self, "hybrid_override_pattern", "")
    if not pattern:
        return []
    return [_NEMOTRON_H_CHAR_TO_BLOCK_TYPE.get(c, c) for c in pattern]


def _patch_nemotron_h_config(config) -> None:
    """Monkey-patch *NemotronHConfig* to support ``'E'`` → ``"moe"`` block type.

    The stock ``configuration_nemotron_h.py`` shipped with the 4B/8B models
    only recognises ``'M'``, ``'*'``, and ``'-'`` in its
    ``hybrid_override_pattern``.  The 30B-A3B model introduces ``'E'`` for
    MoE layers, which causes a ``KeyError`` in the original
    ``layers_block_type`` property.

    Calling this function replaces the property on the class the first time
    and is a no-op on subsequent calls (idempotent).
    """
    config_class = type(config)
    if getattr(config_class, "_edgellm_moe_patched", False):
        return
    config_class.layers_block_type = property(
        _nemotron_h_config_layers_block_type)
    config_class._edgellm_moe_patched = True


def _select_rope_parameters(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Select effective RoPE parameters from config in transformers-compatible order."""
    rope_parameters = config_dict.get("rope_parameters")
    if rope_parameters is None:
        rope_parameters = config_dict.get("rope_scaling")

    if not isinstance(rope_parameters, dict):
        return {}

    # Per-layer rope configuration (e.g. Gemma/ModernBERT style nested dict):
    # choose full_attention first, then first available layer type, then first dict value.
    layer_types = config_dict.get("layer_types")
    if layer_types and set(rope_parameters.keys()).issubset(set(layer_types)):
        if isinstance(rope_parameters.get("full_attention"), dict):
            return dict(rope_parameters["full_attention"])
        for layer_type in layer_types:
            layer_params = rope_parameters.get(layer_type)
            if isinstance(layer_params, dict):
                return dict(layer_params)
        return {}

    if "full_attention" in rope_parameters and isinstance(
            rope_parameters["full_attention"], dict):
        return dict(rope_parameters["full_attention"])

    return dict(rope_parameters)


def _normalize_rope_scaling(rope_params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize rope type aliases for runtime compatibility."""
    rope_scaling = dict(rope_params)
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type is not None:
        rope_scaling.setdefault("rope_type", rope_type)
        rope_scaling.setdefault("type", rope_type)
    return rope_scaling


def _export_rope_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export RoPE fields with compatibility for old/new transformers config formats."""
    rope_config = {}
    rope_params = _select_rope_parameters(config_dict)

    # Rope theta can live in top-level or rope parameter dicts.
    if "rope_theta" in config_dict:
        rope_config["rope_theta"] = config_dict["rope_theta"]
    elif "rope_theta" in rope_params:
        rope_config["rope_theta"] = rope_params["rope_theta"]
    else:
        raise KeyError("Required field 'rope_theta' not found in config")

    if rope_params:
        rope_config["rope_scaling"] = _normalize_rope_scaling(rope_params)
    else:
        rope_config["rope_scaling"] = None

    # Handle LongRoPE original max position field in both old/new layouts.
    if rope_config["rope_scaling"] is not None:
        rope_scaling = rope_config["rope_scaling"]
        rope_type = rope_scaling.get("rope_type")
        if rope_type == "longrope":
            original_max = rope_scaling.get(
                "original_max_position_embeddings",
                config_dict.get("original_max_position_embeddings"))
            if original_max is None:
                raise KeyError(
                    "Required field 'original_max_position_embeddings' not found in config"
                )
            rope_config["original_max_position_embeddings"] = original_max

    # Handle partial_rotary_factor in top-level and rope params.
    if "partial_rotary_factor" in config_dict:
        rope_config["partial_rotary_factor"] = config_dict[
            "partial_rotary_factor"]
    elif "partial_rotary_factor" in rope_params:
        rope_config["partial_rotary_factor"] = rope_params[
            "partial_rotary_factor"]
    else:
        rope_config["partial_rotary_factor"] = 1.0

    return rope_config


def _export_native_llm_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export LLM configuration with required fields.

    Args:
        config_dict: Raw model configuration dictionary.

    Returns:
        Dict[str, Any]: Sanitized LLM configuration for Edge-LLM export.
    """
    required_fields = [
        "vocab_size", "max_position_embeddings", "hidden_size",
        "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads"
    ]

    llm_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        llm_config[field] = config_dict[field]

    # Export rope configuration
    llm_config.update(_export_rope_config(config_dict))

    # Handle head_dim
    if "head_dim" in config_dict:
        llm_config["head_dim"] = config_dict["head_dim"]
    else:
        print(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        llm_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    # Gemma3n LAuReL config
    if "laurel_rank" in config_dict:
        llm_config["laurel_rank"] = config_dict["laurel_rank"]

    # Trajectory config for Alpamayo 1
    if "num_traj_tokens" in config_dict:
        llm_config["num_traj_tokens"] = config_dict["num_traj_tokens"]
    if "traj_token_start" in config_dict:
        llm_config["traj_token_start"] = config_dict["traj_token_start"]

    llm_config["model_type"] = "llm"
    return llm_config


def _validate_hybrid_config(llm_config: Dict[str, Any]) -> None:
    """Validate that all required hybrid config fields are present."""
    # Required output fields for any hybrid model config (Mamba, GDN, or future variants).
    # C++ runtime/builder reads exactly these keys with no fallback.
    required_fields = [
        "num_linear_attn_layers",
        "num_attention_layers",
        "recurrent_state_num_heads",
        "recurrent_state_head_dim",
        "recurrent_state_size",
        "conv_dim",
        "conv_kernel",
    ]

    missing = [f for f in required_fields if f not in llm_config]
    if missing:
        raise KeyError(
            f"Hybrid model config missing required linear attention fields: {missing}. "
            f"Required fields: {required_fields}")


def _export_hybrid_mamba_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export hybrid Mamba model configuration with Mamba-specific fields."""
    required_fields = [
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    llm_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        llm_config[field] = config_dict[field]

    rope_params = _select_rope_parameters(config_dict)
    has_rope = ("rope_theta" in config_dict) or ("rope_theta" in rope_params)
    llm_config["use_rope"] = has_rope
    if has_rope:
        llm_config.update(_export_rope_config(config_dict))
    else:
        llm_config["rope_theta"] = 10000.0
        llm_config["rope_scaling"] = None
        llm_config["partial_rotary_factor"] = 1.0

    if "head_dim" in config_dict:
        llm_config["head_dim"] = config_dict["head_dim"]
    else:
        llm_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    if "partial_rotary_factor" not in llm_config:
        if "partial_rotary_factor" in config_dict:
            llm_config["partial_rotary_factor"] = config_dict[
                "partial_rotary_factor"]
        else:
            llm_config["partial_rotary_factor"] = 1.0

    layers_block_type = _resolve_hybrid_block_types(config_dict)
    num_mamba = sum(1 for t in layers_block_type if t == "mamba")
    num_attention = sum(1 for t in layers_block_type if t == "attention")

    llm_config["num_linear_attn_layers"] = num_mamba
    llm_config["num_attention_layers"] = num_attention
    llm_config["recurrent_state_num_heads"] = config_dict["mamba_num_heads"]
    llm_config["recurrent_state_head_dim"] = config_dict["mamba_head_dim"]
    llm_config["recurrent_state_size"] = config_dict["ssm_state_size"]
    n_groups = config_dict.get("n_groups",
                               config_dict.get("mamba_n_groups", 1))
    llm_config["conv_dim"] = (
        llm_config["recurrent_state_num_heads"] *
        llm_config["recurrent_state_head_dim"] +
        2 * n_groups * llm_config["recurrent_state_size"])
    llm_config["conv_kernel"] = config_dict.get(
        "conv_kernel", config_dict.get("mamba_d_conv", 4))

    _validate_hybrid_config(llm_config)

    # Emit canonical per-layer config for HybridCacheManager when layers_block_type
    # is available (e.g. Qwen3.5 GDN models with explicit layer type lists).
    # For hybrid_override_pattern models (Nemotron-H), the C++ scalar fallback
    # (num_attention_layers + num_linear_attn_layers) handles routing correctly.
    layers_block_type = config_dict.get("layers_block_type", [])
    if layers_block_type:
        layer_types = []
        kv_layer_configs = []
        for lt in layers_block_type:
            if lt == "mamba":
                layer_types.append("mamba")
                kv_layer_configs.append(None)
            elif lt == "attention":
                layer_types.append("attention")
                kv_layer_configs.append({
                    "num_kv_heads":
                    llm_config["num_key_value_heads"],
                    "head_dim":
                    llm_config["head_dim"],
                })
            # Skip non-stateful layer types (e.g. "mlp", "moe") — they
            # have no KV cache or recurrent state and must not appear in
            # the per-layer routing table consumed by HybridCacheManager.
        llm_config["layer_types"] = layer_types
        llm_config["kv_layer_configs"] = kv_layer_configs

    llm_config["model_type"] = "hybrid_mamba"
    return llm_config


def _export_hybrid_gdn_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export hybrid GDN model configuration by extending native LLM config."""
    # Reuse native exporter
    llm_config = _export_native_llm_config(config_dict)

    # Hybrid stack metadata (full_attention / linear_attention)
    layer_types = config_dict.get("layer_types", [])
    llm_config["layer_types"] = layer_types
    if layer_types:
        llm_config["num_attention_layers"] = sum(1 for t in layer_types
                                                 if t == "full_attention")
        llm_config["num_linear_attn_layers"] = sum(1 for t in layer_types
                                                   if t == "linear_attention")
    else:
        # Fallback for incomplete configs: assume all layers are full attention.
        llm_config["num_attention_layers"] = config_dict["num_hidden_layers"]
        llm_config["num_linear_attn_layers"] = 0

    # Emit canonical per-layer KV config for HybridCacheManager
    if layer_types:
        kv_layer_configs = []
        for lt in layer_types:
            if lt in ("full_attention", "attention"):
                kv_layer_configs.append({
                    "num_kv_heads":
                    llm_config["num_key_value_heads"],
                    "head_dim":
                    llm_config["head_dim"],
                })
            elif lt == "linear_attention":
                kv_layer_configs.append(None)
            else:
                kv_layer_configs.append(None)
        # Normalize layer_types to "attention"/"mamba" for C++ parser
        normalized_types = []
        for lt in layer_types:
            if lt in ("full_attention", "attention"):
                normalized_types.append("attention")
            elif lt == "linear_attention":
                normalized_types.append("mamba")
            else:
                normalized_types.append(lt)
        llm_config["layer_types"] = normalized_types
        llm_config["kv_layer_configs"] = kv_layer_configs

    # Linear-attention (GDN) related dimensions
    llm_config["linear_num_key_heads"] = config_dict["linear_num_key_heads"]
    llm_config["recurrent_state_num_heads"] = config_dict[
        "linear_num_value_heads"]
    llm_config["recurrent_state_head_dim"] = config_dict["linear_key_head_dim"]
    llm_config["recurrent_state_size"] = config_dict["linear_value_head_dim"]
    llm_config["conv_dim"] = (2 * llm_config["linear_num_key_heads"] *
                              llm_config["recurrent_state_head_dim"] +
                              llm_config["recurrent_state_num_heads"] *
                              llm_config["recurrent_state_size"])
    llm_config["conv_kernel"] = config_dict["linear_conv_kernel_dim"]

    _validate_hybrid_config(llm_config)

    llm_config["model_type"] = "hybrid_gdn"
    return llm_config


def _export_eagle_base_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export EAGLE base configuration with required fields."""
    required_fields = [
        "vocab_size", "max_position_embeddings", "hidden_size",
        "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads"
    ]

    eagle_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        eagle_config[field] = config_dict[field]

    eagle_config.update(_export_rope_config(config_dict))

    # Handle head_dim
    if "head_dim" in config_dict:
        eagle_config["head_dim"] = config_dict["head_dim"]
    else:
        print(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        eagle_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    eagle_config["model_type"] = f"eagle3_base"
    return eagle_config


def _export_eagle_draft_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Export EAGLE draft configuration with required fields."""
    required_fields = [
        "hidden_size", "max_position_embeddings", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads"
    ]

    draft_config = {}
    for field in required_fields:
        if field not in config_dict:
            raise KeyError(f"Required field '{field}' not found in config")
        draft_config[field] = config_dict[field]

    draft_config.update(_export_rope_config(config_dict))

    # Handle head_dim
    if "head_dim" in config_dict:
        draft_config["head_dim"] = config_dict["head_dim"]
    else:
        print(
            "Warning: head_dim not found in config, calculating as hidden_size // num_attention_heads"
        )
        draft_config["head_dim"] = config_dict["hidden_size"] // config_dict[
            "num_attention_heads"]

    # Handle draft_vocab_size based on EAGLE version
    if "draft_vocab_size" not in config_dict:
        raise KeyError("Required field 'draft_vocab_size' not found in config")
    draft_config["draft_vocab_size"] = config_dict["draft_vocab_size"]

    # Add base model configuration fields
    # The target_hidden_size from the model config represents the base model's hidden dimension
    if "target_hidden_size" in config_dict:
        # Use target_hidden_size * 3 as the base model hidden dimension (as per llm_export.py logic)
        draft_config[
            "base_model_hidden_size"] = config_dict["target_hidden_size"] * 3
    else:
        # Fallback: assume base model hidden size is 3x draft model (Eagle3 default)
        draft_config["base_model_hidden_size"] = config_dict["hidden_size"] * 3
        print(
            f"Warning: target_hidden_size not found, using default 3x draft hidden size: {draft_config['base_model_hidden_size']}"
        )

    # Set model_type for draft
    draft_config["model_type"] = f"eagle3_draft"

    return draft_config


def export_vision_config(config: Any) -> Dict[str, Any]:
    """Export vision encoder configuration with proper model_type."""
    config_dict = config.to_dict()

    has_vision = "vision_config" in config_dict
    has_phi4_vision = "image_embd_layer" in config_dict.get("embd_layer", {})
    if not (has_vision or has_phi4_vision):
        raise KeyError(
            "Required field 'vision_config' or 'image_embd_layer' in 'embd_layer' not found in config"
        )
    # Add TensorRT Edge-LLM version
    config_dict['edgellm_version'] = __version__

    # Export RoPE configuration from text_config if it exists, otherwise from top-level config
    if isinstance(config_dict.get("text_config"), dict):
        config_dict["text_config"].update(
            _export_rope_config(config_dict["text_config"]))
    else:
        config_dict.update(_export_rope_config(config_dict))

    # Set top-level model_type for C++ builder/runtime
    # Check if this is Qwen3-Omni vision (has vision_config.model_type = qwen3_omni_vision_encoder)
    if 'vision_config' in config_dict and config_dict['vision_config'].get(
            'model_type') == 'qwen3_omni_vision_encoder':
        config_dict['model_type'] = 'qwen3_omni_vision_encoder'

    # Return the config_dict. Since MRoPE needs LLM config, ViTRunner will use the LLM config.
    return config_dict


def export_llm_config(config: Any,
                      export_type: str,
                      trt_native_ops: bool = False) -> Dict[str, Any]:
    """Export configuration based on export type and EAGLE version.

    Args:
        config: HuggingFace model config object.
        export_type: Edge-LLM export category — one of ``'llm'``,
            ``'eagle3_base'``, or ``'eagle_draft'``.  Hybrid models
            (nemotron_h, qwen3_5) are auto-detected via ``config.model_type``.
        trt_native_ops: Whether TRT native ops are enabled.
    """
    config_dict = config.to_dict()

    # Extract model name from config class
    config_class_name = config.__class__.__name__
    model_name = config_class_name.lower().replace('config', '')

    # For multimodal models, preserve token IDs before switching to text_config
    multimodal_token_ids = {}
    if "text_config" in config_dict:
        print("Detected multimodal model, using text_config")
        # Automatically preserve any field ending with '_token_id' or '_token_ids' at the top level
        for key, value in config_dict.items():
            if key.endswith('_token_id') or key.endswith('_token_ids'):
                multimodal_token_ids[key] = value
        if multimodal_token_ids:
            print(
                f"Preserved multimodal token IDs: {list(multimodal_token_ids.keys())}"
            )
        config_dict = config_dict["text_config"]

    if config.model_type == 'nemotron_h':
        output_config = _export_hybrid_mamba_config(config_dict)
    elif config.model_type in ['qwen3_5_text', 'qwen3_5']:
        output_config = _export_hybrid_gdn_config(config_dict)
    elif export_type == 'llm':
        output_config = _export_native_llm_config(config_dict)
    elif export_type == 'eagle3_base':
        output_config = _export_eagle_base_config(config_dict)
    elif export_type == 'eagle_draft':
        output_config = _export_eagle_draft_config(config_dict)
    else:
        raise ValueError(f"Unsupported export type: {export_type}")

    # Add model name to output
    output_config["model"] = model_name

    # Add TensorRT Edge-LLM version
    output_config['edgellm_version'] = __version__

    # Add trt_native_ops to output_config
    output_config["trt_native_ops"] = trt_native_ops

    # Restore multimodal token IDs if any were saved
    if multimodal_token_ids:
        output_config.update(multimodal_token_ids)
        print(
            f"Restored multimodal token IDs to output config: {list(multimodal_token_ids.keys())}"
        )

    return output_config


def export_audio_config(config: Any) -> Dict[str, Any]:
    """Export audio encoder configuration with proper model_type."""
    config_dict = config.to_dict()

    has_audio = "audio_config" in config_dict
    if not (has_audio):
        raise KeyError("Required field 'audio_config' not found in config")
    # Add TensorRT Edge-LLM version
    config_dict['edgellm_version'] = __version__

    # Set top-level model_type for C++ builder (reads top-level, not audio_config.model_type)
    config_dict['model_type'] = 'qwen3_omni_audio_encoder'

    return config_dict


def export_action_config(config: Any,
                         max_kv_cache_capacity: int) -> Dict[str, Any]:
    """
    Export action configuration with additional architecture parameters.

    Args:
        config: Model expert config (e.g., Qwen3VLTextModel config).
        max_kv_cache_capacity: Maximum KV cache capacity (must match LLM engine).

    Returns:
        Dict[str, Any]: Configuration dictionary with action-specific fields.
    """
    config_dict = config.to_dict()
    rope_params = _select_rope_parameters(config_dict)
    has_rope = ("rope_theta" in config_dict) or ("rope_theta" in rope_params)
    if has_rope:
        config_dict.update(_export_rope_config(config_dict))
    # Add TensorRT Edge-LLM version
    config_dict['edgellm_version'] = __version__
    # Store max_kv_cache_capacity so the C++ builder and runtime can validate it
    config_dict['max_kv_cache_capacity'] = max_kv_cache_capacity

    return config_dict


def export_code2wav_config(config: Any) -> Dict[str, Any]:
    """Export code2wav configuration with proper model_type."""
    config_dict = config.to_dict()

    has_code2wav = "code2wav_config" in config_dict
    if not (has_code2wav):
        raise KeyError("Required field 'code2wav_config' not found in config")

    # Add TensorRT Edge-LLM version
    config_dict['edgellm_version'] = __version__

    # Set model_type to code2wav for proper type identification
    # Override any existing model_type in code2wav_config
    if 'code2wav_config' in config_dict:
        config_dict['code2wav_config']['model_type'] = 'qwen3_omni_code2wav'

    # Also set top-level model_type for easier parsing
    config_dict['model_type'] = 'qwen3_omni_code2wav'

    return config_dict


def export_tts_talker_config(full_tts_config: Any) -> Dict[str, Any]:
    """Export Qwen3-TTS Talker config for C++ runtime."""
    config_dict = full_tts_config.to_dict()
    talker_raw = config_dict["talker_config"]

    talker_config = export_llm_config(full_tts_config.talker_config, 'llm',
                                      False)
    talker_config["model_type"] = "qwen3_tts_talker"
    talker_config["use_embeddings_input"] = True

    # TTS text embedding dimensions
    talker_config["text_hidden_size"] = talker_raw["text_hidden_size"]
    if "text_vocab_size" in talker_raw:
        talker_config["text_vocab_size"] = talker_raw["text_vocab_size"]
    talker_config["num_code_groups"] = talker_raw["num_code_groups"]

    # Codec control tokens
    for key in [
            "codec_eos_token_id", "codec_think_id", "codec_nothink_id",
            "codec_think_bos_id", "codec_think_eos_id", "codec_pad_id",
            "codec_bos_id"
    ]:
        if key in talker_raw:
            talker_config[key] = talker_raw[key]

    # TTS special tokens (top-level config)
    for key in [
            "tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id",
            "im_start_token_id", "im_end_token_id"
    ]:
        if key in config_dict:
            talker_config[key] = config_dict[key]

    # Language / Speaker (TTS field name is spk_id, not speaker_id)
    if "codec_language_id" in talker_raw:
        talker_config["codec_language_id"] = talker_raw["codec_language_id"]
    if "spk_id" in talker_raw:
        talker_config["speaker_id"] = talker_raw["spk_id"]
        talker_config["available_speakers"] = list(talker_raw["spk_id"].keys())
    if "spk_is_dialect" in talker_raw:
        talker_config["spk_is_dialect"] = talker_raw["spk_is_dialect"]

    talker_config = {k: v for k, v in talker_config.items() if v is not None}
    print(f"Exported TTS Talker config with {len(talker_config)} fields")
    return talker_config


def export_talker_config(full_qwen3_omni_config: Any) -> Dict[str, Any]:
    """Export Talker config: preserve original model structure and fields."""
    config_dict = full_qwen3_omni_config.to_dict()

    if "talker_config" not in config_dict:
        raise KeyError("Required field 'talker_config' not found in config")

    # Top-level LLM fields (vocab_size, hidden_size, etc.)
    result = export_llm_config(
        full_qwen3_omni_config.talker_config.text_config,
        export_type='llm',
        trt_native_ops=False)

    talker_config_raw = config_dict["talker_config"]

    # Override model_type to preserve original Talker model type
    # Original: "qwen3_omni_moe_talker" or "qwen3_omni_talker"
    if "model_type" in talker_config_raw:
        result["model_type"] = talker_config_raw["model_type"]

    # Core talker fields
    result["thinker_hidden_size"] = talker_config_raw["thinker_hidden_size"]
    result["accept_hidden_layer"] = talker_config_raw["accept_hidden_layer"]

    # Architecture metadata
    if "num_code_groups" in talker_config_raw:
        result["num_code_groups"] = talker_config_raw["num_code_groups"]

    # Mark as embedding-input model (no tokenizer needed for build)
    result["use_embeddings_input"] = True

    # Multimodal token IDs
    result["audio_token_id"] = talker_config_raw["audio_token_id"]
    if "audio_start_token_id" in talker_config_raw:
        result["audio_start_token_id"] = talker_config_raw[
            "audio_start_token_id"]
    if "audio_end_token_id" in talker_config_raw:
        result["audio_end_token_id"] = talker_config_raw["audio_end_token_id"]

    result["image_token_id"] = talker_config_raw["image_token_id"]
    if "video_token_id" in talker_config_raw:
        result["video_token_id"] = talker_config_raw["video_token_id"]

    # Role tokens from config (used for chat template)
    # Note: eos_token_id is NOT exported here - it should be obtained from tokenizer at runtime
    # This aligns with standard LLM behavior where stopping logic is in the application layer
    token_id_fields = [
        "user_token_id", "assistant_token_id", "system_token_id",
        "im_start_token_id", "im_end_token_id"
    ]

    for key in token_id_fields:
        if key in config_dict and config_dict[key] is not None:
            result[key] = config_dict[key]

    # Codec control tokens (preserve original field names for consistency)
    result["codec_nothink_id"] = talker_config_raw["codec_nothink_id"]
    result["codec_think_bos_id"] = talker_config_raw["codec_think_bos_id"]
    result["codec_think_eos_id"] = talker_config_raw["codec_think_eos_id"]
    result["codec_pad_id"] = talker_config_raw["codec_pad_id"]
    result["codec_bos_id"] = talker_config_raw["codec_bos_id"]
    result["codec_eos_token_id"] = talker_config_raw[
        "codec_eos_token_id"]  # Keep original field name

    # TTS special tokens (from top-level config)
    result["tts_pad_token_id"] = config_dict.get("tts_pad_token_id", 151671)
    result["tts_bos_token_id"] = config_dict.get("tts_bos_token_id", 151672)
    result["tts_eos_token_id"] = config_dict.get("tts_eos_token_id", 151673)

    # Speaker ID mapping for multi-speaker support
    if "speaker_id" in talker_config_raw and talker_config_raw["speaker_id"]:
        result["speaker_id"] = talker_config_raw["speaker_id"]
        # Set default speaker to first speaker in mapping (typically f245: 2301)
        speaker_ids = list(talker_config_raw["speaker_id"].values())
        result["default_speaker_id"] = speaker_ids[0]
        result["available_speakers"] = list(
            talker_config_raw["speaker_id"].keys())
        print(
            f"Exported {len(result['available_speakers'])} speaker IDs, default: {result['default_speaker_id']}"
        )
    else:
        # Fallback: if no speaker_id mapping in config, use default
        print(
            "Warning: No speaker_id mapping found in config, using default f245"
        )
        result["default_speaker_id"] = 2301  # f245 as fallback

    # Optional metadata fields (preserve if present for full compatibility)
    optional_fields = [
        "output_router_logits", "position_id_per_seconds", "seconds_per_chunk",
        "spatial_merge_size"
    ]
    for field in optional_fields:
        if field in talker_config_raw:
            result[field] = talker_config_raw[field]

    # Validate all required fields are present
    required_fields = ["user_token_id", "assistant_token_id"]
    missing_fields = [
        f for f in required_fields if f not in result or result[f] is None
    ]
    if missing_fields:
        raise ValueError(
            f"Required token ID fields missing from config: {missing_fields}")

    # Filter out remaining None values (optional fields only)
    result = {k: v for k, v in result.items() if v is not None}

    print(f"Exported Talker config with {len(result)} fields")

    return result
