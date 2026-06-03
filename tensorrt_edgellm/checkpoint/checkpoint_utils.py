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
Checkpoint metadata I/O and runtime sidecars next to exported ONNX.

Weights are loaded only via :func:`loader.load_weights`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from ..models.default.modeling_default import CausalLM

logger = logging.getLogger(__name__)

__all__ = [
    "RUNTIME_TOKENIZER_FILENAMES",
    "normalize_rope_scaling_for_runtime",
    "load_checkpoint_config_dicts",
    "load_config_dict",
    "build_runtime_llm_config_dict",
    "write_runtime_artifacts",
]

RUNTIME_TOKENIZER_FILENAMES: Tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "processed_chat_template.json",
)


def normalize_rope_scaling_for_runtime(rope_scaling: Any) -> Any:
    """Normalize HF MRoPE / rope_parameters metadata to the shape expected by
    the C++ runtime.
    """
    if not isinstance(rope_scaling, dict):
        return rope_scaling

    normalized = dict(rope_scaling)
    if "mrope_section" in normalized:
        rope_type = normalized.get("type") or normalized.get("rope_type")
        if rope_type in (None, "default", "mrope"):
            normalized["type"] = "default"
            normalized["rope_type"] = "default"
    # rope_parameters (transformers v5) carries "rope_type" without "type";
    # propagate the alias so Python callers keyed off "type" still work.
    # C++ collectRopeConfig() accepts either key.
    if "type" not in normalized and "rope_type" in normalized:
        normalized["type"] = normalized["rope_type"]
    return normalized


def _torch_dtype_to_config_str(dtype: Any) -> str:
    """Map a ``torch.dtype`` to the string token the runtime parser accepts
    (see cpp/runtime/config/llmEngineConfig.cpp::parseStateDtype).

    Deferred import of torch so importing this module does not drag it in on
    pure-CPU tooling paths.
    """
    import torch
    mapping = {
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
    }
    if dtype not in mapping:
        raise ValueError(
            f"No config-string mapping for torch dtype {dtype!r}. "
            f"Supported: {sorted(v for v in mapping.values())}")
    return mapping[dtype]


def _nested_config_to_dict(sub: Any) -> Dict[str, Any]:
    if isinstance(sub, dict):
        return sub
    if hasattr(sub, "to_dict"):
        return sub.to_dict()
    return {}


def _promote_llm_subconfig(config: Any, root: Dict[str,
                                                   Any]) -> Dict[str, Any]:
    """Return the dict used for LLM architecture fields (text / nested block)."""
    if root.get("num_attention_heads") is not None:
        return root

    for name in ("llm_config", "text_config", "language_config"):
        sub = getattr(config, name, None)
        if sub is None and name in root:
            sub = root[name]
        sub_dict = _nested_config_to_dict(sub)
        if (sub_dict.get("hidden_size") is not None
                and sub_dict.get("num_attention_heads") is not None):
            return sub_dict

    # Qwen3-ASR / Qwen3-Omni: LLM lives at thinker_config.text_config
    thinker = root.get("thinker_config")
    if isinstance(thinker, dict):
        for name in ("text_config", "llm_config", "language_config"):
            sub_dict = _nested_config_to_dict(thinker.get(name, {}))
            if (sub_dict.get("hidden_size") is not None
                    and sub_dict.get("num_attention_heads") is not None):
                return sub_dict

    # Qwen3-TTS: LLM (talker) lives at talker_config
    talker = root.get("talker_config")
    if isinstance(talker, dict):
        if (talker.get("hidden_size") is not None
                and talker.get("num_attention_heads") is not None):
            return talker

    return root


def _promote_alpamayo_llm_config(root: Dict[str, Any]) -> Dict[str, Any]:
    """For Alpamayo-R1: load the full VLM text config from ``vlm_name_or_path``.

    The Alpamayo root config.json is flat and does not embed VLM architecture
    fields.  We load ``AutoConfig.from_pretrained(vlm_name_or_path)`` to get
    the full Qwen3-VL config and then promote its text sub-config (which
    contains ``num_attention_heads``, ``hidden_size``, etc.).
    """
    from transformers import AutoConfig

    # vlm_name_or_path is a default in AlpamayoR1Config, not persisted in
    # config.json.  Fall back to the known default for Alpamayo-R1.
    vlm_name = root.get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
    if not vlm_name:
        logger.warning("alpamayo_r1 config missing vlm_name_or_path; "
                       "falling back to root config")
        return root

    try:
        vlm_cfg = AutoConfig.from_pretrained(vlm_name, trust_remote_code=True)
        vlm_dict = vlm_cfg.to_dict()
    except (ValueError, OSError) as exc:
        logger.warning(
            "Failed to load VLM config from %s (%s); "
            "falling back to root config", vlm_name, exc)
        return root

    # The VLM config (e.g. Qwen3-VL) has a text sub-config at text_config
    # or language_config.  Promote it using the existing helper.
    llm = _promote_llm_subconfig(vlm_cfg, vlm_dict)

    # Alpamayo extends the vocabulary with trajectory tokens; the root
    # config carries the true vocab_size which must override the base VLM's.
    if root.get("vocab_size") is not None:
        llm["vocab_size"] = root["vocab_size"]

    # Qwen3-VL text_config stores mRoPE info under ``rope_parameters``
    # rather than ``rope_scaling``.  Promote it so downstream code and the
    # C++ runtime find it under the expected ``rope_scaling`` key.
    if not llm.get("rope_scaling") and llm.get("rope_parameters"):
        llm["rope_scaling"] = llm["rope_parameters"]

    return llm


def load_checkpoint_config_dicts(
        model_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(root_dict, llm_dict)`` from the checkpoint config.

    Tries ``AutoConfig.from_pretrained`` first (handles registered HF model
    types).  Falls back to reading ``config.json`` directly for custom /
    not-yet-registered model types (e.g. ``qwen3_asr``, ``qwen3_tts``).

    For multimodal models (e.g. Qwen2.5-VL, Qwen3-ASR), the LLM text config
    is promoted out of the nested sub-object by :func:`_promote_llm_subconfig`.
    Any fields lost during promotion are patched back from the raw JSON.
    """
    from transformers import AutoConfig

    raw_path = os.path.join(model_dir, "config.json")
    raw: Dict[str, Any] = {}
    if os.path.exists(raw_path):
        with open(raw_path) as _f:
            raw = json.load(_f)
    elif not os.path.isdir(model_dir):
        # model_dir is likely an HF model ID (e.g. "Qwen/Qwen3-ASR-0.6B").
        # Download config.json from HF Hub so the raw fallback works.
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(model_dir, "config.json")
            with open(local) as _f:
                raw = json.load(_f)
        except (OSError, ImportError, ValueError):
            pass

    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        root = config.to_dict()
    except (ValueError, OSError) as exc:
        # Unknown / not-yet-registered model type — fall back to raw JSON.
        logger.warning(
            "AutoConfig.from_pretrained failed for %s (%s); "
            "falling back to raw config.json.",
            model_dir,
            exc,
        )
        root = raw
        config = root

    # Alpamayo-R1: flat config with no embedded VLM sub-config.
    # Load the full VLM architecture config from vlm_name_or_path and promote
    # the text sub-config so downstream sees a standard Qwen3-VL text config.
    if root.get("model_type") == "alpamayo_r1":
        llm = _promote_alpamayo_llm_config(root)
    else:
        llm = _promote_llm_subconfig(config, root)

    # Patch: for multimodal models where AutoConfig loses top-level fields,
    # merge them in from the raw config.json.  Only fields absent from llm
    # are copied; existing llm fields are never overwritten.
    for key, val in raw.items():
        if key not in llm and val is not None:
            llm[key] = val

    # VLM / transformers v5 rope compatibility: rope_scaling may be null
    # while rope_parameters carries the real config (transformers v5
    # convention), and either may live only in a nested sub-config
    # (e.g. text_config for Qwen3-VL).  Recover into rope_scaling so
    # collectRopeConfig() in C++ detects kMRope correctly and Python
    # readers that key off rope_scaling (e.g. longrope in
    # build_runtime_llm_config_dict) still work.
    if not llm.get("rope_scaling"):
        candidate = llm.get("rope_parameters")
        if not candidate:
            for subkey in ("text_config", "language_config", "llm_config"):
                raw_sub = raw.get(subkey) or {}
                if isinstance(raw_sub, dict):
                    candidate = (raw_sub.get("rope_scaling")
                                 or raw_sub.get("rope_parameters"))
                    if candidate:
                        break
        if isinstance(candidate, dict):
            llm["rope_scaling"] = candidate
    if llm.get("rope_scaling"):
        llm["rope_scaling"] = normalize_rope_scaling_for_runtime(
            llm["rope_scaling"])

    return root, llm


def load_config_dict(model_dir: str) -> Dict[str, Any]:
    """Return only the promoted LLM config dict."""
    return load_checkpoint_config_dicts(model_dir)[1]


def _export_tool_version() -> str:
    """Version string for runtime ``config.json`` (``edgellm_version`` field)."""
    from .._version import __version__
    return __version__


def _determine_model_type(config) -> str:
    """Map ModelConfig to runtime model_type string."""
    if config.is_eagle3_draft:
        return "eagle3_draft"
    if config.eagle_base:
        return "eagle3_base"
    if config.is_mtp_draft:
        return "mtp_draft"
    if config.mtp_base:
        return "mtp_base"
    if config.is_hybrid:
        return "hybrid_mamba"
    return "llm"


def build_runtime_llm_config_dict(model: "CausalLM") -> Dict[str, Any]:
    """JSON object written as ``config.json`` beside the ONNX export."""
    config = model.config
    mc = config.mamba_cfg
    rope_scaling = normalize_rope_scaling_for_runtime(config.rope_scaling)

    out: Dict[str, Any] = {
        "model": config.model_type,
        "model_type": _determine_model_type(config),
        "edgellm_version": _export_tool_version(),
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "rope_scaling": rope_scaling,
        "partial_rotary_factor": config.partial_rotary_factor,
        "num_deepstack_features": config.num_deepstack_features,
    }

    # longrope requires original_max_position_embeddings for scaling factor computation.
    if (isinstance(rope_scaling, dict)
            and rope_scaling.get("type") == "longrope"
            and config.original_max_position_embeddings is not None):
        out["original_max_position_embeddings"] = config.original_max_position_embeddings

    if config.is_hybrid and mc is not None:
        out.update({
            "num_linear_attn_layers":
            config.num_mamba_layers,
            "num_attention_layers":
            config.num_attn_layers,
            "recurrent_state_num_heads":
            mc.num_heads,
            "recurrent_state_head_dim":
            mc.head_dim,
            "recurrent_state_size":
            mc.ssm_state_size,
            "conv_dim":
            mc.conv_dim,
            "conv_kernel":
            mc.conv_kernel,
            # Nemotron-H attention is NoPE
            "use_rope":
            config.num_attn_layers > 0 and not config.is_nemotron_h,
        })

    gc = config.gdn_cfg
    if config.is_hybrid and gc is not None:
        out.update({
            "num_linear_attn_layers": config.num_gdn_layers,
            "num_attention_layers": config.num_attn_layers,
            "recurrent_state_num_heads": gc.num_value_heads,
            "recurrent_state_head_dim": gc.key_head_dim,
            "recurrent_state_size": gc.value_head_dim,
            "conv_dim": gc.conv_dim,
            "conv_kernel": gc.conv_kernel,
            "use_rope": config.num_attn_layers > 0,
        })

    # Emit canonical per-layer config consumed by the C++ HybridCacheManager.
    # Only attention and linear-attention layers carry KV/recurrent state and
    # must appear in the per-layer routing table. MLP layers are skipped.
    if config.is_hybrid and config.layer_types:
        from ..config import LAYER_ATTN, LAYER_GDN, LAYER_MAMBA

        normalized_layer_types: list = []
        kv_layer_configs: list = []
        for lt in config.layer_types:
            if lt == LAYER_ATTN:
                normalized_layer_types.append("attention")
                kv_layer_configs.append({
                    "num_kv_heads": config.num_key_value_heads,
                    "head_dim": config.head_dim,
                })
            elif lt in (LAYER_MAMBA, LAYER_GDN):
                normalized_layer_types.append("mamba")
                kv_layer_configs.append(None)
            # Non-stateful layers (e.g. LAYER_MLP) have no cache slot and are
            # intentionally omitted from the per-layer routing table.
        out["layer_types"] = normalized_layer_types
        out["kv_layer_configs"] = kv_layer_configs

    if config.is_eagle3_draft:
        draft_vocab = config.draft_vocab_size or config.vocab_size
        target_hidden = config.eagle3_target_hidden_size
        out.update({
            "draft_vocab_size": draft_vocab,
            "base_model_hidden_size": target_hidden * 3,
        })

    if config.is_mtp_draft:
        # MTP draft shares vocab with base (no reduced vocab) and receives
        # base hidden states of size hidden_size (not 3x like EAGLE3).
        out.update({
            "draft_vocab_size": config.vocab_size,
            "base_model_hidden_size": config.hidden_size,
        })

    if config.eagle_base:
        # EAGLE3 base: record which layers provide hidden states to the draft.
        n_layers = config.num_hidden_layers
        out["eagle_hidden_state_layers"] = [2, n_layers // 2, n_layers - 4]

    if config.reduced_vocab_size:
        out["reduced_vocab_size"] = config.reduced_vocab_size

    # KV cache dtype is baked in at export time. The C++ runtime parses it
    # strictly from config.json (no engine-introspection back-patching).
    # Mirrors llm_export.py: "fp8" when KV cache is quantised, otherwise "fp16".
    out["kv_cache_dtype"] = ("fp8" if config.quant.kv_cache_quant == "fp8" else
                             "fp16")

    # Hybrid models (Mamba / GDN / Nemotron-H) bake in recurrent-state and
    # conv-state dtypes at export time. The authoritative source is the
    # model class itself — `export_onnx` constructs dummy tensors with these
    # dtypes, which in turn fix the ONNX binding dtypes the engine is built
    # with. Reading them from the same class attribute used there guarantees
    # the config string cannot drift from the engine binding. The C++ runtime
    # validator cross-checks the config dtype against the engine binding at
    # init, so a drift would fail loudly at load time; this keeps both sides
    # pinned to one source.
    if out.get("num_linear_attn_layers", 0) > 0:
        for attr, key in (("RECURRENT_STATE_DTYPE", "recurrent_state_dtype"),
                          ("CONV_STATE_DTYPE", "conv_state_dtype")):
            torch_dtype = getattr(model, attr, None)
            if torch_dtype is None:
                raise AttributeError(
                    f"{type(model).__name__} is hybrid (num_linear_attn_layers>0) "
                    f"but does not expose {attr}. Add a class-level {attr} "
                    f"(torch.dtype) to the model class; its value must match "
                    f"the dtype of the dummy state tensor its export_onnx "
                    f"builds and the dtype mandated by the plugin schema.")
            out[key] = _torch_dtype_to_config_str(torch_dtype)

    return out


def _build_alpamayo_tokenizer(config: Dict[str, Any], out_dir: str) -> None:
    """Build and save the Alpamayo-R1 tokenizer with added trajectory tokens.

    The base tokenizer comes from the VLM (e.g. Qwen3-VL-8B-Instruct).
    Alpamayo adds discrete trajectory tokens (<i0> .. <i767>) and special
    trajectory tokens (<|traj_history|>, <|traj_future|>, etc.) on top.
    """
    vlm_name = config.get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
    if not vlm_name:
        return

    try:
        from transformers import AutoProcessor
        try:
            processor = AutoProcessor.from_pretrained(vlm_name,
                                                      trust_remote_code=True)
        except (OSError, ValueError) as online_exc:
            logger.warning(
                "Failed to load Alpamayo tokenizer from %s (%s); "
                "retrying local cache", vlm_name, online_exc)
            processor = AutoProcessor.from_pretrained(vlm_name,
                                                      trust_remote_code=True,
                                                      local_files_only=True)
        tokenizer = processor.tokenizer

        # Add discrete trajectory tokens
        traj_vocab_size = config.get("traj_vocab_size", 768)
        if traj_vocab_size:
            discrete_tokens = [f"<i{v}>" for v in range(traj_vocab_size)]
            tokenizer.add_tokens(discrete_tokens)

        # Add special trajectory tokens
        _TRAJ_TOKENS = [
            "<|traj_history|>",
            "<|traj_future|>",
            "<|traj_history_start|>",
            "<|traj_future_start|>",
            "<|traj_history_end|>",
            "<|traj_future_end|>",
        ]
        add_special = config.get("add_special_tokens", False)
        if add_special:
            _SPECIAL_TOKENS_KEYS = [
                "prompt_start",
                "prompt_end",
                "image_start",
                "image_pre_tkn",
                "image_end",
                "traj_history_start",
                "traj_history_pre_tkn",
                "traj_history_end",
                "cot_start",
                "cot_end",
                "meta_action_start",
                "meta_action_end",
                "traj_future_start",
                "traj_future_pre_tkn",
                "traj_future_end",
                "traj_history",
                "traj_future",
                "image_pad",
                "vectorized_wm",
                "vectorized_wm_start",
                "vectorized_wm_end",
                "vectorized_wm_pre_tkn",
                "route_start",
                "route_pad",
                "route_end",
                "question_start",
                "question_end",
                "answer_start",
                "answer_end",
            ]
            special_tokens = ["<|" + k + "|>" for k in _SPECIAL_TOKENS_KEYS]
            tokenizer.add_tokens(special_tokens, special_tokens=True)
        else:
            tokenizer.add_tokens(_TRAJ_TOKENS, special_tokens=True)

        os.makedirs(out_dir, exist_ok=True)
        tokenizer.save_pretrained(out_dir)
        logger.info("Saved Alpamayo tokenizer (%d tokens) to %s",
                    len(tokenizer), out_dir)
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("Failed to build Alpamayo tokenizer: %s", exc)


def write_runtime_artifacts(model: "CausalLM",
                            model_dir: str,
                            out_dir: str,
                            fp8_embedding: bool = False,
                            reduced_vocab_dir: str = "") -> None:
    """Write ``config.json``, ``embedding.safetensors``, tokenizer copies, chat template."""
    import torch
    from safetensors.torch import save_file

    from ..chat_template import (process_chat_template,
                                 write_fallback_processed_chat_template)

    os.makedirs(out_dir, exist_ok=True)

    cfg_json = build_runtime_llm_config_dict(model)

    # For VLM models, the C++ VLM runner (qwenViTRunner, internViTRunner)
    # reads vision_config from the LLM config.json.  Preserve it from the
    # original HF config so the runtime can find deepstack_visual_indexes,
    # num_position_embeddings, etc.
    root_cfg = {}
    if model_dir:
        hf_cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(hf_cfg_path):
            with open(hf_cfg_path) as _f:
                root_cfg = json.load(_f)
            if root_cfg.get("vision_config"):
                cfg_json["vision_config"] = root_cfg["vision_config"]

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)
    logger.info("Wrote config.json to %s", out_dir)

    # EAGLE3 draft models don't need embedding.safetensors — the C++ runtime
    # uses the base model's shared embedding table (the builder already skips
    # copying for draft models).
    if model.config.is_eagle3_draft or model.config.is_mtp_draft:
        kind = ("EAGLE3 draft"
                if model.config.is_eagle3_draft else "MTP draft")
        logger.info(
            "%s: skipping embedding.safetensors (uses base model embedding)",
            kind)
    else:
        embed = getattr(model, "embed_tokens", None)
        if embed is None:
            embed = getattr(getattr(model, "model", None), "embed_tokens",
                            None)
        if embed is None:
            embed = getattr(getattr(model, "backbone", None), "embeddings",
                            None)
        if embed is not None:
            weight = embed.weight.data.cpu()
            # C++ runtime requires FP16 (or FP8) embedding; cast if needed.
            if weight.dtype in (torch.float32, torch.bfloat16):
                weight = weight.to(torch.float16)
            embedding_path = os.path.join(out_dir, "embedding.safetensors")
            if fp8_embedding:
                from .embedding_quantization import quantize_embedding_to_fp8
                embedding_fp8, scales = quantize_embedding_to_fp8(weight)
                save_file(
                    {
                        "embedding": embedding_fp8,
                        "embedding_scale": scales
                    }, embedding_path)
                logger.info("Wrote FP8 embedding.safetensors (%s)",
                            list(weight.shape))
            else:
                save_file({"embedding": weight}, embedding_path)
                logger.info("Wrote embedding.safetensors (%s)",
                            list(weight.shape))
        else:
            logger.warning(
                "embed_tokens not found; skipping embedding.safetensors")

    # Alpamayo-R1: tokenizer lives in the VLM checkpoint, not in model_dir.
    # Build it first so that tokenizer files exist before the copy loop
    # (which is a no-op for Alpamayo) and before process_chat_template.
    if root_cfg.get("model_type") == "alpamayo_r1":
        _build_alpamayo_tokenizer(root_cfg, out_dir)

    for fname in RUNTIME_TOKENIZER_FILENAMES:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            logger.info("Copied %s", fname)

    # If tokenizer.json is missing but vocab.json+merges.txt exist (GPT-2
    # format, used by Qwen3-ASR/TTS), generate tokenizer.json using the
    # transformers library so the C++ runtime can load it.
    tok_json_dst = os.path.join(out_dir, "tokenizer.json")
    if not os.path.exists(tok_json_dst) and model_dir:
        vocab_src = os.path.join(model_dir, "vocab.json")
        merges_src = os.path.join(model_dir, "merges.txt")
        if os.path.exists(vocab_src) and os.path.exists(merges_src):
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model_dir)
                tok.save_pretrained(out_dir)
                logger.info(
                    "Generated tokenizer.json from vocab.json+merges.txt")
            except (OSError, ValueError, ImportError):
                logger.warning("Failed to generate tokenizer.json; "
                               "copying vocab.json and merges.txt as fallback")
                shutil.copy2(vocab_src, os.path.join(out_dir, "vocab.json"))
                shutil.copy2(merges_src, os.path.join(out_dir, "merges.txt"))

    # EAGLE3 draft: save d2t (draft-to-target vocab map)
    d2t = getattr(model, "d2t", None)
    if d2t is not None:
        d2t_cpu = d2t.data.cpu().to(torch.int32)
        save_file({"d2t": d2t_cpu}, os.path.join(out_dir, "d2t.safetensors"))
        logger.info("Wrote d2t.safetensors (%s)", list(d2t_cpu.shape))

    from ..vocab_reduction.onnx_export import copy_reduced_vocab_artifacts
    copy_reduced_vocab_artifacts(model, out_dir, reduced_vocab_dir)

    template_dst = os.path.join(out_dir, "processed_chat_template.json")
    if not os.path.exists(template_dst) and model_dir:
        process_chat_template(model_dir, out_dir)
    if not os.path.exists(template_dst):
        write_fallback_processed_chat_template(model_dir, out_dir)
