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
Centralized command configuration
"""

import os
import shlex
from typing import Dict, List, Tuple

from ..config import (DEFAULT_SEARCH_DEPTH, PRE_QUANTIZED_MODELS, ModelType,
                      TestConfig, _find_directory)
from .llm_loader_helpers import get_tensorrt_edgellm_root

# Available LoRA weights mapping
AVAILABLE_LORA_WEIGHTS = {
    "Qwen2.5-0.5B-Instruct": "Jailbreak-Detector-2-XL",
    "Qwen2.5-0.5B-Instruct-FP8": "Jailbreak-Detector-2-XL",
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-Diagrams2SQL-v2",
}


def _llm_loader_module_shell(module: str, args: List[str]) -> str:
    edgellm_root = get_tensorrt_edgellm_root()
    if not edgellm_root:
        raise ValueError(
            "Cannot find tensorrt-edge-llm root (need experimental/ next to it). "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    exp_dir = os.path.join(edgellm_root, "experimental")
    existing = os.environ.get("PYTHONPATH", "")
    py_path = f"{exp_dir}{os.pathsep}{existing}" if existing else exp_dir
    cmd = ["python3", "-m", module] + args
    inner = " ".join(shlex.quote(x) for x in cmd)
    return f"PYTHONPATH={shlex.quote(py_path)} {inner}"


def _generate_merge_lora_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate merge LoRA commands for models with embedded LoRA (e.g., Phi-4)"""
    commands = []
    if not config.merge_lora:
        return commands

    merge_lora_shell = _llm_loader_module_shell(
        "llm_loader.lora.merge_lora_cli", [
            f"--model_dir={config.get_torch_model_dir()}",
            f"--lora_dir={config.get_lora_adapter_dir()}",
            f"--output_dir={config.get_merged_model_dir()}"
        ])
    commands.append((["bash", "-c", merge_lora_shell], 600))

    return commands


def _experimental_llm_quant_shell(
    config: TestConfig,
    input_model_dir: str,
    output_model_dir: str,
    needs_weight_quant: bool,
    needs_kv_cache_quant: bool,
) -> str:
    """``cd <sdk> && python -m experimental.quantization llm ...`` (unified ModelOpt export)."""
    edgellm_root = get_tensorrt_edgellm_root()
    if not edgellm_root:
        raise ValueError(
            "Cannot find tensorrt-edge-llm root (need experimental/ next to it). "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    args: List[str] = [
        "python3",
        "-m",
        "experimental.quantization",
        "llm",
        f"--model_dir={input_model_dir}",
        f"--output_dir={output_model_dir}",
        f"--dataset={config.get_cnn_dailymail_dataset_dir()}",
    ]
    if needs_weight_quant:
        args.append(f"--quantization={config.llm_precision}")
    if config.lm_head_precision != "fp16" and needs_weight_quant:
        args.append(f"--lm_head_quantization={config.lm_head_precision}")
    if needs_kv_cache_quant:
        args.append("--kv_cache_quantization=fp8")
    inner = " ".join(shlex.quote(x) for x in args)
    return f"cd {shlex.quote(edgellm_root)} && {inner}"


def _generate_quantization_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate quantization commands if needed.

    Quantizer choice is paired with the export path so the on-disk weight
    layout matches what the exporter loads:

      * **llm_loader export path** (``can_use_llm_loader(config)`` is True):
        use ``python -m experimental.quantization`` — ModelOpt +
        ``export_hf_checkpoint`` writes a unified HF checkpoint that
        ``llm_loader.export_all_cli`` knows how to unpack (NVFP4 / INT4-AWQ
        weights are stored in packed form alongside ``hf_quant_config.json``).

      * **legacy export path** (trt_native_ops forces
        ``tensorrt-edgellm-export-llm``): use ``tensorrt-edgellm-quantize-llm``.
        The legacy exporter loads the model via stock
        ``AutoModelForCausalLM.from_pretrained``, which cannot unpack
        ModelOpt's compressed NVFP4/INT4-AWQ tensors and fails with
        ``MISMATCH … ckpt: [N, K/2] vs model: [N, K]`` on the gate/up/down
        projections. Pairing the legacy quantizer with the legacy exporter
        keeps the format mutually consistent.

    int4_gptq models are pre-quantized (``is_prequantized()`` returns True)
    so they bypass the weight-quant step here entirely; there are no
    ``int4_gptq + fp8kv`` test cases that would route GPTQ weights into a
    KV-cache-only quant pass.
    """
    commands = []
    # Pre-quantized models ship with weights already quantized; skip this step entirely.
    if config.model_name in PRE_QUANTIZED_MODELS:
        return commands
    # Quantize weights (for non-fp16) and/or KV cache (when fp8_kv_cache is enabled).
    needs_weight_quant = config.llm_precision != "fp16" and not config.is_prequantized(
    )
    needs_kv_cache_quant = bool(config.fp8_kv_cache)
    if needs_weight_quant or needs_kv_cache_quant:
        # Use the merged checkpoint when a model ships a required LoRA
        # adapter, otherwise use the raw torch checkpoint.
        if config.merge_lora:
            input_model_dir = config.get_merged_model_dir()
        else:
            input_model_dir = config.get_torch_model_dir()

        if needs_weight_quant:
            output_model_dir = config.get_quantized_model_dir()
        else:
            # KV-cache-only quantization (fp16 weights)
            output_model_dir = config.get_kv_cache_quantized_model_dir()

        if can_use_llm_loader(config):
            # llm_loader path: experimental.quantization (unified HF, packed).
            shell = _experimental_llm_quant_shell(config, input_model_dir,
                                                  output_model_dir,
                                                  needs_weight_quant,
                                                  needs_kv_cache_quant)
            commands.append((["bash", "-c", shell], 1200))
        else:
            # Legacy export path: legacy tensorrt-edgellm-quantize-llm so
            # the resulting on-disk layout is loadable by stock HF
            # AutoModelForCausalLM that the legacy exporter uses.
            quantize_cmd = [
                "tensorrt-edgellm-quantize-llm",
                f"--model_dir={input_model_dir}",
                f"--output_dir={output_model_dir}",
                f"--dataset_dir={config.get_cnn_dailymail_dataset_dir()}",
            ]
            if needs_weight_quant:
                quantize_cmd.append(f"--quantization={config.llm_precision}")
            if (config.lm_head_precision != "fp16" and needs_weight_quant):
                quantize_cmd.append(
                    f"--lm_head_quantization={config.lm_head_precision}")
            if needs_kv_cache_quant:
                quantize_cmd.append("--kv_cache_quantization=fp8")
            commands.append((quantize_cmd, 1200))

    return commands


def _generate_llm_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate LLM export commands"""
    from .llm_loader_helpers import get_export_model_dir

    model_dir = get_export_model_dir(config)

    llm_cmd = [
        "tensorrt-edgellm-export-llm", f"--model_dir={model_dir}",
        f"--output_dir={config.get_llm_onnx_dir()}"
    ]

    if config.fp8_kv_cache:
        llm_cmd.append("--fp8_kv_cache")

    if config.fp8_embedding:
        llm_cmd.append("--fp8_embedding")

    chat_template_path = config.get_chat_template_file()
    if chat_template_path:
        llm_cmd.append(f"--chat_template={chat_template_path}")

    if config.reduced_vocab_size:
        llm_cmd.append(f"--reduced_vocab_dir={config.get_reduced_vocab_dir()}")

    if config.trt_native_ops:
        llm_cmd.append("--trt_native_ops")

    return [(llm_cmd, 1200)]


def _generate_visual_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate visual model export commands for VLMs"""
    commands = []
    if config.model_type != ModelType.VLM:
        return commands

    visual_export_cmd = [
        "tensorrt-edgellm-export-visual",
        f"--model_dir={config.get_torch_model_dir()}",
        f"--dtype=fp16",
    ]

    fp16_visual_export_cmd = visual_export_cmd.copy()
    fp16_visual_export_cmd.append(
        f"--output_dir={config.get_visual_onnx_dir('fp16')}")
    commands.append((fp16_visual_export_cmd, 1200))

    if config.visual_precision == "fp8":
        fp8_visual_export_cmd = visual_export_cmd.copy()
        fp8_visual_export_cmd.append(f"--quantization=fp8")
        fp8_visual_export_cmd.append(
            f"--output_dir={config.get_visual_onnx_dir('fp8')}")
        fp8_visual_export_cmd.append(
            f"--dataset_dir={config.get_mmmu_dataset_dir()}")
        commands.append((fp8_visual_export_cmd, 1200))
    return commands


def _generate_lora_commands(config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate LoRA processing commands"""
    commands = []
    if not config.lora:
        return commands

    # Insert LoRA command
    lora_cmd = [
        "tensorrt-edgellm-insert-lora",
        f"--onnx_dir={config.get_llm_onnx_dir()}"
    ]
    commands.append((lora_cmd, 120))

    # Process LoRA weights if available
    if config.model_name in AVAILABLE_LORA_WEIGHTS:
        # Get base data directory from environment variable
        edgellm_data_dir = os.environ.get('EDGELLM_DATA_DIR',
                                          '/scratch.edge_llm_cache')

        # Search for the LoRA weights directory
        lora_model_name = AVAILABLE_LORA_WEIGHTS[config.model_name]
        lora_weights_dir = _find_directory(edgellm_data_dir, lora_model_name,
                                           DEFAULT_SEARCH_DEPTH)

        if not lora_weights_dir:
            raise ValueError(
                f"LoRA weights directory '{lora_model_name}' not found under "
                f"'{edgellm_data_dir}' within search depth {DEFAULT_SEARCH_DEPTH}."
            )

        process_lora_cmd = [
            "tensorrt-edgellm-process-lora", f"--input_dir={lora_weights_dir}",
            f"--output_dir={config.get_lora_weights_dir()}"
        ]
        commands.append((process_lora_cmd, 120))
    else:
        raise ValueError(
            f"No LoRA weights available for {config.model_name}. Please add it to AVAILABLE_LORA_WEIGHTS"
        )

    return commands


def _experimental_draft_quant_shell(config: TestConfig) -> str:
    """``cd <sdk> && python -m experimental.quantization draft ...``"""
    edgellm_root = get_tensorrt_edgellm_root()
    if not edgellm_root:
        raise ValueError(
            "Cannot find tensorrt-edge-llm root (need experimental/ next to it). "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    base_model_dir = config.get_torch_model_dir()
    draft_model_dir = config.get_draft_model_dir()
    quantized_draft_dir = config.get_quantized_draft_model_dir()
    args: List[str] = [
        "python3",
        "-m",
        "experimental.quantization",
        "draft",
        f"--base_model_dir={base_model_dir}",
        f"--draft_model_dir={draft_model_dir}",
        f"--output_dir={quantized_draft_dir}",
        f"--quantization={config.draft_llm_precision}",
        f"--dataset={config.get_cnn_dailymail_dataset_dir()}",
    ]
    if (config.draft_lm_head_precision
            and config.draft_lm_head_precision != "fp16"):
        args.append(f"--lm_head_quantization={config.draft_lm_head_precision}")
    inner = " ".join(shlex.quote(x) for x in args)
    return f"cd {shlex.quote(edgellm_root)} && {inner}"


def _generate_draft_quantization_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate draft model quantization commands for EAGLE.

    Uses ``experimental.quantization``; legacy ``tensorrt-edgellm-quantize-draft``
    is not used. Output is a unified ModelOpt ``export_hf_checkpoint`` tree
    consumable by both ``llm_loader.export_all_cli`` and the legacy
    ``tensorrt-edgellm-export-llm`` CLI.
    """
    commands = []
    if not config.is_eagle:
        return commands
    if config.is_mtp:
        return commands

    if config.draft_llm_precision is None:
        raise ValueError("draft_llm_precision not set for EAGLE mode")

    # Only quantize if draft model is not fp16
    if (config.draft_llm_precision != "fp16"
            and config.draft_llm_precision != "int4_gptq"):
        shell = _experimental_draft_quant_shell(config)
        commands.append((["bash", "-c", shell], 900))

    return commands


def _generate_draft_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate draft model export commands for EAGLE (legacy tool only).

    Needed for the ``is_eagle + reduced_vocab`` combination on the legacy
    path: vocab reduction reads ``d2t.safetensors`` from the draft ONNX
    directory, so the draft must be exported before vocab reduction runs.
    llm_loader handles its own draft export via ``run_llm_loader_draft_export``.
    MTP is skipped: the draft is produced by ``export_all_cli --mtp`` in a
    single invocation alongside the base model.
    """
    commands = []
    if not config.is_eagle:
        return commands
    if config.is_mtp:
        return commands

    base_model_dir = config.get_torch_model_dir()
    if (config.draft_llm_precision and config.draft_llm_precision != "fp16"
            and config.draft_llm_precision != "int4_gptq"):
        draft_model_dir = config.get_quantized_draft_model_dir()
    else:
        draft_model_dir = config.get_draft_model_dir()

    export_draft_cmd = [
        "tensorrt-edgellm-export-draft", f"--base_model_dir={base_model_dir}",
        f"--draft_model_dir={draft_model_dir}",
        f"--output_dir={config.get_draft_onnx_dir()}"
    ]

    commands.append((export_draft_cmd, 600))
    return commands


def _generate_llm_loader_draft_export_for_vocab_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Export EAGLE draft early when vocab reduction needs d2t.safetensors."""
    commands = []
    if not (config.is_eagle and config.reduced_vocab_size):
        return commands

    if (config.draft_llm_precision and config.draft_llm_precision != "fp16"
            and config.draft_llm_precision != "int4_gptq"):
        draft_model_dir = config.get_quantized_draft_model_dir()
    else:
        draft_model_dir = config.get_draft_model_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()

    edgellm_root = get_tensorrt_edgellm_root()
    if not edgellm_root:
        raise ValueError(
            "Cannot find tensorrt-edge-llm root (need experimental/ next to it). "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    exp_dir = os.path.join(edgellm_root, "experimental")
    existing = os.environ.get("PYTHONPATH", "")
    py_path = f"{exp_dir}{os.pathsep}{existing}" if existing else exp_dir
    export_shell = (f"PYTHONPATH={shlex.quote(py_path)} "
                    "python3 -m llm_loader.export_all_cli "
                    f"{shlex.quote(draft_model_dir)} \"$tmp_dir\"")
    shell = ("tmp_dir=$(mktemp -d); "
             "trap 'rm -rf \"$tmp_dir\"' EXIT; "
             f"{export_shell}; "
             f"mkdir -p {shlex.quote(draft_onnx_dir)}; "
             f"cp -a \"$tmp_dir/llm/.\" {shlex.quote(draft_onnx_dir)}/")
    commands.append((["bash", "-c", shell], 600))
    return commands


def _generate_vocab_reduction_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate vocabulary reduction commands if needed"""
    commands = []
    if not config.reduced_vocab_size:
        return commands

    torch_model_dir = config.get_torch_model_dir()
    reduced_vocab_dir = config.get_reduced_vocab_dir()

    vocab_reduction_args = [
        f"--model_dir={torch_model_dir}",
        f"--output_dir={reduced_vocab_dir}",
        f"--reduced_vocab_size={config.reduced_vocab_size}",
        f"--method={config.vocab_reduction_method}",
        f"--max_samples={config.vocab_reduction_max_samples}",
    ]

    # Add d2t_path for EAGLE models
    if config.is_eagle:
        # d2t.safetensors is in the draft ONNX directory after export
        d2t_path = os.path.join(config.get_draft_onnx_dir(), "d2t.safetensors")
        vocab_reduction_args.append(f"--d2t_path={d2t_path}")

    vocab_reduction_shell = _llm_loader_module_shell(
        "llm_loader.vocab_reduction", vocab_reduction_args)
    commands.append((["bash", "-c", vocab_reduction_shell], 600))
    return commands


def can_use_llm_loader(config: TestConfig) -> bool:
    """Return True when llm_loader.export_all_cli can replace the legacy export.

    llm_loader handles: pre-quantized LLM, fp16 visual, audio/TTS, EAGLE,
    and dynamic LoRA (via ``llm_loader.lora.{insert_lora_cli,
    process_lora_weights_cli}`` as a post-export step).
    It does NOT support: trt_native_ops or fp8 visual calibration (those still
    require the legacy CLIs).
    """
    if config.trt_native_ops:
        return False
    return True


def generate_pre_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate commands that must run BEFORE the ONNX export step.

    Includes LoRA merge, EAGLE draft quantization, vocab reduction, and
    base-model quantization. When using llm_loader for the export step the
    caller runs these first, then calls llm_loader, then optionally runs
    post-export commands (e.g. fp8 visual / fp8 audio / lora insert).

    All quantization (base + draft) goes through
    ``python -m experimental.quantization``, which writes a unified ModelOpt
    HF checkpoint consumable by both ``llm_loader.export_all_cli`` and the
    legacy ``tensorrt-edgellm-export-llm`` CLI — no path-specific flags
    needed here.
    """
    commands: List[Tuple[List[str], int]] = []
    # Phi-4-Multimodal needs its required vision LoRA merged before
    # quantization/export. The merged checkpoint is then used by both
    # experimental.quantization and llm_loader.
    commands.extend(_generate_merge_lora_commands(config))
    commands.extend(_generate_draft_quantization_commands(config))
    commands.extend(
        _generate_llm_loader_draft_export_for_vocab_commands(config))
    commands.extend(_generate_vocab_reduction_commands(config))
    commands.extend(_generate_quantization_commands(config))
    return commands


def generate_post_llm_loader_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate commands that run AFTER the llm_loader export step.

    Three cases handled today:
      * fp8 visual encoder calibration for VLMs (``visual_precision == "fp8"``)
        — ``experimental.quantization`` does not yet support visual encoder
        quantization, so we still call the legacy
        ``tensorrt-edgellm-export-visual --quantization=fp8`` CLI.
      * fp8 audio encoder calibration for ASR/Omni
        (``audio_precision == "fp8"``) — same situation: legacy
        ``tensorrt-edgellm-export-audio --quantization=fp8`` is still
        needed.
      * Dynamic LoRA insertion (``config.lora``) — uses the new
        ``llm_loader.lora.{insert_lora_cli, process_lora_weights_cli}``
        modules. Mirrors the flow used by
        ``test_llm_loader_lora_export``.

    When ``merge_lora`` is set (e.g. Phi-4 with vision-lora), the merged-vision
    checkpoint is used for the fp8 visual step instead of the raw torch dir:
    the raw Phi-4 checkpoint ships a custom ``modeling_phi4mm.py`` that imports
    ``SlidingWindowCache``, a symbol no longer present in current transformers,
    while the merged dir saved via HF ``save_pretrained`` omits that custom
    module and loads via the stock HF classes.
    """
    commands: List[Tuple[List[str], int]] = []
    if config.model_type == ModelType.VLM and config.visual_precision == "fp8":
        if config.merge_lora:
            model_dir = config.get_merged_model_dir()
        else:
            model_dir = config.get_torch_model_dir()
        visual_export_cmd = [
            "tensorrt-edgellm-export-visual",
            f"--model_dir={model_dir}",
            f"--dtype=fp16",
            f"--quantization=fp8",
            f"--output_dir={config.get_visual_onnx_dir('fp8')}",
            f"--dataset_dir={config.get_mmmu_dataset_dir()}",
        ]
        commands.append((visual_export_cmd, 1200))

    if (config.model_type in (ModelType.ASR, ModelType.OMNI)
            and config.audio_precision == "fp8"):
        # NOTE: --dataset_dir is intentionally omitted. The audio calibration
        # path in tensorrt_edgellm.quantization.audio_quantization streams
        # ``openslr/librispeech_asr`` from HuggingFace (load_dataset with
        # streaming=True) and asserts the dataset name contains
        # "librispeech" — local-directory datasets are not supported. The
        # CLI default (openslr/librispeech_asr) is what the user-guide ASR
        # / TTS examples use.
        audio_export_cmd = [
            "tensorrt-edgellm-export-audio",
            f"--model_dir={config.get_torch_model_dir()}",
            f"--dtype=fp16",
            f"--quantization=fp8",
            f"--output_dir={config.get_audio_onnx_dir('fp8')}",
        ]
        # Omni bundles audio_encoder + code2wav in one model dir; restrict to
        # audio_encoder so code2wav is not re-exported here (llm_loader already
        # produced its fp16 ONNX).
        if config.model_type == ModelType.OMNI:
            audio_export_cmd.append("--export_models=audio_encoder")
        commands.append((audio_export_cmd, 1200))

    if config.lora:
        # Dynamic (text-side) LoRA insertion via the new llm_loader.lora
        # package mirrors the flow used by
        # ``test_llm_loader_lora_export`` — insert LoRA pattern nodes into
        # the exported model.onnx, then process the adapter weights into a
        # runtime-ready safetensors layout.
        insert_cmd = [
            "python3",
            "-m",
            "llm_loader.lora.insert_lora_cli",
            f"--onnx_dir={config.get_llm_onnx_dir()}",
        ]
        commands.append((insert_cmd, 120))

        if config.model_name not in AVAILABLE_LORA_WEIGHTS:
            raise ValueError(
                f"No LoRA weights available for {config.model_name}. "
                f"Please add it to AVAILABLE_LORA_WEIGHTS")
        edgellm_data_dir = os.environ.get("EDGELLM_DATA_DIR",
                                          "/scratch.edge_llm_cache")
        lora_model_name = AVAILABLE_LORA_WEIGHTS[config.model_name]
        lora_weights_dir = _find_directory(edgellm_data_dir, lora_model_name,
                                           DEFAULT_SEARCH_DEPTH)
        if not lora_weights_dir:
            raise ValueError(
                f"LoRA weights directory '{lora_model_name}' not found under "
                f"'{edgellm_data_dir}' within search depth "
                f"{DEFAULT_SEARCH_DEPTH}.")
        process_cmd = [
            "python3",
            "-m",
            "llm_loader.lora.process_lora_weights_cli",
            f"--input_dir={lora_weights_dir}",
            f"--output_dir={config.get_lora_weights_dir()}",
        ]
        commands.append((process_cmd, 120))

    return commands


def generate_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate full legacy export commands (for models that cannot use llm_loader)."""
    commands = []

    # Generate commands in order:
    # 1. Merge LoRA (if needed, e.g., Phi-4 with vision-lora)
    # 2. Quantize draft model (EAGLE only)
    # 3. Export draft model (EAGLE only, writes d2t.safetensors for vocab reduction)
    # 4. Reduce vocabulary (if needed, reads d2t.safetensors for EAGLE)
    # 5. Quantize base model (if needed)
    # 6. Export base model
    # 7. Export visual model (VLM only)
    # 8. Process LoRA (if needed)
    commands.extend(_generate_merge_lora_commands(config))
    commands.extend(_generate_draft_quantization_commands(config))
    commands.extend(_generate_draft_export_commands(config))
    commands.extend(_generate_vocab_reduction_commands(config))
    commands.extend(_generate_quantization_commands(config))
    commands.extend(_generate_llm_export_commands(config))
    commands.extend(_generate_visual_export_commands(config))
    commands.extend(_generate_lora_commands(config))

    return commands


def _generate_draft_build_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate draft model build commands for EAGLE"""
    commands = []

    if not config.is_eagle:
        return commands

    draft_cmd = [executable_files['llm_build']]
    draft_cmd.extend([
        f"--onnxDir={config.get_draft_onnx_dir()}",
        f"--engineDir={config.get_llm_engine_dir()}",
        f"--maxInputLen={config.max_input_len}",
        f"--maxKVCacheCapacity={config.max_seq_len}",
        f"--maxBatchSize={config.max_batch_size}", "--specDraft",
        f"--maxDraftTreeSize={config.max_draft_tree_size}"
    ])
    commands.append((draft_cmd, 1200))

    return commands


def generate_build_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate build commands - returns list of (command, timeout) tuples"""
    commands = []

    if config.model_type == ModelType.LLM:
        # LLM build command
        cmd = [executable_files['llm_build']]
        cmd.extend([
            f"--onnxDir={config.get_llm_onnx_dir()}",
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--maxInputLen={config.max_input_len}",
            f"--maxKVCacheCapacity={config.max_seq_len}",
            f"--maxBatchSize={config.max_batch_size}"
        ])

        if config.is_eagle:
            cmd.append("--specBase")
            cmd.append(f"--maxVerifyTreeSize={config.max_verify_tree_size}")

        if config.max_lora_rank > 0:
            cmd.append(f"--maxLoraRank={config.max_lora_rank}")

        if config.debug:
            cmd.append("--debug")

        commands.append((cmd, 1200))

    elif config.model_type == ModelType.VLM:
        # VLM LLM build command
        llm_cmd = [executable_files['llm_build']]
        llm_cmd.extend([
            f"--onnxDir={config.get_llm_onnx_dir()}",
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--maxInputLen={config.max_input_len}",
            f"--maxKVCacheCapacity={config.max_seq_len}",
            f"--maxBatchSize={config.max_batch_size}"
        ])

        if config.is_eagle:
            llm_cmd.append("--specBase")
            llm_cmd.append(
                f"--maxVerifyTreeSize={config.max_verify_tree_size}")

        if config.max_lora_rank > 0:
            llm_cmd.append(f"--maxLoraRank={config.max_lora_rank}")

        if config.debug:
            llm_cmd.append("--debug")

        commands.append((llm_cmd, 1200))

        # VLM visual build command
        visual_cmd = [executable_files['visual_build']]
        visual_cmd.extend([
            f"--onnxDir={config.get_visual_onnx_dir(config.visual_precision)}",
            f"--engineDir={config.get_visual_engine_dir()}",
            f"--minImageTokens={config.min_image_tokens}",
            f"--maxImageTokens={config.max_image_tokens}",
            f"--maxImageTokensPerImage={config.max_image_tokens_per_image}"
        ])

        if config.debug:
            visual_cmd.append("--debug")

        commands.append((visual_cmd, 1200))

    elif config.model_type == ModelType.TTS:
        # TTS: build talker + code_predictor LLM engines (under
        # ``llm-<llm_prec>-<lm_head_prec>/<talker|code_predictor>``).
        # Optional tokenizer_decoder audio engine — skipped when the legacy
        # ONNX is absent (llm_loader-based exports do not produce it).
        llm_onnx = config.get_llm_onnx_dir()
        for sub, engine_dir in (
            ("talker", config.get_talker_engine_dir()),
            ("code_predictor", config.get_code_predictor_engine_dir()),
        ):
            cmd = [executable_files['llm_build']]
            cmd.extend([
                f"--onnxDir={os.path.join(llm_onnx, sub)}",
                f"--engineDir={engine_dir}",
                f"--maxInputLen={config.max_input_len}",
                f"--maxKVCacheCapacity={config.max_seq_len}",
                f"--maxBatchSize={config.max_batch_size}",
            ])
            if config.debug:
                cmd.append("--debug")
            commands.append((cmd, 1200))

        code2wav_onnx = config.get_code2wav_onnx_dir()
        if os.path.isdir(code2wav_onnx):
            audio_cmd = [executable_files['audio_build']]
            audio_cmd.extend([
                f"--onnxDir={code2wav_onnx}",
                f"--engineDir={config.get_llm_engine_dir()}",
            ])
            if config.debug:
                audio_cmd.append("--debug")
            commands.append((audio_cmd, 1200))

        tokenizer_decoder_onnx = os.path.join(config.get_audio_onnx_dir(),
                                              "tokenizer_decoder")
        if os.path.isdir(tokenizer_decoder_onnx):
            audio_cmd = [executable_files['audio_build']]
            audio_cmd.extend([
                f"--onnxDir={tokenizer_decoder_onnx}",
                f"--engineDir={config.get_llm_engine_dir()}",
            ])
            if config.debug:
                audio_cmd.append("--debug")
            commands.append((audio_cmd, 1200))

    elif config.model_type == ModelType.OMNI:
        # OMNI: shared multimodal engine dir holds both visual + audio engines,
        # plus a separate base LLM engine.
        llm_cmd = [executable_files['llm_build']]
        llm_cmd.extend([
            f"--onnxDir={config.get_llm_onnx_dir()}",
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--maxInputLen={config.max_input_len}",
            f"--maxKVCacheCapacity={config.max_seq_len}",
            f"--maxBatchSize={config.max_batch_size}",
        ])
        if config.max_lora_rank > 0:
            llm_cmd.append(f"--maxLoraRank={config.max_lora_rank}")
        if config.debug:
            llm_cmd.append("--debug")
        commands.append((llm_cmd, 1200))

        multimodal_engine_dir = config.get_multimodal_engine_dir()
        visual_cmd = [executable_files['visual_build']]
        visual_cmd.extend([
            f"--onnxDir={config.get_visual_onnx_dir('fp16')}",
            f"--engineDir={multimodal_engine_dir}",
            f"--minImageTokens={config.min_image_tokens}",
            f"--maxImageTokens={config.max_image_tokens}",
            f"--maxImageTokensPerImage={config.max_image_tokens_per_image}",
        ])
        if config.debug:
            visual_cmd.append("--debug")
        commands.append((visual_cmd, 1200))

        audio_cmd = [executable_files['audio_build']]
        audio_cmd.extend([
            f"--onnxDir={config.get_audio_onnx_dir()}",
            f"--engineDir={multimodal_engine_dir}",
            f"--minTimeSteps={config.min_time_steps}",
            f"--maxTimeSteps={config.max_time_steps}",
        ])
        if config.debug:
            audio_cmd.append("--debug")
        commands.append((audio_cmd, 1200))

    elif config.model_type == ModelType.ASR:
        # ASR: build LLM engine + audio encoder engine.
        llm_cmd = [executable_files['llm_build']]
        llm_cmd.extend([
            f"--onnxDir={config.get_llm_onnx_dir()}",
            f"--engineDir={config.get_llm_engine_dir()}",
            f"--maxInputLen={config.max_input_len}",
            f"--maxKVCacheCapacity={config.max_seq_len}",
            f"--maxBatchSize={config.max_batch_size}",
        ])
        if config.max_lora_rank > 0:
            llm_cmd.append(f"--maxLoraRank={config.max_lora_rank}")
        if config.debug:
            llm_cmd.append("--debug")
        commands.append((llm_cmd, 1200))

        audio_cmd = [executable_files['audio_build']]
        audio_cmd.extend([
            f"--onnxDir={config.get_audio_onnx_dir()}",
            f"--engineDir={config.get_audio_engine_dir()}",
            f"--minTimeSteps={config.min_time_steps}",
            f"--maxTimeSteps={config.max_time_steps}",
        ])
        if config.debug:
            audio_cmd.append("--debug")
        commands.append((audio_cmd, 1200))

    # Add draft model build for EAGLE (must be after base model build)
    commands.extend(_generate_draft_build_commands(config, executable_files))

    return commands


def generate_inference_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate inference commands - returns list of (command, timeout) tuples"""
    commands = []

    if config.model_type == ModelType.TTS:
        cmd = [executable_files['qwen3_tts_inference']]
        cmd.extend([
            f"--talkerEngineDir={config.get_talker_engine_dir()}",
            f"--code2wavEngineDir={config.get_code2wav_engine_dir()}",
            f"--tokenizerDir={config.get_tts_tokenizer_dir()}",
            f"--inputFile={config.get_test_case_file()}",
            f"--outputFile={config.get_output_json_file()}",
            f"--outputAudioDir={config.get_output_audio_dir()}",
            "--dumpProfile",
        ])
        if config.batch_size is not None:
            cmd.append(f"--batchSize={config.batch_size}")
        if config.debug:
            cmd.append("--debug")
        commands.append((cmd, 6000))
        return commands

    cmd = [executable_files['llm_inference']]
    cmd.extend([
        f"--engineDir={config.get_llm_engine_dir()}",
        f"--inputFile={config.get_test_case_file()}",
        f"--outputFile={config.get_output_json_file()}", f"--dumpProfile"
    ])

    # Add EAGLE parameters
    if config.is_eagle:
        cmd.append("--specDecode")
        cmd.append(f"--specDraftTopK={config.eagle_draft_top_k}")
        cmd.append(f"--specDraftStep={config.eagle_draft_step}")
        cmd.append(f"--specVerifyTreeSize={config.max_verify_tree_size}")

    if config.model_type == ModelType.VLM:
        cmd.append(f"--multimodalEngineDir={config.get_visual_engine_dir()}")
    elif config.model_type == ModelType.ASR:
        cmd.append(f"--multimodalEngineDir={config.get_audio_engine_dir()}")
    elif config.model_type == ModelType.OMNI:
        cmd.append(
            f"--multimodalEngineDir={config.get_multimodal_engine_dir()}")

    # Add batch size override if specified
    if config.batch_size is not None:
        cmd.append(f"--batchSize={config.batch_size}")

    if config.debug:
        cmd.append("--debug")

    commands.append((cmd, 6000))
    return commands


def generate_e2e_bench_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate e2e benchmark commands - returns list of (command, timeout) tuples"""
    commands = []

    cmd = [executable_files['llm_inference']]
    cmd.extend([
        f"--engineDir={config.get_llm_engine_dir()}",
        f"--inputFile={config.get_test_case_file()}",
        f"--outputFile={config.get_output_json_file()}", f"--dumpProfile"
    ])

    # Add EAGLE parameters
    if config.is_eagle:
        cmd.append("--specDecode")
        cmd.append(f"--specDraftTopK={config.eagle_draft_top_k}")
        cmd.append(f"--specDraftStep={config.eagle_draft_step}")
        cmd.append(f"--specVerifyTreeSize={config.max_verify_tree_size}")

    if config.model_type == ModelType.VLM:
        cmd.append(f"--multimodalEngineDir={config.get_visual_engine_dir()}")
    elif config.model_type == ModelType.ASR:
        cmd.append(f"--multimodalEngineDir={config.get_audio_engine_dir()}")

    # Add batch size override if specified
    if config.batch_size is not None:
        cmd.append(f"--batchSize={config.batch_size}")

    # Add warmup if specified
    cmd.append(f"--warmup={config.warmup or 10}")

    if config.debug:
        cmd.append("--debug")

    commands.append((cmd, 6000))
    return commands


def generate_kernel_bench_commands(
        config: TestConfig,
        executable_files: Dict[str, str]) -> List[Tuple[List[str], int]]:
    """Generate kernel_bench commands - returns list of (command, timeout) tuples"""
    commands = []

    cmd = [executable_files['llm_bench']]
    cmd.extend([
        f"--engineDir={config.get_llm_engine_dir()}",
        f"--batchSize={config.batch_size or 1}",
        f"--warmup={config.warmup or 2}",
        f"--iterations=10",
        "--profile",
    ])

    if config.bench_mode:
        cmd.append(f"--mode={config.bench_mode}")

    if config.input_len:
        cmd.append(f"--inputLen={config.input_len}")

    if config.past_kv_len:
        cmd.append(f"--pastKVLen={config.past_kv_len}")

    if config.debug:
        cmd.append("--debug")

    commands.append((cmd, 600))
    return commands
