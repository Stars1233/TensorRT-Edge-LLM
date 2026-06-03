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
from .checkpoint_export_helpers import get_tensorrt_edgellm_root

# Available LoRA weights mapping
AVAILABLE_LORA_WEIGHTS = {
    "Qwen2.5-0.5B-Instruct": "Jailbreak-Detector-2-XL",
    "Qwen2.5-0.5B-Instruct-FP8": "Jailbreak-Detector-2-XL",
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-Diagrams2SQL-v2",
}


def _tensorrt_edgellm_module_shell(module: str, args: List[str]) -> str:
    edgellm_root = get_tensorrt_edgellm_root()
    if not edgellm_root:
        raise ValueError(
            "Cannot find tensorrt-edge-llm root. "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    existing = os.environ.get("PYTHONPATH", "")
    py_path = f"{edgellm_root}{os.pathsep}{existing}" if existing else edgellm_root
    cmd = ["python3", "-m", module] + args
    inner = " ".join(shlex.quote(x) for x in cmd)
    return f"PYTHONPATH={shlex.quote(py_path)} {inner}"


def _generate_merge_lora_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate merge LoRA commands for models with embedded LoRA (e.g., Phi-4)"""
    commands = []
    if not config.merge_lora:
        return commands

    merge_lora_shell = _tensorrt_edgellm_module_shell(
        "tensorrt_edgellm.scripts.merge_lora", [
            f"--model_dir={config.get_torch_model_dir()}",
            f"--lora_dir={config.get_lora_adapter_dir()}",
            f"--output_dir={config.get_merged_model_dir()}"
        ])
    commands.append((["bash", "-c", merge_lora_shell], 600))

    return commands


def _llm_quant_shell(
    config: TestConfig,
    input_model_dir: str,
    output_model_dir: str,
    needs_weight_quant: bool,
    needs_kv_cache_quant: bool,
    needs_visual_quant: bool,
    needs_audio_quant: bool,
) -> str:
    """``cd <sdk> && tensorrt-edgellm-quantize llm ...`` (unified ModelOpt export)."""
    edgellm_root = get_tensorrt_edgellm_root()
    if not edgellm_root:
        raise ValueError(
            "Cannot find tensorrt-edge-llm root. "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    args: List[str] = [
        "python3",
        "-m",
        "tensorrt_edgellm.scripts.quantize",
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
    if needs_visual_quant:
        args.append("--visual_quantization=fp8")
    if needs_audio_quant:
        args.append("--audio_quantization=fp8")
    inner = " ".join(shlex.quote(x) for x in args)
    return f"cd {shlex.quote(edgellm_root)} && {inner}"


def _generate_quantization_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate quantization commands if needed.

    Quantization writes a unified ModelOpt checkpoint that
    ``tensorrt_edgellm.scripts.export`` knows how to unpack.

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
    needs_visual_quant = bool(config.visual_precision == "fp8")
    needs_audio_quant = bool(config.audio_precision == "fp8")
    if (needs_weight_quant or needs_kv_cache_quant or needs_visual_quant
            or needs_audio_quant):
        # Use the merged checkpoint when a model ships a required LoRA
        # adapter, otherwise use the raw torch checkpoint.
        if config.merge_lora:
            input_model_dir = config.get_merged_model_dir()
        else:
            input_model_dir = config.get_torch_model_dir()

        if needs_weight_quant or needs_visual_quant or needs_audio_quant:
            output_model_dir = config.get_quantized_model_dir()
        else:
            # KV-cache-only quantization (fp16 weights)
            output_model_dir = config.get_kv_cache_quantized_model_dir()

        shell = _llm_quant_shell(config, input_model_dir, output_model_dir,
                                 needs_weight_quant, needs_kv_cache_quant,
                                 needs_visual_quant, needs_audio_quant)
        commands.append((["bash", "-c", shell], 1200))

    return commands


def _draft_quant_shell(config: TestConfig) -> str:
    """``cd <sdk> && tensorrt-edgellm-quantize draft ...``"""
    edgellm_root = get_tensorrt_edgellm_root()
    if not edgellm_root:
        raise ValueError(
            "Cannot find tensorrt-edge-llm root. "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    base_model_dir = config.get_torch_model_dir()
    draft_model_dir = config.get_draft_model_dir()
    quantized_draft_dir = config.get_quantized_draft_model_dir()
    args: List[str] = [
        "python3",
        "-m",
        "tensorrt_edgellm.scripts.quantize",
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

    Uses ``tensorrt-edgellm-quantize``. Output is a unified ModelOpt
    ``export_hf_checkpoint`` tree consumable by ``tensorrt_edgellm.scripts.export``.
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
        shell = _draft_quant_shell(config)
        commands.append((["bash", "-c", shell], 900))

    return commands


def _generate_tensorrt_edgellm_draft_export_for_vocab_commands(
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
            "Cannot find tensorrt-edge-llm root. "
            "Set LLM_SDK_DIR to the SDK root, or run from a full tensorrt-edge-llm tree."
        )
    existing = os.environ.get("PYTHONPATH", "")
    py_path = f"{edgellm_root}{os.pathsep}{existing}" if existing else edgellm_root
    export_shell = (f"PYTHONPATH={shlex.quote(py_path)} "
                    "python3 -m tensorrt_edgellm.scripts.export "
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

    vocab_reduction_shell = _tensorrt_edgellm_module_shell(
        "tensorrt_edgellm.scripts.reduce_vocab", vocab_reduction_args)
    commands.append((["bash", "-c", vocab_reduction_shell], 600))
    return commands


def generate_pre_export_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate commands that must run BEFORE the ONNX export step.

    Includes LoRA merge, EAGLE draft quantization, vocab reduction, and
    base-model quantization. The caller runs these first, then calls the
    checkpoint exporter, then optionally runs post-export commands such as
    dynamic LoRA insertion.

    All quantization (base + draft) goes through
    ``tensorrt-edgellm-quantize``, which writes a unified ModelOpt
    HF checkpoint consumable by ``tensorrt_edgellm.scripts.export``.
    """
    commands: List[Tuple[List[str], int]] = []
    # Phi-4-Multimodal needs its required vision LoRA merged before
    # quantization/export. The merged checkpoint is then used by
    # ``tensorrt-edgellm-quantize`` and the checkpoint exporter.
    commands.extend(_generate_merge_lora_commands(config))
    commands.extend(_generate_draft_quantization_commands(config))
    commands.extend(
        _generate_tensorrt_edgellm_draft_export_for_vocab_commands(config))
    commands.extend(_generate_vocab_reduction_commands(config))
    commands.extend(_generate_quantization_commands(config))
    return commands


def generate_post_tensorrt_edgellm_commands(
        config: TestConfig) -> List[Tuple[List[str], int]]:
    """Generate commands that run after the checkpoint export step.

    Dynamic LoRA insertion uses
    ``tensorrt-edgellm-insert-lora`` and ``tensorrt-edgellm-process-lora``.
    Visual/audio FP8 calibration is handled before export by
    ``tensorrt-edgellm-quantize``.
    """
    commands: List[Tuple[List[str], int]] = []
    if config.lora:
        # Dynamic LoRA insertion mirrors ``test_checkpoint_lora_export``:
        # insert LoRA pattern nodes into the exported model.onnx, then process
        # the adapter weights into a runtime-ready safetensors layout.
        insert_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.insert_lora",
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
            "tensorrt_edgellm.scripts.process_lora_weights",
            f"--input_dir={lora_weights_dir}",
            f"--output_dir={config.get_lora_weights_dir()}",
        ]
        commands.append((process_cmd, 120))

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
        # Optional tokenizer_decoder audio engine. The checkpoint exporter does
        # not emit this component.
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
    elif config.model_type == ModelType.OMNI:
        cmd.append(
            f"--multimodalEngineDir={config.get_multimodal_engine_dir()}")

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
