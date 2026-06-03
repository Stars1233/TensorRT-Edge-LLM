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
Test configuration utilities for TensorRT Edge-LLM tests.

This module provides a unified TestConfig class that handles both export and runtime 
test configurations, eliminating redundancy across the test framework.
"""

import enum
import os
from dataclasses import dataclass
from typing import Optional, Set

from conftest import EnvironmentConfig
from valid_precisions import (VALID_AUDIO_PRECISIONS, VALID_LLM_PRECISIONS,
                              VALID_LM_HEAD_PRECISIONS,
                              VALID_VISUAL_PRECISIONS)

# Global configuration constants
DEFAULT_SEARCH_DEPTH = 3

# Models that ship pre-quantized (skip tensorrt-edgellm-quantize step).
# These are exported directly from the HF checkpoint without a quantization step.
PRE_QUANTIZED_MODELS: frozenset = frozenset({
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "Qwen3-30B-A3B-NVFP4",
})


def _find_directory(
    root_dir: str,
    target_name: str,
    max_depth: Optional[int] = None,
    require_files: Optional[list] = None,
) -> Optional[str]:
    """Search for a directory with the given name or path.

    Args:
        root_dir: Root directory to start searching from.
        target_name: Target directory name or relative path to find
            (e.g. ``"model"`` or ``"parent/model"``).
        max_depth: Maximum search depth (``None`` for unlimited, ``1`` for
            immediate children only).
        require_files: Optional list of file requirements. Each entry may
            be either a single glob (``"config.json"``) — that pattern
            must match at least one file — or a list of globs
            (``["model*.safetensors", "pytorch_model*.bin"]``) — at least
            one of those globs must match. All outer entries must be
            satisfied. Used to disambiguate HF checkpoint dirs from
            same-named engine cache / ONNX output dirs that happen to be
            deeper in the tree.

    Returns:
        Full path to the first matching directory found, or None if not
        found.
    """
    if root_dir is None or not os.path.exists(root_dir):
        return None

    import glob

    def _is_valid_match(candidate: str) -> bool:
        if not require_files:
            return True
        for pattern in require_files:
            if isinstance(pattern, (list, tuple)):
                # OR: at least one of these globs must match.
                if not any(
                        glob.glob(os.path.join(candidate, p))
                        for p in pattern):
                    return False
            else:
                if not glob.glob(os.path.join(candidate, pattern)):
                    return False
        return True

    def _search(current_dir: str, current_depth: int) -> Optional[str]:
        if max_depth is not None and current_depth > max_depth:
            return None

        try:
            entries = os.listdir(current_dir)
        except PermissionError:
            return None

        candidate_path = os.path.join(current_dir, target_name)
        if os.path.isdir(candidate_path) and _is_valid_match(candidate_path):
            return candidate_path

        if max_depth is None or current_depth < max_depth:
            for entry in entries:
                if entry == '.git':
                    continue
                entry_path = os.path.join(current_dir, entry)

                if os.path.isdir(entry_path):
                    result = _search(entry_path, current_depth + 1)
                    if result:
                        return result

        return None

    return _search(root_dir, 0)


# A real HF checkpoint dir has config.json plus at least one weight file
# (any *.safetensors or *.bin). Used by _find_directory.require_files to
# distinguish HF checkpoints from engine-cache dirs (no config.json).
_HF_CHECKPOINT_FILES = [
    "config.json",
    ["*.safetensors", "*.bin"],
]


class ModelType(enum.Enum):
    """Supported model types"""
    LLM = "llm"
    VLM = "vlm"
    TTS = "tts"
    ASR = "asr"
    OMNI = "omni"


class TaskType(enum.Enum):
    """Supported task types"""
    EXPORT = "export"
    BUILD = "build"
    E2E_BENCH = "e2e_bench"
    INFERENCE = "inference"
    KERNEL_BENCH = "kernel_bench"


@dataclass
class ParameterSpec:
    """Specification for a parameter with validation rules"""
    name: str
    format_hint: str
    required_for: Set[TaskType]
    valid_for: Set[ModelType]
    is_required: bool = True

    def is_valid_for_task_and_model(self, task_type: TaskType,
                                    model_type: ModelType) -> bool:
        """Check if this parameter is valid for the given task and model type"""
        return task_type in self.required_for and model_type in self.valid_for

    def is_required_for_task_and_model(self, task_type: TaskType,
                                       model_type: ModelType) -> bool:
        """Check if this parameter is required for the given task and model type"""
        return self.is_valid_for_task_and_model(
            task_type, model_type) and self.is_required


@dataclass
class TestConfig:
    """
    Test configuration class that handles both export and runtime test configurations.
    """
    # Core identifiers
    param_str: str
    model_name: str
    model_type: ModelType
    task_type: TaskType

    # Precision settings
    llm_precision: str
    lm_head_precision: Optional[str] = None
    visual_precision: Optional[str] = None
    audio_precision: Optional[str] = None

    # EAGLE draft model settings
    draft_model_name: Optional[str] = None
    draft_model_id: Optional[str] = None
    draft_llm_precision: Optional[str] = None
    draft_lm_head_precision: Optional[str] = None

    # Speculative decoding flags
    is_eagle: Optional[bool] = None
    is_mtp: Optional[bool] = None

    # Directory paths
    llm_models_dir: Optional[str] = None
    edgellm_data_dir: Optional[str] = None
    onnx_dir: Optional[str] = None
    engine_dir: Optional[str] = None
    test_log_dir: Optional[str] = None

    # Export LoRA parameters
    lora: Optional[bool] = None
    merge_lora: Optional[bool] = None

    # Engine build parameters
    max_batch_size: Optional[int] = None
    max_input_len: Optional[int] = None
    max_seq_len: Optional[int] = None
    max_lora_rank: Optional[int] = None

    # EAGLE specific build parameters
    max_verify_tree_size: Optional[int] = None
    max_draft_tree_size: Optional[int] = None

    # EAGLE inference parameters
    eagle_draft_top_k: Optional[int] = None
    eagle_draft_step: Optional[int] = None

    # VLM specific build parameters
    min_image_tokens: Optional[int] = None
    max_image_tokens: Optional[int] = None
    max_image_tokens_per_image: Optional[int] = None

    # ASR specific build parameters
    min_time_steps: Optional[int] = None
    max_time_steps: Optional[int] = None

    # Inference parameters
    test_case: Optional[str] = None

    # Path of a per-config preprocessed copy of the test case JSON. Set by
    # tests/defs/utils/command_execution.py helpers when they need to rewrite
    # audio paths or substitute LoRA placeholders without mutating the
    # version-controlled source under tests/test_cases/. ``get_test_case_file``
    # returns this when populated so downstream command generation
    # transparently picks up the rewritten document.
    _test_case_file_override: Optional[str] = None

    # KV cache options
    fp8_kv_cache: Optional[
        bool] = None  # If true, export ONNX/config with FP8 KV cache enabled

    # Embedding options
    fp8_embedding: Optional[
        bool] = None  # If true, write embedding.safetensors in FP8 E4M3 format

    # Benchmark parameters
    batch_size: Optional[int] = None
    input_seq_len: Optional[int] = None
    output_seq_len: Optional[int] = None

    # VLM-specific parameters (no defaults - must be specified)
    text_token_length: Optional[int] = None
    image_token_length: Optional[int] = None

    # Vocabulary reduction parameters
    reduced_vocab_size: Optional[int] = None
    vocab_reduction_method: Optional[
        str] = None  # "input_aware" or "frequency"
    vocab_reduction_max_samples: Optional[int] = None

    warmup: Optional[int] = None

    # kernel_bench parameters
    bench_mode: Optional[str] = None
    input_len: Optional[int] = None
    past_kv_len: Optional[int] = None

    # Debug flag for verbose output
    debug: Optional[bool] = None

    # Declarative parameter specifications
    _PARAMETER_SPECS = [
        # Core parameters for engine identification
        ParameterSpec(
            "max_batch_size", "mxbs", {
                TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE,
                TaskType.KERNEL_BENCH
            }, {
                ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR,
                ModelType.OMNI
            }),
        ParameterSpec(
            "max_input_len", "mxil", {
                TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE,
                TaskType.KERNEL_BENCH
            }, {
                ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR,
                ModelType.OMNI
            }),
        ParameterSpec(
            "max_seq_len", "mxsl", {
                TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE,
                TaskType.KERNEL_BENCH
            }, {
                ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR,
                ModelType.OMNI
            }),
        ParameterSpec("max_lora_rank",
                      "mxlr",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {
                          ModelType.LLM, ModelType.VLM, ModelType.TTS,
                          ModelType.ASR, ModelType.OMNI
                      },
                      is_required=False),

        # Export-specific parameters
        ParameterSpec("lora",
                      "", {TaskType.EXPORT}, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("merge_lora",
                      "", {TaskType.EXPORT}, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec(
            "fp8_kv_cache",
            "fp8kv", {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec(
            "fp8_embedding",
            "fp8emb",
            {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.ASR, ModelType.OMNI},
            is_required=False),
        ParameterSpec(
            "is_eagle",
            "eagle", {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec("is_mtp",
                      "mtp", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("draft_model_id",
                      "", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("draft_llm_precision",
                      "", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("draft_lm_head_precision",
                      "", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("max_verify_tree_size",
                      "mvts",
                      {TaskType.BUILD, TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("max_draft_tree_size",
                      "mdts",
                      {TaskType.BUILD, TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("eagle_draft_top_k",
                      "edtk", {TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("eagle_draft_step",
                      "edst", {TaskType.INFERENCE, TaskType.E2E_BENCH},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),

        # ASR-specific parameters
        ParameterSpec("min_time_steps", "mnts",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.ASR, ModelType.OMNI}),
        ParameterSpec("max_time_steps", "mxts",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.ASR, ModelType.OMNI}),

        # VLM-specific parameters
        ParameterSpec("min_image_tokens", "mnit",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.VLM, ModelType.OMNI}),
        ParameterSpec("max_image_tokens", "mxit",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.VLM, ModelType.OMNI}),
        # mxpiit is optional — most VLM/OMNI test cases use the same value
        # (512), so set_defaults() falls back to 512 when not in the param
        # string. Override per-test by including ``-mxpiit<N>``.
        ParameterSpec("max_image_tokens_per_image",
                      "mxpiit",
                      {TaskType.BUILD, TaskType.E2E_BENCH, TaskType.INFERENCE},
                      {ModelType.VLM, ModelType.OMNI},
                      is_required=False),
        ParameterSpec("visual_precision",
                      "vit", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.VLM, ModelType.OMNI},
                      is_required=False),
        ParameterSpec("audio_precision",
                      "aud", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                          TaskType.INFERENCE
                      }, {ModelType.TTS, ModelType.ASR, ModelType.OMNI},
                      is_required=False),

        # Inference/Benchmark parameters
        ParameterSpec("test_case", "",
                      {TaskType.INFERENCE, TaskType.E2E_BENCH}, {
                          ModelType.LLM, ModelType.VLM, ModelType.TTS,
                          ModelType.ASR, ModelType.OMNI
                      }),
        ParameterSpec("batch_size",
                      "bs", {TaskType.E2E_BENCH, TaskType.INFERENCE}, {
                          ModelType.LLM, ModelType.VLM, ModelType.TTS,
                          ModelType.ASR, ModelType.OMNI
                      },
                      is_required=False),

        # Vocabulary reduction parameters
        ParameterSpec(
            "reduced_vocab_size",
            "rvs", {
                TaskType.EXPORT, TaskType.BUILD, TaskType.E2E_BENCH,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec(
            "vocab_reduction_method",
            "vrm", {TaskType.EXPORT},
            {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        ParameterSpec(
            "vocab_reduction_max_samples",
            "vrms", {TaskType.EXPORT},
            {ModelType.LLM, ModelType.VLM, ModelType.TTS, ModelType.ASR},
            is_required=False),
        # kernel_bench parameters
        ParameterSpec("bench_mode",
                      "mode", {TaskType.KERNEL_BENCH}, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("input_len",
                      "il", {TaskType.KERNEL_BENCH}, {ModelType.LLM},
                      is_required=False),
        ParameterSpec("past_kv_len",
                      "pkv", {TaskType.KERNEL_BENCH}, {ModelType.LLM},
                      is_required=False),
    ]

    @classmethod
    def from_param_string(cls, param_str: str, model_type: ModelType,
                          task_type: TaskType,
                          env_config: EnvironmentConfig) -> 'TestConfig':
        """
        Unified function to parse parameter string and create config with validation.
        
        Handles model names with multiple parts, performs all validation uniformly,
        and constructs the final config object.
        """

        # Validate environment based on task type
        if task_type == TaskType.EXPORT:
            env_config.validate_for_export_tests()
        else:
            env_config.validate_for_pipeline_tests()

        # Parse parameter string
        parts = param_str.split('-')

        # Find precision position and extract model name
        model_parts = []
        llm_precision = None
        lm_head_precision = None
        visual_precision = None
        remaining_parts = []

        # Find the first valid precision to determine where model name ends
        for i, part in enumerate(parts):
            if part in VALID_LLM_PRECISIONS:
                llm_precision = part
                model_parts = parts[:
                                    i]  # Everything before precision is model name

                # Check for lm head precision
                if i + 1 < len(parts) and parts[i + 1].startswith('lm'):
                    lm_head_precision = parts[i + 1][2:]
                    if lm_head_precision not in VALID_LM_HEAD_PRECISIONS:
                        raise ValueError(
                            f"Invalid LM head precision: {lm_head_precision}")
                    remaining_parts = parts[i + 2:]
                else:
                    remaining_parts = parts[i + 1:]
                break
        if not lm_head_precision:
            lm_head_precision = "fp16"

        if not llm_precision:
            raise ValueError(f"No valid precision found in: {param_str}")

        if not model_parts:
            raise ValueError(f"No model name found in: {param_str}")

        model_name = '-'.join(model_parts)

        # Parse remaining parameters
        parsed_params = {}

        i = 0
        while i < len(remaining_parts):
            part = remaining_parts[i]

            # For engine identification
            if part.startswith('mxsl'):
                parsed_params['max_seq_len'] = int(part[4:])
            elif part == "lora":
                parsed_params['lora'] = True
            elif part == "merge_lora":
                parsed_params['merge_lora'] = True
            elif part == "fp8kv":
                parsed_params['fp8_kv_cache'] = True
            elif part == "fp8emb":
                parsed_params['fp8_embedding'] = True
            elif part == "mtp":
                parsed_params['is_mtp'] = True
                parsed_params['is_eagle'] = True
            elif part == "eagle":
                parsed_params['is_eagle'] = True
                # Parse eagle-{draft_id}-{draft_precision}[-lm{draft_lm_head}]
                # Next part should be draft_model_id
                if i + 1 < len(remaining_parts):
                    i += 1
                    parsed_params['draft_model_id'] = remaining_parts[i]

                    # Next part should be draft_llm_precision
                    if i + 1 < len(remaining_parts):
                        i += 1
                        draft_precision = remaining_parts[i]

                        # Check if it starts with 'lm' - this would be draft lm_head
                        if draft_precision.startswith('lm'):
                            raise ValueError(
                                f"Missing draft precision after draft_model_id in: {param_str}"
                            )

                        # Validate draft precision
                        if draft_precision not in VALID_LLM_PRECISIONS:
                            raise ValueError(
                                f"Invalid draft precision: {draft_precision}")
                        parsed_params['draft_llm_precision'] = draft_precision

                        # Check for optional draft lm_head precision
                        if i + 1 < len(remaining_parts) and remaining_parts[
                                i + 1].startswith('lm'):
                            i += 1
                            draft_lm_precision = remaining_parts[i][2:]
                            if draft_lm_precision not in VALID_LM_HEAD_PRECISIONS:
                                raise ValueError(
                                    f"Invalid draft LM head precision: {draft_lm_precision}"
                                )
                            parsed_params[
                                'draft_lm_head_precision'] = draft_lm_precision
            elif part.startswith('mxbs'):
                parsed_params['max_batch_size'] = int(part[4:])
            elif part.startswith('mxil'):
                parsed_params['max_input_len'] = int(part[4:])
            elif part.startswith('mnts'):
                parsed_params['min_time_steps'] = int(part[4:])
            elif part.startswith('mxts'):
                parsed_params['max_time_steps'] = int(part[4:])
            elif part.startswith('mnit'):
                parsed_params['min_image_tokens'] = int(part[4:])
            elif part.startswith('mxit'):
                parsed_params['max_image_tokens'] = int(part[4:])
            elif part.startswith('mxpiit'):
                parsed_params['max_image_tokens_per_image'] = int(part[6:])
            elif part.startswith('mxlr'):
                parsed_params['max_lora_rank'] = int(part[4:])
            # For benchmark parameters
            elif part.startswith('bs'):
                parsed_params['batch_size'] = int(part[2:])
            elif part.startswith('isl'):
                parsed_params['input_seq_len'] = int(part[3:])
            elif part.startswith('osl'):
                parsed_params['output_seq_len'] = int(part[3:])
            elif part.startswith('ttl'):
                parsed_params['text_token_length'] = int(part[3:])
            elif part.startswith('itl'):
                parsed_params['image_token_length'] = int(part[3:])
            elif part.startswith('vit'):
                visual_precision = part[3:]
                if visual_precision in VALID_VISUAL_PRECISIONS:
                    parsed_params['visual_precision'] = visual_precision
                else:
                    raise ValueError(
                        f"Invalid visual precision: {visual_precision}")
            elif part.startswith('aud'):
                audio_precision = part[3:]
                if audio_precision in VALID_AUDIO_PRECISIONS:
                    parsed_params['audio_precision'] = audio_precision
                else:
                    raise ValueError(
                        f"Invalid audio precision: {audio_precision}")
            # For EAGLE parameters
            elif part.startswith('mvts'):
                parsed_params['max_verify_tree_size'] = int(part[4:])
            elif part.startswith('mdts'):
                parsed_params['max_draft_tree_size'] = int(part[4:])
            elif part.startswith('edtk'):
                parsed_params['eagle_draft_top_k'] = int(part[4:])
            elif part.startswith('edst'):
                parsed_params['eagle_draft_step'] = int(part[4:])
            # For vocabulary reduction parameters
            elif part.startswith('rvs'):
                parsed_params['reduced_vocab_size'] = int(part[3:])
            elif part.startswith('vrm'):
                parsed_params['vocab_reduction_method'] = part[3:]
            elif part.startswith('vrms'):
                parsed_params['vocab_reduction_max_samples'] = int(part[4:])
            # For kernel_bench parameters
            elif part.startswith('mode_'):
                parsed_params['bench_mode'] = part[5:]
            elif part.startswith('il') and part[2:].isdigit():
                parsed_params['input_len'] = int(part[2:])
            elif part.startswith('pkv') and part[3:].isdigit():
                parsed_params['past_kv_len'] = int(part[3:])
            else:
                parsed_params['test_case'] = part

            i += 1

        if not visual_precision and model_type in (ModelType.VLM,
                                                   ModelType.OMNI):
            parsed_params['visual_precision'] = "fp16"
        if model_type in (ModelType.TTS, ModelType.ASR, ModelType.OMNI
                          ) and 'audio_precision' not in parsed_params:
            parsed_params['audio_precision'] = "fp16"

        # Create base config object
        config = cls(param_str=param_str,
                     model_name=model_name,
                     model_type=model_type,
                     task_type=task_type,
                     llm_precision=llm_precision,
                     lm_head_precision=lm_head_precision,
                     llm_models_dir=env_config.llm_models_dir
                     if task_type == TaskType.EXPORT else None,
                     edgellm_data_dir=env_config.edgellm_data_dir,
                     onnx_dir=env_config.onnx_dir,
                     engine_dir=env_config.engine_dir
                     if task_type != TaskType.EXPORT else None,
                     test_log_dir=env_config.test_log_dir)

        # Apply validated parameters to config
        for param_name, value in parsed_params.items():
            setattr(config, param_name, value)

        # Validate completeness and set defaults
        config._validate_completeness()

        return config

    def _validate_completeness(self) -> None:
        """Validate that all required parameters are set for the given task and model type"""

        def set_defaults() -> None:
            """Set default values for optional parameters"""
            if self.task_type == TaskType.EXPORT:
                if self.lora is None:
                    self.lora = False
                if self.merge_lora is None:
                    self.merge_lora = False
                if self.fp8_kv_cache is None:
                    self.fp8_kv_cache = False
                if self.is_eagle is None:
                    self.is_eagle = False
                if self.is_mtp is None:
                    self.is_mtp = False
                if self.draft_llm_precision is not None and self.draft_lm_head_precision is None:
                    self.draft_lm_head_precision = "fp16"
                if self.reduced_vocab_size is not None:
                    if self.vocab_reduction_method is None:
                        self.vocab_reduction_method = "input_aware"
                    if self.vocab_reduction_max_samples is None:
                        self.vocab_reduction_max_samples = 50000
            else:  # Runtime tasks
                if self.max_lora_rank is None:
                    self.max_lora_rank = 0
                if self.lora is None:
                    self.lora = self.max_lora_rank > 0
                if self.fp8_kv_cache is None:
                    self.fp8_kv_cache = False
                # max_image_tokens_per_image: VLM/OMNI default. Most test
                # cases use 512; override via ``-mxpiit<N>`` in the param.
                if (self.model_type in (ModelType.VLM, ModelType.OMNI)) and (
                        self.max_image_tokens_per_image is None):
                    self.max_image_tokens_per_image = 512
                if self.is_eagle is None:
                    self.is_eagle = False
                if self.is_mtp is None:
                    self.is_mtp = False
                if self.draft_llm_precision is not None and self.draft_lm_head_precision is None:
                    self.draft_lm_head_precision = "fp16"
                if self.eagle_draft_top_k is None:
                    self.eagle_draft_top_k = 1 if self.is_mtp else 10
                if self.eagle_draft_step is None:
                    self.eagle_draft_step = 3 if self.is_mtp else 6
                if self.max_verify_tree_size is None:
                    self.max_verify_tree_size = 4 if self.is_mtp else 60
                if self.max_draft_tree_size is None:
                    self.max_draft_tree_size = 4 if self.is_mtp else 60

        warmup_env = os.environ.get('WARMUP')
        if warmup_env is not None and self.warmup is None:
            self.warmup = int(warmup_env)

        # Read debug flag from environment variable
        debug_env = os.environ.get('DEBUG')
        if debug_env is not None and self.debug is None:
            self.debug = debug_env.lower() in ('1', 'true', 'yes')

        missing_params = []
        invalid_params = []

        # Get valid and required parameters for current task/model combination
        valid_params = set()
        required_params = set()

        for spec in self._PARAMETER_SPECS:
            if spec.is_valid_for_task_and_model(self.task_type,
                                                self.model_type):
                valid_params.add(spec.name)
                if spec.is_required_for_task_and_model(self.task_type,
                                                       self.model_type):
                    required_params.add(spec.name)

        # Check for invalid parameters (parameters that are set but not allowed)
        for spec in self._PARAMETER_SPECS:
            param_value = getattr(self, spec.name)
            if param_value is not None and spec.name not in valid_params:
                invalid_params.append(spec.name)

        if invalid_params:
            task_desc = f"{self.model_type.value} {self.task_type.value}"
            raise ValueError(
                f"Invalid parameters for {task_desc}: {', '.join(invalid_params)}"
            )

        # Check for missing required parameters
        for spec in self._PARAMETER_SPECS:
            if spec.name in required_params and getattr(self,
                                                        spec.name) is None:
                param_desc = f'{spec.name} ({spec.format_hint})' if spec.format_hint else spec.name
                missing_params.append(param_desc)

        # Raise error if any required parameters are missing
        if missing_params:
            task_desc = f"{self.model_type.value} {self.task_type.value}"
            raise ValueError(
                f"Missing required parameters for {task_desc}: {', '.join(missing_params)}"
            )

        # Set defaults after validation
        set_defaults()

    # Unified path generation methods
    def get_onnx_model_id(self) -> str:
        """Generate unique model identifier"""
        model_id = f"{self.llm_precision}-{self.lm_head_precision}"
        if self.fp8_kv_cache:
            model_id += "-fp8kv"
        if self.reduced_vocab_size:
            model_id += f"-rvs{self.reduced_vocab_size}"
        return model_id

    def get_engine_id(self) -> str:
        """Generate unique engine identifier"""
        mxlr = self.max_lora_rank if self.max_lora_rank is not None else 0
        llm_engine_id = f"{self.get_onnx_model_id()}-mxil{self.max_input_len}-mxbs{self.max_batch_size}-mxlr{mxlr}"
        if self.model_type == ModelType.VLM:
            llm_engine_id += f"-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}"
        if self.model_type == ModelType.OMNI:
            llm_engine_id += (
                f"-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}"
                f"-mnts{self.min_time_steps}-mxts{self.max_time_steps}")
        if self.is_eagle:
            if self.max_verify_tree_size is not None:
                llm_engine_id += f"-mvts{self.max_verify_tree_size}"
            if self.max_draft_tree_size is not None:
                llm_engine_id += f"-mdts{self.max_draft_tree_size}"
        return llm_engine_id

    def get_torch_model_dir(self) -> str:
        """
        Get torch model directory path using dynamic search.
        
        Searches for the model directory under llm_models_dir.
        
        Raises:
            ValueError: If llm_models_dir is not set or model directory is not found
        """

        # Models in llm_models_dir (/scratch.trt_llm_data/llm-models)
        LLM_MODELS_DIR_MAP = {
            "Qwen2.5-0.5B-Instruct":
            "Qwen2.5-0.5B-Instruct",
            "Qwen2.5-1.5B-Instruct":
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct":
            "Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct":
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-VL-3B-Instruct":
            "Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct":
            "Qwen2.5-VL-7B-Instruct",
            "Qwen2-VL-2B-Instruct":
            "Qwen2-VL-2B-Instruct",
            "InternVL3-1B":
            "InternVL3-1B-hf",
            "InternVL3-2B":
            "InternVL3-2B-hf",
            "Llama-3.1-8B-Instruct":
            "llama-3.1-model/Llama-3.1-8B-Instruct",
            "Llama-3.2-1B":
            "llama-3.2-models/Llama-3.2-1B",
            "Llama-3.2-3B":
            "llama-3.2-models/Llama-3.2-3B",
            "Qwen3-0.6B":
            "Qwen3/Qwen3-0.6B",
            "Qwen3-1.7B":
            "Qwen3/Qwen3-1.7B",
            "Qwen3-8B":
            "Qwen3/Qwen3-8B",
            "Qwen3-4B-Instruct-2507":
            "Qwen3/Qwen3-4B-Instruct-2507",
            "Qwen3-VL-2B-Instruct":
            "Qwen3/Qwen3-VL-2B-Instruct",
            "Qwen3-VL-4B-Instruct":
            "Qwen3/Qwen3-VL-4B-Instruct",
            "Qwen3-VL-8B-Instruct":
            "Qwen3/Qwen3-VL-8B-Instruct",
            "Qwen3.5-0.8B":
            "Qwen3.5-0.8B",
            "Qwen3.5-2B":
            "Qwen3.5-2B",
            "Qwen3.5-4B":
            "Qwen3.5-4B",
            "Qwen3.5-9B":
            "Qwen3.5-9B",
            "Qwen3.5-27B":
            "Qwen3.5-27B",
            "Phi-4-multimodal-instruct":
            "Phi-4-multimodal-instruct",
            # Pre-quantized models in llm_models_dir
            "Llama-3.2-1B-FP8":
            "llama-3.2-models/Llama-3.2-1B-FP8",
            "Phi-4-FP8":
            "Phi-4-FP8",
            "Phi-4-multimodal-instruct-FP8":
            "Phi-4-multimodal-instruct-FP8",
            # ASR and TTS models
            "Qwen3-ASR-0.6B":
            "Qwen3/Qwen3-ASR-0.6B",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice":
            "Qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            # Nemotron-H 30B (BF16 base + pre-quantized NVFP4)
            "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
            "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            # Pre-quantized NVFP4 model: exported directly without quantization step
            "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4":
            "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "NVIDIA-Nemotron-3-Nano-4B-BF16":
            "NVIDIA-Nemotron-3-Nano-4B-BF16",
            "NVIDIA-Nemotron-3-Nano-4B-FP8":
            "NVIDIA-Nemotron-3-Nano-4B-FP8",
            # Nemotron-Nano 9B v2 family (BF16 base + FP8/NVFP4 pre-quantized
            # variants all live under llm_models_dir/, not edge_llm_cache/).
            "NVIDIA-Nemotron-Nano-9B-v2":
            "NVIDIA-Nemotron-Nano-9B-v2",
            "NVIDIA-Nemotron-Nano-9B-v2-FP8":
            "NVIDIA-Nemotron-Nano-9B-v2-FP8",
            "NVIDIA-Nemotron-Nano-9B-v2-NVFP4":
            "NVIDIA-Nemotron-Nano-9B-v2-NVFP4",
            # Cosmos VLM
            "Cosmos-Reason2-8B":
            "Cosmos-Reason2-8B",
            # Qwen3.5 35B-A3B (BF16 base; GPTQ-Int4 variant lives in GPTQ map)
            "Qwen3.5-35B-A3B":
            "Qwen3.5-35B-A3B",
            # ASR / TTS larger variants (1.7B family)
            "Qwen3-ASR-1.7B":
            "Qwen3/Qwen3-ASR-1.7B",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice":
            "Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        }

        # GPTQ and pre-quantized models in edgellm_data_dir (/scratch.edge_llm_cache)
        GPTQ_MODELS_DIR_MAP = {
            "Qwen2.5-7B-Instruct-GPTQ-Int4": "Qwen2.5-7B-Instruct-GPTQ-Int4",
            "InternVL3-1B-GPTQ-Int4": "InternVL3-1B-hf-GPTQ-Int4",
            # GPTQ-Int4 large MoE variants
            "Qwen3-30B-A3B-GPTQ-Int4": "Qwen3-30B-A3B-GPTQ-Int4",
            # NVFP4 MoE (pre-quantized, no quantization step needed)
            "Qwen3-30B-A3B-NVFP4": "Qwen3-30B-A3B-NVFP4",
            "Qwen3.5-35B-A3B-GPTQ-Int4": "Qwen3.5-35B-A3B-GPTQ-Int4",
            # Multimodal pre-quantized NVFP4 (LLM + visual + audio).  Test list
            # uses ``Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`` as the
            # canonical name; verify the on-disk dir matches before running.
            "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4":
            "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
            "NVIDIA-Nemotron-3-Nano-4B-NVFP4":
            "NVIDIA-Nemotron-3-Nano-4B-NVFP4",
            # Pre-quantized unified checkpoints (edge_llm_cache/quantized_models/)
            "Qwen2.5-0.5B-Instruct-FP8": "Qwen2.5-0.5B-Instruct-FP8",
            "Qwen2.5-0.5B-Instruct-FP8-KV": "Qwen2.5-0.5B-Instruct-FP8-KV",
            "Qwen2.5-0.5B-Instruct-NVFP4": "Qwen2.5-0.5B-Instruct-NVFP4",
            "Qwen3-0.6B-FP8": "Qwen3-0.6B-FP8",
            "Qwen3-0.6B-INT8-SQ": "Qwen3-0.6B-INT8-SQ",
            "Qwen3-1.7B-FP8": "Qwen3-1.7B-FP8",
            "Qwen3-1.7B-NVFP4": "Qwen3-1.7B-NVFP4",
            "Qwen3-VL-4B-Instruct-NVFP4": "Qwen3-VL-4B-Instruct-NVFP4",
            "Qwen3-VL-2B-Instruct-INT4-AWQ": "Qwen3-VL-2B-Instruct-INT4-AWQ",
        }

        # Determine search directory and model path. A map entry may be either
        # a single directory name (str) or a list of candidates — useful when
        # the same model ships under multiple folder names (e.g. ``InternVL3-1B``
        # as ``InternVL3-1B-hf`` or ``InternVL3-1B``).
        if self.model_name in GPTQ_MODELS_DIR_MAP:
            search_dir = self.edgellm_data_dir
            entry = GPTQ_MODELS_DIR_MAP[self.model_name]
        elif self.model_name in LLM_MODELS_DIR_MAP:
            search_dir = self.llm_models_dir
            entry = LLM_MODELS_DIR_MAP[self.model_name]
        else:
            all_models = list(LLM_MODELS_DIR_MAP.keys()) + list(
                GPTQ_MODELS_DIR_MAP.keys())
            raise ValueError(f"Unsupported model name: '{self.model_name}'. "
                             f"Supported models: {', '.join(all_models)}")

        candidates = [entry] if isinstance(entry, str) else list(entry)
        for model_dir_name in candidates:
            model_dir = _find_directory(search_dir,
                                        model_dir_name,
                                        DEFAULT_SEARCH_DEPTH,
                                        require_files=_HF_CHECKPOINT_FILES)
            if model_dir:
                return model_dir
        raise ValueError(
            f"Model directory not found: none of {candidates} under "
            f"{search_dir} (search depth {DEFAULT_SEARCH_DEPTH}, "
            f"requiring config.json + *.safetensors)")

    def is_prequantized(self) -> bool:
        """Model is pre-quantized if its name already contains the precision identifier."""
        if self.llm_precision == "fp16":
            return False
        if self.llm_precision == "int4_gptq":
            return True
        if self.model_name in PRE_QUANTIZED_MODELS:
            return True
        return self.llm_precision.upper() in self.model_name.upper()

    def get_audio_engine_dir(self) -> str:
        suffix = f"audio-{self.audio_precision}"
        if self.min_time_steps is not None and self.max_time_steps is not None:
            suffix += f"-mnts{self.min_time_steps}-mxts{self.max_time_steps}"
        return os.path.join(self.get_engine_base_dir(), suffix)

    def get_draft_model_dir(self) -> str:
        """
        Get draft model directory using draft_model_id.
        Supports multiple draft models per base model.
        """
        # base_model -> draft_id -> draft_model_path
        MODEL_NAME_TO_DRAFT_MODELS_MAP = {
            "Qwen2.5-VL-7B-Instruct": {
                "v1": "qwen2.5-vl-7b-eagle3-v1",
                "v2": "qwen2.5-vl-7b-eagle3-v2",
                "sgl": "qwen2.5-vl-7b-eagle3-sgl",
            },
            "Llama-3.1-8B-Instruct": {
                "eagle3": "EAGLE3-LLaMA3.1-Instruct-8B",
            },
            "Qwen3-8B": {
                "eagle3": "qwen3_8b_eagle3",
            },
            "Qwen3-4B-Instruct-2507": {
                "v2": "EAGLE3-Qwen3-4B-v2",
                "v2.1": "EAGLE3-Qwen3-4B-v2.1",
            },
            "Qwen3-VL-4B-Instruct": {
                "eagle3": "EAGLE3-Qwen3-VL-4B-v1.1",
            },
            "Qwen3-1.7B": {
                "eagle3": "Qwen3-1.7B_eagle3",
            },
            # Pre-quantized base models with pre-quantized EAGLE3 drafts
            # (unified checkpoints in /scratch.edge_llm_cache/quantized_models/)
            "Qwen3-1.7B-NVFP4": {
                "eagle3": "Qwen3-1.7B-eagle3-NVFP4",
            },
            "Qwen3-VL-4B-Instruct-NVFP4": {
                "eagle3": "EAGLE3-Qwen3-VL-4B-v1.1-NVFP4",
            },
            "Qwen3-VL-8B-Instruct": {
                "v0": "qwen3-vl-8b-eagle3-v0",
            },
            # Add more mappings as needed
        }

        if self.model_name not in MODEL_NAME_TO_DRAFT_MODELS_MAP:
            raise ValueError(
                f"Unsupported base model for EAGLE: '{self.model_name}'. "
                f"Supported models: {', '.join(MODEL_NAME_TO_DRAFT_MODELS_MAP.keys())}"
            )

        draft_models = MODEL_NAME_TO_DRAFT_MODELS_MAP[self.model_name]

        if not self.draft_model_id:
            raise ValueError(
                f"draft_model_id not set. Available draft models for {self.model_name}: "
                f"{', '.join(draft_models.keys())}")

        if self.draft_model_id not in draft_models:
            raise ValueError(
                f"Unsupported draft_model_id '{self.draft_model_id}' for {self.model_name}. "
                f"Available: {', '.join(draft_models.keys())}")

        model_dir_name = draft_models[self.draft_model_id]
        # Search in llm_models_dir first, then fallback to edgellm_data_dir.
        # Require HF checkpoint markers so we don't match same-named engine
        # cache or ONNX output dirs at deeper levels.
        model_dir = _find_directory(self.llm_models_dir,
                                    model_dir_name,
                                    5,
                                    require_files=_HF_CHECKPOINT_FILES)
        if not model_dir:
            model_dir = _find_directory(self.edgellm_data_dir,
                                        model_dir_name,
                                        5,
                                        require_files=_HF_CHECKPOINT_FILES)
        if not model_dir:
            raise ValueError(
                f"Draft model directory not found: '{model_dir_name}' under "
                f"{self.llm_models_dir} or {self.edgellm_data_dir} with search depth 5 "
                f"(requiring config.json + *.safetensors)")
        return model_dir

    def get_onnx_base_dir(self) -> str:
        """Get ONNX model base directory"""
        if not self.onnx_dir:
            raise ValueError("onnx_dir not set")
        return os.path.join(self.onnx_dir, self.model_name)

    def get_engine_base_dir(self) -> str:
        """Get engine base directory"""
        if not self.engine_dir:
            raise ValueError("engine_dir not set")
        return os.path.join(self.engine_dir, self.model_name)

    def get_llm_onnx_dir(self) -> str:
        """Get LLM ONNX model directory. For TTS this contains talker/ and code_predictor/."""
        if self.is_mtp:
            prefix = "llm-base-mtp"
        elif self.is_eagle:
            prefix = "llm-base"
        else:
            prefix = "llm"
        return os.path.join(self.get_onnx_base_dir(),
                            f"{prefix}-{self.get_onnx_model_id()}")

    def get_tts_tokenizer_dir(self) -> str:
        """
        Get tokenizer directory for TTS benchmark/inference.

        Prefer ONNX export output, where ``tokenizer.json`` is written: the
        tensorrt_edgellm-based export drops it under ``llm-<prec>/talker/`` (since
        the talker submodel is what carries the language modeling head), so
        check there first; older exports placed it directly at
        ``llm-<prec>/``. Fall back to the torch model dir
        (``vocab.json`` + ``tokenizer_config.json``) when neither has it.
        """
        onnx_llm_dir = self.get_llm_onnx_dir()
        candidates = [
            os.path.join(onnx_llm_dir, "talker"),
            onnx_llm_dir,
        ]
        for candidate in candidates:
            if os.path.isfile(os.path.join(candidate, "tokenizer.json")):
                return candidate
        return self.get_torch_model_dir()

    def get_audio_onnx_dir(self, precision: Optional[str] = None) -> str:
        """Audio encoder ONNX directory.

        ``precision=None`` uses ``self.audio_precision``. Pass an explicit
        precision (``"fp16"`` / ``"fp8"``) when the test path needs to refer to
        a specific variant.
        """
        p = precision or self.audio_precision
        return os.path.join(self.get_onnx_base_dir(), f"audio-{p}")

    def get_code2wav_onnx_dir(self, precision: Optional[str] = None) -> str:
        """Code2Wav ONNX directory for TTS models."""
        p = precision or self.audio_precision
        return os.path.join(self.get_onnx_base_dir(), f"code2wav-{p}")

    def get_draft_onnx_model_id(self) -> str:
        """Generate unique draft model identifier including draft_model_id"""
        if self.draft_model_id is None:
            raise ValueError("draft_model_id not set")
        if self.draft_llm_precision is None:
            raise ValueError("draft_llm_precision not set")
        if self.draft_lm_head_precision is None:
            raise ValueError("draft_lm_head_precision not set")
        draft_id = f"{self.draft_model_id}-{self.draft_llm_precision}-{self.draft_lm_head_precision}"
        return draft_id

    def get_draft_onnx_dir(self) -> str:
        """Get draft model ONNX directory"""
        if self.is_mtp:
            # MTP draft shares the base checkpoint precision; use the same
            # onnx_model_id suffix so different precisions don't collide.
            return os.path.join(self.get_onnx_base_dir(),
                                f"mtp-draft-{self.get_onnx_model_id()}")
        return os.path.join(self.get_onnx_base_dir(),
                            f"draft-{self.get_draft_onnx_model_id()}")

    def get_quantized_draft_model_dir(self) -> str:
        """Get quantized draft model directory (for export)"""
        if self.draft_llm_precision == "fp16":
            return self.get_draft_model_dir()
        if self.draft_model_id is None:
            raise ValueError("draft_model_id not set")
        quantized_name = f"quantized-{self.draft_model_id}-{self.draft_llm_precision}-{self.draft_lm_head_precision}"
        return os.path.join(self.get_onnx_base_dir(), "quantized-draft",
                            quantized_name)

    def get_visual_onnx_dir(self, precision: str) -> str:
        """Get visual ONNX model directory"""
        return os.path.join(self.get_onnx_base_dir(), f"visual-{precision}")

    def get_llm_engine_dir(self) -> str:
        """Get LLM engine directory"""
        if not self.engine_dir:
            raise ValueError(
                "engine_dir not set - required for LLM engine operations")
        if self.task_type == TaskType.EXPORT:
            raise ValueError(
                "LLM engine directory not available for export tasks")

        if self.is_mtp:
            # MTP shares the base checkpoint; no separate draft_model_id.  Precision
            # info comes from get_engine_id() (already includes llm/lm_head precision).
            prefix = "llm-mtp"
        elif self.is_eagle:
            if self.draft_model_id is None:
                raise ValueError("draft_model_id not set for EAGLE engine")
            if self.draft_llm_precision is None:
                raise ValueError(
                    "draft_llm_precision not set for EAGLE engine")
            prefix = f"llm-eagle-{self.draft_model_id}-{self.draft_llm_precision}"
        else:
            prefix = "llm"

        return os.path.join(self.get_engine_base_dir(),
                            f"{prefix}-{self.get_engine_id()}")

    def get_talker_engine_dir(self) -> str:
        """Get Talker engine directory (TTS only). Subdir under LLM engine dir."""
        return os.path.join(self.get_llm_engine_dir(), "talker")

    def get_code_predictor_engine_dir(self) -> str:
        """Get CodePredictor engine directory (TTS only). Subdir under LLM engine dir."""
        return os.path.join(self.get_llm_engine_dir(), "code_predictor")

    def get_code2wav_engine_dir(self) -> str:
        """Get Code2Wav engine directory (TTS only). Subdir under LLM engine dir."""
        return os.path.join(self.get_llm_engine_dir(), "code2wav")

    def get_visual_engine_dir(self) -> str:
        """Get visual engine directory"""
        return os.path.join(
            self.get_engine_base_dir(),
            f"visual-{self.visual_precision}-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}-mxpiit{self.max_image_tokens_per_image}"
        )

    def get_multimodal_engine_dir(self) -> str:
        """OMNI multimodal engine parent dir.

        visual_build writes to <this>/visual/ and audio_build writes to
        <this>/audio/, so inference / e2e_bench pass
        --multimodalEngineDir=<this> and runtime auto-locates both encoders.
        """
        suffix = (f"multimodal-mnit{self.min_image_tokens}"
                  f"-mxit{self.max_image_tokens}"
                  f"-mxpiit{self.max_image_tokens_per_image}"
                  f"-mnts{self.min_time_steps}"
                  f"-mxts{self.max_time_steps}")
        return os.path.join(self.get_engine_base_dir(), suffix)

    def get_test_case_file(self) -> str:
        """
        Get test case file path using test case name mapping.

        When ``_test_case_file_override`` is set (by helpers that have
        rewritten the test case JSON for this config, e.g. audio
        preprocessing or LoRA placeholder substitution), that path is
        returned so downstream command generation operates on the
        per-config preprocessed copy and never mutates the
        version-controlled source.

        Returns:
            Full path to the test case JSON file

        Raises:
            ValueError: If test_case is not set or not supported
        """
        if self._test_case_file_override:
            return self._test_case_file_override

        if not self.test_case:
            raise ValueError("test_case not set - required for this operation")

        # Test case name to file path mapping
        # Maps logical test case names to their actual file paths.
        # You can adjust this map to match your test case organization.
        mtbench_dataset = ("mtbench_eagle3.json" if self.is_eagle or "qwen3"
                           in str(getattr(self, "model_name", "")).lower() else
                           "mtbench_dataset.json")

        TEST_CASE_NAME_TO_PATH_MAP = {
            # Add test case mappings here, for example:
            "llm_basic":
            "tests/test_cases/llm_basic.json",
            "llm_lora":
            "tests/test_cases/llm_lora.json",
            "asr_basic":
            "tests/test_cases/asr_basic.json",
            "tts_basic":
            "tests/test_cases/tts_basic.json",
            "vlm_basic":
            "tests/test_cases/vlm_basic.json",
            "vlm_lora":
            "tests/test_cases/vlm_lora.json",
            "mtbench":
            f"{self.edgellm_data_dir}/updated_datasets/updated_MTBench/{mtbench_dataset}",
            "mmmu":
            f"{self.edgellm_data_dir}/updated_datasets/mmmu/mmmu_dataset.json",
            "mmmu_pro_4":
            f"{self.edgellm_data_dir}/updated_datasets/MMMU_Pro_4/mmmu_pro_4_dataset.json",
            "mmmu_pro_10":
            f"{self.edgellm_data_dir}/updated_datasets/MMMU_Pro_10/mmmu_pro_10_dataset.json",
            "mmmu_pro_vision":
            f"{self.edgellm_data_dir}/updated_datasets/MMMU_Pro_vision/mmmu_pro_vision_dataset.json",
            "coco":
            f"{self.edgellm_data_dir}/updated_datasets/coco/dataset.json",
            "mmlu_0":
            f"{self.edgellm_data_dir}/updated_datasets/MMLU_zero_shot/mmlu_dataset.json",
            "mmlu_5":
            f"{self.edgellm_data_dir}/updated_datasets/MMLU_five_shot/mmlu_dataset.json",
            "mmlu_pro":
            f"{self.edgellm_data_dir}/updated_datasets/MMLU_Pro/mmlu_pro_dataset.json",
            "mmstar":
            f"{self.edgellm_data_dir}/updated_datasets/MMStar/mmstar_reference.json",
            "aime":
            f"{self.edgellm_data_dir}/updated_datasets/AIME/aime_dataset.json",
            "gsm8k":
            f"{self.edgellm_data_dir}/updated_datasets/GSM8K/gsm8k_dataset.json",
            "gsm8k_10":
            f"{self.edgellm_data_dir}/updated_datasets/GSM8K/gsm8k_dataset_10.json",
            "humaneval":
            f"{self.edgellm_data_dir}/updated_datasets/HumanEval/humaneval_dataset.json",
            "math500":
            f"{self.edgellm_data_dir}/updated_datasets/MATH500/math500_dataset.json",
        }

        if self.test_case not in TEST_CASE_NAME_TO_PATH_MAP:
            raise ValueError(
                f"Unsupported test case: '{self.test_case}'. "
                f"Supported test cases: {', '.join(TEST_CASE_NAME_TO_PATH_MAP.keys())}"
            )
        test_case_path = TEST_CASE_NAME_TO_PATH_MAP[self.test_case]
        if not os.path.exists(test_case_path):
            raise ValueError(f"Test case file not found: '{test_case_path}'")

        return test_case_path

    def get_chat_template_file(self) -> Optional[str]:
        """
        Get custom chat template file path for models that require it.

        Returns:
            Path to chat template JSON file, or None if no custom template for this model
        """
        try:
            from tensorrt_edgellm.chat_templates import get_template_path
        except ImportError:
            return None

        MODEL_TO_TEMPLATE = {
            "NVIDIA-Nemotron-Nano-9B-v2": "nemotron_nano_v2",
            "NVIDIA-Nemotron-Nano-9B-v2-FP8": "nemotron_nano_v2",
            "NVIDIA-Nemotron-Nano-9B-v2-NVFP4": "nemotron_nano_v2",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice": "qwen3tts",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice": "qwen3tts",
        }

        template_id = MODEL_TO_TEMPLATE.get(self.model_name)
        if template_id:
            return get_template_path(template_id)
        return None

    def get_output_json_file(self) -> str:
        """
        Get output JSON file path.
        Always stored on host in log directory for subsequent processing.
        """
        return os.path.join(self.test_log_dir, f"{self.param_str}.json")

    def get_output_audio_dir(self) -> str:
        """
        Get directory for TTS-generated wav files. Uses a subdir per test case
        (param_str) so that different datasets (e.g. SeedTTS_en_meta vs SeedTTS_zh_meta)
        do not overwrite each other's audio_req*.wav.
        """
        return os.path.join(self.test_log_dir, self.param_str)

    def get_lora_weights_dir(self) -> str:
        """Get LoRA weights directory"""
        return os.path.join(self.get_onnx_base_dir(), "lora_weights")

    def get_lora_adapter_dir(self) -> str:
        """
        Get LoRA adapter directory for models with embedded LoRA (e.g., Phi-4).
        Returns the path to the vision-lora subdirectory in the torch model dir.
        """
        LORA_ADAPTER_SUBDIRS = {
            "Phi-4-multimodal-instruct": "vision-lora",
        }
        if self.model_name not in LORA_ADAPTER_SUBDIRS:
            raise ValueError(
                f"No LoRA adapter mapping for model: {self.model_name}. "
                f"Supported models: {', '.join(LORA_ADAPTER_SUBDIRS.keys())}")
        subdir = LORA_ADAPTER_SUBDIRS[self.model_name]
        return os.path.join(self.get_torch_model_dir(), subdir)

    def get_merged_model_dir(self) -> str:
        """Get merged LoRA model directory (for models requiring LoRA merge before quantization)"""
        return os.path.join(self.get_onnx_base_dir(), "merged-vision")

    def get_quantized_model_dir(self) -> str:
        """Get quantized model directory (for export)"""
        if self.llm_precision == "fp16":
            return self.get_torch_model_dir()
        if self.is_eagle and not self.is_mtp:
            prefix = "quantized-base"
        else:
            prefix = "quantized"
        quantized_name = f"{self.llm_precision}-{self.lm_head_precision}"
        if self.fp8_kv_cache:
            quantized_name += "-fp8kv"

        return os.path.join(self.get_onnx_base_dir(), prefix, quantized_name)

    def get_kv_cache_quantized_model_dir(self) -> str:
        """
        Get a derived model directory for KV-cache-only quantization.

        Used when llm_precision == fp16 but fp8_kv_cache is enabled, so we need a distinct
        output directory for `tensorrt-edgellm-quantize --kv_cache_quantization fp8`.
        """
        return os.path.join(self.get_onnx_base_dir(), "quantized-kvcache",
                            self.get_onnx_model_id())

    def get_reduced_vocab_dir(self) -> str:
        """Get reduced vocabulary directory (for export)"""
        if not self.reduced_vocab_size:
            raise ValueError("reduced_vocab_size not set")
        vocab_name = f"vocab-{self.reduced_vocab_size}"
        if self.vocab_reduction_method:
            vocab_name += f"-{self.vocab_reduction_method}"
        return os.path.join(self.get_onnx_base_dir(), "reduced_vocab",
                            vocab_name)

    def get_cnn_dailymail_dataset_dir(self) -> str:
        """Get CNN DailyMail dataset directory for LLM quantization calibration"""
        if not self.llm_models_dir:
            raise ValueError("llm_models_dir not set")
        dataset_dir = _find_directory(
            self.llm_models_dir, os.path.join("datasets", "cnn_dailymail"),
            DEFAULT_SEARCH_DEPTH)
        if not dataset_dir:
            raise ValueError(
                f"CNN DailyMail dataset directory not found under {self.llm_models_dir}"
            )
        return dataset_dir

    def get_mmmu_dataset_dir(self) -> str:
        """Get MMMU dataset directory for visual model quantization calibration"""
        if not self.llm_models_dir:
            raise ValueError("llm_models_dir not set")
        dataset_dir = _find_directory(self.llm_models_dir,
                                      os.path.join("datasets", "MMMU"),
                                      DEFAULT_SEARCH_DEPTH)
        if not dataset_dir:
            raise ValueError(
                f"MMMU dataset directory not found under {self.llm_models_dir}"
            )
        return dataset_dir

    def get_reference_json_file(self) -> Optional[str]:
        """
        Get model-specific reference JSON file for ROUGE score calculation.
        
        Lookup order:
        1. {test_case}_{model_name}_{llm_precision}.json
        2. {test_case}_{model_name}_fp16.json (fallback)
        3. None (use original test_case_file as reference)
        
        Returns:
            Path to reference JSON file, or None if no model-specific reference exists
        """
        if not self.test_case or not self.edgellm_data_dir:
            return None

        reference_dir = os.path.join(self.edgellm_data_dir,
                                     "specific_model_reference")
        if not os.path.isdir(reference_dir):
            return None

        # Try exact precision match first
        exact_match = os.path.join(
            reference_dir,
            f"{self.test_case}_{self.model_name}_{self.llm_precision}.json")
        if os.path.exists(exact_match):
            return exact_match

        # Fallback to fp16 if current precision is not fp16
        if self.llm_precision != "fp16":
            fp16_fallback = os.path.join(
                reference_dir, f"{self.test_case}_{self.model_name}_fp16.json")
            if os.path.exists(fp16_fallback):
                return fp16_fallback

        return None
