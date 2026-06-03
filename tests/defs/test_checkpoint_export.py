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
"""Checkpoint export test suite for pre-quantized models.

Uses `tensorrt-edgellm-export` to export pre-quantized
models (e.g. Nemotron 4B NVFP4, Phi-4) to ONNX format. The exported ONNX
is placed in the same ONNX directory structure used by the standard export
pipeline, so downstream engine build and inference tests can consume it.
"""

import os
import shutil
import tempfile

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from .config import (DEFAULT_SEARCH_DEPTH, ModelType, TaskType, TestConfig,
                     _find_directory)
from .utils.command_generation import AVAILABLE_LORA_WEIGHTS


def test_checkpoint_export(test_param: str, test_logger,
                           env_config: EnvironmentConfig):
    """Export a pre-quantized model via tensorrt_edgellm.scripts.export."""

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)

    # Locate source model checkpoint
    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    # Determine the final ONNX output directory (same layout as standard export)
    llm_onnx_dir = config.get_llm_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)

    # Use a temporary directory for the raw export output
    tmp_dir = tempfile.mkdtemp(prefix="checkpoint_export_")

    try:
        # Build the export command
        # PYTHONPATH must include repository root so the source package imports.
        # The CI job sets this via PYTHONPATH=$LLM_SDK_DIR:$PYTHONPATH.
        export_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            torch_dir,
            tmp_dir,
        ]

        with timer_context(
                f"Exporting {config.model_name} via the checkpoint exporter",
                test_logger):
            result = run_command(export_cmd,
                                 timeout=600,
                                 remote_config=None,
                                 logger=test_logger)
            if not result['success']:
                pytest.fail(
                    f"checkpoint export failed: {result.get('error', 'Unknown error')}"
                )

        # Move the LLM ONNX output to the expected directory.
        # TTS exports two LLM-class sub-models (talker + code_predictor); build
        # expects them under ``<llm_onnx_dir>/{talker,code_predictor}/``. The
        # standard non-TTS export emits a single ``tmp/llm/`` that maps
        # directly to ``<llm_onnx_dir>/``.
        llm_output = os.path.join(tmp_dir, "llm")
        cp_output = os.path.join(tmp_dir, "code_predictor")
        if not os.path.isdir(llm_output):
            pytest.fail(
                f"the checkpoint exporter did not produce llm/ output directory in {tmp_dir}"
            )
        if os.path.isdir(cp_output):
            # TTS layout: tmp/llm/         -> onnx/llm-<prec>/talker/
            #             tmp/code_predictor/ -> onnx/llm-<prec>/code_predictor/
            shutil.copytree(llm_output,
                            os.path.join(llm_onnx_dir, "talker"),
                            dirs_exist_ok=True)
            shutil.copytree(cp_output,
                            os.path.join(llm_onnx_dir, "code_predictor"),
                            dirs_exist_ok=True)
        else:
            # Standard layout: tmp/llm/ -> onnx/llm-<prec>/
            shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

        # If the model also has a visual encoder output, move that too
        visual_output = os.path.join(tmp_dir, "visual")
        if os.path.isdir(visual_output):
            visual_onnx_dir = os.path.join(config.get_onnx_base_dir(),
                                           "visual-fp16")
            shutil.copytree(visual_output, visual_onnx_dir, dirs_exist_ok=True)

        # Same for audio encoder (ASR / Qwen3-Omni / Nemotron-Omni).
        audio_output = os.path.join(tmp_dir, "audio")
        if os.path.isdir(audio_output):
            audio_onnx_dir = os.path.join(config.get_onnx_base_dir(),
                                          "audio-fp16")
            shutil.copytree(audio_output, audio_onnx_dir, dirs_exist_ok=True)

        # Same for Qwen3-Omni Code2Wav vocoder.
        code2wav_output = os.path.join(tmp_dir, "code2wav")
        if os.path.isdir(code2wav_output):
            code2wav_onnx_dir = os.path.join(config.get_onnx_base_dir(),
                                             "code2wav-fp16")
            shutil.copytree(code2wav_output,
                            code2wav_onnx_dir,
                            dirs_exist_ok=True)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Validate the exported ONNX model exists. TTS produces model.onnx under
    # ``talker/`` (the talker is the LLM-class submodel); non-TTS produces it
    # at the root of llm_onnx_dir.
    onnx_candidates = [
        os.path.join(llm_onnx_dir, "model.onnx"),
        os.path.join(llm_onnx_dir, "talker", "model.onnx"),
    ]
    if not any(os.path.exists(p) for p in onnx_candidates):
        pytest.fail(
            f"LLM ONNX model not found after export at any of: {onnx_candidates}"
        )


def _run_checkpoint_export(cmd, timeout, test_logger, label):
    """Run an checkpoint export command and fail on error."""
    with timer_context(label, test_logger):
        result = run_command(cmd,
                             timeout=timeout,
                             remote_config=None,
                             logger=test_logger)
        if not result['success']:
            pytest.fail(
                f"{label} failed: {result.get('error', 'Unknown error')}")


def test_checkpoint_eagle_export(test_param: str, test_logger,
                                 env_config: EnvironmentConfig):
    """Export EAGLE base + draft models via tensorrt_edgellm.scripts.export.

    Uses pre-quantized unified checkpoints for both base and draft models
    so that no on-the-fly quantization is needed during CI.

    Two tensorrt-edgellm-export invocations:
      1. Base model with --eagle-base  -> llm-base ONNX (+ visual for VLMs)
      2. Draft model -> draft ONNX
    """

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)

    # Locate pre-quantized base model checkpoint
    base_torch_dir = config.get_torch_model_dir()
    if not os.path.exists(base_torch_dir):
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_torch_dir}")

    # Locate pre-quantized draft model checkpoint
    draft_torch_dir = config.get_draft_model_dir()
    if not os.path.exists(draft_torch_dir):
        raise FileNotFoundError(
            f"Draft model checkpoint not found: {draft_torch_dir}")

    # Strip quantization suffix from model_name so ONNX paths match downstream
    # build/inference tests.  Export test params include the quantization type
    # in the model name (e.g. "Qwen3-1.7B-NVFP4") to locate the pre-quantized
    # checkpoint, but downstream tests use the base model name ("Qwen3-1.7B").
    _QUANT_SUFFIXES = ("-NVFP4", "-FP8", "-FP8-KV", "-INT8-SQ", "-INT4-AWQ")
    for suffix in _QUANT_SUFFIXES:
        if config.model_name.endswith(suffix):
            config.model_name = config.model_name[:-len(suffix)]
            break

    # Output directories
    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_base = tempfile.mkdtemp(prefix="eagle_base_export_")
    tmp_draft = tempfile.mkdtemp(prefix="eagle_draft_export_")

    try:
        # --- Export base model with --eagle-base ---
        base_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            base_torch_dir,
            tmp_base,
            "--eagle-base",
        ]
        _run_checkpoint_export(
            base_cmd, 600, test_logger,
            f"Exporting EAGLE base {config.model_name} via the checkpoint exporter"
        )

        # Copy base LLM ONNX
        base_llm_out = os.path.join(tmp_base, "llm")
        if not os.path.isdir(base_llm_out):
            pytest.fail(f"Base export did not produce llm/ in {tmp_base}")
        shutil.copytree(base_llm_out, llm_onnx_dir, dirs_exist_ok=True)

        # Copy visual encoder if present (VLM models)
        base_vis_out = os.path.join(tmp_base, "visual")
        if os.path.isdir(base_vis_out):
            visual_onnx_dir = os.path.join(config.get_onnx_base_dir(),
                                           "visual-fp16")
            shutil.copytree(base_vis_out, visual_onnx_dir, dirs_exist_ok=True)

        # --- Export draft model ---
        draft_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            draft_torch_dir,
            tmp_draft,
        ]
        _run_checkpoint_export(
            draft_cmd, 600, test_logger,
            f"Exporting EAGLE draft {config.draft_model_id} via the checkpoint exporter"
        )

        # Copy draft ONNX
        draft_llm_out = os.path.join(tmp_draft, "llm")
        if not os.path.isdir(draft_llm_out):
            pytest.fail(f"Draft export did not produce llm/ in {tmp_draft}")
        shutil.copytree(draft_llm_out, draft_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        shutil.rmtree(tmp_draft, ignore_errors=True)

    # Validate outputs
    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"Base ONNX model not found: {base_onnx}")

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"Draft ONNX model not found: {draft_onnx}")


def test_checkpoint_mtp_export(test_param: str, test_logger,
                               env_config: EnvironmentConfig):
    """Export MTP base + draft from a single checkpoint via --mtp flag."""

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)

    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="mtp_export_")

    try:
        export_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            torch_dir,
            tmp_dir,
            "--mtp",
        ]

        with timer_context(
                f"Exporting MTP {config.model_name} via the checkpoint exporter",
                test_logger):
            result = run_command(export_cmd,
                                 timeout=600,
                                 remote_config=None,
                                 logger=test_logger)
            if not result['success']:
                pytest.fail(
                    f"MTP export failed: {result.get('error', 'Unknown error')}"
                )

        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(f"MTP export did not produce llm/ in {tmp_dir}")
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

        draft_output = os.path.join(tmp_dir, "mtp_draft")
        if not os.path.isdir(draft_output):
            pytest.fail(f"MTP export did not produce mtp_draft/ in {tmp_dir}")
        shutil.copytree(draft_output, draft_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    base_onnx = os.path.join(llm_onnx_dir, "model.onnx")
    if not os.path.exists(base_onnx):
        pytest.fail(f"MTP base ONNX not found: {base_onnx}")

    draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
    if not os.path.exists(draft_onnx):
        pytest.fail(f"MTP draft ONNX not found: {draft_onnx}")


def test_checkpoint_lora_export(test_param: str, test_logger,
                                env_config: EnvironmentConfig):
    """Export a model and insert LoRA patterns via the checkpoint exporter.

    Exports a model ONNX, then runs LoRA insertion and verifies the
    lora_model.onnx has additional LoRA nodes.
    """

    config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                          TaskType.EXPORT, env_config)

    torch_dir = config.get_torch_model_dir()
    if not os.path.exists(torch_dir):
        raise FileNotFoundError(f"Model checkpoint not found: {torch_dir}")

    llm_onnx_dir = config.get_llm_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="lora_export_")

    try:
        # Step 1: Export the model
        export_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.export",
            torch_dir,
            tmp_dir,
        ]

        _run_checkpoint_export(
            export_cmd, 600, test_logger,
            f"Exporting {config.model_name} for LoRA via the checkpoint exporter"
        )

        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(
                f"the checkpoint exporter did not produce llm/ output directory in {tmp_dir}"
            )

        orig_onnx = os.path.join(llm_output, "model.onnx")
        if not os.path.exists(orig_onnx):
            pytest.fail(f"model.onnx not found: {orig_onnx}")

        # Step 2: Insert LoRA patterns
        lora_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.insert_lora",
            "--onnx_dir",
            llm_output,
        ]

        _run_checkpoint_export(lora_cmd, 120, test_logger,
                               f"Inserting LoRA into {config.model_name}")

        lora_onnx = os.path.join(llm_output, "lora_model.onnx")
        if not os.path.exists(lora_onnx):
            pytest.fail(f"lora_model.onnx not found: {lora_onnx}")

        # Verify LoRA model has more nodes than original
        import onnx
        orig_model = onnx.load(orig_onnx, load_external_data=False)
        lora_model = onnx.load(lora_onnx, load_external_data=False)

        if len(lora_model.graph.node) <= len(orig_model.graph.node):
            pytest.fail(f"LoRA model should have more nodes than original. "
                        f"Original: {len(orig_model.graph.node)}, "
                        f"LoRA: {len(lora_model.graph.node)}")

        # Verify LoRA-specific inputs exist
        lora_inputs = [
            i.name for i in lora_model.graph.input if 'lora' in i.name.lower()
        ]
        if not lora_inputs:
            pytest.fail("No LoRA weight inputs found in lora_model.onnx")

        # Step 3: Process LoRA adapter weights for runtime tests.
        if config.model_name not in AVAILABLE_LORA_WEIGHTS:
            pytest.fail(f"No LoRA weights configured for {config.model_name}")

        lora_model_name = AVAILABLE_LORA_WEIGHTS[config.model_name]
        data_dir = config.edgellm_data_dir or os.environ.get(
            "EDGELLM_DATA_DIR", "/scratch.edge_llm_cache")
        lora_weights_dir = _find_directory(data_dir, lora_model_name,
                                           DEFAULT_SEARCH_DEPTH)
        if not lora_weights_dir:
            pytest.fail(
                f"LoRA weights directory '{lora_model_name}' not found under "
                f"{data_dir}")

        process_cmd = [
            "python3",
            "-m",
            "tensorrt_edgellm.scripts.process_lora_weights",
            "--input_dir",
            lora_weights_dir,
            "--output_dir",
            config.get_lora_weights_dir(),
        ]
        _run_checkpoint_export(
            process_cmd, 120, test_logger,
            f"Processing LoRA weights for {config.model_name}")

        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
