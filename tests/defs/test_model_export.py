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
"""Model export test suite for TensorRT Edge-LLM

Uses the checkpoint exporter for ONNX export, including pre-quantized LLM,
fp16 visual, ASR/TTS, EAGLE, and MTP checkpoints.

The full pipeline is: ``tensorrt-edgellm-quantize`` if needed -> export ONNX
-> post-export transforms such as dynamic LoRA insertion.
"""

import os

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.checkpoint_export_helpers import (
    _checkpoint_export_env, run_checkpoint_export, run_command_list,
    run_tensorrt_edgellm_draft_export, run_tensorrt_edgellm_mtp_export)
from .utils.command_generation import (generate_post_tensorrt_edgellm_commands,
                                       generate_pre_export_commands)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_export_result(config: TestConfig) -> None:
    """Simple file validation - fail fast"""
    output_dir = config.get_llm_onnx_dir()

    if config.model_type == ModelType.TTS:
        expected_onnx = [
            os.path.join(output_dir, "talker", "model.onnx"),
            os.path.join(output_dir, "code_predictor", "model.onnx"),
            os.path.join(config.get_code2wav_onnx_dir(), "model.onnx"),
        ]
        for path in expected_onnx:
            if not os.path.exists(path):
                raise FileNotFoundError(f"TTS ONNX not found: {path}")
        return

    llm_onnx = os.path.join(output_dir, "model.onnx")
    if not os.path.exists(llm_onnx):
        raise FileNotFoundError(f"LLM ONNX model not found: {llm_onnx}")

    if config.lora:
        lora_onnx = os.path.join(config.get_llm_onnx_dir(), "lora_model.onnx")
        if not os.path.exists(lora_onnx):
            raise FileNotFoundError(f"LoRA ONNX model not found: {lora_onnx}")

    if config.reduced_vocab_size:
        vocab_map_file = os.path.join(output_dir, "vocab_map.safetensors")
        if not os.path.exists(vocab_map_file):
            raise FileNotFoundError(
                f"vocab_map.safetensors not found in ONNX dir: "
                f"{vocab_map_file}")

    if config.model_type == ModelType.VLM:
        fp16_visual_onnx_dir = config.get_visual_onnx_dir("fp16")
        if not os.path.exists(fp16_visual_onnx_dir):
            raise FileNotFoundError(
                f"Visual ONNX model not found: {fp16_visual_onnx_dir}")
        if config.visual_precision == "fp8":
            fp8_visual_onnx_dir = config.get_visual_onnx_dir("fp8")
            if not os.path.exists(fp8_visual_onnx_dir):
                raise FileNotFoundError(
                    f"Visual ONNX model not found: {fp8_visual_onnx_dir}")

    if config.model_type == ModelType.ASR:
        fp16_audio_onnx_dir = config.get_audio_onnx_dir("fp16")
        if not os.path.exists(fp16_audio_onnx_dir):
            raise FileNotFoundError(
                f"Audio ONNX model not found: {fp16_audio_onnx_dir}")
        if config.audio_precision == "fp8":
            fp8_audio_onnx_dir = config.get_audio_onnx_dir("fp8")
            if not os.path.exists(fp8_audio_onnx_dir):
                raise FileNotFoundError(
                    f"Audio ONNX model not found: {fp8_audio_onnx_dir}")

    if config.model_type == ModelType.OMNI:
        fp16_visual_onnx_dir = config.get_visual_onnx_dir("fp16")
        if not os.path.exists(fp16_visual_onnx_dir):
            raise FileNotFoundError(
                f"Visual ONNX model not found: {fp16_visual_onnx_dir}")
        fp16_audio_onnx_dir = config.get_audio_onnx_dir("fp16")
        if not os.path.exists(fp16_audio_onnx_dir):
            raise FileNotFoundError(
                f"Audio ONNX model not found: {fp16_audio_onnx_dir}")
        if config.audio_precision == "fp8":
            fp8_audio_onnx_dir = config.get_audio_onnx_dir("fp8")
            if not os.path.exists(fp8_audio_onnx_dir):
                raise FileNotFoundError(
                    f"Audio ONNX model not found: {fp8_audio_onnx_dir}")

    if config.is_eagle:
        draft_onnx_dir = config.get_draft_onnx_dir()
        draft_onnx = os.path.join(draft_onnx_dir, "model.onnx")
        if not os.path.exists(draft_onnx):
            raise FileNotFoundError(
                f"Draft ONNX model not found: {draft_onnx}")
        # d2t.safetensors is written by both tensorrt-edgellm-export
        # and tensorrt_edgellm's draft export (via write_runtime_artifacts) when the
        # draft carries a d2t mapping, which is required for reduced-vocab EAGLE.
        if config.reduced_vocab_size:
            d2t_path = os.path.join(draft_onnx_dir, "d2t.safetensors")
            if not os.path.exists(d2t_path):
                raise FileNotFoundError(
                    f"d2t.safetensors not found for reduced-vocab EAGLE: "
                    f"{d2t_path}")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestModelExport:
    """Unified test suite for model export"""

    def test_model_export(self, test_param: str, test_logger,
                          model_type: ModelType,
                          env_config: EnvironmentConfig):
        """Universal export test — handles LLM, VLM, ASR, and TTS.

        ONNX export uses ``tensorrt_edgellm.scripts.export`` for all model
        families.
        """

        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.EXPORT, env_config)

        # Validate pre-existing models
        torch_dir = config.get_torch_model_dir()
        if not os.path.exists(torch_dir):
            raise FileNotFoundError(f"Torch model not found: {torch_dir}")

        if config.is_eagle and not config.is_mtp:
            draft_torch_dir = config.get_draft_model_dir()
            if not os.path.exists(draft_torch_dir):
                raise FileNotFoundError(
                    f"Draft model not found: {draft_torch_dir}")

        # Create output directories
        llm_onnx_dir = config.get_llm_onnx_dir()
        print(f"Creating output directory: {llm_onnx_dir}")
        os.makedirs(llm_onnx_dir, exist_ok=True)

        if config.model_type in (ModelType.TTS, ModelType.ASR, ModelType.OMNI):
            # Pre-create the audio output dirs expected by downstream build
            # steps. FP8 audio exports are produced from a quantized checkpoint.
            os.makedirs(config.get_audio_onnx_dir("fp16"), exist_ok=True)
            if config.audio_precision == "fp8":
                os.makedirs(config.get_audio_onnx_dir("fp8"), exist_ok=True)

        if config.model_type == ModelType.TTS:
            os.makedirs(config.get_code2wav_onnx_dir("fp16"), exist_ok=True)

        if config.model_type == ModelType.OMNI:
            visual_onnx_dir = config.get_visual_onnx_dir("fp16")
            os.makedirs(visual_onnx_dir, exist_ok=True)

        if config.reduced_vocab_size:
            reduced_vocab_dir = config.get_reduced_vocab_dir()
            os.makedirs(reduced_vocab_dir, exist_ok=True)

        if config.llm_precision != "fp16" and not config.is_prequantized():
            quantized_model_dir = config.get_quantized_model_dir()
            os.makedirs(quantized_model_dir, exist_ok=True)
        elif config.fp8_kv_cache:
            kv_cache_quantized_dir = config.get_kv_cache_quantized_model_dir()
            os.makedirs(kv_cache_quantized_dir, exist_ok=True)

        if config.is_eagle:
            draft_onnx_dir = config.get_draft_onnx_dir()
            os.makedirs(draft_onnx_dir, exist_ok=True)
            if (not config.is_mtp and config.draft_llm_precision
                    and config.draft_llm_precision != "fp16"):
                quantized_draft_dir = config.get_quantized_draft_model_dir()
                os.makedirs(quantized_draft_dir, exist_ok=True)

        # Install gptqmodel for GPTQ models
        if config.llm_precision == "int4_gptq":
            install_gptq_cmd = [
                "bash", "-c",
                "BUILD_CUDA_EXT=0 pip install -v gptqmodel==5.7.0 "
                "--no-build-isolation"
            ]
            result = run_command(install_gptq_cmd,
                                 timeout=300,
                                 remote_config=None,
                                 logger=test_logger)
            if not result['success']:
                pytest.fail(f"Failed to install gptqmodel: "
                            f"{result.get('error', 'Unknown error')}")

        if test_logger:
            test_logger.info("Using tensorrt_edgellm export path for %s",
                             config.model_name)

        # 1. Pre-export: quantize base + draft with
        #    ``tensorrt-edgellm-quantize``.
        # 2. Export LLM + fp16 visual + audio/TTS via the checkpoint exporter
        #    (chat_template handled internally by tensorrt_edgellm).
        #    For EAGLE: base is exported with --eagle-base, then the
        #    draft is exported in a second tensorrt_edgellm invocation.
        #    For MTP: base + draft are exported together via --mtp.
        # 3. Post-export: dynamic LoRA insertion/weight processing when needed.
        pre_commands = generate_pre_export_commands(config)
        post_commands = generate_post_tensorrt_edgellm_commands(config)

        with timer_context(
                f"Exporting {config.model_type.value} {config.model_name} "
                f"to {config.llm_precision} (tensorrt_edgellm)", test_logger):
            run_command_list(pre_commands, "Pre-export", test_logger)
            if config.is_mtp:
                run_tensorrt_edgellm_mtp_export(config, test_logger)
            else:
                run_checkpoint_export(config,
                                      test_logger,
                                      eagle_base=config.is_eagle)
                if config.is_eagle:
                    run_tensorrt_edgellm_draft_export(config, test_logger)
            # Post commands may include LoRA commands which require PYTHONPATH
            # to include repository root.
            run_command_list(post_commands,
                             "Post-export",
                             test_logger,
                             env_vars=_checkpoint_export_env(test_logger)
                             or None)

        validate_export_result(config)

    def test_llm_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """LLM export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.LLM,
                               env_config)

    def test_tts_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """TTS export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.TTS,
                               env_config)

    def test_vlm_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """VLM export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.VLM,
                               env_config)

    def test_asr_model_export(self, test_param: str, test_logger,
                              env_config: EnvironmentConfig):
        """ASR export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.ASR,
                               env_config)

    def test_omni_model_export(self, test_param: str, test_logger,
                               env_config: EnvironmentConfig):
        """OMNI (multimodal LLM + visual + audio) export test entry point"""
        self.test_model_export(test_param, test_logger, ModelType.OMNI,
                               env_config)
