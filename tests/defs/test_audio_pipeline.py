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
Test suite for audio pipeline functionality (ASR and TTS).

Tests engine building, benchmarking, and inference using parameterized
configurations. ASR models handle audio-to-text; TTS models handle text-to-speech.
"""
import logging
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_execution import (check_result_failures,
                                      execute_build_test,
                                      execute_e2e_bench_test,
                                      execute_inference_test)


class TestASRPipeline:
    """Test suite for ASR pipeline functionality."""

    def test_engine_build(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test TensorRT engine building for ASR models."""
        config = TestConfig.from_param_string(test_param, ModelType.ASR,
                                              TaskType.BUILD, env_config)

        with timer_context(f"ASR build for {config.model_name}", test_logger):
            result = execute_build_test(config, executable_files,
                                        remote_config, test_logger, env_config)
            if not result['success']:
                pytest.fail(f"Build failed: {result['error']}")

    def test_e2e_bench(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test performance benchmarking for ASR models."""
        config = TestConfig.from_param_string(test_param, ModelType.ASR,
                                              TaskType.E2E_BENCH, env_config)

        with timer_context(f"ASR benchmark for {config.model_name}",
                           test_logger):
            result = execute_e2e_bench_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Benchmark failed: {result['error']}")
            check_result_failures(result)

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test batch inference for ASR models."""
        config = TestConfig.from_param_string(test_param, ModelType.ASR,
                                              TaskType.INFERENCE, env_config)

        with timer_context(f"ASR inference for {config.model_name}",
                           test_logger):
            result = execute_inference_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Inference failed: {result['error']}")
            check_result_failures(result)


class TestTTSPipeline:
    """Test suite for TTS pipeline functionality."""

    def test_engine_build(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test TensorRT engine building for TTS models."""
        config = TestConfig.from_param_string(test_param, ModelType.TTS,
                                              TaskType.BUILD, env_config)

        with timer_context(f"TTS build for {config.model_name}", test_logger):
            result = execute_build_test(config, executable_files,
                                        remote_config, test_logger, env_config)
            if not result['success']:
                pytest.fail(f"Build failed: {result['error']}")

    def test_e2e_bench(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test performance benchmarking for TTS models."""
        config = TestConfig.from_param_string(test_param, ModelType.TTS,
                                              TaskType.E2E_BENCH, env_config)

        with timer_context(f"TTS benchmark for {config.model_name}",
                           test_logger):
            result = execute_e2e_bench_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Benchmark failed: {result['error']}")
            check_result_failures(result)

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test inference for TTS models."""
        config = TestConfig.from_param_string(test_param, ModelType.TTS,
                                              TaskType.INFERENCE, env_config)

        with timer_context(f"TTS inference for {config.model_name}",
                           test_logger):
            result = execute_inference_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Inference failed: {result['error']}")
            check_result_failures(result)
