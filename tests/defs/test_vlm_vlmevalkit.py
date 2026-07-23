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
Test suite for VLM/OMNI VLMEvalKit MMMU evaluation.

Runs inference on MMMU dataset using multimodal models, then evaluates the
results using VLMEvalKit.
"""
import logging
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_execution import execute_vlmevalkit_test


class TestVLMVLMEvalKit:
    """Test suite for multimodal VLMEvalKit MMMU evaluation."""

    def test_vlmevalkit_mmmu(self, test_param: str,
                             executable_files: Dict[str, str],
                             remote_config: Optional[RemoteConfig],
                             test_logger: logging.Logger,
                             env_config: EnvironmentConfig) -> None:
        """
        Test MMMU evaluation using VLMEvalKit for VLM and OMNI models.

        Runs inference on the MMMU dataset, converts the output to VLMEvalKit
        format, and evaluates it.
        """
        env_config.validate_for_vlmevalkit_tests()
        model_type = (ModelType.OMNI
                      if "Omni" in test_param else ModelType.VLM)
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.VLMEVALKIT, env_config)

        with timer_context(f"VLMEvalKit MMMU for {config.model_name}",
                           test_logger):
            result = execute_vlmevalkit_test(config, executable_files,
                                             remote_config, test_logger,
                                             env_config)
            if not result['success']:
                pytest.fail(
                    f"VLMEvalKit MMMU evaluation failed: {result['error']}")
