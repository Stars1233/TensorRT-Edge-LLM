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
"""Common test functions for TensorRT Edge-LLM"""

import glob
import logging
import os
from typing import Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .utils.device import DeviceConfig


def _get_trt_env_vars(env_config: EnvironmentConfig) -> Optional[dict]:
    """Get environment variables for TensorRT library path."""
    if env_config.trt_package_dir:
        trt_lib_path = f"{env_config.trt_package_dir}/lib"
        return {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib_path}"}
    return None


def test_build_project(env_config: EnvironmentConfig,
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger):
    """Test project build - builds all components"""
    execution_mode = "remote" if remote_config else "local"
    device_config = DeviceConfig.auto_detect(remote_config, test_logger)
    test_logger.info(
        f"Building project for {device_config.target} in {execution_mode} mode"
    )
    build_dir = env_config.build_dir

    # Build cmake command with required components only
    cmake_cmd = ['cmake', '..', '-DBUILD_UNIT_TESTS=ON']

    # Use trt_package_dir from env_config
    if env_config.trt_package_dir:
        cmake_cmd.append(f'-DTRT_PACKAGE_DIR={env_config.trt_package_dir}')
    cmake_cmd.append(f'-DCUDA_CTK_VERSION={device_config.cuda_version}')

    if device_config.target in [
            'jetson-orin', 'auto-thor', 'jetson-thor', 'gb10'
    ]:
        cmake_cmd.append(f'-DEMBEDDED_TARGET={device_config.target}')
        cmake_cmd.append(
            '-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake')

    # Enable CuteDSL kernels for Blackwell aarch64 targets.
    if device_config.target in ['auto-thor', 'jetson-thor', 'gb10']:
        # Prebuilt tarballs are committed in kernelSrcs/cuteDSLPrebuilt/.
        # CMake auto-extracts them -- no on-device build needed.
        cmake_cmd.append('-DENABLE_CUTE_DSL=ALL')
        test_logger.info(
            "CuTe DSL: using prebuilt tarball (CMake auto-extracts)")

    # Jetson Orin (SM87): the GDN/SSD cuteDSL kernels have sm_87 variants, but
    # only the unit-test job stages the tarball, so enable cuteDSL only when it
    # is present (the pipeline jobs build without it, as before).
    if device_config.target == 'jetson-orin' and glob.glob(
            os.path.join(env_config.llm_sdk_dir, 'kernelSrcs',
                         'cuteDSLPrebuilt', 'cutedsl_aarch64_sm_87_*.tar.gz')):
        cmake_cmd.append('-DENABLE_CUTE_DSL=ALL')
        cmake_cmd.append('-DCUTE_DSL_ARTIFACT_TAG=sm_87')
        test_logger.info("CuTe DSL: jetson-orin sm_87, using prebuilt tarball "
                         "(CMake auto-extracts)")

    # Enable CuteDSL kernels on x86 when the job staged a prebuilt tarball for
    # the detected SM. The unit-test jobs stage one artifact per SKU (see the
    # build_cutedsl_x86_sm*_artifact jobs); pipeline jobs that only stage the
    # sm_120 tarball keep their previous behavior on other SMs. sm86 reuses the
    # sm_80 artifact: cubins are forward compatible within a major compute
    # capability and the sm_80/sm_86 kernel variant sets are identical.
    x86_cutedsl_tags = {80: 'sm_80', 86: 'sm_80', 100: 'sm_100', 120: 'sm_120'}
    x86_tag = x86_cutedsl_tags.get(device_config.compute_capability)
    if (device_config.target == 'x86' and x86_tag and glob.glob(
            os.path.join(env_config.llm_sdk_dir, 'kernelSrcs',
                         'cuteDSLPrebuilt',
                         f'cutedsl_x86_64_{x86_tag}_*.tar.gz'))):
        cmake_cmd.append('-DENABLE_CUTE_DSL=ALL')
        cmake_cmd.append(f'-DCUTE_DSL_ARTIFACT_TAG={x86_tag}')
        test_logger.info(
            f"CuTe DSL: x86 SM{device_config.compute_capability}, "
            f"using prebuilt {x86_tag} tarball (CMake auto-extracts)")

    build_cmd = ' && '.join([
        f'mkdir -p {build_dir}', f'cd {build_dir}', ' '.join(cmake_cmd),
        'make -j16'
    ])

    with timer_context(f"Building ({execution_mode})", test_logger):
        result = run_command(cmd=['bash', '-c', build_cmd],
                             remote_config=remote_config,
                             timeout=600,
                             logger=test_logger)
        success = result['success']

    if not success:
        pytest.fail("Build failed")

    expected_files = [
        'unitTest',
        'examples/llm/llm_build',
        'examples/llm/llm_inference',
        'examples/multimodal/visual_build',
        'examples/multimodal/audio_build',
    ]
    # Executables that support --help smoke test
    help_check_files = [
        'examples/llm/llm_build',
        'examples/llm/llm_inference',
        'examples/multimodal/visual_build',
        'examples/multimodal/audio_build',
    ]

    env_vars = _get_trt_env_vars(env_config)

    for artifact in expected_files:
        artifact_path = os.path.join(build_dir, artifact)

        result = run_command(cmd=['test', '-f', artifact_path],
                             remote_config=remote_config,
                             timeout=60,
                             logger=test_logger)
        if not result['success']:
            pytest.fail(f"Build artifact not found: {artifact_path}")

        if artifact not in help_check_files:
            continue

        # Verify executable runs correctly with --help
        result = run_command(cmd=[artifact_path, '--help'],
                             remote_config=remote_config,
                             timeout=180,
                             logger=test_logger,
                             env_vars=env_vars)
        if not result['success']:
            output = result.get('output', '')
            test_logger.error(f"Executable --help output:\n{output}")
            pytest.fail(
                f"Executable --help failed (exit code {result['returncode']}): {artifact_path}"
            )


def test_unit_tests(env_config: EnvironmentConfig,
                    remote_config: Optional[RemoteConfig],
                    test_logger: logging.Logger):
    """Test unit tests execution - model independent"""
    execution_mode = "remote" if remote_config else "local"
    test_logger.info(f"Starting unit tests execution in {execution_mode} mode")

    build_dir = env_config.build_dir
    unit_test_cmd = ['bash', '-c', f'cd {build_dir} && ./unitTest']
    env_vars = _get_trt_env_vars(env_config)

    result = run_command(cmd=unit_test_cmd,
                         remote_config=remote_config,
                         timeout=600,
                         logger=test_logger,
                         env_vars=env_vars)

    if not result['success']:
        pytest.fail(
            f"Unit tests failed: {result.get('error', 'Unknown error')}")
