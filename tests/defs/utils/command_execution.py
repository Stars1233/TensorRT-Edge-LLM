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
Test execution functions for TensorRT Edge-LLM tests.

This module contains the simplified test execution functions for build, inference, 
and benchmark tests. Each function is focused on its specific task without 
unnecessary abstraction layers.
"""

import json
import os
import sys
from typing import Any, Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import check_file_exists, run_command, run_with_trt_env

from ..config import ModelType, TaskType, TestConfig
from .accuracy import check_accuracy_with_dataset
from .baseline import (get_baseline, map_accuracy_result_to_csv,
                       parse_perf_from_output, save_to_baseline)
from .command_generation import (generate_build_commands,
                                 generate_e2e_bench_commands,
                                 generate_inference_commands,
                                 generate_kernel_bench_commands)

# Raw audio extensions handled by tensorrt_edgellm.scripts.preprocess_audio.
# When the test case JSON references one of these, the file is converted to a
# .safetensors mel-spectrogram on the fly because the C++ requestFileParser
# only accepts safetensors today.
_RAW_AUDIO_EXTENSIONS = (".flac", ".wav", ".mp3", ".ogg", ".m4a")


def _audio_feature_extractor(model_name: str) -> str:
    """Return the preprocess_audio --feature_extractor for *model_name*.

    Nemotron-Omni uses Parakeet mel features; everything else (Qwen3-Omni,
    Qwen3-ASR, ...) uses the Whisper default.
    """
    return "parakeet" if "nemotron" in model_name.lower() else "whisper"


def _source_test_case_file(config: TestConfig) -> Optional[str]:
    """Return the canonical (un-rewritten) test case JSON path, ignoring any
    per-config preprocessed override that this module may have already set.
    """
    override = config._test_case_file_override
    config._test_case_file_override = None
    try:
        return config.get_test_case_file()
    except ValueError:
        return None
    finally:
        config._test_case_file_override = override


def _write_preprocessed_test_case(config: TestConfig, data: Dict[str, Any],
                                  logger) -> str:
    """Write *data* as a per-config preprocessed test case JSON and return its path.

    Output goes under ``config.test_log_dir`` (a per-test directory pytest
    creates afresh per parametrization, so different ASR/OMNI configs never
    contend for the same path even under pytest-xdist). The write itself is
    atomic via ``os.replace`` to defend against partial-write corruption.
    """
    out_dir = config.test_log_dir or os.path.dirname(
        _source_test_case_file(config) or ".") or "."
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preprocessed_test_case.json")
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    os.replace(tmp_path, out_path)
    if logger:
        logger.debug("Wrote preprocessed test case: %s", out_path)
    return out_path


def _substitute_placeholder_in_test_case(config: TestConfig, placeholder: str,
                                         replacement: str,
                                         logger) -> Dict[str, Any]:
    """Replace ``placeholder`` with ``replacement`` in every string value of
    the test case JSON, write the result to a per-config preprocessed copy,
    and expose it via ``config._test_case_file_override``.

    The rewrite walks the parsed document recursively rather than doing a
    text-level ``str.replace``, so a placeholder that incidentally appears as
    a JSON key or inside a comment-shaped string elsewhere is not silently
    replaced. The version-controlled source JSON is *never* mutated, which
    keeps pytest-xdist parallel runs safe and avoids stale-state-on-rerun
    that the old in-place sed/text rewrite suffered from. Returns a
    ``run_command``-shaped dict so call sites keep their existing failure
    handling.
    """
    try:
        source_file = _source_test_case_file(config)
        if source_file is None or not os.path.isfile(source_file):
            return {
                "success": False,
                "error": f"test case file not found: {source_file!r}",
                "output": "",
            }

        with open(source_file) as f:
            data = json.load(f)

        def _walk(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: _walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_walk(v) for v in node]
            if isinstance(node, str):
                return node.replace(placeholder, replacement)
            return node

        new_data = _walk(data)
        if new_data == data:
            # Placeholder absent — nothing to do, leave override unset so
            # the canonical source path is used downstream.
            return {"success": True, "error": None, "output": ""}

        config._test_case_file_override = _write_preprocessed_test_case(
            config, new_data, logger)
        if logger:
            logger.info("Substituted %s in %s -> %s", placeholder, source_file,
                        config._test_case_file_override)
        return {"success": True, "error": None, "output": ""}
    except (OSError, json.JSONDecodeError) as e:
        return {
            "success": False,
            "error": f"placeholder substitution failed: {e}",
            "output": "",
        }


def _iter_audio_items(test_case_data: Dict[str, Any]):
    """Yield each ``{"type": "audio", "audio": <path>}`` content dict.

    Returning the dict (not just the path) lets the caller rewrite ``audio``
    in place on the parsed document — the document is then serialized to a
    per-config preprocessed copy, never back to the source JSON.
    """
    for req in test_case_data.get("requests", []):
        for msg in req.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if (isinstance(item, dict) and item.get("type") == "audio"
                        and isinstance(item.get("audio"), str)):
                    yield item


def _assert_audio_path_under_cwd(audio_path: str, source_file: str) -> None:
    """Reject audio paths that resolve outside the current working directory.

    Test-case JSONs are version-controlled, but a crafted ``..``-style path
    would otherwise let the preprocess subprocess write its
    ``_<extractor>.safetensors`` output to an arbitrary location reachable
    from the test runner's cwd.
    """
    cwd = os.path.realpath(os.getcwd())
    real = os.path.realpath(audio_path)
    if real != cwd and not real.startswith(cwd + os.sep):
        pytest.fail(f"Audio path escapes repo root: {audio_path} "
                    f"(referenced by {source_file})")


def _copy_preprocessed_audio_to_remote(sf_path: str,
                                       remote_config: Optional[RemoteConfig],
                                       logger) -> None:
    if remote_config is None:
        return

    remote_dir = os.path.dirname(sf_path)
    if remote_dir:
        result = run_command(["mkdir", "-p", remote_dir],
                             remote_config=remote_config,
                             timeout=60,
                             logger=logger)
        if not result["success"]:
            pytest.fail(
                f"Failed to create remote audio cache directory "
                f"{remote_dir}: {result.get('error', 'Unknown error')}")

    remote_target = os.path.join(remote_config.remote_workspace, sf_path)
    result = run_command([
        "sshpass",
        "-e",
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
        sf_path,
        f"{remote_config.user}@{remote_config.host}:{remote_target}",
    ],
                         remote_config=None,
                         timeout=120,
                         logger=logger,
                         env_vars={"SSHPASS": remote_config.password})
    if not result["success"]:
        pytest.fail(f"Failed to copy preprocessed audio to remote workspace: "
                    f"{result.get('error', 'Unknown error')}")


def _preprocess_audio_in_test_case(
        config: TestConfig, logger,
        remote_config: Optional[RemoteConfig]) -> None:
    """Convert raw-audio entries in the ASR/OMNI test case JSON to safetensors.

    For each ``"type": "audio"`` request whose path ends in a raw extension
    (.flac/.wav/.mp3/.ogg/.m4a), runs
    ``python tensorrt_edgellm/scripts/preprocess_audio.py`` to emit a
    ``<basename>_<extractor>.safetensors`` next to the source. The rewritten
    test case (with ``audio`` fields pointing at the generated safetensors) is
    written to a per-config copy under ``config.test_log_dir`` and exposed via
    ``config._test_case_file_override``; the version-controlled source JSON
    is *never* mutated. This makes the helper safe under pytest-xdist (each
    config has its own preprocessed copy) and avoids cross-contamination
    between models that share a fixture (e.g. ASR vs OMNI both consuming
    ``asr_basic.json``).

    Feature extractor selection is by model name: ``parakeet`` for Nemotron-
    Omni, ``whisper`` everywhere else. The generated mel cache is keyed on
    extractor, so the two flavors never share a file.

    No-op for non-ASR/OMNI configs or when the test case has no raw audio.
    In remote execution mode, the generated safetensors file is also copied
    into the same relative path under the remote workspace because inference
    runs on the device.
    """
    if config.model_type not in (ModelType.ASR, ModelType.OMNI):
        return

    source_file = _source_test_case_file(config)
    if source_file is None or not os.path.isfile(source_file):
        return

    with open(source_file) as f:
        data = json.load(f)

    extractor = _audio_feature_extractor(config.model_name)
    # Cache: raw-audio path -> generated safetensors. Dedups preprocess
    # invocations across multiple audio items pointing at the same source.
    converted: Dict[str, str] = {}
    items_to_rewrite = []

    for item in _iter_audio_items(data):
        audio_path = item["audio"]
        if not audio_path.endswith(_RAW_AUDIO_EXTENSIONS):
            continue
        _assert_audio_path_under_cwd(audio_path, source_file)
        if not os.path.isfile(audio_path):
            pytest.fail(
                f"Audio fixture not found: {audio_path} (referenced by "
                f"{source_file})")

        if audio_path not in converted:
            base, _ext = os.path.splitext(audio_path)
            sf_path = f"{base}_{extractor}.safetensors"
            if not os.path.isfile(sf_path):
                cmd = [
                    sys.executable,
                    os.path.join("tensorrt_edgellm", "scripts",
                                 "preprocess_audio.py"),
                    "--input",
                    audio_path,
                    "--output",
                    sf_path,
                    "--feature_extractor",
                    extractor,
                ]
                if logger:
                    logger.info(
                        "Preprocessing %s -> %s (feature_extractor=%s)",
                        audio_path, sf_path, extractor)
                result = run_command(cmd,
                                     remote_config=None,
                                     timeout=300,
                                     logger=logger)
                if not result["success"]:
                    pytest.fail(f"preprocess_audio failed for {audio_path}: "
                                f"{result.get('error', 'Unknown error')}")
            _copy_preprocessed_audio_to_remote(sf_path, remote_config, logger)
            converted[audio_path] = sf_path

        items_to_rewrite.append(item)

    if not items_to_rewrite:
        return

    for item in items_to_rewrite:
        item["audio"] = converted[item["audio"]]

    config._test_case_file_override = _write_preprocessed_test_case(
        config, data, logger)


def check_result_failures(result: Dict[str, Any]) -> None:
    """Check baseline regressions first, then static threshold failures.

    Called by pipeline tests after execute_*_test returns successfully.
    """
    failures = []
    if result.get('baseline_regressions'):
        failures.append("Baseline regression:\n  " +
                        "\n  ".join(result['baseline_regressions']))
    if result.get('threshold_failure'):
        failures.append(result['threshold_failure'])
    if failures:
        pytest.fail("\n\n".join(failures))


def _try_save_baseline(config: TestConfig, test_func: str,
                       result: Dict[str, Any], logger) -> None:
    """Save current result to baseline CSV when BASELINE_CSV is set but has no entry."""
    csv_path = os.environ.get('BASELINE_CSV', 'logs/baseline.csv')
    if not result.get('success', False):
        return
    if result.get('threshold_failure'):
        return
    save_to_baseline(csv_path, config.model_type.value, test_func,
                     config.param_str, result)
    if logger:
        logger.info(
            "No baseline entry for [%s]. "
            "Saved current result to %s", config.param_str, csv_path)


def _check_baseline_regression(config: TestConfig,
                               test_func: str,
                               result: Dict[str, Any],
                               logger,
                               check_perf: bool = False) -> bool:
    """Check accuracy (and optionally perf) regression against baseline CSV.

    Returns True if baseline entry was found (regardless of pass/fail).
    When baseline is found, threshold_failure is cleared since baseline takes priority.
    If no baseline exists, saves the current result for future runs.

    Args:
        check_perf: only True for benchmark tests; inference skips perf comparison.
    """
    baseline = get_baseline()
    if baseline is None:
        _try_save_baseline(config, test_func, result, logger)
        return False

    entry = baseline.find_by_param(config.param_str,
                                   test_func,
                                   model_type_value=config.model_type.value)
    if entry is None:
        _try_save_baseline(config, test_func, result, logger)
        return False

    regressions = []
    all_summaries = []

    current_acc = map_accuracy_result_to_csv(result)
    if current_acc:
        acc_reg, acc_sum = baseline.check_accuracy_regression(
            entry, current_acc)
        regressions.extend(acc_reg)
        all_summaries.extend(acc_sum)

    if check_perf:
        raw_output = result.get('output', '')
        current_perf = parse_perf_from_output(raw_output)
        # Merge accuracy metrics into perf dict; check_perf_regression
        # only looks at columns in PERF_LOWER/HIGHER_IS_BETTER, so extras
        # (e.g. rouge scores) are naturally ignored.
        current_perf.update(current_acc)
        if current_perf:
            perf_reg, perf_sum = baseline.check_perf_regression(
                entry, current_perf)
            regressions.extend(perf_reg)
            all_summaries.extend(perf_sum)

    if logger and all_summaries:
        logger.info("Baseline comparison:\n  " + "\n  ".join(all_summaries))

    if regressions:
        result['baseline_regressions'] = regressions
        if logger:
            logger.warning("Baseline regressions detected:\n  " +
                           "\n  ".join(regressions))

    # Baseline found → it takes priority, discard static threshold result
    result.pop('threshold_failure', None)
    return True


def execute_build_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute build test for any model type"""

    # Generate all build commands
    commands = generate_build_commands(config, executable_files)

    all_outputs = []

    engine_file_map = {
        executable_files['llm_build']: ["llm.engine"],
        executable_files['visual_build']: ["visual.engine"],
        executable_files['audio_build']: [
            os.path.join("audio", "audio_encoder.engine"),
            os.path.join("code2wav", "code2wav.engine"),
        ],
    }

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Build step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        engine_candidates = engine_file_map.get(cmd[0], [])
        engine_dir = next((arg.split('=', 1)[1]
                           for arg in cmd if arg.startswith('--engineDir=')),
                          None) if engine_candidates else None
        if engine_dir:
            skip = False
            for engine_filename in engine_candidates:
                if check_file_exists(os.path.join(engine_dir, engine_filename),
                                     remote_config, logger):
                    if logger:
                        logger.info(
                            f"{engine_filename} already exists in {engine_dir}. Skipping."
                        )
                    all_outputs.append(
                        f"{engine_filename} already exists - skipped")
                    skip = True
                    break
            if skip:
                continue

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.BUILD.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.BUILD.value
    }


def execute_e2e_bench_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute end-to-end benchmark test for any model type"""

    # Handle LoRA weights replacement if needed
    if config.max_lora_rank is not None and config.max_lora_rank > 0:
        # Replace the $LORA_WEIGHTS_DIR placeholder with the resolved path.
        result = _substitute_placeholder_in_test_case(
            config, "$LORA_WEIGHTS_DIR", config.get_lora_weights_dir(), logger)
        if not result['success']:
            result['test_type'] = TaskType.E2E_BENCH.value
            return result

    _preprocess_audio_in_test_case(config, logger, remote_config)

    # Generate all e2e benchmark commands
    commands = generate_e2e_bench_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Benchmark step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.E2E_BENCH.value
            }

    # Calculate metrics based on dataset type
    final_result = {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.E2E_BENCH.value
    }

    try:
        # Use model-specific reference if available, fallback to generic test case file
        reference_file = config.get_reference_json_file(
        ) or config.get_test_case_file()
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(
            config.get_output_json_file(), reference_file, config.test_case,
            logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)

    except Exception as e:
        final_result['error'] = f"Failed to calculate metrics: {str(e)}"
        final_result['success'] = False

    if final_result['success']:
        _check_baseline_regression(config,
                                   'test_e2e_bench',
                                   final_result,
                                   logger,
                                   check_perf=True)

    return final_result


def execute_inference_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute inference test for any model type"""

    # Handle LoRA weights replacement if needed
    if config.max_lora_rank is not None and config.max_lora_rank > 0:
        # Replace the $LORA_WEIGHTS_DIR placeholder with the resolved path.
        result = _substitute_placeholder_in_test_case(
            config, "$LORA_WEIGHTS_DIR", config.get_lora_weights_dir(), logger)
        if not result['success']:
            result['test_type'] = TaskType.INFERENCE.value
            return result

    _preprocess_audio_in_test_case(config, logger, remote_config)

    # Generate all inference commands
    commands = generate_inference_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Inference step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.INFERENCE.value
            }

    # Calculate metrics based on dataset type
    final_result = {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.INFERENCE.value
    }

    try:
        # Use model-specific reference if available, fallback to generic test case file
        reference_file = config.get_reference_json_file(
        ) or config.get_test_case_file()
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(
            config.get_output_json_file(), reference_file, config.test_case,
            logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)

    except Exception as e:
        final_result['error'] = f"Failed to calculate metrics: {str(e)}"
        final_result['success'] = False

    if final_result['success']:
        _check_baseline_regression(config, 'test_inference', final_result,
                                   logger)

    return final_result


def execute_kernel_bench_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute kernel_bench test - validates that the kernel benchmark runs successfully"""

    commands = generate_kernel_bench_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"kernel_bench step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.KERNEL_BENCH.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.KERNEL_BENCH.value
    }
