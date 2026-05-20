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
"""Shared helpers for exporting models via ``llm_loader.export_all_cli``.

Used by both ``test_model_export.py`` (integrated llm_loader path) and
``test_llm_loader_export.py`` (standalone llm_loader tests).

llm_loader handles chat templates internally — it reads the checkpoint's
``tokenizer_config.json`` and writes ``processed_chat_template.json``
alongside the exported ONNX, so no ``--chat_template`` argument is needed.
"""

import logging
import os
import shutil
import tempfile
from typing import Optional

import pytest
from pytest_helpers import run_command, timer_context

from ..config import ModelType, TestConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model directory selection
# ---------------------------------------------------------------------------


def get_export_model_dir(config: TestConfig,
                         use_llm_loader: bool = False) -> str:
    """Determine which checkpoint directory to feed to the exporter.

    After an optional quantization step the weights live in a derived
    directory; otherwise we use the original torch checkpoint.

    This is the single source of truth — ``_generate_llm_export_commands``
    in ``command_generation.py`` delegates to this function as well.
    """
    if config.fp8_kv_cache and config.llm_precision == "fp16":
        return config.get_kv_cache_quantized_model_dir()
    if config.llm_precision != "fp16" and not config.is_prequantized():
        return config.get_quantized_model_dir()
    if config.merge_lora:
        return config.get_merged_model_dir()
    return config.get_torch_model_dir()


# ---------------------------------------------------------------------------
# Output directory mapping
# ---------------------------------------------------------------------------

# Mapping from llm_loader output sub-directory names to a callable that
# returns the target directory given a TestConfig.  Add new entries here
# when llm_loader starts emitting additional top-level output dirs — no
# other copy logic needs to change.
#
# Special cases handled outside this map:
#   - TTS talker/code_predictor: talker is written at ``llm/`` in the
#     exporter output, but downstream engine build expects
#     ``<llm_onnx_dir>/talker/``; code_predictor is copied to
#     ``<llm_onnx_dir>/code_predictor/``.
#   - EAGLE draft: handled by ``run_llm_loader_draft_export`` which copies
#     the draft output to ``config.get_draft_onnx_dir()``.

_OUTPUT_DIR_MAP = {
    # subdir_name: (target_dir_func, description)
    "visual":
    (lambda cfg: cfg.get_visual_onnx_dir("fp16"), "fp16 visual encoder"),
    "audio":
    (lambda cfg: cfg.get_audio_onnx_dir("fp16"), "fp16 audio encoder"),
    "code2wav":
    (lambda cfg: cfg.get_code2wav_onnx_dir("fp16"), "fp16 Code2Wav"),
}


def _copy_subdir(tmp_dir: str, subdir: str, target_dir: str) -> bool:
    """Copy ``tmp_dir/subdir`` → ``target_dir`` if it exists. Return True if copied."""
    src = os.path.join(tmp_dir, subdir)
    if not os.path.isdir(src):
        return False
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(src, target_dir, dirs_exist_ok=True)
    return True


def _log_dir_contents(dir_path: str, label: str, test_logger=None) -> None:
    """Log the files in *dir_path* with sizes in MB. Used to sanity-check
    the llm_loader export output before it is copied to the final location.
    """
    log = test_logger.info if test_logger else logger.info
    if not os.path.isdir(dir_path):
        return
    entries = sorted(os.listdir(dir_path))
    log(f"{label} export produced {len(entries)} files in {dir_path}:")
    for name in entries:
        path = os.path.join(dir_path, name)
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.isfile(
            path) else 0.0
        log(f"  {size_mb:7.2f} MB  {name}")


def copy_llm_loader_output(tmp_dir: str,
                           config: TestConfig,
                           test_logger=None) -> None:
    """Copy llm_loader export output from *tmp_dir* into the directory
    layout expected by downstream engine-build and inference tests.

    Raises ``pytest.fail`` if the mandatory ``llm/`` output is missing.
    """

    def _log(msg, *args):
        if test_logger:
            test_logger.info(msg, *args)
        else:
            logger.info(msg, *args)

    llm_onnx_dir = config.get_llm_onnx_dir()
    llm_output = os.path.join(tmp_dir, "llm")

    if not os.path.isdir(llm_output):
        pytest.fail(f"llm_loader did not produce llm/ output in {tmp_dir}")

    _log_dir_contents(llm_output, "LLM", test_logger=test_logger)

    # --- LLM / Talker ---
    if config.model_type == ModelType.TTS:
        # TTS: exporter writes Talker at llm/model.onnx; downstream
        # engine build expects get_llm_onnx_dir()/talker/.
        dst = os.path.join(llm_onnx_dir, "talker")
        _copy_subdir(tmp_dir, "llm", dst)
        _log("Copied llm/ → %s (talker)", dst)
    else:
        os.makedirs(llm_onnx_dir, exist_ok=True)
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)
        _log("Copied llm/ → %s", llm_onnx_dir)

    # --- CodePredictor (TTS) ---
    # Placed under get_llm_onnx_dir()/code_predictor/ to match engine build.
    cp_src = os.path.join(tmp_dir, "code_predictor")
    if os.path.isdir(cp_src):
        _log_dir_contents(cp_src, "CodePredictor", test_logger=test_logger)
    cp_dst = os.path.join(llm_onnx_dir, "code_predictor")
    if _copy_subdir(tmp_dir, "code_predictor", cp_dst):
        _log("Copied code_predictor/ → %s", cp_dst)

    # --- Generic mapped outputs (visual, audio, future additions) ---
    for subdir, (target_fn, desc) in _OUTPUT_DIR_MAP.items():
        src = os.path.join(tmp_dir, subdir)
        if os.path.isdir(src):
            _log_dir_contents(src, desc.capitalize(), test_logger=test_logger)
        dst = target_fn(config)
        if _copy_subdir(tmp_dir, subdir, dst):
            _log("Copied %s/ → %s (%s)", subdir, dst, desc)


# ---------------------------------------------------------------------------
# Export execution
# ---------------------------------------------------------------------------


def _resolve_experimental_dir() -> Optional[str]:
    """Locate the ``experimental/`` directory containing the llm_loader package.

    Checks ``LLM_SDK_DIR/experimental`` first, then falls back to a path
    relative to this file so the helper works when LLM_SDK_DIR is unset.
    """
    sdk_dir = os.environ.get("LLM_SDK_DIR")
    if sdk_dir:
        candidate = os.path.join(sdk_dir, "experimental")
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    # Fallback: <repo>/tests/defs/utils/llm_loader_helpers.py → <repo>/experimental
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(
        os.path.join(here, "..", "..", "..", "experimental"))
    if os.path.isdir(os.path.join(candidate, "llm_loader")):
        return candidate

    return None


def get_tensorrt_edgellm_root() -> Optional[str]:
    """Parent of ``experimental/`` (the tensorrt-edge-llm tree root on disk).

    Used so subprocesses can run ``python -m experimental.quantization ...`` with
    a ``PYTHONPATH``/cwd that resolves the top-level ``experimental`` package.
    """
    ed = _resolve_experimental_dir()
    if not ed:
        return None
    return os.path.abspath(os.path.join(ed, ".."))


def _llm_loader_env(test_logger=None) -> dict:
    """Return env_vars dict that makes ``llm_loader`` importable by subprocesses."""
    exp_dir = _resolve_experimental_dir()
    if not exp_dir:
        if test_logger:
            test_logger.warning(
                "Could not locate experimental/ directory for llm_loader. "
                "Set LLM_SDK_DIR or install the package.")
        return {}
    existing = os.environ.get("PYTHONPATH", "")
    new_pp = f"{exp_dir}{os.pathsep}{existing}" if existing else exp_dir
    return {"PYTHONPATH": new_pp}


def _run_export_subprocess(model_dir: str,
                           tmp_dir: str,
                           label: str,
                           test_logger,
                           timeout: int,
                           extra_args: Optional[list] = None,
                           failure_prefix: str = "llm_loader export") -> None:
    """Invoke ``python3 -m llm_loader.export_all_cli`` and fail on error.

    Centralizes PYTHONPATH setup, timing, and error handling shared by
    base and draft exports.
    """
    # Optional: EDGELLM_EXPORT_PYTHON (e.g. export venv vs quant venv) set by test harness.
    ex_py = os.environ.get("EDGELLM_EXPORT_PYTHON", "python3")
    export_cmd = [ex_py, "-m", "llm_loader.export_all_cli", model_dir, tmp_dir]
    if extra_args:
        export_cmd.extend(extra_args)

    env_vars = _llm_loader_env(test_logger) or None

    with timer_context(label, test_logger):
        result = run_command(export_cmd,
                             timeout=timeout,
                             remote_config=None,
                             logger=test_logger,
                             env_vars=env_vars)
        if not result['success']:
            pytest.fail(f"{failure_prefix} failed: "
                        f"{result.get('error', 'Unknown error')}")


def run_llm_loader_export(config: TestConfig,
                          test_logger,
                          model_dir: Optional[str] = None,
                          timeout: int = 1200,
                          eagle_base: bool = False) -> None:
    """Run ``llm_loader.export_all_cli`` and copy output to final dirs.

    Args:
        config: Test configuration (determines output directories).
        test_logger: Logger for test output.
        model_dir: Checkpoint directory to export. Defaults to
            ``get_export_model_dir(config)``.
        timeout: Export command timeout in seconds.
        eagle_base: When True, pass ``--eagle-base`` to export an EAGLE3
            base model (adds tree-attention I/O and hidden_states output).
    """
    if model_dir is None:
        model_dir = get_export_model_dir(config, use_llm_loader=True)

    tmp_dir = tempfile.mkdtemp(prefix="llm_loader_export_")
    try:
        label = f"Exporting {config.model_name} via llm_loader"
        extra_args = []
        if eagle_base:
            extra_args.append("--eagle-base")
            label += " (EAGLE base)"
        if config.fp8_embedding:
            extra_args.append("--fp8-embedding")
            label += " (FP8 embedding)"
        if config.reduced_vocab_size:
            extra_args.append(
                f"--reduced-vocab-dir={config.get_reduced_vocab_dir()}")
            label += f" (rvs{config.reduced_vocab_size})"

        _run_export_subprocess(model_dir,
                               tmp_dir,
                               label,
                               test_logger,
                               timeout,
                               extra_args=extra_args)

        copy_llm_loader_output(tmp_dir, config, test_logger=test_logger)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_llm_loader_draft_export(config: TestConfig,
                                test_logger,
                                timeout: int = 1200) -> None:
    """Export the EAGLE draft model via ``llm_loader.export_all_cli``.

    Uses the quantized draft checkpoint when ``draft_llm_precision`` is a
    quantization that actually produces a quantized dir (i.e. non-fp16 and
    non-int4_gptq — int4_gptq ships pre-quantized so no intermediate dir
    is created). Otherwise uses the original torch draft checkpoint. The
    ONNX output is copied to ``config.get_draft_onnx_dir()``.
    """
    if (config.draft_llm_precision and config.draft_llm_precision != "fp16"
            and config.draft_llm_precision != "int4_gptq"):
        draft_model_dir = config.get_quantized_draft_model_dir()
    else:
        draft_model_dir = config.get_draft_model_dir()

    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(draft_onnx_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="llm_loader_draft_export_")
    try:
        _run_export_subprocess(
            draft_model_dir,
            tmp_dir,
            f"Exporting EAGLE draft {config.draft_model_id} via llm_loader",
            test_logger,
            timeout,
            failure_prefix="llm_loader draft export")

        draft_llm_out = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(draft_llm_out):
            pytest.fail(
                f"Draft export did not produce llm/ output in {tmp_dir}")

        _log_dir_contents(draft_llm_out, "Draft", test_logger=test_logger)

        shutil.copytree(draft_llm_out, draft_onnx_dir, dirs_exist_ok=True)
        if test_logger:
            test_logger.info(f"Copied draft/ → {draft_onnx_dir}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_llm_loader_mtp_export(config: TestConfig,
                              test_logger,
                              device: str = "cpu",
                              timeout: int = 600) -> None:
    """Export MTP base + draft from a single checkpoint via ``--mtp``.

    Calls ``llm_loader.export_all_cli --mtp`` on the base model checkpoint and
    copies outputs:
      - ``llm/``       → ``config.get_llm_onnx_dir()``
      - ``mtp_draft/`` → ``config.get_draft_onnx_dir()``
      - ``visual/``    → ``config.get_visual_onnx_dir("fp16")`` (VLM only,
                          copied when present)
    """
    model_dir = get_export_model_dir(config, use_llm_loader=True)

    llm_onnx_dir = config.get_llm_onnx_dir()
    draft_onnx_dir = config.get_draft_onnx_dir()
    os.makedirs(llm_onnx_dir, exist_ok=True)
    os.makedirs(draft_onnx_dir, exist_ok=True)
    for subdir, (target_fn, _) in _OUTPUT_DIR_MAP.items():
        os.makedirs(target_fn(config), exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="mtp_export_")
    try:
        _run_export_subprocess(
            model_dir,
            tmp_dir,
            f"Exporting MTP {config.model_name} via llm_loader",
            test_logger,
            device,
            timeout,
            extra_args=["--mtp"],
            failure_prefix="MTP export")

        llm_output = os.path.join(tmp_dir, "llm")
        if not os.path.isdir(llm_output):
            pytest.fail(f"MTP export did not produce llm/ in {tmp_dir}")
        shutil.copytree(llm_output, llm_onnx_dir, dirs_exist_ok=True)

        draft_output = os.path.join(tmp_dir, "mtp_draft")
        if not os.path.isdir(draft_output):
            pytest.fail(f"MTP export did not produce mtp_draft/ in {tmp_dir}")
        shutil.copytree(draft_output, draft_onnx_dir, dirs_exist_ok=True)

        for subdir, (target_fn, _) in _OUTPUT_DIR_MAP.items():
            _copy_subdir(tmp_dir, subdir, target_fn(config))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Command runner helper
# ---------------------------------------------------------------------------


def run_command_list(commands, label, test_logger, env_vars=None):
    """Execute a list of ``(cmd, timeout)`` tuples, failing on first error."""
    for i, (cmd, timeout) in enumerate(commands):
        result = run_command(cmd,
                             timeout=timeout,
                             remote_config=None,
                             logger=test_logger,
                             env_vars=env_vars)
        if not result['success']:
            pytest.fail(f"{label} step {i+1}/{len(commands)} failed: "
                        f"{result.get('error', 'Unknown error')}")
