# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Few-layer numeric validation smoke test.

Early-fail smoke: runs the first N decoder layers of a checkpoint through both
a PyTorch golden and the EdgeLLM engine, comparing per-round logits + KV /
recurrent / conv state (see ``scripts/few-layer-validation.sh``). The script's
exit code is the PASS/FAIL gate; this test is a thin wrapper that resolves the
checkpoint, points the script at the CI build, and runs it.

Host (server) only: the golden runs PyTorch + transformers locally, so the test
skips under remote (edge-device) execution. The parametrization is driven by the
``test_param`` ids listed in ``tests/test_lists/*.yml`` (one per model)."""

import logging
import os
import sys
from typing import Optional

import pytest
from conftest import EnvironmentConfig
from pytest_helpers import run_command, timer_context

from .config import _HF_CHECKPOINT_FILES, _find_directory
from .utils.device import DeviceConfig

# test_param id -> validation spec.
#   dir_name      : checkpoint dir/relpath, searched recursively under the model
#                   roots (llm_models_dir, then edgellm_data_dir).
#   num_layers    : leading decoder layers to validate.
#   cos_threshold : min per-tensor cosine (measured over the first N layers):
#                   FP16 / GDN-hybrid ~0.9999; NVFP4 is lower (~0.974 worst) because
#                   its Mamba recurrent state has a large dynamic range, so quant
#                   noise shows up strongly per-tensor even when logits stay ~0.98.
#   needs_cutedsl : True for hybrid (Gated DeltaNet / Mamba) models and NVFP4, whose
#                   kernels only build on Blackwell (SM100+).
_FEW_LAYER_MODELS = {
    "Qwen3-0.6B": {
        "dir_name": "Qwen3/Qwen3-0.6B",
        "num_layers": 4,
        "cos_threshold": 0.99,
        "needs_cutedsl": False,
    },
    "NVIDIA-Nemotron-3-Nano-4B-NVFP4": {
        # ModelOpt export keys the body under ``backbone.``; the golden realigns
        # it to the native ``model.`` prefix (see _load_quantized_state).
        "dir_name": "NVIDIA-Nemotron-3-Nano-4B-NVFP4",
        "num_layers": 4,
        "cos_threshold": 0.96,
        "needs_cutedsl": True,
    },
    # GDN hybrid. Not wired into the b100 CI list: its GatedDeltaNet engine
    # needs CuTe DSL kernels that the CI build does not enable
    # (CUTE_DSL_GDN_ENABLED), so engine build fails there. Kept here so it can
    # be run manually on a GDN-enabled build (validated at cos 0.99998).
    "Qwen3.5-0.8B": {
        "dir_name": "Qwen3.5-0.8B",
        "num_layers": 4,
        "cos_threshold": 0.99,
        "needs_cutedsl": True,
    },
}


def _resolve_model_dir(env_config: EnvironmentConfig,
                       dir_name: str) -> Optional[str]:
    """Locate a checkpoint dir under the known model roots (recursive search)."""
    for root in (env_config.llm_models_dir, env_config.edgellm_data_dir):
        found = _find_directory(root,
                                dir_name,
                                require_files=_HF_CHECKPOINT_FILES)
        if found:
            return found
    return None


def test_few_layer_validation(test_param: str, env_config: EnvironmentConfig,
                              remote_config, test_logger: logging.Logger):
    """Functional few-layer golden-vs-engine numeric smoke."""
    spec = _FEW_LAYER_MODELS.get(test_param)
    if spec is None:
        pytest.fail(f"unknown few-layer model id '{test_param}'; "
                    f"known ids: {sorted(_FEW_LAYER_MODELS)}")

    # Host (server) only: the golden runs PyTorch + transformers locally, which
    # the edge boards (remote execution) do not have.
    if remote_config is not None:
        pytest.skip("few-layer validation runs on the host (server) only; "
                    "the PyTorch golden needs torch + transformers")

    # Hybrid (Gated DeltaNet / Mamba) and NVFP4 models need the CuTe DSL kernels,
    # which only build on Blackwell (SM100+; verified on SM100 / B-series). Skip on
    # older architectures rather than fail so the same list stays reusable there.
    if spec["needs_cutedsl"]:
        device_config = DeviceConfig.auto_detect(remote_config, test_logger)
        cc = device_config.compute_capability
        if cc is None or cc < 100:
            pytest.skip(
                f"{test_param} needs CuTe DSL (Blackwell SM100+); this "
                f"runner is compute capability {cc}")

    model_dir = _resolve_model_dir(env_config, spec["dir_name"])
    if model_dir is None:
        pytest.skip(f"checkpoint '{spec['dir_name']}' not found under "
                    f"{env_config.llm_models_dir} / "
                    f"{env_config.edgellm_data_dir}")

    # build_dir may be relative ("build"); the script needs an absolute path
    # since it resolves binaries against it independent of its own CWD.
    build_dir = env_config.build_dir
    if not os.path.isabs(build_dir):
        build_dir = os.path.join(env_config.llm_sdk_dir, build_dir)

    script = os.path.join(env_config.llm_sdk_dir, "scripts",
                          "few-layer-validation.sh")
    cmd = [
        "bash",
        script,
        "--model",
        model_dir,
        "--num-layers",
        str(spec["num_layers"]),
        "--build-dir",
        build_dir,
        # The pytest interpreter has torch + transformers + the TRT wheel.
        "--python",
        sys.executable,
        "--cos",
        str(spec["cos_threshold"]),
    ]

    with timer_context(f"few-layer-validation [{test_param}]", test_logger):
        result = run_command(cmd=cmd,
                             remote_config=remote_config,
                             timeout=1200,
                             logger=test_logger)

    if not result["success"]:
        pytest.fail(f"few-layer validation FAILED for {test_param} "
                    f"(exit {result.get('returncode')}); see log above")
