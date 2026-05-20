# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Action expert export for TensorRT Edge-LLM.

This module provides the main entry point to export the Alpamayo action expert
(expert + action_in_proj + action_out_proj, one flow-matching step) to ONNX.
Export implementations live in the action_models package.
"""

import json
import os

import torch

from ..action_models.alpamayo_model import (Alpamayo1ActionExpertPatch,
                                            export_alpamayo1_action)
from ..llm_models.model_utils import load_hf_model
from .config_export import export_action_config


def action_export(
    model_dir: str,
    output_dir: str,
    max_kv_cache_capacity: int,
    device: str = "cuda",
    dtype: str = "fp16",
) -> str:
    """
    Export the Alpamayo action expert to ONNX.

    Loads an Alpamayo 1 model from model_dir, wraps the expert in the patched
    model, loads weights, and exports one flow-matching step to
    output_dir/model.onnx with config.json.

    Args:
        model_dir: Directory containing the Alpamayo 1 model.
        output_dir: Directory to save the exported ONNX model (model.onnx).
        max_kv_cache_capacity: Maximum KV cache capacity.
        device: Device to load the model on (e.g. "cuda", "cuda:0").
        dtype: Export dtype; only "fp16" is supported.

    Returns:
        Path to the output directory where the exported model is saved.

    Raises:
        ValueError: If dtype is not "fp16" or model type is not supported.
    """
    if dtype != "fp16":
        raise ValueError(
            f"Only fp16 is supported for action export. Got: {dtype}")

    try:
        model, _, _ = load_hf_model(model_dir, dtype, device)
    except Exception as e:
        raise ValueError(f"Could not load model from {model_dir}. Error: {e}")

    model_type = model.config.model_type
    torch_dtype = torch.float16

    os.makedirs(output_dir, exist_ok=True)

    if model_type == "alpamayo_r1":
        print(f"Exporting Alpamayo 1 action expert from {model_dir}")

        patched_model = Alpamayo1ActionExpertPatch(
            model,
            device=device,
            torch_dtype=torch_dtype,
        )

        export_alpamayo1_action(
            patched_model,
            output_dir,
            max_kv_cache_capacity,
            device=device,
            torch_dtype=torch_dtype,
        )

        # from https://github.com/NVlabs/alpamayo/blob/2d2b511ccd626b97aab0c9c211861c0e768551aa/src/alpamayo_r1/models/delta_tokenizer.py#L30
        model.expert.config.num_traj_tokens = 1000
        model.expert.config.traj_token_start = model.config.traj_token_start_idx + model.config.traj_tokenizer_cfg[
            "num_bins"]
        model.expert.config.n_diffusion_tokens = patched_model.n_diffusion_tokens

        config_dict = export_action_config(model.expert.config,
                                           max_kv_cache_capacity)

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(
        f"Action expert export completed for {model_type} with dtype={dtype}, device={device}"
    )
    print(f"Exported to: {output_dir}")
    return output_dir
