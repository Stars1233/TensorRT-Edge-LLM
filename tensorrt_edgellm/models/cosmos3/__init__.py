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
"""Cosmos3-Omni model/export support.

The Python model definitions and export contracts live in ``tensorrt_edgellm``.
The experimental C++ runtime remains under ``experimental_models/cosmos3`` and
consumes the component artifacts emitted by this package.
"""

from .modeling_gen import (ACTION_CHUNK_SIZE, DEFAULT_DOMAIN,
                           DEFAULT_NUM_INFERENCE_STEPS, DEFAULT_RAW_ACTION_DIM,
                           Cosmos3GenConfig, build_cosmos3_gen)
from .modeling_und_prefill import build_cosmos3_und_prefill
from .modeling_vae_encoder import (VAE_NUM_FRAMES, build_cosmos3_vae_encoder,
                                   get_vae_onnx_export_args)
from .weights import load_config_json, split_transformer_weights

__all__ = [
    "ACTION_CHUNK_SIZE",
    "DEFAULT_DOMAIN",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_RAW_ACTION_DIM",
    "VAE_NUM_FRAMES",
    "Cosmos3GenConfig",
    "build_cosmos3_gen",
    "build_cosmos3_und_prefill",
    "build_cosmos3_vae_encoder",
    "get_vae_onnx_export_args",
    "load_config_json",
    "split_transformer_weights",
]
