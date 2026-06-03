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
"""Qwen3.5 model components: visual encoder + LLM backbone."""
from .modeling_qwen3_5_mtp import Qwen3_5MtpDecoderLayer, Qwen3_5MtpDraftModel
from .modeling_qwen3_5_text import Qwen3_5CausalLM, fuse_gdn_input_projections
from .modeling_qwen3_5_visual import Qwen3_5VLVisualModel, build_qwen3_5_visual

__all__ = [
    "Qwen3_5VLVisualModel",
    "build_qwen3_5_visual",
    "Qwen3_5CausalLM",
    "fuse_gdn_input_projections",
    "Qwen3_5MtpDraftModel",
    "Qwen3_5MtpDecoderLayer",
]
