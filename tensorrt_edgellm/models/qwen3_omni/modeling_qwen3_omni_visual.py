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
"""Qwen3-Omni visual encoder — same architecture as Qwen3-VL."""
from ..qwen3_vl.modeling_qwen3_vl_visual import \
    Qwen3VLVisualModel as Qwen3OmniVisualModel
from ..qwen3_vl.modeling_qwen3_vl_visual import \
    build_qwen3_vl_visual as build_qwen3_omni_visual

__all__ = ["Qwen3OmniVisualModel", "build_qwen3_omni_visual"]
