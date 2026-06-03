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
InternVL LLM backbone.

The InternVL3 text decoder uses InternLM2 or a similar transformer architecture.
The default ``AutoModel`` pipeline handles it automatically from ``config.json``.

For export:
    from tensorrt_edgellm.model import AutoModel
    from tensorrt_edgellm.onnx.export import export_onnx
    model = AutoModel.from_pretrained(model_dir)
    export_onnx(model, output_path, model_dir=model_dir)
"""
from ..default.modeling_default import CausalLM as InternVLLanguageModel

__all__ = ["InternVLLanguageModel"]
