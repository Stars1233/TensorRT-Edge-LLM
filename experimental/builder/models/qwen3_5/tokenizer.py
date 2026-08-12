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
"""Qwen3.5 tokenizer artifacts."""

from typing import Any, Dict

_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def patch_chat_template(template: Dict[str, Any],
                        root_config: Dict[str, Any]) -> None:
    """Preserve provider vision placeholders in the runtime template."""
    content_types = template.setdefault("content_types", {})
    content_types.setdefault("image", {})["format"] = _IMAGE_PLACEHOLDER
    if root_config.get("video_token_id") is not None:
        content_types.setdefault("video", {})["format"] = _VIDEO_PLACEHOLDER
