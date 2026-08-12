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
"""Qwen3-Omni tokenizer artifacts."""

from typing import Any, Dict

CHAT_TEMPLATE = "qwen3_omni.json"
OMNI_AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
OMNI_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
OMNI_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _omni_content_formats():
    return (
        ("audio", OMNI_AUDIO_PLACEHOLDER),
        ("image", OMNI_IMAGE_PLACEHOLDER),
        ("video", OMNI_VIDEO_PLACEHOLDER),
    )


def patch_chat_template(template: Dict[str, Any],
                        root_config: Dict[str, Any]) -> None:
    """Patch Qwen3-Omni audio/image/video placeholders."""
    content_types = template.setdefault("content_types", {})
    for content_type, placeholder in _omni_content_formats():
        content_types.setdefault(content_type, {})["format"] = placeholder
    _ = root_config
