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
"""Qwen3-Omni-Next tokenizer and conditioning artifacts."""

import os
import shutil
from typing import Any, Dict

CHAT_TEMPLATE = "qwen3_omni.json"
_AUDIO = "<|audio_start|><|audio_pad|><|audio_end|>"
_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO = "<|vision_start|><|video_pad|><|vision_end|>"


def patch_chat_template(template: Dict[str, Any],
                        root_config: Dict[str, Any]) -> None:
    """Patch the provider's multimodal placeholders."""
    content_types = template.setdefault("content_types", {})
    for kind, placeholder in (("audio", _AUDIO), ("image", _IMAGE), ("video",
                                                                     _VIDEO)):
        content_types.setdefault(kind, {})["format"] = placeholder
    _ = root_config


def patch_runtime_artifacts(output_dir: str, args) -> None:
    """Copy the friendly-name speaker table used by the Next Talker."""
    source = os.path.join(args.model_dir, "voice_map.json")
    if os.path.isfile(source):
        shutil.copy2(source, os.path.join(output_dir, "voice_map.json"))
