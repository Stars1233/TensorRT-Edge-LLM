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
"""Content-type rendering contracts of the hardcoded chat template JSONs."""

import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "tensorrt_edgellm", "chat_templates"))


def _load_template(name):
    with open(os.path.join(_TEMPLATES_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def test_qwen3_omni_media_wrapped_with_boundary_tokens():
    # C++ getMRopePositionIds only applies MRoPE to <|vision_start|>-anchored
    # pad runs; bare pads silently degrade to 1D text positions.
    content_types = _load_template("qwen3_omni.json")["content_types"]
    assert content_types["image"]["format"] == \
        "<|vision_start|><|image_pad|><|vision_end|>"
    assert content_types["video"]["format"] == \
        "<|vision_start|><|video_pad|><|vision_end|>"
    assert content_types["audio"]["format"] == \
        "<|audio_start|><|audio_pad|><|audio_end|>"


def test_qwen3asr_audio_wrapped_with_boundary_tokens():
    # Same contract for the ASR template: the C++ audio runner anchors on the
    # boundary pair, so a bare pad drops the encoder embeddings.
    content_types = _load_template("qwen3asr.json")["content_types"]
    assert content_types["audio"]["format"] == \
        "<|audio_start|><|audio_pad|><|audio_end|>"
