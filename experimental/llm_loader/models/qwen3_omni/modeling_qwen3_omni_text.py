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
Qwen3-Omni LLM backbone (thinker).

The Qwen3-Omni thinker is a standard Qwen3 transformer decoder, but the TTS
pipeline feeds the thinker's full-sequence last-layer normed hidden states
into the Talker's input projection. This subclass enables the extra
``hidden_states`` ONNX output so downstream tooling doesn't need to patch it
in after the fact.
"""
from ..default.modeling_default import CausalLM

__all__ = ["Qwen3OmniLanguageModel"]


class Qwen3OmniLanguageModel(CausalLM):
    """Qwen3-Omni thinker: standard Qwen3 decoder with ``hidden_states`` output.

    The Talker consumes the thinker's full-sequence last-layer normed hidden
    states; exposing them as a named ONNX output avoids a post-export patch.
    """

    emit_hidden_states = True
