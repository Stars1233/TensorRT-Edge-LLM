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
"""Qwen3-Omni model components: visual encoder, audio encoder, LLM backbone,
and Code2Wav vocoder."""
from .modeling_qwen3_omni_audio import (Qwen3OmniAudioEncoder,
                                        build_qwen3_omni_audio)
from .modeling_qwen3_omni_code2wav import (Code2WavModel, build_code2wav,
                                           export_code2wav_onnx)
from .modeling_qwen3_omni_text import Qwen3OmniLanguageModel
from .modeling_qwen3_omni_visual import (Qwen3OmniVisualModel,
                                         build_qwen3_omni_visual)

__all__ = [
    "Qwen3OmniVisualModel",
    "build_qwen3_omni_visual",
    "Qwen3OmniAudioEncoder",
    "build_qwen3_omni_audio",
    "Qwen3OmniLanguageModel",
    "Code2WavModel",
    "build_code2wav",
    "export_code2wav_onnx",
]
