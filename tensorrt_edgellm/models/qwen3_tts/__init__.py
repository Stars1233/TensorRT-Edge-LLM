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
"""Qwen3-TTS model components.

Qwen3-TTS has NO audio encoder. Its components are:
- Talker: Qwen3-architecture LLM decoder for speech codec prediction.
- CodePredictor: Small 5-layer Qwen3 decoder for multi-token prediction.
- Code2Wav: speech_tokenizer decoder for waveform synthesis.

See ``modeling_code_predictor.py`` and ``modeling_qwen3_tts_code2wav.py``
for details.
"""
from .modeling_code_predictor import CodePredictorCausalLM
from .modeling_qwen3_tts_code2wav import export_qwen3_tts_code2wav
from .modeling_qwen3_tts_talker import TalkerCausalLM
from .modeling_qwen3_tts_text import Qwen3TTSLanguageModel

__all__ = [
    "Qwen3TTSLanguageModel",
    "TalkerCausalLM",
    "CodePredictorCausalLM",
    "export_qwen3_tts_code2wav",
]
