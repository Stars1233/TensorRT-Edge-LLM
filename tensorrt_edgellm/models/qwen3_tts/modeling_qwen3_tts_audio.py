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
Qwen3-TTS has **no audio encoder**.

Unlike Qwen3-ASR (which encodes mel spectrograms), Qwen3-TTS is a
text-to-speech model that generates speech codec tokens from text.
Its components are:

- **Talker**: A Qwen3-architecture LLM decoder that predicts speech codec
  tokens from text tokens (``talker.*`` weight prefix, see
  ``modeling_qwen3_tts_talker.py``).
- **CodePredictor**: A small 5-layer Qwen3 decoder for multi-token prediction
  (``talker.code_predictor.*`` weight prefix).
- **Tokenizer (Vocoder)**: Converts codec tokens to waveform — exported via
  the ``tensorrt_edgellm`` package (see ``audio_models/qwen3_tts_audio_model.py``).

Neither the Talker nor the CodePredictor is an "audio encoder" in the
sense of processing mel features. They are LLM decoders exported via the
standard LLM backbone pipeline (``tensorrt_edgellm.model.AutoModel``).
"""

__all__: list = []
