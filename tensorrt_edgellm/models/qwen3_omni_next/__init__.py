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
"""Qwen3-Next Omni model components: visual / audio / text / Talker / CodePredictor.

Each sub-model lives in its own ``modeling_qwen3_omni_next_*`` file.
"""
from .modeling_qwen3_omni_next_audio import (Qwen3OmniNextAudioEncoder,
                                             build_qwen3_omni_next_audio)
from .modeling_qwen3_omni_next_code2wav import \
    Code2WavModel as Qwen3OmniNextCode2WavModel
from .modeling_qwen3_omni_next_code2wav import (
    build_qwen3_omni_next_code2wav, export_qwen3_omni_next_code2wav_onnx,
    load_code2wav_config)
from .modeling_qwen3_omni_next_code_predictor import \
    Qwen3OmniNextCodePredictorCausalLM
from .modeling_qwen3_omni_next_moe_talker import Qwen3OmniNextMoeTalkerCausalLM
from .modeling_qwen3_omni_next_mtp import Qwen3OmniNextMoeMtpDraftModel
from .modeling_qwen3_omni_next_talker import Qwen3OmniNextTalkerCausalLM
from .modeling_qwen3_omni_next_text import Qwen3OmniNextLanguageModel
from .modeling_qwen3_omni_next_visual import (Qwen3OmniNextVisualModel,
                                              build_qwen3_omni_next_visual)

__all__ = [
    "Qwen3OmniNextAudioEncoder",
    "build_qwen3_omni_next_audio",
    "Qwen3OmniNextVisualModel",
    "build_qwen3_omni_next_visual",
    "Qwen3OmniNextLanguageModel",
    "Qwen3OmniNextTalkerCausalLM",
    "Qwen3OmniNextMoeTalkerCausalLM",
    "Qwen3OmniNextMoeMtpDraftModel",
    "Qwen3OmniNextCodePredictorCausalLM",
    "Qwen3OmniNextCode2WavModel",
    "build_qwen3_omni_next_code2wav",
    "export_qwen3_omni_next_code2wav_onnx",
    "load_code2wav_config",
]
