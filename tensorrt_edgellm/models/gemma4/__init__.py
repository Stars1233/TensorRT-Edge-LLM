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
"""Gemma4 text, audio, and paired MTP assistant decoder implementations."""

from .modeling_gemma4_assistant import (Gemma4AssistantDecoderLayer,
                                        Gemma4AssistantForCausalLM)
from .modeling_gemma4_audio import (Gemma4AudioModel,
                                    Gemma4AudioMultimodalEmbedder,
                                    Gemma4AudioWithEmbedder,
                                    build_gemma4_audio)
from .modeling_gemma4_text import (GEMMA4_NVFP4_KEY_REMAP, Gemma4Attention,
                                   Gemma4DecoderLayer, Gemma4ForCausalLM,
                                   Gemma4NvFP4MoEBlock, Gemma4NvFP4MoEExperts,
                                   Gemma4Transformer, Gemma4ValueRMSNorm)
from .modeling_gemma4_visual import Gemma4VisualModel, build_gemma4_visual

__all__ = [
    "GEMMA4_NVFP4_KEY_REMAP",
    "Gemma4Attention",
    "Gemma4ForCausalLM",
    "Gemma4AssistantForCausalLM",
    "Gemma4DecoderLayer",
    "Gemma4AssistantDecoderLayer",
    "Gemma4NvFP4MoEBlock",
    "Gemma4NvFP4MoEExperts",
    "Gemma4Transformer",
    "Gemma4ValueRMSNorm",
    "Gemma4AudioModel",
    "Gemma4AudioMultimodalEmbedder",
    "Gemma4AudioWithEmbedder",
    "build_gemma4_audio",
    "Gemma4VisualModel",
    "build_gemma4_visual",
]
