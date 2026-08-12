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
"""Nemotron-3.5-ASR: FastConformer encoder + RNN-T (LSTM) decoder step."""
from .modeling_nemotron3_5_asr_audio import (Nemotron3_5AsrAudioModel,
                                             build_nemotron3_5_asr_audio)
from .modeling_nemotron3_5_asr_decoder import (Nemotron3_5AsrRNNTStepModel,
                                               build_nemotron3_5_asr_decoder)

__all__ = [
    "Nemotron3_5AsrAudioModel",
    "build_nemotron3_5_asr_audio",
    "Nemotron3_5AsrRNNTStepModel",
    "build_nemotron3_5_asr_decoder",
]
