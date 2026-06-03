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
"""Standalone Qwen3-ASR modeling for quantization.

Pure-PyTorch reimplementation of the Qwen3-ASR audio encoder and a joint
audio + text calibration model, used only during ModelOpt ``mtq.quantize``.

Mirrors the layout in :mod:`tensorrt_edgellm.quantization.models.eagle3_draft`:
this side owns its own modeling so the calibration tree never imports from
``tensorrt_edgellm`` (whose audio modeling threads through
``make_linear`` + ``FP8Linear`` / ``NVFP4Linear`` custom classes that are
needed at ONNX-export time but undesirable during pure-precision
calibration).
"""

from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration  # noqa: F401
from .modeling_qwen3_asr_audio import Qwen3ASRAudioEncoder  # noqa: F401
