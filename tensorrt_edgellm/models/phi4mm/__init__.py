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
"""Phi-4 Multimodal model components: visual encoder + LLM backbone."""
from .modeling_phi4mm_text import Phi4MMLanguageModel
from .modeling_phi4mm_visual import Phi4MMVisualModel, build_phi4mm_visual

__all__ = ["Phi4MMVisualModel", "build_phi4mm_visual", "Phi4MMLanguageModel"]
