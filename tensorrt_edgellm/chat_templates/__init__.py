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
"""Bundled processed chat-template helpers."""

from pathlib import Path
from typing import Optional

from tensorrt_edgellm.chat_template import (
    process_chat_template, write_fallback_processed_chat_template)

_TEMPLATES_DIR = Path(__file__).parent


def get_template_path(model_identifier: str) -> Optional[str]:
    """Return the bundled template path for *model_identifier*, if present."""
    template_path = _TEMPLATES_DIR / f"{model_identifier}.json"
    if template_path.exists():
        return str(template_path)
    return None


__all__ = [
    "get_template_path",
    "process_chat_template",
    "write_fallback_processed_chat_template",
]
