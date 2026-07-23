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
"""Dependency-free attention-scale resolution for calibration models."""

import math
from collections.abc import Mapping
from typing import Any


def _get(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def resolve_attention_scale(config: Any, model_type: str,
                            head_dim: int) -> float:
    """Match runtime attention-scale aliases, validation, and fallbacks."""
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive; got {head_dim}")

    for key in ("attention_scaling", "qk_scale", "scaling"):
        value = _get(config, key)
        if value is not None:
            attention_scale = float(value)
            if not math.isfinite(attention_scale) or attention_scale <= 0.0:
                raise ValueError(
                    f"{key} must be finite and positive; got {value!r}")
            return attention_scale

    if str(model_type).startswith("gemma4"):
        return 1.0

    return 1.0 / (float(head_dim)**0.5)


__all__ = ["resolve_attention_scale"]
