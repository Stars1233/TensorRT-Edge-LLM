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
Sliding-window attention tests for AttentionPlugin.

Split into its own file since sliding window is a distinct feature (and a
scale-up target). Reuses the AttentionPlugin harness/reference from
test_attention_plugin. Validated against the PyTorch reference at cos > 0.99999.

The plugin uses one window semantic everywhere (prefill via CuTe DSL FMHA or
FMHA_v2, decode via XQA): ``sliding_window_size`` is the number of keys
attended in total, query included -- the HF ``sliding_window`` convention. See
sliding_window_mask in test_attention_plugin.
"""

from __future__ import annotations

import pytest
from test_attention_plugin import (BASE, DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                                   AttentionParams, _run_rounds)

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")


@pytest.mark.parametrize("window", [4, 16], ids=lambda w: f"window{w}")
def test_sliding_window_prefill(window):
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        sliding_window_size=window,
                        **BASE)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


# Gemma4 sliding-attention layers use head 256 with a window. Head 256 has no
# CuTe DSL sliding kernel, so this exercises the FMHA_v2 sliding path on every
# SKU.
@pytest.mark.parametrize("num_kv_heads", [2], ids=lambda k: f"kv{k}")
def test_sliding_window_prefill_head256(num_kv_heads):
    cfg = dict(BASE)
    cfg["head_size"] = 256
    cfg["num_kv_heads"] = num_kv_heads
    p = AttentionParams(batch_size=2,
                        seq_len=8,
                        is_prefill=True,
                        sliding_window_size=4,
                        **cfg)
    _run_rounds(p, num_rounds=2, atol=1e-2, rtol=1e-2)


def test_sliding_window_decode():
    # capacity > window so the window actually clips history during decode
    cfg = dict(BASE)
    cfg["kv_cache_capacity"] = 32
    cfg["max_seq_len"] = 24
    cfg["max_position_embeddings"] = 64
    p = AttentionParams(batch_size=2, seq_len=1, sliding_window_size=8, **cfg)
    _run_rounds(p, num_rounds=12, atol=1e-2, rtol=1e-2)
