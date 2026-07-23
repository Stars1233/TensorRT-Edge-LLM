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
ViTAttentionPlugin tests vs a PyTorch reference.

The ViT attention plugin is varlen-packed full (bidirectional / padding-mask)
self-attention: Q/K/V are [total_tokens, H, D], segmented by cu_seqlens
[batch+1], MHA (num_q == num_kv), softmax scale 1/sqrt(D), FP16. Head dims
supported by the CuTe-DSL ViT runner: {64, 72, 80, 128}.

Covers even/uneven batch cases (bs 1/2/3/4/8, seq up to 2048), a head-size /
num-heads sweep, and batch-shuffle invariance (plugin-vs-plugin).

Run:
    python3 -m pytest tests/python-unittests/test_vit_attention_plugin.py -v
"""

from __future__ import annotations

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR, MAX_SEQ,
                              RAGGED_CASES, PluginRunner, assert_close,
                              pf_int32)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

DEV = "cuda"
VIT_HEAD_DIMS = [64, 72, 80, 128]  # CuteDslFMHARunner::canImplementViT
MAX_TOTAL_TOKENS = 8192


def vit_attention_ref(q, k, v, cu_seqlens, head_size):
    """Varlen bidirectional MHA. q/k/v [total_S, H, D] -> out [total_S, H, D]."""
    out = torch.zeros_like(q, dtype=torch.float32)
    scale = head_size**-0.5
    qf, kf, vf = q.float(), k.float(), v.float()
    for b in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        if e <= s:
            continue
        qs = qf[s:e].transpose(0, 1)  # [H, L, D]
        ks = kf[s:e].transpose(0, 1)
        vs = vf[s:e].transpose(0, 1)
        scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale  # [H, L, L]
        w = torch.softmax(scores, dim=-1)
        out[s:e] = torch.matmul(w, vs).transpose(0, 1)  # [L, H, D]
    return out


class ViTRunner:

    def __init__(self, num_heads: int, head_size: int):
        self.h, self.d = num_heads, head_size
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        h, d = self.h, self.d
        F16, I32 = trt.float16, trt.int32
        self.runner.build(
            input_specs=[
                ("q", F16, (-1, h, d)),
                ("k", F16, (-1, h, d)),
                ("v", F16, (-1, h, d)),
                ("cu_seqlens", I32, (-1, )),
                ("max_seqlen_carrier", I32, (-1, )),
            ],
            output_names=["out"],
            plugin_name="ViTAttentionPlugin",
            plugin_version="1",
            plugin_fields=[pf_int32("num_heads", h),
                           pf_int32("head_size", d)],
            profiles={
                "q": ((1, h, d), (512, h, d), (MAX_TOTAL_TOKENS, h, d)),
                "k": ((1, h, d), (512, h, d), (MAX_TOTAL_TOKENS, h, d)),
                "v": ((1, h, d), (512, h, d), (MAX_TOTAL_TOKENS, h, d)),
                "cu_seqlens": ((2, ), (3, ), (MAX_BATCH_P1, )),
                "max_seqlen_carrier": ((1, ), (512, ), (MAX_SEQ, )),
            },
        )

    def run(self, q, k, v, cu_seqlens, max_seqlen):
        out = torch.empty_like(q)
        carrier = torch.zeros(max_seqlen, dtype=torch.int32, device=DEV)
        self.runner.execute({
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens": cu_seqlens,
            "max_seqlen_carrier": carrier,
            "out": out,
        })
        return out


# Max segments (+1 for the cu_seqlens prefix sum): covers the required batch
# cases (up to 8 images) and window-partition shapes (32 windows per call).
MAX_BATCH_P1 = 33


def _cu_seqlens(seqlens):
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32).cumsum(0)
    return cu


def _rand_packed(total, h, d, gen):
    return torch.randn(total, h, d, generator=gen,
                       dtype=torch.float32).to(torch.float16).to(DEV)


# --------------------------------------------------------------------------- #
# Required even/uneven batch cases (bs 1/2/3/4/8) vs PyTorch reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,seqlens", RAGGED_CASES)
def test_ragged_prefill_batch_sizes(label, seqlens):
    h, d = 8, 64
    gen = torch.Generator().manual_seed(hash(label) % 2**31)
    total = sum(seqlens)
    q, k, v = (_rand_packed(total, h, d, gen) for _ in range(3))
    cu = _cu_seqlens(seqlens)
    r = ViTRunner(h, d)
    out = r.run(q, k, v, cu, max(seqlens))
    ref = vit_attention_ref(q, k, v, cu, d)
    assert_close(f"vit[{label}]", ref, out)


# --------------------------------------------------------------------------- #
# Window-partition shape: Qwen2.5-VL window-attention layers reorder tokens
# window-first and feed the plugin many equal 64-token segments (8x8 patch
# windows) through cu_window_seqlens. 32 windows x 64 tokens, with the
# Qwen2.5-VL vision head config (16 heads, head dim 80).
# --------------------------------------------------------------------------- #
def test_window_partition():
    h, d = 16, 80
    seqlens = [64] * 32
    gen = torch.Generator().manual_seed(2564)
    total = sum(seqlens)
    q, k, v = (_rand_packed(total, h, d, gen) for _ in range(3))
    cu = _cu_seqlens(seqlens)
    r = ViTRunner(h, d)
    out = r.run(q, k, v, cu, max(seqlens))
    ref = vit_attention_ref(q, k, v, cu, d)
    assert_close("vit[window-partition]", ref, out)


# --------------------------------------------------------------------------- #
# Head-size / num-heads sweep (CuTe-DSL ViT supports head dim 64/72/80/128)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("head_size", VIT_HEAD_DIMS, ids=lambda d: f"head{d}")
@pytest.mark.parametrize("num_heads", [8, 14, 16], ids=lambda h: f"nheads{h}")
def test_head_sweep(head_size, num_heads):
    seqlens = [60, 128, 16, 300]
    gen = torch.Generator().manual_seed(700 + head_size + num_heads)
    total = sum(seqlens)
    q, k, v = (_rand_packed(total, num_heads, head_size, gen)
               for _ in range(3))
    cu = _cu_seqlens(seqlens)
    r = ViTRunner(num_heads, head_size)
    out = r.run(q, k, v, cu, max(seqlens))
    ref = vit_attention_ref(q, k, v, cu, head_size)
    assert_close(f"vit[h{num_heads}d{head_size}]", ref, out)


# --------------------------------------------------------------------------- #
# Batch-shuffle invariance (plugin-vs-plugin): reordering segments must
# reorder the outputs identically.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,seqlens",
                         [c for c in RAGGED_CASES if len(c[1]) > 1])
def test_batch_invariance(label, seqlens):
    h, d = 8, 64
    gen = torch.Generator().manual_seed(900 + len(seqlens))
    segs = [_rand_packed(s, h, d, gen) for s in seqlens]
    q = torch.cat(segs, 0)
    k = torch.cat([_rand_packed(s, h, d, gen) for s in seqlens], 0)
    v = torch.cat([_rand_packed(s, h, d, gen) for s in seqlens], 0)
    cu = _cu_seqlens(seqlens)
    r = ViTRunner(h, d)
    out0 = r.run(q, k, v, cu, max(seqlens))

    # Shuffle segment order, run again, then un-shuffle and compare.
    perm = list(range(len(seqlens)))
    perm = perm[len(perm) // 2:] + perm[:len(perm) //
                                        2]  # deterministic rotate
    new_seqlens = [seqlens[i] for i in perm]

    def reorder(x):
        offs = [0]
        for s in seqlens:
            offs.append(offs[-1] + s)
        return torch.cat([x[offs[i]:offs[i + 1]] for i in perm], 0)

    qs, ks, vs = reorder(q), reorder(k), reorder(v)
    cu_s = _cu_seqlens(new_seqlens)
    out_s = r.run(qs, ks, vs, cu_s, max(seqlens))

    # out_s should equal the permuted segments of out0.
    out0_perm = reorder(out0)
    assert_close(f"vit-batch-invariance[{label}]", out0_perm, out_s)
