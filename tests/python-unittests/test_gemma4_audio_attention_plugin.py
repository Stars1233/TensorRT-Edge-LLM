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
"""Gemma4AudioAttentionPlugin vs a PyTorch reference.

The plugin fuses the Gemma 4 audio-encoder (Conformer) chunked local attention:
per-dim learned query scaling, fixed key scaling, left-context K/V gather,
content + relative-position (Transformer-XL rel-shift) scores, tanh soft-cap,
local-causal + padding mask, fp32 softmax, value mix. Specialized to the audio
config: head_dim=128, rel_pos_len=13, chunk=12, left_horizon=12, context=24.
"""

from __future__ import annotations

import math

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              PluginRunner, _fail_unsupported, assert_close,
                              pf_float32, pf_int32, poison_padding)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

DEV = "cuda"

# Gemma 4 audio config; the kernel is compile-time specialized to these
# constants (head count is the only free dimension).
NUM_HEADS = 8
HEAD_DIM = 128
REL_POS_LEN = 13  # P = context_size // 2 + 1
CHUNK_SIZE = 12  # C: query block size
LEFT_HORIZON = 12  # L = attention_context_left - 1
CONTEXT_SIZE = 24  # M = chunk + left horizon + right context
LOGIT_CAP = 50.0

MAX_BATCH = 8
MAX_SEQ = 240


def gemma4_audio_attention_ref(q_raw, k_raw, v, gamma, rel_key, valid):
    """Chunked local attention reference. q_raw/k_raw/v [B, S, H, D],
    gamma [D] fp32, rel_key [P, H, D], valid [B, S] bool -> out [B, S, H, D].

    Chunk n query a attends key j = n*C - L + m at position i = n*C + a; slot
    allowed iff 0 <= i-j < L, j in sequence, valid[b,j] (else a -1e9 logit).
    """
    B, S, H, D = q_raw.shape
    P = rel_key.shape[0]
    C, L, M = CHUNK_SIZE, LEFT_HORIZON, CONTEXT_SIZE
    n_chunks = (S + C - 1) // C
    dtype = q_raw.dtype
    dev = q_raw.device

    # Q/K scaling; the kernel stages the scaled values back in the element
    # type before the fp32 dot products.
    q_scalar = D**-0.5 / math.log(2.0)
    k_scale = math.log1p(math.exp(1.0)) / math.log(2.0)
    sp_gamma = torch.nn.functional.softplus(gamma.float())  # [D]
    q = (q_raw.float() * q_scalar * sp_gamma).to(dtype).float()
    k = (k_raw.float() * k_scale).to(dtype).float()
    vf = v.float()

    # Pad queries out to whole chunks: [B, n_chunks, C, H, D].
    pad = n_chunks * C - S
    q_blk = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
    q_blk = q_blk.view(B, n_chunks, C, H, D)

    # Context gather: key position j = n*C - L + m; out-of-range -> zeros.
    j = (torch.arange(n_chunks, device=dev).view(-1, 1) * C - L +
         torch.arange(M, device=dev).view(1, -1))  # [n_chunks, M]
    in_seq = (j >= 0) & (j < S)
    jc = j.clamp(0, S - 1)
    k_ctx = k[:, jc] * in_seq[..., None, None]  # [B, n_chunks, M, H, D]
    v_ctx = vf[:, jc] * in_seq[..., None, None]

    # Content scores AC[a, m] = Q[i] . K[j]  -> [B, H, n_chunks, C, M].
    ac = torch.einsum("bnahd,bnmhd->bhnam", q_blk, k_ctx)

    # Relative scores R[a, t] = Q[i] . P[t], then the Transformer-XL blocked
    # shift: pad the last dim to M+1, flatten, take the first C*M entries.
    r = torch.einsum("bnahd,phd->bhnap", q_blk, rel_key.float())
    r = torch.nn.functional.pad(r, (0, M + 1 - P))  # [B, H, n_chunks, C, M+1]
    bd = r.reshape(B, H, n_chunks, C * (M + 1))[..., :C * M]
    bd = bd.reshape(B, H, n_chunks, C, M)

    # Soft-cap, then the local-causal + padding mask (finite fill).
    logits = LOGIT_CAP * torch.tanh((ac + bd) / LOGIT_CAP)
    dist = (torch.arange(C, device=dev).view(-1, 1) + L -
            torch.arange(M, device=dev).view(1, -1))  # i - j, [C, M]
    window = (dist >= 0) & (dist < L)
    key_ok = valid[:, jc] & in_seq  # [B, n_chunks, M]
    q_pos = (torch.arange(n_chunks, device=dev).view(-1, 1) * C +
             torch.arange(C, device=dev).view(1, -1))  # [n_chunks, C]
    allowed = (window.view(1, 1, 1, C, M)
               & key_ok.view(B, 1, n_chunks, 1, M)
               & (q_pos < S).view(1, 1, n_chunks, C, 1))
    logits = torch.where(allowed, logits, torch.tensor(-1e9, device=dev))

    # fp32 softmax over the M context slots, then the value mix.
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("bhnam,bnmhd->bnahd", probs, v_ctx)
    return out.reshape(B, n_chunks * C, H, D)[:, :S].to(dtype)


class Gemma4AudioRunner:
    """Builds + runs Gemma4AudioAttentionPlugin with dynamic batch/seq."""

    def __init__(self, trt_dtype):
        self.trt_dtype = trt_dtype
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        h, d, p = NUM_HEADS, HEAD_DIM, REL_POS_LEN
        DT, F32, I32, BOOL = self.trt_dtype, trt.float32, trt.int32, trt.bool
        # A build missing this plugin is a missing gate or a broken build.
        registry = trt.get_plugin_registry()
        if registry.get_creator("Gemma4AudioAttentionPlugin", "1", "") is None:
            _fail_unsupported("Gemma4AudioAttentionPlugin not present in this "
                              "plugin library build")
        qkv_prof = ((1, 1, h, d), (1, 120, h, d), (MAX_BATCH, MAX_SEQ, h, d))
        self.runner.build(
            input_specs=[
                ("q_raw", DT, (-1, -1, h, d)),
                ("k_raw", DT, (-1, -1, h, d)),
                ("v", DT, (-1, -1, h, d)),
                ("gamma", F32, (d, )),
                ("rel_key", DT, (p, h, d)),
                ("valid", BOOL, (-1, -1)),
                ("seq_len_carrier", I32, (1, )),
            ],
            output_names=["out"],
            plugin_name="Gemma4AudioAttentionPlugin",
            plugin_version="1",
            plugin_fields=[
                pf_int32("chunk_size", CHUNK_SIZE),
                pf_int32("left_horizon", LEFT_HORIZON),
                pf_int32("context_size", CONTEXT_SIZE),
                pf_float32("logit_cap", LOGIT_CAP),
            ],
            profiles={
                "q_raw": qkv_prof,
                "k_raw": qkv_prof,
                "v": qkv_prof,
                "valid": ((1, 1), (1, 120), (MAX_BATCH, MAX_SEQ)),
            },
        )

    def run(self, q_raw, k_raw, v, gamma, rel_key, valid):
        seq_len = torch.tensor([q_raw.shape[1]], dtype=torch.int32, device=DEV)
        out = torch.empty_like(q_raw)
        self.runner.execute({
            "q_raw": q_raw,
            "k_raw": k_raw,
            "v": v,
            "gamma": gamma,
            "rel_key": rel_key,
            "valid": valid,
            "seq_len_carrier": seq_len,
            "out": out,
        })
        return out


def _make_inputs(batch, seq, gen, torch_dtype=None):
    """Random q/k/v [B, S, H, D], gamma [D] fp32, rel_key [P, H, D]."""
    torch_dtype = torch_dtype or torch.float16
    h, d, p = NUM_HEADS, HEAD_DIM, REL_POS_LEN

    def rand(*shape):
        return torch.randn(*shape, generator=gen,
                           dtype=torch.float32).to(torch_dtype).to(DEV)

    q, k, v = rand(batch, seq, h, d), rand(batch, seq, h,
                                           d), rand(batch, seq, h, d)
    # per_dim_scale is initialized to zeros and learns small values; sample
    # around that regime so softplus(gamma) stays O(1).
    gamma = (torch.randn(d, generator=gen, dtype=torch.float32) * 0.5).to(DEV)
    rel_key = rand(p, h, d)
    return q, k, v, gamma, rel_key


# --------------------------------------------------------------------------- #
# Single-batch prefill across chunk counts: exact / partial-final / many.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seq", [12, 64, 120], ids=lambda s: f"seq{s}")
def test_prefill(seq):
    gen = torch.Generator().manual_seed(1000 + seq)
    q, k, v, gamma, rel_key = _make_inputs(1, seq, gen)
    valid = torch.ones(1, seq, dtype=torch.bool, device=DEV)
    r = Gemma4AudioRunner(trt.float16)
    out = r.run(q, k, v, gamma, rel_key, valid)
    ref = gemma4_audio_attention_ref(q, k, v, gamma, rel_key, valid)
    assert_close(f"gemma4audio[seq{seq}]", ref, out)


# --------------------------------------------------------------------------- #
# Element-type sweep: the kernel supports fp16, bf16, and fp32.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16", "fp32"])
def test_dtype(dtype_name):
    trt_dtype = {
        "fp16": trt.float16,
        "bf16": trt.bfloat16,
        "fp32": trt.float32,
    }[dtype_name]
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype_name]
    gen = torch.Generator().manual_seed(2000)
    q, k, v, gamma, rel_key = _make_inputs(2, 60, gen, torch_dtype)
    valid = torch.ones(2, 60, dtype=torch.bool, device=DEV)
    r = Gemma4AudioRunner(trt_dtype)
    out = r.run(q, k, v, gamma, rel_key, valid)
    ref = gemma4_audio_attention_ref(q, k, v, gamma, rel_key, valid)
    # bf16's 8-bit mantissa lands the cosine at ~0.99998: the dtype's
    # precision floor, not an error.
    kwargs = {"cos_threshold": 0.9999} if dtype_name == "bf16" else {}
    assert_close(f"gemma4audio[{dtype_name}]", ref, out, **kwargs)


# --------------------------------------------------------------------------- #
# Batched prefill (all rows fully valid).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [2, 4, 8], ids=lambda b: f"bs{b}")
def test_batch(batch):
    gen = torch.Generator().manual_seed(3000 + batch)
    q, k, v, gamma, rel_key = _make_inputs(batch, 48, gen)
    valid = torch.ones(batch, 48, dtype=torch.bool, device=DEV)
    r = Gemma4AudioRunner(trt.float16)
    out = r.run(q, k, v, gamma, rel_key, valid)
    ref = gemma4_audio_attention_ref(q, k, v, gamma, rel_key, valid)
    assert_close(f"gemma4audio[bs{batch}]", ref, out)


# --------------------------------------------------------------------------- #
# Ragged valid lengths with poisoned padding; only the first lengths[b] query
# rows are compared (padding-query outputs are unspecified).
# --------------------------------------------------------------------------- #
def test_ragged_padding_poison():
    # Local-causal attention never reads right padding; poisoned holes inside
    # each row are what prove the kernel honors valid[] per key.
    lengths = [72, 50, 33, 12]
    holes = {0: [10, 40], 1: [7], 2: [20, 21], 3: [3]}
    batch, seq = len(lengths), max(lengths)
    gen = torch.Generator().manual_seed(4000)
    q, k, v, gamma, rel_key = _make_inputs(batch, seq, gen)
    valid = torch.zeros(batch, seq, dtype=torch.bool, device=DEV)
    for b, n in enumerate(lengths):
        valid[b, :n] = True
    poison_padding([q, k, v], lengths)
    for b, hs in holes.items():
        for j in hs:
            valid[b, j] = False
            for t in (k, v):  # poison the invalid keys inside the sequence
                t[b, j] = 1e3

    r = Gemma4AudioRunner(trt.float16)
    out = r.run(q, k, v, gamma, rel_key, valid)
    ref = gemma4_audio_attention_ref(q, k, v, gamma, rel_key, valid)
    keep = valid.clone()  # compare exactly the valid queries (holes excluded)
    assert_close("gemma4audio[ragged-poison]", ref[keep], out[keep])
