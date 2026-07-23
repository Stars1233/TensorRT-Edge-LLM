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
causal_conv1d plugin tests vs a PyTorch reference (causal_conv1d_ref).

Depthwise causal conv1d as used by the Nemotron-H mamba mixer (conv_kernel=4).
Centerpiece is ragged prefill (variable per-row lengths in one padded call,
as Nemotron hits during batched MMLU prefill): verifies per-row valid outputs
and the captured conv-state, and poisons the padding so any read past
context_lengths is caught. Decode is tested with a non-zero conv-state to
exercise state carry-over.

Run:
    python3 -m pytest tests/python-unittests/test_causal_conv1d_plugin.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from test_plugin_base import (DEPENDENCIES_AVAILABLE, IMPORT_ERROR,
                              RAGGED_CASES, PluginRunner, assert_close,
                              pf_int32, poison_padding)

if DEPENDENCIES_AVAILABLE:
    import tensorrt as trt
    import torch

pytestmark = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE,
    reason=f"TensorRT/torch CUDA not available: {IMPORT_ERROR}")

DEV = "cuda"


def causal_conv1d_ref(
        x: torch.Tensor,  # [b, s, c]
        weight: torch.Tensor,  # [c, width]  (depthwise; width == conv kernel)
        bias: Optional[torch.Tensor],  # [c]
        conv_state0: torch.Tensor,  # [b, c, width]
        activation: bool,
        context_lengths: Optional[torch.Tensor] = None,  # [b]
):
    """Depthwise causal conv1d reference. Returns (y, final_conv_state).

    Matches the plugin's conventions (verified empirically): the conv-state
    buffer is ``width`` (== kernel) columns holding the most recent ``width``
    inputs (oldest..newest). Output token t uses the last ``width`` inputs
    ending at t; the relevant history for the first new token is the buffer's
    last ``width-1`` columns ``conv_state0[:, :, 1:]`` (zeros on a fresh state,
    giving causal left zero-pad). The captured final state is the last
    ``width`` inputs. For ragged rows only the first ``context_lengths[bi]``
    tokens are valid.
    """
    b, s, c = x.shape
    width = weight.shape[-1]
    xf = x.float().transpose(1, 2)  # [b, c, s]
    wf = weight.float()  # [c, width]
    state0 = conv_state0.float()  # [b, c, width]

    if context_lengths is None:
        context_lengths = torch.full((b, ), s, dtype=torch.int64)
    else:
        context_lengths = context_lengths.to(torch.int64)

    y = torch.zeros((b, c, s), dtype=torch.float32, device=x.device)
    final_state = state0.clone()
    for bi in range(b):
        L = int(context_lengths[bi])
        # Usable history = buffer's last (width-1) columns, then valid inputs.
        seq = torch.cat([state0[bi, :, 1:width], xf[bi, :, :L]], dim=-1)
        for t in range(L):
            window = seq[:, t:t + width]  # [c, width]
            out = (window * wf).sum(-1)
            if bias is not None:
                out = out + bias.float()
            y[bi, :, t] = out
        # Captured state = last `width` inputs (left zero-pad if seq shorter).
        if seq.shape[-1] >= width:
            final_state[bi] = seq[:, -width:]
        else:
            final_state[bi] = torch.nn.functional.pad(
                seq, (width - seq.shape[-1], 0))
    if activation:
        y = torch.nn.functional.silu(y)
    return y.transpose(1, 2), final_state  # y: [b, s, c]


@dataclass
class ConvConfig:
    dim: int = 256  # conv channels (depthwise)
    width: int = 4  # conv kernel (Nemotron conv_kernel=4)
    max_batch: int = 4
    max_seq: int = 64


class ConvRunner:
    """Builds + runs causal_conv1d for prefill (seq>1) or decode (seq=1)."""

    def __init__(self, cfg: ConvConfig):
        self.cfg = cfg
        self.runner = PluginRunner()
        self._build()

    def _build(self):
        c = self.cfg
        dim, w, mb, ms = c.dim, c.width, c.max_batch, c.max_seq
        F16, I32 = trt.float16, trt.int32
        input_specs = [
            ("x", F16, (-1, -1, dim)),
            ("weight", F16, (dim, 1, w)),
            ("bias", F16, (dim, )),
            ("conv_state", F16, (-1, dim, w)),
            ("context_lengths", I32, (-1, )),
        ]
        profiles = {
            "x": ((1, 1, dim), (1, 16, dim), (mb, ms, dim)),
            "weight": ((dim, 1, w), (dim, 1, w), (dim, 1, w)),
            "bias": ((dim, ), (dim, ), (dim, )),
            "conv_state": ((1, dim, w), (1, dim, w), (mb, dim, w)),
            "context_lengths": ((1, ), (1, ), (mb, )),
        }
        self.runner.build(
            input_specs=input_specs,
            output_names=["output", "conv_state_out"],
            plugin_name="causal_conv1d",
            plugin_version="1",
            plugin_fields=[
                pf_int32("stride", 1),
                pf_int32("padding", w - 1),  # causal left pad
                pf_int32("dilation", 1),
                pf_int32("groups", 0),  # 0 -> depthwise (groups == dim)
                pf_int32("use_mtp", 0),
            ],
            profiles=profiles,
        )

    def run(self, x, weight, bias, conv_state, context_lengths):
        out = torch.empty_like(x)
        state_out = torch.empty_like(conv_state)
        self.runner.execute({
            "x": x,
            "weight": weight,
            "bias": bias,
            "conv_state": conv_state,
            "context_lengths": context_lengths,
            "output": out,
            "conv_state_out": state_out,
        })
        return out, state_out


def _rand(cfg, b, s, gen):
    """x is always 3D [b, s, dim] (the plugin treats x as [batch, seq, dim])."""
    dim, w = cfg.dim, cfg.width

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(DEV)

    x = rn(b, s, dim).to(torch.float16)
    weight = (rn(dim, 1, w) * 0.3).to(torch.float16)
    bias = (rn(dim) * 0.1).to(torch.float16)
    return x, weight, bias


# --------------------------------------------------------------------------- #
# Prefill: causal conv output + captured conv-state (fresh zero state)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2, 4], ids=lambda b: f"bs{b}")
def test_prefill(batch):
    cfg = ConvConfig()
    seq = 16
    gen = torch.Generator().manual_seed(10 + batch)
    r = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, seq, gen)
    state0 = torch.zeros(batch,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    ctx = torch.full((batch, ), seq, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, weight, bias, state0.clone(), ctx)
    w2 = weight[:, 0, :]  # [dim, width]
    ref_y, ref_state = causal_conv1d_ref(x, w2, bias, state0, False, ctx)
    for bi in range(batch):
        assert_close(f"y[b{bi}]", ref_y[bi], out[bi], 2e-2, 2e-2)
    assert_close("conv_state_out", ref_state, state_out, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# RAGGED prefill: variable per-row lengths + poisoned padding
# --------------------------------------------------------------------------- #
def test_ragged_prefill():
    cfg = ConvConfig()
    seq = 32
    batch = 3
    gen = torch.Generator().manual_seed(202)
    r = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, seq, gen)
    ctx = torch.tensor([32, 17, 5], dtype=torch.int32, device=DEV)
    poison_padding(x, ctx)
    state0 = torch.zeros(batch,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    out, state_out = r.run(x, weight, bias, state0.clone(), ctx)
    w2 = weight[:, 0, :]
    ref_y, ref_state = causal_conv1d_ref(x, w2, bias, state0, False, ctx)
    for bi in range(batch):
        L = int(ctx[bi])
        assert_close(f"y[b{bi}]", ref_y[bi, :L], out[bi, :L], 2e-2, 2e-2)
    assert_close("conv_state_out", ref_state, state_out, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# Decode (seq=1): state carry-over from a non-zero conv-state.
# The conv-state buffer holds the last `width` inputs; decode rolls left and
# appends the new input.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 2], ids=lambda b: f"bs{b}")
def test_decode(batch):
    cfg = ConvConfig()
    gen = torch.Generator().manual_seed(30 + batch)
    r = ConvRunner(cfg)
    x, weight, bias = _rand(cfg, batch, 1, gen)  # x is [b, 1, dim]
    state0 = (torch.randn(
        batch, cfg.dim, cfg.width, generator=gen, dtype=torch.float32) *
              0.3).to(torch.float16).to(DEV)
    ctx = torch.ones(batch, dtype=torch.int32, device=DEV)
    out, state_out = r.run(x, weight, bias, state0.clone(), ctx)
    w2 = weight[:, 0, :]
    ref_y, ref_state = causal_conv1d_ref(x, w2, bias, state0, False, ctx)
    assert_close("y", ref_y, out, 2e-2, 2e-2)
    assert_close("conv_state_out", ref_state, state_out, 2e-2, 2e-2)


# --------------------------------------------------------------------------- #
# Required even/uneven batch cases (bs 1/2/3/4/8, seq up to 2048).
# --------------------------------------------------------------------------- #
def _ragged_cfg():
    return ConvConfig(dim=256, width=4, max_batch=8, max_seq=2048)


@pytest.mark.parametrize("label,seqlens", RAGGED_CASES)
def test_ragged_prefill_batch_sizes(label, seqlens):
    cfg = _ragged_cfg()
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(1100 + maxlen + bs)
    x, weight, bias = _rand(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding(x, ctx)
    state0 = torch.zeros(bs,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    r = ConvRunner(cfg)
    out, state_out = r.run(x, weight, bias, state0.clone(), ctx)
    ref_y, ref_state = causal_conv1d_ref(x, weight[:, 0, :], bias, state0,
                                         False, ctx)
    for bi in range(bs):
        L = seqlens[bi]
        assert_close(f"conv[{label}].y[{bi}]", ref_y[bi, :L], out[bi, :L])
    assert_close(f"conv[{label}].state", ref_state, state_out)


# --------------------------------------------------------------------------- #
# Batch invariance (plugin-vs-plugin): permuting batch rows permutes outputs.
# --------------------------------------------------------------------------- #
def test_batch_invariance():
    cfg = _ragged_cfg()
    seqlens = [10, 2048, 128]
    bs, maxlen = len(seqlens), max(seqlens)
    gen = torch.Generator().manual_seed(1357)
    x, weight, bias = _rand(cfg, bs, maxlen, gen)
    ctx = torch.tensor(seqlens, dtype=torch.int32, device=DEV)
    poison_padding(x, ctx)
    state0 = torch.zeros(bs,
                         cfg.dim,
                         cfg.width,
                         dtype=torch.float16,
                         device=DEV)
    r = ConvRunner(cfg)
    out0, st0 = r.run(x, weight, bias, state0.clone(), ctx)
    perm = torch.tensor([2, 0, 1], device=DEV)
    out1, st1 = r.run(x[perm].contiguous(), weight, bias, state0.clone(),
                      ctx[perm].contiguous())
    pcpu = perm.cpu()
    for new_i in range(bs):
        orig = int(pcpu[new_i])
        L = seqlens[orig]
        assert_close(f"conv-batch-inv.y[{new_i}]", out0[orig, :L],
                     out1[new_i, :L])
    assert_close("conv-batch-inv.state", st0[pcpu], st1)
