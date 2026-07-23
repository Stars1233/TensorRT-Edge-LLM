#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""Compare the PyTorch golden and EdgeLLM 4-layer dumps.

Loads two safetensors files:
- golden  : produced by ``golden_layer_dump.py`` (left padding).
- edgellm : produced by the C++ runtime when ``EDGELLM_DUMP_LOGITS_KVCACHE_*`` is set
            (right-aligned).

Compares logits and KV cache per round (round 0 = prefill, 1..N = decode), per sequence and
per layer, and checks that both sides' greedy token choices agree each round.

== Format differences and how they are reconciled ==
- **logits**: golden ``round_{r}.logits`` is ``[B, 1, vocab]``; edgellm is ``[B, vocab]``. Both are
  the last real token's logits, so they compare directly.
- **KV layout**: golden stores separate ``round_{r}.layer_{i}.key`` / ``.value`` of shape
  ``[B, kv_heads, seq, head_dim]``; edgellm stores a combined ``round_{r}.layer_{i}.kv`` of shape
  ``[B, 2, kv_heads, maxSeqLen, head_dim]`` (dim 1: 0=key, 1=value), dumped full-length (the engine
  does no truncation). The combined tensor is split, and each side is sliced to ``valid`` below.
- **padding side**: golden is left-padded (real tokens occupy each row's right segment); edgellm is
  right-aligned (real tokens occupy ``[0, valid)``). So within the valid length ``valid`` for
  sequence b at round r:
    golden takes the last ``valid`` columns: ``g[..., -valid:, ...]``
    edgellm takes the first ``valid`` columns: ``e[..., :valid, ...]``
  where ``valid = edgellm round_{r}.context_lengths[b]`` (= prefill real length + r).
- **dtype**: both sides are upcast to float32 before comparing (golden KV/logits are usually fp16,
  edgellm logits are fp32).

== Input-alignment check ==
At round 0, verify that the golden's real prefill lengths (per-row attention_mask sums) equal the
edgellm round_0.context_lengths. If they differ, the two sides consumed different input_ids (most
commonly: one applied a chat template while the other used raw tokenization), the numeric comparison
would be meaningless, and the tool reports this structural error immediately.

== Exit code ==
0 = everything passes; 1 = any round/tensor fails or the structure is misaligned.
"""

from __future__ import annotations

import argparse
import re
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

_ROUND_RE = re.compile(r"^round_(\d+)\.")


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load an entire safetensors file into a {name: tensor} dict."""
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _infer_rounds(d: dict[str, torch.Tensor]) -> list[int]:
    """Infer the available rounds from the keys (looking at round_{r}.logits)."""
    rs = set()
    for k in d:
        m = _ROUND_RE.match(k)
        if m and k.endswith(".logits"):
            rs.add(int(m.group(1)))
    return sorted(rs)


def _infer_layers(d: dict[str, torch.Tensor], round_idx: int) -> list[int]:
    """Infer a round's layers from the keys.

    Attention layers: golden ``.key``/``.value``, edgellm ``.kv``. Recurrent layers (Mamba /
    Gated DeltaNet): both sides ``.recurrent_state`` (+ ``.conv_state``).
    """
    ls = set()
    for k in d:
        m = re.match(
            rf"^round_{round_idx}\.layer_(\d+)\.(key|kv|recurrent_state)$", k)
        if m:
            ls.add(int(m.group(1)))
    return sorted(ls)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Whole-tensor cosine similarity (both operands flattened to one vector).

    Delegates to ``torch.nn.functional.cosine_similarity`` for the usual case.
    The explicit zero-norm guard is kept on purpose: ``F.cosine_similarity``
    clamps the denominator with ``eps`` and would return a near-zero value for
    an all-zero tensor, but here an all-zero side is a meaningful degenerate
    case (e.g. a near-zero gate activation), so we report an exact ``1.0`` when
    both sides are identically zero and ``0.0`` when only one side is zero.
    """
    a = a.flatten()
    b = b.flatten()
    if a.norm() == 0 or b.norm() == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(F.cosine_similarity(a, b, dim=0))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


class Stat:
    """Result of comparing a single tensor."""

    def __init__(self, name: str, cos: float, max_abs: float, allclose: bool):
        self.name = name
        self.cos = cos
        self.max_abs = max_abs
        self.allclose = allclose


def _compare_tensor(name: str, golden: torch.Tensor, edgellm: torch.Tensor,
                    atol: float, rtol: float) -> Stat:
    golden = golden.to(torch.float32)
    edgellm = edgellm.to(torch.float32)
    if golden.shape != edgellm.shape:
        # A shape mismatch is recorded as the worst possible result (usually means the reconcile
        # slicing is wrong or the lengths don't line up).
        return Stat(
            name +
            f" SHAPE-MISMATCH golden{tuple(golden.shape)} edgellm{tuple(edgellm.shape)}",
            0.0, float("inf"), False)
    return Stat(name, cosine_similarity(golden, edgellm),
                _max_abs(golden, edgellm),
                bool(torch.allclose(golden, edgellm, atol=atol, rtol=rtol)))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden",
                   required=True,
                   help="golden safetensors (PyTorch side)")
    p.add_argument("--edgellm",
                   required=True,
                   help="edgellm safetensors (C++ runtime side)")
    p.add_argument("--cos",
                   type=float,
                   default=0.99,
                   help="minimum cosine similarity (default 0.99)")
    p.add_argument("--atol",
                   type=float,
                   default=2e-2,
                   help="atol for allclose (default 2e-2)")
    p.add_argument("--rtol",
                   type=float,
                   default=2e-2,
                   help="rtol for allclose (default 2e-2)")
    p.add_argument("--verbose",
                   action="store_true",
                   help="print every tensor's comparison result")
    p.add_argument(
        "--teacher-forced",
        action="store_true",
        help=
        "the run was teacher-forced (EdgeLLM was fed the golden's tokens), so treat per-round "
        "token disagreement as informational rather than a failure (cosine is the gate)"
    )
    args = p.parse_args()

    golden = _load_safetensors(args.golden)
    edgellm = _load_safetensors(args.edgellm)

    golden_rounds = _infer_rounds(golden)
    edgellm_rounds = _infer_rounds(edgellm)
    common = [r for r in golden_rounds if r in edgellm_rounds]
    print(
        f"[compare] golden rounds={golden_rounds} edgellm rounds={edgellm_rounds} "
        f"-> comparing {common}")
    if golden_rounds != edgellm_rounds:
        print(
            f"[compare] WARNING: round sets differ; only comparing the intersection."
        )
    if not common:
        print("[compare] FAIL: no common rounds to compare.")
        return 1

    bs = int(golden["input_ids"].shape[0])

    # ---- Input-alignment check (using the round-0 lengths) ----
    # Golden real prefill length = per-row attention_mask sum; edgellm = round_0.context_lengths.
    golden_real_len = golden["attention_mask"].sum(-1).to(torch.int64)  # [B]
    edgellm_ctx0 = edgellm["round_0.context_lengths"].to(torch.int64)  # [B]
    if not torch.equal(golden_real_len, edgellm_ctx0):
        print(
            "[compare] FAIL: input misalignment — golden real prefill lengths "
            f"{golden_real_len.tolist()} != edgellm round_0 context_lengths "
            f"{edgellm_ctx0.tolist()}.")
        print(
            "           The two sides consumed different input_ids (commonly: one "
            "applied a chat template, the other used raw tokenization). Skipping the "
            "numeric comparison.")
        return 1
    print(
        f"[compare] input alignment OK: real prefill lengths {golden_real_len.tolist()}"
    )

    overall_ok = True
    worst = None  # (cos, name)

    for r in common:
        ctx = edgellm[f"round_{r}.context_lengths"].to(torch.int64)  # [B]
        layers = _infer_layers(edgellm, r)
        round_stats: list[Stat] = []

        # Token agreement: golden generated_ids[:, r] vs edgellm round_r.generated_token_ids.
        # Under teacher forcing the runtime is fed the golden's tokens, so a mismatch here only
        # means the runtime's own greedy would have differed (a near-tie) — informational, not a
        # failure; cosine is the gate.
        tok_note = ""
        token_mismatch = False
        edgellm_tok_key = f"round_{r}.generated_token_ids"
        if edgellm_tok_key in edgellm and r < golden["generated_ids"].shape[1]:
            golden_tok = golden["generated_ids"][:, r].to(torch.int64)
            edgellm_tok = edgellm[edgellm_tok_key].to(torch.int64)
            agree = int((golden_tok == edgellm_tok).sum())
            tok_note = f"tokens {agree}/{bs}"
            if agree != bs:
                token_mismatch = True
                tok_note += f"  golden={golden_tok.tolist()} edgellm={edgellm_tok.tolist()}"
                if args.teacher_forced:
                    tok_note += "  (teacher-forced: runtime's own greedy differs — informational)"
                elif r == 0:
                    tok_note += "  (round-0 mismatch — check that the inputs really align)"

        for b in range(bs):
            valid = int(ctx[b])
            # ---- logits ----
            golden_logits = golden[f"round_{r}.logits"][
                b, -1, :]  # [vocab] (last position of [B,1,vocab])
            edgellm_logits = edgellm[f"round_{r}.logits"][b, :]  # [vocab]
            round_stats.append(
                _compare_tensor(f"r{r}.b{b}.logits", golden_logits,
                                edgellm_logits, args.atol, args.rtol))
            # ---- per-layer state: attention KV, or recurrent (Mamba / Gated DeltaNet) ----
            for i in layers:
                rec_key = f"round_{r}.layer_{i}.recurrent_state"
                if rec_key in golden and rec_key in edgellm:
                    # Recurrent layer: fixed-size state, no sequence dim -> compare row b directly
                    # (no left/right reconcile needed). conv_state likewise when present.
                    round_stats.append(
                        _compare_tensor(f"r{r}.b{b}.layer_{i}.recurrent_state",
                                        golden[rec_key][b],
                                        edgellm[rec_key][b], args.atol,
                                        args.rtol))
                    conv_key = f"round_{r}.layer_{i}.conv_state"
                    if conv_key in golden and conv_key in edgellm:
                        round_stats.append(
                            _compare_tensor(f"r{r}.b{b}.layer_{i}.conv_state",
                                            golden[conv_key][b],
                                            edgellm[conv_key][b], args.atol,
                                            args.rtol))
                    continue
                # Attention layer.
                # golden: separate key/value, [B, kv_heads, seq, head_dim], take the last valid cols.
                golden_key = golden[f"round_{r}.layer_{i}.key"][b, :,
                                                                -valid:, :]
                golden_value = golden[f"round_{r}.layer_{i}.value"][b, :,
                                                                    -valid:, :]
                # edgellm: combined [B, 2, kv_heads, L, head_dim], split 0/1 and take the first valid cols.
                edgellm_kv = edgellm[f"round_{r}.layer_{i}.kv"]
                edgellm_key = edgellm_kv[b, 0, :, :valid, :]
                edgellm_value = edgellm_kv[b, 1, :, :valid, :]
                round_stats.append(
                    _compare_tensor(f"r{r}.b{b}.layer_{i}.key", golden_key,
                                    edgellm_key, args.atol, args.rtol))
                round_stats.append(
                    _compare_tensor(f"r{r}.b{b}.layer_{i}.value", golden_value,
                                    edgellm_value, args.atol, args.rtol))

        # Round summary
        min_cos_stat = min(round_stats, key=lambda s: s.cos)
        max_abs_stat = max(round_stats, key=lambda s: s.max_abs)
        all_close = all(s.allclose for s in round_stats)
        cos_ok = min_cos_stat.cos >= args.cos
        # A token mismatch fails the round only when NOT teacher-forced; otherwise cosine is the gate.
        token_fail = token_mismatch and not args.teacher_forced
        round_ok = cos_ok and not token_fail
        if not round_ok:
            overall_ok = False
        if worst is None or min_cos_stat.cos < worst[0]:
            worst = (min_cos_stat.cos, min_cos_stat.name)

        tag = "prefill" if r == 0 else f"decode {r}"
        print(f"  round {r} [{tag}]: {tok_note or 'tokens n/a':<24s} "
              f"min_cos={min_cos_stat.cos:.5f} ({min_cos_stat.name}) "
              f"max_abs={max_abs_stat.max_abs:.4g} "
              f"allclose={'Y' if all_close else 'N'} "
              f"-> {'PASS' if (cos_ok and round_ok) else 'FAIL'}")
        if args.verbose:
            for s in sorted(round_stats, key=lambda s: s.name):
                print(
                    f"      {s.name:30s} cos={s.cos:.5f} max_abs={s.max_abs:.4g} "
                    f"allclose={'Y' if s.allclose else 'N'}")

    print(f"[compare] {'PASS' if overall_ok else 'FAIL'}: "
          f"worst cosine {worst[0]:.5f} @ {worst[1]} "
          f"(threshold {args.cos}, atol {args.atol}, rtol {args.rtol})")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
