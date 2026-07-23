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

"""Calibrate the BLASST skip-softmax deployment threshold via NVIDIA ModelOpt.

Skip-softmax (arXiv:2512.12087) skips a KV tile when exp(scale * (m_tile - m_running))
falls below a threshold lambda. A fixed lambda yields wildly different sparsity across
context lengths, so the threshold follows lambda = scale_factor / L with a
model-specific scale factor calibrated per target sparsity.

This tool is a thin wrapper around the OFFICIAL calibration implementation —
``modelopt.torch.sparsity.attention_sparsity`` (``DynamicThresholdCalibrator``, the
same machinery TensorRT-LLM's threshold_scale_factor comes from):

  1. Load the HF checkpoint with ``attn_implementation="eager"`` (the pytorch
     backend patches softmax, which only eager attention calls).
  2. ``mtsa.sparsify(model, config)`` with a calibration config: ModelOpt
     auto-generates the RULER calibration set (default 24 samples across
     descending power-of-2 length bins), runs ONE forward pass evaluating all
     20 built-in threshold trials simultaneously, and fits
     ``scale_factor = a * exp(b * sparsity)`` over every individual
     (sample, threshold) point with scipy curve_fit (sparsity filtered to
     [0.10, 0.90]).
  3. Read back (a, b) and print the deployment threshold for each requested
     target sparsity and engine max context:
     ``lambda = a * exp(b * s_target) / max_context``.

Bake the resulting lambda at AOT-compile time (--skip_softmax_threshold in the
fmha variant args of kernelSrcs/build_cutedsl.py) and validate the deployed
engine END-TO-END (task evals + TTFT A/B vs the dense build) — that is also how
the paper judges accuracy (its ~50% sparsity near-lossless safe zone).

Note on sparsity semantics: ModelOpt measures simulated block sparsity POOLED
ACROSS ALL LAYERS with a single running-max chain over causal-valid blocks;
the deployed kernel's per-layer skip ratio at the same lambda can differ
substantially (two interleaved softmax-stage max chains, first tile never
skips, and per-layer sparsity varies widely around the pooled mean). Treat the
calibrated lambda as the TRT-LLM-ecosystem-consistent starting point and let
end-to-end evals arbitrate; the kernel's verify-build tile counters report the
deployed per-layer sparsity if measurement is needed.

Two subcommands:

  calibrate (default) — the ModelOpt flow above; venv needs torch,
      transformers, nvidia-modelopt, scipy, wonderwords.
  evaluate — the paper's accuracy validation: score a DEPLOYED engine on real
      RULER samples (HF dataset simonjegou/ruler, exact-answer matching per
      task). Run it once per build (dense baseline vs each baked lambda) and
      compare; pair with the prefill benchmark (llm_bench) for TTFT.

Examples:

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
        --model-dir /path/to/Qwen3-1.7B --max-seqlen 4096 \
        --target-sparsity 0.3 0.5 --max-context 4096

    python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
        --model-dir /path/to/Qwen3-1.7B \
        --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
        --max-context 4096 --per-task 20
"""

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_RULER_HF_REPO = "simonjegou/ruler"


def _tty() -> bool:
    return sys.__stdout__ is not None and sys.__stdout__.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty() else text


def stage(cmd: str, idx: int, total: int, text: str) -> None:
    """Slim, labeled stage header: distinguishes calibrate/evaluate stages."""
    head = f"[{cmd} {idx}/{total}] {text}"
    print(f"\n{_c('1;36', head)}\n{_c('36', '-' * min(len(head), 72))}", flush=True)


def result_box(lines: list[str], color: str = "1;37") -> None:
    """Heavy box around the lines that matter."""
    width = max(len(line) for line in lines) + 2
    print()
    print(_c(color, "┏" + "━" * width + "┓"))
    for line in lines:
        print(_c(color, "┃ " + line.ljust(width - 2) + " ┃"))
    print(_c(color, "┗" + "━" * width + "┛"), flush=True)


def program_banner(title: str, subtitle: str = "") -> None:
    """Top-level separator: one per subcommand invocation."""
    bar = "━" * 72
    print(f"\n{_c('1;35', bar)}")
    print(_c("1;35", f"  {title}"))
    if subtitle:
        print(_c("35", f"  {subtitle}"))
    print(_c("1;35", bar))
    print(_c("2", "  (dim '│'-indented lines = library output; "
                  "normal lines = this tool)"), flush=True)


class _IndentDim:
    """Contain a library's stdout: reprint it dim + indented under our stage."""

    def __init__(self):
        self._buf = ""

    def write(self, text: str) -> None:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            sys.__stdout__.write(_c("2", f"  │ {line}") + "\n")

    def flush(self) -> None:
        sys.__stdout__.flush()

    def isatty(self) -> bool:      # some libraries probe the stream
        return False


def evaluate(args) -> int:
    """Score a deployed engine on real RULER samples (exact-answer matching)."""
    import time

    from datasets import load_dataset
    from transformers import AutoTokenizer

    role = args.label or ("baseline" if args.save_results and not args.baseline
                          else "candidate")
    program_banner(f"SKIP-SOFTMAX EVALUATE — {role}",
                   f"engine: {args.engine_dir}   RULER@{args.max_context}, "
                   f"{args.per_task}/task, seed {args.seed}")
    stage(f"evaluate·{role}", 1, 3, "build RULER sample set")
    t0 = time.time()
    cfgs = [4096, 8192, 16384]
    cfg = next((c for c in cfgs if c >= args.max_context), cfgs[-1])
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ds = load_dataset(_RULER_HF_REPO, str(cfg), split="test").shuffle(seed=args.seed)

    max_tok = args.max_context - 196     # headroom for the chat template
    picked: dict = defaultdict(list)
    requests, answers = [], []
    for row in ds:
        if len(picked[row["task"]]) >= args.per_task:
            continue
        prompt = row["context"] + "\n" + row["question"] + row["answer_prefix"]
        if len(tokenizer(prompt).input_ids) > max_tok:
            continue
        picked[row["task"]].append(1)
        requests.append({"messages": [{"role": "user", "content": prompt}]})
        answers.append({"task": row["task"], "answer": row["answer"]})
        if sum(len(v) for v in picked.values()) >= args.per_task * 13:
            break
    dropped = [k for k, v in picked.items() if not v]
    print(f"RULER({cfg}): {len(requests)} samples across "
          f"{sum(1 for v in picked.values() if v)} tasks"
          + (f" (all samples too long for: {dropped})" if dropped else ""))

    print(f"  sample set built in {time.time() - t0:.0f}s")

    stage(f"evaluate·{role}", 2, 3, "run engine (greedy, exact-answer scoring next)")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "ruler_input.json"
        out_path = Path(tmp) / "ruler_output.json"
        json.dump({"temperature": 1.0, "top_p": 1.0, "top_k": 1,   # greedy
                   "max_generate_length": args.max_generate_length,
                   "apply_chat_template": True, "requests": requests},
                  open(in_path, "w"))
        cmd = [str(args.llm_inference), "--engineDir", str(args.engine_dir),
               "--inputFile", str(in_path), "--outputFile", str(out_path)]
        print(_c("2", "  running: " + " ".join(cmd)))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not out_path.exists():
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            print("ERROR: inference produced no output file")
            return 1
        responses = json.load(open(out_path))["responses"]
    ok = sum(1 for r in responses if r.get("output_text"))
    print(f"  {len(responses)} responses ({ok} non-empty) in {time.time() - t0:.0f}s")

    stage(f"evaluate·{role}", 3, 3, "score (exact-answer match per task)")
    per_task, total = defaultdict(list), []
    for resp, ans in zip(sorted(responses, key=lambda r: r["request_idx"]), answers):
        raw = ans["answer"]
        if isinstance(raw, (list, tuple)):     # dataset stores answers as lists
            expected = list(raw)
        else:
            try:                               # stringified list fallback
                val = ast.literal_eval(raw)
                expected = list(val) if isinstance(val, (list, tuple)) else [val]
            except (ValueError, SyntaxError):  # plain answer string
                expected = [raw]
        text = resp.get("output_text", "")
        score = sum(1 for e in expected if str(e) in text) / max(len(expected), 1)
        per_task[ans["task"]].append(score)
        total.append(score)
    task_means = {task: sum(v) / len(v) for task, v in per_task.items()}
    overall = sum(total) / len(total)
    base = json.load(open(args.baseline)) if args.baseline else None
    print(f"\n{'task':<20}{'n':>4}{'score':>9}"
          + ("{:>10}{:>8}".format("baseline", "delta") if base else ""))
    for task in sorted(task_means):
        line = f"{task:<20}{len(per_task[task]):>4}{task_means[task]:>9.3f}"
        if base:
            base_t = base.get("per_task", {}).get(task)
            if base_t is not None:
                line += f"{base_t:>10.3f}{task_means[task] - base_t:>+8.3f}"
        print(line)
    line = f"{'OVERALL':<20}{len(total):>4}{overall:>9.4f}"
    if base:
        line += f"{base['overall']:>10.4f}{overall - base['overall']:>+8.4f}"
    print(line)

    if args.save_results:
        json.dump({"overall": overall, "per_task": task_means,
                   "n": len(total), "seed": args.seed,
                   "per_task_n": {k: len(v) for k, v in per_task.items()}},
                  open(args.save_results, "w"), indent=1)
        print(f"saved results -> {args.save_results}")

    if base:
        if base.get("seed") != args.seed or base.get("n") != len(total):
            print(f"\nWARNING: baseline sampled differently "
                  f"(seed={base.get('seed')}, n={base.get('n')}) — comparison is unpaired")
        for task in sorted(task_means):
            base_t = base.get("per_task", {}).get(task)
            if base_t is not None and base_t - task_means[task] > 0.10:
                print(f"WARNING: {task} dropped {base_t - task_means[task]:+.3f} "
                      f"({base_t:.3f} -> {task_means[task]:.3f}) — small n, but inspect")
        drop = base["overall"] - overall
        verdict = drop <= args.max_drop
        label = f" [{args.label}]" if args.label else ""
        result_box(
            [f"VERDICT: {'PASS' if verdict else 'FAIL'}{label}",
             f"overall  {base['overall']:.4f} -> {overall:.4f}   "
             f"drop {drop:+.4f}  (gate {args.max_drop})"],
            color="1;32" if verdict else "1;31")
        if verdict:
            print(f"RECOMMENDATION: this build{label} is validated for deployment. "
                  "Kernel time is threshold-insensitive at these shapes, so prefer "
                  "the SMALLEST lambda that passes this gate — a larger lambda buys "
                  "no speed and only spends accuracy margin.")
        else:
            print(f"RECOMMENDATION: do NOT deploy this build{label}. Re-bake with a "
                  "smaller lambda (lower --target-sparsity in `calibrate`) and "
                  "re-run this evaluation.")
        return 0 if verdict else 1
    print("\n(no --baseline given: scores reported without a verdict; save this "
          "run with --save-results and pass it as --baseline on the skip build)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_eval = sub.add_parser(
        "evaluate", help="Score a deployed engine on real RULER samples")
    p_eval.add_argument("--model-dir", type=Path, required=True,
                        help="HF checkpoint directory (tokenizer for length filtering)")
    p_eval.add_argument("--engine-dir", type=Path, required=True,
                        help="Deployed engine directory (llm.engine + tokenizer)")
    p_eval.add_argument("--llm-inference", type=Path, required=True,
                        help="Path to the llm_inference binary")
    p_eval.add_argument("--max-context", type=int, default=4096,
                        help="Engine max input length (default: %(default)s)")
    p_eval.add_argument("--per-task", type=int, default=20,
                        help="Samples per RULER task (default: %(default)s)")
    p_eval.add_argument("--max-generate-length", type=int, default=96,
                        help="Generation budget per sample (default: %(default)s)")
    p_eval.add_argument("--seed", type=int, default=0,
                        help="Sample-selection seed (default: %(default)s)")
    p_eval.add_argument("--save-results", type=Path, default=None,
                        help="Write scores to this json (use on the dense "
                             "baseline run)")
    p_eval.add_argument("--baseline", type=Path, default=None,
                        help="Baseline scores json to compare against; prints "
                             "a PASS/FAIL verdict and sets the exit code")
    p_eval.add_argument("--max-drop", type=float, default=0.03,
                        help="Max tolerated OVERALL accuracy drop vs the "
                             "baseline (default: %(default)s)")
    p_eval.add_argument("--label", default=None,
                        help="Human-readable label of the build under test "
                             "(e.g. 'lambda=0.115'), echoed in the verdict")

    p_cal = sub.add_parser(
        "calibrate", help="Calibrate a/b via ModelOpt (default command)")
    p_cal.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Cache dir for ModelOpt's RULER generation data (default: "
             "ModelOpt's ~/.cache/modelopt/data — point this somewhere with room)")
    parser_target = p_cal
    parser_target.add_argument(
        "--model-dir", type=Path, required=True,
        help="HF checkpoint directory")
    parser_target.add_argument(
        "--samples", type=int, default=24,
        help="RULER calibration samples (default: %(default)s, the ModelOpt "
             "default — 1 per task per length bin; increase for robustness)")
    parser_target.add_argument(
        "--max-seqlen", type=int, default=4096,
        help="Longest calibration length; RULER length bins are descending "
             "powers of 2 from here, >= 1024 (default: %(default)s)")
    parser_target.add_argument(
        "--target-sparsity", type=float, nargs="+", default=[0.5],
        help="Target sparsity(ies) to print deployment thresholds for "
             "(default: %(default)s, the paper's near-lossless safe-zone bound)")
    parser_target.add_argument(
        "--max-context", type=int, nargs="+", default=[4096],
        help="Engine max context length(s) for the deployment lambda")
    parser_target.add_argument(
        "--module-pattern", default="*self_attn*",
        help="Wildcard matching the attention modules to calibrate "
             "(default: %(default)s)")
    parser_target.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16"],
        help="Model dtype for the calibration forwards (default: %(default)s)")
    parser_target.add_argument(
        "--gpu", default="0", help="GPU id (default: %(default)s)")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("calibrate", "evaluate", "-h", "--help"):
        argv = ["calibrate"] + argv          # default subcommand
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return evaluate(args)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    from transformers import AutoModelForCausalLM

    import modelopt.torch.sparsity.attention_sparsity as mtsa

    program_banner(f"SKIP-SOFTMAX CALIBRATE — {Path(args.model_dir).name}",
                   f"ModelOpt official, RULER samples={args.samples}, "
                   f"max_seqlen={args.max_seqlen}, targets={args.target_sparsity}")
    stage("calibrate", 1, 3, f"load model (eager attention, {args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=getattr(torch, args.dtype),
        device_map="cuda",
        attn_implementation="eager",       # required: pytorch backend patches softmax
    )
    model.eval()

    config = {
        "sparse_cfg": {
            args.module_pattern: {
                "method": "flash_skip_softmax",
                "backend": "pytorch",
                "enable": True,
            },
            # NOTE: "calibration" is a top-level sparse_cfg key (sibling of the
            # module wildcards) — _extract_calibration_config reads it there.
            "calibration": {
                "target_sparse_ratio": {"prefill": args.target_sparsity[0],
                                        "decode": 0.0},
                "samples": args.samples,
                "max_seqlen": args.max_seqlen,
                **({"cache_dir": str(args.cache_dir)} if args.cache_dir else {}),
            },
            "default": {"enable": False},
        },
    }

    stage("calibrate", 2, 3, "ModelOpt calibration "
          "(20 threshold trials, one forward pass — library output below)")
    import contextlib

    from modelopt.torch.sparsity.attention_sparsity.calibration.calibrate import (
        calibrate_sparse_attention,
    )

    print(_c("2", "  ┌─ modelopt library output " + "─" * 40))
    with contextlib.redirect_stdout(_IndentDim()):     # dim+indent modelopt prints
        # sparsify = convert + calibrate, but it discards the calibration
        # results dict (R^2, observed sparsity range, ...). Run the two steps
        # explicitly to keep it.
        model = mtsa.sparsify(model, {"sparse_cfg": {
            k: v for k, v in config["sparse_cfg"].items() if k != "calibration"}})
        results = calibrate_sparse_attention(model, config)
    print(_c("2", "  └─ end modelopt output " + "─" * 43))

    params = results.get("calibration_results", {}).get("prefill")
    if not params:
        print("ERROR: calibration produced no parameters — see modelopt output above")
        return 1

    a, b = float(params["a"]), float(params["b"])
    r2 = params.get("r_squared")
    n_pts = params.get("num_data_points")
    s_lo = params.get("min_observed_sparsity")
    s_hi = params.get("max_observed_sparsity")
    stage("calibrate", 3, 3, "fitted parameters and deployment thresholds")
    lines = ["fit:  scale_factor = a * exp(b * sparsity)",
             f"      a = {a:.6g}   b = {b:.6g}"
             + (f"   R^2 = {float(r2):.3f}" if r2 is not None else "")
             + (f"   ({n_pts} points)" if n_pts else ""),
             ]
    if s_lo is not None and s_hi is not None:
        lines.append(f"      observed sparsity range: [{s_lo:.1%}, {s_hi:.1%}]"
                     "  (targets beyond it are extrapolated)")
    lines.append("")
    for s_target in args.target_sparsity:
        sf = a * math.exp(b * s_target)
        extra = ("  <- EXTRAPOLATED" if s_hi is not None and s_target > s_hi
                 else "")
        for max_ctx in args.max_context:
            lam = sf / max_ctx
            lines.append(f"target {s_target:>4.0%}  max_ctx {max_ctx:<6}  "
                         f"lambda = {lam:<12.6g} (log2 {math.log2(lam):+.2f})"
                         + extra)
    result_box(lines)
    print(_c("1;33",
             "NEXT: bake the chosen lambda (--skip_softmax_threshold in "
             "kernelSrcs/build_cutedsl.py),\n      rebuild + relink, then run "
             "`evaluate --baseline <dense scores>` — deploy the\n      SMALLEST "
             "lambda that PASSES (kernel time is threshold-insensitive)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
