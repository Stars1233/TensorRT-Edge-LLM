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

import argparse
import json
import sys

from scripts import run_nemo_eval


def test_collect_metric_summary_deduplicates_aliases_and_primary_score(
        tmp_path):
    report = {
        "evaluation": {
            "runtime_seconds": 1017.32,
            "peak_memory_bytes": 55971840,
            "inference_time_seconds": 565.655,
            "scoring_time_seconds": 451.667,
        },
        "score_macro": 0.666345,
        "score": 0.664,
        "runtime_seconds": 1017.32,
        "peak_memory_bytes": 55971840,
    }
    (tmp_path / "report.json").write_text(json.dumps(report))

    metrics = run_nemo_eval._collect_metric_summary(tmp_path,
                                                    excluded_keys=("score", ))

    assert metrics == [
        ("evaluation.runtime_seconds", 1017.32),
        ("evaluation.peak_memory_bytes", 55971840.0),
        ("evaluation.inference_time_seconds", 565.655),
        ("evaluation.scoring_time_seconds", 451.667),
        ("score_macro", 0.666345),
    ]


def test_make_eval_cmd_uses_existing_server_url():
    args = argparse.Namespace(
        nemo_evaluator_bin="nemo-evaluator",
        eval_type="mmlu",
        model_id="test-model",
        engine_dir="",
        output_dir="results",
        parallelism=1,
        max_new_tokens=128,
        request_timeout=60,
        temperature=0.0,
        top_p=1.0,
        limit_samples=10,
        evaluator_overrides={},
        extra_overrides="",
    )
    model_url = "http://localhost:8000/v1/chat/completions"

    cmd = run_nemo_eval._make_eval_cmd(args, model_url)

    assert cmd[cmd.index("--model_url") + 1] == model_url
    assert cmd[cmd.index("--model_type") + 1] == "chat"


def test_parse_args_defaults_to_local_server(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_nemo_eval.py"])

    args = run_nemo_eval.parse_args()

    assert args.model_url == "http://127.0.0.1:8000/v1/chat/completions"
    assert args.engine_dir == ""
