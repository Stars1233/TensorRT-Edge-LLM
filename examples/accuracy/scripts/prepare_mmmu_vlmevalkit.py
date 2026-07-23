#!/usr/bin/env python3
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
Prepare MMMU output xlsx file in VLMEvalKit format from JSON output.

This script combines:
1. MMMU_DEV_VAL.tsv - VLMEvalKit's MMMU dataset metadata (questions, options, answers, etc.)
2. JSON output file - TensorRT Edge LLM inference predictions

To produce an xlsx file compatible with VLMEvalKit evaluation.

Download MMMU_DEV_VAL.tsv from:
    https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv

Usage:
    python prepare_mmmu_vlmevalkit.py \
        --tsv_file /path/to/MMMU_DEV_VAL.tsv \
        --json_file /path/to/outputs/mmmu_predictions.json \
        --output_file /path/to/outputs/Model_MMMU_DEV_VAL.xlsx
"""

import argparse
import json
import os
import re

import pandas as pd


def extract_answer_letter(text: str) -> str:
    """
    Extract the final answer letter (A-J) from a multiple-choice model
    prediction.

    Tries patterns in priority order:
    1. Explicit answer markers (all use A-J): "Answer: A", "**B**", "is C", "(D)", etc.
    2. Fallback: last standalone uppercase letter B-H or J (excludes A and I which
       are common English words; explicit patterns above still match "Answer: A").
    Falls back to the original text if nothing is found.
    """
    if not text:
        return text

    # Patterns searched from the END of the text first (most reliable)
    explicit_patterns = [
        r'[Aa]nswer[:\s*]+\*{0,2}([A-J])\b',  # Answer: B  /  Answer: **B**
        r'\*{1,2}([A-J])\*{1,2}[\s.]*$',  # **B** or *B* at end
        r'\b([A-J])\.\s*$',  # "B." at end of line
        r'\b(?:option|choice|select)[:\s]+([A-J])\b',  # option/choice B
        r'\b(?:is|are|be)[:\s]+\(?([A-J])\)?',  # is B / is (B)
        r'\(([A-J])\)',  # (B)
    ]

    for pat in explicit_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Fallback: last standalone B-J in the text (exclude A/I - common English words)
    standalone = re.findall(r'\b([B-HJ])\b', text)
    if standalone:
        return standalone[-1].upper()

    return text


def extract_prediction(text: str, question_type) -> str:
    """Route prediction processing by MMMU ``question_type``.

    - ``multiple-choice``: extract single answer letter A-J via
      ``extract_answer_letter``.
    - ``open`` (free-form / numerical answers): return the raw prediction
      so VLMEvalKit's open-ended evaluator can score it directly. Running
      letter extraction on open answers would clobber phrases like
      "the answer is 5, hence pattern B" by collapsing them to ``"B"``.
    - Missing / unknown: fall back to MCQ extraction (existing behavior).
    """
    if not text:
        return text
    qt = (question_type or "").strip().lower() if isinstance(
        question_type, str) else ""
    if qt.startswith("open"):
        return text.strip()
    return extract_answer_letter(text)


def load_json_predictions(json_file: str) -> list:
    """Load predictions from JSON output file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Handle both formats: list of responses or dict with 'responses' key
    if isinstance(data, dict) and 'responses' in data:
        responses = data['responses']
    elif isinstance(data, list):
        responses = data
    else:
        raise ValueError(
            f"Unsupported JSON format. Expected dict with 'responses' key or list."
        )

    # Sort by request_idx to ensure correct order
    responses_sorted = sorted(responses, key=lambda x: x.get('request_idx', 0))

    # Extract output_text as predictions
    predictions = [r.get('output_text', '') for r in responses_sorted]

    return predictions


def prepare_mmmu_vlmevalkit_output(
    tsv_file: str,
    json_file: str,
    output_file: str,
) -> None:
    """
    Prepare MMMU output xlsx in VLMEvalKit format.
    
    Args:
        tsv_file: Path to MMMU_DEV_VAL.tsv template file
        json_file: Path to JSON output file with predictions
        output_file: Path to output xlsx file
    """
    # Load TSV template
    print(f"Loading TSV template from: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    print(f"  Total rows: {len(df)}")

    # Filter by validation split
    df_filtered = df[df['split'] == 'validation'].copy()
    print(f"  Rows after filtering by split='validation': {len(df_filtered)}")

    # Remove 'image' column if it exists (VLMEvalKit format doesn't include it)
    if 'image' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['image'])
        print("  Removed 'image' column")

    # Load predictions from JSON
    print(f"\nLoading predictions from: {json_file}")
    predictions = load_json_predictions(json_file)
    print(f"  Number of predictions: {len(predictions)}")

    # Verify counts match
    if len(predictions) != len(df_filtered):
        raise ValueError(
            f"Mismatch: {len(predictions)} predictions vs {len(df_filtered)} dataset rows. "
            "Ensure the JSON output corresponds to the same dataset split.")

    # Route each prediction by MMMU question_type: multiple-choice runs
    # letter extraction; open-ended passes the raw text through (otherwise
    # the regex fallback can hijack a stray letter inside a free-form answer).
    if 'question_type' in df_filtered.columns:
        qtypes = df_filtered['question_type'].tolist()
        n_open = sum(
            1 for q in qtypes
            if isinstance(q, str) and q.strip().lower().startswith("open"))
        print(
            f"  question_type column found: {len(qtypes) - n_open} multiple-choice, {n_open} open"
        )
    else:
        qtypes = [None] * len(df_filtered)
        print(
            "  question_type column missing — falling back to multiple-choice extraction for all rows"
        )

    extracted = [extract_prediction(p, q) for p, q in zip(predictions, qtypes)]
    n_extracted = sum(1 for o, e in zip(predictions, extracted) if o != e)
    print(
        f"  Extracted/normalized answer from {n_extracted}/{len(predictions)} predictions"
    )

    # Add predictions column
    df_filtered['prediction'] = extracted
    print("  Added 'prediction' column")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to xlsx
    print(f"\nSaving output to: {output_file}")
    df_filtered.to_excel(output_file, index=False, engine='openpyxl')
    print(
        f"  Saved {len(df_filtered)} rows with {len(df_filtered.columns)} columns"
    )
    print(f"  Columns: {df_filtered.columns.tolist()}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Prepare MMMU output xlsx in VLMEvalKit format from JSON output")
    parser.add_argument(
        '--tsv_file',
        type=str,
        required=True,
        help='Path to MMMU_DEV_VAL.tsv file. Download from: '
        'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv')
    parser.add_argument('--json_file',
                        type=str,
                        required=True,
                        help='Path to JSON output file with predictions')
    parser.add_argument('--output_file',
                        type=str,
                        required=True,
                        help='Path to output xlsx file')

    args = parser.parse_args()

    prepare_mmmu_vlmevalkit_output(
        tsv_file=args.tsv_file,
        json_file=args.json_file,
        output_file=args.output_file,
    )


if __name__ == '__main__':
    main()
