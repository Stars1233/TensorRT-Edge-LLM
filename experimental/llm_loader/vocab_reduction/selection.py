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
"""Reference vocabulary-map generation for reduced-vocabulary export."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from typing import Any, Optional, Set

import torch
from safetensors.torch import load_file, save_file

from .constants import VOCAB_INFO_NAME, VOCAB_MAP_NAME


def get_vocab_size(config: Any) -> int:
    """Extract vocabulary size from a text or multimodal HF config."""
    if hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    if hasattr(config, "text_config") and hasattr(config.text_config,
                                                  "vocab_size"):
        return int(config.text_config.vocab_size)
    raise AttributeError(
        f"Could not find vocab_size in {type(config).__name__}. Expected "
        "config.vocab_size or config.text_config.vocab_size.")


def extract_d2t_required_tokens(d2t_tensor: torch.Tensor,
                                vocab_size: int) -> Set[int]:
    """Return base-model token IDs referenced by an EAGLE d2t tensor."""
    required_tokens = set()
    print(f"Processing d2t tensor with {len(d2t_tensor)} entries...")

    for reduced_token_id in range(len(d2t_tensor)):
        offset = int(d2t_tensor[reduced_token_id].item())
        base_token_id = reduced_token_id + offset
        if 0 <= base_token_id < vocab_size:
            required_tokens.add(base_token_id)

    print(f"Extracted {len(required_tokens)} required tokens from d2t mapping")
    return required_tokens


def get_special_tokens(tokenizer: Any) -> Set[int]:
    """Return special token IDs that must stay available at runtime."""
    special_tokens = set()

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
        if eos_token_id is None:
            raise ValueError(
                "Tokenizer must have eos_token_id or pad_token_id")
    special_tokens.add(int(eos_token_id))

    if tokenizer.bos_token_id is not None:
        special_tokens.add(int(tokenizer.bos_token_id))
    if tokenizer.pad_token_id is not None:
        special_tokens.add(int(tokenizer.pad_token_id))
    if tokenizer.unk_token_id is not None:
        special_tokens.add(int(tokenizer.unk_token_id))

    return special_tokens


def input_frequency_filter(dataset: Any, tokenizer: Any, target_size: int,
                           exclude_tokens: Set[int]) -> Set[int]:
    """Select tokens by frequency in the dataset input text."""
    from tqdm import tqdm

    print(
        f"Analyzing token frequencies in dataset with {len(dataset)} samples..."
    )
    token_counter = Counter()

    for sample in tqdm(dataset, desc="Tokenizing and counting tokens"):
        article = sample.get("article", "")
        if article:
            token_counter.update(
                tokenizer.encode(article, add_special_tokens=False))

    print(f"Found {len(token_counter)} unique tokens in dataset")

    selected = set()
    for token_id, _ in token_counter.most_common():
        if token_id not in exclude_tokens:
            selected.add(int(token_id))
            if len(selected) >= target_size:
                break

    if len(selected) < target_size:
        raise ValueError(
            f"Not enough unique tokens available. Requested {target_size}, "
            f"but only found {len(selected)} unique tokens in dataset.")

    return selected


def input_aware_filter(dataset: Any, tokenizer: Any, config: Any,
                       target_size: int, exclude_tokens: Set[int]) -> Set[int]:
    """Select tokens with the input-aware summarization heuristic."""
    from tqdm import tqdm

    tolerance_k = 5

    print("Input-aware vocabulary reduction algorithm for summarization task")
    print(f"Analyzing dataset with {len(dataset)} samples...")

    print("[Step 1] Building static vocabulary from output summaries...")
    output_counter = Counter()
    input_counter = Counter()

    for sample in tqdm(dataset, desc="Analyzing summaries and documents"):
        summary = sample.get("highlights", "")
        if summary:
            output_counter.update(
                tokenizer.encode(summary, add_special_tokens=False))

        document = sample.get("article", "")
        if document:
            input_counter.update(
                tokenizer.encode(document, add_special_tokens=False))

    input_tokens = set(input_counter)
    print(f"  - {len(output_counter)} unique tokens in summaries")
    print(f"  - {len(input_tokens)} unique tokens in documents")

    print("[Step 2] Applying input-aware filtering...")
    input_aware = {tid for tid in output_counter if tid in input_tokens}
    print(f"  - {len(input_aware)} tokens pass input-aware filter")

    print("[Step 3] Selecting most frequent task-specific tokens...")
    tolerance_budget = int(target_size * 0.1)
    core_budget = target_size - tolerance_budget

    core_vocab = set()
    for token_id, _ in output_counter.most_common():
        if len(core_vocab) >= core_budget:
            break
        if token_id not in exclude_tokens and token_id in input_aware:
            core_vocab.add(int(token_id))
    for token_id, _ in output_counter.most_common():
        if len(core_vocab) >= core_budget:
            break
        if token_id not in exclude_tokens:
            core_vocab.add(int(token_id))
    for token_id, _ in input_counter.most_common():
        if len(core_vocab) >= core_budget:
            break
        if token_id not in exclude_tokens:
            core_vocab.add(int(token_id))

    print(f"  - Selected {len(core_vocab)} core task-specific tokens")

    print(f"[Step 4] Applying tolerance filtering (k={tolerance_k})...")
    tolerance_tokens = set()

    vocab_size = get_vocab_size(config)
    for token_id in core_vocab:
        for offset in range(-tolerance_k, tolerance_k + 1):
            neighbor_id = token_id + offset
            if (0 <= neighbor_id < vocab_size and neighbor_id not in core_vocab
                    and neighbor_id not in exclude_tokens):
                tolerance_tokens.add(int(neighbor_id))
                if len(tolerance_tokens) >= tolerance_budget:
                    break
        if len(tolerance_tokens) >= tolerance_budget:
            break

    print(f"  - Added {len(tolerance_tokens)} tolerance tokens")

    final_selected = core_vocab | tolerance_tokens
    for counter in (output_counter, input_counter):
        if len(final_selected) >= target_size:
            break
        for token_id, _ in counter.most_common():
            if len(final_selected) >= target_size:
                break
            if token_id not in exclude_tokens:
                final_selected.add(int(token_id))
    for token_id in range(vocab_size):
        if len(final_selected) >= target_size:
            break
        if token_id not in exclude_tokens:
            final_selected.add(int(token_id))

    if len(final_selected) > target_size:
        final_selected = set(sorted(final_selected)[:target_size])
    if len(final_selected) < target_size:
        raise ValueError(
            f"Filter returned {len(final_selected)} tokens but expected "
            f"exactly {target_size}. Core vocab: {len(core_vocab)}, "
            f"tolerance: {len(tolerance_tokens)}")

    return final_selected


def reduce_vocab_size(tokenizer: Any,
                      config: Any,
                      dataset: Any,
                      reduced_vocab_size: int,
                      d2t_tensor: Optional[torch.Tensor] = None,
                      method: str = "frequency") -> torch.Tensor:
    """Create a reduced-vocabulary map from calibration data."""
    vocab_size = get_vocab_size(config)
    if reduced_vocab_size >= vocab_size:
        raise ValueError(
            f"reduced_vocab_size ({reduced_vocab_size}) must be less than "
            f"vocab_size ({vocab_size})")

    if method not in ["frequency", "input_aware"]:
        raise ValueError(
            f"method must be 'frequency' or 'input_aware', got {method!r}")

    required = get_special_tokens(tokenizer)

    if d2t_tensor is not None:
        if reduced_vocab_size <= len(d2t_tensor):
            raise ValueError(
                f"reduced_vocab_size ({reduced_vocab_size}) must be greater "
                f"than d2t_tensor size ({len(d2t_tensor)})")
        required.update(extract_d2t_required_tokens(d2t_tensor, vocab_size))

    remaining_slots = reduced_vocab_size - len(required)
    if remaining_slots < 0:
        raise ValueError(
            f"Required tokens ({len(required)}) exceeds reduced_vocab_size "
            f"({reduced_vocab_size})")

    if method == "frequency":
        additional = input_frequency_filter(dataset, tokenizer,
                                            remaining_slots, required)
    else:
        additional = input_aware_filter(dataset, tokenizer, config,
                                        remaining_slots, required)

    final_tokens = required | additional
    if len(final_tokens) != reduced_vocab_size:
        raise ValueError(
            f"Final vocabulary size ({len(final_tokens)}) does not match "
            f"target ({reduced_vocab_size}). Required: {len(required)}, "
            f"Additional: {len(additional)}")

    print(f"Final vocabulary composition ({method}):")
    print(f"  - Required tokens (d2t + special): {len(required)}")
    print(f"  - Method-selected tokens: {len(additional)}")
    print(f"  - Total vocabulary size: {len(final_tokens)}")

    return torch.tensor(sorted(final_tokens), dtype=torch.int32)


def main() -> None:
    """CLI entry point for generating ``vocab_map.safetensors``."""
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Reduce vocabulary size from calibration data")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory containing tokenizer and config")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Directory to save reduced-vocab artifacts")
    parser.add_argument(
        "--reduced_vocab_size",
        type=int,
        required=True,
        help="Target reduced vocabulary size, less than the original vocab")
    parser.add_argument("--method",
                        type=str,
                        choices=["input_aware", "frequency"],
                        default="input_aware",
                        help="Vocabulary reduction method")
    parser.add_argument("--max_samples",
                        type=int,
                        default=50000,
                        help="Maximum CNN/DailyMail samples to use")
    parser.add_argument(
        "--d2t_path",
        type=str,
        default=None,
        help="Optional EAGLE d2t.safetensors path. Referenced base tokens "
        "are always included.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Loading tokenizer and config from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        config = AutoConfig.from_pretrained(args.model_dir)

        vocab_size = get_vocab_size(config)
        print(f"Original vocabulary size: {vocab_size}")
        print(f"Target reduced vocabulary size: {args.reduced_vocab_size}")
        print(f"Method: {args.method}")

        print("Loading example dataset: cnn_dailymail")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples for vocabulary analysis")

        d2t_tensor = None
        if args.d2t_path:
            print(f"\nLoading d2t tensor from {args.d2t_path}...")
            d2t_data = load_file(args.d2t_path)
            if "d2t" not in d2t_data:
                raise KeyError("d2t tensor not found in d2t.safetensors")
            d2t_tensor = d2t_data["d2t"]
            print(f"Loaded d2t tensor with shape {d2t_tensor.shape}")

        print(f"\n{'=' * 70}")
        print(f"Reducing vocabulary with {args.method!r} method...")
        print(f"{'=' * 70}\n")

        vocab_map = reduce_vocab_size(
            tokenizer=tokenizer,
            config=config,
            dataset=dataset,
            reduced_vocab_size=args.reduced_vocab_size,
            d2t_tensor=d2t_tensor,
            method=args.method)

        vocab_map_path = os.path.join(args.output_dir, VOCAB_MAP_NAME)
        print(f"Saving vocabulary map to {vocab_map_path}...")
        save_file({"vocab_map": vocab_map}, str(vocab_map_path))

        vocab_info = {
            "vocab_size": vocab_size,
            "reduced_vocab_size": int(vocab_map.numel()),
            "method": args.method,
            "dataset": "cnn_dailymail",
            "max_samples": min(args.max_samples, len(dataset)),
        }
        if args.d2t_path:
            vocab_info["d2t_tensor_size"] = len(d2t_tensor)

        vocab_info_path = os.path.join(args.output_dir, VOCAB_INFO_NAME)
        print(f"Saving vocabulary info to {vocab_info_path}...")
        with open(vocab_info_path, "w") as f:
            json.dump(vocab_info, f, indent=2)

        print("Vocabulary reduction completed successfully!")
        print(f"Output files saved to: {args.output_dir}")
        print(f"  - {VOCAB_MAP_NAME}: Vocabulary mapping tensor "
              f"[{int(vocab_map.numel())}]")
        print(f"  - {VOCAB_INFO_NAME}: Vocabulary size information")

    except Exception as exc:
        print(f"Error during vocabulary reduction: {exc}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)
