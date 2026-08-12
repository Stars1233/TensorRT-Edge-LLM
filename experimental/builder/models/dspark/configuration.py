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
"""DSpark target/draft pairing configuration."""

from ...core import contracts
from ...core.bundle import BundleConfig


def _dspark_config(draft: dict) -> dict:
    nested = draft.get("dspark_config") or {}
    return {
        "target_layer_ids":
        nested.get("target_layer_ids", draft.get("target_layer_ids", [])),
        "block_size":
        nested.get("block_size", draft.get("block_size", 7)),
        "mask_token_id":
        nested.get("mask_token_id", draft.get("mask_token_id", 151669)),
        "enable_confidence_head":
        nested.get("enable_confidence_head",
                   draft.get("enable_confidence_head", False)),
        "confidence_head_with_markov":
        nested.get("confidence_head_with_markov",
                   draft.get("confidence_head_with_markov", False)),
        "markov_head_type":
        nested.get("markov_head_type", draft.get("markov_head_type", "")),
        "markov_rank":
        nested.get("markov_rank", draft.get("markov_rank", 0)),
    }


def _validate_dimensions(draft, target) -> None:
    if target.hidden_size != draft.hidden_size:
        raise ValueError("DSpark base/draft hidden sizes must match: "
                         f"{target.hidden_size} != {draft.hidden_size}")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("DSpark base/draft vocab sizes must match: "
                         f"{target.vocab_size} != {draft.vocab_size}")


def _validate_runtime_contract(config, build_args) -> None:
    if build_args is None:
        return
    if build_args.max_verify_tree_size != build_args.max_draft_tree_size + 1:
        raise ValueError(
            "DSpark requires max_verify_tree_size == max_draft_tree_size + 1")
    if build_args.max_draft_tree_size > config.dspark_block_size:
        raise ValueError(
            "DSpark max_draft_tree_size exceeds the checkpoint block_size")
    if build_args.reduced_vocab_dir or build_args.draft_reduced_vocab_dir:
        raise ValueError("DSpark does not support reduced-vocabulary engines")
    markov_type = config.dspark_markov_head_type or "vanilla"
    if markov_type != "vanilla":
        raise ValueError(
            f"DSpark supports markov_head_type='vanilla', got {markov_type!r}")
    if not 0 <= config.dspark_mask_token_id < config.vocab_size:
        raise ValueError(
            "DSpark mask_token_id is outside the draft vocabulary")


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Read the target-hidden and sequential-head contract from the draft."""
    if not paired_draft_dir:
        raise ValueError("DSpark base requires a paired draft checkpoint")
    bundle = BundleConfig.from_pretrained(paired_draft_dir)
    draft = bundle.component_dict(contracts.Component.LLM)
    values = _dspark_config(draft)
    target_layers = [int(index) for index in values["target_layer_ids"]]
    if not target_layers:
        raise ValueError("DSpark draft config must provide target_layer_ids")
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("DSpark target-layer IDs must be unique")
    invalid = [
        index for index in target_layers
        if index < 0 or index >= config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"DSpark target-layer IDs outside base model: {invalid}")
    draft_hidden = int(draft.get("hidden_size", 0))
    if draft_hidden != config.hidden_size:
        raise ValueError("DSpark base/draft hidden sizes must match: "
                         f"{config.hidden_size} != {draft_hidden}")
    draft_vocab = int(draft.get("vocab_size", 0))
    if draft_vocab != config.vocab_size:
        raise ValueError("DSpark base/draft vocab sizes must match: "
                         f"{config.vocab_size} != {draft_vocab}")

    config.dspark_base = True
    config.dspark_target_layer_ids = target_layers
    config.dspark_block_size = int(values["block_size"])
    config.dspark_mask_token_id = int(values["mask_token_id"])
    config.dspark_enable_confidence_head = bool(
        values["enable_confidence_head"])
    config.dspark_confidence_head_with_markov = bool(
        values["confidence_head_with_markov"])
    config.dspark_markov_head_type = str(values["markov_head_type"])
    config.dspark_markov_rank = int(values["markov_rank"])
    _validate_runtime_contract(config, build_args)


def configure_draft(config,
                    *,
                    paired_target=None,
                    build_args=None,
                    **kwargs) -> None:
    """Validate and normalize one DSpark draft checkpoint."""
    if paired_target is None:
        raise ValueError("DSpark draft requires a target config")
    _validate_dimensions(config, paired_target)
    if not config.dspark_target_layer_ids:
        raise ValueError("DSpark draft config must provide target_layer_ids")
    if config.dspark_markov_rank <= 0:
        raise ValueError("DSpark draft requires markov_rank > 0")
    _validate_runtime_contract(config, build_args)
