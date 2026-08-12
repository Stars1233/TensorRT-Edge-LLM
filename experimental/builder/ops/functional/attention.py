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
"""Attention operations."""

from typing import Optional, Sequence, Tuple

from ..tensor import Tensor
from ._operation import operation

KV_PAGE_SIZE = 128

__all__ = ["KV_PAGE_SIZE", "attention", "gemma4_attention", "vit_attention"]


def attention(
    qkv: Tensor,
    past_kv: Tensor,
    context_lengths: Tensor,
    rope_cos_sin: Tensor,
    kvcache_start_index: Tensor,
    kv_page_table: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int = -1,
    enable_fp8_kv_cache: bool = False,
    qkv_scales: Sequence[float] = (1.0, 1.0, 1.0),
    q_norm_gamma: Optional[Tensor] = None,
    k_norm_gamma: Optional[Tensor] = None,
    rms_norm_eps: float = 1e-6,
    attention_scale: Optional[float] = None,
    enable_kv_shared: bool = False,
    context_mask_selector: Optional[Tensor] = None,
    vision_block_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    attention_pos_id: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Run paged decoder attention and return output plus present KV.

    ``qkv`` is packed on the last dimension as Q, K, V. A shared-KV layer
    passes Q alone because K/V come from a donated cache.
    """

    if (attention_mask is None) != (attention_pos_id is None):
        raise ValueError(
            "attention_mask and attention_pos_id must be supplied together")
    if (q_norm_gamma is None) != (k_norm_gamma is None):
        raise ValueError(
            "q_norm_gamma and k_norm_gamma must be supplied together")
    enable_qk_norm = q_norm_gamma is not None
    if enable_qk_norm and enable_kv_shared:
        raise ValueError("QK normalization is not supported with shared KV")
    if vision_block_ids is not None and attention_mask is not None:
        raise ValueError(
            "vision-block attention and tree attention are mutually exclusive")

    attributes = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "enable_tree_attention": int(attention_mask is not None),
        "enable_qk_norm": int(enable_qk_norm),
        "enable_kv_shared": int(enable_kv_shared),
        "enable_context_mask_selector": int(context_mask_selector is not None),
        "enable_fp8_kv_cache": int(enable_fp8_kv_cache),
        "enable_vision_block_attention": int(vision_block_ids is not None),
        "sliding_window_size": sliding_window_size,
        "qkv_scales": qkv_scales,
    }
    if attention_scale is not None:
        attributes["attention_scale"] = attention_scale
    if enable_qk_norm:
        attributes["rms_norm_eps"] = rms_norm_eps
    inputs = [
        qkv, past_kv, context_lengths, rope_cos_sin, kvcache_start_index,
        kv_page_table
    ]
    if enable_qk_norm:
        inputs.extend((q_norm_gamma, k_norm_gamma))
    if context_mask_selector is not None:
        inputs.append(context_mask_selector)
    if attention_mask is not None:
        inputs.extend((attention_mask, attention_pos_id))
    if vision_block_ids is not None:
        inputs.append(vision_block_ids)
    attn_4d, present_kv = operation("attention",
                                    inputs,
                                    output_count=2,
                                    **attributes)
    attn = attn_4d.reshape((0, 0, num_q_heads * head_size))
    return attn, present_kv


def gemma4_attention(q_raw: Tensor, k_raw: Tensor, value: Tensor,
                     gamma: Tensor, relative_key: Tensor, valid: Tensor,
                     sequence_length: Tensor, chunk_size: int,
                     left_horizon: int, context_size: int,
                     logit_cap: float) -> Tensor:
    """Run Gemma4 chunked local audio attention."""
    inputs = [q_raw, k_raw, value, gamma, relative_key, valid, sequence_length]
    return operation("gemma4_attention",
                     inputs,
                     chunk_size=chunk_size,
                     left_horizon=left_horizon,
                     context_size=context_size,
                     logit_cap=logit_cap)


def vit_attention(query: Tensor, key: Tensor, value: Tensor,
                  cu_seqlens: Tensor, max_seqlen_carrier: Tensor,
                  num_heads: int, head_size: int) -> Tensor:
    """Run packed vision-transformer attention."""
    return operation("vit_attention",
                     [query, key, value, cu_seqlens, max_seqlen_carrier],
                     num_heads=num_heads,
                     head_size=head_size)
