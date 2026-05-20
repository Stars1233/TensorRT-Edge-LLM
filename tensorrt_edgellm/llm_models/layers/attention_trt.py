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
TensorRT Native Operations Attention Layer

This module provides an attention implementation using TensorRT's native ONNX operations
(RoPE, Attention, KVCacheUpdate) instead of custom plugins. This enables plugin-free
deployment with TensorRT >= 10.15.

Key differences from plugin-based attention:
- Uses separate K and V cache tensors instead of combined format
- Exports to native ONNX operations that TensorRT recognizes
- No custom plugin library required at runtime
- Limited to vanilla decoding (no tree attention or reusable KV cache in v1)
"""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.onnx import symbolic_helper
from torch.onnx.symbolic_helper import _get_tensor_sizes

from ...common import ONNX_OPSET_VERSION
from .layer_utils import EdgeLLMQKNorm, EdgeLLMQKVProj


@symbolic_helper.parse_args("v", "v", "v")
def symbolic_kv_cache_update(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    cache: torch._C.Value,
    new_kv: torch._C.Value,
    cache_indices: torch._C.Value,
):
    """Symbolic function for ONNX export of KV cache update."""

    cache_shape = _get_tensor_sizes(cache)

    inputs = [cache, new_kv, cache_indices]
    updated_cache = g.op("TensorScatter", *inputs)
    updated_cache.setType(cache.type().with_sizes(cache_shape))

    return updated_cache


@torch.library.custom_op("trt::kv_cache_update_onnx", mutates_args=())
def kv_cache_update_onnx(
    cache: torch.Tensor,
    new_kv: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Dummy implementation for ONNX export, this is not used in the actual inference."""
    return cache.clone()


@symbolic_helper.parse_args("v", "v", "v", "v", "b", "f")
def symbolic_attention(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: torch._C.Value,
    is_causal: bool,
    scale: float,
):
    """Symbolic function for ONNX export of attention."""
    query_shape = _get_tensor_sizes(query)

    inputs = [query, key, value]
    if attn_mask is not None:
        inputs.append(attn_mask)

    # Create attention node with attributes
    attn_output = g.op("Attention",
                       *inputs,
                       is_causal_i=is_causal,
                       TRT_decomposable_i=1,
                       scale_f=scale)
    attn_output.setType(query.type().with_sizes(query_shape))

    return attn_output


@torch.library.custom_op("trt::attention_onnx", mutates_args=())
def attention_onnx(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    """Dummy implementation for ONNX export, this is not used in the actual inference."""
    # Return a dummy tensor with the same shape as query
    return query.clone()


@symbolic_helper.parse_args("v", "v", "v", "v")
def symbolic_rope(
    g: torch.onnx._internal.torchscript_exporter.jit_utils.GraphContext,
    x: torch._C.Value,
    cos: torch._C.Value,
    sin: torch._C.Value,
    position_ids: torch._C.Value,
):
    """Symbolic function for ONNX export of RoPE."""
    x_shape = _get_tensor_sizes(x)

    inputs = [x, cos, sin, position_ids]
    rope_output = g.op("RotaryEmbedding", *inputs)
    rope_output.setType(x.type().with_sizes(x_shape))

    return rope_output


@torch.library.custom_op("trt::rope_onnx", mutates_args=())
def rope_onnx(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Dummy implementation for ONNX export, this is not used in the actual inference."""
    # Return a dummy tensor with the same shape as input
    return x.clone()


class EdgeLLMAttentionTRTNative(nn.Module):
    """
    Multi-headed attention using TensorRT native operations.
    
    This module implements attention using operations that export to TensorRT's
    native ONNX layers: RotaryEmbedding, KVCacheUpdate, and Attention.
    It supports vanilla decoding with separate K and V caches.
    
    Attributes:
        q_proj: Query projection layer
        k_proj: Key projection layer
        v_proj: Value projection layer
        o_proj: Output projection layer
        q_norm: Query normalization layer (optional, for Qwen3 models)
        k_norm: Key normalization layer (optional, for Qwen3 models)
        qk_norm: QK normalization layer (optional, for Llama4 models)
        hidden_size: Hidden dimension size
        num_key_value_heads: Number of key-value heads
        num_attention_heads: Number of attention heads
        head_dim: Dimension of each attention head
        max_position_embeddings: Maximum sequence length for positional embeddings
        qk_scale: Scaling factor for Q@K^T
    """

    def __init__(self, attention_module: nn.Module,
                 eagle3_draft: bool) -> None:
        """
        Initialize the EdgeLLMAttentionTRTNative module.
        
        Args:
            attention_module: Original attention module to extract components from
        """
        super().__init__()

        # Copy projection layers from original attention module
        self.qkv_proj = EdgeLLMQKVProj(attention_module, eagle3_draft)
        self.o_proj = attention_module.o_proj
        self.qk_norm = EdgeLLMQKNorm(attention_module)
        self.eagle3_draft = eagle3_draft

        # Copy configuration attributes from the original attention module
        self.hidden_size: int = attention_module.config.hidden_size
        self.num_key_value_heads: int = attention_module.config.num_key_value_heads
        self.num_attention_heads: int = attention_module.config.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
        self.num_key_value_groups: int = self.num_attention_heads // self.num_key_value_heads

        # Set head dimension
        if hasattr(attention_module.config, 'head_dim'):
            self.head_dim: int = attention_module.config.head_dim
        else:
            self.head_dim: int = attention_module.config.hidden_size // self.num_attention_heads

        # Maximum sequence length for positional embeddings
        self.max_position_embeddings: int = attention_module.config.max_position_embeddings

        # Compute QK scale factor
        self.qk_scale: float = 1.0 / (self.head_dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        slice_kv_cache: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for TensorRT native operations attention computation.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            k_cache: Key cache of shape (batch_size, num_kv_heads, capacity, head_dim)
            v_cache: Value cache of shape (batch_size, num_kv_heads, capacity, head_dim)
            rope_rotary_cos_sin: RoPE rotary embeddings of shape (batch_size, max_position_embeddings, head_dim)
            context_lengths: Context length tensor of shape (batch_size,)
            kvcache_start_index: Start index of KV cache of shape (batch_size), optional
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len), optional
            position_ids: Position IDs of shape (batch_size, seq_len), optional
            slice_kv_cache: Whether to slice the KV cache to the current context length, optional
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Attention output of shape (batch_size, seq_len, hidden_size)
                - Updated K cache
                - Updated V cache
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply Q, K, V projections
        query_states, key_states, value_states, gate_states = self.qkv_proj(
            hidden_states)

        norm_shape = [bsz, q_len, -1, self.head_dim]
        query_states, key_states = self.qk_norm(query_states, key_states,
                                                norm_shape)

        # Convert to FP16 for TensorRT compatibility
        compute_type = torch.float16
        io_type = torch.float16
        query_states = query_states.to(io_type)
        key_states = key_states.to(io_type)
        value_states = value_states.to(io_type)

        if k_cache.dtype != io_type:
            k_cache = k_cache.to(io_type)
        if v_cache.dtype != io_type:
            v_cache = v_cache.to(io_type)

        # Reshape Q, K, V: [batch, seq_len, num_heads * head_dim] -> [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_attention_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        # Extract cos and sin from rope_rotary_cos_sin [batch, max_pos_emb, head_dim]
        # Split into cos [max_pos_emb, head_dim//2] and sin [max_pos_emb, head_dim//2]
        half_dim = self.head_dim // 2
        rope_cos = rope_rotary_cos_sin[:, :, :half_dim]
        rope_sin = rope_rotary_cos_sin[:, :, half_dim:]
        rope_cos = rope_cos[0:1, :, :]
        rope_sin = rope_sin[0:1, :, :]
        rope_cos = rope_cos.reshape(-1, half_dim)
        rope_sin = rope_sin.reshape(-1, half_dim)
        rope_cos = rope_cos.to(compute_type)
        rope_sin = rope_sin.to(compute_type)

        # if position_ids is not provided, use kvcache_start_index to generate it
        # position_ids should have shape (batch_size, seq_len) and for each batch, it start with kvcache_start_index and increment by 1
        if position_ids is None:
            position_ids = kvcache_start_index.unsqueeze(1) + torch.arange(
                q_len,
                device=kvcache_start_index.device,
                dtype=kvcache_start_index.dtype).unsqueeze(0)

        # Apply RoPE to Q and K using onnx ops
        query_states = query_states.to(compute_type)
        key_states = key_states.to(compute_type)
        query_states = self._apply_rope(query_states, rope_cos, rope_sin,
                                        position_ids)
        key_states = self._apply_rope(key_states, rope_cos, rope_sin,
                                      position_ids)
        query_states = query_states.to(io_type)
        key_states = key_states.to(io_type)
        value_states = value_states.to(io_type)

        # Update K and V caches using the TensorRT native KV cache updater
        present_k_cache = kv_cache_update_onnx(k_cache, key_states,
                                               kvcache_start_index)
        present_v_cache = kv_cache_update_onnx(v_cache, value_states,
                                               kvcache_start_index)

        # Get present length for slicing
        present_length = torch.max(kvcache_start_index) + torch.max(
            context_lengths)

        if slice_kv_cache:
            # Slice present K and V from caches [batch, num_heads, present_length, head_dim]
            k_present = present_k_cache[:, :, :present_length, :]
            v_present = present_v_cache[:, :, :present_length, :]
        else:
            k_present = present_k_cache
            v_present = present_v_cache

        # Apply QK scale to query
        query_states = query_states * self.qk_scale

        attn_output = self._compute_attention(query_states, k_present,
                                              v_present, attention_mask)

        # Reshape output: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            bsz, q_len, self.num_attention_heads * self.head_dim)

        # Apply output projection
        attn_output = attn_output.to(io_type)
        if gate_states is not None:
            attn_output = attn_output * torch.sigmoid(
                gate_states.to(attn_output.dtype))
        attn_output = self.o_proj(attn_output)

        return attn_output, present_k_cache, present_v_cache

    def _apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim]
            cos: Cosine values of shape [batch, max_pos_emb, head_dim//2]
            sin: Sine values of shape [batch, max_pos_emb, head_dim//2]
            position_ids: Position indices of shape [batch, seq_len]
            
        Returns:
            Tensor with RoPE applied
        """
        rope_x = rope_onnx(x, cos, sin, position_ids)
        return rope_x

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        This function uses a custom_op implementation that exports to 
        TensorRT's native Attention layer.
        
        Args:
            query: Query tensor of shape [batch, num_heads, seq_q, head_dim]
            key: Key tensor of shape [batch, num_heads, seq_k, head_dim]
            value: Value tensor of shape [batch, num_heads, seq_k, head_dim]
            
        Returns:
            Attention output of shape [batch, num_heads, seq_q, head_dim]
        """

        # Use mask for causal in TensorRT 10.15
        is_causal = False
        if mask is None:
            # Generate causal mask with lower-right alignment
            # For query position i and key position j, allow attention if:
            # j <= i + (seq_k - seq_q), which is equivalent to lower-right alignment
            batch_size, num_heads, seq_q, head_dim = query.shape
            seq_k = key.shape[2]

            # Create row indices (query positions): [seq_q, 1]
            row_indices = torch.arange(seq_q,
                                       device=query.device,
                                       dtype=torch.int32).reshape(seq_q, 1)
            # Create column indices (key positions): [1, seq_k]
            col_indices = torch.arange(seq_k,
                                       device=query.device,
                                       dtype=torch.int32).reshape(1, seq_k)

            # Lower-right alignment: allow if col <= row + (seq_k - seq_q)
            # Equivalent to: col - row <= seq_k - seq_q
            # Or: row - col >= seq_q - seq_k
            offset = seq_k - seq_q
            # causal_mask is True where attention is NOT allowed (will be masked out)
            causal_mask = col_indices > (row_indices + offset)

            # Convert boolean mask to attention mask format
            # Where causal_mask is True (should be masked), use -inf; otherwise 0
            mask = torch.where(
                causal_mask,
                torch.tensor(float('-inf'),
                             device=query.device,
                             dtype=query.dtype),
                torch.tensor(0.0, device=query.device, dtype=query.dtype))
            # Reshape to [1, 1, seq_q, seq_k] for broadcasting
            mask = mask.reshape(1, 1, seq_q, seq_k)

        # Use custom_op for attention
        # Note: scale is already applied to query, so we use scale=1.0
        attn_output = attention_onnx(query,
                                     key,
                                     value,
                                     attn_mask=mask,
                                     is_causal=is_causal,
                                     scale=1.0)

        return attn_output


def register_trt_native_attention_onnx_symbolic_functions() -> None:
    """
    Register symbolic functions for ONNX export of TensorRT native kv cache update, rope, and attention operations.
    Using ONNX TensorScatter, RotaryEmbedding, and Attention, supported by TensorRT >= 10.15.
    """

    from torch.onnx import register_custom_op_symbolic

    # Register our custom symbolic functions
    register_custom_op_symbolic("trt::kv_cache_update_onnx",
                                symbolic_kv_cache_update, ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::rope_onnx", symbolic_rope,
                                ONNX_OPSET_VERSION)
    register_custom_op_symbolic("trt::attention_onnx", symbolic_attention,
                                ONNX_OPSET_VERSION)

    print(
        "Registered ONNX symbolic functions for TensorRT native kv cache update (TensorScatter), rope (RotaryEmbedding), and attention (Attention)"
    )
