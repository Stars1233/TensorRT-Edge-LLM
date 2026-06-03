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
``torch.library.custom_op`` stubs for inference export.

Each op is a trace-time dummy (returns zero tensors of the correct shape/dtype)
paired with a ``register_fake`` for shape propagation in the dynamo exporter.
Domains ``trt::`` / ``trt_edgellm::`` map to ONNX nodes consumed by the
TensorRT plugin runtime.
"""

from typing import List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Custom op: trt::attention_plugin  (unified: vanilla / FP8-KV / EAGLE tree)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::attention_plugin", mutates_args=())
def attention_plugin(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    sliding_window_size: int,
    enable_tree_attention: bool,
    enable_fp8_kv_cache: bool,
    attention_mask: Optional[torch.Tensor] = None,
    attention_pos_id: Optional[torch.Tensor] = None,
    qkv_scales: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified stub for AttentionPlugin covering all feature combinations.

    Feature matrix (all map to the same TRT ``AttentionPlugin``):

    +-----------------------+--------------------+----------------------------+
    | Mode                  | enable_tree_attn   | enable_fp8_kv_cache        |
    +=======================+====================+============================+
    | Vanilla               | False              | False                      |
    +-----------------------+--------------------+----------------------------+
    | FP8 KV cache          | False              | True  (qkv_scales set)     |
    +-----------------------+--------------------+----------------------------+
    | EAGLE tree attention  | True               | False                      |
    +-----------------------+--------------------+----------------------------+
    | EAGLE + FP8 KV        | True               | True  (qkv_scales set)     |
    +-----------------------+--------------------+----------------------------+

    ``enable_tree_attention`` and ``enable_fp8_kv_cache`` are required (no
    default) so that ``torch.export`` always includes them in the FX graph
    — default-matching kwargs get stripped, breaking ONNX translation.

    Callers must always pass ``qkv_scales=[1.0, 1.0, 1.0]`` explicitly so
    the FX graph contains a valid FLOATS value for the ONNX translation.

    When ``enable_tree_attention=True``, ``attention_mask`` and
    ``attention_pos_id`` must be provided (non-None).

    The TRT AttentionPlugin kernel returns a 4-D tensor
    ``[batch, seq_len, num_q_heads, head_size]``.
    The caller (``Attention.forward``) is responsible for reshaping to
    ``[batch, seq_len, num_q_heads * head_size]``.
    """
    batch_size, seq_len, _ = query_states.shape
    past_len = past_key_value.shape[3]
    attn_output = torch.zeros(batch_size,
                              seq_len,
                              num_q_heads,
                              head_size,
                              dtype=query_states.dtype,
                              device=query_states.device)
    present_key_value = torch.zeros(batch_size,
                                    2,
                                    num_kv_heads,
                                    past_len + seq_len,
                                    head_size,
                                    dtype=past_key_value.dtype,
                                    device=past_key_value.device)
    return attn_output, present_key_value


@attention_plugin.register_fake
def _(query_states,
      key_states,
      value_states,
      past_key_value,
      context_lengths,
      rope_rotary_cos_sin,
      kvcache_start_index,
      num_q_heads,
      num_kv_heads,
      head_size,
      sliding_window_size,
      enable_tree_attention,
      enable_fp8_kv_cache,
      attention_mask=None,
      attention_pos_id=None,
      qkv_scales=None):
    batch_size, seq_len, _ = query_states.shape
    past_len = past_key_value.shape[3]
    return (torch.empty(batch_size,
                        seq_len,
                        num_q_heads,
                        head_size,
                        dtype=query_states.dtype,
                        device=query_states.device),
            torch.empty(batch_size,
                        2,
                        num_kv_heads,
                        past_len + seq_len,
                        head_size,
                        dtype=past_key_value.dtype,
                        device=past_key_value.device))


# ---------------------------------------------------------------------------
# Custom op: trt::vit_attention_plugin  (ViT ragged self-attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
def vit_attention_plugin(
    query_states: torch.Tensor,  # [T, num_heads, head_size]
    key_states: torch.Tensor,  # [T, num_heads, head_size]
    value_states: torch.Tensor,  # [T, num_heads, head_size]
    cu_seqlens: torch.Tensor,  # [batch+1] int32
    max_seqlen_carrier: torch.Tensor,  # [] or [1] int32 (scalar)
    num_heads: int,
    head_size: int,
) -> torch.Tensor:
    """ViT ragged self-attention.

    In eager mode, implements varlen SDPA using cu_seqlens to process each
    sequence segment independently.  During dynamo/ONNX tracing the
    register_fake shape propagation is used and this body is not executed.

    Unlike AttentionPlugin, ViT attention has no KV cache and takes ragged
    input with cu_seqlens instead of context_lengths.  RoPE is applied before
    this call.
    """
    import torch.nn.functional as F
    out = torch.empty_like(query_states)
    seqlens = cu_seqlens.tolist()
    for i in range(len(seqlens) - 1):
        start, end = int(seqlens[i]), int(seqlens[i + 1])
        if start >= end:
            continue
        # q/k/v: [S, H, D] -> [1, H, S, D] for SDPA
        q = query_states[start:end].permute(1, 0, 2).unsqueeze(0)
        k = key_states[start:end].permute(1, 0, 2).unsqueeze(0)
        v = value_states[start:end].permute(1, 0, 2).unsqueeze(0)
        attn = F.scaled_dot_product_attention(q, k, v)  # [1, H, S, D]
        out[start:end] = attn.squeeze(0).permute(1, 0, 2)
    return out


@vit_attention_plugin.register_fake
def _(query_states, key_states, value_states, cu_seqlens, max_seqlen_carrier,
      num_heads, head_size):
    return torch.empty_like(query_states)


# ---------------------------------------------------------------------------
# Custom op: trt::fp8_quantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::fp8_quantize", mutates_args=())
def fp8_quantize(
        hidden_states: torch.Tensor,  # float16 input
        scale: torch.Tensor,  # float16 per-tensor scale (scalar)
) -> torch.Tensor:
    """Stub: quantize float16 -> FP8; ONNX export -> QuantizeLinear."""
    return torch.zeros_like(hidden_states)


@fp8_quantize.register_fake
def _(hidden_states, scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::fp8_dequantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::fp8_dequantize", mutates_args=())
def fp8_dequantize(
        weight: torch.Tensor,  # fp8_e4m3fn [out, in]
        weight_scale: torch.Tensor,  # float16 per-tensor scale (scalar)
) -> torch.Tensor:
    """Stub: dequantize FP8 -> float16; ONNX export -> DequantizeLinear."""
    return torch.zeros_like(weight, dtype=torch.float16)


@fp8_dequantize.register_fake
def _(weight, weight_scale):
    return torch.empty_like(weight, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_act_qdq  (activation DynQ + 2DQ -> float16)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_act_qdq", mutates_args=())
def nvfp4_act_qdq(
        hidden_states: torch.Tensor,  # float16 activation
        global_scale: torch.Tensor,  # float32 scalar: amax / (6.0 * 448.0)
) -> torch.Tensor:
    """Stub: NVFP4 activation QDQ (DynQ + 2 trt::DQ). Returns float16.

    In the ONNX graph this emits three nodes matching ModelOpt's
    ``export_fp4(onnx_quantizer_type="dynamic")`` pattern::

        TRT_FP4DynamicQuantize(x, scale_f32, axis=-1, block_size=16, scale_type=17)
            -> (x_f4, sx_f8)
        trt::DequantizeLinear(sx_f8, scale_f32)
            -> dq_scale
        trt::DequantizeLinear(x_f4, dq_scale, axis=-1, block_size=16)
            -> x_dq  [float16]
    """
    return torch.zeros_like(hidden_states)


@nvfp4_act_qdq.register_fake
def _(hidden_states, global_scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::nvfp4_dequantize
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::nvfp4_dequantize", mutates_args=())
def nvfp4_dequantize(
    weight: torch.Tensor,  # uint8 [out, in//2] packed fp4
    weight_scale: torch.Tensor,  # fp8_e4m3fn [out, in//group_size]
    weight_scale_2: torch.Tensor,  # float32 scalar
    group_size: int,
) -> torch.Tensor:
    """Stub: dequantize NVFP4 packed weight to float16."""
    out_features, packed_in = weight.shape
    in_features = packed_in * 2
    return torch.zeros(out_features,
                       in_features,
                       dtype=torch.float16,
                       device=weight.device)


@nvfp4_dequantize.register_fake
def _(weight, weight_scale, weight_scale_2, group_size):
    out_features, packed_in = weight.shape
    return torch.empty(out_features,
                       packed_in * 2,
                       dtype=torch.float16,
                       device=weight.device)


# ---------------------------------------------------------------------------
# Custom op: trt::mxfp8_act_qdq  (activation DynQ + DQ -> float16)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::mxfp8_act_qdq", mutates_args=())
def mxfp8_act_qdq(
        hidden_states: torch.Tensor,  # float16 activation
) -> torch.Tensor:
    """Stub: MXFP8 activation DynQ + DQ. Returns float16.

    In the ONNX graph this emits two nodes::

        TRT_MXFP8DynamicQuantize(x, axis=-1, block_size=32, output_dtype=17)
            -> (x_f8, sx_e8m0)
        TRT_MXFP8DequantizeLinear(x_f8, sx_e8m0,
            axis=-1, block_size=32, output_dtype=10)
            -> x_dq  [float16]
    """
    return torch.zeros_like(hidden_states)


@mxfp8_act_qdq.register_fake
def _(hidden_states):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::mxfp8_weight_dq
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::mxfp8_weight_dq", mutates_args=())
def mxfp8_weight_dq(
    weight: torch.Tensor,  # fp8_e4m3fn [out, in]
    weight_scale: torch.Tensor,  # uint8 (E8M0) [out, in // block_size]
    block_size: int,
) -> torch.Tensor:
    """Stub: dequantize MXFP8 weight (FP8E4M3 + E8M0 scale) to float16.

    Emits::

        TRT_MXFP8DequantizeLinear(weight, weight_scale,
            axis=-1, block_size=block_size, output_dtype=10) -> w_dq [float16]
    """
    return torch.zeros(weight.shape[0],
                       weight.shape[1],
                       dtype=torch.float16,
                       device=weight.device)


@mxfp8_weight_dq.register_fake
def _(weight, weight_scale, block_size):
    return torch.empty(weight.shape[0],
                       weight.shape[1],
                       dtype=torch.float16,
                       device=weight.device)


# ---------------------------------------------------------------------------
# Custom op: trt::int4_groupwise_gemm
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int4_groupwise_gemm", mutates_args=())
def int4_groupwise_gemm(
    hidden_states: torch.Tensor,  # [*, in_features] float16
    qweight: torch.Tensor,  # [out_features//2, in_features] int8 (swizzled)
    scales: torch.Tensor,  # [in_features//group_size, out_features] float16
    gemm_n: int,
    gemm_k: int,
    group_size: int,
) -> torch.Tensor:
    """Stub: INT4 groupwise GEMM - returns zero tensor of correct shape."""
    *leading, _ = hidden_states.shape
    return torch.zeros(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


@int4_groupwise_gemm.register_fake
def _(hidden_states, qweight, scales, gemm_n, gemm_k, group_size):
    *leading, _ = hidden_states.shape
    return torch.empty(*leading,
                       gemm_n,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


# ---------------------------------------------------------------------------
# Custom op: trt::int8_sq_act_qdq  (INT8 SmoothQuant activation QDQ)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int8_sq_act_qdq", mutates_args=())
def int8_sq_act_qdq(
        hidden_states: torch.Tensor,  # float16 smoothed activation [*, in]
        scale: torch.Tensor,  # float32 per-tensor input scale []
) -> torch.Tensor:
    """Stub: symmetric per-tensor INT8 QuantizeLinear + DequantizeLinear.

    In the ONNX graph emits::

        QuantizeLinear(x, scale, output_dtype=INT8) -> q
        DequantizeLinear(q, scale)                  -> dq  [float32]
        Cast(dq, to=FLOAT16)                        -> output
    """
    return torch.zeros_like(hidden_states)


@int8_sq_act_qdq.register_fake
def _(hidden_states, scale):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt::int8_sq_weight_dq  (INT8 per-channel weight dequantize)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::int8_sq_weight_dq", mutates_args=())
def int8_sq_weight_dq(
        weight: torch.Tensor,  # int8 [out, in]
        scale: torch.Tensor,  # float32 [out] per-channel scale
) -> torch.Tensor:
    """Stub: per-channel INT8 DequantizeLinear (axis=0), output float16.

    In the ONNX graph emits::

        DequantizeLinear(weight, scale, axis=0) -> dq  [float32]
        Cast(dq, to=FLOAT16)                    -> output
    """
    return torch.zeros_like(weight, dtype=torch.float16)


@int8_sq_weight_dq.register_fake
def _(weight, scale):
    return torch.empty_like(weight, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::causal_conv1d
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::causal_conv1d", mutates_args=())
def causal_conv1d(
    hidden_states: torch.Tensor,  # [batch, seq_len, conv_dim]
    weight: torch.Tensor,  # [conv_dim, 1, kernel_size]
    bias: torch.Tensor,  # [conv_dim]
    conv_state: torch.Tensor,  # [batch, conv_dim, conv_kernel]
    context_lengths: torch.Tensor,  # [batch] int32
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    collect_intermediate_states: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub: causal conv1d; with MTP enabled it returns an optional 3rd output."""
    if collect_intermediate_states:
        batch_size, seq_len, _ = hidden_states.shape
        intermediate_conv_state = torch.zeros(batch_size,
                                              seq_len,
                                              conv_state.shape[1],
                                              conv_state.shape[2],
                                              dtype=conv_state.dtype,
                                              device=conv_state.device)
        return (torch.zeros_like(hidden_states), conv_state.clone(),
                intermediate_conv_state)
    return torch.zeros_like(hidden_states), conv_state.clone()


@causal_conv1d.register_fake
def _(hidden_states,
      weight,
      bias,
      conv_state,
      context_lengths,
      stride,
      padding,
      dilation,
      groups,
      collect_intermediate_states=False):
    if collect_intermediate_states:
        batch_size, seq_len, _ = hidden_states.shape
        intermediate_conv_state = torch.empty(batch_size,
                                              seq_len,
                                              conv_state.shape[1],
                                              conv_state.shape[2],
                                              dtype=conv_state.dtype,
                                              device=conv_state.device)
        return (torch.empty_like(hidden_states), conv_state.clone(),
                intermediate_conv_state)
    return torch.empty_like(hidden_states), conv_state.clone()


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::update_ssm_state
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::update_ssm_state", mutates_args=())
def update_ssm_state(
    hidden_states: torch.Tensor,  # [batch, seq_len, num_heads, head_dim]
    ssm_a: torch.Tensor,  # [num_heads] float32
    ssm_b: torch.Tensor,  # [batch, seq_len, n_groups, ssm_state_size]
    ssm_c: torch.Tensor,  # [batch, seq_len, n_groups, ssm_state_size]
    ssm_d: torch.Tensor,  # [num_heads] float16
    dt: torch.Tensor,  # [batch, seq_len, num_heads]
    dt_bias: torch.Tensor,  # [num_heads] float16
    state: torch.Tensor,  # [batch, num_heads, head_dim, ssm_state_size]
    context_lengths: torch.Tensor,  # [batch] int32
    dt_softplus: int,
    ngroups: int,
    chunk_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub: Mamba SSM state update. Returns zeros for hidden_states and cloned state."""
    return torch.zeros_like(hidden_states), state.clone()


@update_ssm_state.register_fake
def _(hidden_states,
      ssm_a,
      ssm_b,
      ssm_c,
      ssm_d,
      dt,
      dt_bias,
      state,
      context_lengths,
      dt_softplus,
      ngroups,
      chunk_size=0):
    return torch.empty_like(hidden_states), state.clone()


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::int4_moe_plugin  (sparse MoE with INT4 expert GEMMs)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::int4_moe_plugin", mutates_args=())
def int4_moe_plugin(
    router_logits: torch.
    Tensor,  # [B*S, E] float32 — gate output before softmax
    hidden_states: torch.Tensor,  # [B, S, H] float16
    fc_gate_up_qweights: torch.Tensor,  # [E, K//16, 2*I] Marlin int8
    fc_gate_up_scales: torch.Tensor,  # [E, num_groups, I] float16
    fc_down_qweights: torch.Tensor,  # [E, K//16, 2*D] Marlin int8
    fc_down_scales: torch.Tensor,  # [E, num_groups, D] float16
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    quantization_group_size: int,
) -> torch.Tensor:
    """Stub: fused sparse MoE (softmax + topk + expert INT4 grouped GEMMs).

    The gate GEMM (Linear) is traced separately as a standard MatMul.
    This op receives the router logits and performs softmax + topk routing,
    then dispatches tokens to experts for gate_up + SiLU + down projections
    using Marlin-packed INT4 weights.

    Mirrors ``trt_edgellm::Int4MoePlugin`` in tensorrt_edgellm.
    """
    batch_size, seq_len, _ = hidden_states.shape
    return torch.zeros(batch_size,
                       seq_len,
                       hidden_size,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


@int4_moe_plugin.register_fake
def _(router_logits, hidden_states, fc_gate_up_qweights, fc_gate_up_scales,
      fc_down_qweights, fc_down_scales, num_experts, top_k, hidden_size,
      moe_inter_size, activation_type, quantization_group_size):
    batch_size, seq_len, _ = hidden_states.shape
    return torch.empty(batch_size,
                       seq_len,
                       hidden_size,
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


# ---------------------------------------------------------------------------
# Custom op: trt::gather_nd  (token selection: GatherND with batch_dims=1)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::gather_nd", mutates_args=())
def gather_nd(
        value: torch.Tensor,  # [batch, seq_len, hidden_size] float16
        indices: torch.Tensor,  # [batch, num_tokens] int64
) -> torch.Tensor:
    """Stub: gather tokens from seq_len dim. Exports as GatherND(batch_dims=1).

    Equivalent to ``value[b, indices[b, t], :]`` for each batch b and
    token position t.  Exports as GatherND(batch_dims=1).
    """
    batch_size, num_tokens = indices.shape
    return torch.zeros(batch_size,
                       num_tokens,
                       value.shape[-1],
                       dtype=value.dtype,
                       device=value.device)


@gather_nd.register_fake
def _(value, indices):
    batch_size, num_tokens = indices.shape
    hidden_size = value.shape[-1]
    return torch.empty(batch_size,
                       num_tokens,
                       hidden_size,
                       dtype=value.dtype,
                       device=value.device)


# ---------------------------------------------------------------------------
# Custom op: trt::rope_onnx  (TRT native RotaryEmbedding)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::rope_onnx", mutates_args=())
def rope_onnx(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Stub for TRT native RotaryEmbedding — returns tensor with same shape as input."""
    return x.clone()


@rope_onnx.register_fake
def _(x, cos, sin, position_ids):
    return torch.empty_like(x)


# Custom op: trt::kv_cache_update_onnx  (TRT native KVCacheUpdate)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::kv_cache_update_onnx", mutates_args=())
def kv_cache_update_onnx(
    cache: torch.Tensor,
    new_kv: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Stub for TRT native KVCacheUpdate — returns cache with same shape."""
    return cache.clone()


@kv_cache_update_onnx.register_fake
def _(cache, new_kv, cache_indices):
    return torch.empty_like(cache)


# ---------------------------------------------------------------------------
# Custom op: trt::attention_onnx  (TRT native Attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt::attention_onnx", mutates_args=())
def attention_onnx(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: float,
) -> torch.Tensor:
    """Stub for TRT native Attention — returns tensor with same shape as query."""
    return query.clone()


@attention_onnx.register_fake
def _(query, key, value, attn_mask, is_causal, scale):
    return torch.empty_like(query)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::gated_delta_net  (Qwen3.5 GDN linear attention)
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::gated_delta_net", mutates_args=())
def gated_delta_net(
    q: torch.Tensor,  # [batch, seq, num_k_heads, k_dim]
    k: torch.Tensor,  # [batch, seq, num_k_heads, k_dim]
    v: torch.Tensor,  # [batch, seq, num_v_heads, v_dim]
    a: torch.Tensor,  # [batch, seq, num_v_heads]
    b: torch.Tensor,  # [batch, seq, num_v_heads]
    A_log: torch.Tensor,  # [num_v_heads] float32
    dt_bias: torch.Tensor,  # [num_v_heads] float16
    h0_source: torch.Tensor,  # [batch, num_v_heads, k_dim, v_dim] float32
    context_lengths: torch.Tensor,  # [batch] int32
    k_dim: int,
    v_dim: int,
    collect_intermediate_states: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stub: GatedDeltaNet; with MTP enabled it returns an optional 3rd output."""
    if collect_intermediate_states:
        batch_size, seq_len, num_v_heads, _ = v.shape
        intermediate_recurrent_state = torch.zeros(batch_size,
                                                   seq_len,
                                                   num_v_heads,
                                                   k_dim,
                                                   v_dim,
                                                   dtype=h0_source.dtype,
                                                   device=h0_source.device)
        return (torch.zeros_like(v), h0_source.clone(),
                intermediate_recurrent_state)
    return torch.zeros_like(v), h0_source.clone()


@gated_delta_net.register_fake
def _(q,
      k,
      v,
      a,
      b,
      A_log,
      dt_bias,
      h0_source,
      context_lengths,
      k_dim,
      v_dim,
      collect_intermediate_states=False):
    if collect_intermediate_states:
        batch_size, seq_len, num_v_heads, _ = v.shape
        intermediate_recurrent_state = torch.empty(batch_size,
                                                   seq_len,
                                                   num_v_heads,
                                                   k_dim,
                                                   v_dim,
                                                   dtype=h0_source.dtype,
                                                   device=h0_source.device)
        return (torch.empty_like(v), h0_source.clone(),
                intermediate_recurrent_state)
    return torch.empty_like(v), h0_source.clone()


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::Nvfp4MoePlugin
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::Nvfp4MoePlugin", mutates_args=())
def nvfp4_moe_plugin(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_global_scale: torch.Tensor,
    up_weights: torch.Tensor,
    up_block_scale: torch.Tensor,
    up_global_scale: torch.Tensor,
    down_weights: torch.Tensor,
    down_block_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    up_block_scale_decode: torch.Tensor,
    down_block_scale_decode: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: int,
    routed_scaling_factor: float,
    routing_mode: int,
) -> torch.Tensor:
    return torch.zeros_like(hidden_states)


@nvfp4_moe_plugin.register_fake
def _(router_logits, hidden_states, hidden_global_scale, up_weights,
      up_block_scale, up_global_scale, down_weights, down_block_scale,
      down_global_scale, up_block_scale_decode, down_block_scale_decode,
      e_score_correction_bias, num_experts, top_k, hidden_size, moe_inter_size,
      activation_type, n_group, topk_group, norm_topk_prob,
      routed_scaling_factor, routing_mode):
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------------------
# Custom op: trt_edgellm::NvFP4MoEPluginGeforce
# ---------------------------------------------------------------------------


@torch.library.custom_op("trt_edgellm::NvFP4MoEPluginGeforce", mutates_args=())
def nvfp4_moe_plugin_geforce(
    router_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_qweights: torch.Tensor,
    fc1_blocks_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_qweights: torch.Tensor,
    fc2_blocks_scale: torch.Tensor,
    fc2_alpha: torch.Tensor,
    input_global_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    backend: int,
    io_dtype: int,
    max_routed_rows: int,
) -> torch.Tensor:
    return torch.zeros_like(hidden_states)


@nvfp4_moe_plugin_geforce.register_fake
def _(router_logits, hidden_states, fc1_qweights, fc1_blocks_scale, fc1_alpha,
      fc2_qweights, fc2_blocks_scale, fc2_alpha, input_global_scale,
      down_input_scale, num_experts, top_k, hidden_size, moe_inter_size,
      activation_type, backend, io_dtype, max_routed_rows):
    return torch.empty_like(hidden_states)
