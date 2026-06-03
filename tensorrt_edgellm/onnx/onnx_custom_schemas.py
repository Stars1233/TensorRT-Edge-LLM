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
Register ONNX OpSchemas for custom ops used by dynamo export in this package.


Call :func:`register_tensorrt_edgellm_onnx_custom_schemas` before
``torch.onnx.export(..., optimize=True)`` so custom nodes resolve.

Custom op ``since_version`` (ONNX opset) is fixed below; bump if the runtime
expects a newer schema revision. FP8 here uses standard ``QuantizeLinear`` /
``DequantizeLinear``, not custom FP8 schemas.
"""

from __future__ import annotations

import onnx
from onnx.defs import OpSchema

_SCHEMA_SINCE_VERSION = 23


def _safe_register_schema(schema: OpSchema) -> None:
    """Register *schema* if not already present (idempotent across packages)."""
    try:
        onnx.defs.register_schema(schema)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# trt_edgellm::AttentionPlugin, trt_edgellm::ViTAttentionPlugin
# ---------------------------------------------------------------------------

_attention_plugin_schema = OpSchema(
    name="AttentionPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    "Custom TensorRT attention plugin with RoPE, KV cache, and attention computation.",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="past_key_value",
            description="KV cache tensor",
            type_str="T_KV",
        ),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Context length tensor",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="rope_rotary_cos_sin",
            description="RoPE rotary embeddings (FP32)",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="kvcache_start_index",
            description=
            "KV cache start index tensor of shape [kv_cache_start_batch_size]",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="attention_mask",
            description="Attention mask tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
        OpSchema.FormalParameter(
            name="attention_pos_id",
            description="Position IDs tensor (optional)",
            type_str="tensor(int32)",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="present_key_value",
            description="Updated KV cache tensor",
            type_str="T_KV",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "Input Q/K/V data type.",
        ),
        (
            "T_KV",
            ["tensor(float16)", "tensor(float8e4m3fn)"],
            "KV cache data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_q_heads",
            type=OpSchema.AttrType.INT,
            description="Number of query heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="num_kv_heads",
            type=OpSchema.AttrType.INT,
            description="Number of key-value heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_tree_attention",
            type=OpSchema.AttrType.INT,
            description="Whether to enable tree attention (0(false), 1(true))",
            required=True,
        ),
        OpSchema.Attribute(
            name="enable_fp8_kv_cache",
            type=OpSchema.AttrType.INT,
            description=
            "Whether to use FP8 KV cache (0(false), 1(true)). Optional.",
            required=False,
        ),
        OpSchema.Attribute(
            name="sliding_window_size",
            type=OpSchema.AttrType.INT,
            description=
            "Sliding window size for attention (-1 = none, >0 = window size).",
            required=False,
        ),
        OpSchema.Attribute(
            name="qkv_scales",
            type=OpSchema.AttrType.FLOATS,
            description=
            "QKV dequant scales for FP8 KV cache: [q_scale, k_scale, v_scale]. "
            "Defaults to [1.0, 1.0, 1.0] when the checkpoint has no explicit scales.",
            required=False,
        ),
    ],
)

_vit_attention_plugin_schema = OpSchema(
    name="ViTAttentionPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    "Custom TensorRT ViT attention plugin (separate Q/K/V, no KV cache, no RoPE).",
    inputs=[
        OpSchema.FormalParameter(
            name="q",
            description="Query tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="k",
            description="Key tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="v",
            description="Value tensor [total_S, H, D]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cu_seqlens",
            description="Prefix sum of sequence lengths (int32, shape [B+1])",
            type_str="tensor(int32)",
        ),
        OpSchema.FormalParameter(
            name="max_seqlen_carrier",
            description="Shape-only carrier for max sequence length hint",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output [total_S, H, D]",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)"],
            "Input Q/K/V data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_heads",
            type=OpSchema.AttrType.INT,
            description="Number of attention heads",
            required=True,
        ),
        OpSchema.Attribute(
            name="head_size",
            type=OpSchema.AttrType.INT,
            description="Size of each attention head",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# NVFP4 (trt domain)
# ---------------------------------------------------------------------------

_trt_fp4_dynamic_quantize_schema = OpSchema(
    name="TRT_FP4DynamicQuantize",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    ("Dynamically quantize float16 activations to NVFP4 with per-block scaling."
     ),
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Input activation tensor (float16)",
            type_str="tensor(float16)",
        ),
        OpSchema.FormalParameter(
            name="global_scale",
            description="Global scale constant (float16 scalar)",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="x_fp4",
            description="Packed NVFP4 tensor",
            type_str="T_fp4",
        ),
        OpSchema.FormalParameter(
            name="x_sf",
            description="Per-block scale factors (packed)",
            type_str="T_fp4",
        ),
    ],
    type_constraints=[
        (
            "T_fp4",
            ["tensor(uint8)", "tensor(int8)"],
            "Packed FP4 tensor representation.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Quantization axis (default -1)",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Elements per block (typically 16)",
            required=True,
        ),
        OpSchema.Attribute(
            name="scale_type",
            type=OpSchema.AttrType.INT,
            description="ONNX elem_type for scale factors (17 = FLOAT8E4M3FN)",
            required=True,
        ),
    ],
)

_trt_dequantize_linear_schema = OpSchema(
    name="DequantizeLinear",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT-domain DequantizeLinear for NVFP4 per-block dequantization.",
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Quantized input",
            type_str="T_q",
        ),
        OpSchema.FormalParameter(
            name="x_scale",
            description="Per-block scale factors",
            type_str="T_scale",
        ),
        OpSchema.FormalParameter(
            name="x_zero_point",
            description="Zero-point (optional)",
            type_str="T_q",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="y",
            description="Dequantized output",
            type_str="T_out",
        ),
    ],
    type_constraints=[
        (
            "T_q",
            ["tensor(uint8)", "tensor(int8)", "tensor(float8e4m3fn)"],
            "Quantized tensor type.",
        ),
        (
            "T_scale",
            ["tensor(float16)", "tensor(float)", "tensor(float8e4m3fn)"],
            "Scale tensor type.",
        ),
        (
            "T_out",
            ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"],
            "Dequantized output type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Dequantization axis",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Block size (NVFP4 uses 16)",
            required=False,
        ),
    ],
)

# ---------------------------------------------------------------------------
# MXFP8 (trt domain)
# ---------------------------------------------------------------------------

_trt_mxfp8_dynamic_quantize_schema = OpSchema(
    name="TRT_MXFP8DynamicQuantize",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("Dynamically quantize float16 activations to MXFP8 with per-block "
         "E8M0 scaling."),
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Input activation tensor (float16)",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="x_fp8",
            description="FP8E4M3 quantized tensor",
            type_str="T_fp8",
        ),
        OpSchema.FormalParameter(
            name="x_scale",
            description="E8M0 per-block scale factors (uint8)",
            type_str="T_scale",
        ),
    ],
    type_constraints=[
        (
            "T_fp8",
            ["tensor(float8e4m3fn)", "tensor(uint8)"],
            "FP8 quantized tensor.",
        ),
        (
            "T_scale",
            ["tensor(uint8)"],
            "E8M0 scale tensor (UINT8).",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Quantization axis (default -1)",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Elements per block (typically 32)",
            required=True,
        ),
        OpSchema.Attribute(
            name="output_dtype",
            type=OpSchema.AttrType.INT,
            description="ONNX elem_type for output (17 = FLOAT8E4M3FN)",
            required=True,
        ),
    ],
)

_trt_mxfp8_dequantize_linear_schema = OpSchema(
    name="TRT_MXFP8DequantizeLinear",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT-domain MXFP8 per-block dequantization with E8M0 scales.",
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Quantized input (FP8E4M3)",
            type_str="T_q",
        ),
        OpSchema.FormalParameter(
            name="x_scale",
            description="E8M0 per-block scale factors (uint8)",
            type_str="T_scale",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="y",
            description="Dequantized output",
            type_str="T_out",
        ),
    ],
    type_constraints=[
        (
            "T_q",
            ["tensor(float8e4m3fn)", "tensor(uint8)"],
            "Quantized tensor type.",
        ),
        (
            "T_scale",
            ["tensor(uint8)"],
            "E8M0 scale tensor type.",
        ),
        (
            "T_out",
            ["tensor(float16)", "tensor(float)"],
            "Dequantized output type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="axis",
            type=OpSchema.AttrType.INT,
            description="Dequantization axis",
            required=False,
        ),
        OpSchema.Attribute(
            name="block_size",
            type=OpSchema.AttrType.INT,
            description="Block size (MXFP8 uses 32)",
            required=False,
        ),
        OpSchema.Attribute(
            name="output_dtype",
            type=OpSchema.AttrType.INT,
            description="ONNX elem_type for output (10 = FLOAT16)",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Int4GroupwiseGemmPlugin
# ---------------------------------------------------------------------------

_int4_groupwise_gemm_schema = OpSchema(
    name="Int4GroupwiseGemmPlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TensorRT Int4 groupwise GEMM plugin.",
    inputs=[
        OpSchema.FormalParameter(
            name="input",
            description="Input tensor",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="qweight",
            description="Quantized weights (int8)",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="scales",
            description="Scales (float16)",
            type_str="tensor(float16)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="gemm_n",
            type=OpSchema.AttrType.INT,
            description="Output feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="gemm_k",
            type=OpSchema.AttrType.INT,
            description="Input feature dimension",
            required=True,
        ),
        OpSchema.Attribute(
            name="group_size",
            type=OpSchema.AttrType.INT,
            description="Group size",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::causal_conv1d, update_ssm_state
# ---------------------------------------------------------------------------

_causal_conv1d_schema = OpSchema(
    name="causal_conv1d",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Causal 1D depthwise convolution with persistent state.",
    inputs=[
        OpSchema.FormalParameter(name="x",
                                 description="Input tensor",
                                 type_str="T"),
        OpSchema.FormalParameter(name="weight",
                                 description="Conv weight",
                                 type_str="T"),
        OpSchema.FormalParameter(name="bias",
                                 description="Conv bias",
                                 type_str="T"),
        OpSchema.FormalParameter(name="conv_state",
                                 description="Conv state",
                                 type_str="T"),
        OpSchema.FormalParameter(name="context_lengths",
                                 description="Context lengths per batch",
                                 type_str="T_CL"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output",
                                 description="Conv output",
                                 type_str="T"),
        OpSchema.FormalParameter(name="conv_state_out",
                                 description="Updated conv state",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="intermediate_conv_state_out",
            description="Per-token conv states [batch, seq, dim, width]",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="stride",
                           type=OpSchema.AttrType.INT,
                           description="Stride",
                           required=True),
        OpSchema.Attribute(name="padding",
                           type=OpSchema.AttrType.INT,
                           description="Padding",
                           required=True),
        OpSchema.Attribute(name="dilation",
                           type=OpSchema.AttrType.INT,
                           description="Dilation",
                           required=True),
        OpSchema.Attribute(name="groups",
                           type=OpSchema.AttrType.INT,
                           description="Groups",
                           required=True),
        OpSchema.Attribute(
            name="use_mtp",
            type=OpSchema.AttrType.INT,
            description="Whether to emit per-token intermediate states",
            required=False),
    ],
)

_update_ssm_state_schema = OpSchema(
    name="update_ssm_state",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Mamba selective SSM state update.",
    inputs=[
        OpSchema.FormalParameter(name="x", description="Input x",
                                 type_str="T"),
        OpSchema.FormalParameter(name="A",
                                 description="A parameter",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="B",
                                 description="B parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="C",
                                 description="C parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="D",
                                 description="D parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="dt",
                                 description="dt parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="dt_bias",
                                 description="dt_bias parameter",
                                 type_str="T"),
        OpSchema.FormalParameter(name="state",
                                 description="SSM state",
                                 type_str="T"),
        OpSchema.FormalParameter(name="context_lengths",
                                 description="Context lengths per batch",
                                 type_str="T_CL"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="output",
                                 description="SSM output",
                                 type_str="T"),
        OpSchema.FormalParameter(name="state_out",
                                 description="Updated SSM state",
                                 type_str="T"),
    ],
    type_constraints=[
        ("T", ["tensor(float16)", "tensor(bfloat16)", "tensor(float)"], ""),
        ("T_A", ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="dt_softplus",
                           type=OpSchema.AttrType.INT,
                           description="Apply softplus to dt",
                           required=True),
        OpSchema.Attribute(name="ngroups",
                           type=OpSchema.AttrType.INT,
                           description="Number of groups",
                           required=True),
        OpSchema.Attribute(
            name="chunk_size",
            type=OpSchema.AttrType.INT,
            description="Mamba2 prefill chunk size (0=sequential)",
            required=False),
        OpSchema.Attribute(name="time_step_limit",
                           type=OpSchema.AttrType.FLOATS,
                           description="Time step clamping range",
                           required=False),
    ],
)

# ---------------------------------------------------------------------------
# TRT native attention ops (domain "" — default ONNX domain)
# RotaryEmbedding, TensorScatter, Attention are consumed by TRT >= 10.15
# without any custom plugin; they live in the default ONNX domain.
# ---------------------------------------------------------------------------

_rotary_embedding_schema = OpSchema(
    name="RotaryEmbedding",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT native rotary position embedding applied to Q or K tensors.",
    inputs=[
        OpSchema.FormalParameter(
            name="x",
            description="Input tensor [batch, heads, seq, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cos",
            description="Cosine table [max_pos, head_dim//2]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="sin",
            description="Sine table [max_pos, head_dim//2]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="position_ids",
            description="Position indices [batch, seq] (int32)",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="RoPE-embedded tensor, same shape as x",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
)

_tensor_scatter_schema = OpSchema(
    name="TensorScatter",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=
    "TRT native KV cache scatter update: writes new_kv into cache at cache_indices.",
    inputs=[
        OpSchema.FormalParameter(
            name="cache",
            description="KV cache tensor [batch, kv_heads, capacity, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="new_kv",
            description="New KV values [batch, kv_heads, seq, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="cache_indices",
            description="Write offset per batch item [batch] (int32)",
            type_str="tensor(int32)",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="updated_cache",
            description="Updated KV cache, same shape as cache",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Cache and new_kv data type.",
        ),
    ],
)

_attention_trt_native_schema = OpSchema(
    name="Attention",
    domain="trt",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="TRT native scaled dot-product attention (TRT_decomposable=1).",
    inputs=[
        OpSchema.FormalParameter(
            name="query",
            description="Query tensor [batch, heads, seq_q, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="key",
            description="Key tensor [batch, kv_heads, seq_k, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="value",
            description="Value tensor [batch, kv_heads, seq_k, head_dim]",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="attn_mask",
            description=
            "Additive attention mask [1, 1, seq_q, seq_k] (optional)",
            type_str="T",
            param_option=OpSchema.FormalParameterOption.Optional,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="attn_output",
            description="Attention output, same shape as query",
            type_str="T",
        ),
    ],
    type_constraints=[
        (
            "T",
            ["tensor(float16)", "tensor(float)", "tensor(bfloat16)"],
            "Input and output data type.",
        ),
    ],
    attributes=[
        OpSchema.Attribute(
            name="is_causal",
            type=OpSchema.AttrType.INT,
            description="Whether to apply a causal mask (0=false, 1=true)",
            required=False,
        ),
        OpSchema.Attribute(
            name="TRT_decomposable",
            type=OpSchema.AttrType.INT,
            description="Allow TRT to decompose this node (always 1)",
            required=False,
        ),
        OpSchema.Attribute(
            name="scale",
            type=OpSchema.AttrType.FLOAT,
            description="Attention scale factor (1.0 when Q is pre-scaled)",
            required=False,
        ),
    ],
)

_gated_delta_net_schema = OpSchema(
    name="gated_delta_net",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc="Qwen3.5 GatedDeltaNet linear attention plugin.",
    inputs=[
        OpSchema.FormalParameter(name="q",
                                 description="Query [n, seq, h, k]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="k",
                                 description="Key [n, seq, h, k]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="v",
                                 description="Value [n, seq, hv, v]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="a",
                                 description="A gating tensor [n, seq, hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="b",
                                 description="B gating tensor [n, seq, hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(name="A_log",
                                 description="A_log [hv]",
                                 type_str="T_A"),
        OpSchema.FormalParameter(name="dt_bias",
                                 description="dt_bias [hv]",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="h0_source",
            description="Recurrent state in [n, hv, k, v]",
            type_str="T_A"),
        OpSchema.FormalParameter(
            name="context_lengths",
            description="Valid token count per batch row [n]",
            type_str="T_CL"),
    ],
    outputs=[
        OpSchema.FormalParameter(name="o",
                                 description="Output [n, seq, hv, v]",
                                 type_str="T"),
        OpSchema.FormalParameter(
            name="h0_out",
            description="Recurrent state out [n, hv, k, v]",
            type_str="T_A"),
        OpSchema.FormalParameter(
            name="intermediate_h0_out",
            description="Per-token recurrent states [n, seq, hv, k, v]",
            type_str="T_A",
            param_option=OpSchema.FormalParameterOption.Optional),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], ""),
        ("T_A", ["tensor(float)"], ""),
        ("T_CL", ["tensor(int32)"], ""),
    ],
    attributes=[
        OpSchema.Attribute(name="k_dim",
                           type=OpSchema.AttrType.INT,
                           description="K head dimension",
                           required=True),
        OpSchema.Attribute(name="v_dim",
                           type=OpSchema.AttrType.INT,
                           description="V head dimension",
                           required=True),
        OpSchema.Attribute(
            name="use_mtp",
            type=OpSchema.AttrType.INT,
            description="Whether to emit per-token intermediate states",
            required=False),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Int4MoePlugin
# ---------------------------------------------------------------------------

_int4_moe_plugin_schema = OpSchema(
    name="Int4MoePlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("Fused sparse MoE with INT4 expert GEMMs (Marlin layout).  "
         "Receives router_logits (FP32, from traced gate GEMM) and "
         "hidden_states; performs softmax + topk routing, then per-expert "
         "grouped GEMM (gate_up + SiLU + down) with weighted combine.  "
         "Mirrors trt_edgellm::Int4MoePlugin in tensorrt_edgellm."),
    inputs=[
        OpSchema.FormalParameter(
            name="router_logits",
            description=
            "Router logits (B*S, E) FP32, from gate GEMM + cast, before softmax",
            type_str="tensor(float)",
        ),
        OpSchema.FormalParameter(
            name="hidden_states",
            description="Input hidden states (B, S, D)",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="fc_gate_up_qweights",
            description=
            "Fused gate+up proj Marlin-packed weights (E, K//16, 2*I) int8",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_gate_up_scales",
            description="Fused gate+up proj scales (E, num_groups, I)",
            type_str="T",
        ),
        OpSchema.FormalParameter(
            name="fc_down_qweights",
            description="Down proj Marlin-packed weights (E, K//16, 2*D) int8",
            type_str="tensor(int8)",
        ),
        OpSchema.FormalParameter(
            name="fc_down_scales",
            description="Down proj scales (E, num_groups, D)",
            type_str="T",
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            description="Output tensor (B, S, D)",
            type_str="T",
        ),
    ],
    type_constraints=[
        ("T", ["tensor(float16)"], "FP16 data type."),
    ],
    attributes=[
        OpSchema.Attribute(
            name="num_experts",
            type=OpSchema.AttrType.INT,
            description="Number of experts",
            required=True,
        ),
        OpSchema.Attribute(
            name="top_k",
            type=OpSchema.AttrType.INT,
            description="Top K experts per token",
            required=True,
        ),
        OpSchema.Attribute(
            name="hidden_size",
            type=OpSchema.AttrType.INT,
            description="Hidden size D",
            required=True,
        ),
        OpSchema.Attribute(
            name="moe_inter_size",
            type=OpSchema.AttrType.INT,
            description="MoE intermediate size I",
            required=True,
        ),
        OpSchema.Attribute(
            name="activation_type",
            type=OpSchema.AttrType.INT,
            description="Activation function type (0=SiLU)",
            required=True,
        ),
        OpSchema.Attribute(
            name="quantization_group_size",
            type=OpSchema.AttrType.INT,
            description="Quantization group size G",
            required=True,
        ),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::Nvfp4MoePlugin
# ---------------------------------------------------------------------------

_nvfp4_moe_plugin_schema = OpSchema(
    name="Nvfp4MoePlugin",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("NVFP4 MoE plugin: FP16 hidden + NVFP4 N-major weights. "
         "Dispatches decode or prefill"),
    inputs=[
        OpSchema.FormalParameter("router_logits", "T1",
                                 "Router logits (B*S, E)"),
        OpSchema.FormalParameter("hidden_states", "T2",
                                 "Hidden states (B, S, H)"),
        OpSchema.FormalParameter("hidden_global_scale", "T1",
                                 "Activation global scales (2,) [FC1, FC2]"),
        OpSchema.FormalParameter("up_weights", "T3", "Up weights (E, H, I/2)"),
        OpSchema.FormalParameter(
            "up_block_scale", "T3",
            "Up prefill block scales (E, padUp(I,128), padUp(H/16,4))"),
        OpSchema.FormalParameter("up_global_scale", "T1",
                                 "Up global scales (E,)"),
        OpSchema.FormalParameter("down_weights", "T3",
                                 "Down weights (E, I, H/2)"),
        OpSchema.FormalParameter(
            "down_block_scale", "T3",
            "Down prefill block scales (E, padUp(H,128), padUp(I/16,4))"),
        OpSchema.FormalParameter("down_global_scale", "T1",
                                 "Down global scales (E,)"),
        OpSchema.FormalParameter("up_block_scale_decode", "T3",
                                 "Up decode block scales (E, H/16, I)"),
        OpSchema.FormalParameter("down_block_scale_decode", "T3",
                                 "Down decode block scales (E, I/16, H)"),
        OpSchema.FormalParameter("e_score_correction_bias", "T1",
                                 "Per-expert load-balancing bias (E,)"),
    ],
    outputs=[
        OpSchema.FormalParameter("output", "T2", "Output [B, S, H] FP16"),
    ],
    type_constraints=[
        ("T1", ["tensor(float)"], "FP32"),
        ("T2", ["tensor(float16)"], "FP16"),
        ("T3", ["tensor(int8)"], "INT8"),
    ],
    attributes=[
        OpSchema.Attribute("num_experts", OpSchema.AttrType.INT),
        OpSchema.Attribute("top_k", OpSchema.AttrType.INT),
        OpSchema.Attribute("hidden_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("moe_inter_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("activation_type", OpSchema.AttrType.INT),
        OpSchema.Attribute("n_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("topk_group", OpSchema.AttrType.INT),
        OpSchema.Attribute("norm_topk_prob", OpSchema.AttrType.INT),
        OpSchema.Attribute("routed_scaling_factor", OpSchema.AttrType.FLOAT),
        OpSchema.Attribute("routing_mode", OpSchema.AttrType.INT),
    ],
)

# ---------------------------------------------------------------------------
# trt_edgellm::NvFP4MoEPluginGeforce
# ---------------------------------------------------------------------------

_nvfp4_moe_plugin_geforce_schema = OpSchema(
    name="NvFP4MoEPluginGeforce",
    domain="trt_edgellm",
    since_version=_SCHEMA_SINCE_VERSION,
    doc=("GeForce CuTeDSL NVFP4 MoE plugin: FP16 hidden states, FP4 expert "
         "weights, and FP8 block scales in 6D MMA layout."),
    inputs=[
        OpSchema.FormalParameter("router_logits", "T_ROUTER",
                                 "Router logits [B*S, E] FP32"),
        OpSchema.FormalParameter("hidden_states", "T_HIDDEN",
                                 "Hidden states [B, S, H] FP16"),
        OpSchema.FormalParameter("fc1_qweights", "T_INT8",
                                 "FC1 weights [E, N1, H/2] INT8"),
        OpSchema.FormalParameter(
            "fc1_blocks_scale", "T_INT8",
            "FC1 block scales [E, m_tiles, k_tiles, 32, 4, 4] INT8"),
        OpSchema.FormalParameter("fc1_alpha", "T_ROUTER",
                                 "FC1 global weight scales [E] FP32"),
        OpSchema.FormalParameter("fc2_qweights", "T_INT8",
                                 "FC2 weights [E, H, I/2] INT8"),
        OpSchema.FormalParameter(
            "fc2_blocks_scale", "T_INT8",
            "FC2 block scales [E, m_tiles, k_tiles, 32, 4, 4] INT8"),
        OpSchema.FormalParameter("fc2_alpha", "T_ROUTER",
                                 "FC2 global weight scales [E] FP32"),
        OpSchema.FormalParameter("input_global_scale", "T_ROUTER",
                                 "FC1 activation scales [E] FP32"),
        OpSchema.FormalParameter("down_input_scale", "T_ROUTER",
                                 "FC2 activation scales [E] FP32"),
    ],
    outputs=[
        OpSchema.FormalParameter("output", "T_HIDDEN",
                                 "Output [B, S, H] FP16"),
    ],
    type_constraints=[
        ("T_ROUTER", ["tensor(float)"], "FP32 tensors"),
        ("T_HIDDEN", ["tensor(float16)"], "FP16 tensors"),
        ("T_INT8", ["tensor(int8)"], "INT8 byte tensors"),
    ],
    attributes=[
        OpSchema.Attribute("num_experts", OpSchema.AttrType.INT),
        OpSchema.Attribute("top_k", OpSchema.AttrType.INT),
        OpSchema.Attribute("hidden_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("moe_inter_size", OpSchema.AttrType.INT),
        OpSchema.Attribute("activation_type", OpSchema.AttrType.INT),
        OpSchema.Attribute("backend", OpSchema.AttrType.INT),
        OpSchema.Attribute("io_dtype", OpSchema.AttrType.INT),
        OpSchema.Attribute("max_routed_rows", OpSchema.AttrType.INT),
    ],
)

_ALL_CUSTOM_SCHEMAS: tuple[OpSchema, ...] = (
    _attention_plugin_schema,
    _vit_attention_plugin_schema,
    _trt_fp4_dynamic_quantize_schema,
    _trt_dequantize_linear_schema,
    _trt_mxfp8_dynamic_quantize_schema,
    _trt_mxfp8_dequantize_linear_schema,
    _int4_groupwise_gemm_schema,
    _causal_conv1d_schema,
    _update_ssm_state_schema,
    _rotary_embedding_schema,
    _tensor_scatter_schema,
    _attention_trt_native_schema,
    _gated_delta_net_schema,
    _int4_moe_plugin_schema,
    _nvfp4_moe_plugin_schema,
    _nvfp4_moe_plugin_geforce_schema,
)

_registered_tensorrt_edgellm_schemas: bool = False


def register_tensorrt_edgellm_onnx_custom_schemas() -> None:
    """Register custom ONNX ops for :mod:`tensorrt_edgellm.dynamo_translations`. Idempotent."""
    global _registered_tensorrt_edgellm_schemas
    if _registered_tensorrt_edgellm_schemas:
        return
    for s in _ALL_CUSTOM_SCHEMAS:
        _safe_register_schema(s)
    _registered_tensorrt_edgellm_schemas = True
