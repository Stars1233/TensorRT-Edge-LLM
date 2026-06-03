# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Quantized linear modules: checkpoint buffer names match safetensors keys.


Forwards call :mod:`ops` stubs; ONNX export maps them via registered symbolics.

Layouts (buffers): FP8 — ``weight`` ``[out,in]`` fp8, scales fp16; NVFP4 —
packed uint8 weights and fp8 block scales; AWQ — ``qweight`` ``[in,out//8]``,
``qzeros``, ``scales``; GPTQ — ``qweight`` ``[in//8,out]``, ``qzeros``,
``scales``, ``g_idx``; W4A16 prepacked — ``weight`` ``[out//2,in]`` uint8,
``weight_scale``, optional ``pre_quant_scale``; FP16 — ``weight`` ``[out,in]``.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (QUANT_FP8, QUANT_FP16, QUANT_INT4_AWQ,
                      QUANT_INT4_AWQ_MODELOPT, QUANT_INT4_GPTQ, QUANT_INT8_SQ,
                      QUANT_MXFP8, QUANT_NVFP4, ModelConfig, module_quant_type)
from .ops import (fp8_dequantize, fp8_quantize, int4_groupwise_gemm,
                  int8_sq_act_qdq, int8_sq_weight_dq, mxfp8_act_qdq,
                  mxfp8_weight_dq, nvfp4_act_qdq, nvfp4_dequantize)

logger = logging.getLogger(__name__)


def _require_fp16_input(hidden_states: torch.Tensor, layer_name: str) -> None:
    if hidden_states.dtype != torch.float16:
        raise TypeError(
            f"{layer_name} expects float16 input, got {hidden_states.dtype}")


__all__ = [
    "FP16Linear",
    "FP8Linear",
    "MXFP8Linear",
    "NVFP4Linear",
    "AWQLinear",
    "ModelOptAWQPrepackedLinear",
    "GPTQLinear",
    "INT8SQLinear",
    "make_linear",
]

# ---------------------------------------------------------------------------
# FP16Linear
# ---------------------------------------------------------------------------


class FP16Linear(nn.Module):
    """Plain float16 linear (embed_tokens, lm_head, non-quantised).

    Activations must be float16 (no bfloat16/float32 inputs for now).
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features,
                                               in_features,
                                               dtype=torch.float16),
                                   requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features,
                                                 dtype=torch.float16),
                                     requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "FP16Linear")
        bias = self.bias if self.bias is not None else None
        return F.linear(hidden_states, self.weight, bias)


# ---------------------------------------------------------------------------
# FP8Linear
# ---------------------------------------------------------------------------


class FP8Linear(nn.Module):
    """FP8 E4M3 linear.

    Export emits standard ONNX ``QuantizeLinear`` (``output_dtype=float8e4m3fn``)
    and ``DequantizeLinear`` with per-tensor **float16** scales, then ``MatMul``
    (via ``F.linear``).  NVFP4 is unchanged elsewhere in this package.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features))
        # Per-tensor scales declared as 0-dim scalars to match ModelOpt's
        # unified-checkpoint layout (``shape=[]``).  Avoids the rank mismatch
        # that previously forced visual ``_load_weights`` callers to reshape
        # source tensors with ``view(1)`` before assignment.
        self.register_buffer("weight_scale", torch.ones((),
                                                        dtype=torch.float16))
        self.register_buffer("input_scale", torch.ones((),
                                                       dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "FP8Linear")
        # Activation: ONNX QuantizeLinear -> FP8 + DequantizeLinear -> FP16
        hidden_states_q = fp8_quantize(hidden_states, self.input_scale)
        hidden_states_dq = fp8_dequantize(hidden_states_q, self.input_scale)
        # Weight: DQ FP8 -> FP16 (standard ONNX DequantizeLinear)
        w_fp16 = fp8_dequantize(self.weight, self.weight_scale)
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(hidden_states_dq, w_fp16, bias)


# ---------------------------------------------------------------------------
# NVFP4Linear
# ---------------------------------------------------------------------------


class NVFP4Linear(nn.Module):
    """NVFP4 E2M1 linear with FP8 per-group scales.

    Forward emits ``trt::DequantizeLinear(weight, weight_scale, weight_scale_2,
    block_size=group_size)`` (trt domain) followed by ``F.linear``.

    ``input_scale`` is stored for TRT's activation-quantisation pass; it is
    not emitted as an ONNX op here.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 16,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        num_groups = in_features // group_size

        # weight: packed fp4 stored as int8 (same bits as uint8, TRT needs int8)
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features // 2, dtype=torch.int8))
        # Per-group fp8 scale [out, num_groups]
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, num_groups, dtype=torch.float32))
        self.register_buffer("weight_scale_2", torch.ones(1))
        self.register_buffer("input_scale", torch.ones(1))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "NVFP4Linear")
        # Activation: DynQ + 2x trt::DQ -> float16 activations
        # input_scale is float32: amax / (6.0 * 448.0)
        hidden_states_dq = nvfp4_act_qdq(hidden_states, self.input_scale)
        # Weight: 2xstandard-ONNX DQ -> w_dq (float16)
        w_dq = nvfp4_dequantize(self.weight, self.weight_scale,
                                self.weight_scale_2, self.group_size)
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(hidden_states_dq, w_dq, bias)


# ---------------------------------------------------------------------------
# MXFP8Linear
# ---------------------------------------------------------------------------


class MXFP8Linear(nn.Module):
    """MXFP8 E4M3 linear with E8M0 per-block scales (block_size=32).

    Expects a **unified checkpoint** from ``modelopt.torch.export.export_hf_checkpoint``
    (ModelOpt >= 0.42.0) where weights are already FP8E4M3 and E8M0 scales
    are pre-computed.

    Activation is dynamically quantized at runtime via
    ``TRT_MXFP8DynamicQuantize`` -> ``TRT_MXFP8DequantizeLinear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        num_blocks = in_features // block_size

        # Weight: FP8E4M3 [out, in] — loaded from unified checkpoint.
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        # E8M0 scale: UINT8 [out, in // block_size]
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, num_blocks, dtype=torch.uint8))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "MXFP8Linear")
        # Activation: DynQ + DQ -> float16
        hidden_states_dq = mxfp8_act_qdq(hidden_states)
        # Weight: DQ FP8+E8M0 -> float16
        w_dq = mxfp8_weight_dq(self.weight, self.weight_scale, self.block_size)
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(hidden_states_dq, w_dq, bias)


# ---------------------------------------------------------------------------
# AWQLinear
# ---------------------------------------------------------------------------


class AWQLinear(nn.Module):
    """INT4 AWQ linear (column-packed checkpoint layout).

    Checkpoints use ``qweight`` ``[in, out//8]`` int32; :func:`loader.load_weights`
    repacks to ``[out//2, in]`` int8 for the custom int4 GEMM op.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        # Initialized in AWQ layout [in, out//8] int32; loader repacks to
        # [out//2, in] int8 (swizzled plugin layout) before inference/export.
        self.register_buffer(
            "qweight",
            torch.zeros(in_features, out_features // 8, dtype=torch.int32))
        self.register_buffer(
            "qzeros",
            torch.zeros(in_features // group_size,
                        out_features // 8,
                        dtype=torch.int32))
        self.register_buffer(
            "scales",
            torch.ones(in_features // group_size,
                       out_features,
                       dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "AWQLinear")
        out = int4_groupwise_gemm(
            hidden_states,
            self.qweight,
            self.scales.to(torch.float16),
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# ModelOptAWQPrepackedLinear
# ---------------------------------------------------------------------------


class ModelOptAWQPrepackedLinear(nn.Module):
    """W4A16 AWQ linear with prepacked uint8 weights.

    Checkpoint: ``weight`` ``[out//2, in]`` uint8, ``weight_scale`` ``[out, in//g]``
    float32, optional ``pre_quant_scale`` ``[in]`` float16. The loader repacks
    weights to int8 and transposes scales to ``[in//g, out]`` float16.

    Buffers: ``weight``, ``weight_scale``, ``pre_quant_scale`` (ones if absent).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        # Loaded from checkpoint as uint8; cast to int8 by the loader.
        self.register_buffer(
            "weight",
            torch.zeros(out_features // 2, in_features, dtype=torch.int8))
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features,
                       in_features // group_size,
                       dtype=torch.float16))
        # Optional per-input-channel AWQ smoothing scale.  The loader assigns
        # it from the checkpoint when present; the default of ones is a no-op.
        self.register_buffer("pre_quant_scale",
                             torch.ones(in_features, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "ModelOptAWQPrepackedLinear")
        hidden_states = hidden_states * self.pre_quant_scale
        out = int4_groupwise_gemm(
            hidden_states,
            self.weight,
            self.weight_scale,
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# GPTQLinear
# ---------------------------------------------------------------------------


class GPTQLinear(nn.Module):
    """GPTQ INT4 linear (symmetric or asymmetric, group-quantized) -- int4_gptq format.

    GPTQ packs 8 int4 nibbles per int32 along the *input* axis (row-packed),
    giving ``qweight [in//8, out]`` -- the opposite orientation from AWQ's
    ``[in, out//8]``.  After loading, :func:`repacking.repack_gptq_to_plugin`
    repacks to ``[out//2, in]`` int8 (the same plugin layout as AWQLinear).

    After load, ``int4_act_perm`` orders hidden dim for the int4 GEMM when
    ``desc_act`` reorders groups (otherwise identity).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        zero_point_offset: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.zero_point_offset = int(zero_point_offset)
        # Initialized in GPTQ layout [in//8, out] int32; loader repacks to
        # [out//2, in] int8 (swizzled plugin layout) before inference/export.
        self.register_buffer(
            "qweight",
            torch.zeros(in_features // 8, out_features, dtype=torch.int32))
        self.register_buffer(
            "qzeros",
            torch.zeros(in_features // group_size,
                        out_features // 8,
                        dtype=torch.int32))
        self.register_buffer(
            "scales",
            torch.ones(in_features // group_size,
                       out_features,
                       dtype=torch.float16))
        # g_idx maps each input channel to its group; used when repacking (``desc_act``).
        self.register_buffer(
            "g_idx",
            torch.arange(in_features, dtype=torch.int32) // group_size)
        self.register_buffer("int4_act_perm",
                             torch.arange(in_features, dtype=torch.int64))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "GPTQLinear")
        perm = self.int4_act_perm
        if perm is not None:
            perm_d = perm.to(device=hidden_states.device, dtype=torch.int64)
            hidden_states = hidden_states.index_select(-1, perm_d)
        out = int4_groupwise_gemm(
            hidden_states,
            self.qweight,
            self.scales.to(torch.float16),
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# INT8SQLinear
# ---------------------------------------------------------------------------


class INT8SQLinear(nn.Module):
    """INT8 SmoothQuant W8A8 linear (per-channel weight, per-tensor activation).

    Checkpoint layout (keys match exactly):
        weight          [out, in]   int8   -- symmetric per-channel INT8
        weight_scale    [out]       float32 -- per-channel dequant scale
        input_scale     []          float32 -- per-tensor activation scale
        pre_quant_scale [in]        float16 -- SmoothQuant activation smoother

    Forward ONNX pattern (TRT recognises Q-DQ-MatMul for INT8 GEMM fusion)::

        x_smooth = x16 * pre_quant_scale                      # Mul
        x_dq = QDQ(x_smooth, input_scale)                     # Q + DQ + Cast
        w_dq = DequantizeLinear(weight, weight_scale, axis=0)  # DQ + Cast
        output = F.linear(x_dq, w_dq)                         # MatMul
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale",
                             torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("pre_quant_scale",
                             torch.ones(in_features, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "INT8SQLinear")
        # SmoothQuant activation smoothing (Mul in ONNX)
        hidden_states_smooth = hidden_states * self.pre_quant_scale
        # Activation QDQ: QuantizeLinear + DequantizeLinear (per-tensor INT8)
        hidden_states_dq = int8_sq_act_qdq(hidden_states_smooth,
                                           self.input_scale)
        # Weight dequantize: DequantizeLinear (per-channel, axis=0)
        w_dq = int8_sq_weight_dq(self.weight, self.weight_scale)
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(hidden_states_dq, w_dq, bias)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_linear(
    config: ModelConfig,
    in_features: int,
    out_features: int,
    bias: bool = False,
    module_name: str = "",
) -> nn.Module:
    """Return the right linear layer class for *module_name* under *config*.

    Effective quant type is resolved by :func:`config.module_quant_type` —
    that helper encodes the full decision tree (``quant.excluded`` forces
    FP16; tied ``lm_head`` with no ``layer_overrides`` entry and an
    unquantized backbone is FP16; otherwise ``layer_overrides`` takes
    precedence over the dominant ``quant_type``).  Effective type is always
    a concrete quant string (``fp8``, ``nvfp4``, ...), never
    ``mixed_precision``.

    Args:
        config:        Model config (provides quant type and group_size).
        in_features:   Input feature dimension.
        out_features:  Output feature dimension.
        bias:          Include bias term.
        module_name:   Module path relative to the model root (e.g. ``"lm_head"``).
    """
    quant_type = module_quant_type(module_name, config)

    if quant_type == QUANT_FP16:
        return FP16Linear(in_features, out_features, bias)
    if quant_type == QUANT_FP8:
        return FP8Linear(in_features, out_features, bias)
    if quant_type == QUANT_MXFP8:
        return MXFP8Linear(in_features, out_features, config.quant.group_size,
                           bias)
    if quant_type == QUANT_NVFP4:
        return NVFP4Linear(in_features, out_features, config.quant.group_size,
                           bias)
    if quant_type == QUANT_INT4_AWQ:
        return AWQLinear(in_features, out_features, config.quant.group_size,
                         bias)
    if quant_type == QUANT_INT4_AWQ_MODELOPT:
        return ModelOptAWQPrepackedLinear(in_features, out_features,
                                          config.quant.group_size, bias)
    if quant_type == QUANT_INT4_GPTQ:
        return GPTQLinear(in_features, out_features, config.quant.group_size,
                          config.quant.gptq_zero_point_offset, bias)
    if quant_type == QUANT_INT8_SQ:
        return INT8SQLinear(in_features, out_features, bias)
    raise ValueError(f"Unknown quant_type: {quant_type!r}")
