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
"""Qwen3-Omni Code2Wav vocoder — standalone implementation for ONNX export.

Converts discrete RVQ codec tokens (from Talker/CodePredictor) into
continuous audio waveforms.  Architecture: code embedding → small transformer
with sliding-window attention → ConvNet upsampling → hierarchical decoder.

No HuggingFace runtime dependency — weights are loaded directly from
safetensors via :mod:`tensorrt_edgellm.checkpoint.loader` or manual state_dict.

Checkpoint key prefix: ``code2wav.*``
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["Code2WavModel", "build_code2wav"]

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """SnakeBeta activation: ``x + (1/beta) * sin^2(alpha * x)``.

    Parameters are stored as 1D ``[channels]`` (checkpoint layout) and
    reshaped to ``[1, channels, 1]`` for broadcasting with ``[B, C, T]`` input.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp().unsqueeze(0).unsqueeze(-1)
        beta = self.beta.exp().unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x).pow(2)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm — keeps input dtype throughout for clean ONNX export.

    TensorRT handles mixed-precision internally; forcing FP32 in the ONNX
    graph causes type-mismatch errors in OnnxRuntime validation.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ---------------------------------------------------------------------------
# Convolution blocks
# ---------------------------------------------------------------------------


def _get_extra_padding(x_len: int, kernel_size: int, stride: int,
                       padding: int) -> int:
    """Compute extra right-padding for causal convolution."""
    out_len = (x_len + 2 * padding - kernel_size) // stride + 1
    ideal_len = (out_len - 1) * stride + kernel_size - 2 * padding
    return max(0, ideal_len - x_len)


class CausalConv1d(nn.Module):
    """Causal Conv1d with dynamic left-padding."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 stride: int = 1,
                 groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              dilation=dilation,
                              groups=groups)
        self.stride = stride
        effective_k = (kernel_size - 1) * dilation + 1
        self.padding = effective_k - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra = _get_extra_padding(x.shape[-1], self.conv.kernel_size[0],
                                   self.conv.stride[0], self.padding)
        x = F.pad(x, (self.padding, extra))
        return self.conv(x)


class CausalTransposeConv1d(nn.Module):
    """Causal ConvTranspose1d with HF-compatible trim.

    Matches ``Qwen3OmniCausalTransConvNet``: trims ``kernel_size - stride`` on
    both left and right edges (total ``2*(kernel_size - stride)``).  This
    produces outputs one stride shorter than ``input_len * stride`` but
    matches the upstream HF export bit-for-bit.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride)
        self.left_pad = math.ceil(kernel_size - stride)
        self.right_pad = self.left_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y[..., self.left_pad:y.shape[-1] - self.right_pad]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise causal conv + pointwise expansion."""

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, C, T] → [B, T, C]
        x = self.norm(x)
        x = F.gelu(self.pwconv1(x))
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # [B, T, C] → [B, C, T]
        return residual + x


class DecoderResidualUnit(nn.Module):
    """Decoder residual unit with SnakeBeta activations."""

    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class DecoderStage(nn.Module):
    """Decoder upsampling stage: SnakeBeta + TransConv + ResidualUnits.

    Wraps sub-modules in a ``block`` ModuleList to match checkpoint key layout
    (``decoder.{i}.block.{j}.*``).
    """

    def __init__(self, in_dim: int, out_dim: int, rate: int) -> None:
        super().__init__()
        self.block = nn.ModuleList([
            SnakeBeta(in_dim),
            CausalTransposeConv1d(in_dim,
                                  out_dim,
                                  kernel_size=rate * 2,
                                  stride=rate),
        ])
        for dilation in [1, 3, 9]:
            self.block.append(DecoderResidualUnit(out_dim, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for sub in self.block:
            x = sub(x)
        return x


# ---------------------------------------------------------------------------
# Transformer (pre_transformer)
# ---------------------------------------------------------------------------


class LayerScale(nn.Module):

    def __init__(self, dim: int, initial_scale: float = 0.01) -> None:
        super().__init__()
        self.scale = nn.Parameter(initial_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class Attention(nn.Module):
    """Multi-head attention with ONNX-friendly sliding window mask."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 sliding_window: int, rope_theta: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # RoPE parameters (stored as float32 for precision, cast at forward time)
        inv_freq = 1.0 / (rope_theta
                          **(torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(B, L, self.num_heads,
                                            self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads,
                                            self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads,
                                            self.head_dim).transpose(1, 2)

        # RoPE (compute in float32 for precision, cast to model dtype for ONNX)
        positions = torch.arange(L, device=hidden_states.device)
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(0)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        # Sliding window causal mask (ONNX-friendly, keep model dtype)
        q_pos = torch.arange(L, device=hidden_states.device).unsqueeze(1)
        k_pos = torch.arange(L, device=hidden_states.device).unsqueeze(0)
        valid = (k_pos > q_pos - self.sliding_window) & (k_pos <= q_pos)
        mask = torch.where(
            valid, torch.zeros(1, dtype=dtype, device=hidden_states.device),
            torch.tensor(torch.finfo(dtype).min,
                         dtype=dtype,
                         device=hidden_states.device))

        # Attention (scale as tensor to avoid FP32 upcast in ONNX graph)
        scale = torch.tensor(self.scale, dtype=q.dtype, device=q.device)
        attn = (q @ k.transpose(-2, -1)) * scale + mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


class TransformerLayer(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 intermediate_size: int, sliding_window: int,
                 rope_theta: float, rms_norm_eps: float,
                 layer_scale_init: float) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.self_attn = Attention(hidden_size, num_heads, head_dim,
                                   sliding_window, rope_theta)
        self.self_attn_layer_scale = LayerScale(hidden_size, layer_scale_init)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp_gate_proj = nn.Linear(hidden_size,
                                       intermediate_size,
                                       bias=False)
        self.mlp_up_proj = nn.Linear(hidden_size,
                                     intermediate_size,
                                     bias=False)
        self.mlp_down_proj = nn.Linear(intermediate_size,
                                       hidden_size,
                                       bias=False)
        self.mlp_layer_scale = LayerScale(hidden_size, layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h)
        x = residual + self.self_attn_layer_scale(h)
        # MLP (SwiGLU)
        residual = x
        h = self.post_attention_layernorm(x)
        h = self.mlp_down_proj(
            F.silu(self.mlp_gate_proj(h)) * self.mlp_up_proj(h))
        x = residual + self.mlp_layer_scale(h)
        return x


# ---------------------------------------------------------------------------
# Top-level Code2Wav model
# ---------------------------------------------------------------------------


class Code2WavModel(nn.Module):
    """Standalone Code2Wav vocoder for Qwen3-Omni.

    Converts RVQ codec tokens ``[batch, num_quantizers, code_length]`` to
    waveform ``[batch, 1, code_length * total_upsample]``.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        num_quantizers = config["num_quantizers"]
        codebook_size = config["codebook_size"]
        decoder_dim = config["decoder_dim"]
        upsample_rates = config["upsample_rates"]
        upsampling_ratios = config["upsampling_ratios"]

        # Code embedding
        self.code_embedding = nn.Embedding(codebook_size * num_quantizers,
                                           hidden_size)
        code_offset = torch.arange(num_quantizers).view(1, -1,
                                                        1) * codebook_size
        self.register_buffer("code_offset", code_offset, persistent=False)

        # Pre-transformer
        num_layers = config["num_hidden_layers"]
        num_heads = config["num_attention_heads"]
        head_dim = hidden_size // num_heads
        intermediate_size = config["intermediate_size"]
        sliding_window = config.get("sliding_window", 72)
        rope_theta = config.get("rope_theta", 10000.0)
        rms_norm_eps = config.get("rms_norm_eps", 1e-5)
        layer_scale_init = config.get("layer_scale_initial_scale", 0.01)

        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, head_dim,
                             intermediate_size, sliding_window, rope_theta,
                             rms_norm_eps, layer_scale_init)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, rms_norm_eps)

        # Upsample stages (TransposeConv + ConvNeXt)
        self.upsample = nn.ModuleList()
        for ratio in upsampling_ratios:
            stage = nn.ModuleList([
                CausalTransposeConv1d(hidden_size,
                                      hidden_size,
                                      kernel_size=ratio,
                                      stride=ratio),
                ConvNeXtBlock(hidden_size),
            ])
            self.upsample.append(stage)

        # Decoder
        self.decoder = nn.ModuleList()
        # First conv: hidden_size → decoder_dim
        self.decoder.append(
            CausalConv1d(hidden_size, decoder_dim, kernel_size=7))

        # Hierarchical upsampling stages
        in_dim = decoder_dim
        for rate in upsample_rates:
            out_dim = in_dim // 2
            self.decoder.append(DecoderStage(in_dim, out_dim, rate))
            in_dim = out_dim

        # Final output: SnakeBeta + Conv1d → 1 channel
        self.decoder.append(SnakeBeta(in_dim))
        self.decoder.append(CausalConv1d(in_dim, 1, kernel_size=7))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        # Code embedding: [B, Q, L] → [B, L, H]
        hidden = self.code_embedding(codes + self.code_offset).mean(1)

        # Pre-transformer
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)

        # [B, L, H] → [B, H, L] for conv layers
        hidden = hidden.permute(0, 2, 1)

        # Upsample stages
        for stage in self.upsample:
            for block in stage:
                hidden = block(hidden)

        # Decoder (CausalConv1d, DecoderStage×4, SnakeBeta, CausalConv1d)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)

        return wav.clamp(min=-1, max=1)


# ---------------------------------------------------------------------------
# Builder + export
# ---------------------------------------------------------------------------


def build_code2wav(config: dict, weights: dict,
                   dtype: torch.dtype) -> Code2WavModel:
    """Build Code2Wav model and load weights from checkpoint dict.

    Args:
        config: ``code2wav_config`` dict from the HF config.json.
        weights: Flat dict of all checkpoint tensors (full model).
        dtype: Target dtype (e.g. ``torch.float16``).

    Returns:
        Loaded Code2WavModel ready for export.
    """
    model = Code2WavModel(config)

    # Extract code2wav weights and cast bf16 → fp16
    prefix = "code2wav."
    state_dict = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            if v.dtype == torch.bfloat16:
                v = v.to(torch.float16)
            state_dict[new_key] = v

    # Remap checkpoint keys to model attribute names
    # MLP keys: checkpoint uses mlp.gate_proj etc., model uses mlp_gate_proj
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in [
            (".mlp.gate_proj.", ".mlp_gate_proj."),
            (".mlp.up_proj.", ".mlp_up_proj."),
            (".mlp.down_proj.", ".mlp_down_proj."),
        ]:
            new_k = new_k.replace(old, new)
        # pre_transformer.layers → layers
        new_k = new_k.replace("pre_transformer.layers.", "layers.")
        new_k = new_k.replace("pre_transformer.norm.", "norm.")
        remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.warning("[Code2Wav] Missing keys (%d): %s...", len(missing),
                       missing[:5])
    if unexpected:
        logger.warning("[Code2Wav] Unexpected keys (%d): %s...",
                       len(unexpected), unexpected[:5])

    model.eval().to(dtype)
    return model


def export_code2wav_onnx(model: Code2WavModel, output_path: str,
                         config: dict) -> None:
    """Export Code2Wav to ONNX via dynamo export.

    The caller is responsible for writing ``config.json`` next to the
    exported ONNX (see ``tensorrt-edgellm-export._export_code2wav``).

    Args:
        model: Loaded Code2WavModel.
        output_path: Path for the output ``model.onnx``.
        config: ``code2wav_config`` dict (used for dummy-input shape).
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    num_quantizers = config.get("num_quantizers", 16)
    codebook_size = config.get("codebook_size", 2048)

    codes = torch.randint(0,
                          codebook_size, (2, num_quantizers, 300),
                          dtype=torch.int64,
                          device=next(model.parameters()).device)

    batch_dim = torch.export.Dim("batch", min=1, max=256)
    # Max code_len limited by int32 output: code_len * total_upsample < 2^31
    # total_upsample = prod(upsample_rates + upsampling_ratios) ≈ 3840
    code_len_dim = torch.export.Dim("code_len", min=1, max=8192)

    # Use OPSET 22 to avoid RMSNormalization native op (unsupported by
    # TensorRT 10.13).  OPSET 22 decomposes RMSNorm into basic ops.
    export_program = torch.onnx.export(model, (codes, ),
                                       dynamo=True,
                                       opset_version=22,
                                       input_names=["codes"],
                                       output_names=["waveform"],
                                       dynamic_shapes=({
                                           0: batch_dim,
                                           2: code_len_dim
                                       }, ))
    export_program.save(output_path,
                        include_initializers=True,
                        keep_initializers_as_inputs=False)
