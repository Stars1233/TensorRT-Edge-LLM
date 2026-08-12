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
"""Qwen3-Next Omni Code2Wav vocoder — standalone implementation for ONNX export.

Converts RVQ codec tokens ``[batch, num_quantizers, code_length]`` into
continuous audio waveforms ``[batch, 1, code_length * total_upsample]``.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "Code2WavModel",
    "build_qwen3_omni_next_code2wav",
    "export_qwen3_omni_next_code2wav_onnx",
    "load_code2wav_config",
]

# ---------------------------------------------------------------------------
# Activation — Snake1d (aka SnakeBeta, log-scale α/β)
# ---------------------------------------------------------------------------


class Snake1d(nn.Module):
    """``x + (1/exp(β)) * sin²(exp(α) * x)`` on per-channel α/β."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp().unsqueeze(0).unsqueeze(-1)
        beta = self.beta.exp().unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x).pow(2)


# ---------------------------------------------------------------------------
# Norm + LayerScale
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Keep activation dtype; ONNX-friendly."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class LayerScale(nn.Module):
    """Per-channel scale multiplier (Llama/DiT style, ``gamma`` param)."""

    def __init__(self, dim: int, init_value: float = 1e-2) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


# ---------------------------------------------------------------------------
# Causal 1-D convolutions
# ---------------------------------------------------------------------------


def _extra_causal_padding(x_len: int, kernel: int, stride: int,
                          padding: int) -> int:
    out_len = (x_len + 2 * padding - kernel) // stride + 1
    ideal = (out_len - 1) * stride + kernel - 2 * padding
    return max(0, ideal - x_len)


class CausalConvNet(nn.Module):
    """Left-padded causal Conv1d (matches HF ``CausalConvNet``)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              dilation=dilation,
                              groups=groups)
        self.stride = stride
        self.effective_k = (kernel_size - 1) * dilation + 1
        self.padding = self.effective_k - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra = _extra_causal_padding(x.shape[-1], self.effective_k,
                                      self.stride, self.padding)
        x = F.pad(x, (self.padding, extra))
        return self.conv(x)


class CausalTransConvNet(nn.Module):
    """Causal ConvTranspose1d: trims ``kernel_size - stride`` from both ends."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride)
        pad = kernel_size - stride
        self.pad_right = math.ceil(pad)
        self.pad_left = pad - self.pad_right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # Drop the first ``-pad_left`` and last ``pad_right`` samples (pad_left
        # is non-positive so ``-pad_left`` is the left trim length).
        return y[..., -self.pad_left:y.shape[-1] - self.pad_right]


# ---------------------------------------------------------------------------
# ConvNeXt block (used inside each upsample stage)
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """Depthwise causal conv + pointwise expansion + per-channel γ."""

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = CausalConvNet(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, C, T] → [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # back to [B, C, T]
        return residual + x


# ---------------------------------------------------------------------------
# ResidualUnit / DecoderBlock / Decoder (terminal audio synthesis stack)
# ---------------------------------------------------------------------------


class ResidualUnit(nn.Module):
    """Snake → dilated conv(k=7) → Snake → conv(k=1), with x-skip."""

    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.ModuleList([
            Snake1d(dim),
            CausalConvNet(dim, dim, kernel_size=7, dilation=dilation),
            Snake1d(dim),
            CausalConvNet(dim, dim, kernel_size=1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for m in self.block:
            h = m(h)
        return x + h


class DecoderBlock(nn.Module):
    """Upsample stage: Snake → TransConv(stride=R) → 3× ResidualUnit."""

    def __init__(self, in_dim: int, out_dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.ModuleList([
            Snake1d(in_dim),
            CausalTransConvNet(in_dim,
                               out_dim,
                               kernel_size=2 * stride,
                               stride=stride),
            ResidualUnit(out_dim, dilation=1),
            ResidualUnit(out_dim, dilation=3),
            ResidualUnit(out_dim, dilation=9),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.block:
            x = m(x)
        return x


class Decoder(nn.Module):
    """Initial conv → N DecoderBlocks → final Snake + conv → clamp."""

    def __init__(self, input_channel: int, channels: int,
                 rates: "list[int]") -> None:
        super().__init__()
        layers: "list[nn.Module]" = [
            CausalConvNet(input_channel, channels, kernel_size=7)
        ]
        for i, stride in enumerate(rates):
            in_dim = channels // (2**i)
            out_dim = channels // (2**(i + 1))
            layers.append(DecoderBlock(in_dim, out_dim, stride))
        layers.append(Snake1d(out_dim))
        layers.append(CausalConvNet(out_dim, 1, kernel_size=7))
        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.model:
            x = m(x)
        return x.clamp(min=-1, max=1)


# ---------------------------------------------------------------------------
# RVQ quantizer (semantic + acoustic split)
# ---------------------------------------------------------------------------


class EuclideanCodebook(nn.Module):
    """Codebook stored as ``(embedding_sum, cluster_usage)``.

    HF decodes by ``embedding = embedding_sum / cluster_usage.clamp(eps)``
    at every forward.  Both tensors are static post-training — we fold the
    division into a cached lookup table once at ``_maybe_build_codebook``
    (first forward or explicit call) so the ONNX graph only contains an
    ``F.embedding`` lookup.  This also avoids a torch.onnx dynamo bug with
    ``Parameter.clamp(min=scalar)``.
    """

    def __init__(self,
                 dim: int,
                 codebook_size: int,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = eps
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))
        # Pre-materialised codebook, populated in ``_materialise()`` once
        # weights are loaded.  Registered as a buffer so it moves with the
        # model (``.to(device/dtype)``) and appears in ``state_dict`` without
        # being trainable.
        self.register_buffer("_codebook_embedding",
                             torch.zeros(codebook_size, dim),
                             persistent=False)
        self._materialised = False

    @torch.no_grad()
    def _materialise(self) -> None:
        eps_t = torch.full_like(self.cluster_usage, self.epsilon)
        denom = torch.maximum(self.cluster_usage, eps_t).unsqueeze(-1)
        emb = (self.embedding_sum / denom).to(self._codebook_embedding.dtype)
        self._codebook_embedding = emb
        self._materialised = True

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # ``_materialise`` must be called after weight loading and before any
        # forward call — done once in ``build_qwen3_omni_next_code2wav``.
        return F.embedding(codes, self._codebook_embedding)


class VectorQuantization(nn.Module):
    """Single-codebook VQ layer with optional output projection."""

    def __init__(self,
                 dim: int,
                 codebook_size: int,
                 codebook_dim: Optional[int] = None) -> None:
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim
        if codebook_dim != dim:
            self.project_out = nn.Linear(codebook_dim, dim)
        else:
            self.project_out = nn.Identity()
        self._codebook = EuclideanCodebook(codebook_dim, codebook_size)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        q = self._codebook.decode(codes)
        q = self.project_out(q)
        return q.transpose(1, 2)


class ResidualVectorQuantization(nn.Module):
    """N-layer residual VQ: sum of per-layer decoded embeddings."""

    def __init__(self, *, num_quantizers: int, dim: int,
                 codebook_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantization(dim, codebook_size)
            for _ in range(num_quantizers)
        ])

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [n_q, B, T]
        out = self.layers[0].decode(codes[0])
        for idx in range(1, codes.shape[0]):
            out = out + self.layers[idx].decode(codes[idx])
        return out


class ResidualVectorQuantizer(nn.Module):
    """RVQ with input / output projection Conv1d(1×1)."""

    def __init__(
        self,
        dimension: int,
        n_q: int,
        bins: int,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        force_projection: bool = False,
    ) -> None:
        super().__init__()
        in_d = input_dimension or dimension
        out_d = output_dimension or dimension
        if in_d != dimension or force_projection:
            self.input_proj = nn.Conv1d(in_d, dimension, 1, bias=False)
        else:
            self.input_proj = nn.Identity()
        if out_d != dimension or force_projection:
            self.output_proj = nn.Conv1d(dimension, out_d, 1, bias=False)
        else:
            self.output_proj = nn.Identity()
        self.vq = ResidualVectorQuantization(num_quantizers=n_q,
                                             dim=dimension,
                                             codebook_size=bins)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, n_q, T] → transpose to [n_q, B, T] for per-layer decode
        codes = codes.transpose(0, 1)
        q = self.vq.decode(codes)
        return self.output_proj(q)


class SplitResidualVectorQuantizer(nn.Module):
    """Split RVQ: 1 semantic + (n_q-1) acoustic quantizers."""

    def __init__(
        self,
        dimension: int,
        n_q: int,
        n_q_semantic: int,
        bins: int,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert n_q > n_q_semantic
        kw = dict(
            dimension=dimension,
            bins=bins,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            force_projection=True,
        )
        self.rvq_first = ResidualVectorQuantizer(n_q=n_q_semantic, **kw)
        self.rvq_rest = ResidualVectorQuantizer(n_q=n_q - n_q_semantic, **kw)
        self.n_q_semantic = n_q_semantic

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: [B, n_q=16, T]
        q = self.rvq_first.decode(codes[:, :self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            q = q + self.rvq_rest.decode(codes[:, self.n_q_semantic:])
        return q


# ---------------------------------------------------------------------------
# Pre-transformer — Llama-style attention, fused QKV, sliding-window mask
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    """Mirrors HF ``ModelArgs`` (subset used online)."""
    n_layer: int = 8
    n_head: int = 16
    n_local_heads: int = 16
    dim: int = 1024
    head_dim: int = 64
    intermediate_size: int = 3072
    norm_eps: float = 1e-5
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 16384


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                  sin: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    # cos/sin: [T, head_dim]; q/k: [B, H, T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class Attention(nn.Module):
    """Fused-QKV attention with GQA support and sliding-window causal mask.

    Matches the HF checkpoint keys ``attention.wqkv`` + ``attention.wo``.
    RoPE is Qwen3-style (standard rotate_half on the full head_dim) because
    the YAML config sets ``rope_base=-1``, which routes through
    ``apply_rotary_pos_emb`` (not the polar/interleaved path).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        total_head_dim = (cfg.n_head + 2 * cfg.n_local_heads) * cfg.head_dim
        self.wqkv = nn.Linear(cfg.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, cfg.dim, bias=False)

        inv_freq = 1.0 / (cfg.rope_theta**(
            torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        dtype = x.dtype
        device = x.device
        cfg = self.cfg
        q_size = cfg.n_head * cfg.head_dim
        kv_size = cfg.n_local_heads * cfg.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
        k = k.view(B, T, cfg.n_local_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(B, T, cfg.n_local_heads, cfg.head_dim).transpose(1, 2)

        positions = torch.arange(T, device=device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        q, k = _apply_rotary(q, k, cos, sin)

        # GQA: repeat K/V heads to match Q head count when n_local_heads < n_head.
        if cfg.n_head != cfg.n_local_heads:
            rep = cfg.n_head // cfg.n_local_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = torch.tensor(cfg.head_dim**-0.5, dtype=dtype, device=device)
        attn = (q @ k.transpose(-2, -1)) * scale + mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    """SwiGLU with Llama naming (``w1 / w3 / w2``)."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.attention_layer_scale = LayerScale(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.feed_forward = FeedForward(cfg)
        self.ffn_layer_scale = LayerScale(cfg.dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.attention(self.attention_norm(x), mask)
        x = x + self.attention_layer_scale(h)
        h = self.feed_forward(self.ffn_norm(x))
        x = x + self.ffn_layer_scale(h)
        return x


class Transformer(nn.Module):

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class WindowLimitedTransformer(nn.Module):
    """Thin wrapper around :class:`Transformer` adding channels-first handling,
    optional input/output projection, and a sliding-window causal mask."""

    def __init__(self, cfg: TransformerConfig, input_dim: int,
                 window_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        self.channels_first = True  # YAML: channels_first=true
        if input_dim != cfg.dim:
            self.input_proj = nn.Linear(input_dim, cfg.dim)
            self.output_proj = nn.Linear(cfg.dim, input_dim)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
        # look_ahead_conv is nn.Identity() in this checkpoint — no weights.
        self.look_ahead_conv = nn.Identity()
        # Inner transformer lives on ``self`` directly (not as a submodule)
        # to match HF's ``WindowLimitedTransformer(Transformer)`` inheritance
        # so checkpoint keys are ``layers.N.*`` and ``norm.weight``.
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def _sliding_window_mask(self, T: int, dtype: torch.dtype,
                             device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(T, device=device).unsqueeze(1)
        k_pos = torch.arange(T, device=device).unsqueeze(0)
        valid = (k_pos > q_pos - self.window_size) & (k_pos <= q_pos)
        neg_inf = torch.finfo(dtype).min
        return torch.where(valid, torch.zeros(1, dtype=dtype, device=device),
                           torch.tensor(neg_inf, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)  # [B, C, T] → [B, T, C]
        x = self.input_proj(x)
        x = self.look_ahead_conv(x)
        mask = self._sliding_window_mask(x.shape[1], x.dtype, x.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        x = self.output_proj(x)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Top-level Code2Wav model (mirrors HF ``DAC.decode``)
# ---------------------------------------------------------------------------


@dataclass
class Code2WavConfig:
    """Subset of HF YAML fields we actually need."""
    # Quantizer
    quant_dim: int = 256
    n_q: int = 16
    n_q_semantic: int = 1
    codebook_size: int = 2048
    quant_input_dim: int = 512
    quant_output_dim: int = 512
    # DAC backbone
    latent_dim: int = 1024
    decoder_dim: int = 1536
    decoder_rates: "list[int]" = field(default_factory=lambda: [8, 5, 4, 3])
    upsample_rates: "list[int]" = field(default_factory=lambda: [2, 2])
    # Pre-transformer (post_module in YAML)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    pre_transformer_input_dim: int = 1024
    pre_transformer_window_size: int = 72
    # Pre-conv
    pre_conv_kernel: int = 3


def load_code2wav_config(yaml_path: str) -> Code2WavConfig:
    """Parse the online-codec YAML into a :class:`Code2WavConfig`.

    The YAML contains a ``!!python/object`` tag in some releases; we avoid
    that by extracting only the scalar fields we care about.
    """
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    model = raw["model"]
    dac = model["dac"]
    quant = model["quantizer"]
    pm = model["post_module"]
    args = pm["args"]
    rope = args.get("rope") or {}

    tcfg = TransformerConfig(
        n_layer=args["n_layer"],
        n_head=args["n_head"],
        n_local_heads=(args["n_local_heads"] if args.get("n_local_heads", -1)
                       not in (-1, None) else args["n_head"]),
        dim=args["dim"],
        head_dim=args["head_dim"],
        intermediate_size=args["intermediate_size"],
        norm_eps=float(args.get("norm_eps", 1e-5)),
        rope_theta=float(rope.get("rope_theta", 10000.0)),
        max_position_embeddings=int(rope.get("max_position_embeddings",
                                             16384)),
    )
    return Code2WavConfig(
        quant_dim=quant["dimension"],
        n_q=quant["n_q"],
        n_q_semantic=quant["n_q_semantic"],
        codebook_size=quant["bins"],
        quant_input_dim=quant["input_dimension"],
        quant_output_dim=quant["output_dimension"],
        latent_dim=dac["latet_dim"],  # HF misspelled 'latet_dim'
        decoder_dim=dac["decoder_dim"],
        decoder_rates=list(dac["decoder_rates"]),
        transformer=tcfg,
        pre_transformer_input_dim=pm["input_dim"],
        pre_transformer_window_size=pm["window_size"],
    )


class Code2WavModel(nn.Module):
    """Qwen3-Next Omni Code2Wav vocoder (standalone)."""

    def __init__(self, cfg: Code2WavConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=cfg.quant_dim,
            n_q=cfg.n_q,
            n_q_semantic=cfg.n_q_semantic,
            bins=cfg.codebook_size,
            input_dimension=cfg.quant_input_dim,
            output_dimension=cfg.quant_output_dim,
        )
        self.pre_conv = CausalConvNet(cfg.quant_output_dim,
                                      cfg.latent_dim,
                                      kernel_size=cfg.pre_conv_kernel)
        self.pre_transformer = WindowLimitedTransformer(
            cfg.transformer,
            input_dim=cfg.pre_transformer_input_dim,
            window_size=cfg.pre_transformer_window_size,
        )
        self.upsample = nn.ModuleList([
            nn.ModuleList([
                CausalTransConvNet(cfg.latent_dim,
                                   cfg.latent_dim,
                                   kernel_size=r,
                                   stride=r),
                ConvNeXtBlock(cfg.latent_dim),
            ]) for r in cfg.upsample_rates
        ])
        self.decoder = Decoder(cfg.latent_dim, cfg.decoder_dim,
                               cfg.decoder_rates)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: discrete codec tokens ``[B, n_q=16, T]`` (int64).
        Returns:
            Waveform ``[B, 1, T * total_upsample]`` in ``[-1, 1]``.
        """
        x = self.quantizer.decode(codes)
        x = self.pre_conv(x)
        x = self.pre_transformer(x)
        for stage in self.upsample:
            for m in stage:
                x = m(x)
        return self.decoder(x)


# ---------------------------------------------------------------------------
# Builder + ONNX export
# ---------------------------------------------------------------------------


def build_qwen3_omni_next_code2wav(
        code2wav_dir: str,
        dtype: torch.dtype = torch.float16) -> Code2WavModel:
    """Load the standalone online Code2Wav model from its dedicated dir.

    *code2wav_dir* must contain ``config.yaml`` and ``model_weights.pt`` —
    the release layout of ``qwen3_5_omni_codec_decode_online_*``.
    """
    cfg_path = os.path.join(code2wav_dir, "config.yaml")
    ckpt_path = os.path.join(code2wav_dir, "model_weights.pt")
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Expected config.yaml + model_weights.pt under {code2wav_dir!r}")

    cfg = load_code2wav_config(cfg_path)
    model = Code2WavModel(cfg)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("generator.") for k in state):
        state = {
            k[len("generator."):]: v
            for k, v in state.items() if k.startswith("generator.")
        }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("[Code2Wav] missing keys (%d): %s...", len(missing),
                       missing[:5])
    if unexpected:
        logger.warning("[Code2Wav] unexpected keys (%d): %s...",
                       len(unexpected), unexpected[:5])

    # Bake the per-codebook ``embedding_sum / cluster_usage.clamp(eps)`` into
    # a static lookup table so the ONNX graph is a plain ``F.embedding``.
    for m in model.modules():
        if isinstance(m, EuclideanCodebook):
            m._materialise()

    model = model.to(dtype).eval()
    return model


def export_qwen3_omni_next_code2wav_onnx(model: Code2WavModel,
                                         output_path: str) -> None:
    """Export to ONNX via dynamo.  Dummy codec tokens drive tracing."""
    device = next(model.parameters()).device
    n_q = model.cfg.n_q
    # A short chunk (e.g. 150 codec tokens) is enough for shape inference;
    # the batch + time dims are marked dynamic.
    codes = torch.zeros(1, n_q, 150, dtype=torch.int64, device=device)
    dyn = {
        "codes": {
            0: torch.export.Dim("batch", min=1, max=16),
            2: torch.export.Dim("code_length", min=1, max=4096),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prog = torch.onnx.export(
        model,
        (codes, ),
        dynamo=True,
        input_names=["codes"],
        output_names=["waveform"],
        dynamic_shapes=dyn,
        opset_version=20,
        external_data=True,
        optimize=True,
    )
    prog.save(output_path)
    logger.info("[Code2Wav] Export complete: %s", output_path)
