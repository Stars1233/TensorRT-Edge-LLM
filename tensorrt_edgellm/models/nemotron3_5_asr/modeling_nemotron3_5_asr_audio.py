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
From-scratch Nemotron-3.5-ASR streaming FastConformer encoder.

Reference: HF ``transformers/models/nemotron3_5_asr`` (Nemotron3_5AsrForRNNT,
encoder side) for ``nvidia/nemotron-3.5-asr-streaming-0.6b``.

Architecture (offline / full-utterance mode):
    Causal subsampling (stem Conv2d + 2 depthwise-separable stages, factor=8)
    → Relative positional encoding (sinusoidal, Transformer-XL style)
    → 24 × ConformerBlock with a ``chunked_limited`` attention mask
        FFN1: LayerNorm → Linear → SiLU → Linear → residual (×0.5)
        SelfAttn: LayerNorm → RelPosMultiHeadAttn (masked) → residual
        Conv: LayerNorm → pointwise → GLU → causal depthwise → LN → SiLU
              → pointwise → residual
        FFN2: LayerNorm → Linear → SiLU → Linear → residual (×0.5)
        → LayerNorm
    → prompt conditioning: concat one-hot(prompt_id) → MLP → 1024
    → encoder_projector: Linear 1024 → 640 ("joint space" frames)

All convolutions are causal (left/asymmetric padding), matching NeMo's
CausalConv1D/CausalConv2D, so offline output is frame-identical to the
streaming mode the checkpoint was trained for. The ``chunked_limited`` mask
gives every frame ``sliding_window - 1`` frames of left context and
``num_lookahead_tokens`` frames of (chunk-aligned) right context. The
lookahead is a build-time constant here; one engine per latency mode.

Checkpoint weight key prefixes (identity mapping):
    ``encoder.*``, ``prompt_projector.*``, ``encoder_projector.*``
(``decoder.*`` / ``joint.*`` belong to the RNN-T step model, see
``modeling_nemotron3_5_asr_decoder.py``.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Causal subsampling
# ---------------------------------------------------------------------------


class _CausalSubsamplingLayer(nn.Module):
    """Depthwise-separable stride-2 stage: causal depthwise Conv2d + 1x1."""

    def __init__(self, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels,
                                        channels,
                                        kernel_size,
                                        stride=stride,
                                        groups=channels,
                                        bias=True)
        self.pointwise_conv = nn.Conv2d(channels, channels, 1, bias=True)
        self._pad = (kernel_size - 1, stride - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NeMo CausalConv2D: (kernel-1, stride-1) padding on BOTH time and
        # freq axes (asymmetric, causal on time; freq inherits the same
        # scheme, which is why 128 mels subsample to 17 bins, not 16).
        x = F.pad(x, (self._pad[0], self._pad[1], self._pad[0], self._pad[1]))
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)


class CausalSubsampling(nn.Module):
    """Causal Conv2d subsampling (factor=8) + flatten + linear.

    Checkpoint keys (under ``subsampling.``):
        ``conv_in.{weight,bias}``
        ``layers.{0,1}.depthwise_conv.{weight,bias}``
        ``layers.{0,1}.pointwise_conv.{weight,bias}``
        ``linear.{weight,bias}``
    """

    def __init__(self,
                 mel_bins: int,
                 hidden_size: int,
                 conv_channels: int = 256,
                 kernel_size: int = 3,
                 stride: int = 2,
                 subsampling_factor: int = 8) -> None:
        super().__init__()
        num_layers = int(math.log2(subsampling_factor))
        self.conv_in = nn.Conv2d(1,
                                 conv_channels,
                                 kernel_size,
                                 stride=stride,
                                 bias=True)
        self.layers = nn.ModuleList(
            _CausalSubsamplingLayer(conv_channels, kernel_size, stride)
            for _ in range(num_layers - 1))
        self._pad = (kernel_size - 1, stride - 1)

        # Output length per stage: floor(L/stride) + 1 (causal padding).
        freq_out = mel_bins
        for _ in range(num_layers):
            freq_out = freq_out // stride + 1
        self.linear = nn.Linear(conv_channels * freq_out, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, mel_bins] → [B, floor(T/8)+…, hidden_size]"""
        x = x.unsqueeze(1)  # [B, 1, T, mel]
        x = F.pad(x, (self._pad[0], self._pad[1], self._pad[0], self._pad[1]))
        x = F.relu(self.conv_in(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        # [B, C, T', F'] → [B, T', C*F']  (channel-major flatten)
        B, C, T, F_ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F_)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Relative positional encoding (sinusoidal, positions +(T-1) … -(T-1))
# ---------------------------------------------------------------------------


class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding, interleaved [sin, cos].

    Output shape ``[1, 2T-1, H]`` covering relative distances
    ``+(T-1) … -(T-1)`` (descending), matching the HF/NeMo layout.
    """

    def __init__(self, hidden_size: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len - 1, -max_len, -1,
                                dtype=torch.float32).unsqueeze(1)
        inv_freq = 1.0 / (10000.0**(torch.arange(
            0, hidden_size, 2, dtype=torch.float32) / hidden_size))
        freqs = position * inv_freq.unsqueeze(0)
        pe = torch.zeros(2 * max_len - 1, hidden_size)
        pe[:, 0::2] = torch.sin(freqs)
        pe[:, 1::2] = torch.cos(freqs)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self._max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        center = self._max_len - 1
        return self.pe[:, center - (T - 1):center + T].to(x.dtype)


# ---------------------------------------------------------------------------
# Relative position multi-head attention with chunked-limited mask
# ---------------------------------------------------------------------------


class RelPosMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position bias and bool mask.

    Checkpoint keys (under ``self_attn.``):
        ``q_proj.weight``, ``k_proj.weight``, ``v_proj.weight``,
        ``o_proj.weight``, ``relative_k_proj.weight``,
        ``bias_u`` [H, D], ``bias_v`` [H, D]
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relative_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Skewing trick: absolute-indexed pos scores → relative alignment."""
        B, H, T, L = x.shape
        x = F.pad(x, (1, 0))
        x = x.reshape(B, H, L + 1, T)
        x = x[:, :, 1:, :]
        return x.reshape(B, H, T, L)[:, :, :, :T]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Terms (a)+(c): content attention with global content bias u.
        q_with_u = q + self.bias_u.unsqueeze(0).unsqueeze(2)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))

        # Terms (b)+(d): position attention with global position bias v.
        rel_k = self.relative_k_proj(pos_emb).view(1, -1, H, D).transpose(1, 2)
        q_with_v = q + self.bias_v.unsqueeze(0).unsqueeze(2)
        pos_score = self._rel_shift(
            torch.matmul(q_with_v, rel_k.transpose(-2, -1)))

        scores = (content_score + pos_score) * self.scale
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Conformer convolution module (causal, LayerNorm variant)
# ---------------------------------------------------------------------------


class ConformerConvModule(nn.Module):
    """pointwise → GLU → causal depthwise → LayerNorm → SiLU → pointwise.

    Checkpoint keys (under ``conv.``):
        ``pointwise_conv1.weight`` [2H, H, 1]
        ``depthwise_conv.weight``  [H, 1, K]
        ``norm.{weight,bias}``     (LayerNorm — NOT BatchNorm)
        ``pointwise_conv2.weight`` [H, H, 1]
    """

    def __init__(self, hidden_size: int, kernel_size: int = 9) -> None:
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(hidden_size,
                                         hidden_size * 2,
                                         kernel_size=1,
                                         bias=False)
        self.depthwise_conv = nn.Conv1d(hidden_size,
                                        hidden_size,
                                        kernel_size=kernel_size,
                                        groups=hidden_size,
                                        bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.pointwise_conv2 = nn.Conv1d(hidden_size,
                                         hidden_size,
                                         kernel_size=1,
                                         bias=False)
        self._left_pad = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, H] → [B, T, H]"""
        x = x.transpose(1, 2)  # [B, 2H→H, T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = F.pad(x, (self._left_pad, 0))  # causal
        x = self.depthwise_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# Conformer feed-forward and block
# ---------------------------------------------------------------------------


class ConformerFeedForward(nn.Module):
    """Linear → SiLU → Linear (no bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class ConformerBlock(nn.Module):
    """Macaron Conformer block; see checkpoint keys under ``layers.N.``."""

    def __init__(self, hidden_size: int, num_heads: int,
                 intermediate_size: int, conv_kernel_size: int) -> None:
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.feed_forward1 = ConformerFeedForward(hidden_size,
                                                  intermediate_size)
        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = RelPosMultiHeadAttention(hidden_size, num_heads)
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.conv = ConformerConvModule(hidden_size, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(hidden_size)
        self.feed_forward2 = ConformerFeedForward(hidden_size,
                                                  intermediate_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, attn_mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Nemotron3_5AsrEncoder(nn.Module):
    """FastConformer encoder with chunked-limited streaming attention mask.

    Checkpoint keys (under ``encoder.``): ``subsampling.*``, ``layers.N.*``
    """

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int,
                 intermediate_size: int, mel_bins: int, conv_kernel_size: int,
                 conv_channels: int, subsampling_factor: int,
                 sliding_window: int, num_lookahead_tokens: int,
                 max_position_embeddings: int) -> None:
        super().__init__()
        self.subsampling = CausalSubsampling(
            mel_bins,
            hidden_size,
            conv_channels,
            subsampling_factor=subsampling_factor)
        self.encode_positions = RelPositionalEncoding(
            hidden_size, max_len=max_position_embeddings)
        self.layers = nn.ModuleList(
            ConformerBlock(hidden_size, num_heads, intermediate_size,
                           conv_kernel_size) for _ in range(num_layers))
        # chunked_limited: keys are grouped in chunks of (lookahead+1) frames;
        # a query sees its own chunk plus left_context//chunk_size chunks back.
        self._chunk_size = num_lookahead_tokens + 1
        self._left_context_chunks = (sliding_window - 1) // self._chunk_size

    def _chunked_limited_mask(self, T: int,
                              device: torch.device) -> torch.Tensor:
        idx = torch.arange(T, device=device)
        chunk = torch.div(idx, self._chunk_size, rounding_mode="trunc")
        diff = chunk.unsqueeze(1) - chunk.unsqueeze(0)
        allowed = (diff >= 0) & (diff <= self._left_context_chunks)
        return allowed.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """[B, T_mel, mel_bins] → [B, T, hidden_size]"""
        x = self.subsampling(input_features)
        pos_emb = self.encode_positions(x)
        attn_mask = self._chunked_limited_mask(x.shape[1], x.device)
        for layer in self.layers:
            x = layer(x, pos_emb, attn_mask)
        return x


# ---------------------------------------------------------------------------
# Prompt conditioning
# ---------------------------------------------------------------------------


class PromptProjector(nn.Module):
    """MLP fusing per-frame one-hot language prompt into encoder frames.

    Checkpoint keys (under ``prompt_projector.``):
        ``linear_1.{weight,bias}``, ``linear_2.{weight,bias}``
    """

    def __init__(self, hidden_size: int, num_prompts: int,
                 intermediate_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size + num_prompts, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.relu(self.linear_1(x)))


# ---------------------------------------------------------------------------
# Full audio model
# ---------------------------------------------------------------------------


class Nemotron3_5AsrAudioModel(nn.Module):
    """FastConformer encoder + prompt conditioning + projection to 640-d.

    Output frames feed the RNN-T joint network directly (NOT LLM prompt
    embeddings): ``[batch, encoded_seq_len, decoder_hidden_size]``.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        ec = config["encoder_config"]
        self.num_prompts = config.get("num_prompts", 128)
        self.default_prompt_id = config.get("default_prompt_id", 101)

        self.encoder = Nemotron3_5AsrEncoder(
            hidden_size=ec["hidden_size"],
            num_heads=ec["num_attention_heads"],
            num_layers=ec["num_hidden_layers"],
            intermediate_size=ec["intermediate_size"],
            mel_bins=ec.get("num_mel_bins", 128),
            conv_kernel_size=ec.get("conv_kernel_size", 9),
            conv_channels=ec.get("subsampling_conv_channels", 256),
            subsampling_factor=ec.get("subsampling_factor", 8),
            sliding_window=ec.get("sliding_window", 57),
            num_lookahead_tokens=ec.get("default_num_lookahead_tokens", 3),
            max_position_embeddings=ec.get("max_position_embeddings", 5000),
        )
        self.prompt_projector = PromptProjector(
            hidden_size=ec["hidden_size"],
            num_prompts=self.num_prompts,
            intermediate_size=config.get("prompt_intermediate_size", 2048),
        )
        self.encoder_projector = nn.Linear(ec["hidden_size"],
                                           config["decoder_hidden_size"])
        # One-hot via eye-row gather: exports as a plain Gather instead of a
        # OneHot op (broader TRT support).
        self.register_buffer("prompt_eye",
                             torch.eye(self.num_prompts),
                             persistent=False)

    def forward(self, input_features: torch.Tensor,
                prompt_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [1, T_mel, mel_bins] log-mel frames.
            prompt_ids: [1] int64 language-prompt index (101 = auto).

        Returns:
            [1, T, decoder_hidden_size] encoder frames for the RNN-T joint.
        """
        x = self.encoder(input_features)
        one_hot = self.prompt_eye[prompt_ids].to(x.dtype)
        one_hot = one_hot.unsqueeze(1).expand(-1, x.shape[1], -1)
        fused = self.prompt_projector(torch.cat([x, one_hot], dim=-1))
        return self.encoder_projector(fused)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes)."""
        mel_bins = config["encoder_config"].get("num_mel_bins", 128)
        dtype = next(self.parameters()).dtype
        input_features = torch.zeros(1,
                                     200,
                                     mel_bins,
                                     dtype=dtype,
                                     device=device)
        prompt_ids = torch.full((1, ),
                                self.default_prompt_id,
                                dtype=torch.int64,
                                device=device)
        args = (input_features, prompt_ids)
        input_names = ["input_features", "prompt_ids"]
        output_names = ["encoder_frames"]
        S = torch.export.Dim("mel_seq_len", min=8, max=16384)
        dynamic_shapes = {
            "input_features": {
                1: S
            },
            "prompt_ids": None,
        }
        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading / factory
# ---------------------------------------------------------------------------

_ENCODER_PREFIXES = ("encoder.", "prompt_projector.", "encoder_projector.")


def _load_weights(model: Nemotron3_5AsrAudioModel, weights: dict) -> None:
    """Checkpoint keys map 1:1 onto module paths; decoder/joint keys are
    handled by the RNN-T step model and skipped here."""
    from ...checkpoint.loader import load_submodule_weights

    def _remap(k: str) -> "str | None":
        if k.startswith(_ENCODER_PREFIXES):
            return k
        return None

    load_submodule_weights(model,
                           weights,
                           _remap,
                           label="Nemotron3_5AsrAudioModel")


def build_nemotron3_5_asr_audio(
        config: dict,
        weights: dict,
        dtype: torch.dtype = torch.float16) -> Nemotron3_5AsrAudioModel:
    """Build a :class:`Nemotron3_5AsrAudioModel` with loaded weights.

    Args:
        config:  Full parsed ``config.json`` dict (needs ``encoder_config``).
        weights: Flat ``{key: tensor}`` dict from safetensors.
        dtype:   Target dtype (default ``float16``).
    """
    model = Nemotron3_5AsrAudioModel(config)
    _load_weights(model, weights)
    # Cast AFTER loading: the checkpoint is fp32 and ``_set_tensor``
    # deliberately preserves the source dtype, which would silently leave
    # the whole graph (and its ONNX export) in fp32.
    model.to(dtype)
    model.eval()
    return model


__all__ = [
    "Nemotron3_5AsrAudioModel",
    "build_nemotron3_5_asr_audio",
]
