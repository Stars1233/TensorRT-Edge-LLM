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
From-scratch Nemotron-Omni Parakeet audio encoder + sound projection.

Architecture:
    Subsampling (3× stride-2 Conv2d, factor=8)
    → Relative positional encoding (sinusoidal)
    → 24 × ConformerBlock
        FFN1: LayerNorm → Linear → SiLU → Linear → residual (×0.5)
        SelfAttn: LayerNorm → RelPosMultiHeadAttn → residual
        Conv: LayerNorm → pointwise → GLU → depthwise → BN → SiLU → pointwise → residual
        FFN2: LayerNorm → Linear → SiLU → Linear → residual (×0.5)
        → LayerNorm
    → SoundProjection: RMSNorm → Linear → SquaredReLU → Linear

Checkpoint weight key prefixes:
    Encoder:     ``sound_encoder.encoder.*``
    Projection:  ``sound_projection.*``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module

# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


class Subsampling(nn.Module):
    """Conv2d subsampling (factor=8): depthwise + pointwise stack.

    Treats mel spectrogram as [B, 1, T, mel_bins].

    Checkpoint keys (under ``subsampling.``):
        ``layers.{0,2,3,5,6}.weight``, ``layers.{0,2,3,5,6}.bias``
        ``linear.weight``, ``linear.bias``
    """

    def __init__(self,
                 mel_bins: int,
                 hidden_size: int,
                 conv_channels: int = 256) -> None:
        super().__init__()
        # 3× stride-2: depthwise(1→C,3,s2) → depthwise(C,3,s2)+pointwise(C,1)
        #                                    → depthwise(C,3,s2)+pointwise(C,1)
        self.layers = nn.Sequential(
            nn.Conv2d(1, conv_channels, 3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(conv_channels,
                      conv_channels,
                      3,
                      stride=2,
                      padding=1,
                      groups=conv_channels,
                      bias=True),
            nn.Conv2d(conv_channels, conv_channels, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(conv_channels,
                      conv_channels,
                      3,
                      stride=2,
                      padding=1,
                      groups=conv_channels,
                      bias=True),
            nn.Conv2d(conv_channels, conv_channels, 1, bias=True),
            nn.ReLU(),
        )
        freq_out = mel_bins // 8
        self.linear = nn.Linear(conv_channels * freq_out, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, mel_bins] → [B, T//8, hidden_size]"""
        x = x.unsqueeze(1)  # [B, 1, T, mel]
        x = self.layers(x)  # [B, C, T//8, mel//8]
        B, C, T, F_ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F_)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Relative positional encoding (sinusoidal)
# ---------------------------------------------------------------------------


class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding.

    Generates ``2T - 1`` positions from ``-(T-1)`` to ``+(T-1)`` for
    relative position attention. Output shape: ``[1, 2T-1, H]``.
    """

    def __init__(self, hidden_size: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(2 * max_len - 1, hidden_size)
        # Positions from +(max_len-1) down to -(max_len-1) (descending)
        position = torch.arange(max_len - 1,
                                -(max_len),
                                -1,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) *
            -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)
        self._max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return relative position encoding [1, 2T-1, H]."""
        T = x.shape[1]
        center = self._max_len - 1
        start = center - (T - 1)
        end = center + T
        return self.pe[:, start:end]


# ---------------------------------------------------------------------------
# Relative position multi-head attention
# ---------------------------------------------------------------------------


class RelPosMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position bias.

    Checkpoint keys (under ``self_attn.``):
        ``q_proj.weight``, ``k_proj.weight``, ``v_proj.weight``, ``o_proj.weight``
        ``relative_k_proj.weight``, ``bias_u`` [H, D], ``bias_v`` [H, D]
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 attention_scale: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_scale = attention_scale
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relative_k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.bias_v = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative position shift for Skewing method."""
        B, H, T, L = x.shape
        x = F.pad(x, (1, 0))
        x = x.reshape(B, H, L + 1, T)
        x = x[:, :, 1:, :]
        return x.reshape(B, H, T, L)[:, :, :, :T]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Content-based attention
        q_with_u = q + self.bias_u.unsqueeze(0).unsqueeze(2)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))

        # Position-based attention
        rel_k = self.relative_k_proj(pos_emb).view(1, -1, H, D).transpose(1, 2)
        q_with_v = q + self.bias_v.unsqueeze(0).unsqueeze(2)
        pos_score = torch.matmul(q_with_v, rel_k.transpose(-2, -1))
        pos_score = self._rel_shift(pos_score)

        scores = content_score + pos_score
        if self.attention_scale != 1.0:
            scores = scores * self.attention_scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Conformer convolution module
# ---------------------------------------------------------------------------


class ConformerConvModule(nn.Module):
    """Conformer convolution: pointwise → GLU → depthwise → BN → SiLU → pointwise.

    Checkpoint keys (under ``conv.``):
        ``pointwise_conv1.weight`` [2H, H, 1]
        ``depthwise_conv.weight``  [H, 1, K]
        ``norm.{weight,bias,running_mean,running_var}``
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
                                        padding=kernel_size // 2,
                                        groups=hidden_size,
                                        bias=False)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.pointwise_conv2 = nn.Conv1d(hidden_size,
                                         hidden_size,
                                         kernel_size=1,
                                         bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, H] → [B, T, H]"""
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # [B, H, T]
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# Conformer feed-forward module
# ---------------------------------------------------------------------------


class ConformerFeedForward(nn.Module):
    """Linear → SiLU → Linear.

    Checkpoint keys: ``linear1.weight``, ``linear2.weight``
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


# ---------------------------------------------------------------------------
# Conformer block
# ---------------------------------------------------------------------------


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN1 + SelfAttn + Conv + FFN2 + LayerNorm.

    Checkpoint keys (under ``layers.N.``):
        ``norm_feed_forward1.*``, ``feed_forward1.*``
        ``norm_self_att.*``, ``self_attn.*``
        ``norm_conv.*``, ``conv.*``
        ``norm_feed_forward2.*``, ``feed_forward2.*``
        ``norm_out.*``
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 intermediate_size: int, conv_kernel_size: int,
                 attention_scale: float) -> None:
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.feed_forward1 = ConformerFeedForward(hidden_size,
                                                  intermediate_size)
        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = RelPosMultiHeadAttention(hidden_size, num_heads,
                                                  attention_scale)
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.conv = ConformerConvModule(hidden_size, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(hidden_size)
        self.feed_forward2 = ConformerFeedForward(hidden_size,
                                                  intermediate_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        # Macaron-style: half-step FFN → attn → conv → half-step FFN
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        x = self.norm_out(x)
        return x


# ---------------------------------------------------------------------------
# Full Parakeet encoder
# ---------------------------------------------------------------------------


class ParakeetEncoder(nn.Module):
    """From-scratch Fast Conformer encoder.

    Checkpoint keys (under ``sound_encoder.encoder.``):
        ``subsampling.*``, ``layers.{0..23}.*``
    """

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int,
                 intermediate_size: int, mel_bins: int, conv_kernel_size: int,
                 conv_channels: int, attention_scale: float) -> None:
        super().__init__()
        self.subsampling = Subsampling(mel_bins, hidden_size, conv_channels)
        self.encode_positions = RelPositionalEncoding(hidden_size)
        self.layers = nn.ModuleList([
            ConformerBlock(hidden_size, num_heads, intermediate_size,
                           conv_kernel_size, attention_scale)
            for _ in range(num_layers)
        ])

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """[B, T, mel_bins] → [B, T//8, hidden_size]"""
        x = self.subsampling(input_features)
        pos_emb = self.encode_positions(x)
        for layer in self.layers:
            x = layer(x, pos_emb)
        return x


# ---------------------------------------------------------------------------
# Sound projection
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * x.to(input_dtype)


class SoundProjection(nn.Module):
    """RMSNorm → Linear → SquaredReLU → Linear.

    Checkpoint keys (under ``sound_projection.``):
        ``norm.weight``, ``linear1.weight``, ``linear2.weight``
    """

    def __init__(self,
                 sound_hidden_size: int,
                 projection_hidden_size: int,
                 llm_hidden_size: int,
                 bias: bool = False) -> None:
        super().__init__()
        self.norm = _RMSNorm(sound_hidden_size)
        self.linear1 = nn.Linear(sound_hidden_size,
                                 projection_hidden_size,
                                 bias=bias)
        self.linear2 = nn.Linear(projection_hidden_size,
                                 llm_hidden_size,
                                 bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = torch.pow(F.relu(x), 2)
        return self.linear2(x)


# ---------------------------------------------------------------------------
# Full audio model
# ---------------------------------------------------------------------------


class NemotronOmniAudioModel(nn.Module):
    """From-scratch Parakeet Conformer encoder + sound projection.

    Output: ``[batch, encoded_seq_len, llm_hidden_size]``
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        sc = config["sound_config"]
        llm_cfg = config.get("llm_config", config.get("text_config", {}))
        head_dim = sc["hidden_size"] // sc["num_attention_heads"]
        attention_scale = config_module._get_attention_scaling(
            sc, head_dim, 1.0 / (float(head_dim)**0.5))

        self.encoder = ParakeetEncoder(
            hidden_size=sc["hidden_size"],
            num_heads=sc["num_attention_heads"],
            num_layers=sc["num_hidden_layers"],
            intermediate_size=sc["intermediate_size"],
            mel_bins=sc.get("num_mel_bins", 128),
            conv_kernel_size=sc.get("conv_kernel_size", 9),
            conv_channels=sc.get("subsampling_conv_channels", 256),
            attention_scale=attention_scale,
        )
        self.projection = SoundProjection(
            sound_hidden_size=sc["hidden_size"],
            projection_hidden_size=sc.get("projection_hidden_size", 4096),
            llm_hidden_size=llm_cfg["hidden_size"],
            bias=sc.get("projection_bias", False),
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: [1, seq_len, mel_bins]

        Returns:
            [1, encoded_seq_len, llm_hidden_size]

        The encoder runs at batch=1; the C++ runtime loops over audio clips
        sequentially. The Conformer's depthwise conv is a local operator that
        ignores attention masks, so true cross-clip batching with padding
        causes numerical drift.  Single-clip-per-enqueue avoids that drift.
        """
        x = self.encoder(input_features)
        return self.projection(x)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes)."""
        mel_bins = config["sound_config"].get("num_mel_bins", 128)
        seq_len = 200
        input_features = torch.zeros(1,
                                     seq_len,
                                     mel_bins,
                                     dtype=torch.float16,
                                     device=device)
        args = (input_features, )
        input_names = ["input_features"]
        output_names = ["last_hidden_state"]
        S = torch.export.Dim("seq_len", min=8, max=16384)
        dynamic_shapes = {"input_features": {1: S}}
        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_weights(model: NemotronOmniAudioModel, weights: dict) -> None:
    """Load Parakeet encoder and sound projection weights.

    Checkpoint key → model attribute path:
      ``sound_encoder.encoder.*`` → ``encoder.*``
      ``sound_projection.*``      → ``projection.*``

    Uses ``_set_tensor`` (via ``load_submodule_weights``) so bf16 weights are
    automatically cast to fp16 — ``load_state_dict`` would skip that cast.
    """
    from ...checkpoint.loader import load_submodule_weights

    enc_prefix = "sound_encoder.encoder."
    proj_prefix = "sound_projection."

    def _remap(k: str) -> "str | None":
        if k.startswith(enc_prefix):
            return "encoder." + k[len(enc_prefix):]
        if k.startswith(proj_prefix):
            return "projection." + k[len(proj_prefix):]
        return None

    load_submodule_weights(model,
                           weights,
                           _remap,
                           label="NemotronOmniAudioModel")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_nemotron_omni_audio(
        config: dict,
        weights: dict,
        dtype: torch.dtype = torch.float16) -> NemotronOmniAudioModel:
    """Build and return a :class:`NemotronOmniAudioModel` with loaded weights.

    Args:
        config:  Full parsed ``config.json`` dict (must contain ``sound_config``).
        weights: Flat ``{key: tensor}`` dict from safetensors.
        dtype:   Target dtype (default ``float16``).
    """
    model = NemotronOmniAudioModel(config)
    model.to(dtype)
    _load_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "NemotronOmniAudioModel",
    "build_nemotron_omni_audio",
]
