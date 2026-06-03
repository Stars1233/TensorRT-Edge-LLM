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
"""Pure-PyTorch Qwen3-ASR audio encoder for calibration.

Whisper-style encoder identical in structure to the runtime modeling in
:mod:`tensorrt_edgellm.models.qwen3_asr.modeling_qwen3_asr_audio`,
but every Linear here is a plain :class:`torch.nn.Linear` and weight
loading is the trivial ``load_state_dict(strict=False)`` path. The
runtime sibling is responsible for routing Linears through
``make_linear`` so the resulting ONNX graph picks up FP8 / NVFP4
custom kernels; this calibration-side copy does not need any of that.

The two files are intentionally not merged: the runtime side carries
quant-aware Linear factories and ONNX-friendly tricks that have no place
inside a ModelOpt calibration tree. Keeping them separate is the explicit
design contract — :mod:`tensorrt_edgellm.quantization` must not import from
:mod:`tensorrt_edgellm`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default architecture constants (Qwen3-ASR 1.7B / Qwen3-Omni 30B).
# Qwen3-ASR-0.6B overrides every one of these via its config.json; defaults
# are kept so callers can build the encoder for tests without a config.
_D_MODEL = 1280
_NUM_LAYERS = 32
_NUM_HEADS = 20
_FFN_DIM = 5120
_NUM_MEL_BINS = 128
_MAX_SOURCE_POSITIONS = 1500
_OUTPUT_DIM = 3584
_DOWNSAMPLE_HIDDEN = 480
_N_WINDOW = 50  # Both 0.6B and 1.7B use 50; runtime overrides from HF config.


class SinusoidsPositionEmbedding(nn.Module):
    """Fixed sinusoidal position encoding (Whisper / Qwen3-ASR layout)."""

    def __init__(self,
                 length: int = _MAX_SOURCE_POSITIONS,
                 channels: int = _D_MODEL,
                 max_timescale: int = 10000) -> None:
        super().__init__()
        assert channels % 2 == 0
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment *
            torch.arange(channels // 2, dtype=torch.float32))
        scaled_time = torch.arange(
            length,
            dtype=torch.float32).unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat([torch.sin(scaled_time),
                             torch.cos(scaled_time)],
                            dim=1)
        self.register_buffer("positional_embedding", pos_emb, persistent=False)


class Qwen3ASRAudioAttention(nn.Module):
    """Multi-head self-attention for the audio encoder.

    Checkpoint keys (under ``layers.N.self_attn``)::

        q_proj.{weight,bias}, k_proj.{weight,bias},
        v_proj.{weight,bias}, out_proj.{weight,bias}
    """

    def __init__(self,
                 d_model: int = _D_MODEL,
                 num_heads: int = _NUM_HEADS) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(T, self.num_heads,
                                            self.head_dim).transpose(0, 1)
        k = self.k_proj(hidden_states).view(T, self.num_heads,
                                            self.head_dim).transpose(0, 1)
        v = self.v_proj(hidden_states).view(T, self.num_heads,
                                            self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        scores = scores + attention_mask.unsqueeze(0)
        attn_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(0, 1).reshape(T, -1)
        return self.out_proj(out)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """Single Qwen3-ASR encoder block (pre-norm)."""

    def __init__(self,
                 d_model: int = _D_MODEL,
                 num_heads: int = _NUM_HEADS,
                 ffn_dim: int = _FFN_DIM) -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = Qwen3ASRAudioAttention(d_model, num_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_val = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_val,
                                        max=clamp_val)
        return hidden_states


class Qwen3ASRAudioEncoder(nn.Module):
    """Complete Qwen3-ASR / Qwen3-Omni audio encoder."""

    def __init__(self,
                 num_mel_bins: int = _NUM_MEL_BINS,
                 d_model: int = _D_MODEL,
                 num_layers: int = _NUM_LAYERS,
                 num_heads: int = _NUM_HEADS,
                 ffn_dim: int = _FFN_DIM,
                 max_source_positions: int = _MAX_SOURCE_POSITIONS,
                 output_dim: int = _OUTPUT_DIM,
                 downsample_hidden: int = _DOWNSAMPLE_HIDDEN) -> None:
        super().__init__()
        # Architecture knobs the dataloader needs to read back at runtime
        # (n_window / num_mel_bins drive chunk-shape computation).
        self.config: Dict[str, Any] = {
            "num_mel_bins": num_mel_bins,
            "d_model": d_model,
            "encoder_layers": num_layers,
            "encoder_attention_heads": num_heads,
            "encoder_ffn_dim": ffn_dim,
            "max_source_positions": max_source_positions,
            "output_dim": output_dim,
            "downsample_hidden_size": downsample_hidden,
        }

        self.conv2d1 = nn.Conv2d(1,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv2d2 = nn.Conv2d(downsample_hidden,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv2d3 = nn.Conv2d(downsample_hidden,
                                 downsample_hidden,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        freq_bins = ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(downsample_hidden * freq_bins,
                                  d_model,
                                  bias=False)
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_source_positions, d_model)
        self.layers = nn.ModuleList([
            Qwen3ASRAudioEncoderLayer(d_model, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model, bias=True)
        self.proj2 = nn.Linear(d_model, output_dim, bias=True)

    def forward(
        self,
        padded_feature: torch.Tensor,
        padded_mask_after_cnn_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = padded_feature.unsqueeze(1)  # [C, 1, mel, t]
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        b, c, f, t = x.shape
        x = self.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        pos = self.positional_embedding.positional_embedding[:x.shape[
            1], :].unsqueeze(0).to(x.dtype)
        x = x + pos

        hidden_states = x[padded_mask_after_cnn_indices[:, 0],
                          padded_mask_after_cnn_indices[:, 1]]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


def _cnn_output_length(in_length: int) -> int:
    """Time dimension after 3x stride-2 conv layers (kernel 3, padding 1)."""
    L = in_length
    for _ in range(3):
        L = (L + 2 * 1 - 3) // 2 + 1
    return L


def prepare_audio_inputs(
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    n_window: int,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Build the ragged ``forward()`` inputs from natural mel features + lengths.

    Mirrors what the C++ runtime computes during audio pre-processing.

    Args:
        input_features: ``[B, num_mel_bins, T_mel]`` mel spectrogram.
        feature_lens:   ``[B]`` int64, valid mel frames per row.
        n_window:       Encoder ``n_window`` (50 for both 0.6B and 1.7B;
            runtime always overrides this from HF config). Each chunk
            holds ``n_window * 2`` mel frames.
        dtype:          Output dtype for ``padded_feature`` /
            ``attention_mask``.

    Returns:
        Dict with the three tensors the encoder ``forward()`` expects::

            {
                "padded_feature":                 [C, num_mel_bins, n_window*2],
                "padded_mask_after_cnn_indices":  [T, 2]   int64,
                "attention_mask":                 [T, T]   <dtype>,
            }
    """
    if input_features.dim() != 3:
        raise ValueError(f"input_features must be [B, mel, T], got shape "
                         f"{tuple(input_features.shape)}")
    if feature_lens.dim(
    ) != 1 or feature_lens.shape[0] != input_features.shape[0]:
        raise ValueError(
            f"feature_lens must be [B], got shape "
            f"{tuple(feature_lens.shape)} for batch={input_features.shape[0]}")

    chunk_length = n_window * 2
    device = input_features.device

    chunk_feats: List[torch.Tensor] = []
    chunk_post_cnn_lens: List[int] = []

    for b in range(input_features.shape[0]):
        T_b = int(feature_lens[b].item())
        full_chunks, tail = divmod(T_b, chunk_length)
        chunk_lens_b = [chunk_length] * full_chunks
        if tail > 0:
            chunk_lens_b.append(tail)
        elif not chunk_lens_b:
            chunk_lens_b.append(chunk_length)

        offset = 0
        for L in chunk_lens_b:
            slc = input_features[b, :, offset:offset + L]
            if L < chunk_length:
                slc = F.pad(slc, (0, chunk_length - L))
            chunk_feats.append(slc)
            chunk_post_cnn_lens.append(_cnn_output_length(L))
            offset += L

    padded_feature = torch.stack(chunk_feats, dim=0).to(dtype=dtype,
                                                        device=device)

    cnn_out_max = _cnn_output_length(chunk_length)
    indices_list: List[torch.Tensor] = []
    for c, L in enumerate(chunk_post_cnn_lens):
        if L > 0:
            ts = torch.arange(L, dtype=torch.int64)
            cs = torch.full((L, ), c, dtype=torch.int64)
            indices_list.append(torch.stack([cs, ts], dim=1))
    if indices_list:
        padded_mask_after_cnn_indices = torch.cat(indices_list,
                                                  dim=0).to(device=device)
    else:
        padded_mask_after_cnn_indices = torch.zeros((0, 2),
                                                    dtype=torch.int64,
                                                    device=device)

    T = padded_mask_after_cnn_indices.shape[0]
    fill_val = torch.finfo(dtype).min if dtype.is_floating_point else float(
        "-inf")
    attention_mask = torch.full((T, T), fill_val, dtype=dtype, device=device)
    cursor = 0
    for L in chunk_post_cnn_lens:
        if L > 0:
            attention_mask[cursor:cursor + L, cursor:cursor + L] = 0
            cursor += L

    assert cursor == T, f"cursor={cursor} != T={T} in prepare_audio_inputs"
    assert cnn_out_max >= max(chunk_post_cnn_lens, default=0)

    return {
        "padded_feature": padded_feature,
        "padded_mask_after_cnn_indices": padded_mask_after_cnn_indices,
        "attention_mask": attention_mask,
    }


_CANDIDATE_PREFIXES = (
    "audio_tower.",
    "thinker.audio_tower.",
    "model.audio_tower.",
    "",
)


def _strip_audio_prefix(weights: Dict[str, torch.Tensor],
                        prefix: Optional[str]) -> Dict[str, torch.Tensor]:
    if prefix is None:
        for cand in _CANDIDATE_PREFIXES:
            if cand == "" or any(k.startswith(cand) for k in weights.keys()):
                prefix = cand
                break
        else:
            prefix = ""
    stripped: Dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if not k.startswith(prefix):
            continue
        stripped[k[len(prefix):]] = v
    return stripped


def build_qwen3_asr_audio(
    config: Dict[str, Any],
    weights: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
    prefix: Optional[str] = None,
) -> Qwen3ASRAudioEncoder:
    """Build a :class:`Qwen3ASRAudioEncoder` with loaded weights.

    Args:
        config:  Audio architecture config dict. Accepts either a top-level
            qwen3_asr ``config.json`` (looks under ``thinker_config.audio_config``)
            or a flat audio-only sub-config.
        weights: Flat ``{key: tensor}`` dict from safetensors.
        dtype:   Target dtype (default ``float16``).
        prefix:  Checkpoint key prefix to strip. ``None`` = auto-detect.
    """
    thinker_cfg = config.get("thinker_config", {})
    audio_cfg = (thinker_cfg.get("audio_config") or config.get("audio_config")
                 or config)

    def _get(key: str, default: Any) -> Any:
        return audio_cfg.get(key, config.get(key, default))

    model = Qwen3ASRAudioEncoder(
        num_mel_bins=_get("num_mel_bins", _NUM_MEL_BINS),
        d_model=_get("d_model", _D_MODEL),
        num_layers=_get("encoder_layers", _NUM_LAYERS),
        num_heads=_get("encoder_attention_heads", _NUM_HEADS),
        ffn_dim=_get("encoder_ffn_dim", _FFN_DIM),
        max_source_positions=_get("max_source_positions",
                                  _MAX_SOURCE_POSITIONS),
        output_dim=_get("output_dim", _OUTPUT_DIM),
        downsample_hidden=_get("downsample_hidden_size", _DOWNSAMPLE_HIDDEN),
    )
    model.config["n_window"] = _get("n_window", _N_WINDOW)
    # 0.6B and 1.7B both ship n_window_infer = 800 = 16 * 50; runtime
    # always overrides this from HF config, the default is just for
    # tests that build the encoder from scratch.
    model.config["n_window_infer"] = _get("n_window_infer", _N_WINDOW * 16)

    model = model.to(dtype=dtype)
    state = _strip_audio_prefix(weights, prefix)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Qwen3ASRAudioEncoder: %d missing keys (first 5: %s)",
                       len(missing), missing[:5])
    if unexpected:
        logger.warning(
            "Qwen3ASRAudioEncoder: %d unexpected keys (first 5: %s)",
            len(unexpected), unexpected[:5])
    model.eval()
    return model


__all__ = [
    "SinusoidsPositionEmbedding",
    "Qwen3ASRAudioAttention",
    "Qwen3ASRAudioEncoderLayer",
    "Qwen3ASRAudioEncoder",
    "build_qwen3_asr_audio",
    "prepare_audio_inputs",
]
