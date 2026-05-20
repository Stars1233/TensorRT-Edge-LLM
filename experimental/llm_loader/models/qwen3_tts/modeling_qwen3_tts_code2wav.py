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
"""Qwen3-TTS Code2Wav export.

Qwen3-TTS stores its vocoder under the checkpoint-local ``speech_tokenizer/``
directory.  The Talker and CodePredictor are exported through the LLM path;
this module exports the final codec-token-to-waveform decoder used by the
C++ ``Code2WavRunner``.

The implementation below is intentionally standalone.  It does not import any
external Qwen3-TTS package during export; weights are loaded directly from
``speech_tokenizer/model.safetensors``.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

logger = logging.getLogger(__name__)

__all__ = ["Qwen3TTSCode2WavDecoder", "export_qwen3_tts_code2wav"]


@dataclass
class Qwen3TTSCode2WavConfig:
    codebook_size: int = 2048
    hidden_size: int = 1024
    latent_dim: int = 1024
    max_position_embeddings: int = 8000
    rope_theta: float = 10000.0
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    attention_bias: bool = False
    sliding_window: int = 72
    intermediate_size: int = 3072
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    rms_norm_eps: float = 1e-5
    num_hidden_layers: int = 8
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: tuple[int, ...] = (2, 2)
    decoder_dim: int = 1536
    attention_dropout: float = 0.0
    head_dim: int | None = None
    codebook_dim: int = 512

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Qwen3TTSCode2WavConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {key: value for key, value in config.items() if key in known}
        if "upsample_rates" in kwargs:
            kwargs["upsample_rates"] = tuple(kwargs["upsample_rates"])
        if "upsampling_ratios" in kwargs:
            kwargs["upsampling_ratios"] = tuple(kwargs["upsampling_ratios"])
        return cls(**kwargs)

    @property
    def layer_types(self) -> list[str]:
        return ["sliding_attention"] * self.num_hidden_layers

    @property
    def attention_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads


def _get_extra_padding(x_len: int, kernel_size: int, stride: int,
                       padding: int) -> int:
    n_frames = (x_len - kernel_size + padding) / stride + 1
    ideal_len = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding)
    return int(ideal_len - x_len)


class CausalConv1d(nn.Module):

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
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra = _get_extra_padding(x.shape[-1], self.kernel_size, self.stride,
                                   self.padding)
        x = F.pad(x, (self.padding, extra), mode="constant", value=0)
        return self.conv(x).contiguous()


class CausalTransposeConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride)
        self.right_pad = int(kernel_size - stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.right_pad > 0:
            x = x[..., :x.shape[-1] - self.right_pad]
        return x.contiguous()


class ConvNeXtBlock(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 2, 1)
        return residual + x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, position_embeddings: tuple[torch.Tensor,
                                                                 torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) *
                                                             sin)


class RotaryEmbedding(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig) -> None:
        super().__init__()
        self.head_dim = config.attention_head_dim
        self.rope_theta = config.rope_theta
        self.register_buffer("inv_freq",
                             self._compute_inv_freq(),
                             persistent=False)

    def _compute_inv_freq(self,
                          device: torch.device | None = None) -> torch.Tensor:
        return 1.0 / (self.rope_theta**(torch.arange(
            0, self.head_dim, 2, device=device).float() / self.head_dim))

    def _apply(self, fn: Any) -> "RotaryEmbedding":
        super()._apply(fn)
        self.inv_freq = self._compute_inv_freq(self.inv_freq.device)
        return self

    def forward(
            self, x: torch.Tensor,
            position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :,
                                 None].float().expand(position_ids.shape[0],
                                                      -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(
            x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ position_ids.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len,
                                   head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class Attention(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig, layer_idx: int) -> None:
        super().__init__()
        del layer_idx
        self.head_dim = config.attention_head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (config.num_attention_heads //
                                     config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size,
                                config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim,
                                config.hidden_size,
                                bias=config.attention_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_attention_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(q_shape))
        key = self.k_norm(self.k_proj(hidden_states).view(kv_shape))
        value = self.v_proj(hidden_states).view(kv_shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = _apply_rotary_pos_emb(query, key, position_embeddings)
        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask[:, :, :, :key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1,
                                 dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class Mlp(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size,
                                   config.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(config.hidden_size,
                                 config.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,
                                   config.hidden_size,
                                   bias=False)
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported Qwen3-TTS Code2Wav activation: {config.hidden_act}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((config.hidden_size, ),
                       config.layer_scale_initial_scale,
                       requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class TransformerLayer(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = Mlp(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)
        self.self_attn_layer_scale = LayerScale(config)
        self.mlp_layer_scale = LayerScale(config)
        self.attention_type = "sliding_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings,
                                       attention_mask)
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.mlp_layer_scale(hidden_states)


class TransformerModel(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.window_size = config.sliding_window
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.input_proj(inputs_embeds)
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len,
                                    device=inputs_embeds.device).unsqueeze(0)
        q_pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        valid_mask = (k_pos > q_pos - self.window_size) & (k_pos <= q_pos)
        attention_mask = torch.where(
            valid_mask,
            torch.zeros(valid_mask.shape,
                        dtype=inputs_embeds.dtype,
                        device=inputs_embeds.device),
            torch.full(valid_mask.shape,
                       torch.finfo(inputs_embeds.dtype).min,
                       dtype=inputs_embeds.dtype,
                       device=inputs_embeds.device),
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings,
                                          attention_mask)
        hidden_states = self.norm(hidden_states)
        return self.output_proj(hidden_states)


class SnakeBeta(nn.Module):

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class DecoderResidualUnit(nn.Module):

    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class DecoderBlock(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig, layer_idx: int) -> None:
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2**(layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]
        block: list[nn.Module] = [
            SnakeBeta(in_dim),
            CausalTransposeConv1d(in_dim,
                                  out_dim,
                                  kernel_size=2 * upsample_rate,
                                  stride=upsample_rate),
        ]
        for dilation in (1, 3, 9):
            block.append(DecoderResidualUnit(out_dim, dilation))
        self.block = nn.ModuleList(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        return x


class EuclideanCodebook(nn.Module):

    def __init__(self,
                 dim: int,
                 codebook_size: int,
                 epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))
        self.register_buffer("_embedding", torch.empty(0), persistent=False)

    def materialize_embedding(self) -> None:
        self._embedding = (
            self.embedding_sum /
            self.cluster_usage.clamp(min=self.epsilon)[:, None]).detach()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if self._embedding.numel():
            return F.embedding(codes, self._embedding)
        embedding = self.embedding_sum / self.cluster_usage.clamp(
            min=self.epsilon)[:, None]
        return F.embedding(codes, embedding)


class VectorQuantization(nn.Module):

    def __init__(self,
                 dim: int,
                 codebook_size: int,
                 codebook_dim: int | None = None,
                 epsilon: float = 1e-5) -> None:
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim
        self.project_out = (nn.Linear(codebook_dim, dim)
                            if codebook_dim != dim else nn.Identity())
        self._codebook = EuclideanCodebook(dim=codebook_dim,
                                           codebook_size=codebook_size,
                                           epsilon=epsilon)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        return quantized.transpose(1, 2)


class ResidualVectorQuantization(nn.Module):

    def __init__(self, num_quantizers: int, **kwargs: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.tensor(0.0, device=codes.device)
        for idx, layer in enumerate(self.layers):
            quantized = quantized + layer.decode(codes[idx])
        return quantized


class ResidualVectorQuantizer(nn.Module):

    def __init__(self,
                 dimension: int,
                 input_dimension: int,
                 output_dimension: int,
                 n_q: int,
                 bins: int,
                 force_projection: bool = False) -> None:
        super().__init__()
        if input_dimension == dimension and not force_projection:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Conv1d(input_dimension,
                                        dimension,
                                        1,
                                        bias=False)
        if output_dimension == dimension and not force_projection:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Conv1d(dimension,
                                         output_dimension,
                                         1,
                                         bias=False)
        self.vq = ResidualVectorQuantization(dim=dimension,
                                             codebook_size=bins,
                                             num_quantizers=n_q)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return self.output_proj(quantized)


class SplitResidualVectorQuantizer(nn.Module):

    def __init__(self, n_q: int, n_q_semantic: int, dimension: int,
                 input_dimension: int, output_dimension: int,
                 bins: int) -> None:
        super().__init__()
        if n_q <= n_q_semantic:
            raise ValueError("n_q must be larger than n_q_semantic")
        self.n_q_semantic = n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic,
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            bins=bins,
            force_projection=True,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            bins=bins,
            force_projection=True,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.rvq_first.decode(codes[:, :self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(
                codes[:, self.n_q_semantic:])
        return quantized


class Qwen3TTSCode2WavDecoder(nn.Module):

    def __init__(self, config: Qwen3TTSCode2WavConfig) -> None:
        super().__init__()
        self.config = config
        self.total_upsample = math.prod(config.upsample_rates +
                                        config.upsampling_ratios)
        self.pre_transformer = TransformerModel(config)
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=config.num_semantic_quantizers,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )
        self.pre_conv = CausalConv1d(config.codebook_dim,
                                     config.latent_dim,
                                     kernel_size=3)

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList([
                    CausalTransposeConv1d(config.latent_dim,
                                          config.latent_dim,
                                          kernel_size=factor,
                                          stride=factor),
                    ConvNeXtBlock(config.latent_dim),
                ]))
        self.upsample = nn.ModuleList(upsample)

        decoder: list[nn.Module] = [
            CausalConv1d(config.latent_dim, config.decoder_dim, 7)
        ]
        for idx in range(len(config.upsample_rates)):
            decoder.append(DecoderBlock(config, idx))
        output_dim = config.decoder_dim // 2**len(config.upsample_rates)
        decoder.extend([SnakeBeta(output_dim), CausalConv1d(output_dim, 1, 7)])
        self.decoder = nn.ModuleList(decoder)

    def materialize_codebook_embeddings(self) -> None:
        for module in self.modules():
            if isinstance(module, EuclideanCodebook):
                module.materialize_embedding()

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layers of codes, "
                f"got {codes.shape[1]}")
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1).to(torch.float32)


def _load_speech_tokenizer_weights(
        tokenizer_subdir: str) -> dict[str, torch.Tensor]:
    safetensors_files = sorted(
        glob.glob(os.path.join(tokenizer_subdir, "*.safetensors")))
    if not safetensors_files:
        raise ValueError("Qwen3-TTS Code2Wav export requires speech_tokenizer "
                         f"safetensors weights under {tokenizer_subdir}")

    state_dict: dict[str, torch.Tensor] = {}
    for path in safetensors_files:
        state_dict.update(load_safetensors_file(path, device="cpu"))

    decoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("decoder."):
            decoder_state_dict[key[len("decoder."):]] = value

    if not decoder_state_dict:
        raise ValueError(
            f"No decoder weights found in speech_tokenizer weights: {tokenizer_subdir}"
        )
    return decoder_state_dict


def build_qwen3_tts_code2wav(
    model_dir: str, torch_dtype: torch.dtype
) -> tuple[Qwen3TTSCode2WavDecoder, dict[str, Any]]:
    tokenizer_subdir = os.path.join(model_dir, "speech_tokenizer")
    if not os.path.isdir(tokenizer_subdir):
        raise ValueError(
            "Qwen3-TTS Code2Wav export requires a local speech_tokenizer "
            f"directory at {tokenizer_subdir}")

    with open(os.path.join(tokenizer_subdir, "config.json")) as f:
        tokenizer_config = json.load(f)
    decoder_config_dict = tokenizer_config["decoder_config"]
    decoder_config = Qwen3TTSCode2WavConfig.from_dict(decoder_config_dict)
    model = Qwen3TTSCode2WavDecoder(decoder_config)
    state_dict = _load_speech_tokenizer_weights(tokenizer_subdir)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(dtype=torch_dtype)
    model.materialize_codebook_embeddings()
    return model, decoder_config_dict


def export_qwen3_tts_code2wav(model_dir: str,
                              output_dir: str,
                              torch_dtype: torch.dtype,
                              device: str = "cuda") -> None:
    """Export Qwen3-TTS ``speech_tokenizer`` decoder to ONNX."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")
    for external_data_name in ("onnx_model.data", "model.onnx.data"):
        external_data_path = os.path.join(output_dir, external_data_name)
        if os.path.exists(external_data_path):
            os.remove(external_data_path)

    logger.info("[Code2Wav:TTS] Loading speech tokenizer from %s",
                os.path.join(model_dir, "speech_tokenizer"))
    decoder, decoder_config_dict = build_qwen3_tts_code2wav(
        model_dir, torch_dtype)
    decoder.to(device)

    num_quantizers = decoder.config.num_quantizers
    codes = torch.randint(0,
                          decoder.config.codebook_size,
                          (1, num_quantizers, 128),
                          dtype=torch.int64,
                          device=next(decoder.parameters()).device)

    logger.info("[Code2Wav:TTS] Exporting ONNX to %s", output_path)
    batch_dim = torch.export.Dim("batch", min=1, max=256)
    code_len_dim = torch.export.Dim("code_len", min=1, max=8192)
    export_program = torch.onnx.export(
        decoder,
        (codes, ),
        dynamo=True,
        opset_version=22,
        input_names=["codes"],
        output_names=["waveform"],
        dynamic_shapes=({
            0: batch_dim,
            2: code_len_dim,
        }, ),
        external_data=True,
        optimize=True,
    )
    export_program.save(output_path,
                        include_initializers=True,
                        keep_initializers_as_inputs=False,
                        external_data=True)

    config_dict = dict(decoder_config_dict)
    config_dict["model_type"] = "qwen3_tts_code2wav"
    cfg_out_path = os.path.join(output_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("[Code2Wav:TTS] Wrote config.json: %s", cfg_out_path)
