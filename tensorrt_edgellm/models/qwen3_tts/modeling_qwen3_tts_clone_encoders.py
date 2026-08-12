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
"""Qwen3-TTS voice-clone reference encoders: ONNX export for TensorRT.

Exports two engines consumed by the C++ CloneEncoderRunner (Base checkpoints):

- ``speaker_encoder.onnx``: 24 kHz waveform ``[1, T]`` (dynamic) -> x-vector
  ``[1, enc_dim]``. The mel front-end (reflect pad, STFT 1024/256 with a Hann
  window, slaney mel-128 filterbank, log compression) is folded into the graph
  as a conv-DFT so the runtime feeds raw PCM.
- ``speech_tokenizer_encoder.onnx``: 24 kHz waveform ``[1, BUCKET]`` (static
  40 s bucket) -> RVQ codes ``[BUCKET_FRAMES, 16]``. The bucket is static
  because the Mimi causal-padding shape algebra does not compile under Myelin
  with a dynamic length; the encoder is causal, so zero-padding the tail does
  not disturb leading frames and the runtime keeps the first
  ``floor(num_samples / downsample_rate)`` complete frames.

The ECAPA-TDNN speaker-encoder modules below are ported from the Qwen3-TTS
reference implementation (Apache-2.0) so the export has no ``qwen_tts``
package dependency; weights load from the main checkpoint's
``speaker_encoder.*`` tensors.
"""

import contextlib
import json
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CLONE_BUCKET_SECONDS = 40
SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# ECAPA-TDNN speaker encoder (ported from the Qwen3-TTS reference, Apache-2.0)
# ---------------------------------------------------------------------------
class Res2NetBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale=8,
                 kernel_size=3,
                 dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList([
            TimeDelayNetBlock(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                dilation=dilation,
            ) for i in range(scale - 1)
        ])
        self.scale = scale

    def forward(self, hidden_states):
        outputs = []
        for i, hidden_part in enumerate(
                torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


class SqueezeExcitationBlock(nn.Module):

    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence.

        Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.
        """

        if max_len is None:
            max_len = length.max().long().item(
            )  # using arange to generate mask
        mask = torch.arange(max_len,
                            device=length.device, dtype=length.dtype).expand(
                                len(length), max_len) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(dim)
        std = torch.sqrt(
            (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0],
                             device=hidden_states.device)

        # Make binary mask of shape [N, 1, L]
        mask = self._length_to_mask(lengths * seq_length,
                                    max_len=seq_length,
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        # Apply layers
        attention = self.conv(self.tanh(self.tdnn(attention)))

        # Filter out zero-paddings
        attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=2)
        mean, std = self._compute_statistics(hidden_states, attention)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class TimeDelayNetBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))


class SqueezeExcitationRes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SqueezeExcitationBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels,
                                          res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels,
                                               out_channels)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        return hidden_state + residual


class Qwen3TTSSpeakerEncoder(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://huggingface.co/papers/2005.07143).
    Use for Qwen3TTS extract speaker embedding.
    """

    def __init__(self, config: "_SpeakerEncoderConfig"):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(
                config.enc_channels) != len(config.enc_dilations):
            raise ValueError(
                "enc_channels, enc_kernel_sizes and enc_dilations should have same length"
            )
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            ))

        # SE-Res2Net layers
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                ))

        # Multi-layer feature aggregation
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        # Minimize transpose for efficiency
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states_list = []
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        # Multi-layer feature aggregation
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)

        # Attentive Statistical Pooling
        hidden_states = self.asp(hidden_states)

        # Final linear transformation
        hidden_states = self.fc(hidden_states)

        hidden_states = hidden_states.squeeze(-1)
        return hidden_states


class _SpeakerEncoderConfig:
    """Attribute shim matching Qwen3TTSSpeakerEncoderConfig fields used above."""

    def __init__(self,
                 mel_dim=128,
                 enc_dim=2048,
                 enc_channels=(512, 512, 512, 512, 1536),
                 enc_kernel_sizes=(5, 3, 3, 3, 1),
                 enc_dilations=(1, 2, 3, 4, 1),
                 enc_attention_channels=128,
                 enc_res2net_scale=8,
                 enc_se_channels=128):
        self.mel_dim = mel_dim
        self.enc_dim = enc_dim
        self.enc_channels = list(enc_channels)
        self.enc_kernel_sizes = list(enc_kernel_sizes)
        self.enc_dilations = list(enc_dilations)
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels


# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------
def _slaney_mel_filterbank(num_mels: int, n_fft: int, fmin: float,
                           fmax: float) -> torch.Tensor:
    """Slaney-norm mel filterbank identical to librosa.filters.mel defaults."""

    def hz_to_mel(f):
        f = torch.as_tensor(f, dtype=torch.float64)
        mel = 3.0 * f / 200.0
        log_region = f >= 1000.0
        mel = torch.where(
            log_region, 15.0 + torch.log(f.clamp(min=1e-10) / 1000.0) /
            (math.log(6.4) / 27.0), mel)
        return mel

    def mel_to_hz(m):
        m = torch.as_tensor(m, dtype=torch.float64)
        f = 200.0 * m / 3.0
        log_region = m >= 15.0
        f = torch.where(
            log_region, 1000.0 * torch.exp(
                (math.log(6.4) / 27.0) * (m - 15.0)), f)
        return f

    fft_freqs = torch.arange(n_fft // 2 + 1,
                             dtype=torch.float64) * SAMPLE_RATE / n_fft
    mel_pts = mel_to_hz(
        torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), num_mels + 2))
    fdiff = mel_pts[1:] - mel_pts[:-1]
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)
    enorm = 2.0 / (mel_pts[2:num_mels + 2] - mel_pts[:num_mels])
    return (weights * enorm[:, None]).float()


class SpeakerEncoderExportWrapper(nn.Module):
    """wav [1, T] -> x-vector [1, enc_dim], mel front-end folded in as conv-DFT."""

    def __init__(self,
                 speaker_encoder: Qwen3TTSSpeakerEncoder,
                 num_mels: int = 128):
        super().__init__()
        self.encoder = speaker_encoder
        n_fft, hop = 1024, 256
        window = torch.hann_window(n_fft)
        k = torch.arange(n_fft, dtype=torch.float64)
        bins = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        ang = 2.0 * math.pi * bins[:, None] * k[None, :] / n_fft
        basis = torch.cat([torch.cos(ang), -torch.sin(ang)],
                          dim=0).float() * window[None, :]
        self.register_buffer("dft", basis.unsqueeze(1))
        self.register_buffer(
            "mel_fb",
            _slaney_mel_filterbank(num_mels, n_fft, 0.0, SAMPLE_RATE / 2.0))
        self.hop = hop
        self.pad = (n_fft - hop) // 2
        self.n_bins = n_fft // 2 + 1

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        y = F.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(y, self.dft, stride=self.hop)
        re, im = spec[:, :self.n_bins], spec[:, self.n_bins:]
        mag = torch.sqrt(re * re + im * im + 1e-9)
        mel = torch.log(torch.clamp(self.mel_fb @ mag, min=1e-5))
        return self.encoder(mel.transpose(1, 2))


class TokenizerEncoderExportWrapper(nn.Module):
    """wav [1, BUCKET] -> codes [BUCKET_FRAMES, num_quantizers] via Mimi encode."""

    def __init__(self, mimi_model, num_quantizers: int):
        super().__init__()
        self.mimi = mimi_model
        self.num_quantizers = num_quantizers

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        enc = self.mimi.encode(input_values=wav.unsqueeze(1), return_dict=True)
        return enc.audio_codes[0, :self.num_quantizers].transpose(0, 1)


@contextlib.contextmanager
def _patch_mimi_for_export():
    """Two export-blocking spots in transformers Mimi, patched for the export scope.

    Original methods are restored on exit so other Mimi users in the same
    process are unaffected.

    - torch.cdist has no torch.export decomposition; the algebraic expansion
      ||x-e||^2 = ||x||^2 - 2 x.e + ||e||^2 is argmin-equivalent.
    - The float ceil in the causal-conv padding computation traces into a
      dynamic-shape expression Myelin rejects; integer ceil-div is exact.
    """
    import transformers.models.mimi.modeling_mimi as mimi_mod

    def quantize(self, hidden_states):
        x = hidden_states.float()
        e = self.embed.float()
        dists = (x * x).sum(
            -1, keepdim=True) - 2.0 * x @ e.t() + (e * e).sum(-1)[None, :]
        return dists.argmin(dim=-1)

    def extra_padding(self, hidden_states):
        length = hidden_states.shape[-1]
        numerator = length - self.kernel_size + self.padding_total
        n_frames_minus1 = torch.div(numerator + self.stride - 1,
                                    self.stride,
                                    rounding_mode="floor")
        return n_frames_minus1 * self.stride + self.kernel_size - self.padding_total - length

    orig_quantize = mimi_mod.MimiEuclideanCodebook.quantize
    orig_padding = mimi_mod.MimiConv1d._get_extra_padding_for_conv1d
    mimi_mod.MimiEuclideanCodebook.quantize = quantize
    mimi_mod.MimiConv1d._get_extra_padding_for_conv1d = extra_padding
    try:
        yield
    finally:
        mimi_mod.MimiEuclideanCodebook.quantize = orig_quantize
        mimi_mod.MimiConv1d._get_extra_padding_for_conv1d = orig_padding


def export_qwen3_tts_clone_encoders(model_dir: str, output_dir: str) -> None:
    """Export both voice-clone reference encoders to ONNX (FP32; build with --fp16)."""
    from safetensors.torch import load_file

    os.makedirs(output_dir, exist_ok=True)

    # --- speaker encoder: ECAPA from the main checkpoint's speaker_encoder.* ---
    with open(os.path.join(model_dir, "config.json")) as f:
        root_cfg = json.load(f)
    spk_cfg = root_cfg["speaker_encoder_config"]
    encoder = Qwen3TTSSpeakerEncoder(
        _SpeakerEncoderConfig(
            **{
                k: spk_cfg[k]
                for k in ("mel_dim", "enc_dim", "enc_channels",
                          "enc_kernel_sizes", "enc_dilations",
                          "enc_attention_channels", "enc_res2net_scale",
                          "enc_se_channels") if k in spk_cfg
            })).eval()
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    spk_state = {
        k[len("speaker_encoder."):]: v.float()
        for k, v in state.items() if k.startswith("speaker_encoder.")
    }
    encoder.load_state_dict(spk_state)
    del state

    wrapper = SpeakerEncoderExportWrapper(encoder,
                                          num_mels=spk_cfg.get("mel_dim",
                                                               128)).eval()
    wav = torch.randn(1, SAMPLE_RATE * 3)
    out_path = os.path.join(output_dir, "speaker_encoder.onnx")
    logger.info("[CloneEncoders] Exporting %s", out_path)
    torch.onnx.export(wrapper, (wav, ),
                      out_path,
                      input_names=["wav"],
                      output_names=["speaker_embedding"],
                      dynamic_axes={"wav": {
                          1: "num_samples"
                      }},
                      opset_version=20)

    # --- speech tokenizer encoder: Mimi encoder-only from speech_tokenizer/ ---
    from transformers.models.mimi import MimiConfig, MimiModel
    st_dir = os.path.join(model_dir, "speech_tokenizer")
    with open(os.path.join(st_dir, "config.json")) as f:
        st_cfg = json.load(f)
    with _patch_mimi_for_export():
        mimi = MimiModel(MimiConfig(**st_cfg["encoder_config"])).eval()
        st_state = load_file(os.path.join(st_dir, "model.safetensors"))
        enc_state = {
            k[len("encoder."):]: v.float()
            for k, v in st_state.items() if k.startswith("encoder.")
        }
        missing, unexpected = mimi.load_state_dict(enc_state, strict=False)
        dec_leftover = [
            k for k in missing
            if not k.startswith(("decoder", "upsample", "decoder_transformer"))
        ]
        if dec_leftover:
            raise RuntimeError(
                f"Mimi encoder weights missing: {dec_leftover[:5]}")
        del st_state

        tok = TokenizerEncoderExportWrapper(
            mimi, st_cfg["encoder_valid_num_quantizers"]).eval()
        bucket = torch.randn(1, SAMPLE_RATE * CLONE_BUCKET_SECONDS)
        out_path = os.path.join(output_dir, "speech_tokenizer_encoder.onnx")
        logger.info("[CloneEncoders] Exporting %s (static %ds bucket)",
                    out_path, CLONE_BUCKET_SECONDS)
        torch.onnx.export(tok, (bucket, ),
                          out_path,
                          input_names=["wav"],
                          output_names=["codes"],
                          opset_version=20)
        logger.info("[CloneEncoders] Done: %s", output_dir)
