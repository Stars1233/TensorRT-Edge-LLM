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
Audio model quantization for TensorRT Edge-LLM.

This module provides FP8 quantization for audio encoder models (Qwen3-ASR,
Qwen3-Omni audio encoder) using NVIDIA ModelOpt with LibriSpeech ASR as
the default calibration dataset.
"""

import io

import modelopt.torch.quantization as mtq
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset

from ..scripts.preprocess_audio import extract_mel_spectrogram
from .quantization_utils import quantize_model


class AudioCalibrationDataset(Dataset):

    def __init__(self, data, preprocess_fn):
        self.data = data
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_data = self.data[idx]
        return self.preprocess_fn(raw_data)


def _compute_cnn_output_length(model, num_mel_bins, chunk_length, dtype,
                               device):
    """Probe the CNN layers to determine the temporal output dimension for a given chunk length."""
    dummy = torch.randn(1,
                        1,
                        num_mel_bins,
                        chunk_length,
                        dtype=dtype,
                        device=device)
    with torch.no_grad():
        x = torch.nn.functional.gelu(model.conv2d1(dummy))
        x = torch.nn.functional.gelu(model.conv2d2(x))
        x = torch.nn.functional.gelu(model.conv2d3(x))
    return x.shape[3]


def get_audio_calib_dataloader(model,
                               dataset_dir="openslr/librispeech_asr",
                               max_chunks=4,
                               num_calib_samples=1000):
    """
    Build a calibration dataset from LibriSpeech ASR audio samples.

    Each sample is converted to a mel spectrogram, chunked to full windows,
    and formatted as the patched audio encoder's forward inputs:
    (padded_feature, padded_mask_after_cnn_indices, attention_mask).

    Args:
        model: The patched audio encoder model (Qwen3ASRAudioEncoderPatch
            or Qwen3OmniAudioEncoderPatch).
        dataset_dir: HuggingFace dataset identifier (default: openslr/librispeech_asr).
        max_chunks: Maximum number of chunks per sample to cap memory usage.
        num_calib_samples: Maximum number of dataset samples to use for calibration.

    Returns:
        AudioCalibrationDataset yielding dicts of model inputs.
    """
    assert "librispeech" in dataset_dir.lower(), \
        f"Unsupported dataset: {dataset_dir}. Only librispeech_asr is supported."

    # Stream with audio decoding disabled to avoid the torchcodec dependency.
    # Audio bytes are decoded manually with soundfile in the preprocess function.
    dataset_stream = load_dataset(dataset_dir,
                                  "clean",
                                  split="test",
                                  streaming=True)
    dataset_stream = dataset_stream.cast_column("audio", Audio(decode=False))
    dataset = list(dataset_stream.take(num_calib_samples))

    dtype = next(model.parameters()).dtype
    device = model.device
    num_mel_bins = model.config.num_mel_bins
    n_window = model.config.n_window
    chunk_length = n_window * 2

    cnn_output_len = _compute_cnn_output_length(model, num_mel_bins,
                                                chunk_length, dtype, device)

    def preprocess_fn(data):
        audio_bytes = data["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        mel_np, _ = extract_mel_spectrogram(audio, sample_rate=sr)
        mel = torch.from_numpy(mel_np)
        _, T = mel.shape

        num_chunks = min(max(1, T // chunk_length), max_chunks)
        target_T = num_chunks * chunk_length

        if T < target_T:
            mel = torch.nn.functional.pad(mel, (0, target_T - T))
        else:
            mel = mel[:, :target_T]

        # [mel_bins, num_chunks * chunk_length] -> [num_chunks, mel_bins, chunk_length]
        padded_feature = mel.reshape(num_mel_bins, num_chunks,
                                     chunk_length).permute(1, 0, 2).to(dtype)

        total_elems = num_chunks * cnn_output_len
        padded_mask = torch.ones(num_chunks, cnn_output_len, dtype=torch.bool)
        indices = torch.nonzero(padded_mask)

        attn_mask = torch.full([total_elems, total_elems],
                               torch.finfo(dtype).min,
                               dtype=dtype)
        for i in range(num_chunks):
            s = i * cnn_output_len
            e = (i + 1) * cnn_output_len
            attn_mask[s:e, s:e] = 0

        return {
            "padded_feature": padded_feature,
            "padded_mask_after_cnn_indices": indices,
            "attention_mask": attn_mask,
        }

    return AudioCalibrationDataset(dataset, preprocess_fn)


def quantize_audio(model, precision, dataset_dir="openslr/librispeech_asr"):
    """
    Quantize an audio encoder model to FP8 using LibriSpeech ASR calibration data.

    Args:
        model: Patched audio encoder model (Qwen3ASRAudioEncoderPatch
            or Qwen3OmniAudioEncoderPatch).
        precision: Quantization precision (only "fp8" is supported).
        dataset_dir: HuggingFace dataset identifier for calibration data.

    Returns:
        The quantized model (modified in-place and returned).
    """
    assert precision == "fp8", \
        f"Only fp8(W8A8) is supported for audio encoder quantization. Got: {precision}"

    quant_config = mtq.FP8_DEFAULT_CFG.copy()
    quant_config["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    data_loader = get_audio_calib_dataloader(model, dataset_dir)
    quantized_model = quantize_model(model, quant_config, data_loader)
    return quantized_model
