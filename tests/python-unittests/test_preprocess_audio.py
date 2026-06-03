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
Sanity tests for the preprocess_audio script.

Usage:
    python3 -m pytest tests/python-unittests/test_preprocess_audio.py -v --noconftest
"""

import os
import tempfile

import numpy as np


def test_extract_mel_spectrogram_default():
    """Test mel-spectrogram extraction with default Qwen3-Omni parameters."""
    from tensorrt_edgellm.scripts.preprocess_audio import \
        extract_mel_spectrogram

    audio = np.random.randn(16000).astype(np.float32)
    mel, _ = extract_mel_spectrogram(audio, sample_rate=16000)

    assert mel.ndim == 2
    assert mel.shape[0] == 128
    assert mel.dtype == np.float32


def test_save_and_load_audio_safetensors():
    """Test saving and loading mel-spectrogram in safetensors format."""
    from safetensors.torch import load_file

    from tensorrt_edgellm.scripts.preprocess_audio import \
        save_audio_safetensors

    mel = np.random.randn(128, 100).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.safetensors")
        save_audio_safetensors(mel, path)

        assert os.path.exists(path)
        data = load_file(path)
        assert "mel_spectrogram" in data
        assert data["mel_spectrogram"].shape == (1, 128, 100)


def test_preprocess_single_audio_wav(tmp_path):
    """Test end-to-end preprocessing of a WAV file."""
    import soundfile as sf

    from tensorrt_edgellm.scripts.preprocess_audio import \
        preprocess_single_audio

    audio = np.random.randn(16000).astype(np.float32)
    wav_path = str(tmp_path / "test.wav")
    sf.write(wav_path, audio, 16000)

    out_path = str(tmp_path / "out.safetensors")
    preprocess_single_audio(wav_path, out_path, sample_rate=16000)

    assert os.path.exists(out_path)
