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
"""Shared audio helpers for accuracy benchmarks."""

import os
from pathlib import Path
from typing import Optional

OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech.")

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".flac", ".mp3", ".wav"})


def audio_extension_from_payload(path: Optional[str],
                                 audio_bytes: Optional[bytes] = None,
                                 default: str = ".flac") -> str:
    """Return a runtime-supported extension from path or audio magic bytes."""
    ext = Path(path or "").suffix.lower()
    if ext in SUPPORTED_AUDIO_EXTENSIONS:
        return ext
    if audio_bytes:
        header = audio_bytes[:12]
        if header.startswith(b"fLaC"):
            return ".flac"
        if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
            return ".wav"
        if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF
                                         and header[1] & 0xE0 == 0xE0):
            return ".mp3"
    return default


def write_audio_bytes(audio: bytes, output_path: str) -> str:
    """Write encoded audio bytes without re-encoding the payload."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio)
    return output_path


def save_audio_array_to_wav(audio,
                            output_path: str,
                            sample_rate: int = 16000) -> str:
    """Write a mono waveform array to ``output_path`` as a 16-bit PCM WAV.

    The C++ runtime's ``audioLoader`` + ``MelExtractor`` consumes the WAV
    directly via the CLI's raw-audio path; mel extraction happens in C++.
    """
    import numpy as np
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    return output_path


def save_audio_array_to_flac(audio,
                             output_path: str,
                             sample_rate: int = 16000) -> str:
    """Write a mono waveform array to ``output_path`` as FLAC audio."""
    import numpy as np
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio, sample_rate, format="FLAC")
    return output_path
