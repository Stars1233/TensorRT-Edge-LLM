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
Offline audio preprocessing for TensorRT Edge-LLM inference.

Converts raw audio files (wav, mp3, flac, etc.) into the safetensors format
required by the C++ audioRunner.

Supported feature extractors:
    - **whisper** (default): WhisperFeatureExtractor for Qwen3-Omni / Qwen3-ASR.
      Output shape: [1, mel_bins, time_steps].
    - **parakeet**: ParakeetFeatureExtractor for Nemotron-Omni.
      Output shape: [1, time_steps, mel_bins].

Output format:
    - safetensors file with a single tensor "mel_spectrogram"
    - Dtype: float16

Usage:
    # Qwen3-Omni (default, Whisper feature extractor)
    tensorrt-edgellm-preprocess-audio
        --input /path/to/audio.wav
        --output /path/to/output.safetensors

    # Nemotron-Omni (Parakeet feature extractor)
    tensorrt-edgellm-preprocess-audio
        --input /path/to/audio.wav
        --output /path/to/output.safetensors
        --feature_extractor parakeet

    # With model-specific preprocessor config
    tensorrt-edgellm-preprocess-audio
        --input /path/to/audio.wav
        --output /path/to/output.safetensors
        --preprocessor_config /path/to/preprocessor_config.json

    # Batch mode: process all audio files in a directory
    tensorrt-edgellm-preprocess-audio
        --input /path/to/audio_dir/
        --output /path/to/output_dir/
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SUPPORTED_FEATURE_EXTRACTORS = {"whisper", "parakeet"}


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file and return a mono waveform at the target sample rate.

    Args:
        audio_path: Path to the audio file.
        target_sr: Target sample rate in Hz (default: 16000 for Qwen3-Omni).

    Returns:
        Mono waveform as a 1-D float32 numpy array at *target_sr*.

    Raises:
        RuntimeError: If the file cannot be loaded or resampled.
    """
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32")

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    preprocessor_config: str = None,
    feature_extractor_type: str = "whisper",
) -> tuple:
    """
    Extract a mel-spectrogram from a waveform.

    Supports two feature extractors:

    - **whisper** (default): Uses ``WhisperFeatureExtractor`` for Qwen3-Omni
      / Qwen3-ASR.  Returns shape ``[mel_bins, time_steps]``.
    - **parakeet**: Uses ``ParakeetFeatureExtractor`` for Nemotron-Omni.
      Applies preemphasis (0.97), uses n_fft=512, and performs per-sample
      mean-variance normalization.  Returns shape ``[time_steps, mel_bins]``.

    Args:
        audio: Mono waveform, 1-D float32 array.
        sample_rate: Sample rate of *audio*.
        preprocessor_config: Optional path to a HuggingFace preprocessor
            config JSON.  When provided, the feature extractor is loaded
            from this config so that parameters match the model.
        feature_extractor_type: ``"whisper"`` or ``"parakeet"``.

    Returns:
        Tuple of (mel, num_frames) where *mel* is a float32 numpy array
        whose shape depends on the extractor type (see above) and
        *num_frames* is the number of valid (unpadded) time frames.
    """
    if feature_extractor_type not in SUPPORTED_FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unsupported feature extractor: {feature_extractor_type!r}. "
            f"Choose from {SUPPORTED_FEATURE_EXTRACTORS}")

    if feature_extractor_type == "parakeet":
        return _extract_mel_parakeet(audio, sample_rate, preprocessor_config)
    return _extract_mel_whisper(audio, sample_rate, preprocessor_config)


def _extract_mel_whisper(
    audio: np.ndarray,
    sample_rate: int,
    preprocessor_config: str = None,
) -> tuple:
    """Whisper-based mel extraction for Qwen3-Omni / Qwen3-ASR."""
    try:
        from transformers import WhisperFeatureExtractor
    except ImportError:
        raise ImportError(
            "transformers is required for mel-spectrogram extraction. "
            "Install it with: pip install transformers")

    if preprocessor_config is not None:
        config_dir = os.path.dirname(preprocessor_config) or "."
        feature_extractor = WhisperFeatureExtractor.from_pretrained(config_dir)
    else:
        # Qwen3-Omni default parameters
        feature_extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=sample_rate,
            hop_length=160,
            n_fft=400,
            return_attention_mask=True,
            padding_value=0.0,
        )

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="np",
        return_attention_mask=True,
        padding=False,
        truncation=False,
    )

    # Shape: [1, mel_bins, time_steps] -> [mel_bins, time_steps]
    mel = inputs["input_features"][0]
    # time_steps is the last axis for Whisper
    num_frames = int(inputs["attention_mask"][0].sum()
                     ) if "attention_mask" in inputs else mel.shape[-1]
    return mel, num_frames


def _extract_mel_parakeet(
    audio: np.ndarray,
    sample_rate: int,
    preprocessor_config: str = None,
) -> tuple:
    """Parakeet-based mel extraction for Nemotron-Omni.

    Key differences from Whisper:
      - Preemphasis filter (coefficient 0.97)
      - n_fft=512 (vs 400)
      - Per-sample zero-mean unit-variance normalization
      - Output is time-first: [time_steps, mel_bins]
    """
    try:
        from transformers import ParakeetFeatureExtractor
    except ImportError:
        raise ImportError(
            "transformers>=4.47 is required for ParakeetFeatureExtractor. "
            "Install it with: pip install 'transformers>=4.47'")

    if preprocessor_config is not None:
        config_dir = os.path.dirname(preprocessor_config) or "."
        feature_extractor = ParakeetFeatureExtractor.from_pretrained(
            config_dir)
    else:
        # Nemotron-Omni default parameters
        feature_extractor = ParakeetFeatureExtractor(
            feature_size=128,
            sampling_rate=sample_rate,
            hop_length=160,
            win_length=400,
            n_fft=512,
            preemphasis=0.97,
        )

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="np",
        return_attention_mask=True,
        padding=False,
    )

    # Shape: [1, time_steps, mel_bins] -> [time_steps, mel_bins]
    mel = inputs["input_features"][0]
    # time_steps is the first axis for Parakeet
    num_frames = int(inputs["attention_mask"]
                     [0].sum()) if "attention_mask" in inputs else mel.shape[0]
    return mel, num_frames


def save_audio_safetensors(
    mel: np.ndarray,
    output_path: str,
) -> None:
    """
    Save a mel-spectrogram to safetensors format (FP16) as required by the
    C++ audioRunner.

    Saved tensor:

    - ``mel_spectrogram`` (float16): A leading batch dimension is added if
      the input is 2-D.

      - Whisper input ``[mel_bins, T]`` → ``[1, mel_bins, T]``
      - Parakeet input ``[T, mel_bins]`` → ``[1, T, mel_bins]``


    Args:
        mel: Mel-spectrogram, 2-D or 3-D float32 numpy array.
        output_path: Destination file path (should end with ``.safetensors``).
    """
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("torch and safetensors are required. "
                          "Install them with: pip install torch safetensors")

    if mel.ndim == 2:
        mel = mel[np.newaxis, ...]

    tensors = {
        "mel_spectrogram":
        torch.from_numpy(np.ascontiguousarray(mel)).to(torch.float16),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(tensors, output_path)


def preprocess_single_audio(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    preprocessor_config: str = None,
    feature_extractor_type: str = "whisper",
) -> None:
    """
    End-to-end preprocessing of one audio file.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to the output safetensors file.
        sample_rate: Target sample rate.
        preprocessor_config: Optional path to preprocessor config JSON.
        feature_extractor_type: ``"whisper"`` or ``"parakeet"``.
    """
    print(f"Processing: {input_path}")

    audio = load_audio(input_path, target_sr=sample_rate)
    duration = len(audio) / sample_rate
    print(f"  Audio: {duration:.2f}s, {len(audio)} samples @ {sample_rate} Hz")

    mel, num_frames = extract_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        preprocessor_config=preprocessor_config,
        feature_extractor_type=feature_extractor_type,
    )
    print(f"  Mel-spectrogram: {mel.shape} "
          f"(range [{mel.min():.2f}, {mel.max():.2f}], "
          f"{num_frames} valid frames)")

    save_audio_safetensors(mel, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved: {output_path} ({size_kb:.1f} KB, fp16)")


def main() -> None:
    """
    Main entry point: parse arguments and run audio preprocessing.
    """
    parser = argparse.ArgumentParser(description=(
        "Preprocess audio files into safetensors mel-spectrograms "
        "for TensorRT Edge-LLM inference"), )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=("Path to an audio file (wav/mp3/flac/ogg/m4a) or a directory "
              "of audio files"),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=("Output path.  For a single file, this is the .safetensors "
              "output path.  For a directory input, this is the output "
              "directory."),
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--preprocessor_config",
        type=str,
        default=None,
        help=("Path to a HuggingFace preprocessor_config.json. "
              "When provided, feature extractor settings are loaded from "
              "this file instead of using built-in defaults."),
    )
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="whisper",
        choices=sorted(SUPPORTED_FEATURE_EXTRACTORS),
        help=("Feature extractor type. 'whisper' (default) for "
              "Qwen3-Omni/Qwen3-ASR; 'parakeet' for Nemotron-Omni."),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        if input_path.is_file():
            # Single file mode
            out = str(output_path)
            if not out.endswith(".safetensors"):
                out = str(output_path / (input_path.stem + ".safetensors"))
            preprocess_single_audio(
                str(input_path),
                out,
                sample_rate=args.sample_rate,
                preprocessor_config=args.preprocessor_config,
                feature_extractor_type=args.feature_extractor,
            )
        elif input_path.is_dir():
            # Batch directory mode
            audio_files = sorted(p for p in input_path.iterdir()
                                 if p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if not audio_files:
                print(f"No audio files found in {input_path}")
                sys.exit(1)
            print(f"Found {len(audio_files)} audio files in {input_path}\n")
            output_path.mkdir(parents=True, exist_ok=True)
            for af in audio_files:
                out = str(output_path / (af.stem + ".safetensors"))
                preprocess_single_audio(
                    str(af),
                    out,
                    sample_rate=args.sample_rate,
                    preprocessor_config=args.preprocessor_config,
                    feature_extractor_type=args.feature_extractor,
                )
                print()
        else:
            print(f"Error: {input_path} does not exist")
            sys.exit(1)

        print("Audio preprocessing completed successfully!")

    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
