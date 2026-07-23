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
OmniBench dataset for TensorRT Edge-LLM.

Evaluates multimodal understanding across audio + image + text inputs.
https://huggingface.co/datasets/m-a-p/OmniBench

OmniBench data format:
{
    'index': 42,
    'task type': 'attribute recognition',
    'audio type': 'speech',
    'question': 'What color is the object mentioned in the audio?',
    'options': ['Red', 'Blue', 'Green', 'Yellow'],
    'answer': 'Red',
    'audio': {'array': np.ndarray, 'sampling_rate': 16000},
    'image': PIL.Image,
    'audio content': 'The speaker says: the red ball...',
    'image content': 'A photograph showing colored balls...',
}
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_eval_utils import (OMNI_SYSTEM_PROMPT, audio_extension_from_payload,
                              save_audio_array_to_flac, write_audio_bytes)
from datasets import Audio, Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class OmniBenchDataset(EdgeLLMDataset):
    """
    OmniBench dataset with audio + image + text multimodal inputs.
    https://huggingface.co/datasets/m-a-p/OmniBench

    Extends EdgeLLMDataset with raw audio-file export support.
    """

    def __init__(self, dataset: Dataset, config: DatasetConfig, **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.images_dir = os.path.join(self.output_dir, "images")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        """Format OmniBench prompt with question and multiple choice options.

        Uses the exact prompt format from OmniBench official evaluation:
        audio and image context is implicit (passed via multimodal content).
        """
        question = data["question"]
        options = data["options"]

        prompt = question
        for i, opt in enumerate(options):
            letter = chr(ord('A') + i)
            prompt += f"\n{letter}. {opt}"
        prompt += ("\nAnswer with the option's letter from "
                   "the given choices directly.")
        return prompt

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        """Use Omni system prompt for multimodal understanding."""
        return OMNI_SYSTEM_PROMPT

    MAX_IMAGE_LONG_SIDE = 768

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        """Save OmniBench image, resizing to fit within thinker input budget.

        Audio + vision + text tokens must all fit in maxInputLen (8192).
        With long_side=768, worst-case vision tokens ~729 (square image),
        leaving ~7000 tokens for audio (~1000 max) and text (~400).
        """
        image_paths = []
        image = data.get("image")
        if image is not None:
            idx = data.get("index", id(data))
            image_path = os.path.join(self.images_dir, f"image_{idx}.jpg")
            if not os.path.exists(image_path):
                image = image.convert("RGB")
                w, h = image.size
                long_side = max(w, h)
                if long_side > self.MAX_IMAGE_LONG_SIDE:
                    scale = self.MAX_IMAGE_LONG_SIDE / long_side
                    image = image.resize((int(w * scale), int(h * scale)),
                                         resample=3)  # BICUBIC
                image.save(image_path, "JPEG")
            image_paths.append(image_path)
        return image_paths

    def save_audio(self, data: Dict[str, Any]) -> Optional[str]:
        """Save OmniBench audio for the C++ ``llm_inference`` CLI.

        HF gives ``bytes`` (encoded container) or ``array`` (decoded waveform).
        Encoded audio is copied verbatim; decoded arrays are written as FLAC.
        The runtime decodes and runs the model-specific audio frontend.
        """
        audio_data = data.get("audio")
        if audio_data is None:
            return None

        idx = data.get("index", id(data))
        os.makedirs(self.audio_dir, exist_ok=True)

        if isinstance(audio_data, dict) and audio_data.get("bytes"):
            encoded = audio_data["bytes"]
            ext = audio_extension_from_payload(audio_data.get("path"), encoded)
            out_path = os.path.join(self.audio_dir, f"audio_{idx}{ext}")
            if not os.path.exists(out_path):
                write_audio_bytes(encoded, out_path)
            return out_path

        flac_path = os.path.join(self.audio_dir, f"audio_{idx}.flac")
        if os.path.exists(flac_path):
            return flac_path
        if isinstance(audio_data, dict) and "array" in audio_data:
            arr = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data.get("sampling_rate", 16000)
        else:
            arr = np.array(audio_data, dtype=np.float32)
            sr = 16000
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return save_audio_array_to_flac(arr, flac_path, sample_rate=sr)

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the correct answer letter (A/B/C/D) from OmniBench data."""
        answer_text = data.get("answer")
        if answer_text is None:
            return None
        options = data.get("options", [])
        for i, opt in enumerate(options):
            if opt == answer_text:
                return chr(ord('A') + i)
        return answer_text


def convert_omnibench_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "m-a-p/OmniBench",
        output_dir: Union[str, os.PathLike] = "omnibench_dataset"):
    """
    Convert OmniBench dataset to TensorRT Edge-LLM format.

    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
    """
    print(f"Converting OmniBench from {dataset_name_or_dir} to {output_dir}")

    dataset = load_dataset(dataset_name_or_dir, split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    print(f"Loaded OmniBench with {len(dataset)} examples")

    edge_llm_dataset = OmniBenchDataset(dataset=dataset,
                                        config=config,
                                        output_dir=output_dir)

    print(f"Processing OmniBench with config: {config}")
    edge_llm_dataset.process_and_save_dataset("omnibench_dataset.json")

    print(f"Successfully converted OmniBench to {output_dir}")
    return edge_llm_dataset
