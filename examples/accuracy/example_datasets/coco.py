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
COCO image captioning dataset converter for TensorRT Edge-LLM.

Source: HuggingFace `lmms-lab/COCO-Caption2017` (val split, 5000 images).
Each example carries a real COCO image plus up to 5 human captions; we
emit the first caption as the reference and keep the rest in ``answer``
so downstream rouge / BLEU scoring can pick whichever it likes.

Output layout (aligned with updated_datasets/coco):

    <output_dir>/
        dataset.json           # batch_size / temperature / requests
        images/000000XXXXXX.jpg

The request prompt format follows the existing updated_datasets/coco
template — a system prompt plus a user message containing the image and
a short captioning question.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_dataset
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class COCODataset(EdgeLLMDataset):
    """COCO Caption dataset (val split) converter.

    Expected HF schema (`lmms-lab/COCO-Caption2017`):
        {
            'image': PIL.Image,
            'image_id': 397133,
            'captions': ['A man is standing...', ...],   # up to 5 captions
        }
    """

    DEFAULT_PROMPT = "Please briefly describe the image."

    def __init__(self,
                 dataset: Dataset,
                 config: DatasetConfig,
                 dataset_subdir: str = "coco",
                 prompt: Optional[str] = None,
                 **kwargs):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.dataset_subdir = dataset_subdir
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.prompt = prompt or self.DEFAULT_PROMPT

    def format_user_prompt(self, data: Dict[str, Any]) -> str:
        return self.prompt

    def format_system_prompt(self, data: Dict[str, Any]) -> str:
        return "You are a helpful assistant."

    def save_image(self, data: Dict[str, Any]) -> List[str]:
        image = data.get("image")
        if image is None:
            return []
        image_id = data.get("image_id") or data.get("id") or id(data)
        # COCO val image filenames use 12-digit zero-padded ids.
        try:
            filename = f"{int(image_id):012d}.jpg"
        except (TypeError, ValueError):
            filename = f"{image_id}.jpg"
        image_path = os.path.join(self.images_dir, filename)
        if not os.path.exists(image_path):
            image.convert("RGB").save(image_path, "JPEG", quality=90)
        return [image_path]

    def extract_answer(self, data: Dict[str, Any]) -> Optional[str]:
        captions = data.get("captions") or data.get("answers")
        if captions is None:
            cap = data.get("caption")
            return cap if isinstance(cap, str) else None
        if isinstance(captions, str):
            return captions
        if isinstance(captions, list) and captions:
            return " | ".join(str(c) for c in captions)
        return None


def convert_coco_dataset(
        config: DatasetConfig,
        dataset_name_or_dir: str = "lmms-lab/COCO-Caption2017",
        output_dir: Union[str, os.PathLike] = "coco_output",
        split: str = "val",
        max_samples: Optional[int] = None,
        prompt: Optional[str] = None,
        output_filename: str = "dataset.json"):
    """Convert COCO Caption dataset into the TensorRT Edge-LLM JSON layout.

    Args:
        config: shared DatasetConfig (batch_size / sampling params).
        dataset_name_or_dir: HF repo id (default ``lmms-lab/COCO-Caption2017``)
            or a local directory containing the same structure.
        output_dir: where to write ``dataset.json`` + ``images/``.
        split: HF split (default ``val``; the val split is the one the
            standard COCO-Caption benchmark uses).
        max_samples: optional cap for quick smoke runs.
        prompt: override the per-request user prompt; defaults to
            ``Please briefly describe the image.``.
        output_filename: name of the JSON manifest; defaults to
            ``dataset.json`` to match the existing updated_datasets/coco
            file naming.
    """
    print(f"Loading COCO Caption from {dataset_name_or_dir} (split={split})")
    dataset = load_dataset(dataset_name_or_dir, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to first {len(dataset)} samples")

    converter = COCODataset(dataset=dataset,
                            config=config,
                            output_dir=output_dir,
                            prompt=prompt)
    out_path = converter.process_and_save_dataset(
        output_filename=output_filename)
    print(f"COCO dataset written to {out_path}")
    return out_path
