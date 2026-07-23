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
"""TEMPLATE: add a custom calibration dataset.

This module is a copy-paste starting point -- it is intentionally NOT imported
by ``datasets/__init__.py``, so nothing here is registered until you wire it
in. A calibration dataset is just a generator function, registered by name:

  1. Copy this file to ``datasets/<your_dataset>.py`` (or add a function to
     ``builtin.py``).
  2. Write a generator that yields the right type for your modality:
       * text  : yield str
       * image : yield (image, question)   # image: PIL image, path, or URL
       * audio : yield (audio_bytes, transcript)
     Yield lazily -- the loaders bound the sample count.
  3. Decorate it with ``@register("<modality>", "<name>")``. ``<name>`` is the
     value users pass on the CLI (``--text_dataset <name>`` etc.).
  4. Register it on import: add ``from . import <your_dataset>`` to
     ``datasets/__init__.py`` (next to ``from . import builtin``). If you added
     your function to ``builtin.py`` instead, this step is already done.

After that, ``--<modality>_dataset <name>`` works and an unknown name fails
out with the list of registered datasets.

See ``docs/source/developer_guide/customization/calibration-datasets.md``.
"""

from . import first_present, load_hf_split, local_override_path, register


@register("text", "my_text_dataset")
def my_text_dataset():
    """Text contract: yield one calibration string per item.

    ``load_hf_split`` reads from the Hub id, or from a local file/dir when
    EDGELLM_QUANT_DATASET_MY_TEXT_DATASET is set. You can also read your own
    files here instead of using ``load_hf_split``.
    """
    ds = load_hf_split("your-org/your-text-dataset",
                       name="my_text_dataset",
                       split="train")
    for example in ds:
        text = example.get("text")
        if text:
            yield text


@register("image", "my_image_dataset")
def my_image_dataset():
    """Image contract: yield (image, question) pairs."""
    ds = load_hf_split("your-org/your-image-dataset",
                       name="my_image_dataset",
                       split="validation",
                       streaming=True)
    for example in ds:
        # ``image`` may be a PIL image, a local path, or a URL -- anything the
        # model's AutoProcessor chat template accepts.
        image = first_present(example, ("image", "image_1"))
        question = example.get("question") or ""
        if image is not None and question:
            yield image, question


@register("audio", "my_audio_dataset")
def my_audio_dataset():
    """Audio contract: yield (audio_bytes, transcript) pairs.

    The ASR loader decodes the bytes with soundfile and extracts mel features,
    so just hand it encoded audio (wav/flac/...) and the matching transcript.
    """
    # Replace with your own (path, transcript) source. ``local_override_path``
    # lets a deployment point this at a different location without code edits.
    root = local_override_path("my_audio_dataset") or "/path/to/audio"
    pairs = [(f"{root}/utterance_0.flac", "the ground truth transcript")]
    for path, transcript in pairs:
        transcript = (transcript or "").strip()
        if not transcript:
            continue
        with open(path, "rb") as fh:
            audio_bytes = fh.read()
        if audio_bytes:
            yield audio_bytes, transcript
