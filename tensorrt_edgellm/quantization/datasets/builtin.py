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
"""Built-in calibration datasets, one generator function each.

Each function yields lazily; the calibration loaders bound the count
(``num_samples``) and do the tokenization / feature extraction.
"""

from . import first_present, load_hf_split, register


@register("text", "cnn_dailymail")
def cnn_dailymail():
    """CNN/DailyMail news articles -- the default text dataset."""
    ds = load_hf_split("cnn_dailymail",
                       name="cnn_dailymail",
                       config="3.0.0",
                       split="train")
    for example in ds:
        text = example.get("article")
        if text:
            yield text


@register("text", "wikitext")
def wikitext():
    """WikiText-2 (raw) -- an alternative text dataset."""
    ds = load_hf_split("Salesforce/wikitext",
                       name="wikitext",
                       config="wikitext-2-raw-v1",
                       split="test")
    for example in ds:
        text = example.get("text")
        if text:
            yield text


@register("image", "mmmu")
def mmmu():
    """MMMU image-question pairs -- the default image dataset.

    ``lmms-lab/MMMU`` rows carry numbered ``image_1``..``image_7`` columns, so
    the first present image on each row is used for calibration.
    """
    image_fields = ("image", ) + tuple(f"image_{i}" for i in range(1, 8))
    ds = load_hf_split("lmms-lab/MMMU",
                       name="mmmu",
                       split="validation",
                       streaming=True)
    for example in ds:
        image = first_present(example, image_fields)
        question = example.get("question") or ""
        if image is not None and question:
            yield image, question


@register("audio", "librispeech")
def librispeech():
    """LibriSpeech audio-transcript pairs -- the default audio dataset.

    Yields raw encoded audio bytes; the ASR loader decodes them.
    """
    from datasets import Audio

    ds = load_hf_split("openslr/librispeech_asr",
                       name="librispeech",
                       config="clean",
                       split="train.100",
                       streaming=True)
    # ``decode=False`` avoids torchcodec; the loader decodes bytes itself.
    ds = ds.cast_column("audio", Audio(decode=False))
    for example in ds:
        record = example.get("audio")
        if not record:
            continue
        audio_bytes = record.get("bytes")
        if audio_bytes is None and record.get("path"):
            with open(record["path"], "rb") as fh:
                audio_bytes = fh.read()
        transcript = (example.get("text") or "").strip()
        if audio_bytes and transcript:
            yield audio_bytes, transcript
