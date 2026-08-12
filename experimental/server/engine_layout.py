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
"""Standard directory layouts for TensorRT Edge-LLM.

Defines the file naming conventions expected by the C++ runtime.
Used by ``LLM`` (HLAPI), CI tests, and documentation.

ONNX directory (input to engine build)::

    {onnx_dir}/
        model.onnx
        config.json
        tokenizer.json / tokenizer_config.json
        embedding.safetensors

LLM engine directory::

    {engine_dir}/
        llm.engine
        config.json
        tokenizer.json / tokenizer_config.json
        processed_chat_template.json
        embedding.safetensors

VLM (LLM + Visual encoder)::

    {llm_engine_dir}/              # Same as LLM above
    {multimodal_engine_dir}/
        visual.engine
        config.json

Speculative decoding::

    {spec_decode_engine_dir}/
        spec_base.engine / spec_draft.engine
        base_config.json / draft_config.json
        d2t.safetensors                 # EAGLE only
        tokenizer.json / tokenizer_config.json
        embedding.safetensors
"""

import enum
import os
from typing import Optional


class EngineType(enum.Enum):
    """Detected engine directory type."""

    LLM = "llm"
    VLM = "vlm"
    SPEC_DECODE = "spec_decode"
    UNKNOWN = "unknown"


ONNX_MODEL_FILE = "model.onnx"
LLM_ENGINE_FILE = "llm.engine"
VISUAL_ENGINE_FILE = "visual.engine"
AUDIO_ENGINE_FILE = "audio_encoder.engine"
CODE2WAV_ENGINE_FILE = "code2wav.engine"
SPEC_BASE_ENGINE_FILE = "spec_base.engine"
SPEC_DRAFT_ENGINE_FILE = "spec_draft.engine"
CONFIG_FILE = "config.json"


def validate_onnx_dir(onnx_dir: str) -> bool:
    """Check that a directory contains an ONNX model."""
    return os.path.isfile(os.path.join(onnx_dir, ONNX_MODEL_FILE))


def detect_engine_type(engine_dir: str) -> EngineType:
    """Detect the engine type from directory contents."""
    if not os.path.isdir(engine_dir):
        return EngineType.UNKNOWN
    if os.path.isfile(os.path.join(engine_dir, SPEC_BASE_ENGINE_FILE)):
        return EngineType.SPEC_DECODE
    if os.path.isfile(os.path.join(engine_dir, LLM_ENGINE_FILE)):
        return EngineType.LLM
    if os.path.isfile(os.path.join(engine_dir, VISUAL_ENGINE_FILE)):
        return EngineType.VLM
    return EngineType.UNKNOWN


def validate_llm_engine_dir(engine_dir: str) -> bool:
    """Check that an LLM engine directory has the required files."""
    return os.path.isfile(os.path.join(engine_dir, LLM_ENGINE_FILE))


def validate_multimodal_engine_dir(engine_dir: str) -> bool:
    """Check that a multimodal engine directory has at least one encoder.

    Matches the C++ ``MultimodalRunner::create`` layout: a ``visual/`` or
    ``audio/`` subdirectory holding the respective ``.engine``; legacy
    ``visual.engine`` at root is also accepted.
    """
    return (
        os.path.isfile(os.path.join(engine_dir, "visual", VISUAL_ENGINE_FILE))
        or os.path.isfile(os.path.join(engine_dir, "audio", AUDIO_ENGINE_FILE))
        or os.path.isfile(os.path.join(engine_dir, VISUAL_ENGINE_FILE)))


def validate_spec_decode_engine_dir(engine_dir: str) -> bool:
    """Check that a speculative decoding engine directory has both engines."""
    has_base = os.path.isfile(os.path.join(engine_dir, SPEC_BASE_ENGINE_FILE))
    has_draft = os.path.isfile(os.path.join(engine_dir,
                                            SPEC_DRAFT_ENGINE_FILE))
    return has_base and has_draft


def classify_model_source(path: str) -> str:
    """Classify a path into an LLM init mode: ``engine_dir`` | ``onnx_dir`` | ``model``.

    A prebuilt engine dir (has an ``*.engine``) maps to ``engine_dir``, a prebuilt
    ONNX dir (has ``model.onnx``) to ``onnx_dir``, and a HuggingFace id or
    checkpoint dir to ``model``. Lets a single path route to LLM's three init
    modes without the caller having to know the directory layout.
    """
    if os.path.isdir(path):
        if validate_llm_engine_dir(path) or validate_spec_decode_engine_dir(
                path):
            return "engine_dir"
        if validate_onnx_dir(path):
            return "onnx_dir"
    return "model"


def find_omni_engine_dirs(llm_engine_dir: str) -> Optional[dict]:
    """Auto-detect Qwen3-Omni audio-output engine dirs.

    Expects Talker / CodePredictor / Code2Wav as siblings of the Thinker
    (LLM) engine directory::

        {root}/
            thinker/          # llm_engine_dir (llm.engine)
            talker/           # llm.engine
            code_predictor/   # llm.engine
            code2wav/         # code2wav.engine

    Returns a dict with keys ``talker``, ``code_predictor``, ``code2wav``
    when all three are present, else None.
    """
    parent = os.path.dirname(os.path.abspath(llm_engine_dir))
    talker = os.path.join(parent, "talker")
    code_predictor = os.path.join(parent, "code_predictor")
    code2wav = os.path.join(parent, "code2wav")
    if (os.path.isfile(os.path.join(talker, LLM_ENGINE_FILE))
            and os.path.isfile(os.path.join(code_predictor, LLM_ENGINE_FILE))
            and os.path.isfile(os.path.join(code2wav, CODE2WAV_ENGINE_FILE))):
        return {
            "talker": talker,
            "code_predictor": code_predictor,
            "code2wav": code2wav,
        }
    return None


def find_multimodal_engine_dir(llm_engine_dir: str) -> Optional[str]:
    """Auto-detect a sibling visual engine directory.

    Searches for a sibling of the LLM engine directory that contains
    ``visual/visual.engine`` or ``visual.engine``. Audio-only siblings are
    never auto-attached (no signal for which modality the LLM expects); pass
    them explicitly.
    """
    parent = os.path.dirname(llm_engine_dir)
    if not os.path.isdir(parent):
        return None
    candidates = []
    for entry in sorted(os.listdir(parent)):
        candidate = os.path.join(parent, entry)
        if candidate == llm_engine_dir:
            continue
        if (os.path.isfile(
                os.path.join(candidate, "visual", VISUAL_ENGINE_FILE)) or
                os.path.isfile(os.path.join(candidate, VISUAL_ENGINE_FILE))):
            candidates.append(candidate)
    if len(candidates) > 1:
        raise ValueError(
            "multiple sibling multimodal engine directories found: "
            f"{candidates}; pass --multimodal-engine-dir explicitly")
    return candidates[0] if candidates else None
