# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for `LLM(engine_dir=...)` routing to LLM vs spec-decode paths."""

import pytest

from experimental.server.engine import LLM
from experimental.server.engine_layout import (EngineType, detect_engine_type,
                                               validate_spec_decode_engine_dir)


def _touch(path):
    with open(path, "w"):
        pass


# ---------------------------------------------------------------------------
# engine_layout helpers
# ---------------------------------------------------------------------------


def test_detect_engine_type_spec_decode(tmp_path):
    _touch(tmp_path / "spec_base.engine")
    _touch(tmp_path / "spec_draft.engine")
    assert detect_engine_type(str(tmp_path)) == EngineType.SPEC_DECODE


def test_detect_engine_type_llm(tmp_path):
    _touch(tmp_path / "llm.engine")
    assert detect_engine_type(str(tmp_path)) == EngineType.LLM


def test_detect_engine_type_unknown(tmp_path):
    assert detect_engine_type(str(tmp_path)) == EngineType.UNKNOWN


def test_validate_spec_decode_engine_dir_requires_both(tmp_path):
    assert not validate_spec_decode_engine_dir(str(tmp_path))
    _touch(tmp_path / "spec_base.engine")
    assert not validate_spec_decode_engine_dir(str(tmp_path))
    _touch(tmp_path / "spec_draft.engine")
    assert validate_spec_decode_engine_dir(str(tmp_path))


# ---------------------------------------------------------------------------
# LLM._init_from_engine routing
# ---------------------------------------------------------------------------


class _BareLLM(LLM):
    """LLM subclass that skips runtime loading, exposing just the routing."""

    # pylint: disable=super-init-not-called
    def __init__(self, engine_dir: str, visual_engine_dir: str = ""):
        self._eagle_engine_dir = ""
        self._tool_template_formatter = None
        self._model_id = "test"
        self._init_from_engine(engine_dir, visual_engine_dir)


def test_init_from_engine_routes_spec_decode_dir(tmp_path):
    """Spec-decode dirs promote engine_dir to _eagle_engine_dir."""
    _touch(tmp_path / "spec_base.engine")
    _touch(tmp_path / "spec_draft.engine")

    llm = _BareLLM(str(tmp_path))

    assert llm._engine_dir == str(tmp_path)
    assert llm._model_dir == str(tmp_path)
    # Key routing side-effect: spec-decode dispatch downstream reads
    # `_eagle_engine_dir`, so this promotion is what actually enables the fix.
    assert llm._eagle_engine_dir == str(tmp_path)
    assert llm._visual_engine_dir == ""
    assert llm._is_vlm is False


def test_init_from_engine_ignores_visual_dir_for_spec_decode(tmp_path, caplog):
    """visual_engine_dir is meaningless for spec-decode and must not raise."""
    _touch(tmp_path / "spec_base.engine")
    _touch(tmp_path / "spec_draft.engine")
    visual_dir = tmp_path / "visual"
    visual_dir.mkdir()

    llm = _BareLLM(str(tmp_path), visual_engine_dir=str(visual_dir))

    # Visual dir was ignored (spec-decode engines have no vision).
    assert llm._visual_engine_dir == ""
    assert llm._is_vlm is False


def test_init_from_engine_spec_decode_missing_draft_engine(tmp_path):
    """Half-populated spec-decode dir must raise a clear error."""
    _touch(tmp_path / "spec_base.engine")
    # spec_draft.engine deliberately absent

    with pytest.raises(ValueError, match="spec_base.engine/spec_draft.engine"):
        _BareLLM(str(tmp_path))


def test_init_from_engine_vanilla_llm_still_works(tmp_path):
    """Vanilla llm.engine dirs must continue to route through the LLM path."""
    _touch(tmp_path / "llm.engine")

    llm = _BareLLM(str(tmp_path))

    assert llm._engine_dir == str(tmp_path)
    # Vanilla path does NOT set _eagle_engine_dir.
    assert llm._eagle_engine_dir == ""
    assert llm._is_vlm is False


def test_init_from_engine_unknown_dir_raises(tmp_path):
    """A directory with neither llm.engine nor spec_base.engine is rejected."""
    with pytest.raises(ValueError, match="llm.engine not found"):
        _BareLLM(str(tmp_path))
