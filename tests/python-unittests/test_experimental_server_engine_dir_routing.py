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
"""Tests for `LLM(engine_dir=...)` routing to LLM vs spec-decode paths and
`LLM(model=...)` checkpoint routing to the visual vs audio build branches."""

import json
import os
import shutil

import pytest

from experimental.server import engine as engine_module
from experimental.server.engine import LLM
from experimental.server.engine_layout import (EngineType, detect_engine_type,
                                               find_multimodal_engine_dir,
                                               validate_multimodal_engine_dir,
                                               validate_spec_decode_engine_dir)


def _touch(path):
    with open(path, "w"):
        pass


# ---------------------------------------------------------------------------
# engine_layout helpers
# ---------------------------------------------------------------------------


def test_detect_engine_type(tmp_path):
    unknown = tmp_path / "u"
    unknown.mkdir()
    assert detect_engine_type(str(unknown)) == EngineType.UNKNOWN
    llm = tmp_path / "l"
    llm.mkdir()
    _touch(llm / "llm.engine")
    assert detect_engine_type(str(llm)) == EngineType.LLM
    spec = tmp_path / "s"
    spec.mkdir()
    _touch(spec / "spec_base.engine")
    _touch(spec / "spec_draft.engine")
    assert detect_engine_type(str(spec)) == EngineType.SPEC_DECODE


def test_validate_spec_decode_engine_dir_requires_both(tmp_path):
    assert not validate_spec_decode_engine_dir(str(tmp_path))
    _touch(tmp_path / "spec_base.engine")
    assert not validate_spec_decode_engine_dir(str(tmp_path))
    _touch(tmp_path / "spec_draft.engine")
    assert validate_spec_decode_engine_dir(str(tmp_path))


def test_find_multimodal_engine_dir(tmp_path):
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    # No candidates at all.
    assert find_multimodal_engine_dir(str(llm_dir)) is None
    # Audio-only siblings are never auto-attached (an ASR dir next to a VLM);
    # audio engines must be passed explicitly.
    (tmp_path / "asr" / "audio").mkdir(parents=True)
    _touch(tmp_path / "asr" / "audio" / "audio_encoder.engine")
    assert find_multimodal_engine_dir(str(llm_dir)) is None
    # One visual sibling: picked up.
    (tmp_path / "vlm" / "visual").mkdir(parents=True)
    _touch(tmp_path / "vlm" / "visual" / "visual.engine")
    assert find_multimodal_engine_dir(str(llm_dir)) == str(tmp_path / "vlm")
    # Two visual siblings: ambiguous, must raise instead of picking one.
    (tmp_path / "vlm_b" / "visual").mkdir(parents=True)
    _touch(tmp_path / "vlm_b" / "visual" / "visual.engine")
    with pytest.raises(ValueError, match="multiple sibling"):
        find_multimodal_engine_dir(str(llm_dir))


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
    assert llm._multimodal_engine_dir == ""
    assert llm._is_multimodal is False


def test_init_from_engine_ignores_visual_dir_for_spec_decode(tmp_path, caplog):
    """visual_engine_dir is meaningless for spec-decode and must not raise."""
    _touch(tmp_path / "spec_base.engine")
    _touch(tmp_path / "spec_draft.engine")
    visual_dir = tmp_path / "visual"
    visual_dir.mkdir()

    llm = _BareLLM(str(tmp_path), visual_engine_dir=str(visual_dir))

    # Visual dir was ignored (spec-decode engines have no vision).
    assert llm._multimodal_engine_dir == ""
    assert llm._is_multimodal is False


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
    assert llm._is_multimodal is False


def test_init_from_engine_unknown_dir_raises(tmp_path):
    """A directory with neither llm.engine nor spec_base.engine is rejected."""
    with pytest.raises(ValueError, match="llm.engine not found"):
        _BareLLM(str(tmp_path))


# ---------------------------------------------------------------------------
# LLM._init_from_model routing (checkpoint autobuild, stubbed export/build)
# ---------------------------------------------------------------------------


class _StubBuilderRt:
    """Runtime-module stand-in: builders create the files the real ones do."""

    class LLMBuilderConfig:
        pass

    class VisualBuilderConfig:
        pass

    class AudioBuilderConfig:
        pass

    def __init__(self):
        self.builds = []
        rt = self

        class LLMBuilder:

            def __init__(self, onnx_dir, engine_dir, config):
                self._engine_dir = engine_dir

            def build(self):
                rt.builds.append("llm")
                _touch(os.path.join(self._engine_dir, "llm.engine"))
                return True

        class VisualBuilder:

            def __init__(self, onnx_dir, engine_dir, config):
                self._engine_dir = engine_dir

            def build(self):
                rt.builds.append("visual")
                _touch(os.path.join(self._engine_dir, "visual.engine"))
                return True

        class AudioBuilder:
            # Mirrors the C++ AudioBuilder: appends the audio/ subdirectory to
            # the given engine dir, then writes audio_encoder.engine and copies
            # config.json from the ONNX dir into it.
            def __init__(self, onnx_dir, engine_dir, config):
                self._onnx_dir = onnx_dir
                self._engine_dir = os.path.join(engine_dir, "audio")

            def build(self):
                rt.builds.append("audio")
                os.makedirs(self._engine_dir, exist_ok=True)
                _touch(os.path.join(self._engine_dir, "audio_encoder.engine"))
                shutil.copy(os.path.join(self._onnx_dir, "config.json"),
                            os.path.join(self._engine_dir, "config.json"))
                return True

        self.LLMBuilder = LLMBuilder
        self.VisualBuilder = VisualBuilder
        self.AudioBuilder = AudioBuilder


class _CheckpointLLM(LLM):
    """LLM subclass exercising the real checkpoint routing and engine-dir
    layout, with the heavy ONNX exports replaced by marker files."""

    # pylint: disable=super-init-not-called
    def __init__(self, model_dir: str, multimodal_engine_dir: str = ""):
        self._eagle_engine_dir = ""
        self._tool_template_formatter = None
        self._model_id = "test"
        self.exports = []
        self._init_from_model(model_dir,
                              max_input_len=128,
                              max_batch_size=1,
                              max_kv_cache_capacity=256,
                              multimodal_engine_dir=multimodal_engine_dir)

    def _export_onnx(self):
        self.exports.append("llm")
        os.makedirs(self._onnx_dir, exist_ok=True)
        _touch(os.path.join(self._onnx_dir, "model.onnx"))

    def _export_visual_onnx(self):
        self.exports.append("visual")
        os.makedirs(self._visual_onnx_dir, exist_ok=True)
        _touch(os.path.join(self._visual_onnx_dir, "model.onnx"))

    def _export_audio_onnx(self):
        self.exports.append("audio")
        os.makedirs(self._audio_onnx_dir, exist_ok=True)
        _touch(os.path.join(self._audio_onnx_dir, "model.onnx"))
        # The real audio export writes the runtime sidecar config with the
        # encoder-specific model_type next to the ONNX.
        with open(os.path.join(self._audio_onnx_dir, "config.json"),
                  "w",
                  encoding="utf-8") as f:
            json.dump({"model_type": "qwen3_asr_thinker"}, f)


class _OnnxLLM(LLM):
    """LLM subclass exercising the prebuilt-ONNX routing (no export step)."""

    # pylint: disable=super-init-not-called
    def __init__(self, tmp_path, *, with_visual: bool, with_audio: bool):
        self._eagle_engine_dir = ""
        self._tool_template_formatter = None
        self._model_id = "test"
        onnx_dir = _make_onnx_dir(tmp_path / "llm")
        visual_dir = _make_onnx_dir(tmp_path / "visual") if with_visual else ""
        audio_dir = ""
        if with_audio:
            audio_dir = _make_onnx_dir(tmp_path / "audio")
            with open(os.path.join(audio_dir, "config.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump({"model_type": "qwen3_asr_thinker"}, f)
        self._init_from_onnx(onnx_dir,
                             visual_onnx_dir=visual_dir,
                             audio_onnx_dir=audio_dir,
                             max_input_len=128,
                             max_batch_size=1,
                             max_kv_cache_capacity=256)


def _make_onnx_dir(path):
    os.makedirs(path, exist_ok=True)
    _touch(os.path.join(path, "model.onnx"))
    return str(path)


def _write_checkpoint(tmp_path, config):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(ckpt)


@pytest.fixture
def stub_rt(monkeypatch):
    rt = _StubBuilderRt()
    monkeypatch.setattr(engine_module, "_import_runtime", lambda: rt)
    return rt


def test_init_from_model_asr_builds_audio_engine(tmp_path, stub_rt):
    """qwen3_asr checkpoints export/build audio, never touch the visual path,
    and lay the engine out so the transcription capability checks pass."""
    ckpt = _write_checkpoint(tmp_path, {"model_type": "qwen3_asr"})

    llm = _CheckpointLLM(ckpt)

    assert llm.exports == ["llm", "audio"]
    assert stub_rt.builds == ["llm", "audio"]
    assert llm._is_multimodal is True
    mm_dir = llm._multimodal_engine_dir
    assert os.path.basename(mm_dir) == "multimodal"
    audio_dir = os.path.join(mm_dir, "audio")
    assert os.path.isfile(os.path.join(audio_dir, "audio_encoder.engine"))
    assert validate_multimodal_engine_dir(mm_dir)
    # Transcription endpoint capability check: audio/config.json advertises
    # an ASR model_type even without the in-process _model_type attribute.
    from experimental.server.api_server import _is_asr_model
    assert _is_asr_model(llm, audio_dir)
    assert _is_asr_model(object(), audio_dir)


def test_init_from_model_asr_reuses_cached_artifacts(tmp_path, stub_rt):
    """A second init on the same checkpoint hits the ONNX and engine caches."""
    ckpt = _write_checkpoint(tmp_path, {"model_type": "qwen3_asr"})
    first = _CheckpointLLM(ckpt)

    second = _CheckpointLLM(ckpt)

    assert second.exports == []
    assert stub_rt.builds == ["llm", "audio"]
    assert second._multimodal_engine_dir == first._multimodal_engine_dir


def test_init_from_model_vlm_still_builds_visual(tmp_path, stub_rt):
    """VLM checkpoints keep the visual branch (regression for the split)."""
    ckpt = _write_checkpoint(
        tmp_path, {
            "model_type": "qwen3_vl",
            "vision_config": {
                "image_size": 448,
                "patch_size": 14
            },
        })

    llm = _CheckpointLLM(ckpt)

    assert llm.exports == ["llm", "visual"]
    assert stub_rt.builds == ["llm", "visual"]
    mm_dir = llm._multimodal_engine_dir
    assert os.path.basename(mm_dir) == "multimodal"
    assert os.path.isfile(os.path.join(mm_dir, "visual.engine"))
    assert validate_multimodal_engine_dir(mm_dir)


def test_init_from_onnx_builds_visual_and_audio_into_shared_root(
        tmp_path, stub_rt):
    """Both encoder ONNX dirs build into one multimodal root; neither is
    silently dropped."""
    llm = _OnnxLLM(tmp_path, with_visual=True, with_audio=True)

    assert stub_rt.builds == ["llm", "visual", "audio"]
    mm_dir = llm._multimodal_engine_dir
    assert os.path.isfile(os.path.join(mm_dir, "visual.engine"))
    assert os.path.isfile(os.path.join(mm_dir, "audio",
                                       "audio_encoder.engine"))
    assert validate_multimodal_engine_dir(mm_dir)


def test_init_from_model_omni_builds_visual_and_audio(tmp_path, stub_rt):
    """Omni checkpoints carry both encoders, so neither classification set
    may shadow the other."""
    ckpt = _write_checkpoint(
        tmp_path, {
            "model_type": "qwen3_omni_moe",
            "vision_config": {
                "image_size": 448,
                "patch_size": 14
            },
        })

    llm = _CheckpointLLM(ckpt)

    assert llm.exports == ["llm", "visual", "audio"]
    assert stub_rt.builds == ["llm", "visual", "audio"]
    mm_dir = llm._multimodal_engine_dir
    assert os.path.isfile(os.path.join(mm_dir, "visual.engine"))
    assert os.path.isfile(os.path.join(mm_dir, "audio",
                                       "audio_encoder.engine"))
    assert validate_multimodal_engine_dir(mm_dir)


def test_init_from_model_prebuilt_multimodal_skips_encoder_export(
        tmp_path, stub_rt):
    """A prebuilt multimodal dir wins, so no encoder ONNX is exported for it."""
    ckpt = _write_checkpoint(
        tmp_path, {
            "model_type": "qwen3_vl",
            "vision_config": {
                "image_size": 448,
                "patch_size": 14
            },
        })
    mm_dir = tmp_path / "prebuilt"
    mm_dir.mkdir()
    _touch(mm_dir / "visual.engine")

    llm = _CheckpointLLM(ckpt, multimodal_engine_dir=str(mm_dir))

    assert llm.exports == ["llm"]
    assert stub_rt.builds == ["llm"]
    assert llm._multimodal_engine_dir == str(mm_dir)


def test_bad_multimodal_engine_dir_rejected_before_llm_export(
        tmp_path, stub_rt):
    """A typo must not cost the LLM ONNX export first."""
    ckpt = _write_checkpoint(tmp_path, {"model_type": "qwen3_vl"})
    empty = tmp_path / "prebuilt"
    empty.mkdir()

    with pytest.raises(ValueError, match="no visual or audio encoder engine"):
        _CheckpointLLM(ckpt, multimodal_engine_dir=str(empty))
    assert not os.path.exists(os.path.join(ckpt, ".edgellm"))
    assert stub_rt.builds == []


def test_cached_onnx_still_gets_multimodal_token_ids(tmp_path, stub_rt,
                                                     monkeypatch):
    """Reusing a config cached before this checkpoint gained an encoder must
    still patch the id in, or the runtime reads -1 and drops the embeddings."""
    ckpt = _write_checkpoint(
        tmp_path, {
            "model_type": "qwen3_vl",
            "vision_config": {
                "image_size": 448,
                "patch_size": 14
            },
        })
    cached = os.path.join(ckpt, ".edgellm", "onnx",
                          "llm")  # _artifacts_dir_for_model
    os.makedirs(cached, exist_ok=True)
    _touch(os.path.join(cached, "model.onnx"))
    with open(os.path.join(cached, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "qwen3_vl"}, f)  # pre-encoder config, no id

    monkeypatch.setattr(engine_module, "_ensure_export_package", lambda: None)
    monkeypatch.setattr("tensorrt_edgellm.scripts.export._find_token_id",
                        lambda *a, **kw: 151655)

    llm = _CheckpointLLM(ckpt)

    assert "llm" not in llm.exports, "cached LLM ONNX must not be re-exported"
    with open(os.path.join(cached, "config.json"), encoding="utf-8") as f:
        assert json.load(f)["image_token_id"] == 151655


def test_server_classification_matches_exporter(tmp_path):
    """A hand-copied table silently degrades new multimodal types to
    text-only, so the server must classify everything the exporter builds."""
    from experimental.server.engine import _is_multimodal, _read_model_type
    from tensorrt_edgellm.scripts import export as exporter

    for i, model_type in enumerate(sorted(exporter._VLM_MODEL_TYPES)):
        root = tmp_path / f"v{i}"
        root.mkdir()
        ckpt = _write_checkpoint(root, {"model_type": model_type})
        assert _is_multimodal(ckpt), f"{model_type} not detected as visual"
    for i, model_type in enumerate(sorted(exporter._AUDIO_MODEL_TYPES)):
        root = tmp_path / f"a{i}"
        root.mkdir()
        ckpt = _write_checkpoint(root, {"model_type": model_type})
        assert _read_model_type(ckpt) in exporter._AUDIO_MODEL_TYPES


def test_audio_onnx_dir_rejected_without_onnx_dir(tmp_path):
    """audio_onnx_dir is only meaningful alongside onnx_dir."""
    with pytest.raises(ValueError, match="audio_onnx_dir"):
        LLM(model="hf/model", audio_onnx_dir=str(tmp_path))


def test_init_from_model_text_only_has_no_multimodal_dir(tmp_path, stub_rt):
    """Plain LLM checkpoints build neither encoder."""
    ckpt = _write_checkpoint(tmp_path, {"model_type": "qwen3"})

    llm = _CheckpointLLM(ckpt)

    assert llm.exports == ["llm"]
    assert stub_rt.builds == ["llm"]
    assert llm._multimodal_engine_dir == ""
    assert llm._is_multimodal is False
