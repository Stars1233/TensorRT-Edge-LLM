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
"""Tests for the audio-output server plumbing (no GPU / native module).

Covers /v1/chat/completions audio request parsing, talker-knob and voice
validation, the shared text+audio channel pump, and TTS engine-dir checks.
"""

import threading
from collections import deque
from types import SimpleNamespace

import pytest

from experimental.server.api_server import (_apply_talker_knobs,
                                            _parse_audio_request,
                                            _validate_voice)
from experimental.server.engine import (TTS, AudioParams, _native_audio_params,
                                        _pump_channels, _stream_tts)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeFinishReason:
    NOT_FINISHED = 0
    END_ID = 1
    LENGTH = 2
    CANCELLED = 3
    ERROR = 4
    STOP_WORDS = 5


FAKE_RT = SimpleNamespace(FinishReason=_FakeFinishReason)


def _fake_llm(omni_capable=True, voices=()):
    return SimpleNamespace(omni_capable=omni_capable,
                           list_voices=lambda: list(voices))


class _FakeChannel:
    """Mimics the pybind channel pop/terminal API for one producer run."""

    def __init__(self, chunks, finished_from_start=False):
        self._chunks = deque(chunks)
        self.finished = finished_from_start
        self.cancelled = False

    def wait_pop(self, timeout_ms):
        return self.try_pop()

    def try_pop(self):
        return self._chunks.popleft() if self._chunks else None

    def is_finished(self):
        return self.finished

    def is_cancelled(self):
        return self.cancelled

    def cancel(self):
        self.cancelled = True


def _text_chunk(text, finished=False, reason=_FakeFinishReason.END_ID):
    return SimpleNamespace(
        text=text,
        token_ids=[1],
        finished=finished,
        reason=reason if finished else _FakeFinishReason.NOT_FINISHED,
        logprobs=[],
    )


def _audio_chunk(pcm16=b"", is_final=False):
    return SimpleNamespace(pcm16=pcm16, is_final=is_final, num_frames=1)


def _run_pump(text_channel, audio_channel, run=lambda: None):
    for chan in (text_channel, audio_channel):
        if chan is not None:
            chan.finished = True
    return list(_pump_channels(FAKE_RT, run, text_channel, audio_channel))


# ---------------------------------------------------------------------------
# _apply_talker_knobs / _parse_audio_request / _validate_voice
# ---------------------------------------------------------------------------


def test_talker_knobs_applied_with_types():
    params = AudioParams()
    error = _apply_talker_knobs(
        {
            "talker_temperature": 1,
            "talker_top_k": 5,
            "codec_chunk_frames": 3
        }, params, "")
    assert error is None
    assert params.talker_temperature == 1.0
    assert isinstance(params.talker_temperature, float)
    assert params.talker_top_k == 5
    assert params.codec_chunk_frames == 3


@pytest.mark.parametrize("cfg,fragment", [
    ({
        "talker_top_k": "high"
    }, "must be an integer"),
    ({
        "talker_top_k": 1.5
    }, "must be an integer"),
    ({
        "talker_top_k": True
    }, "must be an integer"),
    ({
        "talker_temperature": "hot"
    }, "must be a number"),
    ({
        "talker_temperature": -0.1
    }, "must be >= 0"),
    ({
        "codec_chunk_frames": 0
    }, "must be >= 1"),
    ({
        "max_audio_length": 0
    }, "must be >= 1"),
])
def test_talker_knobs_rejected(cfg, fragment):
    error = _apply_talker_knobs(cfg, AudioParams(), "audio.")
    assert error is not None and fragment in error
    assert error.startswith(f"'audio.{next(iter(cfg))}'")


def test_parse_audio_request_ignores_text_only():
    assert _parse_audio_request({}, _fake_llm()) == (None, None)
    assert _parse_audio_request({"modalities": ["text"]},
                                _fake_llm()) == (None, None)


@pytest.mark.parametrize("body", [
    {
        "modalities": "audio"
    },
    {
        "modalities": ["audio", "video"]
    },
    {
        "modalities": ["audio"],
        "audio": "pcm16"
    },
    {
        "modalities": ["audio"],
        "audio": {
            "format": "mp3"
        }
    },
    {
        "modalities": ["audio"],
        "audio": {
            "voice": 3
        }
    },
])
def test_parse_audio_request_rejects_malformed(body):
    params, error = _parse_audio_request(body, _fake_llm())
    assert params is None and error


def test_parse_audio_request_requires_omni_engines():
    params, error = _parse_audio_request({"modalities": ["text", "audio"]},
                                         _fake_llm(omni_capable=False))
    assert params is None and "no Omni audio engines" in error


def test_parse_audio_request_full():
    body = {
        "modalities": ["text", "audio"],
        "audio": {
            "voice": "ryan",
            "talker_temperature": 0.8,
            "codec_chunk_frames": 5,
        },
    }
    params, error = _parse_audio_request(body,
                                         _fake_llm(voices=["ryan", "serena"]))
    assert error is None
    assert params.voice == "ryan"
    assert params.talker_temperature == 0.8
    assert params.codec_chunk_frames == 5


def test_validate_voice():
    llm = _fake_llm(voices=["ryan", "serena"])
    assert _validate_voice(llm, "") is None
    assert _validate_voice(llm, "ryan") is None
    assert "available: ryan, serena" in _validate_voice(llm, "bob")
    assert _validate_voice(llm, 3) == "'voice' must be a string"
    # Model without a speaker map: accept anything rather than reject all.
    assert _validate_voice(_fake_llm(voices=[]), "anything") is None


# ---------------------------------------------------------------------------
# _pump_channels
# ---------------------------------------------------------------------------


def test_pump_audio_only_orders_and_terminates():
    audio = _FakeChannel(
        [_audio_chunk(b"a"),
         _audio_chunk(b"b", is_final=True)])
    deltas = _run_pump(None, audio)
    assert [d.audio_bytes for d in deltas] == [b"a", b"b"]
    assert all(not d.text for d in deltas)


def test_pump_empty_final_chunk_yields_nothing():
    audio = _FakeChannel([_audio_chunk(b"", is_final=True)])
    assert _run_pump(None, audio) == []


def test_pump_dual_stream_carries_text_fields():
    text = _FakeChannel(
        [_text_chunk("hello "),
         _text_chunk("world", finished=True)])
    audio = _FakeChannel([_audio_chunk(b"pcm", is_final=True)])
    deltas = _run_pump(text, audio)
    texts = [d for d in deltas if d.text]
    audios = [d for d in deltas if d.audio_bytes]
    assert [d.text for d in texts] == ["hello ", "world"]
    assert texts[0].finish_reason is None
    assert texts[1].finished and texts[1].finish_reason == "stop"
    assert [d.audio_bytes for d in audios] == [b"pcm"]


def test_pump_drains_chunks_pending_after_finish():
    # Producer finished before the consumer's first pop: nothing may be lost.
    audio = _FakeChannel(
        [_audio_chunk(b"x"), _audio_chunk(b"y")], finished_from_start=True)
    deltas = list(_pump_channels(FAKE_RT, lambda: None, None, audio))
    assert [d.audio_bytes for d in deltas] == [b"x", b"y"]


def test_pump_reraises_worker_error():
    audio = _FakeChannel([])

    def _boom():
        raise RuntimeError("talker failed")

    with pytest.raises(RuntimeError, match="talker failed"):
        list(_pump_channels(FAKE_RT, _boom, None, audio))
    assert audio.cancelled


def test_pump_cancels_channels_on_early_close():
    text = _FakeChannel([_text_chunk("partial")])
    audio = _FakeChannel([])
    gen = _pump_channels(FAKE_RT, lambda: None, text, audio)
    assert next(gen).text == "partial"
    gen.close()
    assert text.cancelled and audio.cancelled


# ---------------------------------------------------------------------------
# _native_audio_params / TTS engine-dir validation
# ---------------------------------------------------------------------------


def test_native_audio_params_maps_all_fields():
    captured = SimpleNamespace()
    rt = SimpleNamespace(OmniAudioParams=lambda: captured)
    native = _native_audio_params(
        rt, AudioParams(voice="ryan", talker_top_k=7, codec_chunk_frames=4))
    assert native is captured
    assert captured.speaker_name == "ryan"
    assert captured.talker_top_k == 7
    assert captured.codec_chunk_frames == 4
    assert not hasattr(captured, "voice")


def test_tts_rejects_missing_engine_dirs(tmp_path):
    talker = tmp_path / "talker"
    talker.mkdir()
    with pytest.raises(ValueError, match="code_predictor engine dir"):
        TTS(talker_engine_dir=str(talker))


# ---------------------------------------------------------------------------
# _stream_tts gate ownership
# ---------------------------------------------------------------------------


def _tts_rt(channel):
    return SimpleNamespace(FinishReason=_FakeFinishReason,
                           AudioStreamChannel=lambda: channel,
                           OmniAudioParams=lambda: SimpleNamespace())


def test_stream_tts_acquires_gate_for_direct_callers():
    channel = _FakeChannel([_audio_chunk(b"pcm", is_final=True)],
                           finished_from_start=True)
    sem = threading.Semaphore(1)
    runtime = SimpleNamespace(handle_request_tts=lambda *a: None)
    deltas = list(
        _stream_tts(_tts_rt(channel), runtime, "hi", AudioParams(), sem))
    assert [d.audio_bytes for d in deltas] == [b"pcm"]
    # Acquired on entry, released by the worker: back to its initial count.
    assert sem.acquire(blocking=False)


def test_stream_tts_leaves_gate_to_handoff():
    """With a handoff the HTTP layer already owns the slot; taking it again
    here would deadlock against its own non-blocking acquire."""
    channel = _FakeChannel([_audio_chunk(b"pcm", is_final=True)],
                           finished_from_start=True)
    released = []
    handoff = SimpleNamespace(worker_started=lambda: None,
                              release=lambda: released.append(True))
    runtime = SimpleNamespace(handle_request_tts=lambda *a: None)
    list(
        _stream_tts(_tts_rt(channel),
                    runtime,
                    "hi",
                    AudioParams(),
                    None,
                    admission_handoff=handoff))
    assert released == [True]


def test_stream_tts_holds_infer_guard_during_call():
    """The batcher takes only this lock, so TTS must hold it while the C++
    call runs or batched text would enter the runtime concurrently."""
    channel = _FakeChannel([_audio_chunk(b"pcm", is_final=True)],
                           finished_from_start=True)
    guard = threading.Lock()
    held = []
    runtime = SimpleNamespace(
        handle_request_tts=lambda *a: held.append(guard.locked()))
    list(
        _stream_tts(_tts_rt(channel),
                    runtime,
                    "hi",
                    AudioParams(),
                    None,
                    infer_guard=guard))
    assert held == [True]
    assert not guard.locked()
