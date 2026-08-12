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
"""Tests for request batching bounds and the admission queue."""

import os
import sys
import threading
import time

import pytest

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experimental.server.batching import (BATCH_COMPATIBILITY_FIELDS,
                                          BatcherOverflow, RequestBatcher)


class _FakeRequest:
    """LLMGenerationRequest stand-in: compatibility fields + one-row requests."""

    def __init__(self):
        for field in BATCH_COMPATIBILITY_FIELDS:
            setattr(self, field, 0)
        self.requests = [object()]
        self.stream_channels = []


class _FakeResponse:

    def __init__(self, rows):
        self.output_texts = ["out"] * rows
        self.output_ids = [[1]] * rows
        self.finish_reasons = [0] * rows
        self.logprobs = [None] * rows


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.005)
    return predicate()


def test_overflow_and_pending():
    with pytest.raises(ValueError):
        RequestBatcher(lambda r: None,
                       max_batch_size=1,
                       timeout_ms=0.0,
                       max_pending=0)

    started, release = threading.Event(), threading.Event()

    def handler(request):
        started.set()
        release.wait(5.0)
        return _FakeResponse(len(request.requests))

    batcher = RequestBatcher(handler,
                             max_batch_size=1,
                             timeout_ms=0.0,
                             max_pending=2)
    errors = {}

    def submit(i):
        try:
            batcher.submit(_FakeRequest())
        except Exception as exc:  # noqa: BLE001
            errors[i] = exc

    # Occupy the single-slot worker, then fill the pending queue to max_pending.
    t0 = threading.Thread(target=submit, args=(0, ))
    t0.start()
    assert started.wait(5.0)
    parked = [threading.Thread(target=submit, args=(i, )) for i in (1, 2)]
    for t in parked:
        t.start()
    assert _wait_until(lambda: batcher.pending == 2)

    # Queue full: reject without blocking.
    with pytest.raises(BatcherOverflow):
        batcher.submit(_FakeRequest())

    release.set()
    t0.join(5.0)
    for t in parked:
        t.join(5.0)
    batcher.close()
    assert not errors  # the parked submits drained successfully


class _UnkeyableRequest:
    """Missing the compatibility fields, so _batch_key() raises."""

    def __init__(self):
        self.requests = [object()]
        self.stream_channels = []


def test_worker_survives_bad_request():
    # An unkeyable request must fail only itself, not kill the worker and
    # strand every later submit()'s Future (and the admission slot it holds).
    batcher = RequestBatcher(lambda r: _FakeResponse(len(r.requests)),
                             max_batch_size=2,
                             timeout_ms=0.0)
    try:
        with pytest.raises(Exception):
            batcher.submit(_UnkeyableRequest())
        # Worker still alive: a well-formed request completes.
        assert batcher.submit(_FakeRequest()).response.output_texts == ["out"]
    finally:
        batcher.close()


def test_admission_queue():
    from experimental.server.api_server import _AdmissionQueue

    q = _AdmissionQueue(2)
    assert q.max_depth == 2
    assert q.try_acquire() and q.try_acquire()
    assert q.depth == 2
    assert q.try_acquire() is False  # full -> refused
    q.release()
    assert q.try_acquire() is True
    q.release()
    q.release()
    q.release()  # release below zero is a no-op
    assert q.depth == 0
    # Depth is floored at 1.
    assert _AdmissionQueue(0).max_depth == 1
    assert _AdmissionQueue(-5).max_depth == 1


class _FakeBuffer:

    def __init__(self, is_video, frames=1):
        self.is_video = is_video
        self.frames = frames


class _FakeMediaRequest(_FakeRequest):
    """Request whose single row carries image buffers."""

    def __init__(self, buffers=None):
        super().__init__()
        self.requests = [type("Row", (), {"image_buffers": buffers or []})()]


@pytest.mark.parametrize("frames", [1, 8])
def test_video_request_is_unbatchable(frames):
    from experimental.server.batching import _is_batchable

    # With the Nemotron singleton capability on, a single-frame clip is still a
    # video, so both frame counts run alone.
    assert _is_batchable(_FakeMediaRequest([_FakeBuffer(True, frames)]),
                         True) is False
    # A multi-tile image (frames > 1 but not a video) stays batchable.
    assert _is_batchable(_FakeMediaRequest([_FakeBuffer(False, frames)]),
                         True) is True
    # Without the capability (non-Nemotron), video requests batch like any other.
    assert _is_batchable(_FakeMediaRequest([_FakeBuffer(True, frames)]),
                         False) is True


def test_batcher_runs_video_requests_alone():
    sizes = []

    def handler(request):
        sizes.append(len(request.requests))
        return _FakeResponse(len(request.requests))

    batcher = RequestBatcher(handler,
                             max_batch_size=4,
                             timeout_ms=100.0,
                             video_requires_singleton=True)
    try:
        threads = [
            threading.Thread(target=batcher.submit,
                             args=(_FakeMediaRequest([_FakeBuffer(True,
                                                                  1)]), ))
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5.0)
    finally:
        batcher.close()
    # Two compatible video requests never merge: each runs in its own call.
    assert sizes == [1, 1]
