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
"""Micro-batching for compatible non-streaming server requests."""

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger("edgellm.batching")


class BatcherOverflow(RuntimeError):
    """Raised by ``submit`` when the pending queue is at capacity.

    Lets the upstream admission layer translate backpressure into a retryable
    503 instead of letting requests pile up unboundedly behind the runtime.
    """


# Requests can share one runtime call only when these generation-level settings
# match. Per-row inputs live under LLMGenerationRequest.requests and may differ.
BATCH_COMPATIBILITY_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_generate_length",
    "lora_weights_name",
    "save_system_prompt_kv_cache",
    "apply_chat_template",
    "add_generation_prompt",
    "enable_thinking",
    "disable_spec_decode",
    "num_logprobs",
)


@dataclass
class BatchResult:
    """Runtime response slice for one original HTTP request."""

    response: Any
    index: int


@dataclass
class _QueuedRequest:
    request: Any
    future: Future


@dataclass
class _RuntimeResponseSlice:
    output_texts: List[str]
    output_ids: List[List[int]]
    finish_reasons: List[Any]
    logprobs: List[Any]
    prompt_token_counts: List[int]


def resolve_batch_size(engine_max_batch_size: int,
                       max_queue_batch_size: Optional[int]) -> int:
    """Return the effective HTTP micro-batch size."""
    if max_queue_batch_size is not None:
        if engine_max_batch_size > 0 and max_queue_batch_size > engine_max_batch_size:
            logger.warning(
                "Capping max_queue_batch_size=%d to engine max_batch_size=%d",
                max_queue_batch_size,
                engine_max_batch_size,
            )
            return engine_max_batch_size
        return max_queue_batch_size
    return engine_max_batch_size or 1


def _batch_key(request) -> Tuple[Any, ...]:
    return tuple(
        getattr(request, field) for field in BATCH_COMPATIBILITY_FIELDS)


def _is_batchable(request, video_requires_singleton: bool) -> bool:
    """Only the Nemotron video path forces batch size 1 — its runner enqueues
    video tubelets with shapes that cannot share a call. Other families' video
    requests stay batchable, so gate on the model capability. A single-frame
    clip is still a video, so key off is_video, not frames > 1."""
    if not video_requires_singleton:
        return True
    try:
        for row in request.requests:
            for buf in row.image_buffers:
                if buf.is_video:
                    return False
    except Exception:
        # Fail closed: a request we cannot introspect runs alone.
        return False
    return True


def _copy_batch_settings(source, target) -> None:
    for field in BATCH_COMPATIBILITY_FIELDS:
        setattr(target, field, getattr(source, field))
    target.stream_channels = []


def _copy_response_rows(response, start: int,
                        count: int) -> _RuntimeResponseSlice:
    end = start + count
    logprobs = getattr(response, "logprobs", []) or []
    prompt_tokens = getattr(response, "prompt_token_counts", []) or []
    return _RuntimeResponseSlice(
        output_texts=list(response.output_texts[start:end]),
        output_ids=[list(ids) for ids in response.output_ids[start:end]],
        finish_reasons=list(response.finish_reasons[start:end]),
        logprobs=list(logprobs[start:end]),
        prompt_token_counts=list(prompt_tokens[start:end]),
    )


class RequestBatcher:
    """Batch compatible requests and serialize runtime calls.

    Each submitted request is an LLMGenerationRequest. Batching appends their
    per-row ``requests`` entries into one new LLMGenerationRequest. The combined
    runtime response is split back into one response slice per submitted request.
    """

    def __init__(
        self,
        runtime_handler: Callable[[Any], Any],
        max_batch_size: int,
        timeout_ms: float,
        max_pending: Optional[int] = None,
        video_requires_singleton: bool = False,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if max_pending is not None and max_pending < 1:
            raise ValueError("max_pending must be positive")

        self._video_requires_singleton = video_requires_singleton
        self._runtime_handler = runtime_handler
        self._max_batch_size = max_batch_size
        self._timeout_s = timeout_ms / 1000.0
        self._max_pending = max_pending
        self._cv = threading.Condition()
        self._queue: List[_QueuedRequest] = []
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="edgellm-request-batcher",
            daemon=True,
        )
        self._worker.start()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def timeout_ms(self) -> float:
        return self._timeout_s * 1000.0

    @property
    def pending(self) -> int:
        with self._cv:
            return len(self._queue)

    def submit(self, request) -> BatchResult:
        future: Future = Future()
        item = _QueuedRequest(request=request, future=future)
        with self._cv:
            if self._closed:
                raise RuntimeError("Request batcher is closed")
            if (self._max_pending is not None
                    and len(self._queue) >= self._max_pending):
                raise BatcherOverflow(
                    f"batcher queue full ({self._max_pending} pending)")
            self._queue.append(item)
            self._cv.notify()
        return future.result()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            # The worker must never die: a dead worker blocks every pending
            # submit()'s Future -- and the admission slot each caller holds --
            # forever. Failed items get their exception set instead.
            try:
                batch = self._take_batch()
            except Exception:
                logger.exception("Batcher batch selection failed; continuing")
                continue
            if batch is None:
                return
            if not batch:
                continue
            self._process_batch(batch)

    def _take_batch(self) -> Optional[List[_QueuedRequest]]:
        with self._cv:
            while not self._queue and not self._closed:
                self._cv.wait()
            if not self._queue:
                return None

            first = self._queue.pop(0)
            try:
                key = _batch_key(first.request)
            except Exception as exc:
                # A request missing a compatibility field must not strand the
                # whole worker: fail just this one and move on.
                first.future.set_exception(exc)
                return []
            batch = [first]
            if not _is_batchable(first.request,
                                 self._video_requires_singleton):
                # Nemotron video request: run it alone, never merged.
                return batch
            deadline = time.monotonic() + self._timeout_s

            while len(batch) < self._max_batch_size:
                self._move_compatible_locked(batch, key)
                if len(batch) >= self._max_batch_size or self._closed:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)

            return batch

    def _move_compatible_locked(self, batch: List[_QueuedRequest],
                                key: Tuple[Any, ...]) -> None:
        idx = 0
        while idx < len(self._queue) and len(batch) < self._max_batch_size:
            item = self._queue[idx]
            try:
                compatible = _batch_key(item.request) == key and _is_batchable(
                    item.request, self._video_requires_singleton)
            except Exception:
                # Treat an unkeyable request as incompatible; it will be
                # popped first on a later round and fail cleanly there.
                compatible = False
            if compatible:
                batch.append(self._queue.pop(idx))
            else:
                idx += 1

    def _process_batch(self, batch: List[_QueuedRequest]) -> None:
        try:
            batched_request = self._make_batched_request(batch)
            response = self._runtime_handler(batched_request)
            row_offset = 0
            for item in batch:
                row_count = len(item.request.requests)
                response_slice = _copy_response_rows(response, row_offset,
                                                     row_count)
                item.future.set_result(
                    BatchResult(response=response_slice, index=0))
                row_offset += row_count
        except Exception as exc:
            logger.exception("Batched inference failed")
            for item in batch:
                item.future.set_exception(exc)

    def _make_batched_request(self, batch: List[_QueuedRequest]):
        first_request = batch[0].request
        batched_request = type(first_request)()
        _copy_batch_settings(first_request, batched_request)
        requests = []
        for item in batch:
            requests.extend(item.request.requests)
        batched_request.requests = requests
        return batched_request
