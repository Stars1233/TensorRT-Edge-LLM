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
OpenAI-compatible HTTP server for TensorRT Edge-LLM.

Endpoints:
    GET  /health                  - Health check
    GET  /v1/models               - List available models
    POST /v1/chat/completions     - Chat completion (OpenAI-compatible)
    POST /v1/completions          - Legacy text completion (OpenAI-compatible)

Usage (standalone)::

    # --model takes a HuggingFace id or a local path; a local path is
    # auto-detected as a prebuilt engine dir, a prebuilt ONNX dir, or a
    # checkpoint (exports ONNX + builds an engine).
    python -m experimental.server --model Qwen/Qwen3-1.7B --port 8000
    python -m experimental.server --model /path/to/engine --port 8000

Usage (from LLM object)::

    from experimental.server import LLM
    llm = LLM(model="Qwen/Qwen3-1.7B")
    llm.serve(port=8000)
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import math
import os
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import anthropic_compat as _anthropic
from .audio_preprocess import MAX_AUDIO_UPLOAD_BYTES
from .batching import BatcherOverflow, RequestBatcher, resolve_batch_size
from .engine import (OMNI_AUDIO_SAMPLE_RATE, AudioParams, SamplingParams,
                     _normalize_logit_bias, _validate_logit_bias_spec_decode,
                     finish_reason_name)
from .tool_calling import (ToolConfig, parse_assistant_output,
                           validate_tool_request)
from .video_sampling import MAX_SOURCE_BYTES as MAX_VIDEO_SOURCE_BYTES

logger = logging.getLogger("edgellm.api_server")

# Qwen3-ASR language normalization (mirrors the HF processor's
# resolve_language): ISO codes or full names -> the canonical full name the
# model expects in its system turn. Full official 30-language set.
_ASR_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
_ASR_LANGUAGES.update({v.lower(): v for v in list(_ASR_LANGUAGES.values())})

# Fallback clip-length cap when the audio engine config carries no
# max_time_steps (same value as vLLM's max_audio_clip_s default).
DEFAULT_MAX_AUDIO_CLIP_S = 30.0


class _PrepareError(Exception):
    """Request construction failed (bad input / C++ media decode) -> 400, so
    an infer-stage failure can still surface as 500."""


class _AudioTooLongError(Exception):
    """Decoded audio exceeds the engine's time-step profile -> 413."""


def _audio_duration_limit_s(audio_dir: str) -> float:
    """Longest admissible clip in seconds: the builder's max_time_steps are
    mel frames at a 10 ms hop, so the cap is max_time_steps / 100."""
    try:
        with open(os.path.join(audio_dir, "config.json"),
                  encoding="utf-8") as f:
            steps = json.load(f).get("builder_config",
                                     {}).get("max_time_steps", 0)
        if isinstance(steps, (int, float)) and steps > 0:
            return float(steps) / 100.0
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_MAX_AUDIO_CLIP_S


def _check_audio_durations(request, limit_s: float) -> None:
    """Reject decoded audio longer than the engine profile admits before it
    reaches the C++ audio runner; buffers without PCM metadata are skipped."""
    for req in getattr(request, "requests", None) or []:
        for buf in getattr(req, "audio_buffers", None) or []:
            n = getattr(buf, "num_samples", 0) or 0
            sr = getattr(buf, "sample_rate", 0) or 0
            if n and sr and n / sr > limit_s:
                raise _AudioTooLongError(
                    f"audio is {n / sr:.1f}s long but the loaded engine "
                    f"supports at most {limit_s:.1f}s")


def _parse_content_length(value):
    """None when absent, -1 when malformed (proxies can inject non-integer
    values), else the parsed byte count."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return -1


def _iter_media_refs(messages):
    """Yield every media reference in an OpenAI message list, through the same
    parser the loaders use so no accepted spelling escapes the policy."""
    from .media_source import iter_item_media_refs

    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            yield from iter_item_media_refs(item)


def enforce_local_media_policy(messages, allowed_root):
    """Server-local media is off unless --allowed-local-media-path is set, and
    then only inside that directory: the server binds 0.0.0.0 without auth, so a
    bare path would let any client read arbitrary local files.

    Raises PermissionError, which the routes map to 403."""
    from .media_source import resolve_file_url

    root = Path(allowed_root).resolve() if allowed_root else None
    for ref in _iter_media_refs(messages):
        ref = (ref or "").strip()
        if not ref or ref.startswith(("data:", "http://", "https://")):
            continue
        if root is None:
            raise PermissionError(
                "local media paths are disabled; pass base64/data URLs or "
                "start the server with --allowed-local-media-path")
        path = resolve_file_url(ref) if ref.startswith("file:") else ref
        # resolve() first so that .. and symlinks cannot escape the root.
        if not Path(path).resolve().is_relative_to(root):
            raise PermissionError(
                f"local media path is outside --allowed-local-media-path: {ref}"
            )


def _runtime_prompt_tokens(response, response_idx, fallback):
    """The runtime's templated, media-expanded prompt length; the HF-template
    estimate undercounts multimodal placeholders, so prefer this when present."""
    counts = getattr(response, "prompt_token_counts", None) or []
    if response_idx < len(counts) and counts[response_idx]:
        return int(counts[response_idx])
    return fallback


def _num_field(body, name, default, *, kind, low=None, high=None):
    """Range-checked numeric field: rejects strings, bools and non-finite values
    that would otherwise reach the runtime and surface as a 500."""
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number")
    if kind is int and not float(value).is_integer():
        raise ValueError(f"'{name}' must be an integer")
    value = kind(value)
    if kind is float and not math.isfinite(value):
        raise ValueError(f"'{name}' must be finite")
    if (low is not None and value < low) or (high is not None
                                             and value > high):
        raise ValueError(f"'{name}' out of range")
    return value


def _bool_field(body, name, default):
    value = body.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean")
    return value


# Whole-request cap for JSON endpoints. Derived from the per-video cap so one
# maximum-size video still fits: base64 inflates it by 4/3, plus JSON framing.
MAX_REQUEST_BODY_BYTES = -(-MAX_VIDEO_SOURCE_BYTES // 3) * 4 + 1024 * 1024


def parse_stop(body):
    """OpenAI-compatible ``stop``: null | str | list[str]."""
    stop_raw = body.get("stop")
    if stop_raw is None:
        return []
    if isinstance(stop_raw, str):
        return [stop_raw]
    if isinstance(stop_raw, list) and all(
            isinstance(s, str) for s in stop_raw):
        return list(stop_raw)
    raise ValueError("'stop' must be a string or array of strings")


def parse_sampling_params(body, *, default_max_tokens):
    """Shared validation for the chat and completions bodies. Raises
    ``ValueError`` so both routes answer 400 instead of failing in the runtime."""
    # OpenAI renamed "max_tokens" to "max_completion_tokens"; the modern
    # field takes precedence when both are present.
    max_tokens_key = ("max_completion_tokens"
                      if body.get("max_completion_tokens") is not None else
                      "max_tokens")
    return dict(
        temperature=_num_field(body, "temperature", 0.7, kind=float, low=0.0),
        top_p=_num_field(body, "top_p", 0.9, kind=float, low=0.0, high=1.0),
        top_k=_num_field(body, "top_k", 50, kind=int, low=0),
        max_tokens=_num_field(body,
                              max_tokens_key,
                              default_max_tokens,
                              kind=int,
                              low=1,
                              high=_MAX_COMPLETION_TOKENS),
    )


class _ServerBusy(Exception):
    """Admission full: mapped to backpressure (503 for OpenAI paths, 529 for
    the Anthropic path) so the client retries rather than failing."""


# Bound requests in flight so a client burst gets a fast 503 instead of piling
# up. Admission and the runtime slot are acquired non-blocking: a parked pool
# thread would starve the sync SSE generator that releases the slot. Non-stream
# requests still queue -- in the batcher's own worker, not on a pool thread.
_DEFAULT_REQUEST_QUEUE_SIZE = 32
# With batching, each admitted non-stream request parks a pool thread on the
# batcher; if the depth nears the ASGI pool (default 40) none are left to run
# the streaming generators that release the slot. Warn above this.
_POOL_STARVATION_QUEUE_THRESHOLD = 40


class _AdmissionQueue:
    """Bounds admitted requests (queued + running). ``try_acquire`` is the
    upstream gate; overflow -> the caller returns 503. Execution downstream is
    the batcher (non-stream) or the single runtime slot (stream)."""

    def __init__(self, max_depth: int):
        self._max_depth = max(1, int(max_depth))
        self._lock = threading.Lock()
        self._depth = 0

    def try_acquire(self) -> bool:
        with self._lock:
            if self._depth >= self._max_depth:
                return False
            self._depth += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._depth > 0:
                self._depth -= 1

    @property
    def depth(self) -> int:
        with self._lock:
            return self._depth

    @property
    def max_depth(self) -> int:
        return self._max_depth


class _AdmissionHandoff:
    """Releases the runtime slot -- and, if given, the admission slot -- once,
    when the worker exits if it started (a join timeout must not free the slot
    mid-call), else when the ASGI response exits. Fired from both the SSE
    generator's finally and the ASGI-call finally; the once-guard keeps only
    the first."""

    def __init__(self, sem, on_release=None):
        self._sem = sem
        self._on_release = on_release
        self._lock = threading.Lock()
        self._fired = False
        self._started = False

    def _fire(self):
        with self._lock:
            if self._fired:
                return
            self._fired = True
        self._sem.release()
        if self._on_release is not None:
            self._on_release()

    def worker_started(self):
        with self._lock:
            self._started = True

    def worker_start_failed(self):
        with self._lock:
            self._started = False

    def release(self):
        self._fire()

    def release_if_unstarted(self):
        with self._lock:
            started = self._started
        if not started:
            self._fire()


def _releasing_streaming_response(content, release, **kw):
    """StreamingResponse that releases on ASGI-call exit: a client that
    disconnects before the first body iteration never starts the generator,
    so its finally cannot run."""
    from fastapi.responses import StreamingResponse

    class _Resp(StreamingResponse):

        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                release()

    return _Resp(content, **kw)


def _overloaded_response():
    """OpenAI-path backpressure (503, not 429): the server is saturated, not
    rate-limiting the client -- retry."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=503,
        content={"error": "server overloaded: request queue is full"},
        headers={"Retry-After": "1"})


def _is_asr_model(llm_instance, audio_dir: str) -> bool:
    """The transcription protocol is Qwen3-ASR specific: accept when the LLM
    or the audio encoder identifies as an ASR model type."""
    if "asr" in str(getattr(llm_instance, "_model_type", "") or "").lower():
        return True
    try:
        with open(os.path.join(audio_dir, "config.json"),
                  encoding="utf-8") as f:
            return "asr" in str(json.load(f).get("model_type", "")).lower()
    except (OSError, ValueError):
        return False


THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
IM_END_TOKEN = "<|im_end|>"

# Upper bound on requested output length: the runtime narrows to int32 and,
# with logprobs, allocates from the requested length before KV clamping, so an
# unbounded value can OOM. No real chat request needs more than this.
_MAX_COMPLETION_TOKENS = 1 << 17


def _split_reasoning_and_content(text: str):
    """Split model output into (reasoning_content, content) around <think> tags."""
    think_open = text.find(THINK_OPEN_TAG)
    think_close = text.find(THINK_CLOSE_TAG)
    if think_open != -1 and think_close != -1 and think_close > think_open:
        reasoning = text[think_open + len(THINK_OPEN_TAG):think_close].strip()
        content = text[think_close + len(THINK_CLOSE_TAG):].strip()
        return reasoning, content or None
    return None, text.strip() if text.strip() else None


def _handle_runtime_request(llm_instance, request):
    if hasattr(llm_instance, "_handle_request"):
        return llm_instance._handle_request(request)
    return llm_instance._runtime.handle_request(request)


def _run_nonstream(llm_instance, batcher, admission, build_request):
    """Admit and execute one non-streaming request; return (response, index).

    Admission is the upstream bound (raises ``_ServerBusy`` on overflow).
    Execution is the batcher when enabled, else the single runtime slot.
    ``build_request`` is deferred until after admission so a rejected request
    does no work; its errors propagate to the caller for status mapping.
    """
    if not admission.try_acquire():
        raise _ServerBusy()
    try:
        if batcher is not None:
            # The batcher merges compatible requests, so their inputs decode
            # concurrently by design (bounded by the admission depth).
            request = build_request()
            try:
                result = batcher.submit(request)
            except BatcherOverflow as exc:
                raise _ServerBusy() from exc
            return result.response, result.index
        # Non-batching: hold the runtime slot across build *and* inference so
        # media decode happens under the gate (concurrent requests must not
        # stack decoded buffers while queued behind the runtime lock).
        sem = llm_instance._admission()
        if not sem.acquire(blocking=False):
            raise _ServerBusy()
        try:
            request = build_request()
            return _handle_runtime_request(llm_instance, request), 0
        finally:
            sem.release()
    finally:
        admission.release()


def _parse_audio_request(body: Dict[str, Any], llm_instance):
    """Parse OpenAI-style ``modalities`` / ``audio`` request fields.

    Talker knobs are namespaced inside the ``audio`` object (they must not
    collide with text sampling fields like ``repetition_penalty``). Returns
    ``(AudioParams | None, error_message | None)``; AudioParams is non-None
    only when the request asks for audio output.
    """
    modalities = body.get("modalities")
    if modalities is None:
        return None, None
    if (not isinstance(modalities, list)
            or any(m not in ("text", "audio") for m in modalities)):
        return None, ("'modalities' must be a list containing only "
                      "'text' and/or 'audio'")
    if "audio" not in modalities:
        return None, None
    if not llm_instance.omni_capable:
        return None, ("audio output requested but no Omni audio engines "
                      "(talker/code_predictor/code2wav) are loaded")
    audio_cfg = body.get("audio") or {}
    if not isinstance(audio_cfg, dict):
        return None, "'audio' must be an object"
    fmt = audio_cfg.get("format", "pcm16")
    if fmt != "pcm16":
        return None, "only audio format 'pcm16' is supported"
    voice = audio_cfg.get("voice", "")
    voice_error = _validate_voice(llm_instance, voice)
    if voice_error:
        return None, voice_error

    params = AudioParams(voice=voice)
    error = _apply_talker_knobs(audio_cfg, params, field_prefix="audio.")
    if error:
        return None, error
    return params, None


def _validate_voice(llm_instance, voice) -> Optional[str]:
    """Reject unknown voice names instead of silently using the default.

    Empty voice selects the model default; an empty speaker map (model
    exposes none) skips validation. Returns an error message or None.
    """
    if not isinstance(voice, str):
        return "'voice' must be a string"
    if not voice:
        return None
    voices = llm_instance.list_voices()
    if voices and voice not in voices:
        return f"unknown voice '{voice}'; available: {', '.join(voices)}"
    return None


def _json_error(message: str, status_code: int = 400):
    """JSON error response in the OpenAI-style ``{"error": ...}`` shape."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status_code, content={"error": message})


def _inference_error_response(exc: Exception):
    """Map an inference exception to the shared 400/413/500 JSON response.

    Input longer than the engine's built max_input_len raises with an
    EDGELLM_INPUT_TOO_LONG marker -> 413 (rebuild the engine with a larger
    --maxInputLen to accept longer prompts / larger tool lists); a
    EDGELLM_BAD_MEDIA_COUNT marker is malformed input -> 400.
    """
    if isinstance(exc, (ValueError, KeyError)):
        return _json_error(f"Invalid messages: {exc}")
    if "EDGELLM_INPUT_TOO_LONG" in str(exc):
        return _json_error(str(exc), status_code=413)
    # Placeholder/media count mismatch is malformed client input, not a server
    # fault; keep the runner's diagnostic so the caller can see which one.
    if "EDGELLM_BAD_MEDIA_COUNT" in str(exc):
        return _json_error(str(exc))
    logger.exception("Inference failed")
    return _json_error(str(exc), status_code=500)


def _completion_response(llm_instance,
                         response_id,
                         message,
                         finish_reason,
                         completion_tokens,
                         logprobs=None,
                         prompt_tokens: Optional[int] = None):
    """chat.completion envelope for responses assembled outside the runtime
    response object (the audio path aggregates streamed deltas)."""
    return {
        "id":
        response_id,
        "object":
        "chat.completion",
        "created":
        int(time.time()),
        "model":
        llm_instance._model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": logprobs,
            "finish_reason": finish_reason,
        }],
        "usage":
        _usage_body(prompt_tokens, completion_tokens),
    }


#: Talker knob -> (type, minimum). Keys match AudioParams attribute names.
_TALKER_KNOBS = {
    "talker_temperature": (float, 0),
    "talker_top_p": (float, 0),
    "repetition_penalty": (float, 0),
    "talker_top_k": (int, 0),
    "max_audio_length": (int, 1),
    "codec_chunk_frames": (int, 1),
    "talker_prefill_threshold": (int, 1),
}


def _apply_talker_knobs(cfg: Dict[str, Any], params: AudioParams,
                        field_prefix: str) -> Optional[str]:
    """Validate and apply Talker sampling / chunking knobs from ``cfg``.

    Shared by the chat-completions ``audio`` object and the
    ``/v1/audio/speech`` body. Returns an error message, or None on success.
    """
    for name, (typ, minimum) in _TALKER_KNOBS.items():
        if name not in cfg:
            continue
        value = cfg[name]
        if isinstance(
                value,
                bool) or not isinstance(value, int if typ is int else
                                        (int, float)):
            kind = "an integer" if typ is int else "a number"
            return f"'{field_prefix}{name}' must be {kind}"
        if value < minimum:
            return f"'{field_prefix}{name}' must be >= {minimum}"
        setattr(params, name, typ(value))
    return None


def _create_app(llm_instance,
                *,
                enable_batching: bool = False,
                max_queue_batch_size: Optional[int] = None,
                batch_timeout_ms: float = 10.0,
                request_queue_size: int = _DEFAULT_REQUEST_QUEUE_SIZE,
                allowed_local_media_path: Optional[str] = None):
    """Create a FastAPI app backed by the given LLM instance."""
    try:
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import JSONResponse, PlainTextResponse, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for the server. "
                           "Install: pip install fastapi uvicorn") from exc

    batcher: Optional[RequestBatcher] = None
    admission = _AdmissionQueue(request_queue_size)

    def runtime_handler(request):
        return _handle_runtime_request(llm_instance, request)

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal batcher
        if enable_batching:
            engine_batch_size = int(
                getattr(llm_instance, "max_batch_size", 0) or 0)
            batch_size = resolve_batch_size(engine_batch_size,
                                            max_queue_batch_size)
            # Only the Nemotron video runner requires batch-1; other families'
            # video requests stay batchable.
            try:
                video_singleton = (
                    llm_instance._video_model_family() == "nemotron")
            except Exception:
                video_singleton = False
            batcher = RequestBatcher(
                runtime_handler=runtime_handler,
                max_batch_size=batch_size,
                timeout_ms=batch_timeout_ms,
                # Bounds the *pending* (not-yet-selected) queue; the in-flight
                # batch is separate. Admission caps queued+running upstream, so
                # this is mainly defense-in-depth for direct-driver use.
                max_pending=admission.max_depth,
                video_requires_singleton=video_singleton,
            )
            logger.info(
                "HTTP request batching enabled: max_batch_size=%d timeout_ms=%.3f "
                "request_queue_size=%d",
                batcher.max_batch_size,
                batcher.timeout_ms,
                admission.max_depth,
            )
            # See _POOL_STARVATION_QUEUE_THRESHOLD: a queue near the ASGI pool
            # size lets parked batcher threads starve streaming responses.
            if admission.max_depth >= _POOL_STARVATION_QUEUE_THRESHOLD:
                logger.warning(
                    "request_queue_size=%d is large relative to the server "
                    "thread pool; with batching this risks starving streaming "
                    "responses. Consider a smaller --request-queue-size.",
                    admission.max_depth,
                )
        try:
            yield
        finally:
            if batcher is not None:
                batcher.close()
                batcher = None

    app = FastAPI(
        title="TensorRT Edge-LLM Server",
        version="0.1.0",
        description=
        "OpenAI-compatible inference server powered by TensorRT Edge-LLM",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _cap_upload_body(request, call_next):
        # Starlette spools multipart uploads to disk before the handler runs, so
        # cap at the transport layer: require a Content-Length on the upload
        # route and bound it (margin for the multipart framing).
        if request.url.path == "/v1/audio/transcriptions":
            length = _parse_content_length(
                request.headers.get("content-length"))
            if length is None:
                return JSONResponse(
                    status_code=411,
                    content={"error": "Content-Length required"})
            if length < 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "invalid Content-Length header"})
            if length > MAX_AUDIO_UPLOAD_BYTES + 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error":
                        f"audio upload exceeds the supported "
                        f"maximum of {MAX_AUDIO_UPLOAD_BYTES} bytes"
                    })
        elif request.method == "POST":
            # A JSON body is parsed whole before the handler runs, so count the
            # bytes actually received rather than trusting Content-Length.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error":
                            f"request body exceeds the supported maximum of "
                            f"{MAX_REQUEST_BODY_BYTES} bytes"
                        })
            request._body = bytes(
                body)  # hand the buffered body to the handler
        return await call_next(request)

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": llm_instance.model_dir,
            "speculative_decoding": llm_instance.has_draft_model,
            "batching": {
                "enabled": batcher is not None,
                "max_batch_size": batcher.max_batch_size if batcher else 1,
                "timeout_ms": batcher.timeout_ms if batcher else 0.0,
            },
            "admission": {
                "queue_size": admission.max_depth,
                "in_flight": admission.depth,
            },
        }

    @app.post("/v1/messages")
    def anthropic_messages(body: Dict[str, Any]):
        """Anthropic Messages API adapter -- a thin translation over the
        OpenAI pipeline so Claude-Code-class agents target the server
        directly via ANTHROPIC_BASE_URL with no proxy. Text-only."""

        def _err(status, message):
            code, payload = _anthropic.error_response(status, message)
            return JSONResponse(status_code=code, content=payload)

        def _overloaded():
            # 529 overloaded_error is the Anthropic-native backpressure signal
            # Claude Code retries; the admission queue is full.
            code, payload = _anthropic.error_response(
                529, "server overloaded: request queue is full")
            return JSONResponse(status_code=code,
                                content=payload,
                                headers={"Retry-After": "1"})

        if not getattr(llm_instance, "text_capable", True):
            return _err(400, "this is a TTS-only server; "
                        "use /v1/audio/speech")
        if not body.get("messages"):
            return _err(400, "messages: field required")
        max_tokens = body.get("max_tokens")
        if (not isinstance(max_tokens, int) or isinstance(max_tokens, bool)
                or not 1 <= max_tokens <= _MAX_COMPLETION_TOKENS):
            return _err(
                400, "max_tokens: must be an integer in "
                f"[1, {_MAX_COMPLETION_TOKENS}]")

        try:
            messages, tools, tool_choice, sampling = (
                _anthropic.convert_request(body))
            tool_config = validate_tool_request(messages, tools, tool_choice)
        except ValueError as exc:
            return _err(400, str(exc))

        params = SamplingParams(
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            max_tokens=max_tokens,
            stop=sampling["stop"],
        )
        prompt_tokens = llm_instance.count_prompt_tokens(
            messages, tool_config=tool_config)
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        model_name = body.get("model") or llm_instance._model_id

        if body.get("stream"):
            # Streams can't batch: admit + take the runtime slot, both
            # non-blocking (see the queue-size note); contention -> 529.
            if not admission.try_acquire():
                return _overloaded()
            sem = llm_instance._admission()
            if not sem.acquire(blocking=False):
                admission.release()
                return _overloaded()
            try:
                prebuilt = llm_instance._make_generation_request(
                    messages,
                    params,
                    tools=tool_config.tools,
                    tool_choice=tool_config.tool_choice,
                    tool_config=tool_config)
            except (ValueError, KeyError) as exc:
                sem.release()
                admission.release()
                return _err(400, str(exc))
            except BaseException:
                sem.release()
                admission.release()
                raise
            handoff = _AdmissionHandoff(sem, on_release=admission.release)
            return _releasing_streaming_response(
                _anthropic.stream_run(llm_instance, messages, params,
                                      tool_config, message_id, model_name,
                                      prompt_tokens, prebuilt, handoff),
                handoff.release_if_unstarted,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                })

        def _build():
            return llm_instance._make_generation_request(
                messages,
                params,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice,
                tool_config=tool_config)

        try:
            response, response_idx = _run_nonstream(llm_instance, batcher,
                                                    admission, _build)
        except _ServerBusy:
            return _overloaded()
        except (ValueError, KeyError) as exc:
            return _err(400, f"Invalid messages: {exc}")
        except Exception as exc:
            if "EDGELLM_INPUT_TOO_LONG" in str(exc):
                return _err(413, str(exc))
            logger.exception("Inference failed")
            return _err(500, str(exc))

        raw_text = (response.output_texts[response_idx]
                    if len(response.output_texts) > response_idx else "")
        output_text = raw_text.replace(IM_END_TOKEN, "")
        output_ids = (response.output_ids[response_idx]
                      if len(response.output_ids) > response_idx else [])
        completion_tokens = len(output_ids)
        try:
            # Tool post-processing (parsing/serializing generated calls) can
            # raise; keep it inside the Anthropic error body rather than
            # surfacing a generic framework 500.
            message_body, has_tool_calls = _build_message_body(
                output_text, tool_config, llm_instance.model_dir)
        except Exception as exc:
            logger.exception("Anthropic tool post-processing failed")
            return _err(500, str(exc))
        finish_reason = (finish_reason_name(
            llm_instance._rt, response.finish_reasons[response_idx]) if len(
                response.finish_reasons) > response_idx else "stop")
        # Truncation/cancellation wins over tool_use so clients do not execute
        # potentially half-emitted calls.
        if has_tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        stop_reason = _anthropic.convert_stop_reason(finish_reason)
        blocks = _anthropic.build_content_blocks(
            message_body.get("content"),
            message_body.get("tool_calls") or [])
        return {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": _anthropic._usage(prompt_tokens, completion_tokens),
        }

    @app.post("/v1/messages/count_tokens")
    def anthropic_count_tokens(body: Dict[str, Any]):
        """Claude Code polls this for context management; a 404 here has
        destabilized other servers under the request flood."""
        if not getattr(llm_instance, "text_capable", True):
            code, payload = _anthropic.error_response(
                400, "this is a TTS-only server; use /v1/audio/speech")
            return JSONResponse(status_code=code, content=payload)
        if not body.get("messages"):
            code, payload = _anthropic.error_response(
                400, "messages: field required")
            return JSONResponse(status_code=code, content=payload)
        try:
            messages, tools, tool_choice, _ = _anthropic.convert_request(body)
            tool_config = validate_tool_request(messages, tools, tool_choice)
        except ValueError as exc:
            code, payload = _anthropic.error_response(400, str(exc))
            return JSONResponse(status_code=code, content=payload)
        count = llm_instance.count_prompt_tokens(messages,
                                                 tool_config=tool_config)
        if count is None:
            # Rough estimate rather than an authoritative 0, which would break
            # client-side context budgeting.
            chars = sum(len(str(m.get("content") or "")) for m in messages)
            count = max(1, chars // 4)
        return {"input_tokens": count}

    @app.get("/v1/voices")
    def list_voices():
        """Speaker names accepted as ``voice`` in audio requests."""
        if not getattr(llm_instance, "omni_capable", False):
            return _json_error("no TTS engines loaded on this server")
        return {
            "object":
            "list",
            "data": [{
                "id": name,
                "object": "voice"
            } for name in llm_instance.list_voices()],
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "object":
            "list",
            "data": [{
                "id": llm_instance._model_id,
                "object": "model",
                "owned_by": "tensorrt-edgellm",
            }],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(body: Dict[str, Any]):
        messages = body.get("messages", [])
        if not getattr(llm_instance, "text_capable", True):
            return _json_error(
                "this is a TTS-only server; use /v1/audio/speech")
        if not messages:
            return JSONResponse(status_code=400,
                                content={"error": "messages required"})

        try:
            enforce_local_media_policy(messages, allowed_local_media_path)
        except PermissionError as exc:
            return JSONResponse(status_code=403, content={"error": str(exc)})

        try:
            sampling = parse_sampling_params(body, default_max_tokens=2048)
            stream = _bool_field(body, "stream", False)
            enable_thinking = _bool_field(body, "enable_thinking", False)
            disable_spec_decode = _bool_field(body, "disable_spec_decode",
                                              False)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        temperature = sampling["temperature"]
        top_p = sampling["top_p"]
        top_k = sampling["top_k"]
        max_tokens = sampling["max_tokens"]
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        try:
            logit_bias = _normalize_logit_bias(body.get("logit_bias"))
            _validate_logit_bias_spec_decode(
                logit_bias,
                disable_spec_decode=disable_spec_decode,
                has_draft_model=llm_instance.has_draft_model,
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        # OpenAI logprobs: "logprobs": bool, "top_logprobs": int (0-50, mirrors
        # kMaxLogprobsK). "logprobs": true alone returns the chosen token's
        # logprob per step (num_logprobs=1); top_logprobs raises K. Absent or
        # false disables regardless of top_logprobs.
        req_logprobs_flag = body.get("logprobs", False)
        req_top_logprobs = body.get("top_logprobs")
        num_logprobs = 0
        if req_logprobs_flag:
            if req_top_logprobs is None:
                num_logprobs = 1
            elif (isinstance(req_top_logprobs, bool)
                  or not isinstance(req_top_logprobs, int)
                  or not 0 <= req_top_logprobs <= 50):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "'top_logprobs' must be an integer in [0, 50]"
                    })
            else:
                num_logprobs = max(1, req_top_logprobs)

        try:
            stop = parse_stop(body)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        try:
            tool_config = validate_tool_request(messages, tools, tool_choice)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        # OpenAI "stream_options": {"include_usage": bool} → final usage chunk.
        stream_options = body.get("stream_options")
        if stream_options is not None and not isinstance(stream_options, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "'stream_options' must be an object"})
        try:
            include_usage = bool(stream) and _bool_field(
                stream_options or {}, "include_usage", False)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        # HF-tokenizer prompt count; None (unavailable) degrades to 0.
        prompt_tokens: Optional[int] = None
        if not stream or include_usage:
            prompt_tokens = llm_instance.count_prompt_tokens(
                messages,
                tool_config=tool_config,
                enable_thinking=enable_thinking,
            )

        audio_params, audio_error = _parse_audio_request(body, llm_instance)
        if audio_error:
            return _json_error(audio_error)
        if audio_params is not None and tools:
            return _json_error("tools with audio output not supported")
        if audio_params is not None and num_logprobs > 0:
            return _json_error("logprobs with audio output not supported")

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            disable_spec_decode=disable_spec_decode,
            stop=stop,
            logit_bias=logit_bias,
            num_logprobs=num_logprobs,
        )

        # OpenAI: "logprobs": true without "top_logprobs" returns the chosen
        # token's logprob with an empty top_logprobs list.
        include_top_logprobs = req_top_logprobs is not None

        if not stream and audio_params is not None:
            return _omni_completion(llm_instance, messages, params,
                                    audio_params, response_id, admission,
                                    prompt_tokens)

        if stream:
            # Streams can't batch: admit + take the runtime slot, both
            # non-blocking (see the queue-size note). Prebuild here so bad
            # input stays a 400 and media decodes once.
            if not admission.try_acquire():
                return _overloaded_response()
            sem = llm_instance._admission()
            if not sem.acquire(blocking=False):
                admission.release()
                return _overloaded_response()
            try:
                prebuilt_request = llm_instance._make_generation_request(
                    messages,
                    params,
                    tools=tool_config.tools,
                    tool_choice=tool_config.tool_choice)
            except (ValueError, KeyError) as exc:
                sem.release()
                admission.release()
                return JSONResponse(status_code=400,
                                    content={"error": str(exc)})
            except BaseException:
                sem.release()
                admission.release()
                raise
            handoff = _AdmissionHandoff(sem, on_release=admission.release)
            return _releasing_streaming_response(
                _generate_stream_sse(
                    llm_instance,
                    messages,
                    params,
                    response_id,
                    enable_thinking,
                    tool_config=tool_config,
                    include_top_logprobs=include_top_logprobs,
                    include_usage=include_usage,
                    prompt_tokens=prompt_tokens,
                    prebuilt_request=prebuilt_request,
                    handoff=handoff,
                    audio_params=audio_params,
                ),
                handoff.release_if_unstarted,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        def _build():
            return llm_instance._make_generation_request(
                messages,
                params,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice,
                tool_config=tool_config,
            )

        try:
            response, response_idx = _run_nonstream(llm_instance, batcher,
                                                    admission, _build)
        except _ServerBusy:
            return _overloaded_response()
        except (ValueError, KeyError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid messages: {exc}"},
            )
        except Exception as exc:
            return _inference_error_response(exc)

        return _build_chat_completion_response(
            llm_instance,
            response,
            response_idx,
            response_id,
            tool_config,
            include_top_logprobs=include_top_logprobs,
            include_logprobs=num_logprobs > 0,
            prompt_tokens=prompt_tokens,
        )

    @app.post("/v1/audio/speech")
    def audio_speech(body: Dict[str, Any]):
        """OpenAI-style TTS endpoint (Qwen3-TTS / Qwen3-Omni Talker).

        ``response_format: "pcm"`` (default) streams raw int16 mono PCM at
        24 kHz as chunks are vocoded; ``"wav"`` aggregates and returns one
        WAV file. Talker knobs (``talker_temperature`` etc.) sit at the top
        level of the body.
        """
        if not getattr(llm_instance, "omni_capable", False):
            return _json_error("no TTS engines loaded on this server")
        text = body.get("input")
        if not isinstance(text, str) or not text.strip():
            return _json_error("'input' must be a non-empty string")
        if len(text) > 4096:
            return _json_error("'input' must be at most 4096 characters")
        voice = body.get("voice", "")
        voice_error = _validate_voice(llm_instance, voice)
        if voice_error:
            return _json_error(voice_error)
        response_format = body.get("response_format", "pcm")
        if response_format not in ("pcm", "wav"):
            return _json_error("only response_format 'pcm' (streamed) and "
                               "'wav' are supported")
        params = AudioParams(voice=voice)
        error = _apply_talker_knobs(body, params, field_prefix="")
        if error:
            return _json_error(error)

        # Synthesis is admitted like chat and takes the single runtime slot;
        # both non-blocking, then handed to the worker (see the queue-size
        # note) so a join timeout cannot free them mid-call.
        if not admission.try_acquire():
            return _overloaded_response()
        sem = llm_instance._admission()
        if not sem.acquire(blocking=False):
            admission.release()
            return _overloaded_response()
        handoff = _AdmissionHandoff(sem, on_release=admission.release)

        def _pcm_chunks():
            for delta in llm_instance.generate_speech_stream(
                    text, params, admission_handoff=handoff):
                if delta.audio_bytes:
                    yield delta.audio_bytes

        if response_format == "wav":
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(OMNI_AUDIO_SAMPLE_RATE)
                    wav.writeframes(b"".join(_pcm_chunks()))
            except Exception as exc:
                return _inference_error_response(exc)
            finally:
                handoff.release_if_unstarted()
            return Response(content=buf.getvalue(), media_type="audio/wav")

        return _releasing_streaming_response(
            _pcm_chunks(),
            handoff.release_if_unstarted,
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(OMNI_AUDIO_SAMPLE_RATE),
                "X-Channels": "1",
                "X-Sample-Format": "s16le",
            },
        )

    @app.post("/v1/completions")
    async def completions(body: Dict[str, Any]):
        """OpenAI legacy text-completions: the prompt is tokenized verbatim
        (no chat template), mirroring the pre-templated tool-prompt path."""
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            if len(prompt) != 1:
                return JSONResponse(
                    status_code=400,
                    content={"error": "batch prompts unsupported"})
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt:
            return JSONResponse(status_code=400,
                                content={"error": "prompt required"})
        if body.get("echo"):
            return JSONResponse(status_code=400,
                                content={"error": "'echo' is not supported"})
        if body.get("n") not in (None, 1):
            return JSONResponse(status_code=400,
                                content={"error": "'n' must be 1"})
        if body.get("logprobs") is not None:
            return JSONResponse(
                status_code=400,
                content={"error": "'logprobs' is not supported"})

        try:
            stop = parse_stop(body)
            sampling = parse_sampling_params(body, default_max_tokens=16)
            stream = _bool_field(body, "stream", False)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        params = SamplingParams(**sampling, stop=stop)
        response_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model_name = llm_instance._model_id
        messages = [{"role": "user", "content": prompt}]

        def _build_request():
            # Reuse the chat request builder, then disable templating so the
            # runtime tokenizes the prompt text verbatim.
            request = llm_instance._make_generation_request(messages, params)
            request.apply_chat_template = False
            request.add_generation_prompt = False
            return request

        loop = asyncio.get_running_loop()
        if stream:
            # Admission: non-blocking (503 when busy; a parked pool thread
            # would starve the SSE generator that releases the gate), then
            # built before the SSE response so bad input stays a 400.
            def _admit_and_build():
                sem = llm_instance._admission()
                if not sem.acquire(blocking=False):
                    raise _ServerBusy()
                try:
                    return sem, _build_request()
                except BaseException:
                    sem.release()
                    raise

            try:
                sem, prebuilt_request = await loop.run_in_executor(
                    None, _admit_and_build)
            except _ServerBusy:
                return _overloaded_response()
            except (ValueError, KeyError) as exc:
                return JSONResponse(status_code=400,
                                    content={"error": str(exc)})
            handoff = _AdmissionHandoff(sem)
            return _releasing_streaming_response(
                _completion_stream_sse(llm_instance, messages, params,
                                       response_id, created, model_name,
                                       prebuilt_request, handoff),
                handoff.release_if_unstarted,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        def _prepare_and_infer():
            sem = llm_instance._admission()
            if not sem.acquire(blocking=False):
                raise _ServerBusy()
            try:
                return llm_instance._handle_request(_build_request())
            finally:
                sem.release()

        try:
            response = await loop.run_in_executor(None, _prepare_and_infer)
        except _ServerBusy:
            return _overloaded_response()
        except (ValueError, KeyError) as exc:
            return JSONResponse(status_code=400,
                                content={"error": f"Invalid prompt: {exc}"})
        except Exception as exc:  # noqa: BLE001
            return _inference_error_response(exc)

        output_text = (response.output_texts[0] if response.output_texts else
                       "").replace(IM_END_TOKEN, "")
        output_ids = response.output_ids[0] if response.output_ids else []
        finish_reason = (finish_reason_name(llm_instance._rt,
                                            response.finish_reasons[0])
                         if response.finish_reasons else "stop")
        return {
            "id":
            response_id,
            "object":
            "text_completion",
            "created":
            created,
            "model":
            model_name,
            "choices": [{
                "index": 0,
                "text": output_text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }],
            "usage":
            _usage_body(_runtime_prompt_tokens(response, 0, None),
                        len(output_ids)),
        }

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
            file: UploadFile = File(...),
            model: str = Form(""),
            prompt: str = Form(""),
            language: str = Form(""),
            response_format: str = Form("json"),
            temperature: float = Form(0.0),
    ):
        """OpenAI-compatible ASR endpoint (Whisper SDK): routes the upload through
        the ``input_audio`` chat path (C++ extracts mel + transcribes).
        ``model``/``language`` accepted for SDK compatibility."""
        audio_dir = os.path.join(
            getattr(llm_instance, "_multimodal_engine_dir", "") or "", "audio")
        if not os.path.isdir(audio_dir):
            return JSONResponse(status_code=400,
                                content={
                                    "error":
                                    "the loaded engine has no audio encoder; "
                                    "transcription is unsupported"
                                })
        # The prompt/output protocol below is Qwen3-ASR specific; other
        # audio-capable families (Omni) do chat-audio, not transcription.
        if not _is_asr_model(llm_instance, audio_dir):
            return JSONResponse(status_code=400,
                                content={
                                    "error":
                                    "the loaded model is not an ASR model; "
                                    "transcription is unsupported"
                                })
        if response_format not in ("json", "text"):
            return JSONResponse(status_code=400,
                                content={
                                    "error":
                                    f"unsupported response_format "
                                    f"{response_format!r}; use json or text"
                                })
        # Bounded read: one extra byte detects oversize without buffering
        # an unbounded upload (the base64 copy would double it again).
        raw = await file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
        if len(raw) > MAX_AUDIO_UPLOAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error":
                    f"audio upload exceeds the supported "
                    f"maximum of {MAX_AUDIO_UPLOAD_BYTES} bytes"
                })
        if not raw:
            return JSONResponse(status_code=400,
                                content={"error": "empty audio file"})
        ext = (os.path.splitext(file.filename or "")[1].lstrip(".").lower()
               or "wav")
        content: List[Dict[str, Any]] = [{
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(raw).decode(),
                "format": ext,
            },
        }]
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages = []
        if language:
            # HF Qwen3-ASR protocol: the normalized language name is a
            # system turn ("en" -> "English").
            lang = _ASR_LANGUAGES.get(language.strip().lower())
            if not lang:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"unsupported language {language!r}"})
            messages.append({"role": "system", "content": lang})
        messages.append({"role": "user", "content": content})
        params = SamplingParams(temperature=temperature,
                                top_p=1.0,
                                top_k=1,
                                max_tokens=4096)

        max_audio_s = _audio_duration_limit_s(audio_dir)

        def _prepare_and_infer():
            # Count audio against the same admission bound as chat (the upload
            # read is already capped by the content-length middleware).
            if not admission.try_acquire():
                raise _ServerBusy()
            try:
                sem = llm_instance._admission()
                if not sem.acquire(blocking=False):
                    raise _ServerBusy()
                try:
                    try:
                        request = llm_instance._make_generation_request(
                            messages, params)
                    except (ValueError, KeyError, RuntimeError) as exc:
                        raise _PrepareError(str(exc)) from exc
                    _check_audio_durations(request, max_audio_s)
                    return llm_instance._handle_request(request)
                finally:
                    sem.release()
            finally:
                admission.release()

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None, _prepare_and_infer)
        except _ServerBusy:
            return _overloaded_response()
        except _AudioTooLongError as exc:
            return JSONResponse(status_code=413, content={"error": str(exc)})
        except _PrepareError as exc:
            if "EDGELLM_INPUT_TOO_LONG" in str(exc):
                return JSONResponse(status_code=413,
                                    content={"error": str(exc)})
            return JSONResponse(status_code=400,
                                content={"error": f"Invalid audio: {exc}"})
        except Exception as exc:  # noqa: BLE001
            if "EDGELLM_INPUT_TOO_LONG" in str(exc):
                return JSONResponse(status_code=413,
                                    content={"error": str(exc)})
            logger.exception("Transcription failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})
        text = (response.output_texts[0] if response.output_texts else
                "").replace(IM_END_TOKEN, "").strip()
        # Qwen3-ASR output protocol: "language <LANG><asr_text><text>";
        # split on the delimiter (HF _parse_single_output semantics; the HF
        # _detect_and_fix_repetitions pass is a known gap).
        detected = ""
        if "<asr_text>" in text:
            prefix, text = text.split("<asr_text>", 1)
            text = text.strip()
            # First non-empty prefix line is the language ("language X" or a
            # bare name); "language None" is HF's empty-audio sentinel.
            for line in prefix.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower() == "language none":
                    break
                if line.lower().startswith("language "):
                    detected = line[len("language "):].strip()
                else:
                    detected = line
                break
        if response_format == "text":
            return PlainTextResponse(text)
        body = {"text": text}
        if detected:
            body["language"] = detected
        return body

    return app


class _ThinkingStateMachine:
    """Tracks <think>...</think> boundaries across streaming deltas."""

    def __init__(self, thinking_enabled: bool):
        self._enabled = thinking_enabled
        self._in_think = False
        self._think_opened = False
        self._buf = ""

    def feed(self, text: str):
        """Yield (field, text) pairs: field is 'reasoning' or 'content'."""
        if not self._enabled:
            yield "content", text
            return

        self._buf += text
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(THINK_OPEN_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_OPEN_TAG):
                        safe = self._buf[:-len(THINK_OPEN_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe and self._think_opened:
                            yield "content", safe
                        elif safe:
                            yield "content", safe
                    break
                if idx > 0 and self._think_opened:
                    yield "content", self._buf[:idx]
                elif idx > 0:
                    yield "content", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_OPEN_TAG):]
                self._in_think = True
                self._think_opened = True
            else:
                idx = self._buf.find(THINK_CLOSE_TAG)
                if idx == -1:
                    if len(self._buf) > len(THINK_CLOSE_TAG):
                        safe = self._buf[:-len(THINK_CLOSE_TAG)]
                        self._buf = self._buf[len(safe):]
                        if safe:
                            yield "reasoning", safe
                    break
                if idx > 0:
                    yield "reasoning", self._buf[:idx]
                self._buf = self._buf[idx + len(THINK_CLOSE_TAG):]
                self._in_think = False

    def flush(self):
        """Flush remaining buffer at end of stream."""
        if self._buf:
            field = "reasoning" if self._in_think else "content"
            yield field, self._buf
            self._buf = ""


def _sse_error(message: str) -> str:
    """Terminal SSE event before ``[DONE]``: streaming clients get an actionable
    failure instead of a silent ``finish_reason=error``."""
    return "data: " + json.dumps({"error": {"message": message}}) + "\n\n"


def _omni_completion(llm_instance, messages, params, audio_params, response_id,
                     admission, prompt_tokens):
    """Non-streaming Omni completion: run the streaming pipeline internally
    and return the aggregated text plus one base64 PCM audio blob.

    Admitted like any other request; both gates are taken non-blocking here
    (a parked pool thread would starve the streaming generators that release
    them) and handed to the worker.
    """
    if not admission.try_acquire():
        return _overloaded_response()
    sem = llm_instance._admission()
    if not sem.acquire(blocking=False):
        admission.release()
        return _overloaded_response()
    handoff = _AdmissionHandoff(sem, on_release=admission.release)
    text_parts: List[str] = []
    pcm_parts: List[bytes] = []
    token_count = 0
    finish_reason: Optional[str] = None
    try:
        for delta in llm_instance.generate_stream_with_audio(
                messages,
                params,
                audio_params=audio_params,
                admission_handoff=handoff):
            if delta.text:
                text_parts.append(delta.text)
            token_count += len(delta.token_ids)
            if delta.audio_bytes:
                pcm_parts.append(delta.audio_bytes)
            if delta.finished:
                finish_reason = delta.finish_reason
    except Exception as exc:
        return _inference_error_response(exc)
    finally:
        handoff.release_if_unstarted()

    full_text = "".join(text_parts).replace(IM_END_TOKEN, "")
    reasoning, content = _split_reasoning_and_content(full_text)
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
    message["audio"] = {
        "id": f"audio-{response_id}",
        "data": base64.b64encode(b"".join(pcm_parts)).decode(),
        "format": "pcm16",
        "sample_rate": OMNI_AUDIO_SAMPLE_RATE,
        "transcript": content or "",
    }
    return _completion_response(llm_instance,
                                response_id,
                                message,
                                finish_reason or "stop",
                                token_count,
                                prompt_tokens=prompt_tokens)


def _generate_stream_sse(llm_instance,
                         messages,
                         params,
                         response_id,
                         enable_thinking,
                         tool_config: Optional[ToolConfig] = None,
                         include_top_logprobs: bool = True,
                         include_usage: bool = False,
                         prompt_tokens: Optional[int] = None,
                         prebuilt_request=None,
                         handoff=None,
                         audio_params: Optional[AudioParams] = None):
    """Yield SSE chunks via StreamChannel streaming. ``handoff`` carries the
    admission gate; it is only released here while no worker owns it."""
    try:
        yield from _generate_stream_sse_inner(
            llm_instance, messages, params, response_id, enable_thinking,
            tool_config, include_top_logprobs, include_usage, prompt_tokens,
            prebuilt_request, handoff, audio_params)
    finally:
        if handoff is not None:
            handoff.release_if_unstarted()


def _generate_stream_sse_inner(llm_instance,
                               messages,
                               params,
                               response_id,
                               enable_thinking,
                               tool_config: Optional[ToolConfig] = None,
                               include_top_logprobs: bool = True,
                               include_usage: bool = False,
                               prompt_tokens: Optional[int] = None,
                               prebuilt_request=None,
                               handoff=None,
                               audio_params: Optional[AudioParams] = None):
    """Yield SSE chunks via StreamChannel streaming.

    With ``audio_params`` set (mutually exclusive with tools/logprobs), the
    Omni dual-stream pipeline runs instead and PCM chunks are interleaved as
    ``delta.audio`` per the OpenAI chat-completions audio schema.
    """
    model_name = llm_instance._model_id
    created = int(time.time())

    def emit(delta, **kw):
        return _sse_chunk(response_id,
                          delta,
                          model=model_name,
                          created=created,
                          null_usage=include_usage,
                          **kw)

    yield emit({"role": "assistant"})

    if tool_config is not None and tool_config.parse_output:
        yield from _generate_tool_stream_sse(llm_instance,
                                             messages,
                                             params,
                                             response_id,
                                             tool_config,
                                             prebuilt_request,
                                             handoff,
                                             include_usage=include_usage,
                                             prompt_tokens=prompt_tokens)
        return

    sm = _ThinkingStateMachine(enable_thinking)
    audio_id = f"audio-{response_id}"
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    completion_tokens = 0

    stream_tools = tool_config.tools if tool_config else None
    stream_tool_choice = tool_config.tool_choice if tool_config else None

    if audio_params is not None:
        deltas = llm_instance.generate_stream_with_audio(
            messages,
            params,
            audio_params=audio_params,
            prebuilt_request=prebuilt_request,
            admission_handoff=handoff)
    else:
        deltas = llm_instance.generate_stream(
            messages,
            params,
            tools=stream_tools,
            tool_choice=stream_tool_choice,
            prebuilt_request=prebuilt_request,
            admission_handoff=handoff)
    try:
        for delta in deltas:
            completion_tokens += len(delta.token_ids or [])
            lp_obj: Optional[Dict[str, Any]] = None
            if delta.logprobs:
                # OpenAI streaming schema: choices[0].logprobs = {"content": [...]},
                # one entry per generated token, mirroring the non-streaming _format_logprobs.
                content = []
                for token_id, step in zip(delta.token_ids, delta.logprobs):
                    top = [{
                        "token": e.token,
                        "token_id": e.token_id,
                        "bytes": e.bytes,
                        "logprob": float(e.logprob),
                    } for e in step]
                    chosen = next(
                        (t for t in top if t["token_id"] == token_id), None)
                    content.append({
                        "token":
                        chosen["token"] if chosen else "",
                        "token_id":
                        token_id,
                        "bytes":
                        chosen["bytes"] if chosen else [],
                        "logprob":
                        chosen["logprob"] if chosen else None,
                        "top_logprobs":
                        top if include_top_logprobs else [],
                    })
                lp_obj = {"content": content}
            if delta.text:
                for field, text in sm.feed(delta.text):
                    yield emit({field: text}, logprobs=lp_obj)
                    lp_obj = None  # logprobs only on the first chunk per delta
            if lp_obj is not None:
                yield emit({}, logprobs=lp_obj)
            if delta.audio_bytes:
                yield emit({
                    "audio": {
                        "id": audio_id,
                        "data": base64.b64encode(delta.audio_bytes).decode(),
                        "format": "pcm16",
                        "sample_rate": OMNI_AUDIO_SAMPLE_RATE,
                    }
                })
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
    except Exception as exc:
        logger.exception("Streaming inference failed")
        finish_reason = "error"
        error_message = str(exc)

    for field, text in sm.flush():
        yield emit({field: text})

    # Audio requests surface every mid-stream failure (talker/vocode errors
    # must not truncate silently); the text path keeps its narrower contract.
    if error_message and (audio_params is not None
                          or "EDGELLM_INPUT_TOO_LONG" in error_message):
        yield _sse_error(error_message)
    yield emit({}, finish_reason=finish_reason or "stop")
    if include_usage:
        yield _sse_usage_chunk(response_id,
                               prompt_tokens,
                               completion_tokens,
                               model=model_name,
                               created=created)
    yield "data: [DONE]\n\n"


def _generate_tool_stream_sse(llm_instance,
                              messages,
                              params,
                              response_id,
                              tool_config: ToolConfig,
                              prebuilt_request=None,
                              handoff=None,
                              include_usage: bool = False,
                              prompt_tokens: Optional[int] = None):
    model_name = llm_instance._model_id
    created = int(time.time())

    def emit(delta, **kw):
        return _sse_chunk(response_id,
                          delta,
                          model=model_name,
                          created=created,
                          null_usage=include_usage,
                          **kw)

    text_parts: List[str] = []
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    completion_tokens = 0

    try:
        for delta in llm_instance.generate_stream(
                messages,
                params,
                prebuilt_request=prebuilt_request,
                admission_handoff=handoff,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice):
            completion_tokens += len(delta.token_ids or [])
            if delta.text:
                text_parts.append(delta.text)
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
    except Exception as exc:
        logger.exception("Streaming inference failed")
        finish_reason = "error"
        error_message = str(exc)

    output_text = "".join(text_parts).replace(IM_END_TOKEN, "")
    parsed = parse_assistant_output(output_text, tool_config,
                                    llm_instance.model_dir)
    tool_index = 0
    for event in parsed.events:
        if event["type"] == "reasoning" and event["text"]:
            yield emit({"reasoning": event["text"]})
        elif event["type"] == "content" and event["text"]:
            yield emit({"content": event["text"]})
        elif event["type"] == "tool_call":
            call = event["tool_call"]
            yield emit({
                "tool_calls": [{
                    "index": tool_index,
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": "",
                    },
                }]
            })
            if call.arguments:
                yield emit({
                    "tool_calls": [{
                        "index": tool_index,
                        "function": {
                            "arguments": call.arguments,
                        },
                    }]
                })
            tool_index += 1

    finish = "tool_calls" if tool_index else finish_reason or "stop"
    if error_message and "EDGELLM_INPUT_TOO_LONG" in error_message:
        yield _sse_error(error_message)
    yield emit({}, finish_reason=finish)
    if include_usage:
        yield _sse_usage_chunk(response_id,
                               prompt_tokens,
                               completion_tokens,
                               model=model_name,
                               created=created)
    yield "data: [DONE]\n\n"


def _completion_sse_chunk(response_id: str,
                          created: int,
                          model: str,
                          text: str,
                          finish_reason: Optional[str] = None) -> str:
    """Legacy-completions SSE chunk: incremental ``text``, no chat delta."""
    payload = {
        "id":
        response_id,
        "object":
        "text_completion",
        "created":
        created,
        "model":
        model,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _completion_stream_sse(llm_instance, messages, params, response_id,
                           created, model, prebuilt_request, handoff):
    """Yield legacy-completions SSE chunks, then a finish_reason chunk and
    ``[DONE]``. ``handoff`` carries the admission gate (worker-owned once
    started)."""
    try:
        finish_reason: Optional[str] = None
        error_message: Optional[str] = None
        try:
            for delta in llm_instance.generate_stream(
                    messages,
                    params,
                    prebuilt_request=prebuilt_request,
                    admission_handoff=handoff):
                if delta.text:
                    yield _completion_sse_chunk(response_id, created, model,
                                                delta.text)
                if delta.finished:
                    finish_reason = delta.finish_reason or "stop"
        except Exception as exc:
            logger.exception("Streaming inference failed")
            finish_reason = "error"
            error_message = str(exc)
        if error_message and "EDGELLM_INPUT_TOO_LONG" in error_message:
            yield _sse_error(error_message)
        yield _completion_sse_chunk(response_id, created, model, "",
                                    finish_reason or "stop")
        yield "data: [DONE]\n\n"
    finally:
        if handoff is not None:
            handoff.release_if_unstarted()


def _entry_to_openai(entry) -> Dict[str, Any]:
    """Convert one native (pybind) LogprobEntry into an OpenAI-style dict.

    ``entry.piece`` is raw token bytes: ``token`` is the UTF-8-sanitized string
    (invalid/partial bytes -> U+FFFD) and ``bytes`` carries the raw bytes so
    clients can losslessly reconstruct tokens split across multi-byte chars.
    """
    piece = entry.piece  # bytes
    return {
        "token": piece.decode("utf-8", "replace"),
        "token_id": entry.token_id,
        "bytes": list(piece),
        "logprob": float(entry.logprob),
    }


def _format_logprobs(response,
                     response_idx: int = 0,
                     include_top: bool = True) -> Optional[Dict[str, Any]]:
    """Format C++ logprobs into an OpenAI-compatible logprobs object, or None.

    Each entry carries ``token`` (decoded string), ``bytes`` (raw token bytes),
    ``token_id`` (kept as a superset for internal callers) and ``logprob``.
    ``include_top=False`` emits ``top_logprobs: []`` per entry — OpenAI's shape
    for requests with ``logprobs: true`` but no ``top_logprobs``.

    Note: with non-greedy sampling the sampled token may not appear in the
    top-K candidates (logprobs are computed at temperature=1.0). In that case
    ``token``/``bytes`` are empty and ``logprob`` is ``null`` for the chosen
    token; ``top_logprobs`` still lists the candidates. Greedy is unaffected.
    """
    if not getattr(response, "logprobs", []):
        return None
    output_ids = (response.output_ids[response_idx]
                  if len(response.output_ids) > response_idx else [])
    step_logprobs = (response.logprobs[response_idx]
                     if len(response.logprobs) > response_idx else [])
    if not step_logprobs:
        return None

    content = []
    for token_id, step_topk in zip(output_ids, step_logprobs):
        top = [_entry_to_openai(e) for e in step_topk]
        chosen = next((d for d in top if d["token_id"] == token_id), None)
        content.append({
            "token": chosen["token"] if chosen else "",
            "token_id": token_id,
            "bytes": chosen["bytes"] if chosen else [],
            "logprob": chosen["logprob"] if chosen else None,
            "top_logprobs": top if include_top else [],
        })
    return {"content": content}


def _build_message_body(output_text: str, tool_config: ToolConfig,
                        model_dir: str) -> Tuple[Dict[str, Any], bool]:
    if not tool_config.parse_output:
        reasoning, answer = _split_reasoning_and_content(output_text)
        message_body: Dict[str, Any] = {"role": "assistant"}
        if reasoning is not None:
            message_body["reasoning"] = reasoning
        message_body["content"] = (
            (answer if answer is not None else reasoning) or "")
        return message_body, False

    parsed = parse_assistant_output(output_text, tool_config, model_dir)
    tool_calls = [call.to_openai() for call in parsed.tool_calls]
    message_body: Dict[str, Any] = {"role": "assistant"}
    if parsed.reasoning:
        message_body["reasoning"] = parsed.reasoning
    content = parsed.content.strip()
    message_body["content"] = content if content or not tool_calls else None
    if tool_calls:
        message_body["tool_calls"] = tool_calls
    return message_body, bool(tool_calls)


def _build_chat_completion_response(llm_instance,
                                    response,
                                    response_idx: int,
                                    response_id: str,
                                    tool_config: ToolConfig,
                                    *,
                                    include_top_logprobs: bool = True,
                                    include_logprobs: bool = False,
                                    prompt_tokens: Optional[int] = None):
    raw_text = (response.output_texts[response_idx]
                if len(response.output_texts) > response_idx else "")
    output_text = raw_text.replace(IM_END_TOKEN, "")
    output_ids = (response.output_ids[response_idx]
                  if len(response.output_ids) > response_idx else [])
    completion_tokens = len(output_ids)

    message_body, has_tool_calls = _build_message_body(output_text,
                                                       tool_config,
                                                       llm_instance.model_dir)

    if len(response.finish_reasons) > response_idx:
        finish_reason = finish_reason_name(
            llm_instance._rt, response.finish_reasons[response_idx])
    else:
        finish_reason = "stop"
    if has_tool_calls:
        finish_reason = "tool_calls"

    logprobs_obj = (_format_logprobs(
        response, response_idx=response_idx, include_top=include_top_logprobs)
                    if include_logprobs else None)
    return {
        "id":
        response_id,
        "object":
        "chat.completion",
        "created":
        int(time.time()),
        "model":
        llm_instance._model_id,
        "choices": [{
            "index": 0,
            "message": message_body,
            "logprobs": logprobs_obj,
            "finish_reason": finish_reason,
        }],
        "usage":
        _usage_body(
            _runtime_prompt_tokens(response, response_idx, prompt_tokens),
            completion_tokens),
    }


def _usage_body(prompt_tokens: Optional[int],
                completion_tokens: int) -> Dict[str, int]:
    """OpenAI usage object; None prompt_tokens (counting unavailable) -> 0."""
    pt = prompt_tokens or 0
    return {
        "prompt_tokens": pt,
        "completion_tokens": completion_tokens,
        "total_tokens": pt + completion_tokens,
    }


def _sse_usage_chunk(response_id: str,
                     prompt_tokens: Optional[int],
                     completion_tokens: int,
                     model: str = "",
                     created: Optional[int] = None) -> str:
    """include_usage final chunk: empty choices + usage, right before [DONE]."""
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [],
        "usage": _usage_body(prompt_tokens, completion_tokens),
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_chunk(response_id: str,
               delta: dict,
               finish_reason: Optional[str] = None,
               logprobs: Optional[Dict[str, Any]] = None,
               model: str = "",
               created: Optional[int] = None,
               null_usage: bool = False):
    # OpenAI streaming spec: every chunk carries finish_reason (null until the
    # terminal chunk); some clients detect termination by a non-null value.
    choice: Dict[str, Any] = {
        "delta": delta,
        "index": 0,
        "finish_reason": finish_reason or None,
    }
    if logprobs:
        choice["logprobs"] = logprobs
    payload: Dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [choice],
    }
    if null_usage:
        payload["usage"] = None
    return f"data: {json.dumps(payload)}\n\n"


def run_server(llm_instance,
               host: str = "0.0.0.0",
               port: int = 8000,
               *,
               enable_batching: bool = False,
               batch_timeout_ms: float = 10.0,
               max_queue_batch_size: Optional[int] = None,
               request_queue_size: int = _DEFAULT_REQUEST_QUEUE_SIZE,
               allowed_local_media_path: Optional[str] = None) -> None:
    """Start the OpenAI-compatible server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required. Install: pip install uvicorn") from exc

    app = _create_app(
        llm_instance,
        enable_batching=enable_batching,
        batch_timeout_ms=batch_timeout_ms,
        max_queue_batch_size=max_queue_batch_size,
        request_queue_size=request_queue_size,
        allowed_local_media_path=allowed_local_media_path,
    )
    logger.info("Starting server on %s:%d ...", host, port)
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="TensorRT Edge-LLM OpenAI-compatible server")
    # Single model source. A local path is auto-detected as a prebuilt engine
    # dir, a prebuilt ONNX dir, or a checkpoint, then routed to the matching one
    # of the LLM class's three init modes.
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path. A local path is auto-detected "
        "as a prebuilt engine dir, a prebuilt ONNX dir, or a checkpoint (exports "
        "ONNX + builds an engine).",
    )
    parser.add_argument(
        "--multimodal-engine-dir",
        "--visual-engine-dir",
        dest="multimodal_engine_dir",
        default="",
        help="Pre-built multimodal engine directory (visual and/or audio "
        "encoders) for a prebuilt --model engine dir",
    )
    parser.add_argument(
        "--visual-onnx-dir",
        dest="visual_onnx_dir",
        default="",
        help="Pre-built visual ONNX directory for a VLM "
        "(use when --model is a prebuilt ONNX dir)",
    )
    parser.add_argument(
        "--audio-onnx-dir",
        dest="audio_onnx_dir",
        default="",
        help="Pre-built audio encoder ONNX directory "
        "(use when --model is a prebuilt ONNX dir)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=4096,
        help="Max input sequence length",
    )
    parser.add_argument("--max-batch-size",
                        type=int,
                        default=1,
                        help="Max batch size")
    parser.add_argument(
        "--max-kv-cache-capacity",
        type=int,
        default=8192,
        help="Max KV cache capacity",
    )
    parser.add_argument(
        "--spec-decode-engine-dir",
        dest="spec_decode_engine_dir",
        default="",
        help=
        "Pre-built speculative decoding engine dir (EAGLE, MTP, or DFlash)",
    )
    parser.add_argument("--draft-top-k",
                        type=int,
                        default=10,
                        help="Speculative decoding: tokens per predecessor")
    parser.add_argument("--draft-step",
                        type=int,
                        default=6,
                        help="Speculative decoding: number of draft steps")
    parser.add_argument("--verify-tree-size",
                        type=int,
                        default=60,
                        help="Speculative decoding: verification tree size")
    parser.add_argument(
        "--enable-batching",
        action="store_true",
        help=
        "Enable server-side batching for compatible non-streaming requests",
    )
    parser.add_argument(
        "--allowed-local-media-path",
        default=None,
        help="Directory the server may read local media from. Unset (default) "
        "rejects bare paths and file:// URLs over HTTP.",
    )
    parser.add_argument(
        "--batch-timeout-ms",
        type=float,
        default=10.0,
        help="Maximum wait time for compatible requests when batching",
    )
    parser.add_argument(
        "--max-queue-batch-size",
        type=int,
        default=None,
        help="Maximum HTTP requests to merge into one runtime batch",
    )
    parser.add_argument(
        "--request-queue-size",
        type=int,
        default=_DEFAULT_REQUEST_QUEUE_SIZE,
        help="Maximum concurrently admitted requests (queued + running) before "
        "the server returns backpressure (503 / Anthropic 529)",
    )
    args = parser.parse_args()

    from .engine import LLM
    from .engine_layout import classify_model_source

    # Auto-classify the --model path (prebuilt engine / prebuilt ONNX /
    # checkpoint) and route it to the matching one of LLM's three init modes.
    # Build params are ignored for the prebuilt-engine mode.
    model_arg = args.model
    onnx_dir = ""
    engine_dir = ""
    mode = classify_model_source(model_arg)
    if mode == "engine_dir":
        engine_dir, model_arg = model_arg, ""
    elif mode == "onnx_dir":
        onnx_dir, model_arg = model_arg, ""

    llm = LLM(
        model=model_arg,
        onnx_dir=onnx_dir,
        visual_onnx_dir=args.visual_onnx_dir,
        audio_onnx_dir=args.audio_onnx_dir,
        engine_dir=engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        max_input_len=args.max_input_len,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        eagle_engine_dir=args.spec_decode_engine_dir,
        draft_top_k=args.draft_top_k,
        draft_step=args.draft_step,
        verify_tree_size=args.verify_tree_size,
    )
    llm.serve(
        host=args.host,
        port=args.port,
        enable_batching=args.enable_batching,
        batch_timeout_ms=args.batch_timeout_ms,
        max_queue_batch_size=args.max_queue_batch_size,
        request_queue_size=args.request_queue_size,
        allowed_local_media_path=args.allowed_local_media_path,
    )


if __name__ == "__main__":
    main()
