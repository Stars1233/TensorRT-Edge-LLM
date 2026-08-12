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
vLLM-style one-line API for TensorRT Edge-LLM.

Pipeline: HuggingFace model ID or local path -> ONNX export -> TensorRT engine
build -> inference.

Example::

    from experimental.server import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-1.7B")
    outputs = llm.generate(
        ["What is the capital of France?"],
        SamplingParams(temperature=0.7, max_tokens=256),
    )
    for output in outputs:
        print(output.text)

    # Or start an OpenAI-compatible server:
    llm.serve(port=8000)
"""

import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

from .tool_calling import (ToolConfig, parse_assistant_output,
                           validate_tool_request)
from .tool_chat_template import (ToolChatTemplateFormatter,
                                 needs_tool_chat_template)

logger = logging.getLogger("edgellm.server")

_PLUGIN_LIB_NAME = "libNvInfer_edgellm_plugin.so"
_MAX_LOGIT_BIAS_TOKENS = 1024
_MAX_LOGIT_BIAS_TOKEN_ID = (1 << 31) - 1
_MIN_LOGIT_BIAS = -100.0
_MAX_LOGIT_BIAS = 100.0
_DEFAULT_MAX_INPUT_LEN = 4096
_DEFAULT_MAX_BATCH_SIZE = 1
_DEFAULT_MAX_KV_CACHE_CAPACITY = 8192

_LOGIT_BIAS_SPEC_DECODE_ERROR = (
    "logit_bias is not supported while speculative decoding is enabled; "
    "set disable_spec_decode=true or use a vanilla engine")


def _exporter_model_types():
    """Visual/audio classification read from the exporter, so the server cannot
    drift behind it. The sets are orthogonal: an Omni checkpoint is in both."""
    from tensorrt_edgellm.scripts import export as _export

    return _export._VLM_MODEL_TYPES, _export._AUDIO_MODEL_TYPES


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams:
    """Sampling parameters (mirrors vLLM's SamplingParams)."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    enable_thinking: bool = False
    disable_spec_decode: bool = False
    num_logprobs: int = 0
    stop: List[str] = field(default_factory=list)
    logit_bias: Dict[int, float] = field(default_factory=dict)


@dataclass
class LogprobEntry:
    """One top-K log-probability entry for a single generated token.

    ``token`` is the piece decoded as UTF-8 with ``errors="replace"`` (a
    byte-level BPE token may be only part of a multi-byte character, so it can
    contain U+FFFD); ``bytes`` carries the raw token bytes losslessly.
    """

    token_id: int
    logprob: float
    token: str
    bytes: List[int]


def _convert_logprobs(raw) -> List[List[LogprobEntry]]:
    """Convert the C++/pybind logprobs (list of list of native LogprobEntry with
    a raw-bytes ``piece``) into engine LogprobEntry dataclasses."""
    return [[
        LogprobEntry(token_id=e.token_id,
                     logprob=e.logprob,
                     token=e.piece.decode("utf-8", "replace"),
                     bytes=list(e.piece)) for e in step
    ] for step in raw]


@dataclass
class CompletionOutput:
    """Output of a single generation request."""

    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    logprobs: List[List[LogprobEntry]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class StreamDelta:
    """Single delta from a streaming generation.

    Text deltas carry ``text``/``token_ids``; audio deltas (Omni streaming)
    carry ``audio_bytes`` (int16 LE mono PCM) instead. ``finished`` marks the
    end of the text stream; generator exhaustion ends the audio stream.
    """

    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None
    logprobs: List[List[LogprobEntry]] = field(default_factory=list)
    audio_bytes: Optional[bytes] = None


@dataclass
class AudioParams:
    """Talker / vocoder knobs for one Omni audio-output request."""

    voice: str = ""
    talker_temperature: float = 0.9
    talker_top_k: int = 50
    talker_top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_audio_length: int = 4096
    codec_chunk_frames: int = 10
    talker_prefill_threshold: int = 4


#: Sample rate of Omni Code2Wav PCM output.
OMNI_AUDIO_SAMPLE_RATE = 24000


def _native_audio_params(rt, audio: "AudioParams"):
    """Convert the AudioParams dataclass to the pybind OmniAudioParams."""
    omni_params = rt.OmniAudioParams()
    omni_params.speaker_name = audio.voice
    for name, value in asdict(audio).items():
        if name != "voice":
            setattr(omni_params, name, value)
    return omni_params


def _pump_channels(rt,
                   run,
                   text_channel,
                   audio_channel,
                   sem=None,
                   admission_handoff=None):
    """Drive one generation in a worker thread, yielding StreamDeltas.

    Shared by the Omni dual-stream path (both channels) and the standalone
    TTS path (``text_channel=None``). The drain-once retry after
    is_finished()/is_cancelled() closes the race where the producer finishes
    between an empty pop and the check. ``sem``/``admission_handoff`` follow
    the generate_stream contract: the worker owns the admission gate and
    releases it when the C++ call returns.
    """
    error_holder = [None]

    def _run():
        try:
            run()
        except Exception as error:  # noqa: BLE001 - re-raised below
            error_holder[0] = error
            if text_channel is not None:
                text_channel.cancel()
            audio_channel.cancel()
        finally:
            if sem is not None:
                sem.release()
            if admission_handoff is not None:
                admission_handoff.release()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    if admission_handoff is not None:
        # The worker owns the gate now: a join timeout below must not
        # release it while the C++ call is still running.
        admission_handoff.worker_started()

    text_done = text_channel is None
    audio_done = False
    try:
        while not (text_done and audio_done):
            if not text_done:
                chunk = text_channel.wait_pop(timeout_ms=20)
                if chunk is None and (text_channel.is_finished()
                                      or text_channel.is_cancelled()):
                    chunk = text_channel.try_pop()
                    if chunk is None:
                        text_done = True
                if chunk is not None:
                    reason = finish_reason_name(
                        rt, chunk.reason) if chunk.finished else None
                    yield StreamDelta(
                        text=chunk.text,
                        token_ids=list(chunk.token_ids),
                        finished=chunk.finished,
                        finish_reason=reason,
                        logprobs=_convert_logprobs(chunk.logprobs),
                    )
                    text_done = chunk.finished

            if not audio_done:
                # Text drives pacing while it flows (non-blocking audio poll);
                # once text ends, block on audio instead.
                audio_chunk = audio_channel.wait_pop(
                    timeout_ms=100 if text_done else 0)
                if audio_chunk is None and (audio_channel.is_finished()
                                            or audio_channel.is_cancelled()):
                    audio_chunk = audio_channel.try_pop()
                    if audio_chunk is None:
                        audio_done = True
                if audio_chunk is not None:
                    if audio_chunk.pcm16:
                        yield StreamDelta(audio_bytes=audio_chunk.pcm16)
                    audio_done = audio_chunk.is_final
    finally:
        # Reached normally or via generator close (client disconnect).
        # Cancelling the text channel stops the Thinker decode loop; the
        # audio cancel stops vocoding.
        if not (text_done and audio_done):
            if text_channel is not None:
                text_channel.cancel()
            audio_channel.cancel()
        worker.join(timeout=30.0)
    if error_holder[0] is not None:
        raise error_holder[0]


def _stream_tts(rt,
                runtime,
                text: str,
                audio: "AudioParams",
                sem,
                admission_handoff=None,
                infer_guard=None) -> Generator["StreamDelta", None, None]:
    """Run one standalone TTS request; yields audio-only StreamDeltas.

    ``runtime`` is any pybind object exposing ``handle_request_tts``
    (LLMRuntime with the Omni stack loaded, or the TTS-only TTSRuntime).
    Gate ownership follows generate_stream: ``sem`` is acquired here for
    direct Python callers, while the HTTP layer instead takes it
    non-blocking and hands it over as ``admission_handoff``. ``infer_guard``
    serializes against text inference sharing the same CUDA stream (unused
    by the TTS-only runtime, which serves no text).
    """
    omni_params = _native_audio_params(rt, audio)
    audio_channel = rt.AudioStreamChannel()
    if sem is not None:
        sem.acquire()

    def _run():
        if infer_guard is None:
            runtime.handle_request_tts(text, omni_params, audio_channel)
            return
        with infer_guard:
            runtime.handle_request_tts(text, omni_params, audio_channel)

    yield from _pump_channels(rt,
                              _run,
                              None,
                              audio_channel,
                              sem=sem,
                              admission_handoff=admission_handoff)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_model_dir(model: str) -> str:
    """Resolve a HuggingFace model ID or local path to a local directory."""
    if os.path.isdir(model):
        return os.path.abspath(model)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with: "
            "pip install huggingface_hub") from exc
    logger.info("Downloading %s from Hugging Face Hub ...", model)
    return snapshot_download(model)


def _derive_model_id(model: str, onnx_dir: str, engine_dir: str) -> str:
    """Return a clean id to advertise via /v1/models and echo in responses.

    A local checkpoint/ONNX/engine path (e.g. ``--model /path/to/Qwen3-8B-FP8``)
    would otherwise leak the full filesystem path as the model id; use its
    directory name. A HuggingFace id (``Qwen/Qwen3-1.7B``) is already clean and
    is kept as-is.
    """
    src = model or onnx_dir or engine_dir or ""
    if src and (os.path.isabs(src) or os.path.isdir(src)):
        return os.path.basename(os.path.normpath(src))
    return src


def _artifacts_dir_for_model(model_dir: str) -> str:
    """Return a deterministic directory for ONNX/engine artifacts.

    Stored under ``<model_dir>/.edgellm/``.  If that is not writable
    (e.g. shared filesystem), falls back to ``~/.cache/edgellm/<hash>/``.
    """
    preferred = os.path.join(model_dir, ".edgellm")
    try:
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except OSError:
        digest = hashlib.sha256(
            os.path.abspath(model_dir).encode()).hexdigest()[:12]
        fallback = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "edgellm",
            digest,
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _engine_config_tag(
    max_input_len: int,
    max_batch_size: int,
    max_kv_cache_capacity: int,
) -> str:
    """Return a short tag encoding the engine build config."""
    return f"i{max_input_len}_b{max_batch_size}_kv{max_kv_cache_capacity}"


def _is_multimodal(model_dir: str) -> bool:
    """Visual-encoder model_type in config.json (audio model types are
    detected separately via the exporter audio set)."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type", "") in _exporter_model_types()[0]


def _read_model_type(model_dir: str) -> str:
    """Read model_type from config.json."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return ""
    with open(cfg_path) as f:
        return json.load(f).get("model_type", "")


def _read_vision_config(model_dir: str) -> dict:
    """Read vision_config from config.json for visual builder params."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return json.load(f).get("vision_config", {})


def _read_engine_builder_config(engine_dir: str) -> dict:
    """Read builder_config from an engine directory."""
    cfg_path = os.path.join(engine_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return json.load(f).get("builder_config", {})


def _ensure_plugin_path() -> None:
    """Set EDGELLM_PLUGIN_PATH if not already set.

    Searches relative to this package and common build locations.
    """
    if os.environ.get("EDGELLM_PLUGIN_PATH"):
        return
    project_root = Path(__file__).resolve().parent.parent.parent
    search_dirs = [
        project_root / "build" / "core",
        project_root / "build" / "lib",
    ]
    for d in search_dirs:
        candidate = d / _PLUGIN_LIB_NAME
        if candidate.is_file():
            os.environ["EDGELLM_PLUGIN_PATH"] = str(candidate)
            return


def _import_runtime():
    """Import the C++ pybind module."""
    _ensure_plugin_path()
    try:
        from tensorrt_edgellm import _edgellm_runtime as _rt
        return _rt
    except ImportError:
        pass
    project_root = Path(__file__).resolve().parent.parent.parent
    search_dirs = []
    if os.environ.get("EDGELLM_PYBIND_DIR"):
        search_dirs.append(Path(os.environ["EDGELLM_PYBIND_DIR"]))
    if os.environ.get("BUILD_DIR"):
        search_dirs.append(Path(os.environ["BUILD_DIR"]) / "pybind")
    search_dirs.extend([
        project_root / "experimental" / "pybind" / "build",
        project_root / "build" / "pybind",
    ])
    search_dirs.extend(project_root.glob("build/lib.*"))
    for cand_dir in search_dirs:
        if not cand_dir.is_dir():
            continue
        so_files = list(cand_dir.glob("*_edgellm_runtime*.so"))
        if so_files:
            spec = importlib.util.spec_from_file_location(
                "_edgellm_runtime", so_files[0])
            mod = importlib.util.module_from_spec(spec)
            sys.modules["tensorrt_edgellm._edgellm_runtime"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "Could not import _edgellm_runtime. Build the C++ extension first:\n"
        "  TRT_PACKAGE_DIR=/path/to/tensorrt python experimental/server/setup_pybind.py build_ext --inplace"
    )


def _ensure_export_package() -> None:
    """Ensure the installed checkpoint export package is importable."""
    try:
        import tensorrt_edgellm  # noqa: F401
        return
    except ImportError:
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)


def _normalize_logit_bias(
        logit_bias: Optional[Dict[Any, Any]]) -> Dict[int, float]:
    """Validate and normalize an OpenAI-compatible logit_bias map."""
    if logit_bias is None:
        return {}
    if not isinstance(logit_bias, dict):
        raise ValueError(
            "'logit_bias' must be an object mapping token IDs to bias values")
    if len(logit_bias) > _MAX_LOGIT_BIAS_TOKENS:
        raise ValueError(f"'logit_bias' has {len(logit_bias)} entries; max is "
                         f"{_MAX_LOGIT_BIAS_TOKENS}")

    normalized: Dict[int, float] = {}
    for token, bias in logit_bias.items():
        if isinstance(token, bool):
            raise ValueError(
                f"'logit_bias' token ID {token!r} is not an integer")
        if isinstance(token, int):
            token_id = token
        elif isinstance(token, str):
            try:
                token_id = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"'logit_bias' token ID {token!r} is not an integer"
                ) from exc
        else:
            raise ValueError(
                f"'logit_bias' token ID {token!r} is not an integer")
        if token_id < 0 or token_id > _MAX_LOGIT_BIAS_TOKEN_ID:
            raise ValueError(
                f"'logit_bias' token ID must be in "
                f"[0, {_MAX_LOGIT_BIAS_TOKEN_ID}], got {token_id}")
        if isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be a number")
        try:
            bias_value = float(bias)
        except OverflowError as exc:
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be in "
                f"[{_MIN_LOGIT_BIAS}, {_MAX_LOGIT_BIAS}], got {bias}") from exc
        if (not math.isfinite(bias_value) or bias_value < _MIN_LOGIT_BIAS
                or bias_value > _MAX_LOGIT_BIAS):
            raise ValueError(
                f"'logit_bias' value for token ID {token_id} must be in "
                f"[{_MIN_LOGIT_BIAS}, {_MAX_LOGIT_BIAS}], got {bias_value}")
        normalized[token_id] = bias_value
    return normalized


def _validate_logit_bias_spec_decode(logit_bias: Dict[int, float], *,
                                     disable_spec_decode: bool,
                                     has_draft_model: bool) -> None:
    """Reject logit bias unless speculative decoding is absent or explicitly disabled."""
    if logit_bias and has_draft_model and not disable_spec_decode:
        raise ValueError(_LOGIT_BIAS_SPEC_DECODE_ERROR)


def _engine_config_value(builder_config: dict, field_name: str,
                         requested_value: int) -> int:
    if field_name not in builder_config:
        return requested_value

    engine_value = int(builder_config[field_name])
    if engine_value != requested_value:
        logger.warning(
            "Using %s=%d from engine builder_config instead of requested %d",
            field_name,
            engine_value,
            requested_value,
        )
    return engine_value


# ---------------------------------------------------------------------------
# LLM class
# ---------------------------------------------------------------------------

_STREAM_JOIN_TIMEOUT_S = 5.0


class LLM:
    """vLLM-style entry point for TensorRT Edge-LLM inference.

    Three initialization modes (exactly one of ``model``, ``onnx_dir``,
    or ``engine_dir`` must be provided):

    1. **HuggingFace checkpoint** — exports ONNX, builds engine, loads::

           llm = LLM(model="Qwen/Qwen3-1.7B")

    2. **ONNX directory** — builds engine from ONNX, loads::

           llm = LLM(onnx_dir="/path/to/onnx")

    3. **Pre-built engine** — loads directly::

           llm = LLM(engine_dir="/path/to/engine")
           llm = LLM(engine_dir="...", multimodal_engine_dir="...")

    See :mod:`experimental.server.engine_layout` for the expected
    directory layouts.
    """

    #: Distinguishes full LLM servers from TTS-only ones in the API layer.
    text_capable = True

    def __init__(
        self,
        model: str = "",
        *,
        onnx_dir: str = "",
        visual_onnx_dir: str = "",
        audio_onnx_dir: str = "",
        engine_dir: str = "",
        multimodal_engine_dir: str = "",
        visual_engine_dir: str = "",
        max_input_len: int = _DEFAULT_MAX_INPUT_LEN,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        max_kv_cache_capacity: int = _DEFAULT_MAX_KV_CACHE_CAPACITY,
        eagle_engine_dir: str = "",
        draft_top_k: int = 10,
        draft_step: int = 6,
        verify_tree_size: int = 60,
        talker_engine_dir: str = "",
        code_predictor_engine_dir: str = "",
        code2wav_engine_dir: str = "",
    ):
        sources = sum(bool(s) for s in (model, onnx_dir, engine_dir))
        if sources != 1:
            raise ValueError(
                "Exactly one of 'model', 'onnx_dir', or 'engine_dir' "
                "must be provided.")
        if visual_onnx_dir and not onnx_dir:
            raise ValueError(
                "'visual_onnx_dir' is only supported with 'onnx_dir'; "
                "use 'visual_engine_dir' with 'engine_dir'.")
        if audio_onnx_dir and not onnx_dir:
            raise ValueError(
                "'audio_onnx_dir' is only supported with 'onnx_dir'; "
                "use 'multimodal_engine_dir' with 'engine_dir'.")
        if visual_engine_dir and not engine_dir:
            raise ValueError(
                "'visual_engine_dir' is only supported with 'engine_dir'.")

        # `visual_engine_dir` is the deprecated alias for `multimodal_engine_dir`
        # (the encoder slot now also serves audio).
        multimodal_engine_dir = multimodal_engine_dir or visual_engine_dir

        self._model_id = _derive_model_id(model, onnx_dir, engine_dir)
        self._omni_capable = False
        self._talker_engine_dir = talker_engine_dir
        self._code_predictor_engine_dir = code_predictor_engine_dir
        self._code2wav_engine_dir = code2wav_engine_dir
        self._eagle_engine_dir = eagle_engine_dir
        self._draft_top_k = draft_top_k
        self._draft_step = draft_step
        self._verify_tree_size = verify_tree_size
        self._max_input_len = max_input_len
        self._max_batch_size = max_batch_size
        self._max_kv_cache_capacity = max_kv_cache_capacity
        self._tool_template_formatter: Optional[
            ToolChatTemplateFormatter] = None

        if engine_dir:
            self._init_from_engine(engine_dir, multimodal_engine_dir)
        elif onnx_dir:
            self._init_from_onnx(
                onnx_dir,
                visual_onnx_dir=visual_onnx_dir,
                audio_onnx_dir=audio_onnx_dir,
                multimodal_engine_dir=multimodal_engine_dir,
                max_input_len=max_input_len,
                max_batch_size=max_batch_size,
                max_kv_cache_capacity=max_kv_cache_capacity,
            )
        else:
            self._init_from_model(
                model,
                multimodal_engine_dir=multimodal_engine_dir,
                max_input_len=max_input_len,
                max_batch_size=max_batch_size,
                max_kv_cache_capacity=max_kv_cache_capacity,
            )

        self._load_runtime()

    # ------------------------------------------------------------------
    # Initialization paths
    # ------------------------------------------------------------------

    def _init_from_engine(self, engine_dir: str,
                          multimodal_engine_dir: str) -> None:
        """Load from pre-built engine directories (no export, no build)."""
        from .engine_layout import (EngineType, detect_engine_type,
                                    find_multimodal_engine_dir,
                                    validate_llm_engine_dir,
                                    validate_multimodal_engine_dir,
                                    validate_spec_decode_engine_dir)

        engine_type = detect_engine_type(engine_dir)
        if engine_type == EngineType.SPEC_DECODE:
            if not validate_spec_decode_engine_dir(engine_dir):
                raise ValueError(
                    f"spec_base.engine/spec_draft.engine not found in: {engine_dir}"
                )
            if multimodal_engine_dir:
                logger.warning(
                    "multimodal_engine_dir=%r is ignored for spec-decode engines",
                    multimodal_engine_dir)
            # Spec-decode dir (spec_base.engine + spec_draft.engine): route through
            # the spec-decode path by promoting engine_dir to eagle_engine_dir.
            self._engine_dir = engine_dir
            self._model_dir = engine_dir
            self._eagle_engine_dir = engine_dir
            self._multimodal_engine_dir = ""
            self._is_multimodal = False
            logger.info("Using pre-built spec-decode engine: %s",
                        self._engine_dir)
            return
        if not validate_llm_engine_dir(engine_dir):
            raise ValueError(f"llm.engine not found in: {engine_dir}")
        self._engine_dir = engine_dir
        self._model_dir = engine_dir
        self._is_multimodal = False
        builder_config = _read_engine_builder_config(engine_dir)
        self._max_input_len = _engine_config_value(
            builder_config, "max_input_len",
            getattr(self, "_max_input_len", _DEFAULT_MAX_INPUT_LEN))
        self._max_batch_size = _engine_config_value(
            builder_config, "max_batch_size",
            getattr(self, "_max_batch_size", _DEFAULT_MAX_BATCH_SIZE))
        self._max_kv_cache_capacity = _engine_config_value(
            builder_config, "max_kv_cache_capacity",
            getattr(self, "_max_kv_cache_capacity",
                    _DEFAULT_MAX_KV_CACHE_CAPACITY))

        if multimodal_engine_dir:
            if not validate_multimodal_engine_dir(multimodal_engine_dir):
                raise ValueError(f"no visual or audio encoder engine "
                                 f"found in: {multimodal_engine_dir}")
            self._multimodal_engine_dir = multimodal_engine_dir
            self._is_multimodal = True
        else:
            auto = find_multimodal_engine_dir(engine_dir)
            self._multimodal_engine_dir = auto or ""
            if auto:
                self._is_multimodal = True
                logger.info("Auto-detected visual engine: %s", auto)

        logger.info("Using pre-built engine: %s", self._engine_dir)

    def _init_from_onnx(
        self,
        onnx_dir: str,
        *,
        visual_onnx_dir: str,
        audio_onnx_dir: str = "",
        multimodal_engine_dir: str = "",
        max_input_len: int,
        max_batch_size: int,
        max_kv_cache_capacity: int,
    ) -> None:
        """Build engine from ONNX directories (no export)."""
        from .engine_layout import validate_multimodal_engine_dir
        self._max_input_len = max_input_len
        self._max_batch_size = max_batch_size
        self._max_kv_cache_capacity = max_kv_cache_capacity
        self._onnx_dir = onnx_dir
        self._visual_onnx_dir = visual_onnx_dir
        self._audio_onnx_dir = audio_onnx_dir
        self._model_dir = onnx_dir
        self._is_multimodal = bool(visual_onnx_dir or audio_onnx_dir
                                   or multimodal_engine_dir)

        # Validated up front: a typo must not surface only after the LLM engine
        # build, which can take tens of minutes.
        if multimodal_engine_dir and not validate_multimodal_engine_dir(
                multimodal_engine_dir):
            raise ValueError(f"no visual or audio encoder engine "
                             f"found in: {multimodal_engine_dir}")

        cfg_tag = _engine_config_tag(max_input_len, max_batch_size,
                                     max_kv_cache_capacity)
        artifacts = _artifacts_dir_for_model(onnx_dir)

        spec_decode_engine_dir = self._eagle_engine_dir
        self._engine_dir = (spec_decode_engine_dir
                            if spec_decode_engine_dir else os.path.join(
                                artifacts, "engine", cfg_tag, "llm"))
        if not spec_decode_engine_dir and not os.path.exists(
                os.path.join(self._engine_dir, "llm.engine")):
            self._build_engine()
        else:
            logger.info("Using cached engine: %s", self._engine_dir)

        self._multimodal_engine_dir = ""
        if multimodal_engine_dir:
            # A user-supplied prebuilt encoder wins over auto-built artifacts.
            self._multimodal_engine_dir = multimodal_engine_dir
        elif visual_onnx_dir or audio_onnx_dir:
            # One shared root: the C++ runtime reads visual.engine from it and
            # the audio encoder from its audio/ subdirectory.
            self._multimodal_engine_dir = os.path.join(artifacts, "engine",
                                                       cfg_tag, "multimodal")
            if visual_onnx_dir:
                if not os.path.exists(
                        os.path.join(self._multimodal_engine_dir,
                                     "visual.engine")):
                    self._build_visual_engine()
                else:
                    logger.info("Using cached visual engine: %s",
                                self._multimodal_engine_dir)
            if audio_onnx_dir:
                if not os.path.exists(
                        os.path.join(self._multimodal_engine_dir, "audio",
                                     "audio_encoder.engine")):
                    self._build_audio_engine()
                else:
                    logger.info("Using cached audio engine: %s",
                                self._multimodal_engine_dir)

    def _init_from_model(
        self,
        model: str,
        *,
        multimodal_engine_dir: str = "",
        max_input_len: int,
        max_batch_size: int,
        max_kv_cache_capacity: int,
    ) -> None:
        """Export ONNX + build engine from HuggingFace checkpoint."""
        from .engine_layout import validate_multimodal_engine_dir

        # Validated before the LLM ONNX export below, which can take tens of
        # minutes; _init_from_onnx re-checks for its own direct callers.
        if multimodal_engine_dir and not validate_multimodal_engine_dir(
                multimodal_engine_dir):
            raise ValueError(f"no visual or audio encoder engine "
                             f"found in: {multimodal_engine_dir}")

        self._max_input_len = max_input_len
        self._max_batch_size = max_batch_size
        self._max_kv_cache_capacity = max_kv_cache_capacity

        logger.info("Resolving model: %s", model)
        self._model_dir = _resolve_model_dir(model)
        artifacts = _artifacts_dir_for_model(self._model_dir)
        self._is_multimodal = _is_multimodal(self._model_dir)
        self._model_type = _read_model_type(self._model_dir)
        self._is_audio_model = self._model_type in _exporter_model_types()[1]
        if self._is_multimodal:
            logger.info("Detected VLM model (type=%s)", self._model_type)
        elif self._is_audio_model:
            logger.info("Detected audio model (type=%s)", self._model_type)

        self._onnx_dir = os.path.join(artifacts, "onnx", "llm")
        if not os.path.exists(os.path.join(self._onnx_dir, "model.onnx")):
            self._export_onnx()
        else:
            logger.info("Using cached ONNX: %s", self._onnx_dir)
            self._patch_multimodal_token_ids()

        # A prebuilt multimodal engine dir wins, so exporting encoder ONNX would
        # only be discarded by _init_from_onnx.
        self._visual_onnx_dir = ""
        if self._is_multimodal and not multimodal_engine_dir:
            self._visual_onnx_dir = os.path.join(artifacts, "onnx", "visual")
            if not os.path.exists(
                    os.path.join(self._visual_onnx_dir, "model.onnx")):
                self._export_visual_onnx()
            else:
                logger.info("Using cached visual ONNX: %s",
                            self._visual_onnx_dir)

        self._audio_onnx_dir = ""
        if self._is_audio_model and not multimodal_engine_dir:
            self._audio_onnx_dir = os.path.join(artifacts, "onnx", "audio")
            if not os.path.exists(
                    os.path.join(self._audio_onnx_dir, "model.onnx")):
                self._export_audio_onnx()
            else:
                logger.info("Using cached audio ONNX: %s",
                            self._audio_onnx_dir)

        # Delegate to _init_from_onnx for the build step
        self._init_from_onnx(
            self._onnx_dir,
            visual_onnx_dir=self._visual_onnx_dir,
            audio_onnx_dir=self._audio_onnx_dir,
            multimodal_engine_dir=multimodal_engine_dir,
            max_input_len=max_input_len,
            max_batch_size=max_batch_size,
            max_kv_cache_capacity=max_kv_cache_capacity,
        )

    def _load_runtime(self) -> None:
        """Load the C++ runtime from engine directories."""
        self._rt = _import_runtime()
        logger.info("Loading TensorRT engine from %s ...", self._engine_dir)
        if self._multimodal_engine_dir:
            logger.info("Loading visual engine from %s ...",
                        self._multimodal_engine_dir)
        spec_decode_engine_dir = self._eagle_engine_dir
        if spec_decode_engine_dir:
            logger.info(
                "Speculative decoding enabled (top_k=%d, step=%d, tree=%d)",
                self._draft_top_k,
                self._draft_step,
                self._verify_tree_size,
            )
            self._runtime = self._rt.LLMRuntime(
                self._engine_dir,
                self._multimodal_engine_dir,
                {},
                self._draft_top_k,
                self._draft_step,
                self._verify_tree_size,
            )
        else:
            self._runtime = self._rt.LLMRuntime(
                self._engine_dir,
                self._multimodal_engine_dir,
                {},
            )
        self._runtime.capture_decoding_cuda_graph()
        self._load_omni_runtime()
        logger.info("Engine loaded and ready.")

    def _load_omni_runtime(self) -> None:
        """Load the Qwen3-Omni audio-output stack when its engines exist."""
        from .engine_layout import find_omni_engine_dirs

        dirs = {
            "talker": self._talker_engine_dir,
            "code_predictor": self._code_predictor_engine_dir,
            "code2wav": self._code2wav_engine_dir,
        }
        explicit = any(dirs.values())
        if not all(dirs.values()):
            auto = find_omni_engine_dirs(self._engine_dir) or {}
            dirs = {k: v or auto.get(k, "") for k, v in dirs.items()}
            if not all(dirs.values()):
                if explicit:
                    raise ValueError(
                        "Omni engine dirs partially specified and the rest "
                        f"could not be auto-detected: {dirs}")
                return
            logger.info("Auto-detected Omni engines: talker=%s",
                        dirs["talker"])

        self._runtime.load_omni(dirs["talker"], dirs["code_predictor"],
                                dirs["code2wav"], self._engine_dir)
        self._omni_capable = True
        logger.info("Omni audio output ready.")

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _export_onnx(self) -> None:
        """Export the model checkpoint to ONNX via tensorrt_edgellm."""
        logger.info("Exporting ONNX to %s ...", self._onnx_dir)
        os.makedirs(self._onnx_dir, exist_ok=True)

        _ensure_export_package()
        from tensorrt_edgellm import AutoModel, export_onnx

        model = AutoModel.from_pretrained(self._model_dir, device="cpu")
        output_path = os.path.join(self._onnx_dir, "model.onnx")
        export_onnx(model, output_path, model_dir=self._model_dir)

        self._patch_multimodal_token_ids()
        logger.info("ONNX export complete: %s", output_path)

    def _patch_multimodal_token_ids(self) -> None:
        """Write the media placeholder ids into the LLM config. Idempotent, and
        re-run on a cached ONNX: a config predating the encoder carries no id,
        which the runtime reads as -1 and silently drops the embeddings."""
        if self._is_multimodal:
            _ensure_export_package()
            from tensorrt_edgellm.scripts.export import _find_token_id
            image_token_id = _find_token_id(self._model_dir, "<|image_pad|>")
            cfg_path = os.path.join(self._onnx_dir, "config.json")
            if image_token_id is not None and os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if cfg.get("image_token_id") != image_token_id:
                    cfg["image_token_id"] = image_token_id
                    with open(cfg_path, "w") as f:
                        json.dump(cfg, f, indent=2)
                    logger.info("Patched image_token_id=%d into LLM config",
                                image_token_id)

        if getattr(self, "_is_audio_model", False):
            _ensure_export_package()
            from tensorrt_edgellm.scripts.export import \
                _patch_multimodal_token_ids
            _patch_multimodal_token_ids(self._model_dir, self._onnx_dir,
                                        self._model_type)

    def _export_visual_onnx(self) -> None:
        """Export the visual encoder to ONNX via tensorrt_edgellm."""
        logger.info(
            "Exporting visual ONNX to %s ...",
            self._visual_onnx_dir,
        )
        os.makedirs(self._visual_onnx_dir, exist_ok=True)

        import torch

        _ensure_export_package()
        from tensorrt_edgellm.model import load_model_config
        from tensorrt_edgellm.scripts.export import (_export_visual,
                                                     _load_all_weights,
                                                     _load_config)

        config = _load_config(self._model_dir)
        weights = _load_all_weights(self._model_dir)
        _export_visual(
            self._model_dir,
            self._visual_onnx_dir,
            weights,
            config,
            self._model_type,
            torch.float16,
            load_model_config(self._model_dir),
        )
        logger.info(
            "Visual ONNX export complete: %s",
            self._visual_onnx_dir,
        )

    def _export_audio_onnx(self) -> None:
        """Export the audio encoder to ONNX via tensorrt_edgellm."""
        logger.info(
            "Exporting audio ONNX to %s ...",
            self._audio_onnx_dir,
        )
        os.makedirs(self._audio_onnx_dir, exist_ok=True)

        import torch

        _ensure_export_package()
        from tensorrt_edgellm.model import load_model_config
        from tensorrt_edgellm.scripts.export import (_export_audio,
                                                     _load_all_weights,
                                                     _load_config)

        config = _load_config(self._model_dir)
        weights = _load_all_weights(self._model_dir)
        # Passing the ModelConfig lets _export_audio subset it to the audio
        # tower, so an NVFP4 backbone still exports an FP16 encoder.
        _export_audio(
            self._model_dir,
            self._audio_onnx_dir,
            weights,
            config,
            self._model_type,
            torch.float16,
            load_model_config(self._model_dir),
        )
        logger.info(
            "Audio ONNX export complete: %s",
            self._audio_onnx_dir,
        )

    def _build_engine(self) -> None:
        """Build a TensorRT engine from the ONNX directory."""
        logger.info(
            "Building TensorRT engine: %s -> %s",
            self._onnx_dir,
            self._engine_dir,
        )
        os.makedirs(self._engine_dir, exist_ok=True)

        rt = _import_runtime()
        config = rt.LLMBuilderConfig()
        config.max_input_len = self._max_input_len
        config.max_batch_size = self._max_batch_size
        config.max_kv_cache_capacity = self._max_kv_cache_capacity

        builder = rt.LLMBuilder(self._onnx_dir, self._engine_dir, config)
        if not builder.build():
            raise RuntimeError(
                f"TensorRT engine build failed. "
                f"ONNX dir: {self._onnx_dir}, engine dir: {self._engine_dir}")
        logger.info("Engine build complete: %s", self._engine_dir)

    def _build_visual_engine(self) -> None:
        """Build a TensorRT engine for the visual encoder."""
        logger.info(
            "Building visual TensorRT engine: %s -> %s",
            self._visual_onnx_dir,
            self._multimodal_engine_dir,
        )
        os.makedirs(self._multimodal_engine_dir, exist_ok=True)

        rt = _import_runtime()
        config = rt.VisualBuilderConfig()

        # Derive image token counts from vision_config
        vis_cfg = _read_vision_config(self._model_dir)
        image_size = vis_cfg.get("image_size", 448)
        patch_size = vis_cfg.get("patch_size", 14)
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
        tokens_per_tile = (image_size // patch_size)**2
        # Round up to nearest multiple of tokens_per_tile
        config.min_image_tokens = tokens_per_tile
        config.max_image_tokens = tokens_per_tile * 4
        config.max_image_tokens_per_image = tokens_per_tile * 2

        builder = rt.VisualBuilder(
            self._visual_onnx_dir,
            self._multimodal_engine_dir,
            config,
        )
        if not builder.build():
            raise RuntimeError(f"Visual TensorRT engine build failed. "
                               f"ONNX dir: {self._visual_onnx_dir}, "
                               f"engine dir: {self._multimodal_engine_dir}")
        logger.info(
            "Visual engine build complete: %s",
            self._multimodal_engine_dir,
        )

    def _build_audio_engine(self) -> None:
        """Build a TensorRT engine for the audio encoder.

        The engine and its config.json land in the audio/ subdirectory the
        C++ runtime expects under the multimodal engine dir.
        """
        audio_engine_dir = os.path.join(self._multimodal_engine_dir, "audio")
        logger.info(
            "Building audio TensorRT engine: %s -> %s",
            self._audio_onnx_dir,
            audio_engine_dir,
        )
        os.makedirs(audio_engine_dir, exist_ok=True)

        rt = _import_runtime()
        config = rt.AudioBuilderConfig()
        builder = rt.AudioBuilder(
            self._audio_onnx_dir,
            self._multimodal_engine_dir,  # AudioBuilder appends audio/ itself
            config,
        )
        if not builder.build():
            raise RuntimeError(f"Audio TensorRT engine build failed. "
                               f"ONNX dir: {self._audio_onnx_dir}, "
                               f"engine dir: {audio_engine_dir}")
        logger.info(
            "Audio engine build complete: %s",
            self._multimodal_engine_dir,
        )

    def _tool_template_dirs(self) -> List[str]:
        dirs = [self._model_dir, self._engine_dir]
        if hasattr(self, "_onnx_dir"):
            dirs.append(self._onnx_dir)
        return dirs

    def _get_tool_template_formatter(self) -> ToolChatTemplateFormatter:
        if self._tool_template_formatter is None:
            self._tool_template_formatter = ToolChatTemplateFormatter(
                self._tool_template_dirs())
        return self._tool_template_formatter

    def _tool_choice_for_template(
            self, tool_config: ToolConfig) -> Union[str, Dict[str, Any]]:
        if tool_config.forced_name:
            return {
                "type": "function",
                "function": {
                    "name": tool_config.forced_name
                },
            }
        return tool_config.tool_choice

    def _visual_config(self) -> dict:
        """The visual engine's config.json, read once. The C++ runtime prefers
        the nested <root>/visual/ layout over legacy flat, so read in the same
        order. Empty dict when unavailable."""
        cached = getattr(self, "_visual_config_cache", None)
        if cached is not None:
            return cached
        cfg: dict = {}
        root = getattr(self, "_multimodal_engine_dir", "") or ""
        for cfg_path in (os.path.join(root, "visual", "config.json"),
                         os.path.join(root, "config.json")):
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                except (OSError, ValueError):
                    cfg = {}
                if cfg.get("model_type"):
                    break
        self._visual_config_cache = cfg
        return cfg

    def _video_model_family(self) -> str:
        """Frame-sampling family ("qwen" / "internvl" / "nemotron") from the
        visual engine's model_type. Types without a video path (phi4mm, ...) are
        rejected: their runners read only the first frame."""
        cached = getattr(self, "_video_family_cache", None)
        if cached is not None:
            return cached
        model_type = self._visual_config().get("model_type", "")
        qwen_video_types = ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5",
                            "qwen3_omni")
        # Audio-side model types have no video path (qwen3_omni_audio_encoder,
        # qwen3_omni_code2wav, qwen3_asr*); the omni ones share the qwen3_omni
        # prefix, so exclude before the prefix match.
        is_audio_type = any(tag in model_type
                            for tag in ("audio", "code2wav", "asr"))
        root = getattr(self, "_multimodal_engine_dir", "") or ""
        has_visual = (os.path.isdir(os.path.join(root, "visual"))
                      or os.path.isfile(os.path.join(root, "visual.engine")))
        if "internvl" in model_type and has_visual:
            family = "internvl"
        elif "nemotron" in model_type and not is_audio_type and has_visual:
            family = "nemotron"
        elif (model_type.startswith(qwen_video_types) and not is_audio_type
              and has_visual):
            family = "qwen"
        else:
            # Covers audio-only engines (audio/ but no visual/) and model
            # types whose runners have no video path (phi4mm, gemma, ...).
            raise ValueError(
                f"video input is not supported for model_type={model_type!r}"
                " on this multimodal engine; supported families: Qwen-VL "
                "(qwen2_vl/qwen2_5_vl/qwen3_vl/qwen3_5/qwen3_omni), InternVL, "
                "and Nemotron-Omni")
        self._video_family_cache = family
        return family

    def _video_frame_limits(self) -> dict:
        """Engine-profile inputs for frame-count clamping (see video_sampling):
        builder token bounds from the visual config.json + patch geometry from
        preprocessor_config.json. Empty dict when unavailable (no clamping)."""
        cached = getattr(self, "_video_limits_cache", None)
        if cached is not None:
            return cached
        limits: dict = {}
        cfg = self._visual_config()
        builder = cfg.get("builder_config") or {}
        root = getattr(self, "_multimodal_engine_dir", "") or ""
        pre: dict = {}
        for pre_path in (os.path.join(root, "visual",
                                      "preprocessor_config.json"),
                         os.path.join(root, "preprocessor_config.json")):
            if os.path.isfile(pre_path):
                try:
                    with open(pre_path) as f:
                        pre = json.load(f)
                except (OSError, ValueError):
                    pre = {}
                break
        pre = pre.get("image_processor", pre)
        if builder.get("max_image_tokens"):
            limits = {
                "model_type":
                cfg.get("model_type", ""),
                "min_image_tokens":
                int(builder.get("min_image_tokens", 1)),
                "max_image_tokens":
                int(builder["max_image_tokens"]),
                "max_image_tokens_per_image":
                int(builder.get("max_image_tokens_per_image", 0)),
                "max_cu_seqlen_groups":
                int(builder.get("max_cu_seqlen_groups", 0)),
                "patch_size":
                int(pre.get("patch_size", 0)),
                "merge_size":
                int(pre.get("merge_size", 0)),
                "temporal_patch_size":
                int(pre.get("temporal_patch_size", 2)),
                # Nemotron-Omni video geometry (top-level visual config.json).
                "video_pruning_rate":
                float(cfg.get("video_pruning_rate", 0.0)),
                "video_temporal_patch_size":
                int(cfg.get("video_temporal_patch_size", 2)),
                "video_target_num_patches":
                int(cfg.get("video_target_num_patches", 1024)),
                "downsample_ratio":
                float(cfg.get("downsample_ratio", 0.5)),
            }
        self._video_limits_cache = limits
        return limits

    def _prepare_messages_for_runtime(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        enable_thinking: bool = False,
    ):
        """Prepare messages for the C++ runtime."""
        tool_config = tool_config or validate_tool_request(
            messages, tools, tool_choice)
        template_tools = (tool_config.tools
                          if tool_config.tool_choice != "none" else [])
        image_buffers = _load_image_buffers(self._rt, messages,
                                            self._video_model_family,
                                            self._video_frame_limits)

        if needs_tool_chat_template(messages, template_tools,
                                    tool_config.tool_choice):
            template_tool_choice = None
            if tool_config.tool_choice != "none":
                template_tool_choice = self._tool_choice_for_template(
                    tool_config)
            prompt = self._get_tool_template_formatter().format(
                messages,
                tools=template_tools,
                tool_choice=template_tool_choice,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            cpp_messages = _convert_messages_to_cpp(
                self._rt,
                [{
                    "role": "user",
                    "content": prompt,
                }],
            )
            return cpp_messages, image_buffers, False, False

        cpp_messages = _convert_messages_to_cpp(self._rt, messages)
        return cpp_messages, image_buffers, True, True

    def count_prompt_tokens(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        enable_thinking: bool = False,
    ) -> Optional[int]:
        """Best-effort prompt token count via the HF tokenizer: exact for
        tool-templated requests, within a few tokens for plain ones (HF vs
        C++ template). Multimodal placeholders are counted once, not expanded,
        so multimodal prompts are undercounted. None when counting is
        unavailable."""
        try:
            tool_config = tool_config or validate_tool_request(
                messages, tools, tool_choice)
            template_tools = (tool_config.tools
                              if tool_config.tool_choice != "none" else [])
            template_tool_choice = None
            if template_tools and tool_config.tool_choice != "none":
                template_tool_choice = self._tool_choice_for_template(
                    tool_config)
            formatter = self._get_tool_template_formatter()
            prompt = formatter.format(
                messages,
                tools=template_tools,
                tool_choice=template_tool_choice,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            return formatter.count_tokens(prompt)
        except Exception:
            logger.debug("Prompt token counting unavailable", exc_info=True)
            return None

    def _make_generation_request(
        self,
        messages: List[Dict[str, Any]],
        params: SamplingParams,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        stream_channel: Optional[Any] = None,
    ):
        normalized_logit_bias = _normalize_logit_bias(params.logit_bias)
        _validate_logit_bias_spec_decode(
            normalized_logit_bias,
            disable_spec_decode=params.disable_spec_decode,
            has_draft_model=self.has_draft_model,
        )

        tool_config = tool_config or validate_tool_request(
            messages, tools, tool_choice)
        cpp_messages, image_buffers, apply_template, add_prompt = (
            self._prepare_messages_for_runtime(
                messages,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice,
                tool_config=tool_config,
                enable_thinking=params.enable_thinking,
            ))

        audio_buffers = _load_audio_buffers(self._rt, messages)

        request = self._rt.LLMGenerationRequest()
        req = self._rt.Request(messages=cpp_messages)
        req.image_buffers = image_buffers
        req.audio_buffers = audio_buffers
        req.stop_strings = params.stop
        req.logit_bias = normalized_logit_bias
        request.requests = [req]
        if stream_channel is not None:
            request.stream_channels = [stream_channel]
        request.temperature = params.temperature
        request.top_p = params.top_p
        request.top_k = params.top_k
        request.max_generate_length = params.max_tokens
        request.apply_chat_template = apply_template
        request.add_generation_prompt = add_prompt
        request.enable_thinking = params.enable_thinking
        request.disable_spec_decode = params.disable_spec_decode
        request.num_logprobs = params.num_logprobs
        return request

    def _parse_generation_output(
        self,
        text: str,
        token_ids: List[int],
        finish_reason: Optional[str],
        tool_config: ToolConfig,
    ) -> CompletionOutput:
        if not tool_config.parse_output:
            return CompletionOutput(text=text,
                                    token_ids=token_ids,
                                    finish_reason=finish_reason)

        parsed = parse_assistant_output(text, tool_config, self._model_dir)
        tool_calls = [call.to_openai() for call in parsed.tool_calls]
        return CompletionOutput(
            text=parsed.content,
            token_ids=token_ids,
            finish_reason="tool_calls" if tool_calls else finish_reason,
            tool_calls=tool_calls,
            reasoning=parsed.reasoning or None,
        )

    # ------------------------------------------------------------------
    # Inference API (vLLM-style)
    # ------------------------------------------------------------------

    _infer_lock_guard = threading.Lock()

    def _admission(self):
        """Per-instance gate from media decode through inference completion:
        queued requests must not each pin decoded frames. Semaphore, not Lock --
        streaming releases from the worker/SSE side."""
        sem = self.__dict__.get("_admission_sem")
        if sem is None:
            with LLM._infer_lock_guard:
                sem = self.__dict__.setdefault("_admission_sem",
                                               threading.Semaphore(1))
        return sem

    def _infer_guard(self):
        """Lock serializing every entry into the C++ runtime.

        The batcher runs on its own worker and takes only this lock, never
        the admission semaphore, so the audio paths — which call the runtime
        directly rather than through _handle_request — must take it too.
        """
        lock = self.__dict__.get("_infer_lock")
        if lock is None:
            with LLM._infer_lock_guard:
                lock = self.__dict__.setdefault("_infer_lock",
                                                threading.Lock())
        return lock

    def _handle_request(self, request):
        """Serialized entry to the C++ runtime."""
        # Called unbound on duck-typed objects too, so reach the guard
        # through the class rather than the instance.
        with LLM._infer_guard(self):
            return self._runtime.handle_request(request)

    def generate(
        self,
        prompts: Union[str, List[str], List[List[Dict[str, Any]]]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> List[CompletionOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: A single prompt string, a list of prompt strings, or
                a list of OpenAI-style message lists.
            sampling_params: Sampling configuration. Defaults to
                ``SamplingParams()``.
            tools: Optional OpenAI-compatible tool definitions.
            tool_choice: Optional OpenAI-compatible tool choice.

        Returns:
            List of ``CompletionOutput`` objects, one per prompt.
        """
        params = sampling_params or SamplingParams()

        if isinstance(prompts, str):
            prompts = [prompts]
        message_batches = []
        for p in prompts:
            if isinstance(p, str):
                message_batches.append([{"role": "user", "content": p}])
            elif isinstance(p, list):
                message_batches.append(p)
            else:
                raise TypeError(f"Unsupported prompt type: {type(p)}")

        outputs = []
        for messages in message_batches:
            tool_config = validate_tool_request(messages, tools, tool_choice)
            with self._admission():
                request = self._make_generation_request(
                    messages,
                    params,
                    tools=tool_config.tools,
                    tool_choice=tool_config.tool_choice,
                    tool_config=tool_config,
                )

                response = self._handle_request(request)
            text = response.output_texts[0] if response.output_texts else ""
            ids = response.output_ids[0] if response.output_ids else []
            reason = finish_reason_name(self._rt, response.finish_reasons[0]) \
                if response.finish_reasons else "stop"
            lps = _convert_logprobs(response.logprobs[0]) if (
                params.num_logprobs > 0 and response.logprobs) else []
            out = self._parse_generation_output(text, ids, reason, tool_config)
            out.logprobs = lps
            outputs.append(out)

        return outputs

    def chat(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> CompletionOutput:
        """Single-turn chat completion (convenience wrapper).

        Args:
            messages: OpenAI-style message list.
            sampling_params: Sampling configuration.
            tools: Optional OpenAI-compatible tool definitions.
            tool_choice: Optional OpenAI-compatible tool choice.

        Returns:
            A single ``CompletionOutput``.
        """
        return self.generate([messages],
                             sampling_params,
                             tools=tools,
                             tool_choice=tool_choice)[0]

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        prebuilt_request: Optional[Any] = None,
        admission_handoff: Optional[Any] = None,
    ) -> Generator[StreamDelta, None, None]:
        """Stream generation deltas for a single message list.

        Runs ``handleRequest`` in a background thread with a
        ``StreamChannel`` attached, yielding ``StreamDelta`` objects as
        tokens are produced.
        """
        params = sampling_params or SamplingParams()

        channel = self._rt.StreamChannel.create()
        channel.set_skip_special_tokens(True)

        # Admission spans decode through inference; the HTTP layer acquires
        # it itself before prebuilding (and owns the release), so only the
        # self-building path acquires here.
        sem = None if prebuilt_request is not None else self._admission()
        if sem is not None:
            sem.acquire()
        try:
            if prebuilt_request is not None:
                # Reuse a request built by the caller (the HTTP layer
                # validates media before the SSE response starts); just
                # attach the channel.
                request = prebuilt_request
                request.stream_channels = [channel]
            else:
                request = self._make_generation_request(
                    messages,
                    params,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream_channel=channel,
                )
        except BaseException:
            if sem is not None:
                sem.release()
            raise

        error_holder = [None]

        def _run():
            try:
                self._handle_request(request)
            except Exception as exc:
                error_holder[0] = exc
                channel.cancel()
            finally:
                if sem is not None:
                    sem.release()
                if admission_handoff is not None:
                    admission_handoff.release()

        worker = threading.Thread(target=_run, daemon=True)
        # Transfer gate ownership before start(): the worker owns it the moment
        # it may run (a join timeout below must not release it while the C++
        # call still runs); a start() failure hands it back to the HTTP layer.
        if admission_handoff is not None:
            admission_handoff.worker_started()
        try:
            worker.start()
        except BaseException:
            if admission_handoff is not None:
                admission_handoff.worker_start_failed()
            raise

        try:
            while True:
                chunk = channel.wait_pop(timeout_ms=200)
                if chunk is None:
                    if channel.is_finished() or channel.is_cancelled():
                        break
                    continue
                reason = finish_reason_name(
                    self._rt, chunk.reason) if chunk.finished else None
                yield StreamDelta(
                    text=chunk.text,
                    token_ids=list(chunk.token_ids),
                    finished=chunk.finished,
                    finish_reason=reason,
                    logprobs=_convert_logprobs(chunk.logprobs),
                )
                if chunk.finished:
                    break
        finally:
            # A consumer that stops early (client disconnect closes this
            # generator) must cancel the channel, or the worker keeps
            # generating while holding the inference lock.
            if not (channel.is_finished() or channel.is_cancelled()):
                channel.cancel()
            worker.join(timeout=_STREAM_JOIN_TIMEOUT_S)

        if error_holder[0] is not None:
            raise error_holder[0]

    def generate_stream_with_audio(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
        *,
        audio_params: Optional[AudioParams] = None,
        prebuilt_request: Optional[Any] = None,
        admission_handoff: Optional[Any] = None,
    ) -> Generator[StreamDelta, None, None]:
        """Stream text and audio deltas for a single Omni request.

        Runs the Thinker-Talker streaming pipeline in a background thread.
        Text deltas arrive through a ``StreamChannel`` and PCM chunks through
        an ``AudioStreamChannel``; the two are interleaved into one generator.
        Admission follows generate_stream: the HTTP layer owns the gate when
        it passes ``prebuilt_request``; otherwise it is acquired here.
        """
        if not self.omni_capable:
            raise ValueError("Omni audio output not available: talker / "
                             "code_predictor / code2wav engines not loaded.")
        params = sampling_params or SamplingParams()

        channel = self._rt.StreamChannel.create()
        channel.set_skip_special_tokens(True)
        audio_channel = self._rt.AudioStreamChannel()
        omni_params = _native_audio_params(self._rt, audio_params
                                           or AudioParams())
        # The HTTP layer takes the slot non-blocking and hands it over (with
        # or without a prebuilt request); direct Python callers acquire here.
        owns_gate = (prebuilt_request is not None
                     or admission_handoff is not None)
        sem = None if owns_gate else self._admission()
        if sem is not None:
            sem.acquire()
        try:
            if prebuilt_request is not None:
                request = prebuilt_request
                request.stream_channels = [channel]
            else:
                request = self._make_generation_request(
                    messages,
                    params,
                    stream_channel=channel,
                )
        except BaseException:
            if sem is not None:
                sem.release()
            raise

        def _run():
            # Same serialization as _handle_request: the batcher holds only
            # this lock, so without it batched text would run concurrently.
            with self._infer_guard():
                self._runtime.handle_request_streaming_audio(
                    request, audio_channel, omni_params)

        yield from _pump_channels(self._rt,
                                  _run,
                                  channel,
                                  audio_channel,
                                  sem=sem,
                                  admission_handoff=admission_handoff)

    # ------------------------------------------------------------------
    # Server API
    # ------------------------------------------------------------------

    def generate_speech_stream(
        self,
        text: str,
        audio_params: Optional[AudioParams] = None,
        *,
        admission_handoff: Optional[Any] = None,
    ) -> Generator[StreamDelta, None, None]:
        """Standalone TTS on the Omni stack: synthesize ``text`` directly.

        No Thinker generation pass — the input text goes straight to the
        Talker. Yields audio-only StreamDeltas.
        """
        if not self.omni_capable:
            raise ValueError("TTS not available: Omni audio engines "
                             "(talker/code_predictor/code2wav) not loaded")
        sem = None if admission_handoff is not None else self._admission()
        yield from _stream_tts(self._rt,
                               self._runtime,
                               text,
                               audio_params or AudioParams(),
                               sem,
                               admission_handoff=admission_handoff,
                               infer_guard=self._infer_guard())

    def list_voices(self) -> List[str]:
        """Speaker names accepted as ``voice``; empty when not Omni-capable."""
        if not self.omni_capable:
            return []
        return sorted(self._runtime.get_speaker_names())

    def serve(self,
              host: str = "0.0.0.0",
              port: int = 8000,
              *,
              enable_batching: bool = False,
              batch_timeout_ms: float = 10.0,
              max_queue_batch_size: Optional[int] = None,
              request_queue_size: Optional[int] = None,
              allowed_local_media_path: Optional[str] = None) -> None:
        """Start an OpenAI-compatible HTTP server.

        Args:
            host: Bind address.
            port: Bind port.
            enable_batching: Batch compatible non-streaming HTTP requests.
            batch_timeout_ms: Maximum time to wait for compatible requests.
            max_queue_batch_size: Optional cap for queued HTTP micro-batches.
            request_queue_size: Max concurrently admitted requests (queued +
                running) before the server returns backpressure. None uses the
                server default.
            allowed_local_media_path: Directory HTTP clients may reference local
                media from. Unset rejects bare paths and ``file://`` URLs.
        """
        from .api_server import _DEFAULT_REQUEST_QUEUE_SIZE, run_server

        run_server(
            self,
            host=host,
            port=port,
            enable_batching=enable_batching,
            batch_timeout_ms=batch_timeout_ms,
            max_queue_batch_size=max_queue_batch_size,
            request_queue_size=(request_queue_size if request_queue_size
                                is not None else _DEFAULT_REQUEST_QUEUE_SIZE),
            allowed_local_media_path=allowed_local_media_path,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> str:
        """Path to the resolved model checkpoint."""
        return self._model_dir

    @property
    def engine_dir(self) -> str:
        """Path to the TensorRT engine directory."""
        return self._engine_dir

    @property
    def max_batch_size(self) -> int:
        """Maximum batch size supported by the loaded engine."""
        return self._max_batch_size

    @property
    def has_draft_model(self) -> bool:
        """Whether Eagle speculative decoding is active."""
        return self._runtime.has_draft_model()

    @property
    def omni_capable(self) -> bool:
        """Whether the Omni audio-output stack is loaded."""
        return self._omni_capable


class TTS:
    """TTS-only serving for Qwen3-TTS-style engine sets.

    Loads Talker + CodePredictor + Code2Wav without a Thinker/text engine.
    ``serve()`` exposes ``/v1/audio/speech``; chat endpoints return 400.

    Example::

        from experimental.server import TTS

        tts = TTS(talker_engine_dir="/engines/qwen3-tts/talker")
        tts.serve(port=8000)

    ``code_predictor_engine_dir`` / ``code2wav_engine_dir`` default to the
    talker directory's siblings; ``tokenizer_dir`` defaults to the talker
    directory itself (the standard export layout ships tokenizer files there).
    """

    text_capable = False
    omni_capable = True
    has_draft_model = False

    def __init__(
        self,
        talker_engine_dir: str,
        code_predictor_engine_dir: Optional[str] = None,
        code2wav_engine_dir: Optional[str] = None,
        tokenizer_dir: str = "",
        model: Optional[str] = None,
    ) -> None:
        talker_engine_dir = os.path.abspath(talker_engine_dir)
        base = os.path.dirname(talker_engine_dir)
        code_predictor_engine_dir = (code_predictor_engine_dir
                                     or os.path.join(base, "code_predictor"))
        code2wav_engine_dir = (code2wav_engine_dir
                               or os.path.join(base, "code2wav"))
        for name, path in (("talker", talker_engine_dir),
                           ("code_predictor", code_predictor_engine_dir),
                           ("code2wav", code2wav_engine_dir)):
            if not os.path.isdir(path):
                raise ValueError(f"{name} engine dir not found: {path}")

        self.model_dir = talker_engine_dir
        self._model_id = model or os.path.basename(base) or "tts"
        self._rt = _import_runtime()
        logger.info("Loading TTS engines (talker=%s) ...", talker_engine_dir)
        self._runtime = self._rt.TTSRuntime(
            talker_engine_dir=talker_engine_dir,
            code_predictor_engine_dir=code_predictor_engine_dir,
            code2wav_engine_dir=code2wav_engine_dir,
            tokenizer_dir=tokenizer_dir,
        )
        logger.info("TTS runtime ready")
        self._admission_sem = threading.Semaphore(1)

    def _admission(self):
        """Per-instance admission gate (mirrors LLM._admission)."""
        return self._admission_sem

    def generate_speech_stream(
        self,
        text: str,
        audio_params: Optional[AudioParams] = None,
        *,
        admission_handoff: Optional[Any] = None,
    ) -> Generator[StreamDelta, None, None]:
        """Synthesize ``text``; yields audio-only StreamDeltas."""
        sem = None if admission_handoff is not None else self._admission()
        yield from _stream_tts(self._rt,
                               self._runtime,
                               text,
                               audio_params or AudioParams(),
                               sem,
                               admission_handoff=admission_handoff)

    def list_voices(self) -> List[str]:
        """Speaker names accepted as ``voice``."""
        return sorted(self._runtime.get_speaker_names())

    def serve(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the HTTP server (speech endpoint only)."""
        from .api_server import run_server

        run_server(self, host=host, port=port)


# ---------------------------------------------------------------------------
# Message conversion & image loading
# ---------------------------------------------------------------------------


def finish_reason_name(rt_module, reason) -> Optional[str]:
    """Map a C++ FinishReason enum value to its OpenAI-compatible string.

    NOT_FINISHED maps to None — reaching this function with a non-terminal
    reason indicates a bug; surfacing None instead of silently returning "stop"
    makes it visible. The fallback "stop" catches truly-unknown enum values
    (e.g. future C++ enum additions). STOP_WORDS and END_ID both map to "stop"
    since OpenAI does not distinguish them.
    """
    return {
        rt_module.FinishReason.NOT_FINISHED: None,
        rt_module.FinishReason.END_ID: "stop",
        rt_module.FinishReason.LENGTH: "length",
        rt_module.FinishReason.CANCELLED: "cancelled",
        rt_module.FinishReason.ERROR: "error",
        rt_module.FinishReason.STOP_WORDS: "stop",
    }.get(reason, "stop")


def _convert_messages_to_cpp(rt_module, messages: List[Dict[str, Any]]):
    """Convert Python message dicts to C++ Message objects."""
    cpp_messages = []
    for msg in messages:
        cpp_msg = rt_module.Message()
        cpp_msg.role = msg["role"]
        content = msg["content"]
        contents_list = []
        if isinstance(content, str):
            contents_list.append(rt_module.MessageContent("text", content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    contents_list.append(rt_module.MessageContent(
                        "text", item))
                elif isinstance(item, dict):
                    ct = item.get("type", "text")
                    if ct == "text":
                        contents_list.append(
                            rt_module.MessageContent(
                                "text",
                                item.get("text", ""),
                            ))
                    elif ct == "image":
                        contents_list.append(
                            rt_module.MessageContent(
                                "image",
                                item.get("image", ""),
                            ))
                    elif ct in ("video", "video_url"):
                        # Frames are decoded out-of-band by _load_image_buffers; the chat
                        # template expands this placeholder into the video triplet and the
                        # ViT runner keys off ImageData.isVideo.
                        contents_list.append(
                            rt_module.MessageContent("video", ""))
                    elif ct in ("audio", "input_audio", "audio_url"):
                        # Audio bytes are decoded out-of-band by
                        # `_load_audio_buffers`; the chat template just emits
                        # an opaque audio placeholder here. The per-model
                        # audio runner expands that into model-specific
                        # special tokens (Qwen3: <|audio_start|> +
                        # N×<|audio_pad|> + <|audio_end|>; Nemotron-Omni:
                        # N×<so_embedding>).
                        contents_list.append(
                            rt_module.MessageContent("audio", ""))
                    else:
                        raise ValueError(f"Unsupported content type: {ct}")
        cpp_msg.contents = contents_list
        cpp_messages.append(cpp_msg)
    return cpp_messages


def _load_image_buffers(rt_module,
                        messages: List[Dict[str, Any]],
                        video_family_fn=lambda: "qwen",
                        video_frame_limits_fn=lambda: {}):
    """Build the ordered ImageData list for the messages: images and videos
    share one list the C++ runner matches positionally against the
    <|image_pad|> / <|video_pad|> placeholders, so append in message order."""
    images = []
    items = [
        item for msg in messages if isinstance(msg.get("content"), list)
        for item in msg["content"] if isinstance(item, dict)
    ]
    # Videos and images share one engine token profile: track the remaining
    # budget so multiple media cannot each claim full capacity. Lazy so
    # non-video requests never touch the video family whitelist.
    has_video = any(
        item.get("type") in ("video", "video_url") for item in items)
    family = video_family_fn() if has_video else "qwen"
    if has_video and family == "nemotron":
        # The C++ Nemotron video path handles exactly one video and no mixed-in
        # images per request (batch of one); reject other layouts here rather
        # than letting them fail inside the runner.
        n_videos = sum(1 for it in items
                       if it.get("type") in ("video", "video_url"))
        n_images = sum(1 for it in items
                       if it.get("type") in ("image", "image_url"))
        if n_videos > 1 or n_images > 0:
            raise ValueError(
                "Nemotron-Omni video requests support exactly one video and no "
                f"images (got {n_videos} videos, {n_images} images)")
    limits = video_frame_limits_fn() if has_video else {}
    budget = limits.get("max_image_tokens") if limits else None
    video_tokens = 0
    # Pre-pruning token count for the engine-minimum check: the ViT processes
    # every tubelet, so Nemotron's EVS-pruned estimate would understate what the
    # min-profile actually receives. Non-EVS families track the same value.
    video_raw_tokens = 0
    # Request-wide decoded-pixel budget: several videos each under the
    # per-video ceiling must not jointly exhaust host memory.
    pixel_budget = None
    # Request-wide cu_seqlens group budget (Qwen only; InternVL has no
    # cu_seqlens binding): builder-recorded capacity, else the legacy formula.
    cu_budget = None
    if limits and family != "internvl":
        cu_budget = (limits.get("max_cu_seqlen_groups")
                     or limits["max_image_tokens"] //
                     max(1, limits.get("min_image_tokens", 1)))
    image_upper = 0
    # Phase 1: reserve every image up front so the video sampler's budget is
    # order-independent ([image, video] and [video, image] behave identically).
    if budget is not None:
        from .video_sampling import estimate_image_tokens
        for item in items:
            if item.get("type") != "image":
                continue
            path = item.get("image", "")
            if path and os.path.isfile(path):
                est = estimate_image_tokens(path,
                                            family,
                                            limits,
                                            do_resize=bool(
                                                item.get("do_resize", True)))
                image_upper += est
                budget -= est
                if cu_budget is not None:
                    # One cu_seqlens entry per image (Qwen families only;
                    # InternVL has no cu_seqlens binding).
                    cu_budget -= 1
    # Phase 2: build the buffers in original message order (the C++ runner
    # matches them positionally against the placeholders).
    for item in items:
        itype = item.get("type")
        if itype == "image":
            path = item.get("image", "")
            if path and os.path.isfile(path):
                image = rt_module.load_image_from_path(path)
                image.do_resize = bool(item.get("do_resize", True))
                images.append(image)
        elif itype in ("video", "video_url"):
            from .video_sampling import MAX_DECODE_PIXELS, load_video_buffer
            if pixel_budget is None:
                pixel_budget = MAX_DECODE_PIXELS
            buffer, est_tokens, used_px, used_groups = load_video_buffer(
                rt_module,
                item,
                family,
                frame_limits=limits,
                budget=budget,
                pixel_budget=pixel_budget,
                cu_budget=cu_budget)
            images.append(buffer)
            video_tokens += est_tokens
            raw_tokens = est_tokens
            if family == "nemotron":
                from .video_sampling import _nemotron_tubelet_geometry
                geom = _nemotron_tubelet_geometry(limits)
                if geom:
                    t_frames, tokens_per_tubelet, _q = geom
                    raw_tokens = (-(-buffer.frames // t_frames)) \
                        * tokens_per_tubelet
            video_raw_tokens += raw_tokens
            pixel_budget -= used_px
            if cu_budget is not None:
                cu_budget -= used_groups
            if budget is not None:
                budget -= est_tokens
    # Engine bounds are request-wide (all media accumulate in one ViT batch),
    # so validate after the loop: two videos jointly reaching the minimum are
    # fine, one alone may not be.
    if cu_budget is not None and cu_budget < 0:
        raise ValueError(
            "request media exceed the visual engine's cu_seqlens capacity; "
            "reduce the media count")
    if budget is not None and budget < 0:
        raise ValueError(
            "request media need more visual tokens than the engine's "
            f"budget of {limits['max_image_tokens']}; reduce the media in "
            "the request")
    if (video_raw_tokens or image_upper) and limits and \
            limits.get("min_image_tokens"):
        # The engine minimum is request-wide; the upper estimate is pre-EVS
        # (raw tubelets for Nemotron). It can fall short for a too-short clip or
        # do_resize=false media; resized per-item images are floored above this.
        upper_tokens = video_raw_tokens + image_upper
        if upper_tokens < limits["min_image_tokens"]:
            raise ValueError(
                f"request media yield ~{upper_tokens} visual tokens but the "
                f"engine needs at least {limits['min_image_tokens']}; use "
                "longer videos or raise nframes/fps")
    return images


def _load_audio_buffers(rt_module, messages: List[Dict[str, Any]]):
    """Load audio content from messages into AudioData buffers.

    Returns an empty list when no audio is present, keeping the byte-identical
    fast path for text-only and image-only requests.
    """
    from .audio_preprocess import load_audio_buffers
    return load_audio_buffers(rt_module, messages)
