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

Usage (standalone)::

    python -m experimental.server \\
        --model Qwen/Qwen3-1.7B --port 8000

Usage (from LLM object)::

    from experimental.server import LLM
    llm = LLM(model="Qwen/Qwen3-1.7B")
    llm.serve(port=8000)
"""

import argparse
import json
import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger("edgellm.api_server")

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
IM_END_TOKEN = "<|im_end|>"


def _split_reasoning_and_content(text: str):
    """Split model output into (reasoning_content, content) around <think> tags."""
    think_open = text.find(THINK_OPEN_TAG)
    think_close = text.find(THINK_CLOSE_TAG)
    if think_open != -1 and think_close != -1 and think_close > think_open:
        reasoning = text[think_open + len(THINK_OPEN_TAG):think_close].strip()
        content = text[think_close + len(THINK_CLOSE_TAG):].strip()
        return reasoning, content or None
    return None, text.strip() if text.strip() else None


def _create_app(llm_instance):
    """Create a FastAPI app backed by the given LLM instance."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for the server. "
                           "Install: pip install fastapi uvicorn") from exc

    app = FastAPI(
        title="TensorRT Edge-LLM Server",
        version="0.1.0",
        description=
        "OpenAI-compatible inference server powered by TensorRT Edge-LLM",
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model": llm_instance.model_dir,
            "speculative_decoding": llm_instance.has_draft_model,
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
        if not messages:
            return JSONResponse(status_code=400,
                                content={"error": "messages required"})

        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        top_k = body.get("top_k", 50)
        max_tokens = body.get("max_tokens", 2048)
        stream = body.get("stream", False)
        enable_thinking = body.get("enable_thinking", False)
        disable_spec_decode = body.get("disable_spec_decode", False)

        rt = llm_instance._rt
        from .engine import _convert_messages_to_cpp, _load_image_buffers

        try:
            cpp_messages = _convert_messages_to_cpp(rt, messages)
        except (ValueError, KeyError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid messages: {exc}"},
            )

        image_buffers = _load_image_buffers(rt, messages)

        request = rt.LLMGenerationRequest()
        req = rt.Request(messages=cpp_messages)
        req.image_buffers = image_buffers
        request.requests = [req]
        request.temperature = temperature
        request.top_p = top_p
        request.top_k = top_k
        request.max_generate_length = max_tokens
        request.apply_chat_template = True
        request.add_generation_prompt = True
        request.enable_thinking = enable_thinking
        request.disable_spec_decode = disable_spec_decode

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            from .engine import SamplingParams

            params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                disable_spec_decode=disable_spec_decode,
            )

            return StreamingResponse(
                _generate_stream_sse(
                    llm_instance,
                    messages,
                    params,
                    response_id,
                    enable_thinking,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            response = llm_instance._runtime.handle_request(request)
        except Exception as exc:
            logger.exception("Inference failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})

        raw_text = response.output_texts[0] if response.output_texts else ""
        output_text = raw_text.replace(IM_END_TOKEN, "")
        output_ids = response.output_ids[0] if response.output_ids else []
        completion_tokens = len(output_ids)

        reasoning, answer = _split_reasoning_and_content(output_text)

        message_body: Dict[str, Any] = {"role": "assistant"}
        if reasoning is not None:
            message_body["reasoning"] = reasoning
        message_body["content"] = (
            (answer if answer is not None else reasoning) or "")

        return {
            "id":
            response_id,
            "object":
            "chat.completion",
            "choices": [{
                "index": 0,
                "message": message_body,
                "finish_reason": "stop",
            }],
            "usage": {
                "completion_tokens": completion_tokens,
            },
        }

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


def _generate_stream_sse(llm_instance, messages, params, response_id,
                         enable_thinking):
    """Yield real SSE chunks via StreamChannel streaming."""
    yield _sse_chunk(response_id, {"role": "assistant"})

    sm = _ThinkingStateMachine(enable_thinking)
    finish_reason: Optional[str] = None

    try:
        for delta in llm_instance.generate_stream(messages, params):
            if delta.text:
                for field, text in sm.feed(delta.text):
                    yield _sse_chunk(response_id, {field: text})
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
    except Exception:
        logger.exception("Streaming inference failed")
        finish_reason = "error"

    for field, text in sm.flush():
        yield _sse_chunk(response_id, {field: text})

    yield _sse_chunk(response_id, {}, finish_reason=finish_reason or "stop")
    yield "data: [DONE]\n\n"


def _sse_chunk(response_id: str,
               delta: dict,
               finish_reason: Optional[str] = None):
    choice: Dict[str, Any] = {"delta": delta, "index": 0}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    payload = {"id": response_id, "choices": [choice]}
    return f"data: {json.dumps(payload)}\n\n"


def run_server(llm_instance, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the OpenAI-compatible server."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required. Install: pip install uvicorn") from exc

    app = _create_app(llm_instance)
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
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local checkpoint path",
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
        "--use-trt-native-ops",
        action="store_true",
        default=False,
        help="Use TensorRT native ops instead of custom plugins",
    )
    parser.add_argument(
        "--spec-decode-engine-dir",
        "--eagle-engine-dir",  # deprecated alias, kept for backward compat
        dest="spec_decode_engine_dir",
        default="",
        help="Pre-built speculative decoding engine dir (EAGLE or MTP)",
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
    args = parser.parse_args()

    from .engine import LLM

    llm = LLM(
        model=args.model,
        max_input_len=args.max_input_len,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        use_trt_native_ops=args.use_trt_native_ops,
        eagle_engine_dir=args.spec_decode_engine_dir,
        draft_top_k=args.draft_top_k,
        draft_step=args.draft_step,
        verify_tree_size=args.verify_tree_size,
    )
    llm.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
