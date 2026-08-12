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
"""Anthropic Messages API compatibility layer.

Translates ``POST /v1/messages`` (+ ``/v1/messages/count_tokens``) to the
internal OpenAI-style pipeline so Anthropic-protocol agents (e.g. Claude
Code) can talk to the server directly.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .tool_calling import parse_assistant_output

logger = logging.getLogger("edgellm.anthropic_compat")

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "stop_sequence": "stop_sequence",
}


def _blocks_to_text(content: Any) -> str:
    """Join the text blocks of a string-or-block-array content field."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for blk in content:
        if isinstance(blk, str):
            parts.append(blk)
        elif isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(str(blk.get("text", "")))
    return "\n".join(p for p in parts if p)


def convert_request(
    body: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[Any],
           Dict[str, Any]]:
    """Convert an Anthropic Messages request into (messages, tools,
    tool_choice, sampling) for the internal OpenAI-style pipeline."""
    messages: List[Dict[str, Any]] = []

    system = body.get("system")
    if system:
        system_text = _blocks_to_text(system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        for blk in content if isinstance(content, list) else []:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")
            if btype == "text":
                text_parts.append(str(blk.get("text", "")))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": blk.get("id", ""),
                    "type": "function",
                    "function": {
                        "name":
                        blk.get("name", ""),
                        "arguments":
                        json.dumps(blk.get("input") or {}, ensure_ascii=False),
                    },
                })
            elif btype == "tool_result":
                result = blk.get("content", "")
                if not isinstance(result, str):
                    result = _blocks_to_text(result) or json.dumps(
                        result, ensure_ascii=False)
                if blk.get("is_error"):
                    result = f"[tool error] {result}"
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": blk.get("tool_use_id", ""),
                    "content": result,
                })
            # thinking / redacted_thinking / image blocks are dropped: no
            # reasoning replay channel, and vision goes through the OpenAI
            # endpoint's own block handling.

        if role == "assistant":
            out: Dict[str, Any] = {"role": "assistant"}
            out["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
        else:
            # Anthropic sends tool results as user-role blocks; the internal
            # pipeline expects dedicated tool-role messages first.
            messages.extend(tool_results)
            text = "\n".join(p for p in text_parts if p)
            if text or not tool_results:
                messages.append({"role": "user", "content": text})

    tools = None
    if body.get("tools"):
        tools = []
        for tool in body["tools"]:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            # Server-executed tools (web_search_*, code_execution_*, ...)
            # carry a versioned type and no input_schema. We cannot execute
            # them; offering them to the model would produce tool calls the
            # client does not own, so they are skipped.
            if not isinstance(tool.get("input_schema"), dict):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            })

    tool_choice: Optional[Any] = None
    raw_choice = body.get("tool_choice")
    if isinstance(raw_choice, dict):
        ctype = raw_choice.get("type")
        if ctype == "auto":
            tool_choice = "auto"
        elif ctype == "any":
            tool_choice = "required"
        elif ctype == "tool":
            tool_choice = {
                "type": "function",
                "function": {
                    "name": raw_choice.get("name", "")
                },
            }
        elif ctype == "none":
            tool_choice = "none"

    sampling = {
        "max_tokens": body.get("max_tokens", 2048),
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.9),
        "top_k": body.get("top_k", 50),
        "stop": list(body.get("stop_sequences") or []),
    }
    return messages, tools, tool_choice, sampling


def convert_stop_reason(finish_reason: Optional[str]) -> str:
    return _STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")


def build_content_blocks(
        content: Optional[str],
        tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build Anthropic content blocks from OpenAI-style message parts."""
    blocks: List[Dict[str, Any]] = []
    if content:
        blocks.append({"type": "text", "text": content})
    for call in tool_calls:
        fn = call.get("function", {})
        try:
            tool_input = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            tool_input = {"__raw__": fn.get("arguments")}
        if not isinstance(tool_input, dict):
            # tool_use.input must be an object; strict SDKs reject scalars.
            tool_input = {"value": tool_input}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
            "name": fn.get("name", ""),
            "input": tool_input,
        })
    if not blocks:
        # Anthropic responses always carry at least one block. An empty
        # content array makes Claude Code treat the stream as truncated (and
        # the SDK's get_final_text() raise), so emit an empty text block.
        blocks.append({"type": "text", "text": ""})
    return blocks


def _usage(prompt_tokens: Optional[int],
           completion_tokens: int) -> Dict[str, int]:
    return {
        "input_tokens": prompt_tokens or 0,
        "output_tokens": completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _event(etype: str, payload: Dict[str, Any]) -> str:
    payload = dict(payload)
    payload["type"] = etype
    return f"event: {etype}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _message_start_events(message_id: str, model: str,
                          prompt_tokens: Optional[int]):
    yield _event(
        "message_start", {
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _usage(prompt_tokens, 0),
            }
        })
    yield _event("ping", {})


def _block_events(blocks: List[Dict[str, Any]]):
    for index, blk in enumerate(blocks):
        if blk["type"] == "text":
            yield _event("content_block_start", {
                "index": index,
                "content_block": {
                    "type": "text",
                    "text": ""
                },
            })
            # Chunk the text so clients exercise their delta paths.
            text = blk["text"]
            for i in range(0, len(text), 512):
                yield _event(
                    "content_block_delta", {
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": text[i:i + 512]
                        },
                    })
        else:
            yield _event(
                "content_block_start", {
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": blk["id"],
                        "name": blk["name"],
                        "input": {},
                    },
                })
            yield _event(
                "content_block_delta", {
                    "index": index,
                    "delta": {
                        "type":
                        "input_json_delta",
                        "partial_json":
                        json.dumps(blk["input"], ensure_ascii=False),
                    },
                })
        yield _event("content_block_stop", {"index": index})


def _tail_events(stop_reason: str, completion_tokens: int):
    yield _event(
        "message_delta", {
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None
            },
            "usage": {
                "output_tokens": completion_tokens
            },
        })
    yield _event("message_stop", {})


def stream_events(message_id: str, model: str, prompt_tokens: Optional[int],
                  blocks: List[Dict[str, Any]], stop_reason: str,
                  completion_tokens: int):
    """Emit a complete, spec-ordered Anthropic SSE event sequence.

    Blocks are emitted sequentially and each is closed before the next opens
    — official SDKs hard-fail on interleaved block events.
    """
    yield from _message_start_events(message_id, model, prompt_tokens)
    yield from _block_events(blocks)
    yield from _tail_events(stop_reason, completion_tokens)


_IM_END_TOKEN = "<|im_end|>"


def stream_run(llm_instance,
               messages,
               params,
               tool_config,
               message_id: str,
               model: str,
               prompt_tokens: Optional[int],
               prebuilt_request,
               handoff=None):
    """Streaming generation for /v1/messages.

    message_start + ping flush before inference so clients get liveness
    immediately; generation then runs inside this generator, replayed
    post-parse (tool parsing buffers). ``handoff`` carries the admission gate
    to the runtime worker (released when the native call exits).
    """
    yield from _message_start_events(message_id, model, prompt_tokens)

    text_parts: List[str] = []
    completion_tokens = 0
    finish_reason: Optional[str] = None
    try:
        # Tool parsing needs the full output, so text is buffered rather than
        # streamed. Emit a periodic ping (permitted anywhere in the stream) so
        # long edge-device generations do not trip client idle timeouts, and so
        # a client disconnect raises GeneratorExit here promptly — closing the
        # native generator and releasing the admission slot.
        last_ping = time.monotonic()
        for delta in llm_instance.generate_stream(
                messages,
                params,
                tools=tool_config.tools,
                tool_choice=tool_config.tool_choice,
                prebuilt_request=prebuilt_request,
                admission_handoff=handoff):
            completion_tokens += len(delta.token_ids or [])
            if delta.text:
                text_parts.append(delta.text)
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"
            now = time.monotonic()
            if now - last_ping >= 5.0:
                yield _event("ping", {})
                last_ping = now
    except Exception as exc:
        logger.exception("Anthropic streaming inference failed")
        _, payload = error_response(500, str(exc))
        yield _event("error", payload)
        return

    try:
        output_text = "".join(text_parts).replace(_IM_END_TOKEN, "")
        if tool_config.parse_output:
            parsed = parse_assistant_output(output_text, tool_config,
                                            llm_instance.model_dir)
            tool_calls = [call.to_openai() for call in parsed.tool_calls]
            content = parsed.content.strip()
        else:
            tool_calls = []
            content = output_text.strip()
    except Exception as exc:
        # Tool post-processing (e.g. serializing malformed generated args) can
        # raise after generation; keep it inside the Anthropic error protocol
        # instead of tearing the stream down without an error event.
        logger.exception("Anthropic tool post-processing failed")
        _, payload = error_response(500, str(exc))
        yield _event("error", payload)
        return

    # Truncation/cancellation wins over tool_use so clients do not execute
    # potentially half-emitted calls.
    if tool_calls and (finish_reason or "stop") == "stop":
        finish_reason = "tool_calls"
    blocks = build_content_blocks(content, tool_calls)
    yield from _block_events(blocks)
    yield from _tail_events(convert_stop_reason(finish_reason),
                            completion_tokens)


_ERROR_TYPES = {
    413: "request_too_large",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def error_response(status: int, message: str) -> Tuple[int, Dict[str, Any]]:
    if status in _ERROR_TYPES:
        etype = _ERROR_TYPES[status]
    elif status < 500:
        etype = "invalid_request_error"
    else:
        etype = "api_error"
    return status, {
        "type": "error",
        "error": {
            "type": etype,
            "message": message,
        },
    }
