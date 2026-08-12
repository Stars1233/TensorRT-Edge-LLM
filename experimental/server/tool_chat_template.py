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
"""Tool-aware chat template formatting for agentic workloads."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


class ToolChatTemplateError(ValueError):
    """Tool chat template formatting failed."""


def needs_tool_chat_template(
    messages: Sequence[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> bool:
    """Return whether a request needs full chat template formatting."""

    if tools:
        return True
    if tool_choice and tool_choice != "none":
        return True

    for msg in messages:
        if msg.get("role") == "tool":
            return True
        if msg.get("tool_calls") or msg.get("function_call"):
            return True
    return False


def _json_loads_if_possible(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize OpenAI tool calls for HF chat templates."""
    normalized = copy.deepcopy(tool_call)
    function = normalized.get("function")
    if isinstance(function, dict):
        function["arguments"] = _json_loads_if_possible(
            function.get("arguments", {}))
        return normalized

    if "arguments" in normalized:
        normalized["arguments"] = _json_loads_if_possible(
            normalized["arguments"])
    return normalized


def _flatten_content_blocks(content: Any) -> Any:
    """Collapse a non-empty array of typed text blocks into the plain string
    HF chat templates expect. Anything else -- empty lists, media blocks, raw
    strings, JSON tool-result lists -- passes through unchanged so the later
    role-specific handling (json.dumps of tool results, video normalization)
    still applies."""
    if not isinstance(content, list) or not content:
        return content
    parts: List[str] = []
    for item in content:
        if (isinstance(item, dict)
                and item.get("type") in ("text", "input_text")
                and isinstance(item.get("text"), str)):
            parts.append(item["text"])
        else:
            return content
    return "\n".join(p for p in parts if p)


def normalize_messages_for_tools(
        messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a template-friendly copy of OpenAI-style messages."""
    normalized_messages: List[Dict[str, Any]] = []

    for raw_msg in messages:
        msg = copy.deepcopy(raw_msg)
        # Agentic clients (Claude Code, OpenClaw) send content as
        # [{"type":"text",...}] arrays; collapse pure-text arrays to a string
        # (media/JSON-list content passes through) or the template renders an
        # empty turn and the model never sees the message.
        if isinstance(msg.get("content"), list):
            msg["content"] = _flatten_content_blocks(msg["content"])

        # HF templates only know {"type": "video"}: normalize the OpenAI
        # "video_url" alias here or the template emits no placeholder and the
        # loaded ViT buffer is silently dropped.
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "video_url":
                    ref = item.pop("video_url", None)
                    item["type"] = "video"
                    item["video"] = (ref.get("url", "") if isinstance(
                        ref, dict) else ref)

        if msg.get("function_call") and not msg.get("tool_calls"):
            function_call = msg.pop("function_call")
            if isinstance(function_call, dict):
                msg["tool_calls"] = [{
                    "type": "function",
                    "function": {
                        "name":
                        function_call.get("name", ""),
                        "arguments":
                        _json_loads_if_possible(
                            function_call.get("arguments", {})),
                    },
                }]

        if msg.get("tool_calls"):
            msg["tool_calls"] = [
                _normalize_tool_call(tc) for tc in msg["tool_calls"]
                if isinstance(tc, dict)
            ]

        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if content is None:
                msg["content"] = ""
            elif not isinstance(content, str):
                msg["content"] = json.dumps(content, ensure_ascii=False)

        normalized_messages.append(msg)

    return normalized_messages


class ToolChatTemplateFormatter:
    """Apply HF chat templates for tool-aware requests."""

    def __init__(
        self,
        template_dirs: Iterable[str],
        *,
        template_owner: Optional[Any] = None,
    ) -> None:
        self._template_dirs = [
            str(Path(d)) for d in template_dirs if d and Path(d).is_dir()
        ]
        self._template_owner = template_owner

    def _load_template_owner(self) -> Any:
        if self._template_owner is not None:
            return self._template_owner

        try:
            from transformers import (AutoProcessor, AutoTokenizer,
                                      PreTrainedTokenizerFast)
        except ImportError as exc:
            raise ToolChatTemplateError(
                "transformers is required for tool-aware chat templates."
            ) from exc

        # PreTrainedTokenizerFast fallback ignores config.json (an EdgeLLM engine
        # config in engine-dir mode) that AutoProcessor/AutoTokenizer reject.
        loaders = (AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast)

        errors = []
        for template_dir in self._template_dirs:
            for loader in loaders:
                try:
                    owner = loader.from_pretrained(template_dir,
                                                   trust_remote_code=True)
                except Exception as exc:
                    errors.append(f"{loader.__name__}({template_dir}): {exc}")
                    continue
                if hasattr(owner, "apply_chat_template"):
                    self._template_owner = owner
                    return owner

        detail = "; ".join(errors[-3:]) if errors else "no template dirs"
        raise ToolChatTemplateError(
            "Could not load a tokenizer or processor with apply_chat_template "
            f"from {self._template_dirs}: {detail}")

    def format(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Format messages and tools into a model-native prompt."""
        owner = self._load_template_owner()
        normalized_messages = normalize_messages_for_tools(messages)

        kwargs: Dict[str, Any] = {
            "tools": list(tools or []),
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        try:
            prompt = owner.apply_chat_template(normalized_messages, **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in kwargs:
                raise ToolChatTemplateError(
                    f"Failed to apply tool-aware chat template: {exc}"
                ) from exc
            # Older tokenizers may reject Qwen-style enable_thinking.
            kwargs.pop("enable_thinking", None)
            try:
                prompt = owner.apply_chat_template(normalized_messages,
                                                   **kwargs)
            except Exception as retry_exc:
                raise ToolChatTemplateError(
                    "Failed to apply tool-aware chat template: "
                    f"{retry_exc}") from retry_exc
        except Exception as exc:
            raise ToolChatTemplateError(
                f"Failed to apply tool-aware chat template: {exc}") from exc

        if not isinstance(prompt, str):
            raise ToolChatTemplateError(
                "Tool-aware chat template returned non-string prompt. "
                "Use tokenize=False-compatible tokenizer/processor templates.")
        return prompt

    def count_tokens(self, text: str) -> Optional[int]:
        """Count tokens of a rendered prompt (no special-token wrapper — the
        rendered text already carries them). None if encoding unavailable."""
        owner = self._load_template_owner()
        tokenizer = getattr(owner, "tokenizer", owner)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            return None
        try:
            return len(encode(text, add_special_tokens=False))
        except TypeError:
            return len(encode(text))
