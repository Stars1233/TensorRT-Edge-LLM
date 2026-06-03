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
Chat template extraction for ONNX export sidecars.

Writes ``processed_chat_template.json`` for the C++ runtime tokenizer.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .checkpoint.checkpoint_utils import load_checkpoint_config_dicts

logger = logging.getLogger(__name__)

__all__ = ["process_chat_template", "write_fallback_processed_chat_template"]

# ---------------------------------------------------------------------------
# Hardcoded chat templates for models whose Jinja template is incompatible
# with Jinja2's SandboxedEnvironment (e.g. uses negative list indexing).
# Keys are model_type strings from config.json.
# ---------------------------------------------------------------------------
_TEMPLATES_DIR = Path(__file__).parent / "chat_templates"

_HARDCODED_TEMPLATE_MAP: Dict[str, str] = {
    "phi4mm": "phi4mm.json",
    "phi4_multimodal": "phi4mm.json",
    "qwen3_asr": "qwen3asr.json",
    "qwen3_tts": "qwen3tts.json",
}


def _load_root_config(model_dir: str) -> Dict[str, Any]:
    """Return the root config dict, or empty dict on failure."""
    try:
        root, _ = load_checkpoint_config_dicts(model_dir)
        return root
    except (OSError, ValueError, KeyError):
        return {}


def _is_vlm(model_dir: str) -> bool:
    root = _load_root_config(model_dir)
    has_vision = "vision_config" in root
    embd = root.get("embd_layer") or {}
    has_phi4_vision = "image_embd_layer" in embd
    has_vlm_backend = bool(root.get("vlm_backend"))
    return has_vision or has_phi4_vision or has_vlm_backend


def _is_alpamayo_1_model(model_dir: str) -> bool:
    root = _load_root_config(model_dir)
    return root.get("model_type") == "alpamayo_r1"


def _is_phi4mm_model(model_dir: str) -> bool:
    root = _load_root_config(model_dir)
    return root.get("model_type") in ("phi4mm", "phi4_multimodal")


def _is_qwen3_omni_model(model_dir: str) -> bool:
    root = _load_root_config(model_dir)
    return root.get("model_type") == "qwen3_omni"


def _is_qwen3_asr_model(model_dir: str) -> bool:
    root = _load_root_config(model_dir)
    mt = str(root.get("model_type", "")).lower()
    if "asr" in mt:
        return True
    return any("asr" in str(a).lower()
               for a in (root.get("architectures") or []))


def _is_nemotron_omni_model(model_dir: str) -> bool:
    root = _load_root_config(model_dir)
    return root.get("model_type") in {
        "NemotronH_Nano_VL_V2",
        "NemotronH_Nano_Omni_Reasoning_V3",
    }


@dataclass
class Message:
    role: str
    content: Union[str, List[Dict[str, str]]] = field(default_factory=list)


@dataclass
class SystemMessage(Message):
    role: str = "system"
    content: str = "<placeholder_system_prompt>"


@dataclass
class UserMessage(Message):
    role: str = "user"
    content: str = "<placeholder_user_text>"


@dataclass
class MultimodalUserMessage(Message):
    role: str = "user"
    content: List[Dict[str, str]] = field(
        default_factory=lambda: [{
            "type": "text",
            "text": "<placeholder_user_text>"
        }])

    def add_image_content(self, image: str) -> None:
        self.content.append({"type": "image", "image": image})

    def add_video_content(self, video: str) -> None:
        self.content.append({"type": "video", "video": video})

    def add_audio_content(self, audio: str) -> None:
        self.content.append({"type": "audio", "audio": audio})


@dataclass
class AssistantMessage(Message):
    role: str = "assistant"
    content: str = "<placeholder_assistant_text>"


@dataclass
class AssistantToolCallMessage(Message):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=lambda: [{
        "id": "__SENTINEL_TOOL_CALL_ID__",
        "type": "function",
        "function": {
            "name": "__sentinel_tool__",
            "arguments": "{}",
        },
    }])


@dataclass
class ToolMessage(Message):
    role: str = "tool"
    content: str = "__SENTINEL_TOOL_RESULT__"
    tool_call_id: str = "__SENTINEL_TOOL_CALL_ID__"
    name: str = "__sentinel_tool__"


def _format_messages(
    tokenizer: Any,
    messages: List[Message],
    add_generation_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> str:
    message_dicts = [asdict(msg) for msg in messages]
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(message_dicts, **kwargs)
    except (TypeError, ValueError, KeyError):
        flat_dicts = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content = item.get("text", "")
                        break
            flat_dicts.append({"role": msg.role, "content": content})
        return tokenizer.apply_chat_template(flat_dicts, **kwargs)


def _extract_prefix_suffix(text: str, placeholder: str) -> Tuple[str, str]:
    idx = text.find(placeholder)
    if idx == -1:
        return "", ""
    return text[:idx], text[idx + len(placeholder):]


def _extract_content_pattern(
    tokenizer: Any,
    system_prompt: SystemMessage,
    content_type: str,
    placeholder: str,
    text_only_formatted: str,
    placeholder_text: str,
) -> Optional[str]:
    user_with_content = MultimodalUserMessage()
    if content_type == "image":
        user_with_content.add_image_content(placeholder)
    elif content_type == "video":
        user_with_content.add_video_content(placeholder)
    elif content_type == "audio":
        user_with_content.add_audio_content(placeholder)
    else:
        return None

    with_content_formatted = _format_messages(
        tokenizer, [system_prompt, user_with_content])

    if placeholder_text in text_only_formatted and placeholder_text in with_content_formatted:
        text_pos = text_only_formatted.find(placeholder_text) + len(
            placeholder_text)
        content_pos = with_content_formatted.find(placeholder_text) + len(
            placeholder_text)
        text_only_suffix = text_only_formatted[text_pos:]
        with_content_suffix = with_content_formatted[content_pos:]
        if text_only_suffix and with_content_suffix.endswith(text_only_suffix):
            pattern = with_content_suffix[:-len(text_only_suffix)]
        else:
            pattern = with_content_suffix
        pattern = re.sub(rf"^{content_type.capitalize()} \d+:\s*", "", pattern)
        return pattern if pattern else None
    return None


def write_fallback_processed_chat_template(model_dir: str,
                                           output_dir: str) -> None:
    """Write a minimal ``processed_chat_template.json`` when none exists."""
    os.makedirs(output_dir, exist_ok=True)
    template_dst = os.path.join(output_dir, "processed_chat_template.json")
    if os.path.exists(template_dst):
        return
    fallback = {
        "model_path": model_dir,
        "roles": {
            "system": {
                "prefix": "",
                "suffix": "\n"
            },
            "user": {
                "prefix": "User: ",
                "suffix": "\n"
            },
            "assistant": {
                "prefix": "Assistant: ",
                "suffix": "\n"
            },
        },
        "content_types": {},
        "generation_prompt": "Assistant: ",
        "default_system_prompt": "",
    }
    with open(template_dst, "w") as f:
        json.dump(fallback, f, indent=2)
    logger.info("Wrote fallback processed_chat_template.json")


def _get_model_type(model_dir: str) -> str:
    """Return model_type from config.json, or '' on failure."""
    return _load_root_config(model_dir).get("model_type", "")


def _needs_nemotron_hardcoded_template(model_dir: str) -> bool:
    """Return True if this nemotron_h model uses <SPECIAL_10> tokens.

    Nemotron-Nano-9B-v2 uses <SPECIAL_10>/<SPECIAL_11>/<SPECIAL_12> as chat
    control tokens and needs the hardcoded template.  Nemotron-3-Nano-4B uses
    standard <|im_start|>/<|im_end|> tokens and should fall through to Jinja
    extraction.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        ids = tok.encode("<SPECIAL_10>", add_special_tokens=False)
        return len(ids) == 1  # single token → hardcoded template needed
    except (OSError, ValueError, ImportError):
        return False


def _try_write_hardcoded_template(model_dir: str, output_dir: str) -> bool:
    """If a hardcoded template exists for this model_type, write it and return True."""
    model_type = _get_model_type(model_dir)
    template_file = _HARDCODED_TEMPLATE_MAP.get(model_type)

    # nemotron_h: only use hardcoded template when tokenizer has <SPECIAL_10>
    if not template_file and model_type == "nemotron_h":
        if _needs_nemotron_hardcoded_template(model_dir):
            template_file = "nemotron_nano_v2.json"

    if not template_file:
        return False
    template_path = _TEMPLATES_DIR / template_file
    if not template_path.exists():
        logger.warning("Hardcoded template file not found: %s", template_path)
        return False
    data = json.loads(template_path.read_text())
    data["model_path"] = model_dir  # update to actual path
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "processed_chat_template.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote hardcoded chat template (%s) to %s", template_file,
                out_path)
    return True


def process_chat_template(model_dir: str, output_dir: str) -> None:
    """Extract chat template patterns and write ``processed_chat_template.json``.

    If the model type is in ``_HARDCODED_TEMPLATE_MAP``, that template is used
    directly (highest priority).  Otherwise, attempts standard Jinja-based
    extraction from the tokenizer.  If that fails (e.g. the template uses Python
    constructs incompatible with Jinja2's SandboxedEnvironment), falls back to a
    per-model hardcoded template from ``_HARDCODED_TEMPLATE_MAP``.
    """
    # Highest priority: hardcoded template for this model type.
    if _try_write_hardcoded_template(model_dir, output_dir):
        return

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = None
    is_vlm = _is_vlm(model_dir)
    loaders = [AutoProcessor, AutoTokenizer
               ] if is_vlm else [AutoTokenizer, AutoProcessor]
    # Try model_dir first, then output_dir as fallback (the tokenizer files
    # are already copied there by write_runtime_artifacts before this call).
    search_dirs = [model_dir]
    if output_dir != model_dir:
        search_dirs.append(output_dir)
    for search_dir in search_dirs:
        for ldr in loaders:
            try:
                tok = ldr.from_pretrained(search_dir, trust_remote_code=True)
                if getattr(tok, "chat_template", None):
                    tokenizer = tok
                    break
            except (OSError, ValueError, ImportError, KeyError,
                    AttributeError):
                pass
        if tokenizer is not None:
            break

    if tokenizer is None:
        logger.debug("No chat template found in %s; skipping", model_dir)
        return

    try:
        system_prompt = SystemMessage()
        user_prompt = UserMessage()

        # Some templates (e.g. Qwen3.5) require at least one user message and
        # raise TemplateError when formatting a system-only message.  Always
        # format system+user first (guaranteed to work), then try system-only
        # as a refinement.
        user_formatted = _format_messages(tokenizer,
                                          [system_prompt, user_prompt])

        system_formatted = None
        try:
            system_formatted = _format_messages(tokenizer, [system_prompt])
        except Exception:
            pass

        if system_formatted is not None:
            system_prefix, system_suffix = _extract_prefix_suffix(
                system_formatted, system_prompt.content)
        else:
            # Extract system prefix/suffix from the combined result.
            # e.g. "<|im_start|>system\n<PLACEHOLDER><|im_end|>\n<|im_start|>user\n..."
            system_prefix, system_suffix = _extract_prefix_suffix(
                user_formatted, system_prompt.content)
            # system_suffix will include user segment — trim it below.

        # Find the user segment robustly.  Primary strategy: format a
        # user-only message (no system prompt) to isolate user prefix/suffix
        # without system-related content.  This works for models like Phi-4MM
        # where the tokenizer doesn't inject a default system in user-only mode.
        # For models (e.g. Qwen) that auto-inject a default system prompt, the
        # user-only prefix will contain system_prefix, so we fall back to the
        # original len(system_formatted)-based slice but correct for
        # single-message EOS drift.
        user_only_formatted = None
        try:
            user_only_formatted = _format_messages(tokenizer, [user_prompt])
        except Exception:
            pass

        if (user_only_formatted is not None
                and system_prefix not in (_extract_prefix_suffix(
                    user_only_formatted, user_prompt.content)[0] or "")):
            # User-only formatting is clean (no system injection).
            user_prefix, user_suffix = _extract_prefix_suffix(
                user_only_formatted, user_prompt.content)
        elif system_formatted is not None:
            # Fall back: original approach using len(system_formatted) offset.
            # This works for models where the tokenizer does NOT add single-
            # message EOS to system_formatted (e.g. Qwen, InternVL).
            user_prefix, user_suffix = _extract_prefix_suffix(
                user_formatted[len(system_formatted):], user_prompt.content)
        else:
            # Neither system-only nor user-only worked.  Extract user
            # prefix/suffix by locating the user placeholder after the
            # system placeholder in the combined result.
            sys_end = user_formatted.find(system_prompt.content) + len(
                system_prompt.content)
            user_prefix, user_suffix = _extract_prefix_suffix(
                user_formatted[sys_end:], user_prompt.content)

        if user_prefix and user_prefix in system_suffix:
            system_suffix = system_suffix[:system_suffix.find(user_prefix)]
        elif not user_prefix and system_suffix:
            for length in range(1, len(system_suffix) // 2 + 1):
                candidate = system_suffix[:length]
                if system_suffix.endswith(candidate) and len(
                        system_suffix) > 2 * len(candidate):
                    user_prefix = system_suffix[length:-length]
                    user_suffix = candidate
                    system_suffix = candidate
                    break

        assistant_prompt = AssistantMessage()
        assistant_formatted = _format_messages(
            tokenizer, [system_prompt, user_prompt, assistant_prompt])
        assistant_prefix, assistant_suffix = _extract_prefix_suffix(
            assistant_formatted[len(user_formatted):],
            assistant_prompt.content)

        generation_formatted = _format_messages(tokenizer,
                                                [system_prompt, user_prompt],
                                                add_generation_prompt=True,
                                                enable_thinking=False)
        # Standard case: generation_formatted = user_formatted + gen_prompt.
        generation_prompt = generation_formatted[len(user_formatted):]
        if not generation_prompt and generation_formatted != user_formatted:
            # Some tokenizers (e.g. Phi-4MM) REPLACE the trailing EOS token
            # with the assistant start token rather than appending, so both
            # strings have the same length but differ at the end.  Find the
            # longest common prefix and treat the diverging suffix as the
            # generation prompt.
            common_len = 0
            for i in range(min(len(user_formatted),
                               len(generation_formatted))):
                if user_formatted[i] != generation_formatted[i]:
                    break
                common_len = i + 1
            generation_prompt = generation_formatted[common_len:]

        if _is_alpamayo_1_model(model_dir):
            logger.info("Detected Alpamayo 1 model, adding <|cot_start|> to "
                        "generation prompt")
            generation_prompt = generation_prompt + "<|cot_start|>"

        generation_prompt_thinking = None
        try:
            thinking_formatted = _format_messages(tokenizer,
                                                  [system_prompt, user_prompt],
                                                  add_generation_prompt=True)
            gpt = thinking_formatted[len(user_formatted):]
            if gpt != generation_prompt:
                generation_prompt_thinking = gpt
        except (TypeError, ValueError, KeyError):
            pass

        # Qwen3-Omni override: force both fields to the no-injection variant.
        #
        # Qwen3-Omni Instruct is RLHF'd to never emit ``<think>...</think>``
        # tokens, so the chat template's ``enable_thinking=False`` branch —
        # which prepends ``<think>\n\n</think>\n\n`` to the prompt — provides
        # no semantic benefit.  Worse, the prepended tokens shift the
        # Talker's hardcoded slicing in
        # ``_get_talker_assistant_parts``: positions ``[:, :3]`` /
        # ``[:, 3:4]`` / ``[:, 4:]`` assume the first generated token sits at
        # slice index 3, but the injected ``<think>`` token occupies that
        # slot, breaking Talker prefill alignment (audio tail with junk codec
        # frames; talker hits ``max_new_tokens``).
        #
        # Picking the no-injection variant for ``generation_prompt`` and
        # leaving ``generation_prompt_thinking`` empty makes the C++ runtime
        # fall back to ``generation_prompt`` regardless of the
        # ``enableThinking`` flag (see tokenizer.h ChatTemplateConfig).
        # Result: Qwen3-Omni is immune to the flag at any layer of the stack.
        if _is_qwen3_omni_model(model_dir):
            if generation_prompt_thinking is not None:
                generation_prompt = generation_prompt_thinking
            generation_prompt_thinking = None

        content_types: Dict[str, Any] = {}
        if _is_phi4mm_model(model_dir):
            # Phi-4MM uses <|endoftext10|> (token ID 200010) as image placeholder.
            # The C++ Phi4MMViTRunner hardcodes imageTokenId=200010 and looks for
            # that exact token in the tokenized input to locate image positions.
            content_types = {
                "image": {
                    "format": "<|endoftext10|>"
                },
            }
        elif _is_nemotron_omni_model(model_dir):
            # Nemotron-Omni's HF chat template dumps a Python repr of the
            # content list instead of expanding it.  Emit one real placeholder
            # per item; NemotronOmniViTRunner / NemotronOmniAudioRunner repeat
            # each to the encoder's output length at textPreprocess time.
            content_types = {
                "image": {
                    "format": "<img><image></img>"
                },
                "audio": {
                    "format": "<so_embedding>"
                },
            }
        elif is_vlm:
            user_text_only = MultimodalUserMessage()
            text_only_formatted = _format_messages(
                tokenizer, [system_prompt, user_text_only])
            placeholder_text = user_text_only.content[0]["text"]
            for ctype, cplaceholder in [
                ("image", "<placeholder_image_path>"),
                ("video", "<placeholder_video_path>"),
            ]:
                pattern = _extract_content_pattern(tokenizer, system_prompt,
                                                   ctype, cplaceholder,
                                                   text_only_formatted,
                                                   placeholder_text)
                if pattern:
                    content_types[ctype] = {"format": pattern}
        elif _is_qwen3_omni_model(model_dir) or _is_qwen3_asr_model(model_dir):
            content_types = {
                "audio": {
                    "format": "<|audio_pad|>"
                },
                "image": {
                    "format": "<|image_pad|>"
                },
                "video": {
                    "format": "<|video_pad|>"
                },
            }

        default_system_prompt = ""
        user_only_formatted = _format_messages(tokenizer, [UserMessage()])
        system_start = user_only_formatted.find(system_prefix)
        if system_start != -1:
            content_start = system_start + len(system_prefix)
            content_end = user_only_formatted.find(system_suffix,
                                                   content_start)
            if content_end != -1:
                candidate = user_only_formatted[content_start:content_end]
                if candidate != system_prompt.content:
                    if candidate:
                        default_system_prompt = candidate
                    else:
                        # The template always emits an empty system block even when the user
                        # provides no system message.  The C++ runtime only injects the system
                        # block when default_system_prompt is non-empty, so there is no clean
                        # way to add a truly empty system block through the default_system_prompt
                        # field.  Bake the empty system block into the user prefix instead:
                        #   <|im_start|>system\n<|im_end|>\n<|im_start|>user\n
                        # This produces the EXACT expected token sequence for the common case
                        # where no explicit system message is provided (e.g. the sweep test).
                        user_prefix = system_prefix + system_suffix + user_prefix

        data: Dict[str, Any] = {
            "model_path": model_dir,
            "roles": {
                "system": {
                    "prefix": system_prefix,
                    "suffix": system_suffix
                },
                "user": {
                    "prefix": user_prefix,
                    "suffix": user_suffix
                },
                "assistant": {
                    "prefix": assistant_prefix,
                    "suffix": assistant_suffix
                },
            },
            "content_types": content_types,
            "generation_prompt": generation_prompt,
            "default_system_prompt": default_system_prompt,
        }
        if generation_prompt_thinking is not None:
            data["generation_prompt_thinking"] = generation_prompt_thinking

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "processed_chat_template.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Chat template saved to %s", out_path)

    except Exception as e:
        logger.warning(
            "Jinja template extraction failed for %s (%s); trying hardcoded fallback",
            model_dir, e)
        if not _try_write_hardcoded_template(model_dir, output_dir):
            logger.warning(
                "No hardcoded template for model_type=%r; chat template skipped",
                _get_model_type(model_dir))
