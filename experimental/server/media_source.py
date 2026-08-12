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
"""Shared OpenAI media-source resolution: ``data:`` base64 payloads and
``file://`` URLs, with per-modality strictness/size policies supplied by the
caller (audio keeps lenient decoding; video decodes strictly with a size
cap)."""
from __future__ import annotations

import base64
import binascii
from typing import Optional
from urllib.parse import unquote, urlparse


def decode_base64_payload(payload: str,
                          kind: str,
                          *,
                          strict: bool = False,
                          max_bytes: Optional[int] = None) -> bytes:
    """Decode a raw base64 payload to bytes. ``max_bytes`` is enforced exactly
    on the decoded size; an encoded-length pre-check rejects clearly oversized
    payloads first (sized so base64 padding never mis-rejects)."""
    if max_bytes is not None and len(payload) > -(-max_bytes // 3) * 4:
        raise ValueError(f"{kind} payload exceeds the supported "
                         f"maximum of {max_bytes} bytes")
    try:
        data = base64.b64decode(payload, validate=strict)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 {kind} payload: {exc}") from exc
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError(f"{kind} payload exceeds the supported "
                         f"maximum of {max_bytes} bytes")
    return data


def decode_base64_data_url(url: str,
                           kind: str,
                           *,
                           strict: bool = False,
                           max_bytes: Optional[int] = None) -> bytes:
    """Decode a ``data:<kind>/...;base64,<payload>`` URL to bytes (size policy
    per :func:`decode_base64_payload`)."""
    head, _, payload = url.partition(",")
    if ";base64" not in head:
        raise ValueError(f"data: {kind} URLs must use base64 encoding")
    return decode_base64_payload(payload,
                                 f"data: {kind}",
                                 strict=strict,
                                 max_bytes=max_bytes)


def resolve_file_url(url: str) -> str:
    """Convert a local ``file://`` URL to a filesystem path; non-local hosts
    and empty paths are client errors."""
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc not in ("", "localhost"):
        raise ValueError(
            f"file:// URL must be local (no host); got netloc={parsed.netloc!r}"
        )
    path = unquote(parsed.path)
    if not path:
        raise ValueError("file:// URL has empty path")
    return path


#: Content-item keys carrying a media reference. Which spelling each loader
#: accepts differs (only ``video`` takes both a bare string and ``{"url": ...}``),
#: so the policy reads every key both ways rather than tracking the difference.
MEDIA_REF_KEYS = ("image", "video", "audio", "image_url", "video_url",
                  "audio_url")


def iter_item_media_refs(item: dict):
    """Yield every media URL/path in one content item, in either spelling."""
    if not isinstance(item, dict):
        return
    for key in MEDIA_REF_KEYS:
        ref = item.get(key)
        url = ref.get("url") if isinstance(ref, dict) else ref
        if isinstance(url, str):
            yield url
    # {"type": "video", "frames": [path, ...]} feeds video_sampling directly.
    for frame in item.get("frames") or []:
        if isinstance(frame, str):
            yield frame
