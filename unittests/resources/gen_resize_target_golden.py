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
"""Generate golden tables for unittests/resizeTargetTests.cpp.

Prints C++ initializer rows for the three resize-target functions in cpp/multimodal/imageUtils.h:

- qwenSmartResize (2D): goldens come from the HF REFERENCE implementation
  (transformers qwen2_vl image_processing smart_resize) — true HF-parity goldens.
- qwenSmartResize3D: goldens come from the HF REFERENCE implementation
  (transformers qwen3_vl video_processing smart_resize) — true HF-parity goldens.
- gemma4ResizeTarget: values are REGRESSION PINS computed by an exact Python replica of the C++
  math (std::round == floor(x+0.5) for positive values), NOT an HF reference.

Run from the repo root with a venv that has transformers >= 5.x installed:
    ./venv/bin/python unittests/resources/gen_resize_target_golden.py
"""

import math

from transformers.models.qwen2_vl.image_processing_qwen2_vl import \
    smart_resize as smart_resize_2d
from transformers.models.qwen3_vl.video_processing_qwen3_vl import \
    smart_resize as smart_resize_3d


def gen_qwen_2d():
    # Qwen2/2.5-VL-style config: factor = patchSize(14) * mergeSize(2) = 28,
    # builder defaults minImageTokens=4, maxImageTokensPerImage=512.
    factor = 28
    min_pixels, max_pixels = 4 * factor * factor, 512 * factor * factor
    # NOTE: inputs with a dimension < factor/2 (e.g. 14x2800) are excluded: HF's round() yields 0
    # there and takes the min-pixels rescale branch, while the C++ clamps the rounded dim to
    # >= factor first — a pinned, pre-existing divergence (see
    # QwenSmartResizeTest.SubFactorDimDivergesFromHFByDesign).
    cases = [
        (64, 64),
        (42, 42),
        (70, 70),
        (224, 224),
        (448, 644),
        (1080, 1920),
        (360, 480),
        (28, 5600),
        (812, 1092),
        (1024, 1024),
        (700, 700),
        (56, 84),
    ]
    print(
        "// QwenSmartResizeTest.MatchesHFReferenceGoldens {h, w, hBar, wBar}")
    for h, w in cases:
        h_bar, w_bar = smart_resize_2d(h,
                                       w,
                                       factor=factor,
                                       min_pixels=min_pixels,
                                       max_pixels=max_pixels)
        # The target must be a fixed point (idempotency backs the pre-resize workflow).
        assert smart_resize_2d(h_bar, w_bar, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels) \
            == (h_bar, w_bar), (h, w)
        print(f"    {{{h}, {w}, {h_bar}, {w_bar}}},")


def gen_qwen_3d():
    # Qwen3-VL-style config: factor = patchSize(16) * mergeSize(2) = 32, temporalPatchSize=2,
    # minImageTokens=4, maxImageTokensPerImage=6144 (HF default).
    factor, temporal_factor = 32, 2
    # Token bounds -> 3D pixel budget must carry the temporal factor:
    # tokens = t_bar*h*w / (temporal_factor * factor^2), matching the C++.
    min_pixels = 4 * temporal_factor * factor * factor
    max_pixels = 6144 * temporal_factor * factor * factor
    cases = [
        (1, 64, 64),
        (2, 224, 224),
        (3, 224, 224),
        (4, 224, 224),
        (8, 512, 512),
        (16, 720, 1280),
        (3, 1080, 1920),
        (2, 480, 360),
        (5, 256, 256),
        (1, 1024, 1024),
        (32, 224, 224),
        (2, 800, 600),
        (7, 96, 96),
        (2, 2048, 2048),
    ]
    print(
        "// Qwen3VLSmartResize3DTest.MatchesHFReferenceGoldens {t, h, w, hBar, wBar}"
    )
    for t, h, w in cases:
        h_bar, w_bar = smart_resize_3d(num_frames=t,
                                       height=h,
                                       width=w,
                                       temporal_factor=temporal_factor,
                                       factor=factor,
                                       min_pixels=min_pixels,
                                       max_pixels=max_pixels)
        print(f"    {{{t}, {h}, {w}, {h_bar}, {w_bar}}},")


def gemma4_replica(h, w, max_tok, pool, patch):
    """Exact replica of gemma4ResizeTarget in cpp/multimodal/imageUtils.cpp."""
    max_patches = max_tok * pool * pool
    total_px = float(h) * float(w)
    target_px = float(max_patches) * patch * patch
    factor = math.sqrt(target_px / total_px)
    ideal_h, ideal_w = factor * h, factor * w
    side = pool * patch
    fl = lambda v: int(math.floor(v / side)) * side
    rd = lambda v: int(math.floor(v / side + 0.5)
                       ) * side  # std::round for positive values
    th, tw = fl(ideal_h), fl(ideal_w)
    assert th != 0 or tw != 0
    max_side = (max_patches // (pool * pool)) * side
    if th == 0:
        th = side
        tw = min(max(rd(ideal_w), side), min(max_side, fl(target_px / th)))
    elif tw == 0:
        tw = side
        th = min(max(rd(ideal_h), side), min(max_side, fl(target_px / tw)))
    assert th * tw <= target_px
    return th, tw


def gen_gemma4():
    max_tok, pool, patch = 256, 4, 14
    cases = [
        (224, 224),
        (448, 644),
        (1080, 1920),
        (64, 64),
        (100, 6000),
        (6000, 100),
        (896, 896),
        (500, 700),
    ]
    print(
        "// Gemma4ResizeTargetTest.MatchesPinnedGoldens {h, w, targetH, targetW}"
    )
    for h, w in cases:
        th, tw = gemma4_replica(h, w, max_tok, pool, patch)
        print(f"    {{{h}, {w}, {th}, {tw}}},")


def gemma4_unified_replica(h, w, max_patches, patch, posemb):
    """Exact replica of gemma4UnifiedResizeTarget in cpp/multimodal/imageUtils.cpp."""
    assert h > 0 and w > 0
    target_px = float(max_patches) * patch * patch
    scale = math.sqrt(target_px / (float(h) * w))
    ih, iw = scale * h, scale * w
    fl = lambda v: int(math.floor(v / patch)) * patch
    max_side = min(max_patches, posemb) * patch
    th, tw = min(fl(ih), max_side), min(fl(iw), max_side)
    assert th != 0 or tw != 0
    if th == 0:
        th = patch
        tw = min(int(math.floor(float(w) / h)) * patch, max_side)
    elif tw == 0:
        tw = patch
        th = min(int(math.floor(float(h) / w)) * patch, max_side)
    assert th * tw <= target_px
    return th, tw


def gen_gemma4_unified():
    max_patches, patch, posemb = 256, 48, 64
    cases = [
        (224, 224),
        (448, 644),
        (1080, 1920),
        (64, 64),
        (100, 6000),
        (768, 768),
        (500, 700),
        (48, 4800),
    ]
    print(
        "// Gemma4UnifiedResizeTargetTest.MatchesPinnedGoldens {h, w, targetH, targetW}"
    )
    for h, w in cases:
        th, tw = gemma4_unified_replica(h, w, max_patches, patch, posemb)
        print(f"    {{{h}, {w}, {th}, {tw}}},")


if __name__ == "__main__":
    gen_qwen_2d()
    gen_qwen_3d()
    gen_gemma4()
    gen_gemma4_unified()
