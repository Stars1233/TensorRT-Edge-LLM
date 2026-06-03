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
"""FP8 embedding quantization for tensorrt_edgellm runtime sidecars."""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
FP8_EMBEDDING_BLOCK_SIZE = 128


def quantize_embedding_to_fp8(
    embedding_weight: torch.Tensor,
    block_size: int = FP8_EMBEDDING_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an embedding table to FP8 E4M3 with per-row block scales."""
    if embedding_weight.dim() != 2:
        raise ValueError(
            f"Embedding must be 2D, got {embedding_weight.dim()}D")

    vocab_size, hidden_size = embedding_weight.shape
    if hidden_size % block_size != 0:
        raise ValueError(
            f"Hidden size {hidden_size} must be divisible by block size {block_size}"
        )

    num_groups = hidden_size // block_size
    weight_fp32 = embedding_weight.float()
    weight_reshaped = weight_fp32.view(vocab_size, num_groups, block_size)
    amax = weight_reshaped.abs().amax(dim=-1).clamp(min=1e-4)
    scales = amax / FP8_E4M3_MAX
    quantized = weight_reshaped / scales.unsqueeze(-1)
    quantized = quantized.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    embedding_fp8 = quantized.view(vocab_size,
                                   hidden_size).to(torch.float8_e4m3fn)

    logger.info("Quantized embedding to FP8: [%d, %d], scales: [%d, %d]",
                vocab_size, hidden_size, vocab_size, num_groups)
    return embedding_fp8, scales
