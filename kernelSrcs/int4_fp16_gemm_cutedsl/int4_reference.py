# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Torch host-side INT4 weight helpers for the standalone correctness check.

This module is used ONLY by ``int4_fp16_gemm_ampere.py``'s reference path
(``--skip_ref_check`` omitted); the ``--export_only`` AOT path never imports it,
so Torch is not required to build the kernel library.

Contract (the repo-canonical W4A16 layout):

    C[M, N] = A[M, K] @ dequant(QW[ceil(N/2), K], scales[ceil(K/G), N])^T

  * A       : FP16 row-major [M, K]
  * QW      : UINT8 row-major [ceil(N/2), K], low nibble for even N, high for odd
  * scales  : FP16 row-major [ceil(K/G), N]
  * C       : FP16 row-major [M, N]

``repack_b_for_tile`` produces the fragment-order uint32 buffer the kernel's B
mainloop consumes (the nibble swizzle is done once here on the host so the
in-kernel staging is a single coalesced cp.async).
"""

from __future__ import annotations

import math

import torch

_THREADS = 128
# (bit position, N-offset selector "lo"/"hi", K offset) for the eight nibbles
# of one repacked 32-bit word -- see repack_b_for_tile's docstring.
_NIBBLES = [
    (0, "lo", 0), (4, "lo", 8), (8, "hi", 0), (12, "hi", 8),
    (16, "lo", 1), (20, "lo", 9), (24, "hi", 1), (28, "hi", 9),
]


def _normalize_dim(dim: int, rank: int) -> int:
    if not -rank <= dim < rank:
        raise IndexError(f"dim {dim} out of range for rank {rank}")
    return dim % rank


def pack_signed_int4(q_signed: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
    """Pack signed int4 values along one dimension, low nibble first.

    The repo-canonical layout uses dim=0: logical weights [N, K] become packed
    weights [ceil(N/2), K], matching the CUDA AWQ convention.
    """
    if q_signed.dtype != torch.int8:
        raise TypeError("q_signed must be torch.int8")

    pack_dim = _normalize_dim(dim, q_signed.dim())
    if pack_dim != q_signed.dim() - 1:
        q_signed = q_signed.movedim(pack_dim, -1)

    *leading, packed_extent = q_signed.shape
    if packed_extent % 2:
        pad = torch.zeros((*leading, 1), device=q_signed.device, dtype=q_signed.dtype)
        q_signed = torch.cat([q_signed, pad], dim=-1)

    encoded = q_signed.to(torch.int16) & 0xF
    low = encoded[..., 0::2]
    high = encoded[..., 1::2] << 4
    packed = (low | high).to(torch.uint8)
    if pack_dim != packed.dim() - 1:
        packed = packed.movedim(-1, pack_dim)
    return packed.contiguous()


def unpack_signed_int4(qweight: torch.Tensor, unpacked_size: int, *, dim: int = 0) -> torch.Tensor:
    """Unpack low-nibble-first signed int4 values from a uint8 tensor."""
    unpack_dim = _normalize_dim(dim, qweight.dim())
    if unpack_dim != qweight.dim() - 1:
        qweight = qweight.movedim(unpack_dim, -1)

    packed = qweight.to(torch.int16)
    low = packed & 0xF
    high = (packed >> 4) & 0xF

    unpacked = torch.empty(
        (*qweight.shape[:-1], qweight.shape[-1] * 2),
        device=qweight.device,
        dtype=torch.int16,
    )
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    unpacked = unpacked[..., :unpacked_size]
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked).to(torch.int8)
    if unpack_dim != unpacked.dim() - 1:
        unpacked = unpacked.movedim(-1, unpack_dim)
    return unpacked.contiguous()


def int4_fp16_reference(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Reference for QW[ceil(N/2), K] and scales[ceil(K/G), N]; FP32 accumulate."""
    k = activations.shape[1]
    n = scales.shape[1]
    q_signed = unpack_signed_int4(qweight, n, dim=0).to(torch.float32)
    scale_rows = torch.arange(k, device=scales.device) // group_size
    scale_per_weight = scales[scale_rows, :].t().to(torch.float32)
    weights = q_signed * scale_per_weight
    return (activations.to(torch.float32) @ weights.t()).to(torch.float16)


def make_int4_inputs(
    m: int,
    n: int,
    k: int,
    *,
    group_size: int,
    device: str | torch.device = "cuda",
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic inputs using the repo-canonical packed layout."""
    torch.manual_seed(seed + m + n + k)
    activations = torch.empty((m, k), device=device, dtype=torch.float16).uniform_(-1.0, 1.0)
    q_signed = torch.randint(-8, 8, (n, k), device=device, dtype=torch.int8)
    qweight = pack_signed_int4(q_signed, dim=0)
    scales = torch.empty(
        (math.ceil(k / group_size), n),
        device=device,
        dtype=torch.float16,
    ).uniform_(0.01, 0.2)
    return activations, qweight, scales


def repack_b_for_tile(qweight: torch.Tensor, n: int, k: int, bN: int, bK: int) -> torch.Tensor:
    """Repack canonical ``QW[ceil(N/2), K]`` into fragment-order uint32 words.

    Returns an ``int32`` tensor of shape ``[num_n_blocks * num_k_tiles * kn,
    128]`` (the bit pattern is the uint32 word; int32 is used because torch
    indexing/shifts are int-typed, and the kernel reads it back as ``Uint32``).

    For N-block ``nb``, K-tile ``kt``, word index ``idx = kbl * n_pairs + p`` and
    thread ``t``, the eight biased nibbles (signed + 8) are placed as

        bits  0-3 (Nlo,K0)   4-7 (Nlo,K0+8)   8-11 (Nhi,K0)   12-15 (Nhi,K0+8)
        bits 16-19(Nlo,K0+1) 20-23(Nlo,K0+9) 24-27 (Nhi,K0+1) 28-31 (Nhi,K0+9)

    where ``n_base = t//4``, ``kb = 2*(t%4)``, ``K0 = kt*bK + 16*kbl + kb``,
    ``Nlo = nb*bN + n_base + 64*p``, ``Nhi = Nlo + 32``.  Out-of-range (padding)
    nibbles are biased ``8`` so they dequantize to 0.
    """
    if qweight.dtype != torch.uint8:
        raise TypeError("qweight must be uint8")
    if bN % 64 != 0 or bK % 16 != 0:
        raise ValueError("repacked tile requires bN%64==0 and bK%16==0")
    k_blocks = bK // 16
    n_pairs = bN // 64
    kn = k_blocks * n_pairs
    num_n_blocks = (n + bN - 1) // bN
    num_k_tiles = (k + bK - 1) // bK
    n_pad, k_pad = num_n_blocks * bN, num_k_tiles * bK

    # Biased nibbles (signed + 8), padded with 8 (-> dequant 0) on both axes.
    q_signed = unpack_signed_int4(qweight.detach().cpu(), n, dim=0).to(torch.int64)
    b = torch.full((n_pad, k_pad), 8, dtype=torch.int64)
    b[:n, :k] = (q_signed + 8) & 0xF

    nb = torch.arange(num_n_blocks).view(-1, 1)   # (num_n_blocks, 1)
    kt = torch.arange(num_k_tiles).view(1, -1)    # (1, num_k_tiles)
    out = torch.zeros((num_n_blocks * num_k_tiles * kn, _THREADS), dtype=torch.int64)

    for t in range(_THREADS):
        n_base = t // 4
        kb = 2 * (t % 4)
        for kbl in range(k_blocks):
            for p in range(n_pairs):
                idx = kbl * n_pairs + p
                n_lo = nb * bN + (n_base + 64 * p)        # (num_n_blocks, 1)
                n_hi = n_lo + 32
                k0 = kt * bK + (16 * kbl + kb)            # (1, num_k_tiles)
                word = torch.zeros((num_n_blocks, num_k_tiles), dtype=torch.int64)
                for shift, sel, koff in _NIBBLES:
                    nrow = (n_lo if sel == "lo" else n_hi).expand(num_n_blocks, num_k_tiles)
                    kcol = (k0 + koff).expand(num_n_blocks, num_k_tiles)
                    word |= b[nrow, kcol] << shift
                rows = ((nb * num_k_tiles + kt) * kn + idx).reshape(-1)  # (nb*kt,)
                out[rows, t] = word.reshape(-1)
    return out.to(torch.int32).contiguous()
