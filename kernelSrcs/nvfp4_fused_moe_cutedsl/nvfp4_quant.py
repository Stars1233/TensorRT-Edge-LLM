# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python NVFP4 tensor packing utilities shared by CuTeDSL MoE tooling."""

from __future__ import annotations

from typing import Tuple
import warnings

try:
    import torch
except ImportError as e:
    _TORCH_IMPORT_ERROR = e
    warnings.warn(
        "PyTorch is not available; importing nvfp4_quant.py succeeded, "
        "but NVFP4 helpers require PyTorch and will fail if called.",
        RuntimeWarning,
        stacklevel=2,
    )

    class _MissingTorch:
        def __getattr__(self, name):
            raise ImportError(
                "PyTorch is required to use nvfp4_quant.py helpers."
            ) from _TORCH_IMPORT_ERROR

    torch = _MissingTorch()
else:
    _TORCH_IMPORT_ERROR = None


# Pure-PyTorch NVFP4 (E2M1 + FP8 E4M3 block scale) quantize/dequantize.
_FP4_E2M1_MAG_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX = 6.0


def _fp4_e2m1_mag_tensor(device):
    return torch.tensor(_FP4_E2M1_MAG_VALUES,
                        dtype=torch.float32,
                        device=device)


def _quantize_e2m1_nibbles(values_blockscaled: "torch.Tensor") -> "torch.Tensor":
    """Quantize values already scaled into [-6, 6] to E2M1 nibbles."""
    v = values_blockscaled.clamp(-_FP4_MAX, _FP4_MAX)
    sign = (v < 0).to(torch.uint8) * 8
    mag = v.abs()
    levels = _fp4_e2m1_mag_tensor(values_blockscaled.device)
    dists = (mag.unsqueeze(-1) - levels).abs()
    mag_code = dists.argmin(dim=-1).to(torch.uint8)
    return sign | mag_code


def _pack_nibbles_2per_byte(nibbles: "torch.Tensor") -> "torch.Tensor":
    """Pack uint8 nibbles into bytes, low nibble first."""
    low = nibbles[..., 0::2] & 0x0F
    high = (nibbles[..., 1::2] & 0x0F) << 4
    return low | high


def convert_swizzled_to_linear(a_sf_swizzled: "torch.Tensor", m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = a_sf_swizzled.reshape(1, m_tiles, k_tiles, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k // block_size]


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()

    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    e2m1_values = torch.tensor(_FP4_E2M1_MAG_VALUES,
                               dtype=torch.float32,
                               device=a.device)
    values = e2m1_values[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def dequantize_nvfp4_to_dtype(
    tensor_fp4,
    tensor_sf,
    global_scale,
    dtype,
    device,
    block_size=16,
):
    """Dequantize an NVFP4 tensor back to the requested dtype."""
    del device
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def fp4_quantize(
    x: "torch.Tensor",
    global_scale: "torch.Tensor",
    sf_vec_size: int = 16,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """NVFP4 block quantize.

    Returns ``(fp4_packed, sf)`` where ``sf`` is a uint8 buffer in either
    linear or swizzled physical layout per ``is_sf_swizzled_layout``.
    """
    if sf_vec_size != 16:
        raise NotImplementedError("Only sf_vec_size=16 supported.")
    if sf_use_ue8m0:
        raise NotImplementedError("Only ufp8 (E4M3) scale factors supported.")
    if x.dim() != 2:
        raise ValueError(f"fp4_quantize expects 2D tensor, got {tuple(x.shape)}")

    m, k = x.shape
    if k % sf_vec_size != 0:
        raise ValueError(f"K ({k}) must be a multiple of sf_vec_size ({sf_vec_size})")

    sf_k = k // sf_vec_size
    x_blocks = x.float().view(m, sf_k, sf_vec_size)
    amax = x_blocks.abs().amax(dim=-1)
    block_scale_f32 = amax / _FP4_MAX

    gs = global_scale.view(-1)[0].float().to(x.device)
    sf_stored_f32 = (block_scale_f32 * gs).clamp_min(0.0)
    sf_stored_fp8 = sf_stored_f32.to(torch.float8_e4m3fn)
    sf_linear = sf_stored_fp8.view(torch.uint8)

    block_scale_bc = block_scale_f32.unsqueeze(-1).clamp_min(1e-12)
    normed = x_blocks / block_scale_bc
    nibbles = _quantize_e2m1_nibbles(normed).view(m, k)
    packed = _pack_nibbles_2per_byte(nibbles)

    if not is_sf_swizzled_layout:
        return packed, sf_linear

    m_tiles = (m + 127) // 128
    k_tiles = (sf_k + 3) // 4
    padded_m = m_tiles * 128
    padded_k_sf = k_tiles * 4
    sf_padded = torch.zeros(padded_m,
                            padded_k_sf,
                            dtype=torch.uint8,
                            device=x.device)
    sf_padded[:m, :sf_k] = sf_linear

    sf_5d = sf_padded.view(m_tiles, 4, 32, k_tiles, 4)
    sf_swizzled = sf_5d.permute(0, 3, 2, 1, 4).contiguous()
    return packed, sf_swizzled.view(-1)


def e2m1_and_ufp8sf_scale_to_float(
    e2m1_tensor,
    ufp8_scale_tensor,
    global_scale_tensor,
    sf_vec_size: int = 16,
    ufp8_type: int = 1,
    is_sf_swizzled_layout: bool = True,
    m: int = None,
    k: int = None,
):
    if sf_vec_size != 16:
        raise NotImplementedError("Only sf_vec_size=16 supported.")
    if ufp8_type != 1:
        raise NotImplementedError("Only ufp8_type=1 (FP8 E4M3) supported.")

    if m is not None and k is not None and e2m1_tensor.dim() == 1:
        e2m1_tensor = e2m1_tensor.reshape(m, k // 2)
    m_dim, packed_k = e2m1_tensor.shape
    k_dim = packed_k * 2

    gs = 1.0 / global_scale_tensor.view(-1)[0].float().to(
        e2m1_tensor.device).clamp_min(1e-12)

    if is_sf_swizzled_layout:
        sf_for_upstream = ufp8_scale_tensor
    else:
        sf_k = k_dim // sf_vec_size
        sf_linear = ufp8_scale_tensor.view(m_dim, sf_k)
        m_tiles = (m_dim + 127) // 128
        k_tiles = (sf_k + 3) // 4
        padded_m = m_tiles * 128
        padded_k_sf = k_tiles * 4
        sf_padded = torch.zeros(padded_m,
                                padded_k_sf,
                                dtype=torch.uint8,
                                device=sf_linear.device)
        sf_padded[:m_dim, :sf_k] = sf_linear
        sf_5d = sf_padded.view(m_tiles, 4, 32, k_tiles, 4)
        sf_for_upstream = sf_5d.permute(0, 3, 2, 1, 4).contiguous().view(-1)

    return dequantize_nvfp4_to_dtype(
        e2m1_tensor,
        sf_for_upstream,
        gs,
        dtype=torch.float32,
        device=e2m1_tensor.device,
        block_size=sf_vec_size,
    )


__all__ = [
    "break_fp4_bytes",
    "convert_swizzled_to_linear",
    "dequantize_nvfp4_to_dtype",
    "e2m1_and_ufp8sf_scale_to_float",
    "fp4_quantize",
]
