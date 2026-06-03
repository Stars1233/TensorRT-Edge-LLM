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
From-scratch Nemotron-Omni RADIO vision encoder.

Architecture:
    RADIOEmbeddings
        register tokens [10, H] (position-free)
        Conv2d patch embedding  +  pos_embed [1, N, H] (bilinear interpolated)
    → 32 × RADIOBlock
        norm1 → attn(fused qkv, proj) → residual
        norm2 → mlp(fc1, GELU, fc2) → residual
    → remove register tokens
    → pixel_shuffle downsampling (downsample_ratio=0.5)
    → mlp1 projector: RMSNorm → Linear → SquaredReLU → Linear

Checkpoint weight key prefixes:
    Vision encoder:  ``vision_model.radio_model.model.*``
    Projector:       ``mlp1.*``
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..linear import make_linear

if TYPE_CHECKING:
    from ...config import ModelConfig

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(input_dtype) * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class RADIOEmbeddings(nn.Module):
    """Register tokens + Conv2d patch embedding + learnable position embedding.

    RADIO uses 10 register tokens (not a single CLS). The patch embedder stores
    weight as flat [hidden_size, 3*patch_size*patch_size] and is reshaped to
    Conv2d format during weight loading. Position embedding is stored at max
    resolution and sliced to match the actual image size.

    Checkpoint keys (under ``patch_generator.``):
        ``cls_token.token``   [num_registers, hidden_size]
        ``embedder.weight``   [hidden_size, 3*patch_size*patch_size]
        ``pos_embed``         [1, max_patches, hidden_size]
    """

    def __init__(self,
                 hidden_size: int,
                 patch_size: int,
                 image_size: int,
                 num_registers: int = 10) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size)**2
        self.num_registers = num_registers
        self.cls_token = nn.Module()
        self.cls_token.token = nn.Parameter(
            torch.zeros(num_registers, hidden_size))
        self.embedder = nn.Conv2d(3,
                                  hidden_size,
                                  kernel_size=patch_size,
                                  stride=patch_size,
                                  bias=False)
        # pos_embed covers patches only (no registers). Interpolated at load time.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.size(0)
        patch_embeds = self.embedder(
            pixel_values.to(self.embedder.weight.dtype)).flatten(2).transpose(
                1, 2)
        # Add position embedding to patches only (registers are position-free)
        patch_embeds = patch_embeds + self.pos_embed
        reg_tokens = self.cls_token.token.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat((reg_tokens, patch_embeds), dim=1)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class RADIOAttention(nn.Module):
    """Fused-QKV multi-head attention.

    Checkpoint keys (under ``blocks.N.attn.``):
        ``qkv.weight``, ``qkv.bias``  [3*hidden, hidden]
        ``proj.weight``, ``proj.bias`` [hidden, hidden]
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.embed_dim = hidden_size
        self.qkv = make_linear(
            model_config,
            hidden_size,
            hidden_size * 3,
            bias=True,
            module_name=f"{name_prefix}.qkv" if name_prefix else "")
        self.proj = make_linear(
            model_config,
            hidden_size,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.proj" if name_prefix else "")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads,
                                              self.head_dim).permute(
                                                  2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale,
                         dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(
            B, N, self.embed_dim)
        return self.proj(out)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class RADIOMLP(nn.Module):
    """Two-layer GELU FFN.

    Checkpoint keys: ``mlp.fc1.*``, ``mlp.fc2.*``
    """

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.fc1 = make_linear(
            model_config,
            hidden_size,
            intermediate_size,
            bias=True,
            module_name=f"{name_prefix}.fc1" if name_prefix else "")
        self.fc2 = make_linear(
            model_config,
            intermediate_size,
            hidden_size,
            bias=True,
            module_name=f"{name_prefix}.fc2" if name_prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class RADIOBlock(nn.Module):
    """Single RADIO ViT-H block (pre-norm, no layer scale).

    Checkpoint keys (under ``blocks.N.``):
        ``norm1.*``, ``attn.*``, ``norm2.*``, ``mlp.*``
    """

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = RADIOAttention(
            hidden_size,
            num_heads,
            model_config,
            name_prefix=f"{name_prefix}.attn" if name_prefix else "")
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = RADIOMLP(
            hidden_size,
            intermediate_size,
            model_config,
            name_prefix=f"{name_prefix}.mlp" if name_prefix else "")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class RADIOEncoder(nn.Module):
    """Stack of RADIOBlock layers.

    Checkpoint keys: ``blocks.N.*``
    """

    def __init__(self,
                 num_layers: int,
                 hidden_size: int,
                 num_heads: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            RADIOBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                layer_norm_eps,
                model_config,
                name_prefix=f"{name_prefix}.blocks.{i}" if name_prefix else "")
            for i in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Pixel shuffle
# ---------------------------------------------------------------------------


def _pixel_shuffle(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Spatial downsampling via pixel shuffle (merge spatial → channel).

    Matches the ps_version='v2' logic from the HF model.
    """
    B, H, W, C = x.shape
    scale = int(1.0 / scale_factor)
    new_H = H // scale
    new_W = W // scale
    x = x.view(B, new_H, scale, new_W, scale, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, new_H, new_W, scale * scale * C)
    return x


# ---------------------------------------------------------------------------
# Squared ReLU (used in mlp1 projector)
# ---------------------------------------------------------------------------


class _SquaredReLU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(F.relu(x), 2)


# ---------------------------------------------------------------------------
# Top-level visual model
# ---------------------------------------------------------------------------


class NemotronOmniVisualModel(nn.Module):
    """From-scratch Nemotron-Omni RADIO visual encoder + projector.

    Expects the full ``config.json`` dict (needs ``vision_config``,
    ``llm_config``, ``vit_hidden_size``, ``downsample_ratio``, etc.).

    Output: ``[total_patches * tokens_per_patch, llm_hidden_size]``
    """

    def __init__(self, config: dict, model_config: "ModelConfig") -> None:
        super().__init__()
        llm_cfg = config.get("llm_config", config.get("text_config", {}))
        llm_hidden_size: int = llm_cfg["hidden_size"]
        self.downsample_ratio: float = float(config["downsample_ratio"])

        image_size: int = config["force_image_size"]
        patch_size: int = config["patch_size"]
        hidden_size: int = config["vit_hidden_size"]
        projector_hidden_size: int = config["projector_hidden_size"]

        # ViT-H defaults: 32 layers, 16 heads, intermediate = hidden * 4
        vc = config.get("vision_config", {})
        num_layers = vc.get("num_hidden_layers", 32)
        num_heads = hidden_size // 80  # ViT-H: 1280 / 80 = 16 heads
        intermediate_size = vc.get("intermediate_size", hidden_size * 4)
        layer_norm_eps = 1e-6

        self.patch_generator = RADIOEmbeddings(hidden_size, patch_size,
                                               image_size)
        # Vision blocks live under ``vision_model.radio_model.model.blocks.N.*``
        # in the HF checkpoint; thread that prefix so MIXED_PRECISION
        # ``layer_overrides`` resolve and ``*vision_model*`` wildcards match.
        self.encoder = RADIOEncoder(
            num_layers,
            hidden_size,
            num_heads,
            intermediate_size,
            layer_norm_eps,
            model_config,
            name_prefix="vision_model.radio_model.model")

        # Projector: RMSNorm → Linear → SquaredReLU → Linear
        scale = int(1.0 / self.downsample_ratio)
        in_dim = hidden_size * scale * scale
        # mlp1.0 = RMSNorm, mlp1.1 = Linear, mlp1.2 = SquaredReLU, mlp1.3 = Linear
        self.mlp1 = nn.Sequential(
            _RMSNorm(in_dim),
            make_linear(model_config,
                        in_dim,
                        projector_hidden_size,
                        bias=False,
                        module_name="mlp1.1"),
            _SquaredReLU(),
            make_linear(model_config,
                        projector_hidden_size,
                        llm_hidden_size,
                        bias=False,
                        module_name="mlp1.3"),
        )
        self._llm_hidden_size = llm_hidden_size
        self._feat_side: int = image_size // patch_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [num_images, 3, image_size, image_size]

        Returns:
            image_features: [num_images, tokens_per_image, llm_hidden_size]
        """
        x = self.patch_generator(pixel_values)
        x = self.encoder(x)

        # Remove register tokens (RADIO uses 10 registers, not a single CLS)
        num_reg = self.patch_generator.num_registers
        x = x[:, num_reg:, :]

        B = x.shape[0]
        C = x.shape[2]
        feat_side = self._feat_side
        x = x.reshape(B, feat_side, feat_side, C)
        x = _pixel_shuffle(x, self.downsample_ratio)
        x = x.reshape(B, -1, x.shape[-1])
        x = self.mlp1(x)
        return x

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes)."""
        image_size = config["force_image_size"]
        pixel_values = torch.zeros(2,
                                   3,
                                   image_size,
                                   image_size,
                                   dtype=torch.float16,
                                   device=device)
        args = (pixel_values, )
        input_names = ["input"]
        output_names = ["output"]
        N = torch.export.Dim("num_images", min=1)
        dynamic_shapes = {"pixel_values": {0: N}}
        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_weights(model: NemotronOmniVisualModel, weights: dict) -> None:
    """Load RADIO vision encoder and mlp1 projector weights.

    Checkpoint key → model attribute path:
      ``vision_model.radio_model.model.patch_generator.*`` → ``patch_generator.*``
      ``vision_model.radio_model.model.blocks.*``          → ``encoder.blocks.*``
      ``mlp1.*``                                           → ``mlp1.*``

    Two patch_generator tensors need a per-tensor transform before assignment:

    * ``patch_generator.embedder.weight``: stored flat as ``[hidden, 3*P*P]``;
      reshape to Conv2d ``[hidden, 3, P, P]``.
    * ``patch_generator.pos_embed``: stored at max resolution (e.g. 128*128
      patches); bilinearly interpolate to the model's target resolution.
      RADIO pos_embed covers patches only (registers have no positional
      embedding).
    """
    from ...checkpoint.loader import load_submodule_weights

    vt_prefix = "vision_model.radio_model.model."

    def _remap(k: str) -> "str | None":
        if k.startswith(vt_prefix):
            inner = k[len(vt_prefix):]
            if inner.startswith("patch_generator.") or inner.startswith(
                    "blocks."):
                # blocks.* lives under encoder.* in our module tree.
                return ("encoder." +
                        inner) if inner.startswith("blocks.") else inner
            return None
        if k.startswith("mlp1."):
            return k
        return None

    def _transform(remapped_key: str, v: torch.Tensor) -> torch.Tensor:
        if remapped_key == "patch_generator.embedder.weight" and v.dim() == 2:
            ps = model.patch_generator.embedder.kernel_size[0]
            return v.reshape(v.shape[0], 3, ps, ps)
        if remapped_key == "patch_generator.pos_embed":
            full_pos = v  # [1, max_patches, hidden]
            target_patches = model.patch_generator.num_patches
            stored_patches = full_pos.shape[1]
            if stored_patches != target_patches:
                hidden = full_pos.shape[2]
                stored_side = int(math.sqrt(stored_patches))
                target_side = int(math.sqrt(target_patches))
                patch_pos = full_pos.reshape(1, stored_side, stored_side,
                                             hidden)
                patch_pos = patch_pos.permute(0, 3, 1, 2).float()
                patch_pos = F.interpolate(patch_pos,
                                          size=(target_side, target_side),
                                          mode="bilinear",
                                          align_corners=True).to(
                                              full_pos.dtype)
                patch_pos = patch_pos.permute(0, 2, 3, 1)
                return patch_pos.reshape(1, target_patches, hidden)
        return v

    load_submodule_weights(model,
                           weights,
                           _remap,
                           transform=_transform,
                           label="NemotronOmniVisualModel")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_nemotron_omni_visual(
        config: dict,
        weights: dict,
        model_config: "ModelConfig",
        dtype: torch.dtype = torch.float16) -> NemotronOmniVisualModel:
    """Build and return a :class:`NemotronOmniVisualModel` with loaded weights.

    Args:
        config:       Full parsed ``config.json`` dict.
        weights:      Flat ``{key: tensor}`` dict from safetensors.
        model_config: Top-level ``ModelConfig`` for quantized Linear dispatch.
        dtype:        Target dtype (default ``float16``).
    """
    model = NemotronOmniVisualModel(config, model_config=model_config)
    model.to(dtype)
    _load_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "NemotronOmniVisualModel",
    "build_nemotron_omni_visual",
]
