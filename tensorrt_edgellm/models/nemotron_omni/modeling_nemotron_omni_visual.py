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

The patch embedder is NOT part of the exported graph.  Nemotron-Omni uses a
different embedder for images (``Linear(3·P² → H)``) and videos
(``Linear(T·3·P² → H)``) while sharing every other weight, so the embedder
GEMM (plus the grid-dependent position-embedding add) is executed by the C++
runtime and the single exported model serves both paths.  The embedder /
video_embedder / raw pos_embed weights ship in a safetensors sidecar written
by :meth:`NemotronOmniVisualModel.save_onnx_sidecar`.

Exported graph:
    inputs: ``patch_embeds`` [B, N, H]  (embedder GEMM output + pos_embed)
            ``shuffle_indices`` [M, 4]  (pixel-shuffle gather indices for the
            B×N patch grid; grid-dependent, computed by the runtime)
    → prepend register tokens [10, H] (position-free)
    → 32 × RADIOBlock
        norm1 → attn(fused qkv, proj) → residual
        norm2 → mlp(fc1, GELU, fc2) → residual
    → remove register tokens
    → pixel_shuffle downsampling (downsample_ratio=0.5) as a Gather over
      ``shuffle_indices`` (supports the non-square patch grids produced by
      aspect-preserving video frame resize)
    → mlp1 projector: RMSNorm → Linear → SquaredReLU → Linear

Checkpoint weight key prefixes:
    Vision encoder:  ``vision_model.radio_model.model.*``
    Projector:       ``mlp1.*``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config as config_module
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
    """Weight container for the RADIO patch embedder and register tokens.

    Holds checkpoint weights only — no forward: the patch-embedder GEMM and the
    per-grid pos_embed interpolation run in the C++ runtime. The image embedder
    weight is flat [hidden_size, 3*patch_size*patch_size] (reshaped to Conv2d on
    load); ``video_embedder`` (when the checkpoint carries it) is a Linear over
    T stacked frames; the raw max-resolution pos_embed is kept for the runtime.

    Checkpoint keys (under ``patch_generator.``):
        ``cls_token.token``       [num_registers, hidden_size]
        ``embedder.weight``       [hidden_size, 3*patch_size*patch_size]
        ``video_embedder.weight`` [hidden_size, T*3*patch_size*patch_size]
        ``pos_embed``             [1, max_patches, hidden_size]
    """

    def __init__(self,
                 hidden_size: int,
                 patch_size: int,
                 image_size: int,
                 num_registers: int = 10,
                 video_temporal_patch_size: int = 0) -> None:
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
        # Video patch embedder: T temporally-stacked frames per patch.
        # Present only when the checkpoint carries video_embedder weights.
        self.video_temporal_patch_size = video_temporal_patch_size
        if video_temporal_patch_size > 0:
            self.video_embedder = nn.Linear(video_temporal_patch_size * 3 *
                                            patch_size * patch_size,
                                            hidden_size,
                                            bias=False)
        # Raw checkpoint pos_embed (max resolution) for the sidecar; the C++
        # runtime interpolates it per patch grid. forward() gets pos_embed
        # already added into patch_embeds, so no pos_embed parameter is kept.
        self.register_buffer("pos_embed_orig",
                             torch.zeros(1, 0, hidden_size),
                             persistent=False)


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
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_scale = attention_scale
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
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.attention_scale != 1.0:
            scores = scores * self.attention_scale
        attn = F.softmax(scores, dim=-1)
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
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = RADIOAttention(
            hidden_size,
            num_heads,
            attention_scale,
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
                 attention_scale: float,
                 model_config: "ModelConfig",
                 name_prefix: str = "") -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            RADIOBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                layer_norm_eps,
                attention_scale,
                model_config,
                name_prefix=f"{name_prefix}.blocks.{i}" if name_prefix else "")
            for i in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


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

    The exported graph consumes runtime-computed ``patch_embeds`` (embedder
    GEMM + pos_embed, see module docstring) and grid-dependent
    ``shuffle_indices``; it serves both the image and the video path.

    Output: ``[num_blocks, tokens_per_block, llm_hidden_size]``
    """

    def __init__(self,
                 config: dict,
                 model_config: "ModelConfig",
                 video_temporal_patch_size: int = 0) -> None:
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
        head_dim = hidden_size // num_heads
        attention_scale = config_module._get_attention_scaling(
            vc, head_dim, 1.0 / (float(head_dim)**0.5))

        self.patch_generator = RADIOEmbeddings(
            hidden_size,
            patch_size,
            image_size,
            video_temporal_patch_size=video_temporal_patch_size)
        # Vision blocks live under ``vision_model.radio_model.model.blocks.N.*``
        # in the HF checkpoint; thread that prefix so MIXED_PRECISION
        # ``layer_overrides`` resolve and ``*vision_model*`` wildcards match.
        self.encoder = RADIOEncoder(
            num_layers,
            hidden_size,
            num_heads,
            intermediate_size,
            layer_norm_eps,
            attention_scale,
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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, patch_embeds: torch.Tensor,
                shuffle_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeds: [num_blocks, num_patches, vit_hidden] — embedder
                GEMM output with pos_embed already added (runtime-computed).
            shuffle_indices: [num_out_tokens, scale²] int64 — pixel-shuffle
                gather indices into the num_patches axis for this grid.

        Returns:
            image_features: [num_blocks, num_out_tokens, llm_hidden_size]
        """
        B = patch_embeds.shape[0]
        reg_tokens = self.patch_generator.cls_token.token.unsqueeze(0).expand(
            B, -1, -1).to(patch_embeds.dtype)
        x = torch.cat((reg_tokens, patch_embeds), dim=1)
        x = self.encoder(x)

        # Remove register tokens (RADIO uses 10 registers, not a single CLS)
        num_reg = self.patch_generator.num_registers
        x = x[:, num_reg:, :]

        # Pixel shuffle as a gather: index_select exports as a single ONNX
        # Gather and keeps the graph valid for non-square (h, w) patch grids.
        C = x.shape[2]
        x = torch.index_select(x, 1, shuffle_indices.reshape(-1))
        x = x.reshape(B, shuffle_indices.shape[0],
                      shuffle_indices.shape[1] * C)
        x = self.mlp1(x)
        return x

    def build_shuffle_indices(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Pixel-shuffle gather indices for a (grid_h, grid_w) patch grid.

        Row m = output cell (i, j) with m = i·(grid_w/s) + j; its s² entries
        are the input patch indices (s·i+di)·grid_w + (s·j+dj) in (di, dj)
        row-major order — matching the HF pixel-shuffle (v2) token order.
        The C++ runtime computes the same table.
        """
        s = int(1.0 / self.downsample_ratio)
        oh, ow = grid_h // s, grid_w // s
        i = torch.arange(oh).view(oh, 1, 1, 1)
        j = torch.arange(ow).view(1, ow, 1, 1)
        di = torch.arange(s).view(1, 1, s, 1)
        dj = torch.arange(s).view(1, 1, 1, s)
        idx = (s * i + di) * grid_w + (s * j + dj)
        return idx.reshape(oh * ow, s * s).to(torch.int64)

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes)."""
        image_size = config["force_image_size"]
        patch_size = config["patch_size"]
        hidden_size = config["vit_hidden_size"]
        grid_side = image_size // patch_size
        patch_embeds = torch.zeros(2,
                                   grid_side * grid_side,
                                   hidden_size,
                                   dtype=torch.float16,
                                   device=device)
        shuffle_indices = self.build_shuffle_indices(grid_side,
                                                     grid_side).to(device)
        args = (patch_embeds, shuffle_indices)
        input_names = ["input", "shuffle_indices"]
        output_names = ["output"]
        B = torch.export.Dim("num_blocks", min=1)
        N = torch.export.Dim("num_patches", min=4)
        M = torch.export.Dim("num_out_tokens", min=1)
        dynamic_shapes = {
            "patch_embeds": {
                0: B,
                1: N
            },
            "shuffle_indices": {
                0: M
            },
        }
        return args, input_names, output_names, dynamic_shapes

    def save_onnx_sidecar(self, output_dir: str) -> str:
        """Save the runtime-executed embedder weights next to the ONNX model.

        Tensors (all FP16):
            ``embedder.weight``        [vit_hidden, 3·P²]     (flat GEMM layout)
            ``video_embedder.weight``  [vit_hidden, T·3·P²]   (if present)
            ``pos_embed``              [1, S², vit_hidden]    (raw checkpoint
                resolution; the runtime interpolates it per patch grid)
        """
        import os

        from ..._safetensors_io import save_file

        pg = self.patch_generator
        tensors = {
            "embedder.weight":
            pg.embedder.weight.reshape(pg.embedder.weight.shape[0], -1).to(
                torch.float16).cpu().contiguous(),
            "pos_embed":
            pg.pos_embed_orig.to(torch.float16).cpu().contiguous(),
        }
        if pg.video_temporal_patch_size > 0:
            tensors["video_embedder.weight"] = pg.video_embedder.weight.to(
                torch.float16).cpu().contiguous()
        sidecar_path = os.path.join(output_dir,
                                    "nemotron_omni_embedder.safetensors")
        save_file(tensors, sidecar_path)
        return sidecar_path


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
            if inner == "patch_generator.pos_embed":
                # forward() gets pos_embed pre-added; keep only the raw tensor
                # for the sidecar (the runtime interpolates it per grid).
                return "patch_generator.pos_embed_orig"
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
        return v

    load_submodule_weights(model,
                           weights,
                           _remap,
                           transform=_transform,
                           label="NemotronOmniVisualModel")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def resolve_video_cfg(config: dict, key: str, default):
    """Read a video sizing field, preferring ``vision_config`` (where the
    official checkpoint and vLLM place it) over the top level (older artifacts),
    then the default. Shared with the exporter so both agree on the value."""
    vision_cfg = config.get("vision_config") or {}
    if key in vision_cfg:
        return vision_cfg[key]
    return config.get(key, default)


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
    has_video = ("vision_model.radio_model.model.patch_generator."
                 "video_embedder.weight") in weights
    # Must match export.py so the video_embedder shape agrees with runtime T.
    video_t = ((resolve_video_cfg(config, "video_temporal_patch_size", None)
                or 2) if has_video else 0)
    model = NemotronOmniVisualModel(config,
                                    model_config=model_config,
                                    video_temporal_patch_size=video_t)
    model.to(dtype)
    _load_weights(model, weights)
    model.eval()
    return model


__all__ = [
    "NemotronOmniVisualModel",
    "build_nemotron_omni_visual",
]
