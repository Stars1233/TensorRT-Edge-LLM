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
Qwen3.5 MTP Draft Model for Quantization.

Pure-PyTorch MTP draft model that reuses ``EdgeLLMDecoderLayer`` for the
single decoder block.  The ``quant_forward`` method runs a calibration-friendly
forward pass (no TRT plugins, no KV cache) so that modelopt can collect
activation statistics and quantize the linear layers.

Weight loading
--------------
MTP weights live under the ``mtp.*`` prefix in the base Qwen3.5 checkpoint.
``from_pretrained`` loads them directly from safetensors, stripping the prefix,
and shares ``embed_tokens`` / ``lm_head`` with the base model.
"""

import glob
import os
from typing import Any, Optional

import modelopt.torch.opt as mto
import torch
from safetensors import safe_open
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5Attention,
                                                          Qwen3_5MLP,
                                                          Qwen3_5RMSNorm)

from ..layers.layers import EdgeLLMDecoderLayer

__all__ = ["MtpDraftModel"]


class _Qwen3_5MtpDecoderLayer(nn.Module):
    """Thin wrapper that mirrors a single HF Qwen3.5 decoder layer.

    ``EdgeLLMDecoderLayer`` expects an ``nn.Module`` with ``.self_attn``,
    ``.mlp``, ``.input_layernorm``, and ``.post_attention_layernorm``
    attributes.  This class creates those from the Qwen3.5 text config and
    serves as the input to ``EdgeLLMDecoderLayer(module)``.
    """

    def __init__(self, text_config: Any) -> None:
        super().__init__()
        self.hidden_size = text_config.hidden_size
        self.self_attn = Qwen3_5Attention(text_config, layer_idx=0)
        self.mlp = Qwen3_5MLP(text_config, text_config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(text_config.hidden_size,
                                              text_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            text_config.hidden_size, text_config.rms_norm_eps)


class MtpDraftModel(nn.Module):
    """Qwen3.5 MTP draft model for quantization calibration.

    Architecture (from the HF checkpoint ``mtp.*`` keys)::

        inputs_embeds ─→ pre_fc_norm_embedding ─┐
                                                 ├─ cat ─→ fc ─→ decoder_layer ─→ norm ─→ lm_head
        hidden_states ─→ pre_fc_norm_hidden ────┘

    The single decoder layer is a standard Qwen3.5 full-attention block
    (with gated attention output, QK-norm, and SwiGLU MLP).
    """

    def __init__(self, text_config: Any) -> None:
        super().__init__()
        self.config = text_config
        hidden_size = text_config.hidden_size

        # MTP-specific fusion layers
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden_size,
                                                    text_config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden_size,
                                                 text_config.rms_norm_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Single decoder layer wrapped in EdgeLLMDecoderLayer for quant_forward
        hf_layer = _Qwen3_5MtpDecoderLayer(text_config)
        self.layers = nn.ModuleList([EdgeLLMDecoderLayer(hf_layer, index=0)])

        self.norm = Qwen3_5RMSNorm(hidden_size, text_config.rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size,
                                 text_config.vocab_size,
                                 bias=False)

        # Shared with the base model — set by from_pretrained / calibration
        self.embed_tokens: Optional[nn.Embedding] = None

        # RoPE for calibration forward
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)

    @property
    def device(self):
        return next(self.parameters()).device

    def quant_forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_from_base: torch.Tensor,
    ) -> torch.Tensor:
        """Calibration-friendly forward pass (no KV cache, no TRT plugins).

        Args:
            input_ids: Token IDs of shape ``(batch, seq_len)`` — used to
                look up embeddings via the shared ``embed_tokens``.
            hidden_states_from_base: Last hidden state from the base model,
                shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Logits of shape ``(batch, vocab_size)``.
        """
        assert self.embed_tokens is not None, \
            "embed_tokens must be set (shared from base model) before calling quant_forward"

        inputs_embeds = self.embed_tokens(input_ids)

        # Fuse embeddings and base hidden states
        normed_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        normed_hidden = self.pre_fc_norm_hidden(hidden_states_from_base)
        hidden_states = self.fc(
            torch.cat((normed_embeds, normed_hidden), dim=-1))

        # RoPE for pure-PyTorch attention
        position_ids = torch.arange(0,
                                    input_ids.shape[1],
                                    dtype=input_ids.dtype,
                                    device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Single decoder layer (uses EdgeLLMDecoderLayer.quant_forward)
        for layer in self.layers:
            hidden_states = layer.quant_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
            )

        # Final norm + lm_head on last token
        hidden_states = hidden_states[:, -1]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        text_config: Any,
        device: str = "cuda",
    ) -> "MtpDraftModel":
        """Load MTP draft weights from a Qwen3.5 checkpoint.

        Reads ``mtp.*`` keys from safetensors, strips the ``mtp.`` prefix,
        and loads them into the model.  ``lm_head`` is loaded from the
        checkpoint's ``lm_head.weight`` (or tied from ``embed_tokens``).

        Args:
            model_dir: Path to the HF checkpoint directory containing
                ``mtp.*`` weights in safetensors.
            text_config: Qwen3.5 text config (``config.text_config``).
            device: Target device.

        Returns:
            Loaded MTP draft model.
        """
        model = cls(text_config)

        # Check for a previously saved modelopt quantized model
        quantized_path = os.path.join(model_dir,
                                      "modelopt_mtp_quantized_model.pth")
        if os.path.exists(quantized_path):
            mto.restore(model, quantized_path)
            return model

        # Load MTP weights from safetensors
        mtp_state_dict = {}
        for sf_path in sorted(
                glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(sf_path, framework="pt", device=device) as f:
                for key in f.keys():
                    if key.startswith("mtp."):
                        # Strip "mtp." prefix and remap to our module names.
                        # Checkpoint: mtp.layers.0.self_attn.q_proj.weight
                        # Our model: layers.0.self_attn.qkv_proj.q_proj.weight
                        new_key = key[len("mtp."):]
                        new_key = _remap_attn_key(new_key)
                        mtp_state_dict[new_key] = f.get_tensor(key)
                    elif key == "lm_head.weight":
                        mtp_state_dict["lm_head.weight"] = f.get_tensor(key)
                    elif key in ("model.embed_tokens.weight",
                                 "model.language_model.embed_tokens.weight"):
                        # Fallback for tied embeddings
                        if "lm_head.weight" not in mtp_state_dict:
                            mtp_state_dict["lm_head.weight"] = f.get_tensor(
                                key)

        missing, unexpected = model.load_state_dict(mtp_state_dict,
                                                    strict=False)
        # embed_tokens and rotary_emb are expected to be missing (shared later)
        real_missing = [
            k for k in missing if not k.startswith("embed_tokens")
            and not k.startswith("rotary_emb")
        ]
        if real_missing:
            print(f"Warning: Missing keys in MTP draft model: {real_missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in MTP draft model: {unexpected}")

        model.to(device)
        return model

    def save_pretrained(self, output_dir: str) -> None:
        """Save the quantized MTP draft model."""
        os.makedirs(output_dir, exist_ok=True)
        mto.save(self,
                 os.path.join(output_dir, "modelopt_mtp_quantized_model.pth"))
        print(f"MTP draft model saved to {output_dir}")


def _remap_attn_key(key: str) -> str:
    """Remap HF attention keys to EdgeLLMAttention's QKVProj structure.

    HF checkpoint: ``layers.0.self_attn.q_proj.weight``
    EdgeLLMAttention: ``layers.0.self_attn.qkv_proj.q_proj.weight``
    """
    for proj in ("q_proj", "k_proj", "v_proj"):
        old = f"self_attn.{proj}"
        new = f"self_attn.qkv_proj.{proj}"
        if old in key and f"qkv_proj.{proj}" not in key:
            return key.replace(old, new)
    # o_proj stays as-is (EdgeLLMAttention stores it directly)
    # q_norm / k_norm -> EdgeLLMAttention.qk_norm.q_norm / .k_norm
    for norm in ("q_norm", "k_norm"):
        old = f"self_attn.{norm}"
        new = f"self_attn.qk_norm.{norm}"
        if old in key and "qk_norm" not in key:
            return key.replace(old, new)
    return key
