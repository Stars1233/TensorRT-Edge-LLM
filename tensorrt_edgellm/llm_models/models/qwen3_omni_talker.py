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
Talker and CodePredictor ONNX Export Adapters.

This module contains ONNX export adapters for the Talker subsystem, shared by
Qwen3-Omni and Qwen3-TTS:
- Talker: Audio generation decoder (codec token prediction)
- CodePredictor: Residual code generation decoder (multi-head)

These wrappers use EdgeLLMModel (with attention plugin) internally for optimized inference.

Qwen3-Omni vs Qwen3-TTS differences are handled via optional __init__ parameters
and _from_pretrained / _from_pretrained_tts class methods. The ONNX graph is identical.
"""

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .llm_model import EdgeLLMModel

# ============================================================================
# WAR (Workaround) for CodePredictor FP16 Overflow
# ============================================================================
#
# Problem: CodePredictor's MLP has FP16 overflow in layer 2+ due to large
# intermediate values from act(gate_proj) * up_proj operation.
#
# Root cause: talker_hidden input has large range [-39.5, 72.4], which
# accumulates through layers causing:
#   - Layer 0: act*up range [-0.9, 34.8] ✓
#   - Layer 1: act*up range [-1.7, 149.4] ⚠️
#   - Layer 2: act*up range [-7.1, inf] ❌ Overflow
#
# Solution: Cast gate/up outputs to FP32 before multiplication, then cast
# back to FP16 after down_proj. This is the same approach used in
# Qwen2.5-VL 3B (see visual_models/qwen2_5_vl_model.py).
#
# Reference: Qwen2_5_VLMLPPatch in tensorrt_edgellm/visual_models/qwen2_5_vl_model.py
# ============================================================================


class Qwen3OmniCodePredictorMLPPatchWAR(nn.Module):
    """
    WAR Patch for CodePredictor MLP to prevent FP16 overflow.
    
    This class wraps the original Qwen3OmniMLP and casts intermediate
    computations to FP32 to prevent overflow in the act*up operation.
    """

    def __init__(self, original_mlp: nn.Module) -> None:
        """
        Initialize from original MLP module.
        
        Args:
            original_mlp: Original Qwen3OmniMLP instance
        """
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP32 casting for numerical stability.
        
        Args:
            hidden_state: Input hidden states [batch, seq_len, hidden_dim]
        
        Returns:
            Output after MLP processing [batch, seq_len, hidden_dim]
        """
        # Apply gate and up projections (still in FP16)
        gate_output = self.gate_proj(hidden_state)
        up_output = self.up_proj(hidden_state)

        # WAR: Cast to FP32 before act*up to prevent overflow
        gate_output = gate_output.to(torch.float32)
        up_output = up_output.to(torch.float32)

        # Compute act(gate) * up in FP32
        intermediate = self.act_fn(gate_output) * up_output

        # Cast down_proj weights to FP32 for consistent computation
        # Note: This modifies the weight in-place for ONNX export
        self.down_proj.weight.data = self.down_proj.weight.data.to(
            torch.float32)
        if self.down_proj.bias is not None:
            self.down_proj.bias.data = self.down_proj.bias.data.to(
                torch.float32)

        # Apply down projection in FP32, then cast back to FP16
        output = self.down_proj(intermediate)
        return output.to(torch.float16)


def apply_code_predictor_mlp_war(model: nn.Module,
                                 start_layer: int = 0) -> None:
    """
    Apply MLP WAR patch to CodePredictor model layers.
    
    This function replaces the MLP modules in specified layers with
    Qwen3OmniCodePredictorMLPPatchWAR to prevent FP16 overflow.
    
    Args:
        model: Qwen3OmniTalkerCodePredictorModel instance
        start_layer: First layer to apply WAR (default: 0 for all layers)
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model does not have 'layers' attribute")

    num_layers = len(model.layers)
    print(
        f"[WAR] Applying MLP FP32 workaround to CodePredictor layers {start_layer}-{num_layers-1}"
    )

    for layer_idx in range(start_layer, num_layers):
        layer = model.layers[layer_idx]
        if hasattr(layer, 'mlp'):
            original_mlp = layer.mlp
            layer.mlp = Qwen3OmniCodePredictorMLPPatchWAR(original_mlp)
            print(f"  Layer {layer_idx}: MLP replaced with WAR patch")


class Qwen3OmniTalkerPatch(nn.Module):
    """
    ONNX export adapter for Qwen3-Omni Talker with EdgeLLM attention plugin.
    
    Talker is a decoder-only transformer that generates coarse audio codes.
    Architecture: Projection → EdgeLLMModel (attention plugin) → Codec Head
    """

    def __init__(self,
                 model,
                 codec_head,
                 text_projection,
                 hidden_projection=None):
        """
        Initialize from pre-trained talker model components.
        
        Args:
            model: Talker transformer backbone (Qwen3Omni or Qwen3TTS)
            codec_head: Linear layer for codec token prediction
            text_projection: MLP for projecting text embeddings → talker dimension
            hidden_projection: MLP for projecting thinker hidden states → talker dimension
                (Omni only; None for TTS which has no Thinker)
        """
        super().__init__()

        # Projection layers (Talker-specific)
        self.text_projection = text_projection
        self.hidden_projection = hidden_projection

        # Transformer with EdgeLLM attention plugin (EdgeLLMModel auto-handles missing embed_tokens)
        self.transformer = EdgeLLMModel(model, is_eagle_base=False)

        # Output head (Talker-specific)
        self.codec_head = codec_head

        # Expose attributes for compatibility
        self.config = self.transformer.config
        self.torch_dtype = self.transformer.torch_dtype
        self.model = self.transformer  # For save_embedding_table check

    @classmethod
    def _from_pretrained(cls, talker_model):
        """
        Create patch model from original Qwen3OmniTalkerForConditionalGeneration.
        
        Args:
            talker_model: Original Qwen3OmniTalkerForConditionalGeneration instance
            
        Returns:
            Qwen3OmniTalkerPatch instance
        """
        # Alias codec_embedding as embed_tokens for save_embedding_table to work
        talker_model.model.embed_tokens = talker_model.model.codec_embedding

        return cls(model=talker_model.model,
                   codec_head=talker_model.codec_head,
                   text_projection=talker_model.text_projection,
                   hidden_projection=talker_model.hidden_projection)

    @classmethod
    def _from_pretrained_tts(cls, talker_model):
        """Create from Qwen3TTSTalkerForConditionalGeneration."""
        talker_model.model.embed_tokens = talker_model.model.codec_embedding
        instance = cls(
            model=talker_model.model,
            codec_head=talker_model.codec_head,
            text_projection=talker_model.text_projection,
        )
        instance.is_tts = True
        instance._text_embedding_weight = talker_model.model.text_embedding.weight.data
        return instance

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through talker with EdgeLLM attention plugin.
        
        NOTE: Projection layers (text_projection, hidden_projection) have been moved to C++ runtime.
        Runtime should handle: inputs_embeds = text_projected * (1-mask) + hidden_projected * mask
        
        Talker is a pure autoregressive decoder — causal mask and position encoding
        are handled internally by the attention plugin, so attention_mask and
        position_ids are not needed as explicit inputs.
        
        Args:
            inputs_embeds: [batch, seq_len, hidden_dim] - Pre-projected embeddings from runtime
            past_key_values: Tuple of cached key-value tensors (EdgeLLM format)
            rope_rotary_cos_sin: [batch, seq_len, head_dim] - RoPE embeddings
            context_lengths: [batch] - Current sequence lengths
            last_token_ids: [batch, 1] - Indices of last tokens to extract logits from
            kvcache_start_index: [batch] - KV cache start indices
            
        Returns:
            Tuple of:
                - logits: [batch, 1, vocab_size] - Codec token predictions for last tokens
                - hidden_states: [batch, seq_len, hidden_dim] - Last layer hidden states
                - present_key_values: Tuple of updated KV cache tensors
        """
        hidden_states, present_kv, _ = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            output_hidden_states=False,
        )

        from ..layers.gather_nd import custom_gather_nd
        last_hidden_state = custom_gather_nd(hidden_states, last_token_ids, 1)

        logits = self.codec_head(last_hidden_state)
        logits = logits.to(torch.float32)  # FP32 for sampling

        return logits, hidden_states, present_kv


class Qwen3OmniCodePredictorPatch(nn.Module):
    """
    ONNX export adapter for CodePredictor with EdgeLLM attention plugin.
    
    CodePredictor is a small autoregressive model that generates residual
    audio codes to complement the coarse code from Talker. It has multiple
    LM heads (one per residual code). Shared by Qwen3-Omni and Qwen3-TTS.
    
    NOTE: ONNX includes lm_head as an input tensor (not weight) to enable
    dynamic lm_head selection at runtime via CUDA Graphs.
    """

    def __init__(self, model, lm_heads, small_to_mtp_projection=None):
        """
        Initialize from pre-trained code predictor model components.
        
        Args:
            model: CodePredictor transformer backbone (Qwen3Omni or Qwen3TTS)
            lm_heads: ModuleList of Linear layers for residual code prediction
            small_to_mtp_projection: Linear projection from talker hidden to CP dimension
                (TTS only; None for Omni where dimensions already match)
        """
        super().__init__()

        # Transformer with EdgeLLM attention plugin (EdgeLLMModel auto-handles missing embed_tokens)
        self.transformer = EdgeLLMModel(model, is_eagle_base=False)

        # Multiple LM heads - NOT included in ONNX, exported as safetensors for runtime
        self.lm_heads = lm_heads
        self.num_code_groups = len(
            lm_heads) + 1  # +1 for coarse code from Talker
        self.small_to_mtp_projection = small_to_mtp_projection

        # Expose attributes for compatibility
        self.config = self.transformer.config
        self.torch_dtype = self.transformer.torch_dtype
        self.model = self.transformer  # For save_embedding_table check

    @classmethod
    def _from_pretrained(cls, code_predictor_model):
        """
        Create patch model from original CodePredictor.
        
        Args:
            code_predictor_model: Original Qwen3OmniTalkerCodePredictorModelForConditionalGeneration
            
        Returns:
            Qwen3OmniCodePredictorPatch instance
        """
        # Alias codec_embedding as embed_tokens for save_embedding_table to work
        # Note: codec_embedding is a ModuleList, save_embedding_table will handle it specially
        code_predictor_model.model.embed_tokens = code_predictor_model.model.codec_embedding

        # WAR: Apply MLP FP32 workaround to prevent FP16 overflow
        # Problem: talker_hidden has large range [-39.5, 72.4], causing overflow in layer 2+ MLP
        # Solution: Cast act*up computation to FP32 (same as Qwen2.5-VL 3B WAR)
        # Apply to ALL layers (0-4) for safety, since overflow propagates
        apply_code_predictor_mlp_war(code_predictor_model.model, start_layer=0)

        return cls(model=code_predictor_model.model,
                   lm_heads=code_predictor_model.lm_head)

    @classmethod
    def _from_pretrained_tts(cls, code_predictor_model):
        """Create from Qwen3TTSTalkerCodePredictorModelForConditionalGeneration."""
        code_predictor_model.model.embed_tokens = code_predictor_model.model.codec_embedding
        # TODO: Verify if TTS CodePredictor needs MLP WAR for FP16 overflow
        return cls(
            model=code_predictor_model.model,
            lm_heads=code_predictor_model.lm_head,
            small_to_mtp_projection=code_predictor_model.
            small_to_mtp_projection,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through code predictor transformer with lm_head as input.
        
        NOTE: lm_head_weight is passed as an input tensor (not a fixed weight) to enable
        dynamic lm_head selection at runtime via CUDA Graphs.
        
        CodePredictor is a pure autoregressive decoder — causal mask and position
        encoding are handled internally by the attention plugin.
        
        Args:
            inputs_embeds: [batch, seq_len, hidden_dim] - Input embeddings
            past_key_values: Tuple of cached key-value tensors (EdgeLLM format)
            rope_rotary_cos_sin: [batch, seq_len, head_dim] - RoPE embeddings
            context_lengths: [batch] - Current sequence lengths
            last_token_ids: [batch, 1] - Indices of last tokens to extract hidden states from
            kvcache_start_index: [batch] - KV cache start indices
            lm_head_weight: [vocab_size, hidden_dim] - LM head weight matrix
                (passed as input tensor for dynamic lm_head selection via 15 CUDA Graphs)
            
        Returns:
            Tuple of:
                - logits: [batch, 1, vocab_size] - Output logits (lm_head applied)
                - hidden_states: [batch, seq_len, hidden_dim] - Full sequence hidden states (for residual)
                - present_key_values: Tuple of updated KV cache tensors
        """
        hidden_states, present_kv, _ = self.transformer(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            output_hidden_states=False,
        )

        # Extract last token hidden states using gather
        from ..layers.gather_nd import custom_gather_nd
        last_hidden_state = custom_gather_nd(hidden_states, last_token_ids, 1)

        # Apply lm_head: logits = last_hidden @ lm_head_weight.T
        # last_hidden: [batch, 1, hidden_dim]
        # lm_head_weight: [vocab_size, hidden_dim]
        # logits: [batch, 1, vocab_size]
        logits = torch.matmul(last_hidden_state, lm_head_weight.T)
        logits = logits.to(
            torch.float32)  # Cast to FP32 for sampling (project standard)

        return logits, hidden_states, present_kv


# ============================================================================
# ONNX Export Utilities for Qwen3-Omni Submodels
# ============================================================================


def create_qwen3_omni_dummy_inputs(
        model: nn.Module,
        model_type: str,
        fp8_kv_cache: bool = False) -> Dict[str, Any]:
    """
    Create dummy inputs for Qwen3-Omni submodels (Talker/CodePredictor) ONNX export.
    
    Args:
        model: Qwen3OmniTalkerPatch or Qwen3OmniCodePredictorPatch
        model_type: "talker" or "code_predictor"
        fp8_kv_cache: Whether to use FP8 precision for KV cache
        
    Returns:
        Dictionary of dummy inputs matching the model's forward signature
    """
    batch_size = 1
    seq_len = 100 if model_type == "talker" else 2
    past_len = 0

    model_config = model.config
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_hidden_layers
    num_kv_heads = model_config.num_key_value_heads
    num_heads = model_config.num_attention_heads
    head_dim = getattr(model_config, 'head_dim', hidden_size // num_heads)
    partial_rotary_factor = getattr(model_config, 'partial_rotary_factor', 1.0)
    rotary_dim = int(head_dim * float(partial_rotary_factor))
    if rotary_dim <= 0 or rotary_dim > head_dim:
        rotary_dim = head_dim
    max_position_embeddings = model_config.max_position_embeddings
    device = next(model.parameters()).device

    print(
        f"Creating dummy inputs for {model_type} with batch_size={batch_size}, seq_len={seq_len}"
    )

    # Create common inputs
    # KV cache must be FP16 (AttentionPlugin requirement), regardless of model dtype
    past_key_values = []
    for _ in range(num_layers):
        past_kv = torch.randn(batch_size,
                              2,
                              num_kv_heads,
                              seq_len,
                              head_dim,
                              dtype=torch.float16,
                              device=device)
        if fp8_kv_cache:
            past_kv = past_kv.to(torch.float8_e4m3fn)
        past_key_values.append(past_kv)

    dummy_inputs = {
        'past_key_values':
        tuple(past_key_values),
        'rope_rotary_cos_sin':
        torch.randn(batch_size,
                    max_position_embeddings,
                    rotary_dim,
                    dtype=torch.float32,
                    device=device),
        'context_lengths':
        torch.full([batch_size],
                   past_len + seq_len,
                   dtype=torch.int32,
                   device=device),
        'last_token_ids':
        torch.full([batch_size, 1],
                   seq_len - 1,
                   dtype=torch.int64,
                   device=device),
        'kvcache_start_index':
        torch.zeros(batch_size, dtype=torch.int32, device=device),
    }

    # Model-specific inputs
    if model_type == "talker":
        # Talker now takes pre-projected inputs_embeds (projection moved to runtime)
        dummy_inputs.update({
            'inputs_embeds':
            torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=torch.float16,
                        device=device),
        })
    elif model_type == "code_predictor":
        # lm_head_weight is an input tensor for dynamic selection via 15 CUDA Graphs
        vocab_size = model.config.vocab_size  # 2048 (number of audio codec codes per codebook)
        # I/O is FP16; MLP layers run in FP32 (via CodePredictor MLP WAR) to avoid overflow
        dummy_inputs.update({
            'inputs_embeds':
            torch.randn(batch_size,
                        seq_len,
                        hidden_size,
                        dtype=torch.float16,
                        device=device),
            'lm_head_weight':
            torch.randn(vocab_size,
                        hidden_size,
                        dtype=torch.float16,
                        device=device),
        })

    return dummy_inputs


def export_qwen3_omni_submodel_to_onnx(model: nn.Module,
                                       dummy_inputs: Dict[str, Any],
                                       output_dir: str,
                                       model_type: str) -> None:
    """
    Export Qwen3-Omni submodels (Talker/CodePredictor) to ONNX with EdgeLLM attention plugin.
    
    Args:
        model: Qwen3OmniTalkerPatch or Qwen3OmniCodePredictorPatch
        dummy_inputs: Dictionary containing model-specific dummy inputs
        output_dir: Directory to save ONNX model
        model_type: "talker" or "code_predictor"
    """
    # Import here to avoid circular dependency
    from ...onnx_export.onnx_utils import export_onnx
    from ..layers.attention_plugin import \
        register_attention_plugin_onnx_symbolic_functions
    from ..layers.gather_nd import register_gather_nd_onnx_symbolic_functions

    print(f"Exporting {model_type} to ONNX format: {output_dir}")
    model.eval()

    model_config = model.config
    num_layers = model_config.num_hidden_layers

    # Prepare inputs based on model type
    # NOTE: Talker/CodePredictor are pure autoregressive decoders.
    # attention_mask and position_ids are NOT passed — the attention plugin
    # handles causal masking and position encoding internally.
    is_talker = model_type == "talker"
    if is_talker:
        # Talker forward: (inputs_embeds, past_kv, rope, context_lengths, last_token_ids, kvcache_start_index)
        inputs = (
            dummy_inputs['inputs_embeds'],
            dummy_inputs['past_key_values'],
            dummy_inputs['rope_rotary_cos_sin'],
            dummy_inputs['context_lengths'],
            dummy_inputs['last_token_ids'],
            dummy_inputs['kvcache_start_index'],
        )
        input_names = (['inputs_embeds'] +
                       [f'past_key_values_{i}' for i in range(num_layers)] + [
                           'rope_rotary_cos_sin', 'context_lengths',
                           'last_token_ids', 'kvcache_start_index'
                       ])
        dynamic_axes = {
            "inputs_embeds": {
                0: "batch_size",
                1: "seq_len"
            },
        }
    else:
        # CodePredictor forward: (inputs_embeds, past_kv, rope, context_lengths, last_token_ids, kvcache_start_index, lm_head_weight)
        inputs = (
            dummy_inputs['inputs_embeds'],
            dummy_inputs['past_key_values'],
            dummy_inputs['rope_rotary_cos_sin'],
            dummy_inputs['context_lengths'],
            dummy_inputs['last_token_ids'],
            dummy_inputs['kvcache_start_index'],
            dummy_inputs['lm_head_weight'],
        )
        input_names = (
            ['inputs_embeds'] +
            [f'past_key_values_{i}' for i in range(num_layers)] + [
                'rope_rotary_cos_sin', 'context_lengths', 'last_token_ids',
                'kvcache_start_index', 'lm_head_weight'
            ])
        dynamic_axes = {
            "inputs_embeds": {
                0: "batch_size",
                1: "seq_len"
            },
        }

    # Common dynamic axes
    dynamic_axes.update({
        **{
            f"past_key_values_{i}": {
                0: "batch_size",
                3: "past_len"
            }
            for i in range(num_layers)
        },
        **{
            f"present_key_values_{i}": {
                0: "batch_size",
                3: "present_kv_cache_len"
            }
            for i in range(num_layers)
        },
        "rope_rotary_cos_sin": {
            0: "rope_batch_size",
            1: "max_position_embeddings"
        },
        "context_lengths": {
            0: "batch_size"
        },
        "last_token_ids": {
            0: "batch_size"
        },
        "kvcache_start_index": {
            0: "kv_cache_start_batch_size"
        },
    })

    # Output names
    if is_talker:
        # Talker: output logits, hidden_states, and KV cache
        output_names = (['logits', 'hidden_states'] +
                        [f'present_key_values_{i}' for i in range(num_layers)])
        dynamic_axes['hidden_states'] = {0: "batch_size", 1: "num_tokens"}
    else:
        # CodePredictor: output logits (after lm_head MatMul), hidden_states (for residual), and KV cache
        # NOTE: lm_head is applied in ONNX via MatMul with lm_head_weight input
        output_names = (['logits', 'hidden_states'] +
                        [f'present_key_values_{i}' for i in range(num_layers)])
        dynamic_axes['hidden_states'] = {0: "batch_size", 1: "num_tokens"}
        dynamic_axes['logits'] = {0: "batch_size"}

    # Register ONNX symbolic functions
    register_attention_plugin_onnx_symbolic_functions()
    register_gather_nd_onnx_symbolic_functions()

    # Export to ONNX
    export_onnx(model, inputs, output_dir, input_names, output_names,
                dynamic_axes)
    print(f"ONNX export completed.")


def save_qwen3_omni_talker_projections(model: Qwen3OmniTalkerPatch,
                                       output_dir: str) -> None:
    """
    Save Talker projection weights and optional text_embedding for C++ runtime.

    Omni: text_projection + hidden_projection
    TTS:  text_projection + text_embedding (no hidden_projection; no Thinker)
    """
    from safetensors.torch import save_file

    # text_projection (Omni + TTS)
    text_proj_dict = {
        "text_proj.fc1.weight":
        model.text_projection.linear_fc1.weight.data.cpu().half(),
        "text_proj.fc1.bias":
        model.text_projection.linear_fc1.bias.data.cpu().half(),
        "text_proj.fc2.weight":
        model.text_projection.linear_fc2.weight.data.cpu().half(),
        "text_proj.fc2.bias":
        model.text_projection.linear_fc2.bias.data.cpu().half(),
    }
    save_file(text_proj_dict,
              os.path.join(output_dir, "text_projection.safetensors"))
    print(f"Saved text_projection.safetensors to {output_dir}")

    # hidden_projection (Omni only — projects thinker hidden states to talker dimension)
    if model.hidden_projection is not None:
        hidden_proj_dict = {
            "hidden_proj.fc1.weight":
            model.hidden_projection.linear_fc1.weight.data.cpu().half(),
            "hidden_proj.fc1.bias":
            model.hidden_projection.linear_fc1.bias.data.cpu().half(),
            "hidden_proj.fc2.weight":
            model.hidden_projection.linear_fc2.weight.data.cpu().half(),
            "hidden_proj.fc2.bias":
            model.hidden_projection.linear_fc2.bias.data.cpu().half(),
        }
        save_file(hidden_proj_dict,
                  os.path.join(output_dir, "hidden_projection.safetensors"))
        print(f"Saved hidden_projection.safetensors to {output_dir}")

    # text_embedding (TTS only — TTS has no Thinker, text embedding is self-contained)
    if getattr(model, 'is_tts', False):
        save_file(
            {"text_embedding": model._text_embedding_weight.cpu().half()},
            os.path.join(output_dir, "text_embedding.safetensors"),
        )
        print(f"Saved text_embedding.safetensors to {output_dir}")


def save_qwen3_omni_code_predictor_embeddings(
        model: Qwen3OmniCodePredictorPatch, output_dir: str) -> None:
    """
    Save all codec embeddings into a single codec_embeddings.safetensors file.

    Omni: 15 embeddings (num_code_groups=16).  TTS: 31 embeddings (num_code_groups=32).
    """
    from safetensors.torch import save_file

    codec_embeddings = model.transformer.embed_tokens

    embedding_dict = {}
    for i, codec_emb in enumerate(codec_embeddings):
        embedding_dict[f"embedding_{i}"] = codec_emb.weight.data.cpu()

    save_file(embedding_dict,
              os.path.join(output_dir, "codec_embeddings.safetensors"))

    print(
        f"Saved codec_embeddings.safetensors ({len(codec_embeddings)} embeddings) to {output_dir}"
    )
    print(
        f"  Codebook size: {codec_embeddings[0].weight.shape[0]}, embedding dim: {codec_embeddings[0].weight.shape[1]}"
    )


def save_qwen3_omni_code_predictor_lm_heads(model: Qwen3OmniCodePredictorPatch,
                                            output_dir: str) -> None:
    """
    Save all lm_head weights into a single lm_heads.safetensors file.

    Omni: 15 heads (num_code_groups=16).  TTS: 31 heads (num_code_groups=32).
    """
    from safetensors.torch import save_file

    lm_heads = model.lm_heads

    weight_dict = {}
    for i, lm_head in enumerate(lm_heads):
        weight_dict[f"lm_head_{i}.weight"] = lm_head.weight.data.cpu()

    save_file(weight_dict, os.path.join(output_dir, "lm_heads.safetensors"))

    print(
        f"Saved lm_heads.safetensors ({len(lm_heads)} heads) to {output_dir}")
    print(
        f"  Codebook size: {lm_heads[0].weight.shape[0]}, hidden size: {lm_heads[0].weight.shape[1]}"
    )
