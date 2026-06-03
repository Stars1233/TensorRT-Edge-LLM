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
Patches the NemotronH (hybrid Mamba+Attention) model so it can be loaded
and quantized without working ``mamba_ssm`` / ``causal_conv1d`` CUDA
extensions.

NemotronH ships a custom ``modeling_nemotron_h.py`` that imports
``mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`` unconditionally and
``causal_conv1d.{causal_conv1d_fn, causal_conv1d_update}`` when
``is_causal_conv1d_available()`` returns True. In environments where the
host CUDA toolkit major version differs from the CUDA toolkit torch was
built against, those packages cannot ship working CUDA extensions, so
the runtime imports fail with::

    ModuleNotFoundError: No module named 'causal_conv1d_cuda'
    ImportError: mamba-ssm is required by the Mamba model but cannot be imported

For ONNX export / quantization calibration we do not need the fused
CUDA kernels — pure-PyTorch substitutes are sufficient. ``apply()``
installs:

  * ``mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`` — pure-PyTorch
    grouped gated RMS norm (einops + F.silu).
  * ``causal_conv1d`` — sentinel module exposing
    ``causal_conv1d_fn = causal_conv1d_update = None``; the modeling code
    has explicit ``None`` fallbacks for these symbols and uses a slower
    PyTorch path when the CUDA functions are unavailable.
  * ``mamba_ssm.ops.triton.selective_state_update`` and
    ``mamba_ssm.ops.triton.ssd_combined`` — sentinels so the
    ``is_mamba_2_ssm_available()`` branch in modeling code resolves to a
    benign ``None`` even when ``mamba_ssm`` is partially installed.

It also patches ``AutoModelForCausalLM.from_config`` to inject ONNX-export
friendly forward / mask methods. The implementation is self-contained so
``tensorrt_edgellm.quantization`` stays decoupled from the model package.

Originally adapted from
https://github.com/NVIDIA/TensorRT-LLM/blob/a1964bcbbcbbe1d6f4e0750ec5ff4d58ca7e81fb/tensorrt_llm/_torch/auto_deploy/models/patches/nemotron_h.py
"""
import contextlib
import importlib.util
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# NemotronHConfig.layers_block_type — extend stock impl with 'E' (MoE expert)
# ---------------------------------------------------------------------------

_NEMOTRON_H_CHAR_TO_BLOCK_TYPE: Dict[str, str] = {
    "M": "mamba",
    "*": "attention",
    "-": "mlp",
    "E": "moe",
}


def _nemotron_h_config_layers_block_type(self) -> List[str]:
    """Replacement ``layers_block_type`` property for ``NemotronHConfig``.

    Stock 4B/8B configuration_nemotron_h.py only recognises ``M`` / ``*`` /
    ``-``; the 30B-A3B model introduces ``E`` for MoE layers and trips a
    ``KeyError``. This replacement adds ``E -> "moe"``.
    """
    pattern = getattr(self, "hybrid_override_pattern", "")
    if not pattern:
        return []
    return [_NEMOTRON_H_CHAR_TO_BLOCK_TYPE.get(c, c) for c in pattern]


def _patch_nemotron_h_config(config: Any) -> None:
    """Idempotent monkey-patch of the NemotronHConfig class on the instance."""
    config_class = type(config)
    if getattr(config_class, "_edgellm_moe_patched", False):
        return
    config_class.layers_block_type = property(
        _nemotron_h_config_layers_block_type)
    config_class._edgellm_moe_patched = True


# ---------------------------------------------------------------------------
# Pure-PyTorch substitutes for missing CUDA kernels
# ---------------------------------------------------------------------------


# Forked from
# https://github.com/state-spaces/mamba/blob/6b32be06d026e170b3fdaf3ae6282c5a6ff57b06/mamba_ssm/ops/triton/layernorm_gated.py
def _rms_norm_ref(x,
                  weight,
                  bias,
                  z=None,
                  eps=1e-6,
                  group_size=None,
                  norm_before_gate=True,
                  upcast=True):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd *
                                                                   weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) +
                              eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


# ---------------------------------------------------------------------------
# Method-level patches for NemotronH modules (ONNX-export-friendly behavior)
# ---------------------------------------------------------------------------


def _nemotron_h_model_update_mamba_mask(self, attention_mask, cache_position):
    return None


def _nemotron_h_model_update_causal_mask(self, attention_mask, input_tensor,
                                         cache_position):
    return None


def _nemotron_h_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    device = hidden_states.device
    with contextlib.ExitStack() as stack:
        if device.type == "cuda":
            stack.enter_context(
                torch.cuda.stream(torch.cuda.default_stream(device)))
        residual = hidden_states
        hidden_states = self.norm(
            hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states,
                                       cache_params=cache_params,
                                       cache_position=cache_position)
        elif self.block_type == "attention":
            hidden_states = self.mixer(hidden_states,
                                       cache_position=cache_position)
            hidden_states = hidden_states[0]
        elif self.block_type in ["mlp", "moe"]:
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        hidden_states = residual + hidden_states
        return hidden_states


def _nemotron_h_moe_dense(self, hidden_states: torch.Tensor,
                          topk_indices: torch.Tensor,
                          topk_weights: torch.Tensor) -> torch.Tensor:
    """Dense MoE computation safe for ONNX export.

    Replaces the original sparse routing (``torch.where`` + ``index_add_``)
    with a fully-traceable dense loop: every expert runs on every token, and
    contributions are gated by the soft routing weights.
    """
    n_experts = len(self.experts)
    orig_dtype = hidden_states.dtype
    acc = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)

    for expert_idx in range(n_experts):
        expert = self.experts[expert_idx]
        expert_weight = ((topk_indices == expert_idx).to(topk_weights.dtype) *
                         topk_weights).sum(dim=-1, keepdim=True)
        expert_out = expert(hidden_states.to(orig_dtype))
        acc = acc + expert_out.to(topk_weights.dtype) * expert_weight

    return acc.to(orig_dtype)


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, List[Tuple[str, Callable]]] = {
    "NemotronHModel": [
        ("_update_causal_mask", _nemotron_h_model_update_causal_mask),
        ("_update_mamba_mask", _nemotron_h_model_update_mamba_mask),
    ],
    "NemotronHBlock": [("forward", _nemotron_h_block_forward)],
    "NemotronHMOE": [("moe", _nemotron_h_moe_dense)],
}


def get_model_from_config_patched(config, **kwargs):
    if type(config).__name__ == "NemotronHConfig":
        _patch_nemotron_h_config(config)
    model = _from_config_original(config, **kwargs)
    for _, module in model.named_modules():
        if (module_name :=
                type(module).__name__) in CUSTOM_MODULE_PATCHES.keys():
            patches = CUSTOM_MODULE_PATCHES[module_name]
            for method_name, method_patch in patches:
                setattr(module, method_name,
                        types.MethodType(method_patch, module))
    return model


# ---------------------------------------------------------------------------
# sys.modules stubs
# ---------------------------------------------------------------------------
#
# IMPORTANT: only stub the LEAF submodules that the custom modeling code
# unconditionally imports — never put the parent package (``mamba_ssm``,
# ``causal_conv1d``) in sys.modules with a synthetic ``types.ModuleType``,
# because such modules have ``__spec__ = None``. Transformers'
# ``is_mamba_2_ssm_available()`` / ``is_causal_conv1d_available()`` call
# ``importlib.util.find_spec("<pkg>")`` which raises
# ``ValueError: <pkg>.__spec__ is None`` when the package entry exists in
# sys.modules without a valid spec.
#
# Strategy:
#   * ``mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`` — modeling code
#     does ``from ... import rmsnorm_fn`` UNCONDITIONALLY, wrapped in
#     try/except that re-raises ImportError. We must populate this submodule
#     even when ``mamba_ssm`` itself is absent. The ``from <leaf> import``
#     statement short-circuits to ``sys.modules`` without forcing parent
#     imports, so this works without touching parent packages.
#   * ``causal_conv1d.{causal_conv1d_fn, causal_conv1d_update}`` — modeling
#     code only does ``from causal_conv1d import ...`` when
#     ``is_causal_conv1d_available()`` returns True. If causal_conv1d is
#     genuinely uninstalled, that branch is skipped and the modeling code
#     uses ``None`` fallbacks; we don't need to stub. If causal_conv1d is
#     INSTALLED but broken (e.g. SKIP_CUDA_BUILD), we override the import
#     with a sentinel module exposing ``None`` symbols.
#   * ``mamba_ssm.ops.triton.{selective_state_update, ssd_combined}`` —
#     modeling code only imports these when ``is_mamba_2_ssm_available()``
#     returns True. Same reasoning: stub only when the package is
#     installed-but-broken; otherwise leave alone.


def _stub_layernorm_gated() -> None:
    """Stub the leaf ``mamba_ssm.ops.triton.layernorm_gated`` submodule with
    a pure-PyTorch ``rmsnorm_fn``.

    Always installed (regardless of whether ``mamba_ssm`` itself is
    available) so the unconditional ``from … import rmsnorm_fn`` in
    modeling_nemotron_h.py succeeds. The leaf-only stub does not break
    ``importlib.util.find_spec("mamba_ssm")`` because the parent package
    entry in sys.modules is left untouched.
    """
    name = "mamba_ssm.ops.triton.layernorm_gated"
    stub = types.ModuleType(name)
    stub.rmsnorm_fn = _rms_norm_ref
    sys.modules[name] = stub


def _stub_if_broken(pkg_name: str, sentinel_attrs: Dict[str, Any]) -> None:
    """Override ``sys.modules[pkg_name]`` with a sentinel only if the
    package is installed-but-broken (find_spec succeeds).

    If the package is genuinely uninstalled (find_spec returns None /
    raises), modeling code's ``is_*_available()`` check returns False and
    its ``else`` branch sets the symbols to ``None`` itself — no stub
    needed.
    """
    try:
        spec = importlib.util.find_spec(pkg_name)
    except (ValueError, ImportError):
        spec = None
    if spec is None:
        return
    stub = types.ModuleType(pkg_name)
    stub.__spec__ = importlib.util.spec_from_loader(pkg_name, loader=None)
    for attr, value in sentinel_attrs.items():
        setattr(stub, attr, value)
    sys.modules[pkg_name] = stub


def apply() -> None:
    """Apply all NemotronH patches. Idempotent."""
    AutoModelForCausalLM.from_config = get_model_from_config_patched
    _stub_layernorm_gated()
    _stub_if_broken("causal_conv1d", {
        "causal_conv1d_fn": None,
        "causal_conv1d_update": None,
    })
    _stub_if_broken("mamba_ssm.ops.triton.selective_state_update",
                    {"selective_state_update": None})
    _stub_if_broken(
        "mamba_ssm.ops.triton.ssd_combined", {
            "mamba_chunk_scan_combined": None,
            "mamba_split_conv1d_scan_combined": None,
        })
