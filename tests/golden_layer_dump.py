#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""PyTorch golden: run only the first N decoder layers and dump logits + KV cache.

Produces the "PyTorch golden" used by the EdgeLLM 4-layer numeric validation: instead of
running the whole LLM, it runs only the first few decoder blocks and dumps each round's
logits and KV cache to a safetensors file. ``compare_layer_dumps.py`` then checks that
output numerically against the EdgeLLM inference dump.

== Aligning with EdgeLLM: feed the same requests JSON and apply the same chat template ==
This script reads the exact input JSON format ``llm_inference`` uses (``tests/test_cases/*.json``:
``batch_size`` / ``top_k`` / ``max_generate_length`` / ``requests[].messages``) and applies the
chat template to each request's messages with ``add_generation_prompt=True`` and
``enable_thinking=False`` -- matching EdgeLLM's applyChatTemplate defaults. Feeding both sides the
same file is what makes their input_ids identical, which is the precondition for a
position-by-position numeric comparison.

== Design notes ==
- model     : Qwen3-0.6B by default (a plain full-attention decoder, 28 layers natively). Loaded
              with ``num_hidden_layers=N`` so only the first N layers are built/loaded; embed /
              final norm / lm_head are unaffected, so the standard forward naturally does
              "N layers + norm + lm_head" and produces logits.
- logits    : each round stores only the last position's logits (``[B, 1, vocab]``). The EdgeLLM
              engine gathers the last token before lm_head and computes logits for it alone, so
              full-sequence prefill logits have no counterpart to compare against.
- KV cache  : dump K and V for all N layers.
- regime    : prefill + a fixed number of greedy decode rounds (ignoring EOS), matching the EdgeLLM
              run (``EDGELLM_IGNORE_EOS=1`` + ``--maxGenerateLength``).
- batching  : supports batches with ragged (per-row differing) lengths. Uses **left padding** (HF's
              built-in ragged batching only supports left padding): real tokens sit on the right, so
              each decode step's new token appends at the shared tail. Padding is masked via
              attention_mask and position_ids are derived from the mask's cumsum (pad tokens do not
              consume position ids). Left padding only changes layout, not values; the layout
              difference is reconciled by the comparison tool using sequence lengths.
- greedy    : take the argmax of each row's last-token logits (greedy only; if the input JSON's
              ``top_k`` is not 1 the script just warns and still uses argmax).
- dtype     : fp16 by default.
- attn      : SDPA by default (fp32-accumulated scores, matching the engine's numerically-stable
              attention), falling back to eager if a model lacks SDPA. Plain eager fp16 attention
              overflows for models without qk-norm whose K can be large (e.g. Qwen2 K~130 ->
              QK^T exceeds the fp16 max -> NaN), so it is not used as the default.
- quant     : a compressed checkpoint (e.g. NVFP4) keeps HF's architecture but its recipe-quantized
              projections are swapped for fake-quant linears (same trt::nvfp4_* ops the export uses),
              so the golden tracks the engine's FP4 numerics rather than a weight-only FP16 dequant.

== Output safetensors schema (single file) ==
    __metadata__: { model_name, num_layers, dtype, attn, num_decode_rounds,
                    batch_size, input_file, chat_template, padding_side, kv_layout,
                    seq_len_prefill }
    input_ids                  int64  [B, S]            # S = padded prefill length
    attention_mask             int64  [B, S]            # 1 = real token, 0 = pad (at prefill)
    generated_ids              int64  [B, N]            # the N greedy tokens per row
    round_{r}.logits           fp16   [B, 1, vocab]     # last position only (last-token)
    round_{r}.layer_{i}.key    fp16   [B, kv_heads, S+r, head_dim]   # i = 0..num_layers-1
    round_{r}.layer_{i}.value  fp16   [B, kv_heads, S+r, head_dim]
r=0 is prefill, r=1..N are decode steps; the KV cache length (pad columns included) grows by one
each round (S+r). The KV layout is HF-native ``[batch, num_kv_heads, seq, head_dim]`` and is recorded
in the metadata; reconciling it against EdgeLLM's layout is left to the comparison tool.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from golden_quant_linears import (_detect_gptq_zero_point_offset,
                                  _GoldenAWQLinear, _GoldenFP8Linear,
                                  _GoldenGPTQLinear, _GoldenINT8SQLinear,
                                  _GoldenMXFP8Linear, _GoldenNVFP4Linear)
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Prefer the local Qwen3-0.6B checkpoint; fall back to the HF repo id if it isn't present.
DEFAULT_CKPT_CANDIDATES = [
    "/home/scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B",  # computelab cluster
    "/scratch.edge_llm_cache/models/Qwen3-0.6B-FP8",  # EdgeLM CI machine
    "Qwen/Qwen3-0.6B",  # else, get model from HF
]
# Default requests JSON: the same file EdgeLLM's llm_inference consumes (same format, same
# chat template), so both sides tokenize identically.
DEFAULT_INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "test_cases", "ragged_batch.json")

# EdgeLLM's applyChatTemplate defaults (see cpp/runtime/llmRuntimeUtils.h); the golden matches them.
_ADD_GENERATION_PROMPT = True
_ENABLE_THINKING = False


def _pick_ckpt(user_ckpt: str | None) -> str:
    """Resolve the checkpoint path: prefer the user's, else the first candidate that exists."""
    if user_ckpt:
        return user_ckpt
    for cand in DEFAULT_CKPT_CANDIDATES:
        # Only local paths are existence-checked; an HF repo id (has '/' but isn't an absolute
        # path) is returned as the fallback as-is.
        if os.path.isdir(cand):
            return cand
    return DEFAULT_CKPT_CANDIDATES[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt",
        default=None,
        help=
        "checkpoint path or HF repo id (default: auto-detect local Qwen3-0.6B)"
    )
    p.add_argument(
        "--input-file",
        default=DEFAULT_INPUT_FILE,
        help=
        "requests JSON (llm_inference format; default test_cases/ragged_batch.json)"
    )
    p.add_argument("--num-layers",
                   type=int,
                   default=4,
                   help="number of leading decoder layers to run (default 4)")
    p.add_argument(
        "--decode-rounds",
        type=int,
        default=None,
        help=
        "number of greedy decode rounds; default derived from the input JSON's "
        "max_generate_length - 1")
    p.add_argument("--dtype",
                   default="fp16",
                   choices=["fp16", "bf16", "fp32"],
                   help="precision to run the golden in (default fp16)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda / cpu")
    p.add_argument("--out",
                   default="/tmp/qwen3_0.6b_golden_4layer.safetensors",
                   help="output safetensors path")
    return p.parse_args()


_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32
}


def _apply_chat_template(tokenizer, messages: list) -> list[int]:
    """Apply the chat template to one request's messages and return a token-id list.

    Matches EdgeLLM: add_generation_prompt=True, enable_thinking=False. Some tokenizers' templates
    don't accept the enable_thinking kwarg, in which case we fall back to omitting it. With
    transformers 5.x, ``tokenize=True`` returns a BatchEncoding (dict), so we uniformly extract
    input_ids into a list[int].
    """
    try:
        res = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=_ADD_GENERATION_PROMPT,
            enable_thinking=_ENABLE_THINKING,
            tokenize=True)
    except TypeError:
        res = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=_ADD_GENERATION_PROMPT,
            tokenize=True)
    if hasattr(res, "keys"):  # BatchEncoding / dict -> take input_ids
        res = res["input_ids"]
    return list(res)


def main() -> None:
    args = parse_args()
    ckpt = _pick_ckpt(args.ckpt)
    torch_dtype = _DTYPE_MAP[args.dtype]

    with open(args.input_file, encoding="utf-8") as f:
        req_cfg = json.load(f)
    requests = req_cfg["requests"]
    max_gen = int(req_cfg.get("max_generate_length", 6))
    top_k = int(req_cfg.get("top_k", 1))
    # One dump per round including prefill: rounds = 0..decode_rounds (decode_rounds+1 in total).
    # To match EdgeLLM's max_generate_length rounds (round 0..max_gen-1), set decode_rounds = max_gen-1.
    decode_rounds = args.decode_rounds if args.decode_rounds is not None else max(
        1, max_gen - 1)
    if top_k != 1:
        print(
            f"[golden] WARNING: input top_k={top_k} != 1; the golden only does greedy (argmax)."
        )
    print(
        f"[golden] ckpt={ckpt} num_layers={args.num_layers} dtype={args.dtype} "
        f"device={args.device} batch_size={len(requests)} decode_rounds={decode_rounds}"
    )

    # ---- 1. Load the tokenizer (left padding) ----
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Apply the chat template, then left-pad into a batch ----
    id_lists = [
        _apply_chat_template(tokenizer, r["messages"]) for r in requests
    ]
    enc = tokenizer.pad({"input_ids": id_lists},
                        padding=True,
                        return_tensors="pt")
    input_ids = enc["input_ids"].to(args.device)  # [B, S] (S = padded length)
    attention_mask = enc["attention_mask"].to(
        args.device)  # [B, S], 1 = real token, 0 = pad
    bs, seq_len_prefill = input_ids.shape
    real_lengths = attention_mask.sum(-1).tolist()
    print(
        f"[golden] padded seq_len={seq_len_prefill}, real_lengths={real_lengths}"
    )

    # ---- 3. Override the layer count to N, then load ----
    # Use transformers' native modeling (trust_remote_code=False). Only Nemotron-H ships remote
    # code, and that copy's hybrid cache mis-sizes the Mamba state (uses expand*hidden instead of
    # mamba_num_heads*mamba_head_dim); the native NemotronH path is correct and matches EdgeLLM.
    # Qwen3.5 / Qwen3 / Llama carry no remote code, so this is a no-op for them.
    config = AutoConfig.from_pretrained(ckpt)
    # Hybrid VLM configs (e.g. Qwen3.5) nest the decoder hyper-params under text_config; the
    # layer count / layer_types live there, not at the top level.
    layer_cfg = getattr(config, "text_config", config)
    assert hasattr(
        layer_cfg, "num_hidden_layers"
    ), "this model's config has no num_hidden_layers; handle separately"
    assert args.num_layers <= layer_cfg.num_hidden_layers, \
        f"--num-layers={args.num_layers} exceeds the model's native layer count {layer_cfg.num_hidden_layers}"
    # Truncate to the first N layers. The mechanism differs by family:
    #  - Nemotron-H: ``layers_block_type`` is the source of truth; ``num_hidden_layers`` and
    #    ``hybrid_override_pattern`` are read-only properties derived from its length.
    #  - Qwen3.5: ``layer_types`` (linear_attention / full_attention) alongside num_hidden_layers.
    #  - plain decoders: just num_hidden_layers.
    if getattr(layer_cfg, "layers_block_type", None):
        layer_cfg.layers_block_type = layer_cfg.layers_block_type[:args.
                                                                  num_layers]
    else:
        layer_cfg.num_hidden_layers = args.num_layers
        if getattr(layer_cfg, "layer_types", None):
            layer_cfg.layer_types = layer_cfg.layer_types[:args.num_layers]
    # Per-layer type list (recurrent / attention / mlp / moe), used to dump state by type and to
    # renumber the dumped layers densely so they line up with EdgeLLM's cache index (see _dump_round).
    layer_types = _layer_type_list(layer_cfg, args.num_layers)
    print(f"[golden] layer_types={layer_types}")
    quantized = _is_quantized_ckpt(ckpt, config)
    if quantized:
        print("[golden] quantized checkpoint detected -> fake-quant linears "
              "(NVFP4/FP8/MXFP8/INT8SQ/AWQ/GPTQ)")
        # Drop the quant config so HF builds plain nn.Linear (which we then patch).
        # Otherwise a recognized backend (e.g. AWQ) would swap in its own quant
        # modules (needing autoawq, and lacking the in/out_features we read).
        cfg_objs = [config] + ([layer_cfg] if layer_cfg is not config else [])
        for cfg_obj in cfg_objs:
            if getattr(cfg_obj, "quantization_config", None) is not None:
                cfg_obj.quantization_config = None

    def _build(attn_impl: str):
        if quantized:
            # Quantized ckpt: HF can't load packed quant weights into nn.Linear, so build the
            # architecture empty, swap the recipe-quantized projections for fake-quant linears,
            # then load the packed weights + scales. Attention stays HF fp16.
            m = AutoModelForCausalLM.from_config(config,
                                                 attn_implementation=attn_impl)
            m = m.to(torch_dtype)
            _load_quantized_state(m, ckpt)
            return m
        return AutoModelForCausalLM.from_pretrained(
            ckpt,
            config=config,
            dtype=torch_dtype,
            attn_implementation=attn_impl)

    # Prefer SDPA: it accumulates QK^T in fp32 (numerically stable, matching the engine's
    # attention). Plain eager fp16 attention overflows for models without qk-norm whose K can be
    # large (e.g. Qwen2 K~130 -> QK^T exceeds the fp16 max -> NaN). Fall back to eager if a model
    # does not implement SDPA.
    attn_impl = "sdpa"
    try:
        model = _build(attn_impl)
    except (ValueError, RuntimeError) as exc:
        attn_impl = "eager"
        print(
            f"[golden] SDPA unavailable ({type(exc).__name__}); using eager attention"
        )
        model = _build(attn_impl)
    model.to(args.device)
    model.eval()

    # position_ids: pad tokens must not consume position ids -> cumsum(mask) - 1, clamped to >= 0.
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    tensors: dict[str, torch.Tensor] = {}
    generated: list[torch.Tensor] = []  # each element is [B, 1]

    # ---- 4. Prefill (round 0) ----
    # cache_position = absolute index of each token in the cache. Nemotron-H needs it to size its
    # causal / mamba masks correctly (with a Mamba-only truncation there are no attention layers, so
    # the cache's get_seq_length() is 0 and the length can't be inferred); plain / Qwen3.5 models
    # accept it as the standard HF argument, so passing it is uniformly safe.
    cache_position = torch.arange(seq_len_prefill, device=args.device)
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True)
    past = out.past_key_values
    _dump_round(tensors, 0, out.logits, past, layer_types)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
    generated.append(next_token)

    # ---- 5. Run a fixed number of greedy decode rounds (ignoring EOS) ----
    for r in range(1, decode_rounds + 1):
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((bs, 1), dtype=attention_mask.dtype, device=args.device)
        ],
                                   dim=1)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:,
                                                                           -1:]
        # The new token sits at absolute cache index seq_len_prefill + (r - 1).
        cache_position = torch.tensor([seq_len_prefill + r - 1],
                                      device=args.device)
        with torch.no_grad():
            out = model(input_ids=next_token,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past,
                        cache_position=cache_position,
                        use_cache=True)
        past = out.past_key_values
        _dump_round(tensors, r, out.logits, past, layer_types)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # Only record tokens that are fed forward: each stored token is the input to
        # the next round, so the final round's argmax (r == decode_rounds) has no
        # consumer and is intentionally skipped. generated_ids therefore has
        # decode_rounds columns (the teacher-forced inputs); every round's logits are
        # still compared regardless of token recording.
        if r < decode_rounds:
            generated.append(next_token)

    # ---- 6. Store input_ids / attention_mask / generated_ids + metadata, then write out ----
    generated_ids = torch.cat(generated, dim=1)  # [B, N]
    tensors["input_ids"] = input_ids.to("cpu", torch.int64)
    tensors["attention_mask"] = enc["attention_mask"].to("cpu", torch.int64)
    tensors["generated_ids"] = generated_ids.to("cpu", torch.int64)

    metadata = {
        "model_name": ckpt,
        "num_layers": str(args.num_layers),
        "dtype": args.dtype,
        "attn": attn_impl,
        "num_decode_rounds": str(decode_rounds),
        "batch_size": str(bs),
        "input_file": args.input_file,
        "chat_template":
        f"add_generation_prompt={_ADD_GENERATION_PROMPT},enable_thinking={_ENABLE_THINKING}",
        "padding_side": "left",
        "kv_layout": "[batch, num_kv_heads, seq, head_dim]",
        "seq_len_prefill": str(seq_len_prefill),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)
    for b in range(bs):
        print(f"[golden] row {b}: generated_ids={generated_ids[b].tolist()} "
              f"decoded={tokenizer.decode(generated_ids[b])!r}")
    print(f"[golden] wrote {len(tensors)} tensors -> {args.out}")


def _layer_type_list(layer_cfg, num_layers: int) -> list[str]:
    """Return the per-layer type for the first ``num_layers`` layers.

    Mirrors EdgeLLM's C++ ``_parse_layer_types`` so the two sides agree on what each layer is:
      - ``recurrent`` : Mamba / Gated DeltaNet / linear_attention (fixed-size recurrent + conv state)
      - ``attention`` : full self-attention (KV cache)
      - ``mlp`` / ``moe`` : feed-forward only, **no cache state**

    Reads ``layers_block_type`` / ``layer_types`` (Qwen3.5: ``linear_attention`` / ``full_attention``)
    if present, else ``hybrid_override_pattern`` (Nemotron-H: ``M`` mamba, ``-`` mlp, ``*`` attention,
    ``E`` moe), else falls back to all-attention.
    """
    raw = getattr(layer_cfg, "layers_block_type", None) or getattr(
        layer_cfg, "layer_types", None)
    if raw:
        out = []
        for bt in list(raw)[:num_layers]:
            s = str(bt).lower()
            if s == "linear_attention" or "mamba" in s:
                out.append("recurrent")
            elif s == "moe":
                out.append("moe")
            elif "mlp" in s:
                out.append("mlp")
            else:
                out.append("attention")
        return out
    pattern = getattr(layer_cfg, "hybrid_override_pattern", None)
    if pattern:
        pmap = {"M": "recurrent", "-": "mlp", "*": "attention", "E": "moe"}
        return [pmap[c] for c in pattern[:num_layers] if c in pmap]
    return ["attention"] * num_layers


def _state_at(past, abs_idx: int, ltype: str):
    """Return ``(a, b)`` cache tensors for absolute layer ``abs_idx``.

    For a recurrent layer ``(recurrent_state, conv_state)``; for an attention layer ``(key, value)``.
    Indexed by absolute decoder layer (the cache holds an entry per layer, including the stateless
    MLP slots that the caller skips). transformers 5.x exposes ``cache.layers[i]`` with
    ``.recurrent_states`` / ``.conv_states`` (Mamba / Gated DeltaNet) or ``.keys`` / ``.values``
    (attention); older attention-only caches expose ``key_cache`` / ``value_cache`` lists.
    """
    layers = getattr(past, "layers", None)
    if layers is not None and abs_idx < len(
            layers):  # transformers 5.x unified cache.
        layer = layers[abs_idx]
        if ltype == "recurrent":
            return layer.recurrent_states, getattr(layer, "conv_states", None)
        return layer.keys, layer.values
    if hasattr(past, "key_cache"):  # legacy attention-only cache.
        return past.key_cache[abs_idx], past.value_cache[abs_idx]
    return past[abs_idx][0], past[abs_idx][1]


def _dump_round(
    tensors: dict[str, torch.Tensor],
    round_idx: int,
    logits: torch.Tensor,
    past,
    layer_types: list[str],
) -> None:
    """Stash one round's logits and per-layer state into the tensors dict (moved to CPU).

    State is dumped by layer type: attention -> ``.key`` / ``.value``; recurrent (Mamba / Gated
    DeltaNet) -> ``.recurrent_state`` / ``.conv_state``. **MLP / MoE layers carry no cache state and
    are skipped.** Crucially, the dumped layers are renumbered *densely* (``layer_0, layer_1, ...``
    over only the state-bearing layers) so the index lines up with EdgeLLM, whose cache never holds
    MLP layers. Using the absolute decoder index here would desync the two sides on hybrid models
    that interleave MLP layers (e.g. Nemotron-H's ``M-M-`` -> golden ``{0,2}`` vs EdgeLLM ``{0,1}``).
    """
    # Store only the last position's logits ([B, 1, vocab]). The EdgeLLM engine gathers the last
    # token before lm_head and computes logits for it alone; under left padding the last real token
    # is always the rightmost column (index -1) for every row.
    tensors[f"round_{round_idx}.logits"] = (
        logits[:, -1:, :].detach().to("cpu").contiguous())

    dense = 0  # dense cache index over state-bearing layers; matches EdgeLLM's layer numbering.
    for abs_idx, ltype in enumerate(layer_types):
        if ltype in ("mlp", "moe"):
            continue  # no cache state -> not dumped and does not consume a dense index.
        prefix = f"round_{round_idx}.layer_{dense}."
        a, b = _state_at(past, abs_idx, ltype)
        if ltype == "recurrent":
            tensors[prefix +
                    "recurrent_state"] = a.detach().to("cpu").contiguous()
            if b is not None and b.numel() > 0:
                tensors[prefix +
                        "conv_state"] = b.detach().to("cpu").contiguous()
        else:
            tensors[prefix + "key"] = a.detach().to("cpu").contiguous()
            tensors[prefix + "value"] = b.detach().to("cpu").contiguous()
        dense += 1


def _is_quantized_ckpt(ckpt: str, config) -> bool:
    """A checkpoint is quantized if it ships a quant config (ModelOpt / compressed)."""
    return (os.path.exists(os.path.join(ckpt, "hf_quant_config.json"))
            or getattr(config, "quantization_config", None) is not None)


def _set_submodule(root: torch.nn.Module, dotted: str,
                   new: torch.nn.Module) -> None:
    """Replace ``root.<dotted>`` with ``new`` (handles ModuleList integer indices)."""
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def _load_quantized_state(model: torch.nn.Module, ckpt: str) -> None:
    """Patch the recipe-quantized Linears to fake-quant and load all weights.

    A projection ``P`` is quantized iff the checkpoint has ``P.weight_scale``
    (NVFP4/FP8/MXFP8/INT8-SQ) or ``P.qweight`` (INT4 AWQ/GPTQ); the recipe is read
    straight from the artifact. Dispatch per projection: ``qweight`` + ``g_idx``
    -> GPTQ; ``qweight`` -> AWQ; ``weight_scale_2`` -> NVFP4; uint8 ``weight_scale``
    -> MXFP8; int8 ``weight`` -> INT8 SmoothQuant; else FP8. Supports
    mixed-precision checkpoints. Non-quantized tensors (embed / norm / lm_head /
    bias) load normally; keys for the dropped (truncated) layers are ignored.
    """
    # Gather all checkpoint tensors (single- or multi-file).
    state: dict[str, torch.Tensor] = {}
    shards = [
        f for f in os.listdir(ckpt)
        if f.endswith(".safetensors") and "index" not in f
    ]
    for shard in shards:
        state.update(load_file(os.path.join(ckpt, shard)))

    # ModelOpt / upstream Mamba-family checkpoints (e.g. NVFP4 Nemotron-H exported
    # via the bundled remote modeling) store the body under ``backbone.`` while
    # transformers' native modeling uses ``base_model_prefix`` (``model.``).
    # ``from_pretrained`` reconciles this; we build via ``from_config`` +
    # ``load_state_dict`` (no remap), so realign the prefix here -- otherwise every
    # non-quantized Mamba param (embeddings / norm / dt_bias / A_log / conv1d ...)
    # shows up as a missing key.
    prefix = model.base_model_prefix
    if prefix != "backbone" and any(k.startswith("backbone.") for k in state):
        state = {
            (f"{prefix}.{k[len('backbone.'):]}" if k.startswith("backbone.") else k):
            v
            for k, v in state.items()
        }

    # Find quantized projection prefixes (relative to the CausalLM module).
    # FP family carries ``.weight_scale``; INT4 AWQ/GPTQ carry ``.qweight``.
    prefixes = sorted(
        set(k[:-len(".weight_scale")]
            for k in state if k.endswith(".weight_scale"))
        | set(k[:-len(".qweight")] for k in state if k.endswith(".qweight")))
    counts = {
        "NVFP4": 0,
        "FP8": 0,
        "MXFP8": 0,
        "INT8SQ": 0,
        "AWQ": 0,
        "GPTQ": 0
    }
    for prefix in prefixes:
        try:
            orig = model.get_submodule(prefix)
        except AttributeError:
            continue  # belongs to a dropped (truncated) layer
        has_bias = f"{prefix}.bias" in state
        in_f, out_f = orig.in_features, orig.out_features
        if f"{prefix}.qweight" in state:
            # INT4 W4A16: group_size from scales shape [in//g, out].
            group_size = in_f // state[f"{prefix}.scales"].shape[0]
            if f"{prefix}.g_idx" in state:
                offset = _detect_gptq_zero_point_offset(
                    state[f"{prefix}.qzeros"])
                new = _GoldenGPTQLinear(in_f, out_f, group_size, has_bias,
                                        offset)
                counts["GPTQ"] += 1
            else:
                new = _GoldenAWQLinear(in_f, out_f, group_size, has_bias)
                counts["AWQ"] += 1
        elif f"{prefix}.weight_scale_2" in state:
            new = _GoldenNVFP4Linear(in_f, out_f, has_bias)
            counts["NVFP4"] += 1
        elif state[f"{prefix}.weight_scale"].dtype == torch.uint8:
            new = _GoldenMXFP8Linear(in_f, out_f, has_bias)
            counts["MXFP8"] += 1
        elif state[f"{prefix}.weight"].dtype == torch.int8:
            # W8A8 SmoothQuant: full int8 weight + per-channel scale (NVFP4's int8
            # packed weight is caught earlier by weight_scale_2).
            new = _GoldenINT8SQLinear(in_f, out_f, has_bias)
            counts["INT8SQ"] += 1
        else:
            new = _GoldenFP8Linear(in_f, out_f, has_bias)
            counts["FP8"] += 1
        _set_submodule(model, prefix, new)
    n_built = sum(counts.values())

    missing, unexpected = model.load_state_dict(state, strict=False)
    # Re-tie lm_head <-> embed_tokens if the config says so (the ckpt stores only the
    # shared embedding, so lm_head.weight legitimately shows up as "missing").
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    # `missing` should be empty for the layers we kept; `unexpected` = dropped layers.
    _IGNORE_MISSING = ("rotary_emb", "lm_head.weight")
    real_missing = [
        m for m in missing if not any(s in m for s in _IGNORE_MISSING)
    ]
    if real_missing:
        raise RuntimeError(
            f"quantized load: {len(real_missing)} missing keys, e.g. {real_missing[:3]}"
        )
    active = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    print(f"[golden] quantized: patched {n_built} linears ({active}); "
          f"ignored {len(unexpected)} keys from dropped layers")


if __name__ == "__main__":
    main()
