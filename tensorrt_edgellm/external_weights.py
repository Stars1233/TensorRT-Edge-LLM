# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Externalized model-weight helpers for checkpoint exports."""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

EXTERNAL_WEIGHT_INT4_FFN = "int4_ffn"
EXTERNAL_WEIGHT_INT4_MOE = "int4_moe"
EXTERNAL_WEIGHT_LM_HEAD = "lm_head"
EXTERNAL_WEIGHT_ALL = "all"
EXTERNAL_WEIGHT_KINDS = (
    EXTERNAL_WEIGHT_INT4_FFN,
    EXTERNAL_WEIGHT_INT4_MOE,
    EXTERNAL_WEIGHT_LM_HEAD,
)
EXTERNAL_WEIGHT_CHOICES = (*EXTERNAL_WEIGHT_KINDS, EXTERNAL_WEIGHT_ALL)


def resolve_externalize_weights(externalize_weights) -> "list[str]":
    """Normalize and validate requested external weight kinds."""
    if externalize_weights is None:
        return []
    if isinstance(externalize_weights, str):
        values = [externalize_weights]
    else:
        values = list(externalize_weights)

    requested: list[str] = []
    for value in values:
        if value == EXTERNAL_WEIGHT_ALL:
            return list(EXTERNAL_WEIGHT_KINDS)
        if value not in EXTERNAL_WEIGHT_KINDS:
            supported = ", ".join(EXTERNAL_WEIGHT_CHOICES)
            raise ValueError(f"Unsupported external weight kind '{value}'. "
                             f"Supported kinds: {supported}")
        if value not in requested:
            requested.append(value)
    return requested


def patch_external_weight_manifest(
        out_dir: str,
        external_weight_files: "list[dict[str, object]]") -> None:
    """Add external weight file metadata to runtime ``config.json``."""
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["external_weight_files"] = external_weight_files
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Patched config.json with %d external weight file(s)",
                len(external_weight_files))


def reject_quantized_lm_head_externalization(
        model, model_dir: str,
        requested_external_weights: "list[str]") -> None:
    """Refuse ``--externalize-weights lm_head`` when the LM head is quantized.

    The LM-head externalization path assumes a single dense fp16 weight
    initializer it can move into a safetensors external weight file.  When the
    LM head is quantized (FP8 / NVFP4 / INT4 AWQ / ...), error out here before
    the expensive ONNX export and tell the user to drop ``lm_head`` from
    ``--externalize-weights`` or re-quantize without ``--lm_head_quantization``.

    The check delegates to :func:`config.module_quant_type`, which encodes
    the same lm_head decision :func:`models.linear.make_linear` uses
    (``quant.excluded``, tied embeddings, MIXED_PRECISION layer overrides,
    dominant ``quant_type``), so detection automatically tracks any future
    ``hf_quant_config.json`` shapes that the parser learns to handle.
    """
    if EXTERNAL_WEIGHT_LM_HEAD not in requested_external_weights:
        return

    from .config import QUANT_FP16, module_quant_type

    lm_head_quant = module_quant_type("lm_head", model.config)
    if lm_head_quant == QUANT_FP16:
        return

    location = f" at {model_dir!r}" if model_dir else ""
    raise ValueError(
        f"--externalize-weights lm_head requires an fp16 LM head, but the "
        f"checkpoint{location} has lm_head quantized as {lm_head_quant!r}. "
        "Re-quantize without --lm_head_quantization, or omit 'lm_head' "
        "from --externalize-weights.")


def _is_dense_ffn_int4_weight_name(
        name: str, layer_types: "list[str] | tuple[str, ...]") -> bool:
    """Return True when ``name`` belongs to a dense FFN int4 plugin input.

    The attention Q/K/V/O projections also use Int4GroupwiseGemmPlugin; keep
    them embedded in ONNX for now.  Dense decoder FFNs use ``*.mlp.*`` names in
    the default loader path.  Nemotron-H uses ``backbone.layers.N.mixer`` for
    both attention and MLP-only layers, so check ``layer_types`` before marking
    those weights external.
    """
    normalized = name.replace("/", ".").lower()
    ffn_projections = (".gate_proj.", ".up_proj.", ".down_proj.")
    if ".mlp." in normalized and any(p in normalized for p in ffn_projections):
        return True

    match = re.search(
        r"backbone\.layers\.(\d+)\.mixer\.(up_proj|down_proj)\.",
        normalized,
    )
    if match is None:
        return False

    layer_idx = int(match.group(1))
    return layer_idx < len(layer_types) and layer_types[layer_idx] == "mlp"


def _find_int4_ffn_weight_initializers(onnx_model, model) -> "list[str]":
    """Return dense FFN Int4GroupwiseGemmPlugin qweight/scale initializers."""
    initializers = {init.name for init in onnx_model.graph.initializer}
    layer_types = tuple(getattr(model.config, "layer_types", ()))

    external_names: list[str] = []
    external_name_set: set[str] = set()
    for node in onnx_model.graph.node:
        if node.domain != "trt_edgellm" or node.op_type != "Int4GroupwiseGemmPlugin":
            continue
        if len(node.input) < 3:
            continue
        if not _is_dense_ffn_int4_weight_name(node.input[1], layer_types):
            continue
        for tensor_name in (node.input[1], node.input[2]):
            if tensor_name in initializers and tensor_name not in external_name_set:
                external_names.append(tensor_name)
                external_name_set.add(tensor_name)
    return external_names


def _find_int4_moe_weight_initializers(onnx_model) -> "list[str]":
    """Return Int4MoePlugin qweight initializers, leaving scales embedded."""
    initializers = {init.name for init in onnx_model.graph.initializer}

    external_names: list[str] = []
    external_name_set: set[str] = set()
    for node in onnx_model.graph.node:
        if node.domain != "trt_edgellm" or node.op_type != "Int4MoePlugin":
            continue
        if len(node.input) < 5:
            continue
        for tensor_name in (node.input[2], node.input[4]):
            if tensor_name in initializers and tensor_name not in external_name_set:
                external_names.append(tensor_name)
                external_name_set.add(tensor_name)
    return external_names


def _lm_head_weight_candidate_shapes(model) -> "set[tuple[int, ...]]":
    """Return plausible LM-head shapes: actual weight first, then config vocab/hidden fallbacks."""
    shapes: set[tuple[int, ...]] = set()

    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is not None:
        shapes.add(tuple(int(dim) for dim in weight.shape))

    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    vocab_sizes = [
        getattr(config, "reduced_vocab_size", None),
        # EAGLE3 draft heads project to a reduced draft vocabulary; the ONNX
        # MatMul weight is then (hidden, draft_vocab), which the vocab_size /
        # reduced_vocab_size fallbacks alone do not cover.
        getattr(config, "draft_vocab_size", None),
        getattr(config, "vocab_size", None),
    ]
    for vocab_size in vocab_sizes:
        if hidden_size is None or vocab_size is None:
            continue
        hidden = int(hidden_size)
        vocab = int(vocab_size)
        shapes.add((vocab, hidden))
        shapes.add((hidden, vocab))
    return shapes


def _find_initializer_from_value(
        value_name: str, initializers: dict, producer_by_output: dict,
        candidate_shapes: "set[tuple[int, ...]]") -> str:
    """Trace simple view/cast values back to a candidate initializer."""
    if value_name in initializers:
        init = initializers[value_name]
        if tuple(init.dims) in candidate_shapes:
            return value_name
        return ""

    producer = producer_by_output.get(value_name)
    if producer is None or producer.op_type not in ("Cast", "Identity",
                                                    "Transpose"):
        return ""

    for input_name in producer.input:
        tensor_name = _find_initializer_from_value(input_name, initializers,
                                                   producer_by_output,
                                                   candidate_shapes)
        if tensor_name:
            return tensor_name
    return ""


def _find_lm_head_weight_initializer(onnx_model, model) -> "str | None":
    """Return the ONNX initializer that backs the final LM-head projection."""
    initializers = {init.name: init for init in onnx_model.graph.initializer}
    candidate_shapes = _lm_head_weight_candidate_shapes(model)
    if not candidate_shapes:
        return None

    shape_matches = [
        init.name for init in onnx_model.graph.initializer
        if tuple(init.dims) in candidate_shapes
    ]
    named_matches = [
        name for name in shape_matches
        if "lm_head" in name.replace("/", ".").lower()
    ]
    if len(named_matches) == 1:
        return named_matches[0]
    if len(shape_matches) == 1:
        return shape_matches[0]

    producer_by_output = {
        output_name: node
        for node in onnx_model.graph.node
        for output_name in node.output
    }

    # Start at the public logits output.  Exporters commonly insert a final
    # Cast or Identity, so walk through those before looking for the Gemm/MatMul
    # that implements the LM-head projection.
    logits_outputs = [
        output.name for output in onnx_model.graph.output
        if output.name == "logits"
    ]
    worklist = logits_outputs or [
        output.name for output in onnx_model.graph.output
    ]
    seen_values: set[str] = set()
    # LogSoftmax appears between logits and the head MatMul for EAGLE3 draft
    # models (which emit log-probs); treat it as a passthrough so the walk
    # reaches the projection weight.
    passthrough_ops = {
        "Cast", "Identity", "Reshape", "Transpose", "LogSoftmax"
    }

    while worklist:
        value_name = worklist.pop()
        if value_name in seen_values:
            continue
        seen_values.add(value_name)

        node = producer_by_output.get(value_name)
        if node is None:
            continue

        # For the final projection, the weight is normally the second Gemm or
        # MatMul input.  It can either be an initializer directly or the input
        # to a small Transpose inserted by ONNX export for torch.nn.Linear.
        if node.op_type in ("Gemm", "MatMul"):
            for input_name in list(node.input)[1:]:
                tensor_name = _find_initializer_from_value(
                    input_name, initializers, producer_by_output,
                    candidate_shapes)
                if tensor_name:
                    return tensor_name
            continue

        if node.op_type in passthrough_ops:
            worklist.extend(node.input[:1])
            continue

        if node.op_type == "Add":
            worklist.extend(input_name for input_name in node.input
                            if input_name not in initializers)

    if len(named_matches) > 1:
        logger.warning(
            "Multiple lm_head-shaped initializers matched by name: %s",
            named_matches)
    elif len(shape_matches) > 1:
        logger.warning("Multiple LM-head-shaped initializers found: %s",
                       shape_matches)
    return None


def _write_external_weight_file(out_dir: str, external_weight_file: str,
                                tensor_names: "list[str]",
                                initializers: dict) -> str:
    """Write selected ONNX initializers to a safetensors external weight file."""
    import numpy as np
    import onnx
    import torch
    from safetensors.torch import save_file

    external_weight_tensors = {}
    for tensor_name in tensor_names:
        init = initializers[tensor_name]
        array = onnx.numpy_helper.to_array(init)
        external_weight_tensors[tensor_name] = torch.from_numpy(
            np.array(array, copy=True))
    external_weight_path = os.path.join(out_dir, external_weight_file)
    save_file(external_weight_tensors, external_weight_path)
    return external_weight_path


def _add_external_weight_inputs(onnx_model, tensor_names: "list[str]",
                                initializers: dict) -> None:
    """Expose selected initializers as fixed-shape ONNX graph inputs."""
    import onnx

    graph_input_names = {inp.name for inp in onnx_model.graph.input}
    for tensor_name in tensor_names:
        if tensor_name not in graph_input_names:
            init = initializers[tensor_name]
            onnx_model.graph.input.append(
                onnx.helper.make_tensor_value_info(
                    tensor_name,
                    init.data_type,
                    list(init.dims),
                ))
            graph_input_names.add(tensor_name)


def _remove_externalized_initializers(onnx_model,
                                      tensor_names: "list[str]") -> None:
    """Remove selected tensors from graph initializers after adding inputs."""
    external_name_set = set(tensor_names)
    retained_initializers = [
        init for init in onnx_model.graph.initializer
        if init.name not in external_name_set
    ]
    del onnx_model.graph.initializer[:]
    onnx_model.graph.initializer.extend(retained_initializers)


def _save_externalized_onnx_model(onnx_path: str, onnx_model) -> None:
    """Save ONNX after all requested externalization passes are complete."""
    import onnx

    out_dir = os.path.dirname(os.path.abspath(onnx_path))
    ext_path = os.path.join(out_dir, "model.onnx.data")
    if os.path.isfile(ext_path):
        old_size = os.path.getsize(ext_path)
        logger.info("Removing stale external data %s (%.2f GB) before re-save",
                    ext_path, old_size / 1e9)
        os.remove(ext_path)
    onnx.save_model(
        onnx_model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        convert_attribute=True,
    )


def externalize_model_weights(
        onnx_path: str,
        model,
        externalize_weights=None) -> "list[dict[str, object]]":
    """Move requested ONNX initializer weights into external weight files."""
    import onnx

    requested = resolve_externalize_weights(externalize_weights)
    if not requested:
        return []

    onnx_model = onnx.load(onnx_path)
    initializers = {init.name: init for init in onnx_model.graph.initializer}
    out_dir = os.path.dirname(os.path.abspath(onnx_path))

    manifest: list[dict[str, object]] = []
    external_names: list[str] = []
    external_name_set: set[str] = set()

    def add_external_weight_file(kind: str, external_weight_file: str,
                                 tensor_names: "list[str]",
                                 manifest_kind: str) -> None:
        new_names = [
            name for name in tensor_names if name not in external_name_set
        ]
        if not new_names:
            return
        external_weight_path = _write_external_weight_file(
            out_dir, external_weight_file, new_names, initializers)
        external_names.extend(new_names)
        external_name_set.update(new_names)
        manifest.append({
            "file": external_weight_file,
            "kind": manifest_kind,
            "tensors": new_names,
        })
        logger.info("Externalized %d %s tensor(s) to %s", len(new_names), kind,
                    external_weight_path)

    for kind in requested:
        if kind == EXTERNAL_WEIGHT_INT4_FFN:
            names = _find_int4_ffn_weight_initializers(onnx_model, model)
            if not names:
                logger.warning(
                    "External int4_ffn weights requested, but no dense FFN "
                    "Int4GroupwiseGemmPlugin initializers were found")
                continue
            add_external_weight_file("FFN Int4GroupwiseGemmPlugin",
                                     "external_int4_ffn_weights.safetensors",
                                     names, "int4_ffn_weights")
            continue

        if kind == EXTERNAL_WEIGHT_INT4_MOE:
            names = _find_int4_moe_weight_initializers(onnx_model)
            if not names:
                logger.warning("External int4_moe weights requested, but no "
                               "Int4MoePlugin qweight initializers were found")
                continue
            add_external_weight_file("Int4MoePlugin qweight",
                                     "external_int4_moe_weights.safetensors",
                                     names, "int4_moe_weights")
            continue

        if kind == EXTERNAL_WEIGHT_LM_HEAD:
            tensor_name = _find_lm_head_weight_initializer(onnx_model, model)
            if tensor_name is None:
                logger.warning(
                    "External lm_head weight requested, but no unique "
                    "LM-head initializer was found")
                continue
            add_external_weight_file("LM-head weight",
                                     "external_lm_head_weight.safetensors",
                                     [tensor_name], "lm_head_weight")

    if not external_names:
        return []

    _add_external_weight_inputs(onnx_model, external_names, initializers)
    _remove_externalized_initializers(onnx_model, external_names)
    _save_externalized_onnx_model(onnx_path, onnx_model)
    return manifest
