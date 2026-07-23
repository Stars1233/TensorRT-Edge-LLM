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
Shared harness for Edge-LLM TensorRT plugin unit tests.

Builds a single-plugin TensorRT engine from a declarative IO spec and drives it
using torch CUDA tensors for device memory (``tensor.data_ptr()`` for
``set_tensor_address`` and the current torch CUDA stream for
``execute_async_v3``).

Usage:
    runner = PluginRunner()
    runner.build(
        input_specs=[("x", trt.float16, (-1, 16))],
        output_names=["y"],
        plugin_name="MyPlugin", plugin_version="1",
        plugin_fields=[trt.PluginField("k", np.int32([3]), trt.PluginFieldType.INT32)],
        profiles={"x": ((1, 16), (4, 16), (8, 16))},
    )
    runner.execute({"x": x_gpu, "y": y_gpu})   # input shapes inferred from tensors
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

# Conditional imports for the GPU/TensorRT dependencies, so the module imports
# cleanly (and its tests skip) on a host where TensorRT or a CUDA-capable torch
# is unavailable, instead of failing collection.
try:
    import tensorrt as trt
    import torch

    DEPENDENCIES_AVAILABLE = torch.cuda.is_available()
    IMPORT_ERROR = None if DEPENDENCIES_AVAILABLE else "CUDA not available for torch"
except ImportError as e:  # pragma: no cover - exercised only without deps
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

    class _DummyModule:

        def __getattr__(self, name):
            return None

    trt = _DummyModule()
    torch = _DummyModule()

# --------------------------------------------------------------------------- #
# Plugin library discovery / registration
# --------------------------------------------------------------------------- #
_PLUGIN_LIB_NAME = "libNvInfer_edgellm_plugin.so"
# Default build directory. Set EDGELLM_PLUGIN_LIB to point at the library
# directly if it was built elsewhere.
_plugins_loaded = False


def find_plugin_library() -> Optional[str]:
    """Locate the Edge-LLM plugin shared library, or return None if missing.

    Honors EDGELLM_PLUGIN_LIB when set, then looks under build/ relative to the
    current working directory and to this file's repository root (the project
    is built into build/, see tests/README).
    """
    env_path = os.environ.get("EDGELLM_PLUGIN_LIB")
    if env_path:
        return os.path.abspath(env_path) if os.path.exists(env_path) else None
    candidates = [
        os.path.join("build", _PLUGIN_LIB_NAME),
        os.path.join(os.path.dirname(__file__), "..", "..", "build",
                     _PLUGIN_LIB_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def load_edgellm_plugins(logger) -> str:
    """Load the plugin library + initialize the TRT plugin registry (idempotent).

    Returns the resolved library path. Raises RuntimeError if not found.
    """
    global _plugins_loaded
    path = find_plugin_library()
    if path is None:
        raise RuntimeError(
            f"Could not find {_PLUGIN_LIB_NAME}. Build the project first, or "
            f"set EDGELLM_PLUGIN_LIB to the library path.")
    ctypes.CDLL(path)
    trt.init_libnvinfer_plugins(logger, "")
    _plugins_loaded = True
    return path


# --------------------------------------------------------------------------- #
# dtype helpers
# --------------------------------------------------------------------------- #
def _trt_to_torch_dtype_map():
    m = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    # Optional dtypes depending on TRT/torch versions.
    if hasattr(trt, "bfloat16") and hasattr(torch, "bfloat16"):
        m[trt.bfloat16] = torch.bfloat16
    if hasattr(trt, "fp8") and hasattr(torch, "float8_e4m3fn"):
        m[trt.fp8] = torch.float8_e4m3fn
    if hasattr(trt, "int64"):
        m[trt.int64] = torch.int64
    return m


def trt_dtype_to_torch(dtype):
    """Map a TensorRT DataType to the matching torch dtype."""
    return _trt_to_torch_dtype_map()[dtype]


def make_field(name: str, value, field_type) -> "trt.PluginField":
    """Build a trt.PluginField from a python scalar / sequence."""
    np_dtype = {
        trt.PluginFieldType.INT32: np.int32,
        trt.PluginFieldType.FLOAT32: np.float32,
    }[field_type]
    arr = np.asarray(value, dtype=np_dtype).reshape(-1)
    return trt.PluginField(name, arr, field_type)


def pf_int32(name: str, value) -> "trt.PluginField":
    return make_field(name, value, trt.PluginFieldType.INT32)


def pf_float32(name: str, value) -> "trt.PluginField":
    return make_field(name, value, trt.PluginFieldType.FLOAT32)


# --------------------------------------------------------------------------- #
# Engine runner
# --------------------------------------------------------------------------- #
@dataclass
class InputSpec:
    name: str
    dtype: object  # trt DataType
    shape: Tuple[int, ...]  # may contain -1 for dynamic dims


class PluginRunner:
    """Builds and executes a single-plugin TensorRT engine with torch buffers."""

    def __init__(self, verbose: bool = False):
        sev = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        self.logger = trt.Logger(sev)
        if not _plugins_loaded:
            load_edgellm_plugins(self.logger)
        self.engine = None
        self.context = None
        self.device = torch.device("cuda")

    def build(
        self,
        *,
        input_specs: Sequence[Tuple[str, object, Tuple[int, ...]]],
        output_names: Sequence[str],
        plugin_name: str,
        plugin_version: str,
        plugin_fields: Sequence["trt.PluginField"],
        profiles: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...],
                                  Tuple[int, ...]]],
        plugin_namespace: str = "",
        workspace_bytes: int = 1 << 30,
    ):
        """Construct the engine. ``input_specs`` order defines plugin input order.

        ``profiles`` maps each input name to (min, opt, max) shape tuples.
        ``output_names`` are assigned to plugin outputs 0..N in order.
        """
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace_bytes)
        # Required for in-place / aliased plugin IO (e.g. KV cache update).
        if hasattr(trt.PreviewFeature, "ALIASED_PLUGIN_IO_10_03"):
            config.set_preview_feature(
                trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)

        inputs = []
        for name, dtype, shape in input_specs:
            inputs.append(network.add_input(name, dtype, shape))

        registry = trt.get_plugin_registry()
        creator = registry.get_creator(plugin_name, plugin_version,
                                       plugin_namespace)
        if creator is None:
            raise RuntimeError(
                f"{plugin_name} v{plugin_version} not found in plugin registry"
            )
        fc = trt.PluginFieldCollection(list(plugin_fields))
        plugin = creator.create_plugin(plugin_name, fc,
                                       trt.TensorRTPhase.BUILD)
        if plugin is None:
            # The plugin is not built for this architecture (e.g. a CuTe DSL
            # plugin compiled only for newer SMs); skip rather than fail.
            pytest.skip(f"{plugin_name} plugin not available in this build")

        layer = network.add_plugin_v3(inputs, [], plugin)
        for i, oname in enumerate(output_names):
            layer.get_output(i).name = oname
            network.mark_output(layer.get_output(i))

        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in profiles.items():
            profile.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            # The engine could not be built for this config on this device
            # (e.g. FP8 KV cache on an SM without FP8 tensor cores); skip.
            pytest.skip(
                f"engine build unsupported for {plugin_name} on this device")
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        return self

    def execute(self,
                tensors: Dict[str, "torch.Tensor"],
                input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None):
        """Bind torch tensors to every IO tensor and run.

        Input shapes default to each input tensor's own shape. Aliased outputs
        are handled by passing the same torch tensor under both binding names.
        All tensors must be CUDA tensors with matching dtype/layout.
        """
        ctx = self.context
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in tensors:
                raise KeyError(f"Missing device tensor for binding '{name}'")
            t = tensors[name]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = (input_shapes or {}).get(name, tuple(t.shape))
                if not ctx.set_input_shape(name, shape):
                    raise RuntimeError(f"set_input_shape failed for {name}")
            if not ctx.set_tensor_address(name, t.data_ptr()):
                raise RuntimeError(f"set_tensor_address failed for {name}")
        # Execute on torch's current stream so the plugin is ordered after the
        # torch ops that produced the input tensors; a separate stream would not
        # be synchronized with them and could read partially-written inputs.
        stream = torch.cuda.current_stream()
        ok = ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
COS_SIM_THRESHOLD = 0.99999
# Element-wise abs/rel tolerance (allclose semantics). Complements the cosine
# check, which is insensitive to a global scale factor.
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2


def cosine_sim(expected: "torch.Tensor", actual: "torch.Tensor") -> float:
    """Cosine similarity between two tensors (flattened, fp32)."""
    e = expected.float().flatten().cpu()
    a = actual.float().flatten().cpu()
    denom = (e.norm() * a.norm()).clamp_min(1e-12)
    return float(torch.dot(e, a) / denom)


def assert_close(name: str,
                 expected: "torch.Tensor",
                 actual: "torch.Tensor",
                 atol: float = DEFAULT_ATOL,
                 rtol: float = DEFAULT_RTOL,
                 cos_threshold: float = COS_SIM_THRESHOLD):
    """Assert the plugin output matches the reference on two criteria:

    1. cosine similarity >= cos_threshold (default 0.99999) -- catches
       structural errors (wrong layout, dropped/extra terms, state errors);
    2. element-wise closeness ``|actual - expected| <= atol + rtol*|expected|``
       (allclose semantics) -- catches magnitude / scale errors that cosine,
       being scale-invariant, would miss.
    """
    e = expected.float()
    a = actual.float()
    cos = cosine_sim(e, a)
    diff = (e - a).abs()
    n_viol = int((diff > atol + rtol * e.abs()).sum())
    if cos < cos_threshold or n_viol:
        max_abs = float(diff.max()) if diff.numel() else 0.0
        raise AssertionError(
            f"{name}: cos_sim={cos:.6f} (need >= {cos_threshold}); "
            f"allclose violations={n_viol}/{e.numel()} "
            f"(atol={atol}, rtol={rtol}, max_abs={max_abs:.5f})")


# --------------------------------------------------------------------------- #
# Required batch / sequence-length cases
# --------------------------------------------------------------------------- #
# Each entry is (label, [per-row sequence lengths]), covering batch sizes
# 1/2/3/4/8 with both even and uneven length patterns.
RAGGED_CASES = [
    ("bs1", [1536]),
    ("bs2_even", [1024, 1024]),
    ("bs3_uneven", [10, 2048, 128]),
    ("bs4_even", [512, 512, 512, 512]),
    ("bs8_uneven", [10, 96, 240, 480, 800, 1200, 1664, 2048]),
]
MAX_BATCH = 8
MAX_SEQ = 2048


def poison_padding(tensors, context_lengths, value: float = 1e3):
    """Fill each row's padding region ``[context_lengths[b]:]`` with ``value``
    across all given ``[batch, seq, ...]`` tensors. Any kernel that reads past a
    row's valid length then corrupts the result, catching missing
    context-length masking. ``tensors`` may be a single tensor or an iterable.
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors, )
    for t in tensors:
        seq = t.shape[1]
        for b in range(t.shape[0]):
            valid = int(context_lengths[b])
            if valid < seq:
                t[b, valid:] = value
