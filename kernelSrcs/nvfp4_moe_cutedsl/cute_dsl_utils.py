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

"""Shared utilities for NVFP4 MoE decode GEMV kernels (CuTe DSL).

Adapted from nvfp4_fused_moe_cutedsl/cute_dsl_utils.py with only the
subset needed for the scalar GEMV kernel (no MMA/TMA/cluster helpers).
"""

from __future__ import annotations

import ctypes
import functools
import os
from typing import Union

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type

# Register CuPy ndarray as a JIT argument adapter so CuPy arrays can be
# passed directly to compiled CuTe DSL kernels at runtime.
# (cutlass 4.4.1 only registers numpy.ndarray and torch.Tensor by default.)
import cupy as _cp
from cutlass.cute.runtime import TensorAdapter as _TensorAdapter
from cutlass.base_dsl.runtime.jit_arg_adapters import (
    JitArgAdapterRegistry as _Registry,
)
_Registry.register_jit_arg_adapter(_cp.ndarray)(_TensorAdapter)


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def _cuda_device_id(device: int | str | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    device_str = str(device)
    if device_str == "cuda":
        return None
    if device_str.startswith("cuda:"):
        return int(device_str.split(":", 1)[1])
    return int(device)


def _ensure_cuda_context(device: int | str | None = None):
    import cupy as cp

    device_id = _cuda_device_id(device)
    if device_id is None:
        device_id = cp.cuda.runtime.getDevice()
    cp.cuda.Device(device_id).use()
    cp.cuda.runtime.free(0)
    return cp, device_id


@functools.cache
def get_compute_capability(device: int | str | None = None) -> int:
    cp, device_id = _ensure_cuda_context(device)
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return int(props["major"]) * 10 + int(props["minor"])


@functools.cache
def get_num_sm(device: int | str | None = None) -> int:
    cp, device_id = _ensure_cuda_context(device)
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    return props["multiProcessorCount"]


def cute_compile_options(default: str = "--opt-level 2") -> str:
    options = default
    gpu_arch = os.environ.get("EDGE_LLM_CUTE_DSL_GPU_ARCH")
    ptxas_options = os.environ.get("EDGE_LLM_CUTE_DSL_PTXAS_OPTIONS")
    if gpu_arch:
        options += f" --gpu-arch={gpu_arch}"
    if ptxas_options:
        options += f" --ptxas-options='{ptxas_options}'"
    return options


# WAR for CuTeDSL make_ptr implementation
class _Pointer(Pointer):
    """Runtime representation of a pointer for AOT tracing."""

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned"
        )

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(
            self._dtype.mlir_type, self._addr_space, self._assumed_align
        )

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        if expected_py_type is Pointer or (
            isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer
        ):
            return True
        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Create a pointer from a memory address for AOT tracing.

    :param dtype: Data type of the pointer elements
    :param value: Memory address as integer or ctypes pointer
    :param mem_space: Memory address space, defaults to AddressSpace.generic
    :param assumed_align: Alignment in bytes, defaults to None
    :return: A pointer object
    """
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}"
        )

    return _Pointer(address_value, dtype, mem_space, assumed_align=assumed_align)
