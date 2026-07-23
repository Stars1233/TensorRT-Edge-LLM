#!/usr/bin/env python3
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
"""Shared helpers for the FP16 MoE grouped-GEMM AOT export."""

from __future__ import annotations

import pathlib

# Max expert count: sizes the AOT trace buffers and caps the persistent grid.
# Not baked into the cubin -- the runtime group_count argument carries E.
MAX_NUM_EXPERTS = 256
DESCRIPTOR_ALIGNMENT = 16
TENSORMAP_ALIGNMENT = 128
BYTES_PER_TENSORMAP = 128
TENSORMAPS_PER_BLOCK = 3


def get_max_active_clusters(family: str) -> int:
    """Return a conservative persistent grid size for one kernel family."""
    import cutlass

    hardware_info = cutlass.utils.HardwareInfo()
    if family == "ampere":
        # Ampere does not support CTA clusters; querying cluster occupancy
        # returns CUDA_ERROR_INVALID_CLUSTER_SIZE. The grouped scheduler uses
        # an ordinary one-CTA-per-SM persistent grid on this family.
        return min(MAX_NUM_EXPERTS,
                   hardware_info.get_device_multiprocessor_count())
    if family not in ("blackwell", "blackwell_geforce"):
        raise ValueError(f"Unsupported f16_moe family: {family}")
    return min(
        MAX_NUM_EXPERTS,
        hardware_info.get_max_active_clusters(1),
    )


def make_ptr(data_type, value: int, assumed_align: int | None = None):
    """Build a typed CuTe global-memory pointer for a CuPy allocation."""
    import cutlass.cute as cute

    try:
        import cute_utils
    except ImportError:
        import sys

        nvfp4_dir = pathlib.Path(
            __file__).resolve().parents[1] / "nvfp4_moe_cutedsl"
        sys.path.insert(0, str(nvfp4_dir))
        import cute_utils

    return cute_utils.make_ptr(
        data_type,
        value,
        cute.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def verify_export(output_dir: str, file_name: str) -> tuple[str, str]:
    """Require a CuTeDSL export to produce nonempty header and object files."""
    header = pathlib.Path(output_dir) / f"{file_name}.h"
    obj = pathlib.Path(output_dir) / f"{file_name}.o"
    missing = [
        str(path) for path in (header, obj)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"CuTeDSL export did not produce nonempty artifacts: {missing}")
    return str(header), str(obj)
