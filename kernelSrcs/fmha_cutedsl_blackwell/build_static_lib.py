# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""AOT-compile CuTe DSL FMHA kernels into a static library for CMake linking.

Usage: python build_static_lib.py [--output_dir DIR] [--arch ARCH] [-j JOBS] [--verbose] [--clean]

Output: {output_dir}/{arch}/  libcutedsl_fmha_{arch}.a  libcuda_dialect_runtime_static.a
                               metadata.json             include/cutedsl_fmha_all.h  include/*.h

NOTE: Each fmha.py invocation uses the GPU. Use -j 1 if GPU memory is limited.
"""

import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_OUTPUT_DIR = (_SCRIPT_DIR / "../../cpp/kernels/contextAttentionKernels/cuteDSLArtifact").resolve()
_CUTLASS_DSL_VERSION = "4.4.1"
_CUPY_VERSIONS = {12: ("cupy-cuda12x", "12.3.0"), 13: ("cupy-cuda13x", "13.6.0")}

_LLM = ["--is_causal", "--is_persistent", "--export_only", "--bottom_right_align"]
_VIT = ["--is_persistent", "--export_only", "--vit_mode"]


@dataclass
class KernelVariant:
    name: str
    q_shape: str
    k_shape: str
    flags: list = field(default_factory=list)


KERNEL_VARIANTS = [
    KernelVariant("fmha_d64",      "1,1024,14,64",  "1,1024,1,64",   _LLM),
    KernelVariant("fmha_d128",     "1,1024,14,128", "1,1024,1,128",  _LLM),
    KernelVariant("fmha_d64_sw",   "1,1024,14,64",  "1,1024,1,64",   _LLM + ["--window_size", "4096,-1"]),
    KernelVariant("fmha_d128_sw",  "1,1024,14,128", "1,1024,1,128",  _LLM + ["--window_size", "4096,-1"]),
    KernelVariant("vit_fmha_d64",  "1,1024,14,64",  "1,1024,14,64",  _VIT),
    KernelVariant("vit_fmha_d72",  "1,1024,14,72",  "1,1024,14,72",  _VIT),
    KernelVariant("vit_fmha_d80",  "1,1024,14,80",  "1,1024,14,80",  _VIT),
    KernelVariant("vit_fmha_d128", "1,1024,14,128", "1,1024,14,128", _VIT),
]


def detect_arch(override=None):
    if override:
        m = override.lower().replace("-", "_")
        if m in ("x86_64", "amd64"): return "x86_64"
        if m in ("aarch64", "arm64"): return "aarch64"
        raise ValueError(f"Unsupported --arch: {override!r}. Use 'x86_64' or 'aarch64'.")
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"): return "x86_64"
    if m in ("aarch64", "arm64"): return "aarch64"
    raise RuntimeError(f"Unsupported architecture: {platform.machine()!r}. Use --arch to override.")


def check_dependencies():
    errors = []

    # nvidia-cutlass-dsl
    try:
        ver = importlib.metadata.version("nvidia-cutlass-dsl")
        if ver != _CUTLASS_DSL_VERSION:
            errors.append(f"nvidia-cutlass-dsl: found {ver}, need {_CUTLASS_DSL_VERSION}\n"
                          f"  Fix: pip install nvidia-cutlass-dsl=={_CUTLASS_DSL_VERSION}")
            lib_dir = None
        else:
            spec = importlib.util.find_spec("nvidia_cutlass_dsl")
            pkg_dir = Path(next(iter(spec.submodule_search_locations))) if spec.submodule_search_locations \
                else Path(spec.origin).parent
            lib_dir = pkg_dir / "lib"
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"nvidia-cutlass-dsl not found.\n"
                      f"  Fix: pip install nvidia-cutlass-dsl=={_CUTLASS_DSL_VERSION}")
        lib_dir, ver = None, "unknown"

    # cupy
    cuda_ver = _nvcc_version()
    if cuda_ver:
        major = int(cuda_ver.split(".")[0])
        if major in _CUPY_VERSIONS:
            cupy_pkg, cupy_req = _CUPY_VERSIONS[major]
            try:
                found = importlib.metadata.version(cupy_pkg)
                if found != cupy_req:
                    errors.append(f"cupy: found {cupy_pkg}=={found}, need {cupy_req}\n"
                                  f"  Fix: pip install {cupy_pkg}=={cupy_req}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"cupy not found.\n  Fix: pip install {cupy_pkg}=={cupy_req}")
        else:
            errors.append(f"Unsupported CUDA major version {major} for cupy.")
    else:
        errors.append("Could not detect CUDA version (is nvcc on PATH?).")

    if not shutil.which("ar"):
        errors.append("'ar' not found on PATH. Install binutils.")

    if errors:
        print("Dependency check failed:\n" + "\n".join(f"  • {e}" for e in errors))
        sys.exit(1)

    # All checks passed — these are guaranteed non-None past this point.
    assert ver is not None and lib_dir is not None and cuda_ver is not None
    print(f"  nvidia-cutlass-dsl=={ver} ✓  CUDA {cuda_ver} ✓  ar ✓")
    return ver, lib_dir, cuda_ver


def _nvcc_version():
    # Parse "V12.8.61" token from `nvcc --version` output.
    try:
        out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
        for token in out.split():
            if token.startswith("V") and token[1].isdigit():
                return token[1:].split(",")[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def _compile_one(variant, staging_dir, verbose):
    # Invoke fmha.py to AOT-compile one kernel variant into a .o + .h pair.
    # fmha.py drives the CuTe DSL JIT compiler on the local GPU and writes
    # cubin-embedded PTX into a relocatable object file.
    cmd = [sys.executable, str(_SCRIPT_DIR / "fmha.py"),
           "--q_shape", variant.q_shape, "--k_shape", variant.k_shape,
           "--output_dir", str(staging_dir),
           "--file_name", variant.name, "--function_prefix", variant.name,
           *variant.flags]
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(_SCRIPT_DIR), capture_output=not verbose, text=True)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        return variant.name, False, elapsed, (result.stderr or result.stdout or "")[-2000:]
    obj, hdr = staging_dir / f"{variant.name}.o", staging_dir / f"{variant.name}.h"
    if not obj.exists() or not hdr.exists():
        return variant.name, False, elapsed, f"{obj.name} / {hdr.name} not found after successful exit"
    return variant.name, True, elapsed, ""


def compile_variants(staging_dir, jobs, verbose):
    # Run all kernel variants in parallel using a process pool (each needs a GPU context).
    print(f"\nCompiling {len(KERNEL_VARIANTS)} kernel variants (jobs={jobs})...")
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_compile_one, v, staging_dir, verbose): v for v in KERNEL_VARIANTS}
        for future in concurrent.futures.as_completed(futures):
            name, ok, elapsed, msg = future.result()
            print(f"  {'✓' if ok else '✗'} {name:<20} ({elapsed:.1f}s)")
            if not ok:
                failures.append((name, msg))
    if failures:
        for name, msg in failures:
            print(f"\n  [{name}]\n{msg}")
        sys.exit(1)


def _check_obj_name_collision(kernel_objs, runtime_objs):
    # Detect basename collisions between kernel and runtime object files.
    # Two members with the same name in an archive cause silent linker issues
    # (only the first match is used). Abort early with a clear message if found.
    kernel_names = {f.name for f in kernel_objs}
    runtime_names = {f.name for f in runtime_objs}
    collision = kernel_names & runtime_names
    if collision:
        raise RuntimeError(
            f"Object name collision between kernel and runtime archives: {collision}\n"
            "Rename the affected kernel variant(s) in KERNEL_VARIANTS to resolve.")


def build(args):
    arch = detect_arch(args.arch)
    output_dir = Path(args.output_dir) / arch
    print(f"Target arch : {arch}\nOutput dir  : {output_dir}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    print("\nChecking dependencies...")
    dsl_ver, lib_dir, cuda_ver = check_dependencies()

    staging_dir = Path(tempfile.mkdtemp(prefix="cutedsl_fmha_"))
    try:
        compile_variants(staging_dir, args.jobs, args.verbose)

        # Extract libcuda_dialect_runtime_static.a objects into a subdirectory to avoid
        # filesystem collisions with kernel objects, then merge everything into one archive.
        runtime = lib_dir / "libcuda_dialect_runtime_static.a"
        if not runtime.exists():
            raise FileNotFoundError(f"{runtime} not found. Verify nvidia-cutlass-dsl installation.")
        runtime_obj_dir = staging_dir / "runtime_objs"
        runtime_obj_dir.mkdir()
        subprocess.run(["ar", "x", str(runtime)], cwd=str(runtime_obj_dir), check=True)
        runtime_objs = sorted(runtime_obj_dir.glob("*.o"))

        # Guard against archive member name collisions before packing.
        kernel_obj_files = [staging_dir / f"{v.name}.o" for v in KERNEL_VARIANTS]
        _check_obj_name_collision(kernel_obj_files, runtime_objs)

        # Pack kernel objects and runtime objects into a single static archive.
        # Delivering one .a simplifies CMake linking — no separate runtime lib needed.
        output_dir.mkdir(parents=True, exist_ok=True)
        lib_path = output_dir / f"libcutedsl_fmha_{arch}.a"
        subprocess.run(["ar", "rcs", str(lib_path)]
                       + [str(o) for o in kernel_obj_files]
                       + [str(o) for o in runtime_objs], check=True)
        print(f"\n  Created {lib_path.name} ({lib_path.stat().st_size // 1024} KB)")

        # Copy per-variant headers and write an umbrella header that includes them all.
        inc_dir = output_dir / "include"
        inc_dir.mkdir(exist_ok=True)
        for v in KERNEL_VARIANTS:
            shutil.copy2(staging_dir / f"{v.name}.h", inc_dir)
        umbrella = inc_dir / "cutedsl_fmha_all.h"
        umbrella.write_text("#pragma once\n// Auto-generated by build_static_lib.py -- do not edit\n"
                            + "".join(f'#include "{v.name}.h"\n' for v in KERNEL_VARIANTS))

        # Write build provenance so the artifact directory is self-describing.
        (output_dir / "metadata.json").write_text(json.dumps({
            "arch": arch, "cuda_version": cuda_ver, "cutlass_dsl_version": dsl_ver,
            "build_date": datetime.now(timezone.utc).isoformat(),
            "variants": [v.name for v in KERNEL_VARIANTS],
        }, indent=2) + "\n")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\nDone. Artifacts written to: {output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR),
                   help=f"Root output dir (artifacts go into {{output_dir}}/{{arch}}/). Default: {_DEFAULT_OUTPUT_DIR}")
    p.add_argument("--arch", default=None, help="x86_64 or aarch64 (default: auto-detected)")
    p.add_argument("-j", "--jobs", type=int, default=4,
                   help="Parallel compile jobs (use -j 1 if GPU memory is limited). Default: 4")
    p.add_argument("--verbose", action="store_true", help="Show fmha.py output per variant")
    p.add_argument("--clean", action="store_true", help="Remove output arch dir before building")
    build(p.parse_args())


if __name__ == "__main__":
    main()
