# CuTe DSL FMHA Kernels (Blackwell SM10X/SM110)

Fused multi-head attention kernels compiled ahead-of-time from CuTe DSL Python
source. Kernel artifacts (static library + headers) are pre-generated offline by
`build_static_lib.py` and checked into the repo. CMake simply links the
prebuilt artifacts — no Python, CUTLASS DSL, CuPy, or Blackwell GPU is needed
at CMake build time.

## Quick Start: Building the Kernel Library

Run on a machine with a **Blackwell or Thor GPU** (SM100, SM101/SM110) and CUDA 12.x or 13.x.

**1. Create a virtual environment and install dependencies**

```bash
python3 -m venv build_kernel_venv
source build_kernel_venv/bin/activate

pip install nvidia-cutlass-dsl==4.4.1

# Pick the cupy variant that matches your CUDA version:
pip install cupy-cuda12x==12.3.0  # CUDA 12.x
or
pip install cupy-cuda13x==13.6.0  # CUDA 13.x
```

**2. Compile all kernel variants into a static library**

> **GPU requirement:** compilation must run on a Blackwell datacenter GPU (e.g. B200, GB200)
> or on Thor (SM10X or SM110). Other GPUs are not supported.

```bash
cd tensorrt-edge-llm
python kernelSrcs/fmha_cutedsl_blackwell/build_static_lib.py [--clean] [-j 4]
```

This produces the following artifacts under
`cpp/kernels/contextAttentionKernels/cuteDSLArtifact/{arch}/`:

```
libcutedsl_fmha_{arch}.a          — all 8 kernel .o files + libcuda_dialect_runtime_static.a merged in
metadata.json                     — build provenance (CUDA ver, DSL ver, date)
include/
    cutedsl_fmha_all.h            — umbrella header
    fmha_d64.h, fmha_d128.h, ... — per-variant headers
```

Commit the generated `cuteDSLArtifact/{arch}/` directory to the repo so that
CMake builds on edge devices require no Python or GPU.

**3. Enable in CMake**

For Thor (cross-compiled from x86):

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=auto-thor \
    -DENABLE_CUTE_DSL_FMHA=ON
```

For x86 GPU systems:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION> \
    -DENABLE_CUTE_DSL_FMHA=ON
```

See [Section 2](#2-building) for full CMake options and details.

---

## Origin

`fmha.py` is derived from the CUTLASS example at
`examples/python/CuTeDSL/blackwell/fmha.py`, and `fmha_helpers.py` from
`examples/python/CuTeDSL/helpers/fmha_helpers.py`, both taken from CUTLASS commit
[`b9847690c5838ac3d909ebc163ed16c388802485`](https://github.com/NVIDIA/cutlass/commit/b9847690c5838ac3d909ebc163ed16c388802485).

Local modifications are captured in `fmha.patch`.

## Key Improvements

- **Runtime Parameter Flexibility** — converted batch size, sequence length, and
  number of heads from compile-time constants to runtime arguments, with
  negligible performance overhead.
- **Sliding Window Attention** — added sliding window attention support per
  Xiaomi's requirements.
- **Prefix Cache Optimization** — eliminated temporary KV cache allocation and
  layout conversion before FMHA, reducing memory footprint.
- **Dependency Removal** — removed PyTorch dependency for standalone C++
  execution.

## 1. Kernel Variants

The build produces eight AOT-compiled kernel objects (`.o` + `.h` pairs):

| Variant | Head Dim | SWA | Mode | Causal |
|---|---|---|---|---|
| `fmha_d64` | 64 | No | LLM | Yes |
| `fmha_d128` | 128 | No | LLM | Yes |
| `fmha_d64_sw` | 64 | Yes | LLM | Yes |
| `fmha_d128_sw` | 128 | Yes | LLM | Yes |
| `vit_fmha_d64` | 64 | No | ViT | No |
| `vit_fmha_d72` | 72 | No | ViT | No |
| `vit_fmha_d80` | 80 | No | ViT | No |
| `vit_fmha_d128` | 128 | No | ViT | No |

**LLM variants** use a fused KV cache layout `[B, 2, H_kv, S_k, D]` with causal
masking and bottom-right alignment (`WINDOW_MASK_INFERENCE`).

**ViT variants** use packed variable-length separate Q/K/V tensors
`[total_S, H, D]` with `cu_seqlens` for ragged batching, bidirectional attention.

## 2. Building

### 2.1. Generating Prebuilt Artifacts

Artifacts must be regenerated whenever the kernel source (`fmha.py`) or the
CUTLASS DSL version changes. Run on a machine with a Blackwell GPU:

```bash
cd tensorrt-edge-llm
python kernelSrcs/fmha_cutedsl_blackwell/build_static_lib.py [--clean] [-j 4]
```

**Prerequisites** (only needed when running `build_static_lib.py`):

| Dependency | Version | Notes |
|---|---|---|
| `nvidia-cutlass-dsl` | 4.4.1 | |
| `cupy-cuda12x` | 12.3.0 | CUDA 12.x |
| `cupy-cuda13x` | 13.6.0 | CUDA 13.x |

The script compiles all kernel variants in parallel, assembles them into a
static library, and copies the dialect runtime, placing everything under
`cpp/kernels/contextAttentionKernels/cuteDSLArtifact/{arch}/`:

```
cuteDSLArtifact/x86_64/
    libcutedsl_fmha_x86_64.a          — all 8 kernel .o files + libcuda_dialect_runtime_static.a merged in
    metadata.json                      — build provenance (CUDA ver, DSL ver, date)
    include/
        cutedsl_fmha_all.h             — umbrella header
        fmha_d64.h, fmha_d128.h, ...  — per-variant headers
```

Commit the `cuteDSLArtifact/{arch}/` directory after generation.

**Script options:**

| Flag | Default | Description |
|---|---|---|
| `--output_dir DIR` | `cpp/kernels/contextAttentionKernels/cuteDSLArtifact` | Root output dir |
| `--arch ARCH` | auto-detected | `x86_64` or `aarch64` |
| `-j JOBS` | `4` | Parallel compile jobs (use `-j 1` if GPU memory is limited) |
| `--verbose` | off | Show `fmha.py` output for each variant |
| `--clean` | off | Remove the arch-specific output dir before building |

### 2.2. CMake Configuration

No Python or GPU required. CMake links the prebuilt artifacts directly:

```bash
cmake -DENABLE_CUTE_DSL_FMHA=ON \
      -DTRT_PACKAGE_DIR=/path/to/TensorRT \
      ..
```

`ENABLE_CUTE_DSL_FMHA` defaults to `OFF`. Set it to `ON` to enable CuTe DSL
FMHA support (requires prebuilt artifacts to be present).

`cmake/CuteDslFMHA.cmake`:

1. Detects host CPU architecture.
2. Validates that `libcutedsl_fmha_{arch}.a` and `include/cutedsl_fmha_all.h`
   exist under `cuteDSLArtifact/{arch}/`.
3. Links the single static archive into the plugin and unit tests; defines
   `CUTE_DSL_FMHA_ENABLED`.

If artifacts are missing, CMake exits with a clear error pointing to
`build_static_lib.py`.

### 2.3. Standalone Kernel Compilation

To compile a single variant outside CMake (e.g. for testing):

```bash
cd kernelSrcs/fmha_cutedsl_blackwell

# LLM d128, no sliding window
python3 fmha.py \
  --q_shape 1,1024,14,128 --k_shape 1,1024,1,128 \
  --is_causal --is_persistent --bottom_right_align \
  --export_only --output_dir ./out --file_name fmha_d128 --function_prefix fmha_d128

# LLM d64, with sliding window
python3 fmha.py \
  --q_shape 1,1024,14,64 --k_shape 1,1024,1,64 \
  --is_causal --is_persistent --bottom_right_align \
  --window_size 4096,-1 \
  --export_only --output_dir ./out --file_name fmha_d64_sw --function_prefix fmha_d64_sw

# ViT d64
python3 fmha.py \
  --q_shape 1,1024,14,64 --k_shape 1,1024,14,64 \
  --is_persistent --vit_mode \
  --export_only --output_dir ./out --file_name vit_fmha_d64 --function_prefix vit_fmha_d64
```

Each invocation produces `<file_name>.h` and `<file_name>.o` in `--output_dir`.

To run reference accuracy checks without exporting artifacts (`--export_only` not
set):

```bash
# LLM accuracy reference: multi-round prefill check (plugin-aligned behavior)
python3 fmha.py \
  --q_shape 1,8,8,128 --k_shape 1,64,8,128 \
  --is_causal --is_persistent --bottom_right_align

# ViT accuracy reference: single-shot packed-input check
python3 fmha.py \
  --q_shape 1,8,8,72 --k_shape 1,8,8,72 \
  --is_persistent --vit_mode
```

## 3. Runtime Integration

### 3.1. C++ Runner

`CuteDslFMHARunner` (`cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.{h,cpp}`)
provides the C++ interface:

- **Module loading**: `loadLLMKernelModule()` / `loadViTKernelModule()` — loads
  the AOT-compiled CUDA libraries. Thread-safe (static, guarded by mutex).
- **Dispatch**: `canImplement(headSize, smVersion)` — returns `true` for
  SM >= 100 and head dim 64 or 128.
- **LLM run**: `run(qPtr, kvPtr, oPtr, cuKVSeqLens, stream, slidingWindowSize)`
  — dispatches to the appropriate d64/d128 + SWA/non-SWA variant.
- **ViT run**: `run(qPtr, kPtr, vPtr, oPtr, cuSeqLens, totalSeqLen, maxSeqLen, batchSize, stream)`
  — dispatches to the appropriate d64/d72/d80/d128 variant.

### 3.2. Plugin Integration

The attention plugin (`cpp/plugins/attentionPlugin/attentionPlugin.cpp`) uses
CuTe DSL FMHA as the primary path on Blackwell, with automatic fallback to
FMHA_v2:

1. At construction, checks `CUTE_DSL_FMHA_ENABLED` compile flag and
   `canImplement()`.
2. Attempts to load kernel modules; falls back to FMHA_v2 on failure.
3. At runtime, CuTe DSL path uses a dedicated RoPE kernel
   (`launchApplyRopeWriteKVSplitQKV`) that writes directly into the fused KV
   cache layout `[B, 2, H_kv, S, D]`.

### 3.3. Sliding Window Attention

- Plugin attribute `sliding_window_size`: `-1` means disabled (default).
- At the C++ runtime boundary, `-1` is converted to `INT_MAX`.
- Runner dispatches to `_sw` variants when `slidingWindowSize < INT_MAX`.
- `window_size_right` is always `0` (causal-only), baked as a compile-time
  constant.
- `bottom_right_align` is always enabled, producing correct masking for both
  normal prefill and chunked prefill.

## 4. Patch Details

The patch adapts the upstream FMHA example for ahead-of-time (AOT) compilation
and integration into TensorRT Edge-LLM:

- **Replace PyTorch with CuPy/NumPy** — removes the `torch` dependency entirely;
  GPU tensor operations use CuPy and CPU reference computations use NumPy.
- **Fused KV cache layout** — instead of separate K and V tensors `(B, S, H, D)`,
  uses a single interleaved KV cache `(B, 2, H_kv, S_k, D)`, eliminating
  temporary allocation and layout conversion at runtime.
- **Tensor-based kernel API** — `__call__` now accepts `cute.Tensor` objects
  (`q_tensor`, `kv_cache`, `o_tensor`) directly, extracting problem dimensions
  from tensor shapes rather than a separate `problem_size` tuple.
- **Dynamic tensor marking** — marks batch size, sequence length, and number of
  heads as dynamic dimensions (`mark_bshd_dynamic`, `mark_kv_cache_dynamic`),
  allowing these to be runtime arguments instead of compile-time constants.
- **Compile-time sliding window dispatch** — adds a `use_sliding_window` flag;
  when `False`, `window_size_left` is passed as `None` at compile time to
  eliminate left-side window masking code for better performance.
- **AOT export support** — adds `--output_dir`, `--export_only`, `--file_name`,
  and `--function_prefix` CLI arguments. `export_to_c()` is called only when
  `--export_only` is set, producing `.h` and `.o` artifacts in `--output_dir`.
  `--export_only` also skips the reference check and benchmarking.
- **Causal-only window_size_right** — `window_size_right` is always 0 (causal)
  and set as a compile-time constant.
- **Remove variable-length sequence support** — `cum_seqlen_q`/`cum_seqlen_k`
  (nested tensor) paths are removed.
- **ViT mode** — `--vit_mode` compiles a separate variant with packed varlen
  separate Q/K/V and bidirectional (non-causal) attention for vision
  transformer workloads.

## 5. File Map

| File | Description |
|---|---|
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` | CuTe DSL kernel source (LLM + ViT variants) |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha_helpers.py` | Helper utilities from CUTLASS |
| `kernelSrcs/fmha_cutedsl_blackwell/build_static_lib.py` | Pre-build script: compiles all variants and assembles the static library |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.patch` | Diff against upstream CUTLASS example |
| `kernelSrcs/fmha_cutedsl_blackwell/fp8_prescale.patch` | FP8 pre-scaling patch (future) |
| `cmake/CuteDslFMHA.cmake` | CMake module: validates and links prebuilt artifacts |
| `cpp/kernels/contextAttentionKernels/cuteDSLArtifact/{arch}/` | Prebuilt artifacts (committed to repo) |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.h` | C++ runner header |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.cpp` | C++ runner implementation |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | TRT plugin integration |
| `cpp/kernels/posEncoding/applyRopeWriteKV.cu` | RoPE kernel for CuTe DSL KV layout |
