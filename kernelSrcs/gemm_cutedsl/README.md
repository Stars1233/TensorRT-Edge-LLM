# CuTe DSL GEMM Kernels (Ampere / Blackwell / Blackwell GeForce)

FP16 GEMM kernels compiled ahead-of-time from CuTe DSL Python source for
replacing Qwen3-Omni Talker-side cuBLAS GEMM (`cublasGemmEx`).

These kernels implement:

`C = A @ B^T`

with:

- `A`: activation tensor
- `B`: row-major weight tensor
- `C`: output tensor
- FP16 input / FP16 output / FP32 accumulation

Kernel artifacts (static library + headers) are generated locally by
`kernelSrcs/build_cutedsl.py`. CMake links those local artifacts directly.

## Quick Start: Building the Kernel Library

Run on a machine with a supported GPU and CUDA 12.x or 13.x.

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv build_kernel_venv
source build_kernel_venv/bin/activate

# Pick cuda-python and cupy variant that matches your CUDA version
# before installing `nvidia-cutlass-dsl`:
pip install cuda-python==12.8.* cupy-cuda12x==12.3.0 # CUDA 12.x
# or
pip install cuda-python cupy-cuda13x==13.6.0 # CUDA 13.x

pip install nvidia-cutlass-dsl==4.5.1
```

If your environment hits a `ModuleNotFoundError` while importing CUTLASS DSL,
install:

```bash
pip install "jax[cpu]"
```

This is an environment workaround, not a GEMM kernel requirement.

### 2. Compile GEMM variants into a static library

```bash
cd tensorrt-edge-llm

# Build the GEMM variant(s) supported by the current GPU
python kernelSrcs/build_cutedsl.py --kernels gemm

# Or compile explicitly for a target SM
python kernelSrcs/build_cutedsl.py --kernels gemm --gpu_arch sm_80
python kernelSrcs/build_cutedsl.py --kernels gemm --gpu_arch sm_110
python kernelSrcs/build_cutedsl.py --kernels gemm --gpu_arch sm_121
```

This produces artifacts under:

`cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/`

where `artifact_tag` is currently `sm_<NN>` (for example `sm_80`, `sm_110`,
or `sm_121`).

```text
libcutedsl_{arch}.a          — merged static library (kernel .o + DSL runtime)
metadata.json                — build provenance and compiled variants
include/
    cutedsl_all.h            — umbrella header
    gemm_ampere_fp16.h
    gemm_blackwell_fp16.h
    gemm_bw_geforce_fp16.h
```

Keep the generated `cuteDSLArtifact/{arch}/{artifact_tag}/` directory locally so
subsequent CMake builds can reuse it. No git check-in is required.

## Supported Variants

| Variant | SM | Target GPU family | Kernel source |
|---|---|---|---|
| `gemm_ampere_fp16` | 80 / 86 / 87 / 89 | Ampere / Ada-like path | `gemm_ampere.py` |
| `gemm_blackwell_fp16` | 100 / 101 / 103 / 110 | Blackwell datacenter / Thor | `gemm_blackwell.py` |
| `gemm_bw_geforce_fp16` | 120 / 121 | Blackwell GeForce / GB10 | `gemm_blackwell_geforce.py` |

## Validation Status

The current implementation has been validated on:

| Variant | Hardware | Status |
|---|---|---|
| `gemm_ampere_fp16` | A100 (`SM80`) | Python run + AOT export passed |
| `gemm_blackwell_fp16` | Thor (`SM110`) | Python run + AOT export passed |
| `gemm_bw_geforce_fp16` | n1auto / GB10 (`SM121`) | Python run + AOT export passed |

## CMake Configuration

Enable GEMM CuTe DSL support via:

```bash
cmake -DENABLE_CUTE_DSL=gemm \
      -DTRT_PACKAGE_DIR=/path/to/TensorRT \
      ..
```

To enable multiple CuTe DSL groups together:

```bash
cmake -DENABLE_CUTE_DSL=ALL ...
cmake -DENABLE_CUTE_DSL="fmha;gdn;gemm" ...
```

`cmake/CuteDsl.cmake`:

1. Detects host CPU architecture.
2. Resolves the artifact tag (for example `sm_110` or `sm_121`) from
   `CUTE_DSL_ARTIFACT_TAG` or the target platform when unambiguous.
3. Reads `metadata.json`.
4. Validates that `libcutedsl_{arch}.a` and `include/cutedsl_all.h` exist under
   `cuteDSLArtifact/{arch}/{artifact_tag}/`.
5. Links the static library and defines:
   - `CUTE_DSL_GEMM_ENABLED`
   - `CUTE_DSL_GEMM_AMPERE_ENABLED`
   - `CUTE_DSL_GEMM_BLACKWELL_ENABLED`
   - `CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED`

If multiple artifact tags exist for the same CPU architecture, pass
`-DCUTE_DSL_ARTIFACT_TAG=<tag>` explicitly.

## Standalone Kernel Testing

### Ampere

```bash
cd kernelSrcs/gemm_cutedsl
python gemm_ampere.py --mnk 128,128,128 --skip_ref_check
python gemm_ampere.py --mnk 1,2048,2048 --skip_ref_check
python gemm_ampere.py --mnk 1,1024,2048 --skip_ref_check
```

### Blackwell datacenter / Thor

```bash
cd kernelSrcs/gemm_cutedsl
python gemm_blackwell.py --mnk 128,128,128 --skip_ref_check
```

### Blackwell GeForce / GB10

```bash
cd kernelSrcs/gemm_cutedsl
python gemm_blackwell_geforce.py --mnk 128,128,128 --skip_ref_check
```

### Single-variant AOT export

```bash
cd kernelSrcs/gemm_cutedsl

python gemm_ampere.py --mnk 256,512,128 \
  --export_only --output_dir ./out --file_name gemm_ampere_fp16 --function_prefix gemm_ampere_fp16

python gemm_blackwell.py --mnk 256,512,128 \
  --export_only --output_dir ./out --file_name gemm_blackwell_fp16 --function_prefix gemm_blackwell_fp16

python gemm_blackwell_geforce.py --mnk 256,512,128 \
  --export_only --output_dir ./out --file_name gemm_bw_geforce_fp16 --function_prefix gemm_bw_geforce_fp16
```

## Architecture Notes

### Ampere

Uses `cp.async` + `LdMatrix` + `MmaF16BF16Op`. Exported ABI is 2D:

- `A`: `[M, K]`
- `B`: `[N, K]`
- `C`: `[M, N]`

### Blackwell datacenter / Thor

Uses `tcgen05.mma` (UMMA) + TMA. Exported ABI is 3D with batch `L=1`:

- `A`: `[M, K, 1]`
- `B`: `[N, K, 1]`
- `C`: `[M, N, 1]`

Thor (`SM110`) requires explicitly using Blackwell-family shared-memory
capacity in the kernel because cuTe DSL 4.5.1 does not auto-detect it
reliably.

### Blackwell GeForce / GB10

Uses the `Sm120` / WGMMA-style path with TMA. Exported ABI is also 3D with
batch `L=1`:

- `A`: `[M, K, 1]`
- `B`: `[N, K, 1]`
- `C`: `[M, N, 1]`

## C++ Integration

`CuteDslGemmRunner` lives in:

- `cpp/kernels/talkerMLPKernels/cuteDslGemmRunner.h`
- `cpp/kernels/talkerMLPKernels/cuteDslGemmRunner.cpp`

It dispatches by runtime SM version:

- `SM80-89` → Ampere GEMM
- `SM100-119` → Blackwell GEMM
- `SM120+` → Blackwell GeForce GEMM

`talkerMLPKernels.cu` uses `CuteDslGemmRunner::run()` to implement:

- `invokeTalkerMLP()`
- `invokeLinearLayer()`

replacing the old cuBLAS-based path.

## Dependency Summary

### Required

| Dependency | Version |
|---|---|
| `nvidia-cutlass-dsl` | `4.5.1` |
| `cuda-python` | compatible with local CUDA |
| `cupy-cuda12x` | `12.3.0` for CUDA 12.x |
| `cupy-cuda13x` | `13.6.0` for CUDA 13.x |

### Optional

| Dependency | Purpose |
|---|---|
| `jax[cpu]` | Workaround if the installed CUTLASS DSL package import path requires `jax` |
