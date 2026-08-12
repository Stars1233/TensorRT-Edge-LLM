# CuTe DSL GEMM Kernels (Ampere / Blackwell / Blackwell GeForce)

This directory hosts two CuTe DSL GEMM artifact groups, both compiled
ahead-of-time from Python source:

- **`gemm`** — FP16 GEMM that replaces the Qwen3-Omni Talker-side cuBLAS GEMM
  (`cublasGemmEx`). Documented in the sections below.
- **`gemm_nvfp4`** — NVFP4 blockscaled GEMM for Blackwell datacenter / Thor.
  See [NVFP4 Blockscaled GEMM](#nvfp4-blockscaled-gemm).

The FP16 `gemm` kernels implement:

`C = A @ B^T`

with:

- `A`: activation tensor
- `B`: row-major weight tensor
- `C`: output tensor
- FP16 input / FP16 output / FP32 accumulation

Kernel artifacts (static library + headers) are generated locally by
`kernelSrcs/build_cutedsl.py`. CMake links those local artifacts directly.

## Artifact Development

If you modify this kernel or its registry entries, manually regenerate the
`gemm` group before running CMake. Otherwise, CMake uses the matching prebuilt
tarball by default. Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow)
for the supported Docker and local-venv commands, dependency versions,
cross-compilation, artifact layout, and CMake configuration.

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

CMake derives the available GEMM implementation families from generated
metadata and defines `CUTE_DSL_GEMM_ENABLED` plus the applicable Ampere,
Blackwell, or Blackwell GeForce family definition.

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

## NVFP4 Blockscaled GEMM

The `gemm_nvfp4` group is a warp-specialized NVFP4 (blockscaled,
`sf_vec_size = 16`) GEMM for Blackwell datacenter and Thor, sourced from
`gemm_blackwell_nvfp4_ws.py`. It is kept in its own group so
`build_cutedsl.py --kernels gemm_nvfp4` builds independently of the FP16 `gemm`
variants. `M`, `N`, and `K` are runtime dimensions.

| Variant | Output dtype | MMA N-tile | Supported SMs |
|---|---|---|---|
| `gemm_blackwell_nvfp4_ws_fp16_tn64`  | FP16     | 64  | 100 / 101 / 103 / 110 |
| `gemm_blackwell_nvfp4_ws_fp16_tn128` | FP16     | 128 | 100 / 101 / 103 / 110 |
| `gemm_blackwell_nvfp4_ws_fp16_tn256` | FP16     | 256 | 100 / 101 / 103 / 110 |
| `gemm_blackwell_nvfp4_ws_fp8_tn64`   | FP8-E4M3 | 64  | 100 / 101 / 103 / 110 |
| `gemm_blackwell_nvfp4_ws_fp8_tn128`  | FP8-E4M3 | 128 | 100 / 101 / 103 / 110 |

The kernel body splits load / MMA / store across warp roles (epilogue warps
0–3, MMA warp 4, TMA-load warp 5) to hide TMA latency behind MMA issue; `tn256`
pairs the larger N-tile with the persistent tile scheduler.

CMake sets the umbrella `CUTE_DSL_GEMM_NVFP4_ENABLED` when the group is active,
plus a per-variant `CUTE_DSL_GEMM_BLACKWELL_NVFP4_WS_<DTYPE>_TN<N>_ENABLED`
define for each exported variant.

### Building

Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow),
substituting `gemm_nvfp4` for the group name. Single-variant export:

```bash
cd kernelSrcs/gemm_cutedsl
python gemm_blackwell_nvfp4_ws.py --mnk 128,512,128 \
  --mma_tiler_n 128 --sf_vec_size 16 --c_dtype fp16 \
  --export_only --output_dir ./out \
  --file_name gemm_blackwell_nvfp4_ws_fp16_tn128 \
  --function_prefix gemm_blackwell_nvfp4_ws_fp16_tn128
```
