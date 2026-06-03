# SSD (Structured State Space Duality) CuTe DSL Kernels

CuTe DSL implementation of the Mamba2 SSD chunk-scan prefill kernel for
TensorRT Edge-LLM. Prebuilt artifacts (static library + headers) are checked
into the repo; CMake links them directly — no Python or GPU needed at build time.

Adapted from:
- [Mamba SSM Triton kernels](https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/triton/) (Apache-2.0) — SM80+ chunk scan pipeline
- [FlashInfer Mamba2 Blackwell kernel](https://github.com/flashinfer-ai/flashinfer/pull/2709) (Apache-2.0) — SM100+ persistent kernel

Local modifications:
- **CuTe DSL port** — rewrote Triton kernels as CuTe DSL with SM80 warp MMA + cp.async, removing PyTorch/Triton dependency
- **Multi-variant AOT compilation** -- 4 SM80 variants (D x N, each in {64, 128}) + 4 Blackwell variants (D=64, N in {64, 128}, has_init_states in {false, true}), each a compile-time specialization
- **Runtime parameter flexibility** -- batch, nheads, ngroups, seq_len, `context_lengths` (per-batch) are runtime arguments
- **End-to-end varlen support** -- kernel always-varlen at AOT (`has_varlen=True`); plugin plumbs `context_lengths` to runner; metadata (seq_idx / chunk_indices / chunk_offsets / seq_chunk_cumsum) built fully on-device (CUDA-graph friendly, no D2H sync)
- **`has_init_states` Blackwell variants** -- accept caller-provided initial SSM state at chunk 0; powers chunked prefill / continuous batching unit tests; default zero-state variant is the production fast path
- **Blackwell N=64 support** — fixed TMA partition shape mismatch in FlashInfer kernel to support dstate=64
- **C++ plugin integration** — CuteDslSSDRunner with multi-module dispatch, AOT static library pattern matching FMHA/GDN
- **Dependency removal** — removed PyTorch; uses CuPy/NumPy for standalone testing

## Kernel Variants

`DIM` (headdim) and `DSTATE` (state dim) are **compile-time** constants that
define each AOT variant. All other parameters (batch, heads, groups, seq_len)
are **runtime** arguments.

### Non-Blackwell (SM80+)

| Variant | DIM | DSTATE | Notes |
|---|---|---|---|
| `ssd_prefill_d128_n128` | 128 | 128 | Nemotron-Nano-9B-v2 |
| `ssd_prefill_d64_n128` | 64 | 128 | Nemotron-3-Nano-4B, 30B-A3B |
| `ssd_prefill_d128_n64` | 128 | 64 | — |
| `ssd_prefill_d64_n64` | 64 | 64 | — |

### Blackwell (SM100-110 — TMA + TMEM + WGMMA)

SM120+ (GB10/GB20) lacks TMEM/wgmma and uses the non-Blackwell fallback.

| Variant | DIM | DSTATE | `has_init_states` | Notes |
|---|---|---|---|---|
| `ssd_prefill_blackwell_d64_n128` | 64 | 128 | false | Nemotron-3-Nano-4B, 30B-A3B production fast path |
| `ssd_prefill_blackwell_d64_n128_init_states` | 64 | 128 | true | Chunked prefill / continuous batching |
| `ssd_prefill_blackwell_d64_n64` | 64 | 64 | false | -- |
| `ssd_prefill_blackwell_d64_n64_init_states` | 64 | 64 | true | Chunked prefill / continuous batching |

Blackwell native kernels are limited to DIM=64 due to SM100 TMEM capacity
(512 columns). DIM=128 models use the non-Blackwell fallback on Blackwell GPUs.

`has_init_states` is a compile-time constexpr: `false` is the production fast
path (state arrives zeroed); `true` adds an extra TMA pipeline stage to load
the initial state at chunk 0 (~30% slower vs zero-state). Runner dispatches
on `params.has_init_states`.

### Compile-time vs Runtime Parameters

| Parameter | Compile / Runtime | Affects variant? |
|---|---|---|
| `DIM` (headdim) | Compile-time | Yes |
| `DSTATE` (state dim) | Compile-time | Yes |
| `has_init_states` (Blackwell only) | Compile-time | Yes (x2 binaries per DxN) |
| `has_varlen` (Blackwell only) | Compile-time (always `True`) | No (single binary handles uniform + varlen) |
| `CHUNK_SIZE` | Compile-time (fixed 128) | No (same for all) |
| `batch` (n) | Runtime | No |
| `nheads` | Runtime | No |
| `ngroups` | Runtime | No |
| `seq_len` | Runtime | No |
| `context_lengths` (per-batch valid lengths) | Runtime | No (degenerate metadata for uniform) |

## Chunk Scan Pipeline

5-kernel tiled matmul pipeline (CHUNK_SIZE=128):

| Step | Kernel | Operation |
|------|--------|-----------|
| 1 | `cumsum` | Prefix sum of `A*dt` → decay factors per chunk |
| 2 | `chunk_state` | `B^T @ (decay*dt*X)` → per-chunk state contribution |
| 3 | `state_passing` | Sequential scan over chunks (`nchunks` only) |
| 4 | `bmm` | `C @ B^T` → CB matrix |
| 5 | `chunk_scan` | `(CB*mask)@X + C@state` → output |

Features: FP16 I/O, FP32 accumulation, D skip connection (`has_D`),
z-gating/SiLU (`has_z`), AOT export for C++ plugin integration.

## Building Prebuilt Artifacts

Run on a machine with the target GPU and CUDA 12.x/13.x:

```bash
# Pick cuda-python and cupy variant that matches your CUDA version
# before installing `nvidia-cutlass-dsl`:
pip install cuda-python==12.8.* cupy-cuda12x==12.3.0 # CUDA 12.x
# or
pip install cuda-python cupy-cuda13x==13.6.0 # CUDA 13.x

pip install nvidia-cutlass-dsl==4.5.1

cd tensorrt-edge-llm

# Non-Blackwell variants (all 4 D×N combos):
python kernelSrcs/build_cutedsl.py --kernels ssd --gpu_arch sm_87 [--clean]

# Blackwell variants (D=64 only):
python kernelSrcs/build_cutedsl.py --kernels ssd --gpu_arch sm_100 [--clean]
```

Output under `cpp/kernels/cuteDSLArtifact/{arch}/`:

```
libcutedsl_{arch}.a    — all variant .o files + libcuda_dialect_runtime_static.a
metadata.json          — build provenance (groups, variants, CUDA ver, DSL ver)
include/
    cutedsl_all.h      — umbrella header
    ssd_prefill_d128_n128.h
    ssd_prefill_d64_n128.h
    ...
```

## CMake

```bash
cmake -DENABLE_CUTE_DSL=ssd ...
```

To enable SSD with other kernel groups:

```bash
cmake -DENABLE_CUTE_DSL=ALL ...
```

## Standalone Test / Export

```bash
cd kernelSrcs/ssd_cutedsl

# Non-Blackwell accuracy check (production: D=64, N=128)
python3 ssd_prefill.py --n 1 --nheads 8 --dim 64 --dstate 128 --seq_len 1024

# Non-Blackwell varlen (per-batch context lengths)
python3 ssd_prefill.py --n 4 --nheads 8 --dim 64 --dstate 128 --seq_len 2048 \
    --cl 2048,1024,512,256

# Blackwell accuracy check (production fast path)
python3 ssd_prefill_blackwell.py --n 1 --nheads 8 --dim 64 --dstate 128 --seq_len 1024

# Blackwell with `has_init_states=true` (chunked prefill variant)
python3 ssd_prefill_blackwell.py --n 1 --nheads 8 --dim 64 --dstate 128 \
    --seq_len 1024 --has_init_states

# AOT export (single variant)
python3 ssd_prefill.py --export_only --dim 64 --dstate 128 \
    --output_dir ./out --file_name ssd_prefill_d64_n128 --function_prefix ssd_prefill_d64_n128
```

> Note: `ssd_prefill_blackwell.py` defaults to `--dim 128 --dstate 128` which
> exceeds SM100 TMEM (512 cols). Always pass `--dim 64` (production config) or
> see `kernelSrcs/build_cutedsl.py` for the supported Blackwell variant matrix.

## C++ Integration

`CuteDslSSDRunner` (`cpp/kernels/mamba/cuteDslSSDRunner.{h,cpp}`): call
`loadKernelModules()` once, then `runPrefill(SSDParams, stream)`. Dispatches
to the correct D×N variant at runtime. On SM100+, D=64 uses the Blackwell
native kernel; all other configs fall back to non-Blackwell.

Plugin (`cpp/plugins/mamba/mambaPlugin.cpp`): integrates via the SSD runner.

## Tensor Shapes

| Tensor | Shape | Dtype |
|---|---|---|
| `x` | `(N, seq_len, nheads, dim)` | FP16 |
| `dt` | `(N, seq_len, nheads)` | FP16 |
| `A` | `(nheads,)` | FP32 |
| `dt_bias` | `(nheads,)` | FP16 |
| `B` | `(N, seq_len, ngroups, dstate)` | FP16 |
| `C` | `(N, seq_len, ngroups, dstate)` | FP16 |
| `D` | `(nheads,)` | FP16 |
| `z` | `(N, seq_len, nheads, dim)` | FP16 |
| `output` | `(N, seq_len, nheads, dim)` | FP16 |
| `state` (in/out) | `(N, nheads, dim, dstate)` | FP16 |
| `context_lengths` | `(N,)` | INT32 |

## Files

| File | Description |
|------|-------------|
| `ssd_prefill.py` | Non-Blackwell 5-kernel chunk scan pipeline + AOT export |
| `ssd_prefill_blackwell.py` | Blackwell native kernel (TMA/TMEM/WGMMA) |
