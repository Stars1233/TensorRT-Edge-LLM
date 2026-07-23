# CuTe DSL FMHA Kernels (Blackwell SM10X/SM110)

Fused multi-head attention kernels compiled ahead-of-time from CuTe DSL Python
source. Kernel artifacts (static library + headers) are generated locally by
`kernelSrcs/build_cutedsl.py`. CMake simply links those local artifacts — no
Python, CUTLASS DSL, CuPy, or Blackwell GPU is needed at CMake build time.

## Quick Start: Building the Kernel Library

Run on a machine with a **Blackwell or Thor GPU** (SM100, SM101/SM110) and CUDA 12.x or 13.x.

**1. Create a virtual environment and install dependencies**

```bash
python3 -m venv build_kernel_venv
source build_kernel_venv/bin/activate

# Pick cuda-python and cupy variant that matches your CUDA version
# before installing `nvidia-cutlass-dsl`:
pip install cuda-python==12.8.* cupy-cuda12x==12.3.0 # CUDA 12.x
# or
pip install cuda-python cupy-cuda13x==13.6.0 # CUDA 13.x

# CUDA 13: install the [cu13] extra. CUDA 12: install the base package.
pip install 'nvidia-cutlass-dsl[cu13]==4.6.0'  # CUDA 13.x
# CUDA 12.x: pip install 'nvidia-cutlass-dsl==4.6.0'
```

**2. Compile all kernel variants into a static library**

> **GPU requirement:** compilation must run on a Blackwell datacenter GPU (e.g. B200, GB200)
> or on Thor (SM100 or SM110). Other GPUs are not supported.

```bash
cd tensorrt-edge-llm
python kernelSrcs/build_cutedsl.py --kernels fmha --gpu_arch sm_100 [--clean] [-j 4]
```

This produces the following artifacts under
`cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/`:

`artifact_tag` is currently `sm_<NN>` (for example `sm_100`, `sm_110`, or
`sm_121`).

```
libcutedsl_{arch}.a          — all FMHA kernel .o files + libcuda_dialect_runtime_static.a merged in
metadata.json                — build provenance (CUDA ver, DSL ver, date, groups)
include/
    cutedsl_all.h            — umbrella header
    fmha_d64.h, fmha_d128.h, ... — per-variant headers
```

Keep the generated `cuteDSLArtifact/{arch}/{artifact_tag}/` directory locally so
CMake builds can reuse it without re-running Python each time.

**3. Enable in CMake**

For Thor (cross-compiled from x86):

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=auto-thor \
    -DENABLE_CUTE_DSL=fmha
```

For x86 GPU systems:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION> \
    -DENABLE_CUTE_DSL=fmha
```

To enable both FMHA and GDN:

```bash
cmake .. -DENABLE_CUTE_DSL=ALL ...
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

`fmha.py` contains two independently compiled kernel classes. The generic class
handles D32-D128, while `BlackwellFusedMultiHeadAttentionForwardD256` uses a
D256-specific TMEM and pipeline layout. The host selects the class before
`cute.compile`, so the generated hot kernels contain no head-dimension branch.

| Input / KV dtype | KV layout | Head dims | Variants per head dim |
|---|---|---|---|
| FP16 | Contiguous | 64, 128, 256 | regular, sliding window |
| FP8 E4M3 | Contiguous | 64, 128, 256 | regular, sliding window |
| FP16 | Paged | 64, 128, 256 | regular, sliding window |
| FP8 E4M3 | Paged | 64, 128, 256 | regular, sliding window |
| FP16 | Packed ViT | 64, 72, 80, 128 | bidirectional |

The eight D256 AOT names are `fmha_d256`, `fmha_d256_sw`,
`fmha_d256_fp8`, `fmha_d256_sw_fp8`, `fmha_d256_paged`,
`fmha_d256_sw_paged`, `fmha_d256_paged_fp8`, and
`fmha_d256_sw_paged_fp8`. D256 ViT is not supported.

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
python kernelSrcs/build_cutedsl.py --kernels fmha --gpu_arch sm_100 [--clean] [-j 4]
```

**Prerequisites** (only needed when running `build_cutedsl.py`):

| Dependency | Version | Notes |
|---|---|---|
| `nvidia-cutlass-dsl` | 4.6.0 | CUDA 13: `[cu13]` extra; CUDA 12: base package |
| `cupy-cuda12x` | 12.3.0 | CUDA 12.x |
| `cupy-cuda13x` | 13.6.0 | CUDA 13.x |

The script compiles all FMHA kernel variants in parallel, assembles them into a
static library merged with the DSL runtime, placing everything under
`cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/`:

```
cuteDSLArtifact/x86_64/sm_100/
    libcutedsl_x86_64.a            — all kernel .o files + libcuda_dialect_runtime_static.a merged in
    metadata.json                  — build provenance (CUDA ver, DSL ver, date, groups)
    include/
        cutedsl_all.h              — umbrella header
        fmha_d64.h, fmha_d128.h, ... — per-variant headers
```

Keep the `cuteDSLArtifact/{arch}/{artifact_tag}/` directory after generation so
later CMake builds can reuse it.

**Script options:**

| Flag | Default | Description |
|---|---|---|
| `--kernels GROUPS` | `ALL` | `fmha`, `gdn`, `fmha,gdn`, or `ALL` |
| `--gpu_arch SM` | `` (device native) | e.g. `sm_100` for Blackwell; omit on-device |
| `--output_dir DIR` | `cpp/kernels/cuteDSLArtifact` | Root output dir (artifacts go under `{DIR}/{arch}/sm_<NN>/`) |
| `--arch ARCH` | auto-detected | `x86_64` or `aarch64` |
| `-j JOBS` | `4` | Parallel compile jobs (use `-j 1` if GPU memory is limited) |
| `--verbose` | off | Show `fmha.py` output for each variant |
| `--clean` | off | Remove the arch-specific output dir before building |

### 2.2. CMake Configuration

No Python or GPU required at CMake time. CMake links the locally generated
artifacts directly:

```bash
cmake -DENABLE_CUTE_DSL=fmha \
      -DTRT_PACKAGE_DIR=/path/to/TensorRT \
      ..
```

`ENABLE_CUTE_DSL` defaults to `OFF`. Values: `OFF`, `fmha`, `gdn`, `fmha;gdn`, `ALL`.

`cmake/CuteDsl.cmake`:

1. Detects host CPU architecture.
2. Resolves the artifact tag from `CUTE_DSL_ARTIFACT_TAG` or the target
   platform when unambiguous.
3. Reads `metadata.json` to determine which groups are available.
4. Validates that `libcutedsl_{arch}.a` and `include/cutedsl_all.h`
   exist under `cuteDSLArtifact/{arch}/{artifact_tag}/`.
5. Links the single static archive into the plugin and unit tests; defines
   `CUTE_DSL_FMHA_ENABLED` (and/or `CUTE_DSL_GDN_ENABLED`).

If artifacts are missing, CMake exits with a clear error pointing to
`build_cutedsl.py`.

If multiple artifact tags exist for the same CPU architecture, pass
`-DCUTE_DSL_ARTIFACT_TAG=<tag>` explicitly.

### 2.3. Standalone Kernel Compilation

To compile a single variant outside the build script (e.g. for testing):

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


## 5. Skip-Softmax (BLASST) Threshold Calibration

The kernel implements BLASST skip-softmax ([arXiv:2512.12087](https://arxiv.org/abs/2512.12087)):
with `skip_softmax_threshold` (lambda) set at construction, a KV tile whose local
row max falls below the running max by more than `ln(lambda)` is skipped whole
(exp / row-sum / P*V elided). `None` (default) compiles the feature out — the
kernel is bit-identical to the dense build. Restricted to plain causal attention
(constructor assert; no sliding window, no ViT/bidirectional) and used by the
prefill/context path only.

`calibrate_skip_softmax.py` covers the full lambda lifecycle with two
subcommands and staged, verbose output:

```
calibrate (default) ── ModelOpt official calibration ──▶ a, b, deploy lambda
      │                                                        │
      │                                    bake lambda into build_cutedsl.py,
      │                                    rebuild artifact + relink (manual)
      ▼                                                        ▼
evaluate ── RULER accuracy of the deployed engine ──▶ PASS/FAIL + recommendation
```

### `calibrate` — lambda via ModelOpt (official)

A fixed lambda yields wildly different sparsity across context lengths, so the
threshold follows `lambda = scale_factor / L` with a model-specific scale
factor. The subcommand wraps the official calibration in
`modelopt.torch.sparsity.attention_sparsity` (the same machinery behind
TensorRT-LLM's `threshold_scale_factor`): ModelOpt auto-generates a RULER
calibration set (default 24 samples across power-of-2 length bins), runs one
forward pass evaluating 20 built-in threshold trials at once, and fits
`scale_factor = a * exp(b * sparsity)` with scipy. Requires `torch`,
`transformers`, `nvidia-modelopt`, `scipy`, `wonderwords`; the model loads
with `attn_implementation="eager"`.

```bash
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
    --model-dir /path/to/Qwen3-1.7B --max-seqlen 4096 \
    --target-sparsity 0.3 0.5 --max-context 4096 \
    --cache-dir /path/with/room/modelopt-cache   # RULER gen cache; ModelOpt
                                                 # defaults to ~/.cache (quota!)
# [calibrate 1/3] load model ... [calibrate 2/3] ModelOpt calibration
#   (library output is dim and '│'-indented, this tool's lines are plain)
# [calibrate 3/3] fitted parameters and deployment thresholds
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ a = 115.037   b = 4.6992   R^2 = 0.733   (278 points)         ┃
# ┃ observed sparsity range: [10.3%, 74.7%]  (beyond = extrapolated)
# ┃ target  30%  max_ctx 4096    lambda = 0.115008  (log2 -3.12)  ┃
# ┃ target  50%  max_ctx 4096    lambda = 0.294372  (log2 -1.76)  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

ModelOpt's sparsity is a simulated, all-layers-pooled metric — the deployed
kernel's per-layer skip ratio at the same lambda can differ substantially.
Treat the calibrated lambda as the ecosystem-consistent starting point and let
`evaluate` arbitrate which target actually deploys.

### Deploying a candidate lambda

`lambda` is baked at AOT-compile time (there is no runtime knob): add
`--skip_softmax_threshold <lambda>` to the `fmha_d64`/`fmha_d128` variant args
in `kernelSrcs/build_cutedsl.py`, rebuild the artifact
(`build_cutedsl.py --kernels fmha`), and relink with `ENABLE_CUTE_DSL=fmha`.

### `evaluate` — RULER accuracy verdict for the deployed engine

The paper's accuracy instrument is RULER (its ~50%-sparsity safe-zone
conclusions come from it; retrieval-style tasks degrade first). The subcommand
samples real RULER items (HF `simonjegou/ruler`, tokenizer-filtered to the
engine's max input length), runs the deployed engine greedily, scores by
exact-answer matching per task, and — given a baseline — prints a PASS/FAIL
verdict plus a deployment recommendation (exit code follows, so it can gate
CI). Pair with `llm_bench --mode prefill` for TTFT.

```bash
# 1) dense baseline: save its scores
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
    --model-dir /path/to/Qwen3-1.7B \
    --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
    --max-context 4096 --save-results ruler_dense.json

# 2) each skip build: compare, get the verdict
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
    --model-dir /path/to/Qwen3-1.7B \
    --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
    --max-context 4096 --baseline ruler_dense.json --label "lambda=0.115"
# ...per-task score table with baseline/delta columns...
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ 
# ┃ VERDICT: PASS [lambda=0.115]                          ┃
# ┃ overall  0.7685 -> 0.7653   drop +0.0032  (gate 0.03) ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# 
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ VERDICT: FAIL [lambda=0.294]                          ┃
# ┃ overall  0.7685 -> 0.7147   drop +0.0537  (gate 0.03) ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# RECOMMENDATION: this build [lambda=0.115] is validated for deployment. ...
```

### Reference result (Qwen3-1.7B NVFP4, max_context 4096, B200)

Calibrated `a = 115.0, b = 4.70` (R² 0.73). RULER 200 samples x 10 tasks,
gate 0.03; TTFT from `llm_bench --mode prefill --inputLen 4096`:

| build | RULER overall | verdict | TTFT S=4096 |
|---|---|---|---|
| dense | 0.7685 | baseline | 12.18 ms |
| lambda=0.115 (target 30%) | 0.7653 (-0.003) | **PASS** | 11.70 ms (1.04x) |
| lambda=0.294 (target 50%) | 0.7147 (-0.054, qa/multiquery collapse) | **FAIL** | 11.68 ms (1.04x) |

Kernel time is threshold-insensitive at these shapes, so deploy the SMALLEST
lambda that passes the gate — a larger lambda buys no speed and only spends
accuracy margin.

## 6. File Map

| File | Description |
|---|---|
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` | CuTe DSL kernel source (LLM + ViT variants) |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha_helpers.py` | Helper utilities from CUTLASS |
| `kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py` | Skip-softmax threshold scale-factor calibration tool |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.patch` | Diff against upstream CUTLASS example |
| `kernelSrcs/fmha_cutedsl_blackwell/fp8_prescale.patch` | FP8 pre-scaling patch (future) |
| `kernelSrcs/build_cutedsl.py` | Unified pre-build script: compiles all CuTe DSL variants (FMHA + GDN) |
| `cmake/CuteDsl.cmake` | Unified CMake module: validates and links prebuilt artifacts |
| `cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/` | Local artifacts generated by `build_cutedsl.py` |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.h` | C++ runner header |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.cpp` | C++ runner implementation |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | TRT plugin integration |
| `cpp/kernels/posEncoding/applyRopeWriteKV.cu` | RoPE kernel for CuTe DSL KV layout |
