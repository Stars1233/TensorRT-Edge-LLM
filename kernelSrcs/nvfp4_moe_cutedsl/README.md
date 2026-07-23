# SM110 NVFP4 MoE CuTeDSL Kernels

This directory contains the Thor SM110 split FC1 / FC2 CuTeDSL backend used by
[`Nvfp4MoePlugin`](../../cpp/plugins/nvfp4MoePlugin/). It exports the
`nvfp4_moe` artifact group from [`kernelSrcs/build_cutedsl.py`](../build_cutedsl.py);
today the only variants in that group target SM110/Thor. The SM120 / SM121 fused decode + prefill kernels live in
[`kernelSrcs/nvfp4_fused_moe_cutedsl/`](../nvfp4_fused_moe_cutedsl/) and feed
[`NvFP4MoEPluginGeforce`](../../cpp/plugins/nvfp4MoePluginGeforce/) via the
`nvfp4_fused_moe` group.

Pipeline produced by the AOT pack:

1. **FC1 (gather grouped GEMM + activation + FP4 requant)** — gathers routed
   tokens by expert, runs the FP4 blockscaled grouped GEMM, applies SwiGLU /
   ReLU², and requantizes the intermediate activation back to NVFP4 (packed
   FP4 bytes + FP8-E4M3 scale-factor blocks).
2. **FC2 (grouped GEMM + router-scale finalize / scatter)** — applies
   `alpha = input_gsf * weight_gsf` in the kernel epilogue, multiplies by the
   per-token router weight, and scatter-reduces the result back to the original
   token layout, emitting `[T, H]` FP16.

## 1. Supported Hardware

| GPU | SM | Status |
|---|---|---|
| NVIDIA Thor | SM110 | Primary target (aarch64 cross build) |

The kernels can also be exported on a Blackwell datacenter card (SM100 / SM103)
for local iteration, but the only AOT artifact shipped is `aarch64/sm_110/`.
Blackwell datacenter prefill / decode lives in the `nvfp4_fused_moe` group; the
SM120 / SM121 GeForce path lives in `nvfp4_fused_moe_cutedsl/`.

## 2. Kernel Variants

The `nvfp4_moe` group exports 6 AOT-compiled kernel objects (`.o` + `.h`
pairs):

### FC1 — Gather grouped GEMM + activation + FP4 requant (4 variants)

| Variant | Activation | MMA N-tile |
|---|---|---|
| `nvfp4_moe_sm110_fc1_relu2_n128`  | ReLU²  | 128 |
| `nvfp4_moe_sm110_fc1_relu2_n256`  | ReLU²  | 256 |
| `nvfp4_moe_sm110_fc1_swiglu_n128` | SwiGLU | 128 |
| `nvfp4_moe_sm110_fc1_swiglu_n256` | SwiGLU | 256 |

### FC2 — Finalize with scatter-reduce (2 variants)

| Variant | MMA N-tile |
|---|---|
| `nvfp4_moe_sm110_fc2_n128_fp16` | 128 |
| `nvfp4_moe_sm110_fc2_n256_fp16` | 256 |

FC2 outputs FP16 directly back into the token layout. The runner currently
picks `n128` unconditionally (see `selectMmaTilerN` in
[`cuteDslNvfp4MoeSm110Runner.cpp`](../../cpp/kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.cpp));
`n256` is exported for follow-up benchmarking.

## 3. Tensor Contract

The AOT pack specializes `top_k = 8` (compile-time). `num_experts` (E), hidden
size, and intermediate size are runtime dimensions: the FC1/FC2 wrappers take E
as a runtime `l` argument, so one cubin serves any expert count. The
`--dummy-experts` value used during AOT export only sizes the trace buffers and
is not baked into the cubin. The SM110 runner restricts E to the
product-supported set `{128, 256}` (`CuteDslNvfp4MoeSm110Runner::canImplement`).

- FC1 weights: `[E, N1, H / 2]` (N1 = `2 * I` for SwiGLU, `I` for ReLU²)
- FC1 scales:  `[E, ceil(N1 / 128), ceil((H / 16) / 4), 32, 4, 4]`
- FC2 weights: `[E, H, I / 2]`
- FC2 scales:  `[E, ceil(H / 128), ceil((I / 16) / 4), 32, 4, 4]`
- Grouped MoE metadata: tile-to-expert, tile limits, permuted-to-expanded row mapping
- Output: FP16 `[T, H]`

Alignment rules enforced by the SM110 runner: `H % 128 == 0`, `I % 64 == 0`,
`FC1_N % 128 == 0`.

## 4. Quick Start

Run on Thor (aarch64) or on a Blackwell box for local iteration.

### 4.1. Install AOT dependencies

On CUDA 13 the `[cu13]` extra is **mandatory**; on CUDA 12 install the base package:

```bash
python3 -m pip install \
  'nvidia-cutlass-dsl[cu13]==4.6.0' \
  cupy-cuda13x==13.6.0 \
  cuda-python
```

| Dependency | Version | Notes |
|---|---|---|
| `nvidia-cutlass-dsl` | `4.6.0` | Pinned; CuTeDSL surface used by all groups. **On CUDA 13 install the `[cu13]` extra; on CUDA 12 install the base package.** |
| `cupy-cuda13x`       | `13.6.0` | CUDA 13.x ; use `cupy-cuda12x==12.3.0` for CUDA 12.x |
| `cuda-python`        | matches CUDA | Required by every group's AOT export |


### 4.2. Generate the AOT artifact pack

```bash
python3 kernelSrcs/build_cutedsl.py \
  --kernels nvfp4_moe \
  --gpu_arch sm_110 \
  --arch aarch64 \
  --clean
```

Output goes to `cpp/kernels/cuteDSLArtifact/aarch64/sm_110/` and contains
`libcutedsl_aarch64.a` (merged static archive), `metadata.json`, and the
per-variant headers under `include/`.

### 4.3. Build the plugin

```bash
cmake -S . -B build \
  -DENABLE_CUTE_DSL=nvfp4_moe \
  -DCMAKE_CUDA_ARCHITECTURES=110a \
  -DAARCH64_BUILD=ON \
  -DCUTE_DSL_ARTIFACT_TAG=sm_110 \
  ...
cmake --build build -j
```

CMake auto-defines `CUTE_DSL_NVFP4_MOE_ENABLED`; the plugin uses this backend
whenever `getSMVersion() == 110`.

## 5. `build_cutedsl.py` flags

| Flag | Default | Description |
|---|---|---|
| `--kernels GROUPS` | `ALL` | `nvfp4_moe`, `fmha`, `gdn`, or `ALL` (comma-separated list also accepted) |
| `--gpu_arch SM`    | auto  | e.g. `sm_110` for Thor, `sm_100` for Blackwell |
| `--arch ARCH`      | auto  | `aarch64` (Thor) or `x86_64` |
| `--output_dir DIR` | `cpp/kernels/cuteDSLArtifact` | Root output dir |
| `-j JOBS`          | `4`   | Parallel export jobs |
| `--clean`          | off   | Wipe the arch-specific output dir before building |
| `--verbose`        | off   | Per-variant compilation log |

## 6. Standalone Kernel Compilation

To compile a single variant manually (debugging / iteration):

```bash
cd kernelSrcs

# FC1: ReLU², N-tile = 128
python nvfp4_moe_cutedsl/export_fc1_kernel.py \
  --activation relu2 --mma_tiler_n 128 \
  --dummy-experts 128 --dummy-top-k 8 \
  --output_dir ./out \
  --file_name nvfp4_moe_sm110_fc1_relu2_n128 \
  --function_prefix nvfp4_moe_sm110_fc1_relu2_n128 \
  --export_only

# FC1: SwiGLU, N-tile = 128
python nvfp4_moe_cutedsl/export_fc1_kernel.py \
  --activation swiglu --mma_tiler_n 128 \
  --dummy-experts 128 --dummy-top-k 8 \
  --output_dir ./out \
  --file_name nvfp4_moe_sm110_fc1_swiglu_n128 \
  --function_prefix nvfp4_moe_sm110_fc1_swiglu_n128 \
  --export_only

# FC2: N-tile = 128, FP16 output
python nvfp4_moe_cutedsl/export_fc2_kernel.py \
  --mma_tiler_n 128 --output_dtype fp16 \
  --dummy-experts 128 --dummy-top-k 8 \
  --output_dir ./out \
  --file_name nvfp4_moe_sm110_fc2_n128_fp16 \
  --function_prefix nvfp4_moe_sm110_fc2_n128_fp16 \
  --export_only
```

`--activation` accepts `relu2` and `swiglu`; `--mma_tiler_n` accepts `128` and
`256`; FC2's `--output_dtype` accepts `bf16` and `fp16` (the AOT pack only
ships `fp16`).

## 7. Python Kernel-Class Usage

The two AOT export scripts in Section 6 wrap the underlying CuTeDSL kernel
classes. When iterating on a single variant (e.g. adding a new activation,
debugging a TMEM layout), it is often easier to instantiate the class directly
in Python instead of going through `export_fc{1,2}_kernel.py`. The snippets
below match the `Example:` blocks in the kernel module docstrings.

### 7.1. FC1 — gather grouped GEMM + activation + FP4 requant

`BlockScaledContiguousGatherGroupedGemmKernel` lives in
[`blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`](blockscaled_contiguous_gather_grouped_gemm_act_fusion.py).
`use_2cta_instrs` is inferred from `mma_tiler_mn[0]` (`True` when M=256, `False`
when M=128):

```python
from blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    BlockScaledContiguousGatherGroupedGemmKernel,
)

gemm = BlockScaledContiguousGatherGroupedGemmKernel(
    sf_vec_size=16,
    mma_tiler_mn=(256, 128),  # use_2cta_instrs=True since M=256
    cluster_shape_mn=(2, 1),
    vectorized_f32=True,
)
gemm(
    a=a_tensor,
    b=b_tensor,
    c=c_tensor,
    sfa=sfa_tensor,
    sfb=sfb_tensor,
    sfc_tensor=None,
    input_global_scale_tensor=input_global_scale_tensor,
    down_input_scale_tensor=None,
    tile_idx_to_expert_idx=tile_idx_to_expert_idx,
    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
    token_id_mapping_tensor=token_id_mapping_tensor,
    num_non_exiting_tiles=num_non_exiting_tiles,
    alpha=alpha,
    max_active_clusters=max_active_clusters,
    stream=stream,
)
```

### 7.2. FC2 — grouped GEMM + router-scale finalize / scatter

`Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel` lives in
[`blockscaled_contiguous_grouped_gemm_finalize_fusion.py`](blockscaled_contiguous_grouped_gemm_finalize_fusion.py):

```python
from blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)

gemm = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
    sf_vec_size=16, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1),
)
gemm(
    a_tensor, b_tensor, sfa_tensor, sfb_tensor, out_tensor,
    max_active_clusters, stream,
)
```

Both classes enforce the constraints listed in their docstrings — MMA tiler M
∈ {128, 256}, MMA tiler N ∈ {64, 128, 192, 256}, cluster shape M/N positive
powers of two with total cluster size ≤ 16 (and ≤ 4 for scale-factor
multicasts). The SM110 AOT pack only ships `m_tile_size = 128` (1-CTA, see
Section 8); FC1 and FC2 vary along the N tile via `mma_tiler_mn = (128, 128)`
and `(128, 256)` (see Section 2). The docstring examples above show
`(256, 128)` purely to illustrate the 2-CTA `use_2cta_instrs` path and are
**not** the configuration shipped by [`build_cutedsl.py`](../build_cutedsl.py)
for SM110.

## 8. Important Notes

- **Alpha scaling.** FC1 applies per-expert `alpha = input_gsf * weight_gsf`
  inside the kernel epilogue, before the fused activation. Alpha is a `[E]`
  FP32 tensor on the plugin's input slot.
- **SwiGLU weight interleave.** SwiGLU FC1 weights must be laid out as 32-col
  interleaved `(up, gate)` chunks along the N axis (`moe_inter_size = 2 * I`).
  Plain `[up..., gate...]` concatenation produces wrong results silently.
  See `repack_nvfp4_qwen3_moe_experts` in
  [`tensorrt_edgellm/checkpoint/repacking.py`](../../tensorrt_edgellm/checkpoint/repacking.py).
- **PDL.** Programmatic Dependent Launch is currently disabled
  (`EDGELLM_ENABLE_PDL = False` in [`cute_utils.py`](cute_utils.py)).
- **Tile size.** Only `m_tile_size = 128` (1-CTA) is supported.

## 9. Validation

No standalone reference probe is shipped. The SM110 NVFP4 MoE contract is
validated end-to-end through
[`unittests/nvfp4MoeCuteDslSm110Tests.cu`](../../unittests/nvfp4MoeCuteDslSm110Tests.cu)
(`CuteDslNvfp4MoeSm110Test.accuracy`).

## 10. File Map

| File | Description |
|---|---|
| `blockscaled_contiguous_gather_grouped_gemm_act_fusion.py` | FC1 kernel: gather grouped GEMM + activation + FP4 requant |
| `blockscaled_contiguous_grouped_gemm_finalize_fusion.py`   | FC2 kernel: grouped GEMM + router-scale finalize / scatter |
| `export_fc1_kernel.py` | FC1 AOT export script (invoked by `build_cutedsl.py`) |
| `export_fc2_kernel.py` | FC2 AOT export script (invoked by `build_cutedsl.py`) |
| `export_common.py`     | Shared AOT export helpers (dummy pointers, SF buffer sizing) |
| `custom_pipeline.py`   | SM110 CuTeDSL pipeline helper |
| `cute_utils.py`        | CuTeDSL utility helpers (PTX helpers, PDL gate, etc.) |
| `moe_compat.py`        | Compatibility helpers for the split SM110 path |
