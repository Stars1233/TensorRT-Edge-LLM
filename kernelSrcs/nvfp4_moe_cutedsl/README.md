# CuTe DSL NvFP4 MoE Kernels (Blackwell SM100/SM103, Thor SM110)

NvFP4 MoE FC1 and FC2 kernels compiled ahead-of-time from CuTe DSL Python
source. These kernels implement the gating/up-projection (FC1) and output
projection (FC2) layers of Mixture-of-Experts blocks with FP4 blockscaled
arithmetic on Blackwell/Thor tensor cores (tcgen05.mma).

Kernel artifacts (static library + headers) are pre-generated offline by
`kernelSrcs/build_cutedsl.py` and checked into the repo. CMake simply links the
prebuilt artifacts -- no Python, CUTLASS DSL, CuPy, or Blackwell GPU is needed
at CMake build time.

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

pip install nvidia-cutlass-dsl==4.5.1
```

**2. Compile all kernel variants into a static library**

> **GPU requirement:** compilation must run on a Blackwell datacenter GPU (e.g.
> B200, GB200) or on Thor (SM110). Other GPUs are not supported.

```bash
cd tensorrt-edge-llm
python kernelSrcs/build_cutedsl.py --kernels nvfp4_moe --gpu_arch sm_100 [--clean] [-j 4]
```

This produces the following artifacts under
`cpp/kernels/cuteDSLArtifact/{arch}/`:

```
libcutedsl_{arch}.a          -- all 8 kernel .o files + libcuda_dialect_runtime_static.a merged in
metadata.json                -- build provenance (CUDA ver, DSL ver, date, groups)
include/
    cutedsl_all.h            -- umbrella header
    nvfp4_moe_fc1_*.h        -- FC1 per-variant headers (6 variants)
    nvfp4_moe_fc2_*.h        -- FC2 per-variant headers (2 variants)
```

Commit the generated `cuteDSLArtifact/{arch}/` directory to the repo so that
CMake builds on edge devices require no Python or GPU.

**3. Enable in CMake**

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUTE_DSL=nvfp4_moe
```

To enable all kernel groups (FMHA + GDN + NvFP4 MoE):

```bash
cmake .. -DENABLE_CUTE_DSL=ALL ...
```

---

## 1. Kernel Variants

The build produces **8 AOT-compiled kernel objects** (.o + .h pairs):

### FC1 -- Contiguous Grouped GEMM (6 variants)

| Variant | Activation | N-Tile | Use Case |
|---|---|---|---|
| `nvfp4_moe_fc1_identity_n128` | identity | 128 | Decomposed pipeline (alpha + activation applied externally) |
| `nvfp4_moe_fc1_identity_n256` | identity | 256 | Decomposed pipeline, wider N |
| `nvfp4_moe_fc1_relu2_n128` | relu2 | 128 | Nemotron 3 Nano (fused activation) |
| `nvfp4_moe_fc1_relu2_n256` | relu2 | 256 | Nemotron 3 Nano, wider N |
| `nvfp4_moe_fc1_swiglu_n128` | swiglu | 128 | Qwen3-30B-A3B (fused SwiGLU) |
| `nvfp4_moe_fc1_swiglu_n256` | swiglu | 256 | Qwen3-30B-A3B, wider N |

**Architecture**: 7 warps (4 epilogue, 1 MMA, 1 TMA, 1 scheduler), 224 threads/CTA.
Uses `StaticPersistentTileScheduler` + runtime lookup tables -- zero TMA descriptor
updates. Only 2 AOT variants per activation (vs 16 for the original bucketed FC1).

### FC2 -- Finalize with Scatter-Reduce (2 variants)

| Variant | N-Tile | Use Case |
|---|---|---|
| `nvfp4_moe_fc2_n128` | 128 | Output projection with fused scatter-reduce |
| `nvfp4_moe_fc2_n256` | 256 | Output projection, wider N |

**Architecture**: 7 warps (4 epilogue, 1 MMA, 1 TMA, 1 scheduler), 224 threads/CTA.
Epilogue fuses: permuted-to-token index mapping, alpha scaling, router
`token_final_scales`, and block-reduce scatter-reduce (deterministic).

## 2. Building

### 2.1. Generating Prebuilt Artifacts

Artifacts must be regenerated whenever the kernel source changes or the
CUTLASS DSL version changes. Run on a machine with a Blackwell GPU:

```bash
cd tensorrt-edge-llm
python kernelSrcs/build_cutedsl.py --kernels nvfp4_moe --gpu_arch sm_100 [--clean] [-j 4]
```

**Prerequisites** (only needed when running `build_cutedsl.py`):

| Dependency | Version | Notes |
|---|---|---|
| `nvidia-cutlass-dsl` | 4.5.1 | |
| `cuda-python` | 12.8.* for CUDA 12.x | Install before `nvidia-cutlass-dsl`; use the CUDA 13.x resolved version for CUDA 13.x |
| `cupy-cuda12x` | 12.3.0 | CUDA 12.x |
| `cupy-cuda13x` | 13.6.0 | CUDA 13.x |

**Script options:**

| Flag | Default | Description |
|---|---|---|
| `--kernels GROUPS` | `ALL` | `nvfp4_moe`, `fmha`, `gdn`, or `ALL` |
| `--gpu_arch SM` | (device native) | e.g. `sm_100` for Blackwell, `sm_110` for Thor |
| `--output_dir DIR` | `cpp/kernels/cuteDSLArtifact` | Root output dir |
| `--arch ARCH` | auto-detected | `x86_64` or `aarch64` |
| `-j JOBS` | `4` | Parallel compile jobs (use `-j 1` if GPU memory is limited) |
| `--verbose` | off | Show per-variant compilation output |
| `--clean` | off | Remove the arch-specific output dir before building |

### 2.2. Standalone Kernel Compilation

To compile a single variant outside the build script (e.g. for testing):

```bash
cd kernelSrcs

# FC1: identity activation, N-tile=128
python nvfp4_moe_cutedsl/export_fc1_kernel.py \
    --activation identity --mma_tiler_n 128 \
    --output_dir ./out --file_name nvfp4_moe_fc1_identity_n128 \
    --function_prefix nvfp4_moe_fc1_identity_n128

# FC1: relu2, N-tile=256
python nvfp4_moe_cutedsl/export_fc1_kernel.py \
    --activation relu2 --mma_tiler_n 256 \
    --output_dir ./out --file_name nvfp4_moe_fc1_relu2_n256 \
    --function_prefix nvfp4_moe_fc1_relu2_n256

# FC2: N-tile=128
python nvfp4_moe_cutedsl/export_fc2_kernel.py \
    --mma_tiler_n 128 \
    --output_dir ./out --file_name nvfp4_moe_fc2_n128 \
    --function_prefix nvfp4_moe_fc2_n128
```

Each invocation produces `<file_name>.h` and `<file_name>.o` in `--output_dir`.

### 2.3. CMake Configuration

No Python or GPU required at CMake time. CMake links the prebuilt artifacts:

```bash
cmake -DENABLE_CUTE_DSL=nvfp4_moe \
      -DTRT_PACKAGE_DIR=/path/to/TensorRT \
      ..
```

`cmake/CuteDsl.cmake` validates the artifacts and defines
`CUTE_DSL_NVFP4_MOE_ENABLED` for conditional C++ code.

## 3. Kernel Accuracy

### FC1 (Contiguous Grouped GEMM)

Tested against f32 reference computed from the same FP4-quantized inputs:

| Dimensions | Activation | Median Cosine | Determinism |
|---|---|---|---|
| Nemotron (K=2688, N=1856) | identity | 0.999999 | 100% byte-exact |
| Nemotron (K=2688, N=1856) | relu2 | 0.999999 | 100% byte-exact |
| Qwen3 (K=2048, N=1536) | swiglu | 0.999999 | 100% byte-exact |

### FC2 (Finalize with Scatter-Reduce)

Tested against BF16 reference (measures combined FP4 quantization + kernel error):

| Dimensions | Median Cosine | Mag Ratio | Determinism |
|---|---|---|---|
| Nemotron (K=1856, N=2688, top_k=6) | 0.9910 | 1.0001 | 100% byte-exact (block-reduce) |

The ~1% cosine gap vs BF16 reference is entirely from FP4 quantization loss.

### E2E MoE Block (FC1 + FC2)

| Dimensions | Tier-1 Cosine (BF16 ref) | Tier-2 Cosine (FP4-aware ref) |
|---|---|---|
| Nemotron (H=2688, N=1856, 128 experts, top_k=6) | 0.981 | 0.996-0.999 |

## 4. Supported Hardware

| GPU | SM | CuteDSL Patches | Status |
|---|---|---|---|
| NVIDIA Blackwell | SM100/SM103 | None | Native support |
| NVIDIA Thor | SM110 | 2 patches required (see below) | Byte-exact with Blackwell |

CuteDSL handles architecture targeting internally during AOT export -- no
`-arch` or `-gencode` flags are needed for kernel compilation. The patches
below are only about CuteDSL's internal arch allowlists.

### Thor (SM110): Required CuteDSL Patches

Thor requires two CuteDSL patches (tested on 4.4.1, 4.4.2, 4.5.0, and 4.5.1) before GEMM
kernels can compile. Non-MMA kernels (FP4 quantize, MoE gather) work without
patches.

**Root cause**: CuteDSL internally remaps `sm_110 → sm_101a`, which belongs to
the `sm_110f` family, **not** `sm_100f`. Two ops have hardcoded allowlists that
only include `sm_100f`-family architectures.

**Apply patches:**

```bash
SITE_PKG=$(python3 -c "import nvidia_cutlass_dsl; print(nvidia_cutlass_dsl.__path__[0])")
MMA_FILE="$SITE_PKG/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py"
COPY_FILE="$SITE_PKG/python_packages/cutlass/cute/nvgpu/tcgen05/copy.py"

# Patch 1: Add sm_101a to BlockScaledMmaOp.admissible_archs (mma.py ~line 307)
#   The FP4 blockscaled MMA op has a static allowlist [sm_100a, sm_103a] that
#   is missing sm_101a (Thor). The regular MmaOp uses a dynamic filter that
#   includes sm_110f family, but BlockScaledMmaOp was never updated.
sed -i '/        Arch.sm_103a,/a\        Arch.sm_101a,' "$MMA_FILE"

# Patch 2: Add sm_110f family check to Cp4x32x128bOp (copy.py ~line 735)
#   The tcgen05 copy op checks only sm_100f family. Other copy ops in the
#   same file already use the correct dual-family pattern (sm_100f or sm_110f).
sed -i 's/if not arch.is_family_of(Arch.sm_100f):/if not (arch.is_family_of(Arch.sm_100f) or arch.is_family_of(Arch.sm_110f)):/' "$COPY_FILE"

# Clear bytecode cache
find "$SITE_PKG" -name "*.pyc" -delete
```

**Notes**:
- These patches are safe for Blackwell -- they only add `sm_101a` to allowlists.
- `CUTE_DSL_ARCH` env var is **not** effective for overriding Thor's architecture.
  CuteDSL remaps `sm_110 → sm_101` regardless, and `CUTE_DSL_ARCH=sm_100a`
  produces `cudaErrorNoKernelImageForDevice` on Thor.
- ptxas may emit a harmless warning about `__launch_bounds__(512, 4)` on the
  FP4 quantize kernel because Thor cannot schedule 2048 threads/SM. The hint
  is ignored; the kernel runs correctly.

## 5. Important Notes

- **Alpha scaling**: FC1 applies per-expert `alpha = input_gsf * weight_gsf`
  inside the kernel epilogue, before the fused activation (relu2/swiglu/identity).
  The alpha tensor is passed as a `(L,)` float32 input to the kernel launch.
- **SwiGLU weights**: Must be preprocessed with `interleave_linear_and_gate(weight,
  group_size=32, dim=1)`. Plain `[up..., gate...]` concatenation produces wrong
  results silently.
- **FC2 block-reduce**: Uses `use_blkred=True` for deterministic scatter-reduce.
- **Tile size**: Only `m_tile_size=128` (1CTA) is supported.

## 6. File Map

| File | Description |
|---|---|
| `kernelSrcs/nvfp4_moe_cutedsl/blockscaled_contiguous_grouped_gemm_n_major.py` | FC1 kernel: N-major grouped GEMM with in-flight SMEM nibble transpose + fused activation |
| `kernelSrcs/nvfp4_moe_cutedsl/blockscaled_contiguous_grouped_gemm_finalize_n_major.py` | FC2 kernel: N-major grouped GEMM with in-flight SMEM nibble transpose + fused scatter-reduce |
| `kernelSrcs/nvfp4_moe_cutedsl/utils.py` | PTX helpers (make_ptr, atomics, block-reduce, grid dep control) |
| `kernelSrcs/nvfp4_moe_cutedsl/common.py` | AOT export helpers (dummy pointer creation, SF buffer sizing) |
| `kernelSrcs/nvfp4_moe_cutedsl/export_fc1_kernel.py` | FC1 AOT export script (invoked by build_cutedsl.py) |
| `kernelSrcs/nvfp4_moe_cutedsl/export_fc2_kernel.py` | FC2 AOT export script (invoked by build_cutedsl.py) |
| `kernelSrcs/build_cutedsl.py` | Unified build script: compiles all CuTe DSL variants |
| `cmake/CuteDsl.cmake` | CMake module: validates and links prebuilt artifacts |
| `cpp/kernels/cuteDSLArtifact/{arch}/` | Prebuilt artifacts (committed to repo) |
