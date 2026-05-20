# NvFP4 Fused MoE CuTe DSL Kernels (SM120/SM121)

End-to-end fused Mixture-of-Experts kernels for Blackwell GeForce
(SM120 / SM121). Each kernel fuses route/pack + FC1 + activation +
quantize + FC2 + scatter into a single resident-grid launch, eliminating
the host-side FC1/FC2 handoff of the decomposed `nvfp4_moe` pipeline.

## Shape support

The kernel wrappers are **shape-polymorphic** in `I` (moeInterSize),
`E` (numExperts), and `top_k`: these are passed as runtime `Int32`
arguments and the TMA descriptors are rebuilt at each launch.

The `hidden_size` (K) is a **compile-time variant axis** baked into
each AOT binary via the `--hidden_size` flag on the export scripts.
This is required because `_setup_attributes()` computes the pipeline
stage count (`ab_stage`) with a Python-level divisibility loop that
cannot trace with a symbolic K. The currently supported value is
`H = 2048`; see `CuteDslNvfp4MoeRunner::kSupportedHiddenSize` in the
C++ runner. Supporting additional K values requires rebuilding with a
different `--hidden_size` and adding a dispatch entry to the runner.

## Backends

| Backend | Script | Best for |
|---|---|---|
| **Decode** | `export_decode_kernel.py` | Small routed sets (`routed_rows <= 640`). Resident-grid barrier between route/pack and compute phases. |
| **Prefill** | `export_prefill_kernel.py` | Large routed sets. Global task-queue driven producer/consumer overlap. |

## Variants (build_cutedsl.py group: `nvfp4_fused_moe`)

10 active variants: 5 activations x 2 backends, all using MMA N-tile 128.

The C++ runner currently dispatches n128. Add a new N-tile dispatch axis only
after rebuilding the artifact pack with matching wrapper symbols and validating
accuracy/perf across the full matrix.

| Name | Backend | Activation |
|---|---|---|
| `nvfp4_fused_moe_decode_identity_n128` | decode | identity |
| `nvfp4_fused_moe_decode_silu_n128` | decode | silu |
| `nvfp4_fused_moe_decode_swiglu_n128` | decode | swiglu |
| `nvfp4_fused_moe_decode_gelu_n128` | decode | gelu |
| `nvfp4_fused_moe_decode_relu2_n128` | decode | relu2 |
| `nvfp4_fused_moe_prefill_identity_n128` | prefill | identity |
| `nvfp4_fused_moe_prefill_silu_n128` | prefill | silu |
| `nvfp4_fused_moe_prefill_swiglu_n128` | prefill | swiglu |
| `nvfp4_fused_moe_prefill_gelu_n128` | prefill | gelu |
| `nvfp4_fused_moe_prefill_relu2_n128` | prefill | relu2 |

All variants target SM120/SM121 only.

## Build

```bash
# Build all SM120-compatible kernels (auto-detects GPU):
python kernelSrcs/build_cutedsl.py

# Build only the fused MoE group:
python kernelSrcs/build_cutedsl.py --kernels nvfp4_fused_moe

# Force SM121 (e.g., on a host without the target GPU):
python kernelSrcs/build_cutedsl.py --kernels nvfp4_fused_moe --gpu_arch sm_121
```

Artifacts land in `cpp/kernels/cuteDSLArtifact/<arch>/<tag>/`:
- `libcutedsl_<arch>.a` (merged static library)
- `include/nvfp4_fused_moe_*.h` (per-variant headers)
- `include/cutedsl_nvfp4_fused_moe_all.h` (group umbrella header)

## CuTeDSL SM121 Source Patch

`nvidia-cutlass-dsl==4.4.1` has an SM120-only admissible-architecture check
in the block-scaled warp MMA ops used by these fused MoE kernels. The SM121
fused MoE artifact build sets `CUTE_DSL_ARCH=sm_121a` to generate a real SM121
image, so the unpatched package rejects the compatible SM120/SM121 op path
before AOT export starts.

SM120 builds do not need this patch. SM121 fused MoE artifact generation does.
The build script skips this group for `sm_121` when building `--kernels ALL`.
To build `nvfp4_fused_moe` explicitly on SM121, apply the source patch below
manually to the installed `nvidia_cutlass_dsl/python_packages` directory. It
extends the specific SM120 block-scaled warp MMA admissible-architecture list to
include `sm_121a`, `sm_120f`, and `sm_121f`, and fixes the
`MmaSM120BlockScaledOp.__post_init__` check to honor that list.

<!-- cutedsl-sm121-patch-begin -->
```diff
--- a/cutlass/cute/nvgpu/warp/mma.py
+++ b/cutlass/cute/nvgpu/warp/mma.py
@@ -131,3 +131,6 @@ class MmaSM120BlockScaledOp(MmaOp):
     admissible_archs = [
         "sm_120a",
+        "sm_121a",
+        "sm_120f",
+        "sm_121f",
     ]
@@ -135,7 +138,8 @@ class MmaSM120BlockScaledOp(MmaOp):
     def __post_init__(self) -> None:
         # Verify arch
         arch = CuTeDSL._get_dsl().get_arch_enum()
-        if not arch == Arch.sm_120a:
+        arch_name = getattr(arch, "name", str(arch).rsplit(".", 1)[-1])
+        if arch_name not in self.admissible_archs:
             raise OpError(
                 self,
                 f"expects arch to be one of {self.admissible_archs}, but got {arch}",
```
<!-- cutedsl-sm121-patch-end -->

## Manual export (single variant)

```bash
cd kernelSrcs
python nvfp4_fused_moe_cutedsl/export_decode_kernel.py \
    --activation swiglu \
    --mma_tiler_n 128 \
    --hidden_size 2048 \
    --output_dir /tmp/staging \
    --file_name nvfp4_fused_moe_decode_swiglu_n128 \
    --function_prefix nvfp4_fused_moe_decode_swiglu_n128 \
    --verbose
```

## CMake integration

`cmake/CuteDsl.cmake` auto-detects the `nvfp4_fused_moe` group from
`metadata.json` and sets `CUTE_DSL_NVFP4_FUSED_MOE_ENABLED` on the
link targets. C++ runner code should be guarded by:

```cpp
#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
#include "cutedsl_nvfp4_fused_moe_all.h"
// ... use generated _Kernel_Module_t / _wrapper symbols ...
#endif
```

## Attribution

The two kernel backends are ported from the
[`b12x`](https://github.com/lukealonso/b12x) kernel library by Luke Alonso:

| This repo | b12x origin |
|---|---|
| `moe_decode_kernel.py` | `b12x/moe/fused/static.py` (`MoEStaticKernel`) |
| `moe_prefill_kernel.py` | `b12x/moe/fused/dynamic.py` (`MoEDynamicKernelBackend`) |

The core compute body (FC1 → activation → quant → FC2 → scatter), the
queue-driven producer/consumer model, and the resident-grid barrier scheme
are derived from that work. Local changes include: additional activation
variants (`identity`, `gelu`, `swiglu`), explicit SM121 patch workflow, and
integration with the TRT Edge-LLM plugin system.

## Dependencies

- `nvidia-cutlass-dsl == 4.4.1`
- `cuda-python` (provides `cuda.bindings.driver`)
- `cupy-cuda13x` (GPU memory allocation during AOT compilation)

Pick the `cuda-python` and CuPy variant that matches your CUDA version before
installing `nvidia-cutlass-dsl`:

```bash
pip install cuda-python==12.8.* cupy-cuda12x==12.3.0 # CUDA 12.x
# or
pip install cuda-python cupy-cuda13x==13.6.0 # CUDA 13.x

pip install nvidia-cutlass-dsl==4.4.1
```

## TensorRT plugin

The FP16 variants of this kernel family are wrapped as a TensorRT plugin at
[`cpp/plugins/nvfp4MoePluginGeforce/`](../../cpp/plugins/nvfp4MoePluginGeforce/).
See that plugin's
[`README.md`](../../cpp/plugins/nvfp4MoePluginGeforce/README.md) for the
supported-shapes contract and integration instructions.
