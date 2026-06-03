# NvFP4MoEPluginGeforce

TensorRT plugin that wraps the fused NVFP4 MoE kernel in
[`kernelSrcs/nvfp4_fused_moe_cutedsl/`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/)
(CuTeDSL SM120 / SM121, decode + prefill backends).

This plugin is an **additive** counterpart of the Marlin-based
[`Nvfp4MoePlugin`](../nvfp4MoePlugin/nvfp4MoePlugin.cpp); it does not
replace it. Pick one or the other at ONNX-build time.

## Supported shapes

The AOT-exported CuTeDSL kernel is **shape-polymorphic** in `I` / `E` /
`top_k`: the wrappers in
[`export_decode_kernel.py`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/export_decode_kernel.py)
and
[`export_prefill_kernel.py`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/export_prefill_kernel.py)
build `cute.Tensor` layouts from raw pointers and runtime `Int32`
extents, so one AOT binary handles any (I, E, top_k) tuple meeting the
alignment contract below.

One dim is a **compile-time variant axis** and requires a kernel rebuild
to change:
* The **MMA N-tile** currently dispatches n128. Add another dispatch axis
  after rebuilding the artifact pack and re-enabling n256 in
  `CuteDslNvfp4MoeRunner::selectMmaTilerN`.

The **hidden_size (K)** is runtime-shaped in the AOT wrapper. The only
constraint is `H > 0 && H % kHiddenSizeAlignment == 0`, where
`kHiddenSizeAlignment = kCuteDslTileK * kStaticAbStage = 128 * 2 = 256`
in the current build. That divisor guarantees the K-tile mainloop pipeline
drains cleanly with the AOT-baked `ab_stage = 2`. Validated for H in
`{1024, 2048}`; other multiples of 256 are expected to work by shape
polymorphism but should be accuracy-checked before production use.

| Parameter | Supported values |
|---|---|
| `io_dtype` | FP16 |
| `activation_type` | identity, silu, swiglu, gelu, relu2 |
| `backend` | decode + prefill (`auto` picks decode when `num_tokens*top_k <= 640` else prefill; mirrors `select_sm120_moe_backend`) |
| `hidden_size` (H) | positive multiple of `kHiddenSizeAlignment` (= 256 = `kCuteDslTileK * kStaticAbStage`) |
| `moe_inter_size` (I) | `I > 0`, `I % kLevelTileN == 0` (128) |
| `num_experts` (E) | `E > 0` |
| `top_k` | `0 < top_k <= E` |

The plugin's `configurePlugin` enforces the hidden-size divisibility and
the alignment rules, emitting a clear error when they are violated.
`CuteDslNvfp4MoeRunner::canImplement` is the authoritative source.

### Decode vs prefill

Both backends' AOT binaries are shape-polymorphic in `(H, I, E, top_k)`
-- weights and per-expert metadata are passed as pointers + runtime
`Int32` extents, so TMA descriptors are rebuilt on every launch. The
difference is purely the per-enqueue execution pattern:

* **Decode** — resident-grid kernel with a block-wide barrier between
  route/pack and compute phases. Lower launch overhead; best when
  `num_tokens*top_k <= 640`.
* **Prefill** — global task-queue-driven producer/consumer overlap.
  Higher launch overhead but scales to hundreds of routed rows per
  expert; best for prefill workloads.

`backend=auto` applies the same cutover
(`kDecodePrefillCutoverRoutedRows = 640`) as
[`select_sm120_moe_backend`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/moe_dispatch.py).

## Files

- [`nvfp4MoePluginGeforce.h`](nvfp4MoePluginGeforce.h) / [`nvfp4MoePluginGeforce.cpp`](nvfp4MoePluginGeforce.cpp) — the `IPluginV3` implementation.
- [`../../kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeRunner.{h,cpp}`](../../kernels/moe/nvfp4_cutedsl/) — the AOT-module owner: module load/unload, shape-check, workspace layout, and wrapper dispatch. The runner is allocation-free on the enqueue path; the plugin owns the per-instance identity expert-id table (via `IGpuAllocator` in `attachToContext`) and threads it in through `CuteDslNvfp4MoeParams::weightExpertIds` / `globalToLocalExpertIds`.
- [`../../../kernelSrcs/nvfp4_fused_moe_cutedsl/README.md`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/README.md) — kernel variants, AOT build flow, and data-layout notes.

## ONNX input surface

10 inputs, 1 output:

```
router_logits      fp32    [T, E]         # pre-softmax; plugin applies moeTopkSoftmax
hidden_states      fp16    [B, S, H]      # per-token NVFP4 quant done inside the kernel
fc1_qweights       int8    [E, N1, H/2]   # N1 = 2*I (swiglu) or I (identity)
fc1_blocks_scale   int8    [E, m_tiles_1, k_tiles_1, 32, 4, 4]
fc1_alpha          fp32    [E]
fc2_qweights       int8    [E, H, I/2]
fc2_blocks_scale   int8    [E, m_tiles_2, k_tiles_2, 32, 4, 4]
fc2_alpha          fp32    [E]
input_global_scale fp32    [E]
down_input_scale   fp32    [E]
-> output          fp16    [B, S, H]
```

Block scales use the contiguous physical CuTeDSL NVFP4 layout
`[E, m_tiles, k_tiles, 32, 4, 4]`.

## Build

1. Generate the AOT artifact (requires `nvidia-cutlass-dsl==4.5.1`,
   `cuda-python`, `cupy-cuda13x`, and a sm_120 / sm_121 GPU):

   ```bash
   python kernelSrcs/build_cutedsl.py --kernels nvfp4_fused_moe
   ```

   `nvidia-cutlass-dsl==4.5.1` supports the `sm_121a` architecture used by
   SM121.
   No local CuTeDSL source patch is required for fused MoE artifact generation on
   SM121. `build_cutedsl.py --kernels nvfp4_fused_moe --gpu_arch sm_121` sets
   `CUTE_DSL_ARCH=sm_121a` for fused MoE so the generated images target SM121
   directly.

   Output lands in `cpp/kernels/cuteDSLArtifact/<arch>/<tag>/` and
   includes `cutedsl_nvfp4_fused_moe_all.h`.

2. Build the plugin with CuTeDSL linked in:

   ```bash
   cmake -S . -B build -DENABLE_CUTE_DSL=nvfp4_fused_moe ...
   cmake --build build -j
   ```

   CMake auto-defines `CUTE_DSL_NVFP4_FUSED_MOE_ENABLED` on the plugin
   target. Without it, the plugin still compiles but every `enqueue` call
   returns an error (the plugin creator still registers, so deserialize
   paths work in build-only environments).

## Retargeting other shapes

For `(H, I, E, top_k)` changes that still match the alignment contract
(`H > 0 && H % 256 == 0`, `I % 128 == 0`, `0 < top_k <= E`): no rebuild
needed. Configure the plugin with the desired tuple and the runner will use
the n128 N-tile variant at launch.

If a new MMA N-tile granularity is needed (e.g. `n64` / `n512`), add a
new `KernelVariant` row per activation / backend to
`build_cutedsl.py`, add the matching static modules and the switch arm
in `selectMmaTilerN` / `DISPATCH_*_ACTIVATION` in
[`cuteDslNvfp4MoeRunner.{h,cpp}`](../../kernels/moe/nvfp4_cutedsl/),
and rebuild.

## Validation

There is no active Python unit-test entry for this plugin. Avoid validating
production routing by instantiating Python-only helper modules directly; model
integration should be tested at the export path that explicitly emits
`NvFP4MoEPluginGeforce`.

### Thor sign-off checklist (runner-test equivalent)

1. `mount-thor-sshfs` the workspace onto Thor.
2. `python kernelSrcs/build_cutedsl.py --kernels nvfp4_fused_moe --gpu_arch sm_121`
3. `build-edge-llm-package` with `-DENABLE_CUTE_DSL=nvfp4_fused_moe`.
