# CuTe DSL GDN Kernels (Ampere through Blackwell)

Prefill and decode kernels for Gated Delta Net (GDN), AOT-compiled from CuTe
DSL Python source. Prebuilt artifacts (static library + headers) are generated
locally; CMake links them directly — no Python or GPU needed at build time.

## Kernel Variants

Exact `supported_sms` per variant live in `kernelSrcs/build_cutedsl.py`
(`KERNEL_VARIANTS`); the table below is a summary.

| Variant | seq_len | Supported SMs | Verified On | Notes |
|---|---|---|---|---|
| `gdn_decode` | 1 | SM80+ | Ampere (SM80), Blackwell (SM110) | small/large-batch dispatch at runtime (threshold n=32) |
| `gdn_prefill` | > 1 | SM80+ | Ampere (SM80), Blackwell (SM110) | per-row context masking via `context_lengths`; should work on Hopper (SM90) but not yet tested |
| `gdn_prefill_blackwell` | > 1 | SM100+ only | Blackwell (SM110) | uses TMA + warp-level pipeline; requires `cu_seqlens`; not built on Ampere/Hopper/Orin |
| `gdn_decode_mtp_cache` | 1..16 | SM80+ | Ampere (SM80), Blackwell (SM110) | linear MTP/DFlash verification with per-step checkpoint cache |
| `gdn_decode_tree` | 1..128 | SM80+ | Blackwell (SM110) | DDTree verification with parent/depth-driven per-node checkpoints |
| `gdn_decode_tree_split_v_precomputed` | 1..128 | SM80+ | Blackwell (SM110) | DDTree split-V verification using precomputed q/k scales and gate values |

All variants require CUDA 12.6+.

## Building Prebuilt Artifacts

Run on a machine with a supported GPU and CUDA 12.x/13.x:

```bash
# Pick cuda-python and cupy variant that matches your CUDA version
# before installing `nvidia-cutlass-dsl`:
pip install cuda-python==12.8.* cupy-cuda12x==12.3.0 # CUDA 12.x
# or
pip install cuda-python cupy-cuda13x==13.6.0 # CUDA 13.x

# CUDA 13: install the [cu13] extra. CUDA 12: install the base package.
pip install 'nvidia-cutlass-dsl[cu13]==4.6.0'  # CUDA 13.x
# CUDA 12.x: pip install 'nvidia-cutlass-dsl==4.6.0'

cd tensorrt-edge-llm

# With explicit SM (x86 or cross-compile):
python kernelSrcs/build_cutedsl.py --kernels gdn --gpu_arch sm_87 [--clean]

# On-device (device-native SM, e.g. Thor SM110):
python kernelSrcs/build_cutedsl.py --kernels gdn [--clean]
```

Output under `cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/`:

`artifact_tag` is currently `sm_<NN>` (for example `sm_87`, `sm_110`, or
`sm_121`).

```
libcutedsl_{arch}.a    — per-variant .o + libcuda_dialect_runtime_static.a
metadata.json          — build provenance (groups, variants, CUDA ver, DSL ver)
include/
    cutedsl_all.h      — umbrella header
    gdn_decode.h
    gdn_prefill.h
    gdn_prefill_blackwell.h   — present only when that variant was built
    gdn_decode_mtp_cache.h
    gdn_decode_tree.h
    gdn_decode_tree_split_v.h
    gdn_decode_tree_split_v_precomputed.h
```

Keep the `cuteDSLArtifact/{arch}/{artifact_tag}/` directory locally so
downstream CMake builds can reuse it without re-running Python.

Key script flags: `--kernels gdn`, `--gpu_arch` (e.g. `sm_87` for Orin,
omit for device-native SM on Thor), `--arch` (default: auto), `--verbose`,
`--clean`.

## CMake

```bash
cmake -DENABLE_CUTE_DSL=gdn ...
```

`cmake/CuteDsl.cmake` resolves the artifact tag, reads `metadata.json`,
validates the prebuilt artifacts under `cuteDSLArtifact/{arch}/{artifact_tag}/`,
and links `libcutedsl_{arch}.a`; defines `CUTE_DSL_GDN_ENABLED`. If
`gdn_prefill_blackwell` appears in `metadata.json`, it also defines
`CUTE_DSL_GDN_BLACKWELL_ENABLED` so the C++ runner can load the Blackwell
prefill module. Fails with a clear error if artifacts are missing.

If multiple artifact tags exist for the same CPU architecture, pass
`-DCUTE_DSL_ARTIFACT_TAG=<tag>` explicitly.

To enable both GDN and FMHA:

```bash
cmake -DENABLE_CUTE_DSL=ALL ...
```

## Standalone Test / Export

```bash
cd kernelSrcs/gdn_cutedsl

# accuracy check (SM80+ class GPUs)
python3 gdn_decode.py --n 4 --h 8 --hv 8 --k 128 --v 128
python3 gdn_prefill.py --n 8 --h 8 --hv 8 --k 128 --v 128 --seq_len 16
python3 gdn_decode_mtp.py --n 4 --h 8 --hv 8 --k 128 --v 128 --seq_len 4 --cache
python3 gdn_decode_tree.py --n 2 --h 4 --hv 8 --k 128 --v 128 --seq_len 8

# Blackwell prefill (needs a supported Blackwell-class GPU)
python3 gdn_prefill_blackwell.py --n 4 --h 8 --hv 8 --k 128 --v 128 --seq_len 128

# AOT export (single variant)
python3 gdn_decode.py --export_only --output_dir ./out --file_name gdn_decode --function_prefix gdn_decode
python3 gdn_prefill.py --export_only --output_dir ./out --file_name gdn_prefill --function_prefix gdn_prefill
python3 gdn_prefill_blackwell.py --export_only --output_dir ./out --file_name gdn_prefill_blackwell --function_prefix gdn_prefill_blackwell
python3 gdn_decode_tree.py --export_only --output_dir ./out --file_name gdn_decode_tree --function_prefix gdn_decode_tree
```

`context_lengths_preset` options — decode: `all_ones`, `first_half_active`;
prefill: `full`, `half`, `staggered`; Blackwell prefill: `full`, `half`.

## Tensor Shapes

| Tensor | Decode | Prefill | Dtype |
|---|---|---|---|
| `q`, `k` | `(N,1,H,K)` | `(N,T,H,K)` | FP16 |
| `v`, `o` | `(N,1,HV,V)` | `(N,T,HV,V)` | FP16 |
| `a`, `b` | `(N,1,HV)` | `(N,T,HV)` | FP16 |
| `A_log`, `dt_bias` | `(HV,)` | same | FP16 |
| `h0_source` | `(N,HV,K,V)` batch-dense | same | FP32 |
| `context_lengths` | `(N,)` device | same | INT32 |

Blackwell prefill additionally uses `cu_seqlens` (`(N+1,)` INT32) and separate
`h0` input/output views; see the generated `gdn_prefill_blackwell.h` and
`CuteDslGDNRunner` for the exact layout.

DDTree decode additionally uses `tree_parent_ids` and `tree_depths`
(`(N,S)` INT32) plus `intermediate_states` (`(N,S,HV,K,V)` FP32).  Root reads
`h0_source`; a valid child reads `intermediate_states[parent]`; invalid or
padded nodes write zero output and a harmless `h0_source` checkpoint.  The
DDTree path currently supports `K=V=128` and `S<=128`.

The split-V precomputed DDTree variant consumes runtime scratch generated by
`launchGdnDDTreePrecompute`: `qk_scales` (`(N,S,H,2)` FP32) stores scaled Q/K
normalization factors, and `gate_values` (`(N,S,HV,2)` FP32) stores `g` and
`beta`. This avoids repeating Q/K reductions and gate exponentials in each
V-tile CTA.

`h0_source` base pointer ≥ 16-byte aligned; vectorized-dim offset multiple of
`4×elem_size`. `context_lengths[i]==0` in decode skips row i (zeros output,
leaves h0 unchanged).

## C++ Integration

`CuteDslGDNRunner` (`cpp/kernels/gdnKernels/`): call `loadKernelModules()` once,
then `run(GDNParams, stream)` — dispatches decode vs sequential prefill vs
Blackwell prefill (when `CUTE_DSL_GDN_BLACKWELL_ENABLED` and `smVersion >= 100`),
MTP cache decode, or DDTree decode. `canImplement(kDim, vDim, smVersion)` guards
SM80+, K=V=128.

Plugin (`cpp/plugins/gatedDeltaNet/`): 9 inputs — `q, k, v, a, b, A_log,
dt_bias, h0_source, context_lengths`.
