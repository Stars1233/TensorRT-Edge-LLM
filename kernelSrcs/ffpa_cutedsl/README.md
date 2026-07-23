# CuTe DSL FFPA Kernel

CuTe DSL AOT adapt from the FFPA-style FMHA forward kernel from
[xlite-dev/ffpa-attn](https://github.com/xlite-dev/ffpa-attn), using the
Ampere instruction floor:

- `mma.sync.m16n8k16`
- `cp.async`
- `ldmatrix`
- warp shuffle reductions

The current registry scope is **D=512 causal FP16**. GQA is supported at
runtime: the exported kernel takes `num_kv_heads` as a launch argument, so a
single AOT kernel serves MHA and any GQA/MQA group size (the only constraint
is `num_head % num_kv_heads == 0`). The Python kernel also supports dense mode
and BF16, but those are not exported by default.

Supported artifact targets:

| SM | Role |
|----|------|
| 80, 86, 87, 89 | Native Ampere/Ada path. The default tuning was validated on sm_86. |
| 100, 101, 110, 120, 121 | Portability fallback. Prefer `fmha_cutedsl_blackwell/` for peak Blackwell performance. |

## Quick Start

```bash
python3 -m venv build_kernel_venv
source build_kernel_venv/bin/activate
pip install nvidia-cutlass-dsl==4.5.1 cupy-cuda12x==12.3.0 cuda-python

cd tensorrt-edge-llm
python kernelSrcs/build_cutedsl.py --kernels ffpa --gpu_arch sm_86 --clean -j 4

cmake .. -DENABLE_CUTE_DSL=ffpa -DTRT_PACKAGE_DIR=/path/to/TensorRT
```

Use `cupy-cuda13x==13.6.0` instead of `cupy-cuda12x` for CUDA 13.x.
If `--gpu_arch` is omitted, `build_cutedsl.py` auto-detects the local GPU.

Artifacts are written under `cpp/kernels/cuteDSLArtifact/{arch}/sm_<NN>/`:

```text
libcutedsl_{arch}.a
include/
    cutedsl_all.h
    cutedsl_ffpa_all.h
    ffpa_d512_causal.h
metadata.json
```

When the `ffpa` group is active, CMake defines `CUTE_DSL_FFPA_ENABLED` for
targets passed through `cute_dsl_setup()`. Correctness is covered by
`unittests/cuteDslFFPARunnerTest.cpp` — a parameterized accuracy sweep
plus a causal prefix-equivalence property test, all comparing against the
in-tree FP32 BSHD reference.

## Exported Variant

| Variant | D | Mask | KV group | Dtype | Tuning |
|---------|---:|------|---------:|-------|--------|
| `ffpa_d512_causal` | 512 | causal | runtime | FP16 | `Br=64, Bc=16, threads=128` |
| `ffpa_d512_causal_visionblock` | 512 | causal + vision-block overlay | runtime | FP16 | `Br=64, Bc=16, threads=128` |

Compile-time axes:

- `head_dim`
- `is_causal`
- `vision_block` (Gemma4 vision-block overlay; requires `is_causal`)
- `Br`, `Bc`, `num_threads`
- `skip_rescale`

Runtime axes:

- batch size
- `seqlen_q`, `seqlen_k`
- Q-head count
- `num_kv_heads` (GQA group size = Q-head count / `num_kv_heads`; `1` = MQA,
  equal to Q-head count = MHA). The kernel derives the group size internally;
  the C++ runner passes `numKVHeads` from `CuteDslFFPAParams` and rejects any
  `numQHeads % numKVHeads != 0`.
- `mCuSeqLenQ` / `mCuSeqLenK` — `(B+1,)` Int32 cumulative sequence lengths
  carrying the *logical* per-batch valid lengths.  Per batch
  `b`, `seqlen_q_b = cu_q[b+1] - cu_q[b]` and `seqlen_k_b = cu_k[b+1] -
  cu_k[b]`; the causal mask is bottom-right aligned with offset
  `seqlen_k_b - seqlen_q_b` (0 for right-padded prefill, the KV prefix length
  for chunked prefill).  Padding K/V positions beyond `seqlen_k_b` are never
  attended, padding Q rows are residual-masked, and whole-padding Q tiles
  write exact zeros.  Pass uniform cumulative lengths (`[0, S, 2S, ...]`) to
  recover the previous dense behaviour.  Varlen behaviour is covered by
  `unittests/cuteDslFFPARunnerTest.cpp` (ragged poisoned-padding and
  chunked-prefill cases against the FP32 reference).
- `mBlockBegin` / `mBlockEnd` (`ffpa_d512_causal_visionblock` only) —
  `(B, S_q)` Int32 vision-block interval tensors for Gemma4 Unified.  Per
  query row `q` they carry an extra allowed KV interval
  `[blockBegin[q], blockEnd[q]]` so contiguous image blocks attend
  bidirectionally: `allow(q, k) = causal(q, k) OR blockBegin[q] <= k <=
  blockEnd[q]`.  Text rows hold the `-1/-1` sentinel (empty interval).  The
  per-CTA KV traversal bound is extended to the Q tile's max `blockEnd`, so
  the extra cost is bounded by the block length past the diagonal.  The
  plain `ffpa_d512_causal` variant is compiled from the same script with
  the tensors set to `None` — its codegen and C ABI are unchanged.  The
  C++ runner (`CuteDslFFPARunner::run`) selects the overlay variant iff
  `CuteDslFFPAParams::blockBegin/blockEnd` are non-null.

The AOT export bakes the BSND layout with `D=512` as the innermost contiguous
dim, so the per-tensor batch and seq strides must match `S * H * D` and
`H * D` respectively. The C++ runner derives those values from the
dimensions in `CuteDslFFPAParams`; direct callers of the AOT wrapper must
do the same. Passing non-contiguous strides (e.g., a padded D) is not
supported.

## Kernel Notes

This is the whole-tile FFPA form. One CTA owns one `(Q tile, head, batch)`
triple, keeps `sQ`, `sK`, and `sV` in shared memory, and sweeps KV blocks in
reverse with FA2-style online softmax.

For the default D=512 tuning:

| Buffer | Size |
|--------|------|
| `sQ` | `64 * 512 * 2 = 64 KB` |
| `sK` | `16 * 512 * 2 = 16 KB` |
| `sV` | `16 * 512 * 2 = 16 KB` |
| Total | `96 KB` |

That fits the 99 KB Ampere/Ada opt-in shared-memory budget. The output epilogue
aliases `sO` over `sQ`, so no extra shared-memory output tile is allocated.

`acc_O` uses `64 * 512 / 128 = 256` FP32 accumulator registers per thread,
which sits at the per-thread register cap, so ptxas spills some of `acc_O`
to local memory. The spill is absorbed by L1 in practice but shows up as
the dominant Long-Scoreboard stall in `ncu`. Alternative tile/thread
configurations (e.g., `Br=32, Bc=32, threads=64`) are reachable via the
CLI flags but have not been validated and are not exported.

`skip_rescale` is enabled for the exported variant. It snaps a near-unity
online-softmax rescale factor to `1.0`, matching the upstream FFPA tuning.

**Maintainer note:** the AOT-export-time tensor setup in
`_create_bsnd_tensor` deliberately uses `mark_compact_shape_dynamic` *only*
and does not call `mark_layout_dynamic`. The latter would mark every
stride dynamic — including the one carrying the static `D=512` — and that
static factor is what lets the IR verifier prove 16-byte alignment for the
128-bit `cp.async` source pointer. Do not reintroduce `mark_layout_dynamic`
without first reworking the alignment chain.

## Standalone Commands

Export the registry variant (FP16, the registry's default):

```bash
cd kernelSrcs/ffpa_cutedsl
python3 fmha.py \
  --head_dim 512 \
  --m_block_size 64 --n_block_size 16 --num_threads 128 \
  --dtype Float16 \
  --is_causal --skip_rescale \
  --export_only \
  --output_dir ./out \
  --file_name ffpa_d512_causal \
  --function_prefix ffpa_d512_causal
```

Export a dense tuning variant (drop `--is_causal`; FP16):

```bash
python3 fmha.py \
  --head_dim 512 \
  --m_block_size 64 --n_block_size 16 --num_threads 128 \
  --dtype Float16 \
  --skip_rescale \
  --export_only \
  --output_dir ./out \
  --file_name ffpa_d512 \
  --function_prefix ffpa_d512
```

Smoke launch without exporting (matches the registry-exported variant —
causal, FP16, default tuning):

```bash
python3 fmha.py \
  --batch_size 1 --seqlen_q 1024 --seqlen_k 1024 --num_head 8 \
  --head_dim 512 \
  --m_block_size 64 --n_block_size 16 --num_threads 128 \
  --dtype Float16 \
  --is_causal --skip_rescale
```

For ptxas diagnostics:

```bash
FFPA_PTXAS_VERBOSE=1 python3 fmha.py --head_dim 512 ...
FFPA_PTXAS_OPTS="..." python3 fmha.py --head_dim 512 ...
```

## Adding Variants

Add a `KernelVariant` in `kernelSrcs/build_cutedsl.py` under the `ffpa` group.
Keep `script="ffpa_cutedsl/fmha.py"` and adjust `script_args`.

Useful axes:

- `--dtype BFloat16` to swap the exported dtype from FP16 back to BF16
  (BF16 is also the kernel script's CLI default — the registry pins
  FP16 explicitly).
- `--head_dim 256` for an unvalidated smaller-D sweep.
- `--m_block_size 32 --n_block_size 32 --num_threads 64` for an
  alternative-tuning sweep.

## Adapted from

- [xlite-dev/ffpa-attn](https://github.com/xlite-dev/ffpa-attn) (Apache-2.0)
  — FFPA-style FMHA forward kernel by DefTruth and the xlite-dev community.
  Original CUDA sources at `csrc/cuffpa/prefill.cuh` +
  `csrc/cuffpa/ffpa_attn_fwd.cuh`.
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) (Apache-2.0) — CuTeDSL
  Ampere FA2 example used as the structural skeleton, at
  `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py`.
