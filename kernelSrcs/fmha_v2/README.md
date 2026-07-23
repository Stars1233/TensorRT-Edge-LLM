# FMHA_v2

This document outlines the process for generating the pre-compiled FMHA_V2 CUDA kernel binaries (`.cubin` files) and how to run the associated unit tests.

## 1. Generating Kernel Binaries (CUBINs)

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout 0ffa77af51b272ba27424564ed253096d6f0f11a
git apply /path/to/gen_fmha_cubin.patch
cd cpp/kernels/fmha_v2
```

### 1.1. Generate cubins for SM 80, 86, 87, 89, 100, 101

These architectures can be compiled with a **CUDA 12.8** Toolkit.

**Steps:**
```bash
# 1) Generate the arch–specific .cu sources & headers
export GENERATE_EDGE_LLM=1 GENERATE_CUBIN=1 ENABLE_SM10X=1
python3 setup.py

# 2) Build the cubins (old BERT parameter layout)
make cubin_demobert -j$(nproc)

# 3) Avoid overwrite
mv generated generated_cuda128
mv cubin cubin_cuda128
```
---

### 1.2. Generate cubins for SM 120, 121

These newer architectures require a more recent **CUDA 12.9** or higher Toolkit.

**Steps:**

1.  **Generate SM 12x CUBINs:**

```bash
export GENERATE_EDGE_LLM=1 GENERATE_CUBIN=1 ENABLE_SM12X=1 

# 1) Generate Blackwell-only .cu sources & headers
python3 setup.py

# 2) Build the cubins (old BERT parameter layout)
make cubin_demobert -j$(nproc)

# 3) Avoid overwrite
mv generated generated_cuda129
mv cubin cubin_cuda129
```

2.  **Merge the new CUBIN files:**

Merge the `generated_cuda128` and `generated_cuda129` dirs into a single cubin dir, located at `cpp/kernels/contextAttentionKernels/cubin`.

### 1.3. Generate standalone custom-mask cubins

The vision-block attention (packed custom mask, `ContextAttentionMaskType::CUSTOM_MASK`)
kernels are generated in a separate round so the padding/causal/sliding cubins
above are not touched. With `GENERATE_CUSTOM_MASK_ONLY=1`, `setup.py` emits one
standalone `*_custom_mask_sm<XX>` cubin for the head-size-256 separate-q-k-v
layout (the only consumer: Gemma4 Unified sliding layers), and
`generated/fmha_cubin.cpp` contains only the custom-mask externs/metadata
rows.

```bash
# Round A (SM 80/86/87/89/100/101; sm101 cubins need a CUDA 12.8/12.9 nvcc, see note above)
export GENERATE_EDGE_LLM=1 GENERATE_CUBIN=1 GENERATE_CUSTOM_MASK_ONLY=1 ENABLE_SM10X=1
python3 setup.py && make cubin_demobert -j$(nproc)
mv generated generated_cuda128_custom && mv cubin cubin_cuda128_custom

# Round B (SM 120/121)
unset ENABLE_SM10X; export ENABLE_SM12X=1
python3 setup.py && make cubin_demobert -j$(nproc)
mv generated generated_cuda129_custom && mv cubin cubin_cuda129_custom
```

Merge: copy the new `*_custom_mask_sm*.cubin.cpp` files into
`cpp/kernels/contextAttentionKernels/cubin/` and append the generated extern
declarations and metadata rows to the end of the matching per-SM
`#ifndef EXCLUDE_SM_<XX>` blocks of `fmha_cubin.h` (pre-existing lines stay
byte-identical).

> **Semantics note:** `gen_fmha_cubin.patch` changes the flash kernels
> (`fused_multihead_flash_attention_kernel_noloop{,_tiled}.h`) so that
> CUSTOM_MASK kernel traits apply the packed mask on **every** kv tile.  Stock
> fmha_v2 only applies it in the diagonal band and to its right
> (`kv_loop >= kv_mask_loop_start`), which silently attends keys left of the
> band — wrong for masks that fold a sliding window into the bits (vision-block
> prefill needs `k <= q - W` masked out whenever S > W).  The change is guarded
> by `Kernel_traits::CUSTOM_MASK`, so causal/sliding/padding kernels are
> bit-identical to stock.


## Kernel Unit Test
```bash
ln -s generated_cuda128 generated

make bin/fmha.exe -j$(nproc)

bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 1024 -d 128  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s
bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 128 -d 64  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s -force-non-tiled
bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 1024 -d 128  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s -separate-q-k-v
bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 128 -d 64  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s -force-non-tiled -separate-q-k-v
bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 1024 -d 128  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s -contiguous-q-kv
bin/fmha.exe -v 1 -runs 5 -warm-up-runs 2  -s 128 -d 64  -b 1 -causal-mask -grouped-query-attention 2 -h 14 -fix-s -force-non-tiled -contiguous-q-kv
```