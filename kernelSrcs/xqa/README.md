# XQA - A set of optimized kernels for generation-phase MQA/GQA

This document outlines the process for generating the pre-compiled XQA CUDA kernel binaries (`.cubin` files) and how to run the associated unit tests.

The xqa source is from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) commit [`a4b4ed45`](https://github.com/NVIDIA/TensorRT-LLM/commit/a4b4ed45359167eb6cf3c2100d5d0dcd326bc588).

## 1. Generating Kernel Binaries (CUBINs)

CMake is the only supported entry point for generating XQA cubin C++ blobs.
During configure, CMake derives the required SM architectures from
`CMAKE_CUDA_ARCHITECTURES`, enumerates the expected outputs, and wires the
generated sources from the build tree into `edgellmKernels` and `edgellmCore`.

Generated files are written under:

```bash
<build-dir>/generated/xqa/cubin/
```

Edits to existing files under `kernelSrcs/xqa/` automatically regenerate the
build-tree cubins on the next build of a target that depends on them. Re-run
CMake configuration if you add or remove XQA source files, change the generated
cubin output list, or change `CMAKE_CUDA_ARCHITECTURES`.

Do not run `gen_cubins.py` directly and do not check generated XQA cubin blobs
into the source tree.

### 1.1. Supported SM selection

The CMake integration supports `80`, `86`, `87`, `89`, `100`, `101`, `120`,
and `121`. `110` is normalized to `101` for XQA runtime lookup.

---

## 2. Kernel Unit Tests

The project includes a suite of unit tests to verify the correctness of the attention kernels.

The test executable will be located in your build directory (e.g., `build/`). You can use `gtest_filter` to run specific tests.

**To run all primary attention and tree-attention decoding tests:**
```bash
./build/unitTest --gtest_filter=XQAAttentionDecodingTest.*:XQATreeAttentionDecodingTest.*
```

**To list all available tests:**
```bash
./build/unitTest --gtest_list_tests
```

## 3. Generating New Cubins

If you encounter a scenario that requires a kernel with parameters not covered by the pre-compiled cubins, you can generate a new one.

To do this, modify the compile configurations in `kernelSrcs/xqa/gen_cubins.py`. Locate the `edgellm_config_list` or `edgellm_config_list_spec_dec` and modify the `CompileMacroOption` entries to match your required parameters. Then re-run CMake configure/build so the generated output list and build-tree cubins are updated through the normal CMake target.

For example:
```python
edgellm_config_list = [[
    CompileMacroOption('DTYPE', 'dt', ['__half']),
    CompileMacroOption('HEAD_ELEMS', 'd', [128, 64, 32]),
    CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
    CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0]),
    # 0 denotes contiguous kv cache; 128 denotes paged kv cache page size.
    CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV', [0, 128]),
    CompileMacroOption('SLIDING_WINDOW', 'sw', [0, 1]),
    CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [1, 2, 3, 4, 5, 6, 7, 8]),
    CompileMacroOption('M_TILESIZE', 'm', [8]),
    CompileMacroOption('SPEC_DEC', 'spec_dec', [0]),
]]
```
After modifying the script, re-run CMake configure/build to generate the new cubin in the build tree.

**NOTE：** The adapt_source.patch file in this directory records the adaptations made for EdgeLLM to the source XQA code. However, you do not need to apply it, as these changes have already been included in the source.
