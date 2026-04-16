# Limitations and Known Issues

This page documents the known limitations and issues for each release version of TensorRT Edge-LLM.

## 0.6.1
- Some large models (7B-8B) may encounter cudaMallocAsync issue. Please increase huge page by `echo 15658 | sudo tee /proc/sys/vm/nr_hugepages`

## 0.6.0
- TensorRT 10.15 may cause accuracy degradation with NVFP4 for some models. Use TensorRT 10.13.3.9 shipped with Jetpack 7.1 instead.
- Setting `--maxBatchSize=1` may cause accuracy degradation or engine build failure for some models. Setting `--maxBatchSize=2` can resolve this.
- Cutedsl kernels may cause a hanging issue for some inputs when runtime batch size > 1, and therefore the kernels are only enabled when runtime batch size = 1.
