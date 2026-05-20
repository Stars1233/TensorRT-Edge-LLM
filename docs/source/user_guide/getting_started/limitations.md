# Limitations and Known Issues

This page documents the known limitations and issues for each release version of TensorRT Edge-LLM.

## 0.7.0

- ~~`experimental.quantization` exposes FP8 visual-tower quantization for supported VLM checkpoints. Other visual-tower precisions are not exposed in this release.~~ Fixed in 0.7.1.
- ~~Qwen3-TTS is supported end-to-end through a hybrid export workflow: `llm_loader` exports Talker and CodePredictor, while Code2Wav tokenizer-decoder export uses the deprecated audio exporter.~~ Fixed in 0.7.1.
- Reduced-vocabulary exports with packed INT4 LM heads require `group_size=128` and a reduced vocabulary size that is a multiple of 128.

## 0.6.1

- Some large models (7B-8B) may encounter cudaMallocAsync issue on DriveOS. Please increase huge page by `echo 15658 | sudo tee /proc/sys/vm/nr_hugepages`

## 0.6.0

- TensorRT 10.15 may cause accuracy degradation with NVFP4 for some models. Use TensorRT 10.13.3.9 shipped with Jetpack 7.1 instead.
- ~~Setting `--maxBatchSize=1` may cause accuracy degradation or engine build failure for some models. Setting `--maxBatchSize=2` can resolve this.~~ Fixed in 0.7.0.
- ~~Cutedsl kernels may cause a hanging issue for some inputs when runtime batch size > 1, and therefore the kernels are only enabled when runtime batch size = 1.~~ Fixed in 0.7.0.
