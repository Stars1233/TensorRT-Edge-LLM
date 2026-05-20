# Python Export Pipeline (Deprecated)

The historical FX-tracing export package remains available in 0.7.1 for
compatibility and for components that are not yet fully covered by
`llm_loader`. For new enablement, use:

1. `experimental/quantization` to create unified quantized HuggingFace-style
   checkpoints when starting from FP16/BF16 source checkpoints.
2. `experimental/llm_loader` to export ONNX and runtime sidecars from FP16/BF16,
   pre-quantized, or locally quantized checkpoints.
3. `llm_build`, `visual_build`, `audio_build`, `action_build`, and the runtime
   examples to build engines and run inference.

The `tensorrt_edgellm/` folder will be removed in 0.8.0 after the
`experimental/quantization` -> `experimental/llm_loader` workflow reaches full
feature parity for all models and features.

## Replacement Map

| Legacy responsibility | Current workflow |
|---|---|
| ModelOpt quantization inside the export package | `python -m experimental.quantization ...` |
| LLM ONNX export | `python -m llm_loader.export_all_cli <checkpoint> <output_dir>` |
| VLM/audio/Omni/VLA component export | `llm_loader.export_all_cli` component orchestration |
| EAGLE base export | `llm_loader.export_all_cli --eagle-base` |
| MTP export | `llm_loader.export_all_cli --mtp` |
| LoRA graph insertion and adapter processing | `python -m llm_loader.lora.insert_lora_cli` and `python -m llm_loader.lora.process_lora_weights_cli` |
| Static LoRA merge | `python -m llm_loader.lora.merge_lora_cli` before quantization/export |
| FP8 KV cache | `experimental.quantization --kv_cache_quantization fp8`; `llm_loader` detects metadata during export |
| FP8 embedding sidecar | `llm_loader.export_all_cli --fp8-embedding` |
| Vocabulary reduction | `python -m llm_loader.vocab_reduction`, then `llm_loader.export_all_cli --reduced-vocab-dir` |

## Where To Go

- [Installation](../../user_guide/getting_started/installation.md)
- [Quick Start Guide](../../user_guide/getting_started/quick-start-guide.md)
- [Quantization](../../user_guide/features/quantization.md)
- [Checkpoint-Based Model Loader Design](llm-loader.md)
- [Experimental Quantization Package Design](experimental-quantization.md)
- [LoRA](../../user_guide/features/lora.md)
- [Vocabulary Reduction](../../user_guide/features/reduce-vocab.md)
