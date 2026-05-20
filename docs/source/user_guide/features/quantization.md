# Quantization

Use `experimental.quantization` when you start from an FP16/BF16 checkpoint and need to create a unified quantized checkpoint for `llm_loader`.

The quantization CLI writes a unified HuggingFace-style checkpoint directory that `llm_loader` can export directly.

Skip this step when you already have a supported pre-quantized HuggingFace checkpoint.

## Setup

Install `experimental/quantization/requirements.txt` from the Installation Guide before running the quantization CLI.

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
cd $EDGE_LLM_PATH
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH
python -m experimental.quantization --help
```

## Quantize An LLM

```bash
python -m experimental.quantization llm \
  --model_dir Qwen/Qwen3-0.6B \
  --output_dir /tmp/qwen3_0_6b_nvfp4 \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4
```

The output directory is a HuggingFace-style checkpoint that `llm_loader` can export directly. See [Quick Start Guide](../getting_started/quick-start-guide.md) for the full export-build-inference workflow.

## Enable FP8 KV Cache

```bash
python -m experimental.quantization llm \
  --model_dir Qwen/Qwen3-8B \
  --output_dir /tmp/qwen3_nvfp4_fp8kv \
  --quantization nvfp4 \
  --kv_cache_quantization fp8
```

When this checkpoint is exported with `llm_loader`, FP8 KV cache is detected from the checkpoint metadata automatically. See [FP8 KV Cache](FP8KV.md) for details on FP8 KV cache behavior and platform requirements.

## Quantize A Visual Tower To FP8

For supported VLM checkpoints, the standalone quantizer can also calibrate the
visual tower to FP8. Use a multimodal calibration dataset so the visual path sees
real image activations:

```bash
python -m experimental.quantization llm \
  --model_dir Qwen/Qwen3-VL-2B-Instruct \
  --output_dir /tmp/qwen3_vl_fp8_visual \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4 \
  --visual_quantization fp8 \
  --dataset lmms-lab/MMMU
```

## Quantize An EAGLE3 Draft

```bash
python -m experimental.quantization draft \
  --base_model_dir /path/to/base_model \
  --draft_model_dir /path/to/eagle3_draft \
  --output_dir /tmp/eagle3_draft_fp8 \
  --quantization fp8
```

## Quantize Embedding Table To FP8

FP8 embedding quantization is applied at export time via `llm_loader`, not during the quantization step. Pass `--fp8-embedding` when exporting the quantized checkpoint. See [FP8 Embedding](fp8-embedding.md) for details and usage examples.

## Supported Methods

| Component | Methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| KV cache | `fp8` |
| Visual tower | `fp8` |

## Notes

- The package writes unified checkpoints only. It does not export ONNX or build TensorRT engines.
- Audio calibration is not implemented.
- GPTQ checkpoints are loaded as pre-quantized checkpoints; this package does not create GPTQ models.
