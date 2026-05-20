# Experimental Quantization Package Design

`experimental.quantization` is a standalone checkpoint quantization package. It converts FP16 HuggingFace checkpoints into quantized HuggingFace-style checkpoints that the checkpoint-based loader can export directly.

For command-line usage, see [Quantization](../../user_guide/features/quantization.md).

## Design Goals

- Keep quantization separate from ONNX export.
- Preserve a HuggingFace-compatible checkpoint layout.
- Store quantization metadata in checkpoint config files so `llm_loader` can infer export behavior from the checkpoint.
- Share the same output contract for locally quantized and downloaded pre-quantized checkpoints.

## Package Layout

| Path | Role |
|---|---|
| `experimental/quantization/cli.py` | User-facing `llm` and `draft` commands |
| `experimental/quantization/quantize.py` | Model loading, ModelOpt configuration, calibration, and checkpoint writing |
| `experimental/quantization/utils.py` | Shared checkpoint and config helpers |
| `experimental/llm_loader/config.py` | Reads quantization metadata during ONNX export |
| `experimental/llm_loader/checkpoint/loader.py` | Loads and repacks quantized checkpoint tensors |

## Flow

```text
FP16 checkpoint
  -> experimental.quantization
  -> quantized safetensors + quantization metadata
  -> llm_loader
  -> ONNX + runtime sidecars
```

The quantization package stops after writing the checkpoint. ONNX export, TensorRT engine build, and runtime execution remain owned by `llm_loader`, `llm_build`, and `llm_inference`.

## Artifact Contract

The output directory must contain:

- `config.json` with model architecture fields.
- Quantization metadata such as `quantization_config`, `hf_quant_config.json`, or equivalent fields understood by `llm_loader`.
- One or more `.safetensors` checkpoint shards.
- Tokenizer and processor files needed by the runtime.

`llm_loader` uses this metadata to select quantized linear layers, repack checkpoint tensors, and enable FP8 KV cache when the checkpoint marks KV cache quantization as `fp8`. No separate FP8 KV export flag is required in the `llm_loader` path.

## Supported Methods

| Component | Methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| KV cache | `fp8` |
| Visual tower | `fp8` |

Backbone and LM-head methods can be combined for mixed-precision checkpoints.
Visual FP8 requires multimodal calibration data so visual quantizers observe
image activations.

## Limitations

- Audio calibration is not implemented.
- Model-specific export workarounds that belong to ONNX export are intentionally not included.
- This package produces checkpoints only. It does not build ONNX or TensorRT engines.
