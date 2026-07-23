# Standalone Quantization for TensorRT Edge-LLM

Decoupled quantization pipeline that produces **unified HuggingFace
checkpoints** (safetensors + config) consumable by `tensorrt_edgellm`.
Runs in a clean venv with the dependencies listed in `requirements.txt`.
See `requirements.txt` for the full dependency list.

## Design

### Problem

The old quantization path imported model-specific loaders, custom decoder
layers, ONNX plugins, and TensorRT native ops. This coupled quantization
(a GPU-only, PyTorch-only step) with the export pipeline, making the venv
heavy and brittle.

### Solution

```
┌──────────────────────────────────────────────┐
│           tensorrt_edgellm/quantization          │
│                                              │
│  quantization_configs.py  Recipe presets     │
│  quantize.py              Load → quant → save│
│  models/                                     │
│    eagle3_draft.py        Eagle3 draft model │
│    dflash_draft.py        DFlash draft model │
│  ../scripts/quantize.py    CLI entry point   │
└──────────────────┬───────────────────────────┘
                   │ unified checkpoint
                   ▼
┌──────────────────────────────────────────────┐
│          tensorrt_edgellm             │
│                                              │
│  Load checkpoint → build ONNX models        │
└──────────────────────────────────────────────┘
```

**Key decisions:**

1. **Model loading** uses only `AutoModelForCausalLM` /
   `AutoModelForImageTextToText`.  Required preprocessing, such as
   Phi-4-Multimodal LoRA merge, should be run before this quantization step.

2. **`models/` subfolder** holds standalone model implementations for
   architectures not available in HuggingFace `transformers`.  For
   example, EAGLE3 (speculative-decoding draft model) is reimplemented
   using standalone PyTorch building blocks (`RMSNorm`, `SwiGLUMLP`,
   `RotaryEmbedding`) with no dependency on `transformers` model
   classes.  Only the calibration `forward` is implemented; the full
   inference forward with KV-cache and GatherND belongs in the export
   layer.

3. **Calibration** uses explicit datasets by modality: text data for LLM,
   LM-head, KV-cache, CodePredictor, MTP, EAGLE3, and DFlash quantization;
   image-question data when `--visual_quantization fp8` is requested; and
   audio-transcript data for Qwen3-ASR/audio quantization.

4. **Output** is always a unified HuggingFace checkpoint produced by
   `modelopt.torch.export.export_hf_checkpoint`.  This is the format
   that `tensorrt_edgellm` reads.

### Supported quantization methods

| Backbone        | LM-head         | KV-cache | Visual tower | Audio tower | CodePredictor |
|-----------------|-----------------|----------|--------------|-------------|---------------|
| `fp8`           | `fp8`           | `fp8`    | `fp8`        | `fp8`       | `fp8`         |
| `int4_awq`      | `int4_awq`      | —        | —            | —           | —             |
| `nvfp4`         | `nvfp4`         | —        | —            | —           | —             |
| `mxfp8`         | `mxfp8`         | —        | —            | —           | —             |
| `int8_sq`       | —               | —        | —            | —           | —             |

Any backbone method can be combined with any lm_head method.  Examples:

- `nvfp4` backbone + `nvfp4` lm_head → full NVFP4
- `nvfp4` backbone + `fp8` lm_head → mixed precision
- `nvfp4` backbone + `int4_awq` lm_head → INT4 lm_head
- `fp8` backbone + `fp8` lm_head + `fp8` kv_cache → full FP8

## Usage

### LLM quantization

```bash
tensorrt-edgellm-quantize llm \
    --model_dir /path/to/Qwen3.5-0.8B \
    --output_dir /path/to/output \
    --quantization nvfp4 \
    --lm_head_quantization nvfp4
```

Calibration uses the default `cnn_dailymail` text dataset; pick another with
`--text_dataset <name>`.

### Speculative draft quantization

```bash
tensorrt-edgellm-quantize draft \
    --base_model_dir /path/to/base_model \
    --draft_model_dir /path/to/eagle3_draft \
    --output_dir /path/to/output \
    --quantization fp8
```

The `draft` command auto-detects DFlash draft checkpoints from
`config.json::dflash_config`; otherwise it uses the Eagle3 draft flow.

DFlash draft quantization has been validated with NVFP4:

```bash
tensorrt-edgellm-quantize draft \
    --base_model_dir /path/to/base_model \
    --draft_model_dir /path/to/dflash_draft \
    --output_dir /path/to/output \
    --quantization nvfp4 \
    --lm_head_quantization nvfp4
```

DFlash uses the base model only for calibration hidden states and for the
LM-head fallback when the draft checkpoint does not carry its own LM head.
The DFlash target-hidden projector `fc` is intentionally excluded from PTQ so
the exporter can keep its full-FP32 accumulation path.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--quantization` | None | Backbone method |
| `--lm_head_quantization` | None | LM-head method |
| `--kv_cache_quantization` | None | KV-cache method (fp8) |
| `--visual_quantization` | None | Visual tower method (fp8) |
| `--audio_quantization` | None | Audio tower method (fp8) |
| `--cp_quantization` | None | Qwen3-Omni / Qwen3-TTS CodePredictor method (fp8) |
| `--dtype` | fp16 | Loading dtype |
| `--device` | cuda | CUDA device |
| `--text_dataset` | cnn_dailymail | Registered text dataset name (`cnn_dailymail`, `wikitext`) |
| `--image_dataset` | mmmu | Registered image dataset name (used with `--visual_quantization`) |
| `--audio_dataset` | librispeech | Registered audio dataset name (Qwen3-ASR / audio tower) |
| `--num_samples` | 512 | Number of calibration samples |

Calibration datasets are selected **by name**. An unknown name fails out with
the list of registered datasets and a pointer to the customization guide. To
add your own dataset, or run a built-in offline, see
[Calibration Dataset Customization](../../docs/source/developer_guide/customization/calibration-datasets.md).

### Python API

```python
from tensorrt_edgellm.quantization import quantize_and_export

quantize_and_export(
    model_dir="/path/to/model",
    output_dir="/path/to/output",
    quantization="nvfp4",
    lm_head_quantization="nvfp4",
    text_dataset="cnn_dailymail",   # registered name, instance, or None
)
```

## End-to-end workflow

This package covers **step 1** (quantization) of the TensorRT Edge-LLM
export pipeline.  The full workflow is:

```bash
# 1. Quantize (this package — runs on any CUDA host)
tensorrt-edgellm-quantize llm \
    --model_dir /data/Qwen3.5-0.8B \
    --output_dir /tmp/qwen35-nvfp4 \
    --quantization nvfp4 --lm_head_quantization nvfp4

# 2. Export to ONNX (tensorrt_edgellm — runs on any CUDA host)
tensorrt-edgellm-export \
    /tmp/qwen35-nvfp4 /tmp/qwen35-onnx
```

Steps 3 (TRT engine build) and 4 (inference) run on the **edge device**
using the C++ runtime.  See the
[Quick Start Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)
for engine build and inference instructions.
