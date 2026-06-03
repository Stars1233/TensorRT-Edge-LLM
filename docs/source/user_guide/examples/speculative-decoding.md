# Speculative Decoding

## EAGLE3

EAGLE3 [https://arxiv.org/abs/2503.01840] uses a smaller draft model to accelerate generation. This guide covers EAGLE for both text-only LLMs and vision-language models.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

### Supported Draft Models

| Base Model | Draft Model |
|------------|-------------|
| Llama-3.1-8B-Instruct | [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) |
| Qwen3-1.7B | [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |
| Qwen3-4B | [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) |
| Qwen3-8B | [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) |
| Qwen2.5-VL-7B-Instruct | [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) |

Any EAGLE3-compatible draft model on HuggingFace can be tried however TensorRT Edge-LLM team does not test the accuracy or acceptance rate. Search for [eagle3 models](https://huggingface.co/models?search=eagle3) to find additional options.

Draft model quantization is supported via `tensorrt-edgellm-quantize` with
`fp8`, `int4_awq`, `nvfp4`, `mxfp8`, and `int8_sq` backbone quantization. For
example:

```bash
tensorrt-edgellm-quantize draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization nvfp4 \
  --output_dir $MODEL_NAME/quantized-draft
```

Note that quantizing the draft model will cause a drop in acceptance rate compared to running it in FP16, which may reduce the overall speedup.

---

### LLM EAGLE

**Example model:** Llama-3.1-8B-Instruct with [EAGLE draft model](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)

#### Step 1: Quantize and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct
cd $WORKSPACE_DIR

# Download EAGLE draft model to workspace
git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
cd EAGLE3-LLaMA3.1-Instruct-8B && git lfs pull && cd ..

# Quantize base model
tensorrt-edgellm-quantize llm \
  --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-base

# Export base model with EAGLE flag
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-base \
  $MODEL_NAME/onnx/base_export \
  --eagle-base

# Quantize draft model
tensorrt-edgellm-quantize draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-draft

# Export draft model
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-draft \
  $MODEL_NAME/onnx/draft_export

# Put outputs in the layout used by the build steps below
mkdir -p $MODEL_NAME/onnx/base $MODEL_NAME/onnx/draft
cp -a $MODEL_NAME/onnx/base_export/llm/. $MODEL_NAME/onnx/base/
cp -a $MODEL_NAME/onnx/draft_export/llm/. $MODEL_NAME/onnx/draft/
```

#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct
cd /path/to/TensorRT-Edge-LLM

# Build base model EAGLE engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --specBase

# Build draft model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxDraftTreeSize 60 \
  --specDraft
```

Build time: < 5 minutes

#### Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --specDecode
```

---

### VLM EAGLE

EAGLE for vision-language models combines accelerated text generation with image understanding.

**Example model:** Qwen2.5-VL-7B-Instruct with [EAGLE3 draft model](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl)

#### Step 1: Quantize and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-7B-Instruct
cd $WORKSPACE_DIR

# Download EAGLE draft model to workspace
git clone https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl
cd qwen2.5-vl-7b-eagle3-sgl && git lfs pull && cd ..

# Quantize base model
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-base

# Export base LLM and FP16 visual encoder
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-base \
  $MODEL_NAME/onnx/base_export \
  --eagle-base

# Quantize draft model
tensorrt-edgellm-quantize draft \
  --base_model_dir Qwen/Qwen2.5-VL-7B-Instruct \
  --draft_model_dir qwen2.5-vl-7b-eagle3-sgl \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized-draft

# Export draft model
tensorrt-edgellm-export \
  $MODEL_NAME/quantized-draft \
  $MODEL_NAME/onnx/draft_export

# Put outputs in the layout used by the build steps below
mkdir -p $MODEL_NAME/onnx/base $MODEL_NAME/onnx/draft $MODEL_NAME/onnx/visual
cp -a $MODEL_NAME/onnx/base_export/llm/. $MODEL_NAME/onnx/base/
cp -a $MODEL_NAME/onnx/base_export/visual/. $MODEL_NAME/onnx/visual/
cp -a $MODEL_NAME/onnx/draft_export/llm/. $MODEL_NAME/onnx/draft/
```

#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

#### Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-7B-Instruct
cd /path/to/TensorRT-Edge-LLM

# Build base model EAGLE engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/base \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --specBase

# Build draft model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 60 \
  --maxDraftTreeSize 60 \
  --specDraft

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

Build time: < 5 minutes

#### Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --specDecode
```
---

## MTP (Multi-Token Prediction)

MTP is a speculative decoding method built into certain model architectures. Unlike EAGLE which uses a separately trained draft model, MTP uses lightweight prediction heads that are part of the base model checkpoint. The runtime reuses the same EAGLE speculative decoding pipeline with `topK=1` (linear draft chain).

So far any Qwen3.5 dense model with `num_draft_layers > 0` in its config is MTP-capable.

---

### Example

**Example model:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

#### Step 1: Export (x86 Host)

MTP export produces both the base model and draft model ONNX from a single checkpoint using `tensorrt_edgellm`:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3.5-4B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Export
tensorrt-edgellm-export \
  Qwen/Qwen3.5-4B \
  $WORKSPACE_DIR/$MODEL_NAME/onnx \
  --mtp
```

This produces:
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/llm/` — MTP base model (hybrid attention + GDN layers, with tree-attention and intermediate state outputs)
- `$WORKSPACE_DIR/$MODEL_NAME/onnx/mtp_draft/` — MTP draft head (single attention layer with Add-based hidden state fusion)


#### Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $WORKSPACE_DIR/$MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```


#### Step 3: Build Engines

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3.5-4B
cd /path/to/TensorRT-Edge-LLM

# Build MTP base engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxVerifyTreeSize 7 \
  --specBase

# Build MTP draft engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/mtp_draft \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxDraftTreeSize 7 \
  --specDraft
```


#### Step 4: Run Inference

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json \
  --specDecode \
  --specDraftTopK 1 \
  --specDraftStep 3 \
  --specVerifyTreeSize 4
```

**Key differences from EAGLE:**
- `--specDraftTopK 1`: MTP uses a linear chain (no branching), so topK=1
- `--specDraftStep 3`: Number of MTP draft tokens (typically 3-7, matching the model's MTP head count)
- `--specVerifyTreeSize 4`: Equals `draftStep + 1` for the linear chain
