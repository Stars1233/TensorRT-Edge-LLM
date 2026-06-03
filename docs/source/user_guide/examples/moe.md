# MoE (Mixture of Experts)

Complete workflow for Mixture of Experts (MoE) models using a pre-quantized GPTQ-Int4 model.

**Currently supported models:**
- [Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4)
- [nvidia/Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4)

> **Note:** MoE export always runs on **CPU**. No GPU or device flag is required for the export step.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Export (x86 Host, CPU-only)

Export always runs on CPU; no GPU is required:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-30B-A3B-GPTQ-Int4
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

tensorrt-edgellm-export \
  Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  $MODEL_NAME/exported

mkdir -p $MODEL_NAME/onnx
cp -a $MODEL_NAME/exported/llm/. $MODEL_NAME/onnx/
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engine (Edge Device)

`llm_build` is the same as for regular LLMs:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-30B-A3B-GPTQ-Int4
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 3072 \
  --maxKVCacheCapacity 4096
```

## Step 4: Run Inference (Edge Device)

`llm_inference` is the same as for regular LLMs:

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json
```

MoE export uses CPU-only; build and inference use the same `llm_build` and `llm_inference` as standard LLMs.
