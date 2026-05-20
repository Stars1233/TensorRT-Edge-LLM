# Phi-4-Multimodal

Phi-4-Multimodal requires merging its vision LoRA adapter into the base model before quantization and export.

**Example model:** Phi-4-multimodal-instruct

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Merge, Quantize, and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Phi-4-multimodal-instruct
export MODEL_DIR=$WORKSPACE_DIR/$MODEL_NAME
export EDGE_LLM_PATH=$HOME/tensorrt-edge-llm   # path to TensorRT-Edge-LLM repo root
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Clone Phi-4-multimodal-instruct from HuggingFace
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
cd Phi-4-multimodal-instruct && git lfs pull && cd ..

# Merge vision LoRA adapter into base model
python -m llm_loader.lora.merge_lora_cli \
  --model_dir Phi-4-multimodal-instruct \
  --lora_dir Phi-4-multimodal-instruct/vision-lora \
  --output_dir $MODEL_DIR/merged

# Quantize merged model
cd $EDGE_LLM_PATH
python -m experimental.quantization llm \
  --model_dir $MODEL_DIR/merged \
  --output_dir $MODEL_DIR/quantized \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4

# Export language model and visual encoder
python -m llm_loader.export_all_cli \
  $MODEL_DIR/quantized \
  $MODEL_DIR/onnx
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_DIR/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Phi-4-multimodal-instruct
cd ~/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1536 \
  --maxKVCacheCapacity 2048

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 256 \
  --maxImageTokens 1024 \
  --maxImageTokensPerImage 512
```

## Step 4: Run Inference (Thor Device)

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json
```
