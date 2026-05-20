# LoRA (Low-Rank Adaptation)

TensorRT Edge-LLM supports LoRA through the checkpoint-based `llm_loader` workflow.

Use dynamic LoRA when you need to select adapters at runtime. Use static LoRA
merge when the adapter is always required, such as the Phi-4-Multimodal
`vision-lora` adapter.

## Setup

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH
```

## Dynamic Runtime LoRA

This workflow exports the base checkpoint, inserts LoRA inputs into the ONNX
graph, processes HuggingFace adapter weights, then builds an engine with a
maximum adapter rank.

```bash
# Step 1: Export the base model with llm_loader
python -m llm_loader.export_all_cli \
  /path/to/base_model \
  /tmp/onnx_output

# Step 2: Insert LoRA support into the exported LLM graph
python -m llm_loader.lora.insert_lora_cli \
  --onnx_dir /tmp/onnx_output/llm

# Step 3: Convert each adapter to the runtime sidecar format
python -m llm_loader.lora.process_lora_weights_cli \
  --input_dir /path/to/adapter1 \
  --output_dir /tmp/onnx_output/llm/lora_weights/adapter1

python -m llm_loader.lora.process_lora_weights_cli \
  --input_dir /path/to/adapter2 \
  --output_dir /tmp/onnx_output/llm/lora_weights/adapter2

# Step 4: Build the engine with LoRA support
./build/examples/llm/llm_build \
  --onnxDir /tmp/onnx_output/llm \
  --engineDir engines \
  --maxBatchSize 1 \
  --maxLoraRank 64

# Step 5: Run inference with adapter selection in input.json
./build/examples/llm/llm_inference \
  --engineDir engines \
  --inputFile input.json \
  --outputFile output.json
```

## Static LoRA Merge

Static merge permanently applies a LoRA adapter to the base HuggingFace
checkpoint before optional quantization and ONNX export.

```bash
# Step 1: Merge LoRA into the base checkpoint
python -m llm_loader.lora.merge_lora_cli \
  --model_dir Phi-4-multimodal-instruct \
  --lora_dir Phi-4-multimodal-instruct/vision-lora \
  --output_dir merged_model

# Step 2: Optional quantization of the merged checkpoint
python -m experimental.quantization llm \
  --model_dir merged_model \
  --output_dir quantized_model \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4

# Step 3: Export the checkpoint with llm_loader
python -m llm_loader.export_all_cli \
  quantized_model \
  onnx_output
```

If you do not need weight quantization, export `merged_model` directly in step 3.

## Input Format

Specify available adapters and select one adapter per request:

```json
{
    "available_lora_weights": {
        "french": "/path/to/lora_weights/french/processed_adapter_model.safetensors",
        "medical": "/path/to/lora_weights/medical/processed_adapter_model.safetensors"
    },
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Translate to French: Hello world"}
            ],
            "lora_name": "french"
        },
        {
            "messages": [
                {"role": "user", "content": "What is aspirin?"}
            ],
            "lora_name": "medical"
        }
    ]
}
```

All requests in the same batch must use the same LoRA adapter. To disable LoRA,
omit `lora_name` or set it to an empty string.

## Script Reference

### `llm_loader.lora.insert_lora_cli`

Inserts LoRA patterns into an exported `llm_loader` ONNX model and creates
`lora_model.onnx` in the same directory.

| Argument | Required | Description |
|----------|----------|-------------|
| `--onnx_dir` | Yes | Directory containing `model.onnx` and `config.json` |

### `llm_loader.lora.process_lora_weights_cli`

Processes HuggingFace LoRA adapter weights for runtime use.

| Argument | Required | Description |
|----------|----------|-------------|
| `--input_dir` | Yes | Directory with `adapter_config.json` and `adapter_model.safetensors` |
| `--output_dir` | Yes | Output directory for processed weights |

The output contains `processed_adapter_model.safetensors` and `config.json`.

### `llm_loader.lora.merge_lora_cli`

Permanently merges LoRA weights into a base HuggingFace checkpoint.

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_dir` | Yes | - | Base model directory |
| `--lora_dir` | Yes | - | LoRA adapter directory |
| `--output_dir` | Yes | - | Output directory for merged checkpoint |
| `--device` | No | `cuda` | Device used while merging |
| `--torch-dtype` | No | `float16` | Model dtype used while merging |

## Build Parameters

| Parameter | Description |
|-----------|-------------|
| `--maxLoraRank` | Maximum LoRA rank to support. Set to `0` to disable dynamic LoRA. |

## Notes

- Static merge produces a single checkpoint and does not require runtime LoRA flags.
- Dynamic LoRA enables adapter switching without rebuilding, but all adapters must have rank less than or equal to `--maxLoraRank`.
- CUDA graphs are captured separately for each LoRA configuration.
