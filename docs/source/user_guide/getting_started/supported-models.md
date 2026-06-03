# Supported Models

> **Code Location:** `tensorrt_edgellm/` (checkpoint export), `tensorrt_edgellm/quantization/` (checkpoint quantization), `experimental/server/` (Python API/server), `cpp/` (runtime)
>
> **Pre-Quantized Checkpoints:** When a supported pre-quantized checkpoint is available, the [checkpoint exporter](../../developer_guide/software-design/checkpoint-export.md) can export it directly without a separate quantization step.

## Support Policy

TensorRT Edge-LLM supports the checkpoint IDs listed below. Dense LLM families include official dense checkpoints below 30B parameters. Larger dense checkpoints and non-dense variants require case-by-case validation. MoE, multimodal, audio, TTS, omni, and EAGLE support is limited to the listed rows.

The model coverage list is not comprehensive, and not every listed checkpoint has been fully verified on every supported platform and precision. If a listed model does not export, build, or run correctly, please report an issue with the checkpoint ID, precision, platform, and command line used.

The model class names were checked against the installed `transformers==5.9.0` package and the upstream [Transformers model source tree](https://github.com/huggingface/transformers/tree/main/src/transformers/models). Checkpoint IDs are linked to their Hugging Face pages and grouped into original checkpoints and quantized checkpoints.

## Precision Notes

- Dense precision set: FP16/BF16 checkpoints, ModelOpt FP8/MXFP8/FP4/NVFP4/INT4 AWQ/INT8 SmoothQuant checkpoints, and INT4 GPTQ checkpoints. INT8 GPTQ is not supported.
- Jetson Orin supports FP16, INT8, and INT4 runtime precision in the supported JetPack configurations. Do not select FP8, MXFP8, FP4, or NVFP4 checkpoints for Orin.
- For INT4 engine builds on Jetson Orin devices with less system memory, such as Jetson Orin Nano, pass `--externalize-weights int4_ffn` for dense checkpoints or `--externalize-weights int4_ffn int4_moe` for MoE checkpoints to reduce engine build memory.
- For FP16/BF16 source checkpoints, use the [Quantization](../features/quantization.md) script to create a unified quantized checkpoint for `tensorrt_edgellm`, then export the generated checkpoint.
- FP8 KV cache is detected automatically from checkpoint metadata by `tensorrt_edgellm`.
- `tensorrt-edgellm-export` exports visual encoders. Use `tensorrt-edgellm-quantize --visual_quantization fp8` before export when FP8 visual weights are required.
- MXFP8 and FP4/NVFP4 require Blackwell-class hardware for runtime execution.

## Support Matrix

### Dense LLM

| Model Series | Transformers Class | `tensorrt_edgellm` Handling | Supported Precisions |
|--------------|--------------------|-----------------------|----------------------|
| Llama 3.x Instruct | [`LlamaForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | `llama` -> default `CausalLM` | Dense precision set |
| Qwen2/Qwen2.5 dense | [`Qwen2ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) | `qwen2` -> default `CausalLM` | Dense precision set |
| Qwen3 dense | [`Qwen3ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | `qwen3` -> default `CausalLM` | Dense precision set |
| Qwen3.5/3.6 text | [`Qwen3_5ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | `qwen3_5_text` -> `Qwen3_5CausalLM` | Dense precision set |
| Nemotron Nano dense | [`NemotronHForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | `nemotron_h` -> `NemotronHCausalLM` | BF16, FP8, NVFP4 |

<details>
<summary><b>Llama 3.x Instruct</b> checkpoints</summary>

**Original:**
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

**Quantized:**
- [nvidia/Llama-3.1-8B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)
- [nvidia/Llama-3.1-8B-Instruct-NVFP4](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4)

</details>

<details>
<summary><b>Qwen2/Qwen2.5 dense and Qwen-derived dense</b> checkpoints</summary>

**Original:**
- [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B)
- [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)
- [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
- [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
- [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [Qwen/Qwen2-Math-1.5B](https://huggingface.co/Qwen/Qwen2-Math-1.5B)
- [Qwen/Qwen2-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-Math-1.5B-Instruct)
- [Qwen/Qwen2-Math-7B](https://huggingface.co/Qwen/Qwen2-Math-7B)
- [Qwen/Qwen2-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Math-7B-Instruct)
- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen/Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)
- [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [Qwen/Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
- [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
- [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)
- [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [Qwen/Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B)
- [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
- [Qwen/Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B)
- [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Qwen/Qwen2.5-Coder-14B](https://huggingface.co/Qwen/Qwen2.5-Coder-14B)
- [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
- [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- [Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [Qwen/Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B)
- [Qwen/Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)

**Quantized:**
- [Qwen/Qwen2-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-AWQ)
- [Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-AWQ)
- [Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-7B-Instruct-AWQ)
- [Qwen/Qwen2-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ)
- [Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ)
- [Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ)
- [Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ)
- [Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)
- [Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ)
- [Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ)
- [Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-AWQ)
- [Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ)
- [Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-Coder-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-AWQ)
- [Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4)

</details>

<details>
<summary><b>Qwen3 dense</b> checkpoints</summary>

**Original:**
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)
- [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base)

**Quantized:**
- [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ)
- [Qwen/Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ)
- [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ)
- [nvidia/Qwen3-8B-FP8](https://huggingface.co/nvidia/Qwen3-8B-FP8)
- [nvidia/Qwen3-8B-NVFP4](https://huggingface.co/nvidia/Qwen3-8B-NVFP4)
- [nvidia/Qwen3-14B-FP8](https://huggingface.co/nvidia/Qwen3-14B-FP8)
- [nvidia/Qwen3-14B-NVFP4](https://huggingface.co/nvidia/Qwen3-14B-NVFP4)

</details>

<details>
<summary><b>Qwen3.5/3.6 text</b> checkpoints</summary>

**Qwen3.5:**
- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)
- [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base)
- [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)

**Qwen3.6 (same architecture as Qwen3.5):**
- [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)

</details>

<details>
<summary><b>Nemotron Nano dense</b> checkpoints</summary>

**Original:**
- [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)

**Quantized:**
- [nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8)
- [nvidia/NVIDIA-Nemotron-3-Nano-4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-NVFP4)
- [nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8)
- [nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4)

</details>

---

### MoE

| Model Series | Transformers Class | `tensorrt_edgellm` Handling | Supported Precisions |
|--------------|--------------------|-----------------------|----------------------|
| Qwen3-MoE | [`Qwen3MoeForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) | `qwen3_moe` -> `Qwen3MoeCausalLM` | INT4, NVFP4 |
| Qwen3.5/3.6-MoE | [`Qwen3_5MoeForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) | `qwen3_5_moe` -> `Qwen3_5MoeCausalLM` + `Qwen3_5VLVisualModel` | INT4 GPTQ, NVFP4 |
| Nemotron3-MoE | [`NemotronHForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | `nemotron_h` -> `NemotronHCausalLM` | NVFP4 only |

For NVFP4 MoE exports, the `--nvfp4-moe-backend` flag selects the plugin backend:

- `thor` — uses `Nvfp4MoePlugin` (SM100/101/110, CuTe DSL kernels). Default when checkpoint config does not specify.
- `geforce` — uses `NvFP4MoEPluginGeforce` (SM120/121).

```bash
tensorrt-edgellm-export \
    /path/to/Qwen3-MoE-NVFP4 \
    /tmp/qwen3_moe_onnx \
    --nvfp4-moe-backend thor
```

<details>
<summary><b>Qwen3-MoE</b> checkpoints</summary>

- [Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4)
- [nvidia/Qwen3-30B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4)

</details>

<details>
<summary><b>Qwen3.5/3.6-MoE</b> checkpoints</summary>

- [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4)

</details>

<details>
<summary><b>Nemotron3-MoE</b> checkpoints</summary>

- [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)

</details>

---

### VLM

| Model Series | Transformers Class | `tensorrt_edgellm` Handling | Supported Precisions |
|--------------|--------------------|-----------------------|----------------------|
| Qwen2.5-VL | [`Qwen2_5_VLForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py) | `qwen2_5_vl` + `Qwen2_5VLVisualModel` | Dense precision set for LLM backbone |
| Qwen3-VL / compatible | [`Qwen3VLForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) | `qwen3_vl` + `Qwen3VLVisualModel` | Dense precision set for LLM backbone |
| Qwen3.5/3.6 VLM | [`Qwen3_5ForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | `qwen3_5` -> `Qwen3_5CausalLM` + `Qwen3_5VLVisualModel` | VLM original checkpoints only |
| InternVL3 / InternVL3.5 HF format | [`InternVLForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/internvl/modeling_internvl.py) | `internvl_chat` / `internvl` + InternVL visual models | Dense precision set for LLM backbone |
| Phi-4-Multimodal | [`Phi4MultimodalForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py) | `phi4mm` / `phi4_multimodal` + `Phi4MMVisualModel` | Merge vision LoRA, then dense precision set for the LLM backbone |

<details>
<summary><b>Qwen2.5-VL</b> checkpoints</summary>

**Original:**
- [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

**Quantized:**
- [Qwen/Qwen2.5-VL-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ)
- [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)
- [nvidia/Qwen2.5-VL-7B-Instruct-FP8](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-FP8)
- [nvidia/Qwen2.5-VL-7B-Instruct-NVFP4](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4)

</details>

<details>
<summary><b>Qwen3-VL / compatible</b> checkpoints</summary>

**Original:**
- [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen/Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking)
- [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Qwen/Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking)
- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
- [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)

**Quantized:**
- [nvidia/Cosmos-Reason2-2B-FP8](https://huggingface.co/nvidia/Cosmos-Reason2-2B-FP8)
- [nvidia/Cosmos-Reason2-2B-NVFP4](https://huggingface.co/nvidia/Cosmos-Reason2-2B-NVFP4)
- [nvidia/Cosmos-Reason2-8B-FP8](https://huggingface.co/nvidia/Cosmos-Reason2-8B-FP8)
- [nvidia/Cosmos-Reason2-8B-NVFP4](https://huggingface.co/nvidia/Cosmos-Reason2-8B-NVFP4)

</details>

<details>
<summary><b>Qwen3.5/3.6 VLM</b> — same checkpoints as Qwen3.5/3.6 text</summary>

Qwen3.5 and Qwen3.6 checkpoints are unified text+VLM models. The same checkpoints listed under [Qwen3.5/3.6 text](#dense-llm) are used; `tensorrt_edgellm` selects the VLM path (`qwen3_5` handler) when visual inputs are provided.

</details>

<details>
<summary><b>InternVL3 / InternVL3.5 HF format</b> checkpoints</summary>

**Original:**
- [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)
- [OpenGVLab/InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf)
- [OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf)
- [OpenGVLab/InternVL3-9B](https://huggingface.co/OpenGVLab/InternVL3-9B)
- [OpenGVLab/InternVL3-9B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-9B-Instruct)
- [OpenGVLab/InternVL3-14B-hf](https://huggingface.co/OpenGVLab/InternVL3-14B-hf)
- [OpenGVLab/InternVL3_5-1B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF)
- [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF)
- [OpenGVLab/InternVL3_5-4B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF)
- [OpenGVLab/InternVL3_5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF)
- [OpenGVLab/InternVL3_5-14B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF)

**Quantized:**
- [OpenGVLab/InternVL3-1B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-1B-AWQ)
- [OpenGVLab/InternVL3-2B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-2B-AWQ)
- [OpenGVLab/InternVL3-8B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-8B-AWQ)
- [OpenGVLab/InternVL3-14B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-14B-AWQ)

</details>

<details>
<summary><b>Phi-4-Multimodal</b> checkpoints</summary>

- [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

</details>

---

### VLA

| Model Series | Transformers Class | `tensorrt_edgellm` Handling | Supported Precisions |
|--------------|--------------------|-----------------------|----------------------|
| Alpamayo R1 | Checkpoint architecture `alpamayo_r1`; VLM backbone compatible with [`Qwen3VLForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) | `qwen3_vl` + `Qwen3VLVisualModel` + `AlpamayoAction` | FP16 |

<details>
<summary><b>Alpamayo R1</b> checkpoints</summary>

- [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)

</details>

---

### Audio / Speech

| Model Series | Transformers Class | `tensorrt_edgellm` Handling | Supported Precisions |
|--------------|--------------------|-----------------------|----------------------|
| Qwen3-ASR | Checkpoint architecture `Qwen3ASRForConditionalGeneration`; text backbone compatible with [`Qwen3ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | `Qwen3ASRLanguageModel` + `QwenAudioEncoder` | FP16 |

<details>
<summary><b>Qwen3-ASR</b> checkpoints</summary>

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

</details>

---

### TTS

| Model Series | Transformers Class | `tensorrt_edgellm` Handling | Supported Precisions |
|--------------|--------------------|-----------------------|----------------------|
| Qwen3-TTS | Checkpoint architecture `Qwen3TTSForConditionalGeneration`; talker/code-predictor decoders compatible with [`Qwen3ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | `TalkerCausalLM` + `CodePredictorCausalLM` + Code2Wav from `speech_tokenizer/` | FP16 |

<details>
<summary><b>Qwen3-TTS</b> checkpoints</summary>

- [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)

</details>

---

### Omni

| Model Series | Transformers Class | `tensorrt_edgellm` Handling | Supported Precisions |
|--------------|--------------------|-----------------------|----------------------|
| [Nemotron-Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/blob/main/modeling.py) | Checkpoint architecture `NemotronH_Nano_Omni_Reasoning_V3`; LLM is Nemotron-H compatible with [`NemotronHForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | `NemotronHCausalLM` + `NemotronOmniVisualModel` + `NemotronOmniAudioModel` | NVFP4 only |

<details>
<summary><b>Nemotron-Omni</b> checkpoints</summary>

- [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)

</details>

---

Qwen3-ASR and Qwen3-TTS use checkpoint architecture names that are not present in the installed `transformers==5.3.0` package, so TensorRT Edge-LLM handles their speech/audio/talker/Code2Wav components with local model implementations. Qwen3-TTS support is limited to the CustomVoice checkpoints listed above.

## EAGLE3 Draft Models

EAGLE3 draft checkpoints are detected by `draft_vocab_size` in `config.json` and exported with `Eagle3DraftModel`. Draft checkpoints can be quantized with `tensorrt-edgellm-quantize` using the same ModelOpt methods exposed by the draft quantization CLI: `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, and `int8_sq` for the backbone; `fp8`, `int4_awq`, `nvfp4`, and `mxfp8` for the LM head; and `fp8` for KV cache.

| Draft checkpoint | Base model | Draft config class |
|------------------|------------|--------------------|
| [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `LlamaForCausalLM`-style draft |
| [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `LlamaForCausalLMEagle3`-style draft |
| [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | `Eagle3LlamaForCausalLM`-style draft |
| [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `LlamaForCausalLMEagle3`-style draft |
| [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `LlamaForCausalLMEagle3`-style draft |
| [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `LlamaForCausalLMEagle3`-style draft |
