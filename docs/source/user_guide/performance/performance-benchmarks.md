# Performance Benchmarks

> **Platform:** NVIDIA Jetson AGX Thor Developer Kit (Blackwell, SM110)

## Definitions

| Term | Description |
|------|-------------|
| **Prefill time** | Average wall-clock time (ms) to process the input prompt |
| **Prefill throughput** | Prompt tokens processed per second during prefill (tok/s) |
| **Generation throughput** | Tokens generated per second during decoding (tok/s) |
| **Batch size** | Number of concurrent sequences (BS=1 = single-user latency, BS=8 = multi-user throughput) |
| **Acceptance rate** | Average tokens accepted per EAGLE verify step (higher is better) |
| **Speedup** | EAGLE generation throughput / vanilla generation throughput (same model, precision, batch size) |
| **ViT time** | Total visual encoder processing time per inference run (ms) |
| **ViT throughput** | Image tokens processed per second by the visual encoder (tok/s) |
| **GPU memory** | Peak GPU memory usage during inference (MB) |
| **MTP** | Multi-token prediction speculative decoding |

### Precision Key

| Precision | Description | Platform Requirement |
|-----------|-------------|---------------------|
| FP16 | Half-precision float | All platforms |
| FP8 | 8-bit float | SM89+ (Ada Lovelace and newer) |
| INT4 AWQ | 4-bit integer (AWQ quantization) | All platforms |
| INT4 GPTQ | 4-bit integer (GPTQ quantization) | All platforms |
| NVFP4 | NVIDIA 4-bit float | SM100+ (Blackwell and newer) |

---

## v0.7.1 Results

> **SDK Version:** TensorRT Edge-LLM 0.7.1 &nbsp;|&nbsp; **TensorRT:** 10.13.3.9

### LLM — Vanilla Decoding

| Model | Precision | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|-----------|:-----:|-------------:|:--------------:|----------------:|-------------------:|-------------:|
| Qwen3-1.7B | NVFP4 | 1 | 11.2 | 370 | 32,990 | 173.2 | 1,531 |
| Qwen3-1.7B | NVFP4 | 8 | 132.5 | 2,959 | 22,324 | 935.8 | 1,475 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 1 | 133.2 | 370 | 2,777 | 72.6 | 15,916 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 8 | 1,344.5 | 2,959 | 2,201 | 215.9 | 15,894 |
| Nemotron-3-Nano-4B | NVFP4 | 1 | 127.4 | 383 | 3,004 | 64.7 | 3,568 |
| Nemotron-3-Nano-4B | NVFP4 | 8 | 986.6 | 3,062 | 3,104 | 312.9 | 3,592 |

### Vision Language Model — Vanilla Decoding

| Model | LLM Prec | ViT Prec | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|----------|----------|:-----:|-------------:|:--------------:|----------------:|--------------:|------------:|------------:|-------------------:|-------------:|
| Qwen2.5-VL-7B-Instruct | NVFP4 | FP16 | 1 | 31.0 | 376 | 12,124 | 27.8 | 344 | 12,392 | 49.5 | 5,224 |
| Qwen3.5-0.8B | NVFP4 | FP16 | 1 | 9.4 | 287 | 30,410 | 4.3 | 262 | 60,606 | 287.0 | 1,192 |
| Qwen3.5-2B | NVFP4 | FP16 | 1 | 12.7 | 287 | 22,550 | 10.5 | 262 | 25,000 | 164.4 | 1,694 |
| Qwen3.5-27B | NVFP4 | FP16 | 1 | 103.3 | 287 | 2,775 | 15.0 | 262 | 17,483 | 16.1 | 14,725 |
| Nemotron-3-Nano-Omni-30B-A3B | NVFP4 | FP16 | 1 | 226.0 | 1,663 | 7,358 | 121.3 | 1,635 | 13,477 | 31.3 | 20,327 |

### LLM — EAGLE Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Qwen3-1.7B | Qwen3-1.7B_eagle3 | [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |

> **Note:** Both base and draft models are quantized to NVFP4.

| Model | Base Prec | Draft Prec | Batch | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup | GPU Mem (MB) |
|-------|-----------|------------|:-----:|-------------:|:--------------:|-------------------:|:-----------:|--------:|-------------:|
| Qwen3-1.7B | NVFP4 | NVFP4 | 1 | 12.2 | 370 | 339.0 | 3.7 | 1.96x | 1,534 |
| Qwen3-1.7B | NVFP4 | NVFP4 | 8 | 132.3 | 2,959 | 984.9 | 3.7 | 1.05x | 1,466 |

### Vision Language Model — MTP Speculative Decoding

> **Note:** MTP uses the model's built-in draft heads; no external draft checkpoint is required.
> **Highlight:** MTP is the main v0.7.1 performance improvement, increasing Qwen3.5 VLM BS=1 generation throughput by 1.21x to 2.12x over vanilla decoding.

| Model | Base Prec | Draft Prec | ViT Prec | Batch | Prefill (ms) | Prefill Tokens | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) | Accept Rate | GPU Mem (MB) |
|-------|-----------|------------|----------|:-----:|-------------:|:--------------:|--------------:|------------:|------------:|-------------------:|:-----------:|-------------:|
| Qwen3.5-0.8B | NVFP4 | NVFP4 | FP16 | 1 | 9.2 | 287 | 4.2 | 263 | 62,500 | 348.5 | 2.1 | 1,210 |
| Qwen3.5-0.8B | NVFP4 | NVFP4 | FP16 | 8 | 69.0 | 2,227 | 27.6 | 2,042 | 74,074 | 1,056.7 | 2.2 | 1,375 |
| Qwen3.5-2B | NVFP4 | NVFP4 | FP16 | 1 | 13.8 | 287 | 10.9 | 262 | 24,096 | 236.9 | 2.4 | 1,662 |
| Qwen3.5-2B | NVFP4 | NVFP4 | FP16 | 8 | 89.7 | 2,227 | 75.1 | 2,040 | 27,174 | 787.2 | 2.4 | 1,647 |
| Qwen3.5-27B | NVFP4 | NVFP4 | FP16 | 1 | 111.1 | 287 | 14.4 | 262 | 18,215 | 34.2 | 2.8 | 14,680 |
| Qwen3.5-27B | NVFP4 | NVFP4 | FP16 | 8 | 811.0 | 2,227 | 108.2 | 2,038 | 18,832 | 146.7 | 2.8 | 14,705 |

---

## v0.7.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.7.0 &nbsp;|&nbsp; **TensorRT:** 10.13

### LLM — Vanilla Decoding

| Model | Precision | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|-----------|:-----:|-------------:|:--------------:|----------------:|-------------------:|-------------:|
| Qwen3-1.7B | NVFP4 | 1 | 13.9 | 370 | 26,683 | 170.4 | 1,453 |
| Qwen3-1.7B | NVFP4 | 8 | 150.5 | 2,959 | 19,663 | 798.8 | 1,491 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 1 | 125.3 | 370 | 2,951 | 81.3 | 15,938 |
| Qwen3-30B-A3B-GPTQ-Int4 | INT4 GPTQ | 8 | 1,342.2 | 2,959 | 2,204 | 223.2 | 15,961 |
| Nemotron-3-Nano-4B | NVFP4 | 1 | 126.8 | 383 | 3,018 | 65.4 | 3,647 |
| Nemotron-3-Nano-4B | NVFP4 | 8 | 1,017.6 | 3,062 | 3,009 | 315.4 | 3,684 |

### Vision Language Model — Vanilla Decoding

| Model | LLM Prec | ViT Prec | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) | GPU Mem (MB) |
|-------|----------|----------|-------------:|:--------------:|----------------:|-------------------:|-------------:|
| Qwen3.5-0.8B | NVFP4 | FP16 | 7.0 | 753 | 107,571 | 232.2 | 1,052 |
| Qwen3.5-2B | NVFP4 | FP16 | 13.8 | 753 | 54,565 | 111.0 | 1,671 |
| Qwen3.5-27B | NVFP4 | FP16 | 122.6 | 753 | 6,143 | 10.5 | 14,985 |
| Nemotron-3-Nano-Omni-30B-A3B | NVFP4 | FP16 | 846.7 | 1,663 | 1,964 | 24.5 | 20,267 |

### LLM — EAGLE Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Qwen3-1.7B | Qwen3-1.7B_eagle3 | [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) |

> **Note:** Both base and draft models are quantized to NVFP4.

| Model | Base Prec | Draft Prec | Batch | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup |
|-------|-----------|------------|:-----:|-------------:|:--------------:|-------------------:|:-----------:|--------:|
| Qwen3-1.7B | NVFP4 | NVFP4 | 1 | 14.5 | 370 | 312.4 | 3.75 | 1.83x |
| Qwen3-1.7B | NVFP4 | NVFP4 | 8 | 153.5 | 2,959 | 828.8 | 3.73 | 1.04x |

---

## v0.4.0 Results

> **SDK Version:** TensorRT Edge-LLM 0.4.0 &nbsp;|&nbsp; **TensorRT:** 10.13

### LLM — Vanilla Decoding

| Model | Precision | Batch | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | Generation (tok/s) |
|-------|-----------|:-----:|-------------:|:--------------:|----------------:|-------------------:|
| Llama-3.1-8B-Instruct | INT4 AWQ | 1 | 215.5 | 383 | 1,777 | 50.8 |
| Llama-3.1-8B-Instruct | INT4 AWQ | 8 | 2737.4 | 3064 | 1,119 | 135.3 |
| Llama-3.1-8B-Instruct | NVFP4 | 1 | 31.0 | 383 | 12,355 | 54.9 |
| Llama-3.1-8B-Instruct | NVFP4 | 8 | 387.6 | 3064 | 7,905 | 308.7 |
| Qwen3-0.6B | INT4 AWQ | 1 | 21.0 | 366 | 17,429 | 270.2 |
| Qwen3-0.6B | INT4 AWQ | 8 | 241.8 | 2927 | 12,104 | 828.0 |
| Qwen3-0.6B | NVFP4 | 1 | 8.8 | 366 | 41,591 | 318.6 |
| Qwen3-0.6B | NVFP4 | 8 | 95.4 | 2927 | 30,681 | 1562.4 |
| Qwen3-4B-Instruct-2507 | INT4 AWQ | 1 | 116.2 | 364 | 3,133 | 76.4 |
| Qwen3-4B-Instruct-2507 | INT4 AWQ | 8 | 1502.3 | 2911 | 1,938 | 240.3 |
| Qwen3-4B-Instruct-2507 | NVFP4 | 1 | 22.9 | 364 | 15,895 | 90.2 |
| Qwen3-4B-Instruct-2507 | NVFP4 | 8 | 301.9 | 2911 | 9,642 | 507.4 |
| Qwen3-8B | INT4 AWQ | 1 | 212.0 | 366 | 1,726 | 47.7 |
| Qwen3-8B | INT4 AWQ | 8 | 2719.1 | 2927 | 1,076 | 162.3 |
| Qwen3-8B | NVFP4 | 1 | 32.8 | 366 | 11,159 | 53.7 |
| Qwen3-8B | NVFP4 | 8 | 425.8 | 2927 | 6,874 | 372.2 |

### Vision Language Model — Vanilla Decoding

| Model | LLM Prec | ViT Prec | Prefill (ms) | Prefill Tokens | Prefill (tok/s) | ViT Time (ms) | ViT Tok/Run | ViT (tok/s) | Generation (tok/s) |
|-------|----------|----------|-------------:|:--------------:|----------------:|---------------:|:-----------:|------------:|-------------------:|
| Qwen2.5-VL-7B-Instruct | INT4 AWQ | FP16 | 195.1 | 376 | 1,927 | 51.1 | 344 | 6,732 | 53.1 |
| Qwen2.5-VL-7B-Instruct | INT4 AWQ | FP8 | 195.1 | 376 | 1,927 | 42.7 | 344 | 8,056 | 53.1 |
| Qwen2.5-VL-7B-Instruct | NVFP4 | FP16 | 25.7 | 376 | 14,631 | 51.0 | 344 | 6,745 | 57.7 |
| Qwen2.5-VL-7B-Instruct | NVFP4 | FP8 | 25.7 | 376 | 14,631 | 42.6 | 344 | 8,075 | 57.6 |
| Qwen3-VL-2B-Instruct | INT4 AWQ | FP16 | 39.4 | 283 | 7,183 | 19.0 | 262 | 13,789 | 144.4 |
| Qwen3-VL-2B-Instruct | INT4 AWQ | FP8 | 39.4 | 283 | 7,183 | 15.4 | 262 | 17,013 | 144.7 |
| Qwen3-VL-2B-Instruct | NVFP4 | FP16 | 10.1 | 283 | 28,020 | 19.0 | 262 | 13,789 | 180.8 |
| Qwen3-VL-2B-Instruct | NVFP4 | FP8 | 10.1 | 283 | 28,020 | 15.5 | 262 | 16,903 | 181.0 |

> **Note:** ViT time = per-token ViT latency x image tokens per run. FP8 ViT reduces visual encoder time by ~17% compared to FP16 with negligible impact on generation throughput.

### LLM — EAGLE Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Llama-3.1-8B-Instruct | EAGLE3-LLaMA3.1-Instruct-8B | [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) |
| Qwen3-8B | qwen3_8b_eagle3 | [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) |

> **Note:** Both base and draft models are quantized to the same precision (INT4 AWQ or NVFP4) as listed in the table below.

| Model | Base Prec | Draft Prec | Batch | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup |
|-------|-----------|------------|:-----:|-------------:|:--------------:|-------------------:|:-----------:|--------:|
| Llama-3.1-8B-Instruct | INT4 AWQ | INT4 AWQ | 1 | 215.2 | 382 | 81.0 | 5.25 | 1.59x |
| Llama-3.1-8B-Instruct | INT4 AWQ | INT4 AWQ | 8 | 2735.5 | 3056 | 118.0 | 5.21 | 0.87x |
| Llama-3.1-8B-Instruct | NVFP4 | NVFP4 | 1 | 30.8 | 382 | 189.2 | 5.21 | 3.45x |
| Llama-3.1-8B-Instruct | NVFP4 | NVFP4 | 8 | 413.1 | 3056 | 484.7 | 5.15 | 1.57x |
| Qwen3-8B | INT4 AWQ | INT4 AWQ | 1 | 212.2 | 366 | 66.1 | 4.36 | 1.39x |
| Qwen3-8B | INT4 AWQ | INT4 AWQ | 8 | 2719.1 | 2927 | 99.1 | 4.31 | 0.61x |
| Qwen3-8B | NVFP4 | NVFP4 | 1 | 33.1 | 366 | 151.7 | 4.26 | 2.82x |
| Qwen3-8B | NVFP4 | NVFP4 | 8 | 429.1 | 2927 | 457.7 | 4.25 | 1.23x |

> **Note:** EAGLE speculative decoding provides the greatest speedup at BS=1 (latency-bound). At BS=8, base model compute is already well-utilized, limiting speculative acceleration. See [Speculative Decoding](../examples/speculative-decoding.md) for setup instructions.

### Vision Language Model — EAGLE Speculative Decoding

#### Draft Models

| Base Model | Draft Model | Source |
|------------|-------------|--------|
| Qwen2.5-VL-7B-Instruct | qwen2.5-vl-7b-eagle3-sgl | [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) |

> **Note:** Both base and draft models are quantized to the same precision as listed in the table below.

| Model | Base Prec | Draft Prec | ViT Prec | Prefill (ms) | Prefill Tokens | Generation (tok/s) | Accept Rate | Speedup |
|-------|-----------|------------|----------|-------------:|:--------------:|-------------------:|:-----------:|--------:|
| Qwen2.5-VL-7B-Instruct | INT4 AWQ | INT4 AWQ | FP16 | 195.1 | 376 | 57.3 | 3.66 | 1.08x |
| Qwen2.5-VL-7B-Instruct | NVFP4 | NVFP4 | FP16 | 25.8 | 376 | 149.6 | 3.82 | 2.59x |
| Qwen2.5-VL-7B-Instruct | NVFP4 | NVFP4 | FP8 | 32.8 | 376 | 117.3 | 3.76 | 2.04x |

---

## Key Observations

### v0.7.1

- **MTP speculative decoding:** This is the v0.7.1 performance highlight. Qwen3.5 MTP improves BS=1 generation throughput by 1.21x for 0.8B, 1.44x for 2B, and 2.12x for 27B over vanilla decoding, with BS=8 throughput up to 1,056.7 tok/s for Qwen3.5-0.8B.

### v0.7.0

- **MoE support:** Qwen3-30B-A3B-GPTQ-Int4 (MoE, 3B active params out of 30B) achieves 81.3 tok/s at BS=1 and 223.2 tok/s at BS=8 with INT4 GPTQ, demonstrating efficient sparse model inference on edge.
- **Small model throughput:** Qwen3-1.7B with NVFP4 delivers 170.4 tok/s at BS=1 and 798.8 tok/s at BS=8, suitable for latency-sensitive edge applications.
- **Qwen3.5 VLM family:** Ranges from 232.2 tok/s (0.8B) to 10.5 tok/s (27B), providing a scalable VLM option across memory and throughput budgets.
- **Nemotron-3-Nano-Omni-30B-A3B:** The first audio+video multimodal model benchmarked, achieving 24.5 tok/s generation at 20 GB GPU memory.

### v0.4.0

- **NVFP4 delivers highest throughput**: NVFP4 achieves 1.1–2.3x higher generation throughput than INT4 AWQ, with substantially faster prefill (e.g., 31 ms vs 216 ms for Llama-3.1-8B at BS=1).
- **EAGLE at BS=1 provides meaningful speedup**: 1.4–3.5x for LLMs, best for Llama-3.1-8B NVFP4 (3.45x). The draft model acceptance rate is high for Llama (~5.2 tokens/step) and moderate for Qwen3-8B (~4.3 tokens/step).
- **EAGLE at BS=8 has limited benefit**: At high batch sizes, base model compute is already well-utilized. Speedup drops to <1x for INT4 AWQ and 1.2–1.6x for NVFP4.
- **Qwen3-0.6B achieves the highest throughput**: 1562 tok/s at BS=8 with NVFP4 — a lightweight model well-suited for latency-sensitive edge applications.

### General

- All benchmarks use default TensorRT Edge-LLM inference settings on Jetson AGX Thor. Production performance may vary with system-level tuning (power mode, memory configuration, thermal management).
