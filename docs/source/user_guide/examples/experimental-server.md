# Experimental High-Level Python API and Server

The experimental Python API wraps export, engine build, engine loading, generation, streaming, and OpenAI-compatible serving.

> **Status:** Experimental. API may change between releases.

## Prerequisites

Build the C++ runtime and pybind extension first:

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
cd $EDGE_LLM_PATH
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH

mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
```

Install server dependencies:

```bash
cd $EDGE_LLM_PATH
pip install -r requirements.txt
pip install -r experimental/llm_loader/requirements.txt
pip install pybind11 fastapi uvicorn
```

Run examples from the repository root with the same `PYTHONPATH`. The
repository root makes `experimental.server` importable; the `experimental`
directory makes the `llm_loader` top-level package importable.

## Python API

From a HuggingFace checkpoint:

```python
from experimental.server import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-1.7B")
outputs = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0].text)
```

From existing ONNX or engine directories:

```python
from experimental.server import LLM

llm = LLM(onnx_dir="/path/to/llm_onnx")
llm = LLM(engine_dir="/path/to/llm_engine")
```

Streaming:

```python
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir="/path/to/llm_engine")

for delta in llm.generate_stream(
    [{"role": "user", "content": "Tell me a story."}],
    SamplingParams(max_tokens=256),
):
    print(delta.text, end="", flush=True)
```

## OpenAI-Compatible Server

```bash
python -m experimental.server \
  --model Qwen/Qwen3-1.7B \
  --port 8000
```

Query:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128}'
```

Streaming query:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128, "stream": true}'
```

## Common Inputs

`LLM` requires exactly one source:

| Source | Meaning |
|---|---|
| `model` | HuggingFace model ID or local checkpoint; export, build, then load |
| `onnx_dir` | Existing ONNX directory; build then load |
| `engine_dir` | Existing engine directory; load only |

For VLMs, also pass `visual_onnx_dir` or `visual_engine_dir`.

## Sampling Parameters

| Parameter | Default | Description |
|---|---:|---|
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `top_k` | `50` | Top-K sampling |
| `max_tokens` | `2048` | Maximum generated tokens |
| `enable_thinking` | `False` | Enables Qwen-style thinking output |
| `disable_spec_decode` | `False` | Disables EAGLE for one request |

## EAGLE

```python
from experimental.server import LLM, SamplingParams

llm = LLM(
    eagle_engine_dir="/path/to/eagle/engines",
    draft_top_k=10,
    draft_step=6,
    verify_tree_size=60,
)

outputs = llm.generate(
    ["Explain quantum computing."],
    SamplingParams(max_tokens=256),
)
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List models |
| `POST` | `/v1/chat/completions` | Chat completions with optional SSE streaming |

## Notes

- Chat templates are applied in the C++ runtime, not in Python.
- Thinking output is returned in `reasoning`; final answer text is returned in `content`.
- Supported finish reasons are `stop`, `length`, `cancelled`, and `error`.
