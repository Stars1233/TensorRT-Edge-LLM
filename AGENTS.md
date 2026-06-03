# AGENTS.md

TensorRT Edge-LLM: NVIDIA C++/CUDA/Python inference runtime for deploying LLMs and VLMs on edge devices (Jetson Orin, Thor, DRIVE platforms).

> If an `AGENTS.local.md` file exists alongside this file, read and respect it — it contains developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**
- Read and follow `CODING_GUIDELINES.md` for ALL code changes (C++ and Python)
- NVIDIA copyright header on ALL new files (update year on modified files) — see `LICENSE_HEADER` for the SPDX template (pre-commit `insert-license` hook auto-injects it for `.py`, `.cpp`, `.cu`, `.cuh`, `.h`, `.hpp` files)
- `git commit -s` (DCO sign-off required). Never attribute AI tools in sign-off line. Always rely on `git` to do the sign off instead of directly adding sign off in commit message.
- Do not add co-authors to the git commit message unless explicitly instructed to do so by the user.
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
- PR title format: Conventional Commits style (e.g., `feat: Add Qwen3 support`, `fix #700: Memory leak in runtime`)
- Set `TRT_PACKAGE_DIR` for all C++ builds; set `LLM_SDK_DIR` for all Python tests
- Set `LD_LIBRARY_PATH` before running any built binary: `export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib:$LD_LIBRARY_PATH`
- Git submodules must be initialized: `git submodule update --init` (googletest, nlohmann/json, NVTX)
- Model validation must exercise `export -> build -> inference` in that order. Export-only checks are useful smoke
  tests, but they are not sufficient evidence that a model works.

## Common Commands

| Task | Command |
|------|---------|
| Build (minimal) | `mkdir -p build && cd build && cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR && make -j$(nproc)` |
| Build (with unit tests) | `cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DBUILD_UNIT_TESTS=ON && make -j$(nproc)` |
| Build (cross-compile AArch64) | `cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DAARCH64_BUILD=ON && make -j$(nproc)` |
| Build (NVTX profiling) | `cmake .. -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR -DENABLE_NVTX_PROFILING=ON && make -j$(nproc)` |
| C++ unit tests (all) | `./build/unitTest` |
| C++ unit tests (filter) | `./build/unitTest --gtest_filter="LoggerTest.*"` |
| Python package (install) | `pip install -r requirements.txt && python -m build --wheel --outdir dist . && pip install dist/*.whl` |
| Python tool extras | `pip install ".[tools]"` |
| Python test suite | `pytest --priority=l0_pipeline_a30 -v` |
| Single Python test | `pytest tests/defs/test_model_export.py -v` |
| Python unit tests | `pytest tests/python-unittests/ -v` |
| Pre-commit (all files) | `pre-commit run --all-files` |
| Pre-commit (specific file) | `pre-commit run --files cpp/runtime/llmInferenceRuntime.cpp` |
| Format C++ manually | `git-clang-format --style file` |
| Code coverage | `./scripts/run_coverage.sh --trt-package-dir $TRT_PACKAGE_DIR` |

For the full end-to-end pipeline (quantize → ONNX export → engine build → inference), installation instructions, and build options, see `docs/source/developer_guide/getting-started/`.

## Architecture

For detailed software design, see `docs/source/developer_guide/software-design/`.

The pipeline is: `HuggingFace Model → Python Export (quantize + ONNX) → C++ Engine Builder (TRT engine) → C++ Runtime (inference)`.

**C++ Runtime (`cpp/`)** uses a single unified `LLMInferenceRuntime` class for all inference via `handleRequest()`. It supports both vanilla autoregressive decoding (single base engine) and speculative decoding modes (EAGLE, MTP — base + draft engines) through a pluggable `DecodingStrategy` layer.

**C++ sub-packages:** `common/` (tensor, logging, utils), `kernels/` (FMHA/RoPE/MoE/Mamba/EAGLE), `plugins/` (TRT custom plugins), `builder/` (ONNX→TRT), `tokenizer/`, `multimodal/`, `profiling/`, `sampler/`.

**Python package (`tensorrt_edgellm/`)** is the checkpoint-based export frontend. It implements LLM architectures from scratch using ONNX builtin + custom ops (the only format EdgeLLM's compiler accepts). Instead of tracing HuggingFace FX graphs (which are unstable), it reads the stable HF checkpoint weights directly.
- `model.py` — `AutoModel.from_pretrained()` factory with registry-based dispatch
- `config.py` — `ModelConfig`/`QuantConfig` for parsing HF `config.json`
- `checkpoint/loader.py` — Safetensors weight loading; `repacking.py` — weight repacking
- `onnx/export.py` — Export via `torch.onnx.export(dynamo=True)`; `onnx_custom_schemas.py` — custom op definitions; `dynamo_translations.py` — custom translation rules
- `models/` — Per-architecture implementations: `default/` (standard decoder + Mamba hybrid), `nemotron_h/` (hybrid Mamba2), `qwen3_moe/` (sparse MoE)
- `models/ops.py` — Shared custom operations; `models/linear.py` — Shared linear layer implementations
- `quantization/` — ModelOpt-based checkpoint quantization implementation
- `scripts/` — All user-facing Python command entry points
- Supported quant formats: `fp16`, `fp8`, `nvfp4`, `int4_awq`, `int4_awq_modelopt`, `int4_gptq`, `int8_sq`, `mixed_precision`

### CLI Entry Points (from `pyproject.toml`)

| Command | Script |
|---------|--------|
| `tensorrt-edgellm-quantize` | `tensorrt_edgellm.scripts.quantize:main` |
| `tensorrt-edgellm-export` | `tensorrt_edgellm.scripts.export:main` |
| `tensorrt-edgellm-insert-lora` | `tensorrt_edgellm.scripts.insert_lora:main` |
| `tensorrt-edgellm-process-lora` | `tensorrt_edgellm.scripts.process_lora_weights:main` |
| `tensorrt-edgellm-merge-lora` | `tensorrt_edgellm.scripts.merge_lora:main` |
| `tensorrt-edgellm-reduce-vocab` | `tensorrt_edgellm.scripts.reduce_vocab:main` |
| `tensorrt-edgellm-preprocess-audio` | `tensorrt_edgellm.scripts.preprocess_audio:main` |

## Key Files

| File | Role |
|------|------|
| `cpp/runtime/llmInferenceRuntime.{h,cpp}` | Unified runtime entry point — vanilla + speculative decoding (`handleRequest()`) |
| `cpp/runtime/llmEngineRunner.{h,cpp}` | Core TRT execution engine |
| `cpp/common/tensor.{h,cpp}` | RAII GPU/CPU tensor abstraction |
| `cpp/sampler/sampling.{cu,h}` | GPU token sampling |
| `tensorrt_edgellm/__init__.py` | Python API entry points |
| `tensorrt_edgellm/model.py` | Checkpoint model factory |
| `tensorrt_edgellm/models/default/modeling_default.py` | Base LLM model (Llama, Qwen, etc.) |
| `tensorrt_edgellm/quantization/quantize.py` | Quantization orchestration |
| `tensorrt_edgellm/scripts/export.py` | ONNX export orchestration |
| `tests/conftest.py` | Pytest configuration, YAML-driven test selection |

## Anti-Patterns / Gotchas

- **Pre-commit modifies files in-place** — if hooks fail, files are already modified. Re-stage (`git add`) and commit again.
- **`east-const` style enforced** — `.clang-format` uses `QualifierAlignment: Right`, so write `int const x` not `const int x`.
- **Integration tests need GPUs + models** — always set `LLM_SDK_DIR`, `ONNX_DIR`, `ENGINE_DIR`, and `LLM_MODELS_DIR`. C++ unit tests don't need models.
- **CUDA code coverage limitation** — `*.cu` files are excluded from gcov coverage; only `.cpp` files are instrumented.
- **FMHA kernels are SM-specific** — built per SM arch. When adding new SM support, update `cpp/CMakeLists.txt` FMHA build lists and optionally `cmake/CuteDslFMHA.cmake` for Blackwell+.
- **Plugin shared library** — `NvInfer_edgellm_plugin` is shared (not static) because TRT loads plugins dynamically.
- **One concern per PR** — avoid scope creep. If a PR touches unrelated areas, split it.
- **HF checkpoint consistency** — Python model classes in `models/` must stay compatible with HuggingFace checkpoint tensor names when adding new models.
- **Pinned dependencies** — `transformers`, `nvidia-modelopt`, `onnx`, and `torch` versions are pinned in `pyproject.toml`. Changing them can break export/quantization. Check compatibility before bumping.

## Development Workflow

1. Clone and init submodules: `git clone --recurse-submodules <repo-url>`
2. Install pre-commit: `pip install pre-commit && pre-commit install`
3. Make changes following `CODING_GUIDELINES.md`
4. Build and test locally (see Common Commands)
5. Commit with sign-off: `git commit -s`
6. If pre-commit modifies files, re-stage and commit again

## Branching Policy and PRs

- Branches should be pushed to the developer's fork (usually `origin`)
- PRs target `main` unless fixing a release branch bug (e.g., `release/0.4.0`)
- PR title: Conventional Commits — `feat:`, `fix:`, `chore:`, `BREAKING CHANGE:`, or `None:`
  - NVIDIA developers: include JIRA number or NVBUG ID when applicable
- See `CONTRIBUTING.md` for full policies

## CI / Testing

CI tests are YAML-driven and parametrized by `--priority`.

| Layer | Location | Notes |
|-------|----------|-------|
| Pre-commit | `.pre-commit-config.yaml` | isort, yapf, clang-format v20, cmake-format, codespell, autoflake, insert-license, ruff |
| C++ unit tests | `unittests/` | GTest, 30 files. Build with `-DBUILD_UNIT_TESTS=ON`. |
| Python unit tests | `tests/python-unittests/` | Attention plugin/native/utils tests |
| Integration tests | `tests/defs/` | Export, LLM/VLM pipeline, package sanity |
| Test lists (YAML) | `tests/test_lists/` | Per-GPU/platform test parametrization |
| Code coverage | `scripts/run_coverage.sh` | gcov + SonarQube report |

For model-facing changes, the expected test shape is `tensorrt-edgellm-export`
followed by engine build and runtime inference. Use export-only priorities only
as producer/smoke coverage for downstream pipeline jobs.

### CI Stages

| Stage | Purpose |
|-------|---------|
| `setup` | Pre-commit validation, cache cleanup |
| `l0_test` | MR-triggered — export, pipeline, unit tests per GPU/device |
| `l1_test` | Manual (`L1=true`) — extended coverage |
| `build-sonar` | SonarQube static analysis |

### Test Priorities

| Priority | GPU/Device | Type |
|----------|-----------|------|
| `l0_checkpoint_export_ampere` | A30 (x86) | `tensorrt_edgellm` Ampere export |
| `l0_checkpoint_export` | B100/Thor (x86) | `tensorrt_edgellm` FP8/NVFP4 export |
| `l0_pipeline_a30` | A30 | Full pipeline |
| `l0_pipeline_orin` | Jetson Orin (remote) | On-device pipeline |
| `l0_pipeline_rtx5080` | RTX 5080 | FP8 small model pipeline |
| `l0_pipeline_jedha` | Jedha (SM110) | Long accuracy + EAGLE + larger models |
| `l0_pipeline_thor_1` | Drive Thor 1 (remote) | On-device FP8/NVFP4 pipeline |
| `l0_pipeline_thor_2` | Drive Thor 2 (remote) | On-device FP8+KV pipeline |
| `l0_python_ut` | Any | Python unit tests |

Runtime test parameter format: `ModelName-Precision-[LmHeadPrecision-]MaxSeqLen-MaxBatchSize-MaxInputLen-[Additional-Params]`.
Export tests omit sequence length, batch, and input length parameters. See `tests/README.md`.

## Key Documentation

| Topic | Path |
|-------|------|
| Installation & quick start | `docs/source/developer_guide/getting-started/` |
| Software design (runtime, export, builder) | `docs/source/developer_guide/software-design/` |
| Features (LoRA, FP8KV, system prompt cache, vocab reduction) | `docs/source/developer_guide/features/` |
| Customization & plugins | `docs/source/developer_guide/customization/` |
| C++ / Python API reference | `docs/source/cpp_api/`, `docs/source/python_api.rst` |
| Design principles | `design/general/design_principles.adoc` |
| Accuracy benchmarks | `examples/accuracy/README.md` |
| Coding guidelines | `CODING_GUIDELINES.md` |
| Contributing & PR policies | `CONTRIBUTING.md` |
| Changelog | `CHANGELOG.md` |
