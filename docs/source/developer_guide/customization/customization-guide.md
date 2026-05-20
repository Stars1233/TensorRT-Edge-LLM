# Customization Guide

New model and feature enablement should target the checkpoint-based workflow:
`experimental/quantization` for checkpoint quantization and `experimental/llm_loader`
for ONNX export. This guide describes the main extension points used by that
workflow.

For user commands, see [Quantization](../../user_guide/features/quantization.md)
and [Checkpoint-Based Model Loader Design](../software-design/llm-loader.md).

The deprecated `tensorrt_edgellm/` export package remains available in 0.7.0 and 0.7.1 for
compatibility, but new model and feature work should target the experimental
workflow. The deprecated export package is scheduled for removal in 0.8.0 after
full feature parity is available in `experimental/quantization` and
`experimental/llm_loader`.

## Export Customization Points

| Area | Files | What to update |
|------|-------|----------------|
| Text model | `experimental/llm_loader/models/<family>/` | Add a native `torch.nn.Module` implementation when the default decoder is not enough. Keep parameter names aligned with the HuggingFace checkpoint. |
| Registration | `experimental/llm_loader/__init__.py` | Register custom text classes with `register_model("<model_type>", ModelClass)`. |
| Checkpoint parsing | `experimental/llm_loader/config.py`, `experimental/llm_loader/checkpoint/` | Add model-specific config fields, tensor remapping, fused-weight splitting, or quantization metadata handling. |
| Weight loading | `experimental/llm_loader/checkpoint/loader.py`, `experimental/llm_loader/checkpoint/repacking.py` | Load safetensors, remap checkpoint keys, and repack quantized tensors into the layout consumed by the exported graph. |
| Encoder export | `experimental/llm_loader/onnx/export_encoder.py` | Register visual, audio, or action encoder builders and config extraction rules. |
| Component orchestration | `experimental/llm_loader/export_all_cli.py` | Add model-type classification for LLM-only, VLM, audio, TTS, Omni, VLA, EAGLE, or MTP checkpoints. |
| Custom ops | `experimental/llm_loader/models/ops.py`, `experimental/llm_loader/onnx/dynamo_translations.py`, `experimental/llm_loader/onnx/onnx_custom_schemas.py` | Add torch stubs, ONNX translations, and schemas for TensorRT Edge-LLM custom ops. |
| Runtime/plugins | `cpp/plugins/`, `cpp/runtime/`, `cpp/multimodal/`, `cpp/action/` | Add runtime support only when the exported graph or model I/O needs new behavior. |

Update [Supported Models](../../user_guide/getting_started/supported-models.md)
and add focused export tests when adding a new family.

## Quantization Customization Points

The standalone quantization package writes unified HuggingFace-style
checkpoints. It does not export ONNX or build TensorRT engines.

| Area | Files | What to update |
|------|-------|----------------|
| Quantization recipes | `experimental/quantization/quantization_configs.py` | Add ModelOpt recipe presets, exclusions, or component-specific overrides. |
| Model loading and calibration | `experimental/quantization/quantize.py` | Add model loading fallbacks, calibration data handling, or pre-save checkpoint fixups. |
| EAGLE draft quantization | `experimental/quantization/models/eagle3_draft.py` | Update draft-model calibration and checkpoint writing. |
| CLI surface | `experimental/quantization/cli.py` | Expose a new supported option after the implementation and tests exist. |

Supported methods are documented in
[Quantization](../../user_guide/features/quantization.md). GPTQ checkpoints are
loaded as pre-quantized checkpoints; this package does not create GPTQ models.

## Adding A Text Model

1. Check whether the default `CausalLM` implementation can load the checkpoint.
2. If the architecture needs custom behavior, add a model implementation under
   `experimental/llm_loader/models/<family>/`.
3. Register the `model_type` in `experimental/llm_loader/__init__.py`.
4. Add any required config promotion, tensor-key remapping, or quantized-weight
   handling in `experimental/llm_loader/config.py` and `checkpoint/`.
5. Export with `python -m llm_loader.export_all_cli <checkpoint> <output_dir>`
   and verify `llm_build` plus `llm_inference`.

## Adding A Multimodal Or Action Component

Use `experimental/llm_loader/export_all_cli.py` as the component dispatcher.
Each exported component should have a stable subdirectory and a `config.json`
that the C++ builder can consume.

| Component | Typical output | Builder |
|-----------|----------------|---------|
| LLM thinker/base | `llm/` | `examples/llm/llm_build` |
| Visual encoder | `visual/` | `examples/multimodal/visual_build` |
| Audio encoder | `audio/` | `examples/multimodal/audio_build` |
| TTS code predictor | `code_predictor/` | `examples/llm/llm_build` |
| Omni Code2Wav | `code2wav/` or model-specific audio layout | `examples/multimodal/audio_build` |
| Alpamayo action expert | `action/` | `examples/multimodal/action_build` |
| MTP draft | `mtp_draft/` | `examples/llm/llm_build --specDraft` |

Keep preprocessing and runtime sidecars explicit. For example, Alpamayo adds
trajectory tokens during runtime artifact writing and requires the action
engine's `max_kv_cache_capacity` to match the LLM engine build.

## Custom Operators

Custom operators are declared as `torch.library.custom_op` stubs in
`experimental/llm_loader/models/ops.py`, translated in
`experimental/llm_loader/onnx/dynamo_translations.py`, and registered as ONNX
schemas in `experimental/llm_loader/onnx/onnx_custom_schemas.py`.

Use this path when a PyTorch expression must lower to a TensorRT Edge-LLM
plugin, specialized runtime op, or fixed ONNX node pattern. Runtime support must
exist before documenting a new custom op as supported.

## Runtime Customization

The C++ runtime still owns engine build, tokenization, sampling, multimodal
preprocessing, LoRA adapter loading, EAGLE/MTP execution, and action inference.
Common extension points include:

| Area | Files |
|------|-------|
| LLM engine profiles | `cpp/builder/llmBuilder.cpp`, `examples/llm/llm_build.cpp` |
| Visual/audio/action builders | `cpp/builder/`, `examples/multimodal/` |
| Multimodal runners | `cpp/multimodal/` |
| Action runners | `cpp/action/` |
| Tokenization | `cpp/tokenizer/` |
| Sampling | `cpp/sampler/` |
| Runtime orchestration | `cpp/runtime/` |

Prefer adding runtime behavior only after the exported ONNX contract and sidecar
files are stable.

### Minimal Runtime Request

Use the 0.7 runtime request structure when embedding TensorRT Edge-LLM in a C++
application. Requests are batched as `request.requests`, each request contains
chat-template messages, and generated text is returned in `response.outputTexts`.
For text-only engines, pass an empty `multimodalEngineDir`; set it to the
visual or audio engine directory for multimodal workflows.

```cpp
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

int main()
{
    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    std::string engineDir = "/path/to/engine";
    std::string multimodalEngineDir = "";  // empty string = LLM-only, no multimodal engines
    std::unordered_map<std::string, std::string> loraWeightsMap{};

    trt_edgellm::rt::LLMInferenceRuntime runtime(
        engineDir, multimodalEngineDir, loraWeightsMap, stream);

    trt_edgellm::rt::LLMGenerationRequest request;
    request.requests.resize(1);
    request.temperature = 1.0F;
    request.topK = 50;
    request.topP = 0.8F;
    request.maxGenerateLength = 100;

    trt_edgellm::rt::Message userMsg;
    userMsg.role = "user";
    userMsg.contents.push_back(
        trt_edgellm::rt::Message::MessageContent{
            "text", "What is the capital of France?"});
    request.requests[0].messages.push_back(std::move(userMsg));

    trt_edgellm::rt::LLMGenerationResponse response;
    if (runtime.handleRequest(request, response, stream)
        && !response.outputTexts.empty())
    {
        std::cout << "Generated: " << response.outputTexts[0] << std::endl;
    }

    cudaStreamDestroy(stream);
    return 0;
}
```
