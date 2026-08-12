# Nemotron-3.5-ASR (experimental runtime)

Nemotron-3.5-ASR model definitions and ONNX export live in the main Python package under `tensorrt_edgellm.models.nemotron3_5_asr` (exported through the unified `tensorrt-edgellm-export`). This directory keeps the separate experimental runtime: the C++ RNN-T transcription runtime and its inference CLI.

The model is an offline (batch 1) RNN-T transducer — a FastConformer encoder plus an LSTM prediction network and joint network — with **no LLM backbone**. Decoding is a greedy transducer loop over two engines (a dynamic-length encoder and a static-shape single-step engine), not autoregressive LLM generation, which is why the runtime does not reuse the LLM inference path.

The supported input is the HF checkpoint `nvidia/nemotron-3.5-asr-streaming-0.6b`.

## Layout

```text
nemotron3_5_asr/
  cpp/        # experimental C++ RNN-T transcription runtime
  examples/   # nemotron_asr_inference CLI
```

Unlike Cosmos3, the engine build is **not** a dedicated builder here: the encoder and the RNN-T step engines are produced by the shared `audio_build` tool (auto-detected from each `config.json`), so only the model-specific inference runtime lives under `experimental_models/`.

## Quickstart

The `nemotron_asr_inference` binary is produced by the standard Edge-LLM
experimental-model build (configure with `-DBUILD_EXPERIMENTAL_MODELS=ON`); this
Quickstart assumes it, the shared `audio_build`, and the plugin library are
already built. `$BUILD_DIR` below points at that build tree.

```bash
# 1. Export ONNX from the HF checkpoint (the unified exporter detects
#    Nemotron-3.5-ASR automatically). Produces $ONNX_DIR/audio (encoder ONNX +
#    config.json + tokenizer) and $ONNX_DIR/rnnt_decoder (step ONNX + config).
tensorrt-edgellm-export "nvidia/nemotron-3.5-asr-streaming-0.6b" "$ONNX_DIR"

# 2. Build both engines with the shared audio_build (build type is auto-detected
#    from config.json: encoder_config -> audio encoder, rnnt_decoder_config ->
#    static RNN-T step). Engines land in $ENGINE_DIR/audio and
#    $ENGINE_DIR/rnnt_decoder.
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
"$BUILD_DIR/examples/multimodal/audio_build" \
    --onnxDir "$ONNX_DIR/audio"         --engineDir "$ENGINE_DIR" --maxTimeSteps 8192
"$BUILD_DIR/examples/multimodal/audio_build" \
    --onnxDir "$ONNX_DIR/rnnt_decoder"  --engineDir "$ENGINE_DIR"

# 3. Assemble one runtime dir: both engines + config.json + tokenizer.
mkdir -p "$RUN_DIR"
cp "$ENGINE_DIR/audio/audio_encoder.engine" "$ENGINE_DIR/rnnt_decoder/rnnt_step.engine" "$RUN_DIR/"
cp "$ONNX_DIR/audio/config.json" "$ONNX_DIR/audio/tokenizer.json" "$ONNX_DIR/audio/tokenizer_config.json" "$RUN_DIR/"

# 4. Transcribe. --promptId selects the language prompt (default = config
#    default_prompt_id = automatic language detection; the model emits an
#    <xx-XX> language tag).
"$BUILD_DIR/experimental_models/nemotron3_5_asr/examples/nemotron_asr_inference" \
    --engineDir "$RUN_DIR" --audioFile "$AUDIO"        # e.g. clip.wav / .mp3 / .flac
```

See `docs/source/developer_guide/models/nemotron3_5_asr.md` for the architecture and runtime-design details.
