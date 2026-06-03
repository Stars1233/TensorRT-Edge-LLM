# Experimental Docker Build

This Dockerfile builds TensorRT Edge-LLM for Jetson Thor from the current
checkout using `nvcr.io/nvidia/pytorch:26.04-py3`.

Use the wrapper script from the repository root. It checks for the default
CuTe DSL prebuilt tarball and, if it is missing, compiles the CuTe DSL artifact
before invoking `docker build`.

```bash
experimental/docker/build_container.sh
```

The build extracts the CuTe DSL tarball from `kernelSrcs/cuteDSLPrebuilt/` and
does not generate CuTe DSL artifacts during Docker build. To customize the
image name, set `EXPERIMENTAL_DOCKER_IMAGE` before running the wrapper.

Run the OpenAI-compatible experimental server:

```bash
docker run --runtime nvidia --rm -it --network host \
  -v /data:/data \
  tensorrt-edge-llm:experimental \
  python3 -m experimental.server --model Qwen/Qwen3-1.7B
```

Model downloads and generated artifacts are cached under `/data`.
