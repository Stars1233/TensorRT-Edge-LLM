# Installation


TensorRT Edge-LLM has two separate components that need to be installed on different systems:

1. **Experimental Quantization and `llm_loader`** (runs on x86 host with GPU)
2. **C++ Runtime** (Jetson Thor, NVIDIA DRIVE / DriveOS, or optional x86 developer build)

---

## Part 1: Experimental Quantization and `llm_loader` (x86 Host with GPU)

The experimental quantization package and `llm_loader` convert and quantize models. This must run on an x86 Linux system with an NVIDIA GPU.

### System Requirements

- **Platform**: x86-64 Linux system
- **Recommended OS**: Ubuntu 22.04, 24.04
- **GPU**: NVIDIA GPU with Compute Capability 8.0+ (Ampere or newer)
- **CUDA**: 12.x or 13.x
- **Python**: 3.10+

#### Memory Requirements

**GPU Memory (VRAM):**
- General rule: ~2-3x model size for most operations, ~5-6x model size for FP8 ONNX export
- Small models (0.6B-3B): 8-16GB
- Large models (7B-8B): 20-48GB
- Very large models (13B+): 48GB+

**CPU Memory (RAM):**
- General rule: ~2-3x model size for most operations, **~18-20x** model size for FP8 ONNX export
- Small models (0.6B-3B): 8-16GB (48GB+ for FP8 ONNX export)
- Large models (7B-8B): 20-48GB (128GB+ for FP8 ONNX export)
- Very large models (13B+): 48GB+

> **Note:** FP8 ONNX export currently requires significantly higher CPU (up to 20x model size) and GPU (up to 6x model size) memory due to internal processing. This is a known issue and is being actively optimized.

**Verify Your Prerequisites:**

```bash
# Check CUDA installation
nvcc --version
# Should show CUDA 12.x or 13.x

# Check GPU and available memory
nvidia-smi
# Look for GPU memory (e.g., "24576MiB" for 24GB)

# Check Python version
python3 --version
# Should show Python 3.10 or higher
```

**If CUDA is not installed:**

Download and install CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads). Choose version 12.x or 13.x for your system.

After installation, verify with `nvcc --version` and `nvidia-smi`.

### Installing

For a containerized environment for clean installation, it is recommended to use the NVIDIA PyTorch Docker image:

```bash
# Pull the recommended Docker image
docker pull nvcr.io/nvidia/pytorch:25.12-py3

# Run the container with GPU support
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash
```

**1. Clone Repository**

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**2. Install Python Dependencies**

If you are not using container, it is recommended to use a virtual environment:
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
```

Install the base and checkpoint-based loader dependencies. Add the standalone
quantization dependencies only when you run `experimental.quantization`. Do not
install the repository as a pip package for this workflow; the examples run
directly from the checkout through `PYTHONPATH`.

```bash
# Required for llm_loader export, high-level API, and server workflows
pip3 install -r requirements.txt
pip3 install -r experimental/llm_loader/requirements.txt

# Required only when running experimental.quantization
pip3 install -r experimental/quantization/requirements.txt
```

This installs all required Python dependencies including:
- PyTorch
- Transformers
- NVIDIA Model Optimizer
- ONNX
- ONNX Script and ONNX GraphSurgeon
- And all other required dependencies

> **Note:** For specific version requirements, please refer to `requirements.txt`, `experimental/llm_loader/requirements.txt`, `experimental/quantization/requirements.txt`, and `pyproject.toml`.

**3. Configure and Verify the Experimental Export Workflow**

Use the virtual environment created in Step 2 for this checkout. Do not mix
packages from older release branches into the same environment.

The recommended export path is `experimental.quantization` -> `llm_loader`.
Use the quantization package only when you need to create a unified quantized
checkpoint from an FP16/BF16 source checkpoint before export. Pre-quantized
HuggingFace checkpoints can be exported directly with `llm_loader`.

```bash
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH

# Verify the recommended quantization, export, LoRA, and vocabulary tools
python -m experimental.quantization --help
python -m llm_loader.export_all_cli --help
python -m llm_loader.lora.merge_lora_cli --help
python -m llm_loader.vocab_reduction --help
```

The deprecated `tensorrt_edgellm/` export package remains available in 0.7.1 for
compatibility and for components that are not yet fully covered by
`llm_loader`. It will be removed in 0.8.0 after the
`experimental/quantization` -> `experimental/llm_loader` workflow reaches full
feature parity for all models and features.

**4. Configure HuggingFace Access (Optional)**

Some models on HuggingFace require you to accept terms before downloading. This is **not required** for the quick start example (Qwen3-0.6B).

**Models that require HuggingFace login:**
- Llama family (Llama 3.x)
- Phi-4-Multimodal
- Alpamayo-R1-10B
- Other models marked as "gated" on HuggingFace

**To configure access:**

```bash
# Install HuggingFace CLI and login
hf auth login
# Enter your HuggingFace access token when prompted
```

> **How to get a token:** Visit [HuggingFace Settings - Tokens](https://huggingface.co/settings/tokens), create a new token (read access is sufficient), and copy it.

> **For the quick start guide:** You can skip this step and proceed to verification.

**You're done with export pipeline setup!** You can now quantize and export models with the experimental workflow. The ONNX files will be transferred to the Edge device for runtime deployment.

---

## Part 2: C++ Runtime (Edge Device)

The C++ runtime builds and executes models on the target. **Jetson Thor:** follow the steps below on the device and use `EMBEDDED_TARGET=jetson-thor`. **NVIDIA DRIVE / DriveOS:** run the same flow inside the DriveOS SDK Docker image with **`EMBEDDED_TARGET=auto-thor`**, then copy **`build/`** to the DRIVE system. **x86:** optional local build using the **Alternative** `cmake` block (no toolchain).

### System Requirements

**Target Platform:**
- NVIDIA Jetson Thor
- JetPack 7.1
- CUDA 13.x (included in JetPack)
- TensorRT 10.x+ (included in JetPack)
- Disk Space: ~20-50GB for ONNX files and TensorRT engines

### Build Instructions

**1. Install System Dependencies (on Edge device)**

```bash
sudo apt update
sudo apt install -y \
    cmake \
    build-essential \
    git
```

**2. Verify CUDA and TensorRT Installation**

After JetPack is installed, TensorRT should be installed in /usr

```bash
# Check CUDA version
nvcc --version  # Should show CUDA 13.x

# Check TensorRT version
dpkg -l | grep tensorrt  # Should show TensorRT 10.x+
```

**3. Clone Repository (on Edge device)**

```bash
# Clone to home directory (used in all examples)
cd ~
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive
```

**4. Configure Build**

On your Jetson Thor device, configure the build with the following command:

```bash
mkdir build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
```

**NVIDIA DRIVE / DriveOS:** The `cmake` line is the same except **`EMBEDDED_TARGET=auto-thor`**.

**Alternative: Building on x86 GPU Systems (Optional for Developers)**

If you want to build and test on an x86 workstation with NVIDIA GPU (for development purposes before deploying to Edge devices), you can use this configuration instead:

```bash
mkdir build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION>
```

> **Note:** Replace `/usr/local/TensorRT-10.x.x` with your actual TensorRT installation path. Use `dpkg -l | grep tensorrt` to find it, or download from [NVIDIA TensorRT downloads](https://developer.nvidia.com/tensorrt). Replace `<YOUR_CUDA_VERSION>` with your actual CUDA version (e.g., `13.0`). Use `nvcc --version` to check your CUDA version.

**CMake Options:**

| Option | Description | Default |
|:-------|:------------|:--------|
| `TRT_PACKAGE_DIR` | Path to TensorRT installation. Auto-detected; manual hint to disambiguate multiple versions. | N/A |
| `CMAKE_TOOLCHAIN_FILE` | **Required for Edge devices**: Use `cmake/aarch64_linux_toolchain.cmake` for Edge device builds. **Not needed for GPU builds** | N/A |
| `EMBEDDED_TARGET` | **Required for Edge devices**: `jetson-thor` (Jetson) or `auto-thor` (DRIVE / DriveOS). **Not needed for GPU builds** | N/A |
| `CUDA_CTK_VERSION` | CUDA Toolkit version (such as 13.0). Important for matching target platform. | 13.0 |
| `BUILD_UNIT_TESTS` | Build unit tests | OFF |
| `ENABLE_COVERAGE` | Enable gcov code coverage instrumentation (see [Code Coverage](../../developer_guide/testing/code-coverage.md)) | OFF |
| `ENABLE_CUTE_DSL` | Enable prebuilt CuTe DSL kernels: `OFF` (default), `ALL`, or a group list such as `gdn`, `fmha`, `gemm`, or `ssd` | OFF |
| `CUTE_DSL_ARTIFACT_TAG` | Optional artifact tag under `cpp/kernels/cuteDSLArtifact/<arch>/`, for example `sm_110` or `sm_121`. Required when multiple local artifact tags exist for the same CPU architecture. | auto |

**Building with CuTe DSL Kernels (Optional)**

CuTe DSL binaries are prebuilt and shipped with the repository. Add `-DENABLE_CUTE_DSL=ALL` (or a group selection such as `gdn`, `fmha`, `gemm`, or `ssd`) to the CMake configure command when a model or kernel path needs them. Qwen3.5 GDN requires `-DENABLE_CUTE_DSL=gdn` or `-DENABLE_CUTE_DSL=ALL`.

If you have multiple local artifact tags for the same CPU architecture, also
pass `-DCUTE_DSL_ARTIFACT_TAG=<tag>`.

> **For supported model families, precisions, and hardware notes**, see [Supported Models](supported-models.md).

**5. Build Project**

```bash
make -j$(nproc)
```

Build time: ~1-2 minutes depending on hardware.

**6. Verify Build**

```bash
# Test C++ examples
./examples/llm/llm_build --help
./examples/llm/llm_inference --help
```

**You're done with C++ runtime setup!** You can now build engines and run inference on the Edge device.

---

## Next Steps

After installation, proceed to the [Quick Start Guide](quick-start-guide.md) for a complete end-to-end workflow, or see the [Examples](../examples/) for detailed pipeline stages and advanced use cases.

---

## Troubleshooting

### Common Installation Issues

**Issue: Python module import errors**

Solution: Ensure the virtual environment is activated and `PYTHONPATH` points to
both the repository root and `experimental/`:
```bash
source venv/bin/activate
export EDGE_LLM_PATH=/path/to/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_PATH:$EDGE_LLM_PATH/experimental:$PYTHONPATH
python -m llm_loader.export_all_cli --help
```

**Issue: `nvcc: command not found`**

Solution: Ensure JetPack 7.1 is properly installed with CUDA support:
```bash
# Verify CUDA installation
nvcc --version
# Should show CUDA 13.x
```

**Issue: `TensorRT not found` during CMake**

Solution: Specify TensorRT package directory. This directory should contain `lib` and `include` directories, and we are looking for the `nvinfer` library and header:
```bash
cmake .. \
    -DTRT_PACKAGE_DIR=/usr/local/TensorRT-10.x.x \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor
```

**Issue: Thread issue during C++ build**

Solution: Reduce parallel jobs or even use sequential build:
```bash
make -j  # Instead of make -j$(nproc)
```

### Getting Help

- **Documentation**: Check the `docs/source/developer_guide` directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Community**: Join the NVIDIA Developer Forums

## Uninstalling

**Experimental Quantization and `llm_loader` (x86 Host):**
- Deactivate and remove virtual environment: `deactivate && rm -rf venv`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`

**C++ Runtime (Edge Device):**
- Remove build directory: `rm -rf build`
- Remove repository (optional): `rm -rf TensorRT-Edge-LLM`
