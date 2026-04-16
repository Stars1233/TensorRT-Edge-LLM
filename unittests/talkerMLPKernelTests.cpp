/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/checkMacros.h"
#include "common/tensor.h"
#include "kernels/talkerMLPKernels/talkerMLPKernels.h"
#include "testUtils.h"

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <tuple>
#include <vector>

using namespace trt_edgellm;

namespace
{

float siluF32(float x)
{
    return x / (1.0f + std::exp(-x));
}

// CPU reference: output = FC2(SiLU(FC1(input) + bias1)) + bias2
// Weight layout: row-major [outDim, inDim] (same as PyTorch Linear.weight)
void referenceTalkerMLP(std::vector<half> const& input, std::vector<half> const& fc1Weight,
    std::vector<half> const& fc1Bias, std::vector<half> const& fc2Weight, std::vector<half> const& fc2Bias,
    std::vector<half>& output, int64_t numTokens, int64_t inputDim, int64_t hiddenDim, int64_t outputDim)
{
    std::vector<float> workspace(numTokens * hiddenDim, 0.0f);
    for (int64_t n = 0; n < numTokens; ++n)
    {
        for (int64_t h = 0; h < hiddenDim; ++h)
        {
            float acc = 0.0f;
            for (int64_t k = 0; k < inputDim; ++k)
            {
                acc += __half2float(input[n * inputDim + k]) * __half2float(fc1Weight[h * inputDim + k]);
            }
            acc += __half2float(fc1Bias[h]);
            workspace[n * hiddenDim + h] = siluF32(acc);
        }
    }

    for (int64_t n = 0; n < numTokens; ++n)
    {
        for (int64_t o = 0; o < outputDim; ++o)
        {
            float acc = 0.0f;
            for (int64_t h = 0; h < hiddenDim; ++h)
            {
                acc += workspace[n * hiddenDim + h] * __half2float(fc2Weight[o * hiddenDim + h]);
            }
            acc += __half2float(fc2Bias[o]);
            output[n * outputDim + o] = __float2half(acc);
        }
    }
}

} // namespace

// ============================================================================
// Fixture for tests that require cuBLAS (invokeTalkerMLP)
// ============================================================================
class TalkerMLPTest : public ::testing::Test
{
protected:
    cudaStream_t stream{};
    void* cublasLib{nullptr};
    void* cublasHandle{nullptr};

    void SetUp() override
    {
        cudaSetDevice(0);
        CUDA_CHECK(cudaStreamCreate(&stream));

        cublasLib = dlopen("libcublas.so", RTLD_LAZY);
        if (!cublasLib)
        {
            GTEST_SKIP() << "cuBLAS not available";
        }
        auto createFn = reinterpret_cast<int (*)(void**)>(dlsym(cublasLib, "cublasCreate_v2"));
        if (!createFn || createFn(&cublasHandle) != 0)
        {
            dlclose(cublasLib);
            cublasLib = nullptr;
            GTEST_SKIP() << "Failed to create cuBLAS handle";
        }
    }

    void TearDown() override
    {
        if (cublasHandle && cublasLib)
        {
            auto destroyFn = reinterpret_cast<int (*)(void*)>(dlsym(cublasLib, "cublasDestroy_v2"));
            if (destroyFn)
            {
                destroyFn(cublasHandle);
            }
        }
        if (cublasLib)
        {
            dlclose(cublasLib);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// ============================================================================
// Fixture for tests that only need CUDA (no cuBLAS dependency)
// ============================================================================
class TalkerKernelTest : public ::testing::Test
{
protected:
    cudaStream_t stream{};

    void SetUp() override
    {
        cudaSetDevice(0);
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void TearDown() override
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// ===== invokeTalkerMLP =====

TEST_F(TalkerMLPTest, MLPAccuracy)
{
    // (numTokens, inputDim, hiddenDim, outputDim)
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> testCases = {
        {1, 64, 64, 32},
        {2, 64, 64, 32},
        {4, 2048, 2048, 1024},
    };

    for (auto const& [numTokens, inputDim, hiddenDim, outputDim] : testCases)
    {
        SCOPED_TRACE("numTokens=" + std::to_string(numTokens) + ", inputDim=" + std::to_string(inputDim)
            + ", hiddenDim=" + std::to_string(hiddenDim) + ", outputDim=" + std::to_string(outputDim));

        // Scale init range down for large dimensions to avoid FP16 overflow in accumulation
        float const initScale = (inputDim > 256) ? 0.1f : 0.5f;

        std::vector<half> hostInput(numTokens * inputDim);
        std::vector<half> hostFc1W(hiddenDim * inputDim);
        std::vector<half> hostFc1B(hiddenDim);
        std::vector<half> hostFc2W(outputDim * hiddenDim);
        std::vector<half> hostFc2B(outputDim);

        uniformFloatInitialization(hostInput, -initScale * 2, initScale * 2);
        uniformFloatInitialization(hostFc1W, -initScale, initScale);
        uniformFloatInitialization(hostFc1B, -0.1f, 0.1f);
        uniformFloatInitialization(hostFc2W, -initScale, initScale);
        uniformFloatInitialization(hostFc2B, -0.1f, 0.1f);

        std::vector<half> refOutput(numTokens * outputDim);
        referenceTalkerMLP(
            hostInput, hostFc1W, hostFc1B, hostFc2W, hostFc2B, refOutput, numTokens, inputDim, hiddenDim, outputDim);

        rt::Coords inputShape{numTokens, inputDim};
        rt::Coords fc1WShape{hiddenDim, inputDim};
        rt::Coords fc1BShape{hiddenDim};
        rt::Coords fc2WShape{outputDim, hiddenDim};
        rt::Coords fc2BShape{outputDim};
        rt::Coords outputShape{numTokens, outputDim};
        rt::Coords workspaceShape{numTokens, hiddenDim};

        rt::Tensor gpuInput(inputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc1W(fc1WShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc1B(fc1BShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc2W(fc2WShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuFc2B(fc2BShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuOutput(outputShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
        rt::Tensor gpuWorkspace(workspaceShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

        copyHostToDevice(gpuInput, hostInput);
        copyHostToDevice(gpuFc1W, hostFc1W);
        copyHostToDevice(gpuFc1B, hostFc1B);
        copyHostToDevice(gpuFc2W, hostFc2W);
        copyHostToDevice(gpuFc2B, hostFc2B);

        kernel::invokeTalkerMLP(
            cublasHandle, gpuInput, gpuFc1W, gpuFc1B, gpuFc2W, gpuFc2B, gpuOutput, gpuWorkspace, stream);

        auto const gpuResult = copyDeviceToHost<half>(gpuOutput);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // For large-dim GEMM, allow a small fraction of outliers due to FP16 accumulation differences
        auto [rtol, atol] = getTolerance<half>();
        int32_t mismatches = 0;
        for (size_t i = 0; i < gpuResult.size(); ++i)
        {
            if (!isclose(gpuResult[i], refOutput[i], rtol, atol))
            {
                ++mismatches;
            }
        }
        EXPECT_LT(mismatches, std::max(1, static_cast<int32_t>(gpuResult.size() / 100)))
            << "Too many mismatches: " << mismatches << " / " << gpuResult.size();
    }
}

// ===== Gather / Scatter =====

TEST_F(TalkerKernelTest, GatherScatterRoundTrip)
{
    int64_t const srcTokens = 8;
    int64_t const hiddenDim = 64;
    int64_t const numIndices = 4;

    std::vector<half> hostSource(srcTokens * hiddenDim);
    uniformFloatInitialization(hostSource, -1.0f, 1.0f);
    std::vector<int32_t> hostIndices = {2, 5, 0, 7};

    rt::Coords srcShape{srcTokens, hiddenDim};
    rt::Coords idxShape{numIndices};
    rt::Coords gatherShape{numIndices, hiddenDim};
    rt::Coords scatterShape{srcTokens, hiddenDim};

    rt::Tensor gpuSource(srcShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuIndices(idxShape, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    rt::Tensor gpuGatherOut(gatherShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    rt::Tensor gpuScatterOut(scatterShape, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);

    copyHostToDevice(gpuSource, hostSource);
    copyHostToDevice(gpuIndices, hostIndices);
    CUDA_CHECK(cudaMemset(gpuScatterOut.rawPointer(), 0, srcTokens * hiddenDim * sizeof(half)));

    kernel::invokeGather(gpuSource, gpuIndices, gpuGatherOut, stream);
    kernel::invokeScatter(gpuGatherOut, gpuIndices, gpuScatterOut, stream);

    auto const scatterResult = copyDeviceToHost<half>(gpuScatterOut);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t idx = 0; idx < numIndices; ++idx)
    {
        int32_t const srcRow = hostIndices[idx];
        for (int64_t d = 0; d < hiddenDim; ++d)
        {
            EXPECT_TRUE(isclose(scatterResult[srcRow * hiddenDim + d], hostSource[srcRow * hiddenDim + d], 0.f, 0.f))
                << "Gather-scatter round trip mismatch at row " << srcRow << " dim " << d;
        }
    }
}

// ===== SumReduceOverSequence =====

// ===== TalkerLogitAdjust =====

// Helper: upload logits and seenTokens to GPU, run kernel, download result.
static std::vector<float> runTalkerLogitAdjust(std::vector<float> const& hostLogits, int32_t suppressStart,
    int32_t suppressEnd, int32_t codecEosId, std::vector<int32_t> const& hostSeenTokens, float repetitionPenalty,
    cudaStream_t stream)
{
    int32_t const vocabSize = static_cast<int32_t>(hostLogits.size());

    rt::Tensor gpuLogits({1, static_cast<int64_t>(vocabSize)}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT);
    CUDA_CHECK(
        cudaMemcpy(gpuLogits.rawPointer(), hostLogits.data(), vocabSize * sizeof(float), cudaMemcpyHostToDevice));

    int32_t const maxSeen = std::max(static_cast<int32_t>(hostSeenTokens.size()), 1);
    rt::Tensor gpuSeen({static_cast<int64_t>(maxSeen)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32);
    if (!hostSeenTokens.empty())
    {
        CUDA_CHECK(cudaMemcpy(gpuSeen.rawPointer(), hostSeenTokens.data(), hostSeenTokens.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice));
    }

    kernel::invokeTalkerLogitAdjust(gpuSeen, gpuLogits, suppressStart, suppressEnd, codecEosId,
        static_cast<int32_t>(hostSeenTokens.size()), repetitionPenalty, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return copyDeviceToHost<float>(gpuLogits);
}

// Suppression range [suppressStart, suppressEnd) is set to -inf, except codecEosId.
TEST_F(TalkerKernelTest, TalkerLogitAdjust_SuppressionWithEosExempt)
{
    int32_t const vocabSize = 256;
    int32_t const suppressStart = 50;
    int32_t const suppressEnd = 150;
    int32_t const codecEosId = 100;

    auto result = runTalkerLogitAdjust(
        std::vector<float>(vocabSize, 1.0f), suppressStart, suppressEnd, codecEosId, {}, 1.0f, stream);

    for (int32_t i = 0; i < vocabSize; ++i)
    {
        if (i >= suppressStart && i < suppressEnd && i != codecEosId)
        {
            EXPECT_TRUE(std::isinf(result[i]) && result[i] < 0) << "Token " << i << " should be -inf";
        }
        else
        {
            EXPECT_FLOAT_EQ(result[i], 1.0f) << "Token " << i << " should be unchanged";
        }
    }
}

// No exception token (-1): all tokens in suppress range become -inf.
TEST_F(TalkerKernelTest, TalkerLogitAdjust_SuppressionNoExempt)
{
    int32_t const vocabSize = 128;
    int32_t const suppressStart = 0;
    int32_t const suppressEnd = 64;

    auto result = runTalkerLogitAdjust(
        std::vector<float>(vocabSize, 2.0f), suppressStart, suppressEnd, /*codecEosId=*/-1, {}, 1.0f, stream);

    for (int32_t i = 0; i < suppressEnd; ++i)
    {
        EXPECT_TRUE(std::isinf(result[i]) && result[i] < 0) << "Token " << i << " should be -inf";
    }
    for (int32_t i = suppressEnd; i < vocabSize; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], 2.0f) << "Token " << i << " should be unchanged";
    }
}

// Repetition penalty applied to seenTokens: positive logit divided, negative multiplied.
TEST_F(TalkerKernelTest, TalkerLogitAdjust_RepetitionPenalty)
{
    int32_t const vocabSize = 64;
    int32_t const suppressStart = 50;
    int32_t const suppressEnd = 60;
    float const penalty = 2.0f;

    // token 5: positive logit (4.0 → 4.0/2 = 2.0)
    // token 10: negative logit (-4.0 → -4.0*2 = -8.0)
    std::vector<float> hostLogits(vocabSize, 1.0f);
    hostLogits[5] = 4.0f;
    hostLogits[10] = -4.0f;

    auto result = runTalkerLogitAdjust(hostLogits, suppressStart, suppressEnd,
        /*codecEosId=*/-1, {5, 10}, penalty, stream);

    EXPECT_FLOAT_EQ(result[5], 2.0f) << "Positive logit should be divided by penalty";
    EXPECT_FLOAT_EQ(result[10], -8.0f) << "Negative logit should be multiplied by penalty";
    // Unseen tokens in normal range unchanged
    EXPECT_FLOAT_EQ(result[0], 1.0f) << "Unseen token should be unchanged";
    EXPECT_FLOAT_EQ(result[20], 1.0f) << "Unseen token should be unchanged";
}
