/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef CUTE_DSL_FMHA_ENABLED

#include <cuda_fp16.h>

#include <climits>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "common/cudaUtils.h"
#include "contextAttnReference.h"
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#include "kernels/posEncoding/applyRopeWriteKV.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

bool isSupportedCuteDslTestSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 110;
}

void expectHalfOutputsClose(rt::Tensor const& actualTensor, rt::Tensor const& expectedTensor, std::string const& label)
{
    ASSERT_EQ(actualTensor.getShape().volume(), expectedTensor.getShape().volume()) << label;

    auto const actual = copyDeviceToHost<half>(actualTensor);
    auto const expected = copyDeviceToHost<half>(expectedTensor);
    auto const& shape = actualTensor.getShape();

    bool nanDetected = false;
    int64_t closeWithin1e3 = 0;
    int64_t const totalElements = static_cast<int64_t>(actual.size());

    for (int64_t idx = 0; idx < totalElements; ++idx)
    {
        float const actualValue = __half2float(actual[static_cast<size_t>(idx)]);
        float const expectedValue = __half2float(expected[static_cast<size_t>(idx)]);

        ASSERT_TRUE(isclose(actual[static_cast<size_t>(idx)], expected[static_cast<size_t>(idx)], 1e-2f, 1e-2f))
            << label << " mismatch at index=" << formatTensorIndex(shape, idx) << " flat_index=" << idx
            << " expected=" << expectedValue << " actual=" << actualValue;

        if (isclose(actual[static_cast<size_t>(idx)], expected[static_cast<size_t>(idx)], 1e-3f, 1e-3f))
        {
            ++closeWithin1e3;
        }

        nanDetected = nanDetected || std::isnan(actualValue);
    }

    float const passRate1e3 = static_cast<float>(closeWithin1e3) / static_cast<float>(totalElements);
    EXPECT_GT(passRate1e3, 0.9f) << label;
    EXPECT_FALSE(nanDetected) << label;
}

void runViTAccuracyCase(std::vector<int32_t> const& cuSeqLens, int32_t numHeads, int32_t headDim, int32_t maxSeqLen)
{
    int32_t const batchSize = static_cast<int32_t>(cuSeqLens.size()) - 1;
    int32_t const totalSeqLen = cuSeqLens.back();

    size_t const qkvSize = static_cast<size_t>(totalSeqLen) * numHeads * headDim;

    std::vector<half> qInput(qkvSize);
    std::vector<half> kInput(qkvSize);
    std::vector<half> vInput(qkvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    rt::Tensor qTensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({totalSeqLen, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    copyHostToDevice(qTensor, qInput);
    copyHostToDevice(kTensor, kInput);
    copyHostToDevice(vTensor, vInput);
    copyHostToDevice(cuSeqLensTensor, cuSeqLens);

    cudaStream_t stream = nullptr;

    rt::launchFmhaReferenceCompact(
        qTensor, kTensor, vTensor, outputReference, cuSeqLensTensor, maxSeqLen, false, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CuteDslFMHARunner runner(numHeads, numHeads, headDim);
    runner.run(qTensor.dataPointer<half>(), kTensor.dataPointer<half>(), vTensor.dataPointer<half>(),
        outputCuteDsl.dataPointer<half>(), cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen, maxSeqLen, batchSize,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectHalfOutputsClose(outputCuteDsl, outputReference,
        "ViT CuTe DSL FMHA headDim=" + std::to_string(headDim) + " numHeads=" + std::to_string(numHeads));
}

void runLlmAccuracyCase(int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headDim)
{
    size_t const qSize = static_cast<size_t>(batchSize) * seqLen * numQHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * seqLen * numKVHeads * headDim;

    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    rt::Tensor qCute({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kCute({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vCute({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor qReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vReference({batchSize, seqLen, numKVHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(qCute, qInput);
    copyHostToDevice(kCute, kInput);
    copyHostToDevice(vCute, vInput);
    copyHostToDevice(qReference, qInput);
    copyHostToDevice(kReference, kInput);
    copyHostToDevice(vReference, vInput);

    rt::Tensor kvCacheCute({batchSize, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvCacheReference({batchSize, 2, numKVHeads, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputCuteDsl({batchSize, seqLen, numQHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cosSinCache({1, seqLen, headDim}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kvCacheEndLens({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLens({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);

    CUDA_CHECK(cudaMemset(kvCacheCute.rawPointer(), 0, kvCacheCute.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(kvCacheReference.rawPointer(), 0, kvCacheReference.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(outputReference.rawPointer(), 0, outputReference.getShape().volume() * sizeof(half)));
    CUDA_CHECK(cudaMemset(outputCuteDsl.rawPointer(), 0, outputCuteDsl.getShape().volume() * sizeof(half)));

    std::vector<int32_t> kvCacheEndLensHost(static_cast<size_t>(batchSize), seqLen);
    std::vector<int32_t> cuKVSeqLensHost(static_cast<size_t>(batchSize + 1));
    for (int32_t idx = 0; idx <= batchSize; ++idx)
    {
        cuKVSeqLensHost[static_cast<size_t>(idx)] = idx * seqLen;
    }

    copyHostToDevice(kvCacheEndLens, kvCacheEndLensHost);
    copyHostToDevice(cuKVSeqLens, cuKVSeqLensHost);

    cudaStream_t stream = nullptr;
    std::vector<float> cosSinCacheHost(static_cast<size_t>(cosSinCache.getShape().volume()));
    uniformFloatInitialization(cosSinCacheHost, -1.0f, 1.0f);
    copyHostToDevice(cosSinCache, cosSinCacheHost);

    rt::Tensor emptyScaleTensor{};
    kernel::launchApplyRopeWriteKVSplitQKV(
        cosSinCache, kvCacheEndLens, qCute, kCute, vCute, kvCacheCute, emptyScaleTensor, stream);
    kernel::launchApplyRopeWriteKV(cosSinCache, std::nullopt, qReference, kReference, vReference, kvCacheReference,
        emptyScaleTensor, stream, true);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CuteDslFMHARunner runner(numQHeads, numKVHeads, headDim, batchSize, seqLen, seqLen);
    runner.run(qCute.dataPointer<half>(), kvCacheCute.dataPointer<half>(), outputCuteDsl.dataPointer<half>(),
        cuKVSeqLens.dataPointer<int32_t>(), stream, INT_MAX);

    rt::launchFmhaReferenceBshd(qReference, kReference, vReference, outputReference, true, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectHalfOutputsClose(outputCuteDsl, outputReference,
        "LLM CuTe DSL FMHA batch=" + std::to_string(batchSize) + " seqLen=" + std::to_string(seqLen)
            + " numQHeads=" + std::to_string(numQHeads) + " numKVHeads=" + std::to_string(numKVHeads)
            + " headDim=" + std::to_string(headDim));
}

} // namespace

TEST(CuteDslFMHARunnerTest, vitAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    if (!CuteDslFMHARunner::loadViTKernelModule())
    {
        GTEST_SKIP() << "Failed to load CuTe DSL ViT FMHA kernel module";
    }

    struct ViTCase
    {
        std::vector<int32_t> cuSeqLens;
        int32_t numHeads;
        int32_t headDim;
        int32_t maxSeqLen;
    };

    std::vector<ViTCase> const cases{
        {{0, 32, 60, 88, 128}, 14, 64, 128},
        {{0, 16, 64}, 14, 72, 128},
        {{0, 24, 80, 144}, 14, 80, 160},
        {{0, 100, 200, 300}, 14, 128, 512},
    };

    for (auto const& testCase : cases)
    {
        std::string cuSeqLensStr = "[";
        for (size_t i = 0; i < testCase.cuSeqLens.size(); ++i)
        {
            if (i)
                cuSeqLensStr += ",";
            cuSeqLensStr += std::to_string(testCase.cuSeqLens[i]);
        }
        cuSeqLensStr += "]";
        SCOPED_TRACE(::testing::Message() << "numHeads=" << testCase.numHeads << " headDim=" << testCase.headDim
                                          << " maxSeqLen=" << testCase.maxSeqLen << " cuSeqLens=" << cuSeqLensStr);
        runViTAccuracyCase(testCase.cuSeqLens, testCase.numHeads, testCase.headDim, testCase.maxSeqLen);
    }
}

TEST(CuteDslFMHARunnerTest, llmAccuracy)
{
    int32_t const rawSmVersion = getSMVersion();
    if (!isSupportedCuteDslTestSm(rawSmVersion))
    {
        GTEST_SKIP() << "CuTe DSL FMHA unit tests only run on SM100/101/110. Current SM=" << rawSmVersion;
    }

    if (!CuteDslFMHARunner::loadLLMKernelModule())
    {
        GTEST_SKIP() << "Failed to load CuTe DSL LLM FMHA kernel module";
    }

    struct LlmCase
    {
        int32_t batchSize;
        int32_t seqLen;
        int32_t numQHeads;
        int32_t numKVHeads;
        int32_t headDim;
    };

    std::vector<LlmCase> const cases{
        {2, 32, 8, 8, 64},
        {2, 48, 16, 4, 64},
        {1, 24, 8, 8, 128},
        {1, 32, 12, 4, 128},
    };

    for (auto const& testCase : cases)
    {
        SCOPED_TRACE(::testing::Message() << "batchSize=" << testCase.batchSize << " seqLen=" << testCase.seqLen
                                          << " numQHeads=" << testCase.numQHeads
                                          << " numKVHeads=" << testCase.numKVHeads << " headDim=" << testCase.headDim);
        runLlmAccuracyCase(
            testCase.batchSize, testCase.seqLen, testCase.numQHeads, testCase.numKVHeads, testCase.headDim);
    }
}

#endif
