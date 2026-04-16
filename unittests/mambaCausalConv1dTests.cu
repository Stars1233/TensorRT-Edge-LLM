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

#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/causalConv1d.h"
#include "testUtils.h"

using namespace trt_edgellm;
using namespace nvinfer1;

void runCausalConv1dReference(int32_t batch, int32_t seqLen, int32_t dim, int32_t width, int32_t padding,
    std::vector<half> const& x, std::vector<half> const& weight, std::vector<half> const& bias,
    std::vector<half>& outRef)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t s = 0; s < seqLen; ++s)
        {
            int32_t const inBase = s - padding;
            for (int32_t d = 0; d < dim; ++d)
            {
                float acc = __half2float(bias[d]);
                for (int32_t k = 0; k < width; ++k)
                {
                    int32_t const inPos = inBase + k;
                    if (inPos >= 0 && inPos < seqLen)
                    {
                        int64_t const xIdx
                            = static_cast<int64_t>(b) * seqLen * dim + static_cast<int64_t>(inPos) * dim + d;
                        int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                        acc += __half2float(x[xIdx]) * __half2float(weight[wIdx]);
                    }
                }
                int64_t const outIdx = static_cast<int64_t>(b) * seqLen * dim + static_cast<int64_t>(s) * dim + d;
                outRef[outIdx] = __float2half(acc);
            }
        }
    }
}

void runCausalConv1dTest(int32_t batch, int32_t seqLen, int32_t dim, int32_t width)
{
    std::vector<half> xHost(batch * seqLen * dim);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);
    std::vector<half> outputRef(batch * seqLen * dim, __float2half(0.F));

    uniformFloatInitialization<half>(xHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);

    runCausalConv1dReference(batch, seqLen, dim, width, width - 1, xHost, weightHost, biasHost, outputRef);

    auto xDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outputDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(xDevice, xHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);
    CUDA_CHECK(cudaMemset(outputDevice.rawPointer(), 0, outputDevice.getMemoryCapacity()));

    trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1d(xDevice, weightDevice, biasOpt, outputDevice, 1, width - 1, 1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outputHost = copyDeviceToHost<half>(outputDevice);

    for (size_t i = 0; i < outputRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outputHost[i], outputRef[i], 1e-3F, 1e-3F))
            << "Output mismatch at index " << i << ": got " << __half2float(outputHost[i]) << ", expected "
            << __half2float(outputRef[i]);
    }
}

TEST(MambaCausalConv1d, Width2)
{
    runCausalConv1dTest(2, 16, 128, 2);
}

TEST(MambaCausalConv1d, Width3)
{
    runCausalConv1dTest(2, 23, 128, 3);
}

TEST(MambaCausalConv1d, Width4)
{
    runCausalConv1dTest(2, 31, 256, 4);
}

// ---------------------------------------------------------------------------
// invokeCaptureConvState tests
// ---------------------------------------------------------------------------

void runCaptureConvStateReference(int32_t batch, int32_t seqLen, int32_t dim, int32_t width, std::vector<half> const& x,
    std::vector<half>& convStateRef)
{
    std::fill(convStateRef.begin(), convStateRef.end(), __float2half(0.F));
    int32_t const tailLen = (seqLen >= width) ? width : seqLen;
    int32_t const tailStart = seqLen - tailLen;
    int32_t const dstOffset = width - tailLen;
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            for (int32_t t = 0; t < tailLen; ++t)
            {
                int64_t const srcIdx = (static_cast<int64_t>(b) * seqLen + tailStart + t) * dim + d;
                int64_t const dstIdx = (static_cast<int64_t>(b) * dim + d) * width + dstOffset + t;
                convStateRef[dstIdx] = x[srcIdx];
            }
        }
    }
}

void runCaptureConvStateTest(int32_t batch, int32_t seqLen, int32_t dim, int32_t width)
{
    std::vector<half> xHost(batch * seqLen * dim);
    uniformFloatInitialization<half>(xHost, -0.5F, 0.5F);

    std::vector<half> convStateRef(batch * dim * width);
    runCaptureConvStateReference(batch, seqLen, dim, width, xHost, convStateRef);

    auto xDevice = rt::Tensor({batch, seqLen, dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto convStateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(xDevice, xHost);

    mamba_ssm::invokeCaptureConvState(xDevice, convStateDevice, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const convStateHost = copyDeviceToHost<half>(convStateDevice);

    for (size_t i = 0; i < convStateRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(convStateHost[i], convStateRef[i], 1e-3F, 1e-3F))
            << "CaptureConvState mismatch at index " << i << ": got " << __half2float(convStateHost[i]) << ", expected "
            << __half2float(convStateRef[i]);
    }
}

TEST(MambaCaptureConvState, SeqGtWidth)
{
    runCaptureConvStateTest(2, 16, 128, 4);
}

TEST(MambaCaptureConvState, SeqEqWidth)
{
    runCaptureConvStateTest(2, 4, 64, 4);
}

TEST(MambaCaptureConvState, SeqLtWidth)
{
    runCaptureConvStateTest(2, 2, 64, 4);
}

// ---------------------------------------------------------------------------
// invokeCausalConv1dDecode tests
// ---------------------------------------------------------------------------

void runCausalConv1dDecodeReference(int32_t batch, int32_t dim, int32_t width, std::vector<half> const& convState,
    std::vector<half> const& weight, std::vector<half> const& bias, std::vector<half>& outRef)
{
    for (int32_t b = 0; b < batch; ++b)
    {
        for (int32_t d = 0; d < dim; ++d)
        {
            float acc = __half2float(bias[d]);
            for (int32_t k = 0; k < width; ++k)
            {
                int64_t const sIdx = (static_cast<int64_t>(b) * dim + d) * width + k;
                int64_t const wIdx = static_cast<int64_t>(d) * width + k;
                acc += __half2float(convState[sIdx]) * __half2float(weight[wIdx]);
            }
            int64_t const outIdx = static_cast<int64_t>(b) * dim + d;
            outRef[outIdx] = __float2half(acc);
        }
    }
}

void runCausalConv1dDecodeTest(int32_t batch, int32_t dim, int32_t width)
{
    std::vector<half> convStateHost(batch * dim * width);
    std::vector<half> weightHost(dim * width);
    std::vector<half> biasHost(dim);

    uniformFloatInitialization<half>(convStateHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(weightHost, -0.5F, 0.5F);
    uniformFloatInitialization<half>(biasHost, -0.5F, 0.5F);

    std::vector<half> outRef(batch * dim);
    runCausalConv1dDecodeReference(batch, dim, width, convStateHost, weightHost, biasHost, outRef);

    auto convStateDevice = rt::Tensor({batch, dim, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto weightDevice = rt::Tensor({dim, 1, width}, rt::DeviceType::kGPU, DataType::kHALF);
    auto biasDevice = rt::Tensor({dim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto outDevice = rt::Tensor({batch, 1, dim}, rt::DeviceType::kGPU, DataType::kHALF);

    copyHostToDevice(convStateDevice, convStateHost);
    copyHostToDevice(weightDevice, weightHost);
    copyHostToDevice(biasDevice, biasHost);

    trt_edgellm::rt::OptionalInputTensor biasOpt = std::optional(std::cref(biasDevice));
    mamba_ssm::invokeCausalConv1dDecode(convStateDevice, weightDevice, biasOpt, outDevice, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const outHost = copyDeviceToHost<half>(outDevice);

    for (size_t i = 0; i < outRef.size(); ++i)
    {
        EXPECT_TRUE(isclose(outHost[i], outRef[i], 1e-3F, 1e-3F))
            << "Decode mismatch at index " << i << ": got " << __half2float(outHost[i]) << ", expected "
            << __half2float(outRef[i]);
    }
}

TEST(MambaCausalConv1dDecode, Width2)
{
    runCausalConv1dDecodeTest(2, 128, 2);
}

TEST(MambaCausalConv1dDecode, Width4)
{
    runCausalConv1dDecodeTest(2, 256, 4);
}

TEST(MambaCausalConv1dDecode, LargeDim)
{
    runCausalConv1dDecodeTest(4, 512, 4);
}
