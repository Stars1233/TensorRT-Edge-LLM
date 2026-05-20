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

#pragma once

#include "benchLogger.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "profiling/layerProfiler.h"

#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace trt_edgellm;

// ==================== Fill Helpers ====================

//! Fill a device tensor with random floating-point data, converting to the target dtype.
inline void fillRandomData(rt::Tensor& tensor, float minVal, float maxVal, nvinfer1::DataType dtype, uint64_t seed = 0)
{
    size_t vol = tensor.getShape().volume();
    std::vector<float> hostData(vol);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(minVal, maxVal);

    for (size_t i = 0; i < vol; ++i)
    {
        hostData[i] = dis(gen);
    }

    if (dtype == nvinfer1::DataType::kFLOAT)
    {
        CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(float), cudaMemcpyHostToDevice));
    }
    else if (dtype == nvinfer1::DataType::kHALF)
    {
        std::vector<half> halfData(vol);
        for (size_t i = 0; i < vol; ++i)
            halfData[i] = __float2half(hostData[i]);
        CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), halfData.data(), vol * sizeof(half), cudaMemcpyHostToDevice));
    }
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        std::vector<__nv_bfloat16> bf16Data(vol);
        for (size_t i = 0; i < vol; ++i)
            bf16Data[i] = __float2bfloat16(hostData[i]);
        CUDA_CHECK(
            cudaMemcpy(tensor.rawPointer(), bf16Data.data(), vol * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }
}

//! Fill a device tensor with a constant int32 value.
inline void fillInt32(rt::Tensor& tensor, int32_t val)
{
    size_t vol = tensor.getShape().volume();
    std::vector<int32_t> hostData(vol, val);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(int32_t), cudaMemcpyHostToDevice));
}

//! Fill a device tensor with a constant int8 value.
inline void fillInt8(rt::Tensor& tensor, int8_t val)
{
    size_t vol = tensor.getShape().volume();
    std::vector<int8_t> hostData(vol, val);
    CUDA_CHECK(cudaMemcpy(tensor.rawPointer(), hostData.data(), vol * sizeof(int8_t), cudaMemcpyHostToDevice));
}

// ==================== Run Loop Templates ====================

//! Run warmup iterations with layer profiling disabled.
template <typename ResetFn, typename StepFn>
int runWarmupLoop(std::string const& modeName, int32_t warmupCount, ResetFn const& resetState, StepFn const& step,
    cudaStream_t stream)
{
    LOG_INFO("=== %s Warmup (%d iterations) ===", modeName.c_str(), warmupCount);
    layerProfiler::disableLayerProfilers();

    for (int i = 0; i < warmupCount; ++i)
    {
        resetState();
        if (!step())
        {
            LOG_ERROR("%s warmup iteration %d failed", modeName.c_str(), i);
            return EXIT_FAILURE;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return EXIT_SUCCESS;
}

//! Run layer profiling loop: collects per-layer timing and kernel breakdown per iteration.
template <typename ResetFn, typename StepFn>
int runLayerProfilingLoop(std::string const& modeName, int32_t iterations, bool accumulateForCsv,
    ResetFn const& resetState, StepFn const& step, std::map<std::string, LayerMetadata> const& layerMetadata,
    std::vector<KernelTimes>& timesPerIter, OrderedLayerTimings& layerTimings, cudaStream_t stream)
{
    LOG_INFO("=== %s Layer Profiling (non-CUDA-graph, %d iterations) ===", modeName.c_str(), iterations);

    for (int iter = 0; iter < iterations; ++iter)
    {
        resetState();
        layerProfiler::LayerProfiler::getInstance().reset();
        layerProfiler::enableLayerProfilers();

        if (!step())
        {
            LOG_ERROR("%s iteration %d failed", modeName.c_str(), iter);
            return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        layerProfiler::disableLayerProfilers();

        auto metrics = layerProfiler::LayerProfiler::getInstance().getMetrics();
        KernelTimes times = extractKernelTimes(metrics, layerMetadata);
        timesPerIter.push_back(times);

        if (accumulateForCsv)
        {
            accumulateLayerTimings(metrics, layerTimings);
        }
    }

    return EXIT_SUCCESS;
}

//! Run E2E timing with repeated single-step iterations. Optionally captures a CUDA graph first.
template <typename ResetFn, typename StepFn>
float runRepeatedE2ETiming(
    std::string const& modeName, int32_t iterations, ResetFn const& resetState, StepFn const& step, cudaStream_t stream,
    bool useCudaGraph = false, std::function<bool()> const& captureGraph = []() { return false; })
{
    if (useCudaGraph)
    {
        LOG_INFO("=== Capturing CUDA Graph for %s ===", modeName.c_str());
        resetState();
        if (captureGraph())
        {
            LOG_INFO("CUDA graph captured successfully.");
        }
        else
        {
            LOG_WARNING("Failed to capture CUDA graph, falling back to non-graph execution.");
        }
    }

    LOG_INFO("=== %s E2E Timing (%d iterations) ===", modeName.c_str(), iterations);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> e2eTimes;
    for (int iter = 0; iter < iterations; ++iter)
    {
        resetState();

        CUDA_CHECK(cudaEventRecord(start, stream));
        if (!step())
        {
            LOG_ERROR("%s E2E iteration %d failed", modeName.c_str(), iter);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return -1.0f;
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        e2eTimes.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<double> e2eTimesDouble(e2eTimes.begin(), e2eTimes.end());
    auto [mean, std] = computeStats(e2eTimesDouble);
    LOG_INFO("%s E2E Time: %.4f +/- %.4f ms", modeName.c_str(), mean, std);
    return static_cast<float>(mean);
}

//! Run sequential E2E timing: runs decodeSteps in a single timed block. Used for osl>1 decode.
template <typename ResetFn, typename StepFn, typename PostStepFn, typename CaptureGraphFn>
float runSequentialE2ETiming(std::string const& modeName, int32_t decodeSteps, ResetFn const& resetState,
    StepFn const& step, PostStepFn const& postStep, bool useCudaGraph, CaptureGraphFn const& captureGraph,
    cudaStream_t stream)
{
    if (useCudaGraph)
    {
        LOG_INFO("=== Capturing CUDA Graph for %s ===", modeName.c_str());
        resetState();
        if (captureGraph())
        {
            LOG_INFO("CUDA graph captured successfully.");
        }
        else
        {
            LOG_WARNING("Failed to capture CUDA graph, falling back to non-graph execution.");
        }
    }

    LOG_INFO("=== %s E2E Timing (steps=%d) ===", modeName.c_str(), decodeSteps);

    resetState();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int32_t t = 0; t < decodeSteps; ++t)
    {
        if (!step())
        {
            LOG_ERROR("%s E2E step %d/%d failed", modeName.c_str(), t, decodeSteps);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return -1.0f;
        }
        postStep(t);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float totalTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTimeMs, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    LOG_INFO("E2E Time: %.3f ms (steps=%d)", totalTimeMs, decodeSteps);
    LOG_INFO("Per-step avg: %.3f ms", totalTimeMs / decodeSteps);
    LOG_INFO("Throughput: %.2f tokens/sec", 1000.0f * decodeSteps / totalTimeMs);

    return totalTimeMs;
}
