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

#include "trtUtils.h"
#include "checkMacros.h"
#include "cudaUtils.h"
#include <optional>
#include <string_view>

namespace trt_edgellm
{

std::optional<std::pair<cudaGraph_t, cudaGraphExec_t>> captureTRTCudaGraph(
    nvinfer1::IExecutionContext* context, cudaStream_t stream)
{
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    bool executeStatus{true};
    try
    {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        executeStatus &= context->enqueueV3(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(instantiateCudaGraph(&graphExec, graph));
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("Failed to capture CUDA graph: %s", e.what());
        // Clean up any CUDA error if the context is not graph-capturable.
        // We do not want to check return value here, so add static_cast<void> to suppress coverity error.
        static_cast<void>(cudaGetLastError());
        // Stop the capture mode and clear CUDA error status if the stream is still in capturing mode.
        cudaStreamCaptureStatus streamStatus;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &streamStatus));
        if (streamStatus != cudaStreamCaptureStatusNone)
        {
            static_cast<void>(cudaStreamEndCapture(stream, &graph));
            static_cast<void>(cudaGetLastError());
        }
        // At this point, there should be no more cuda errors.
        CUDA_CHECK(cudaGetLastError());
        return std::nullopt;
    }

    if (!executeStatus)
    {
        return std::nullopt;
    }

    return std::make_pair(graph, graphExec);
}

std::string dimsToString(nvinfer1::Dims const& dims) noexcept
{
    std::ostringstream oss;
    oss << "[";
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

bool hasDynamicDims(nvinfer1::Dims const& dims) noexcept
{
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            return true;
        }
    }
    return false;
}

bool dimsEqual(nvinfer1::Dims const& lhs, nvinfer1::Dims const& rhs) noexcept
{
    if (lhs.nbDims != rhs.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < lhs.nbDims; ++i)
    {
        if (lhs.d[i] != rhs.d[i])
        {
            return false;
        }
    }
    return true;
}

bool isEngineInput(nvinfer1::ICudaEngine const& engine, std::string const& tensorName) noexcept
{
    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i)
    {
        std::string const bindingName = engine.getIOTensorName(i);
        if (bindingName == tensorName)
        {
            return engine.getTensorIOMode(bindingName.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        }
    }
    return false;
}

std::string printEngineInfo(nvinfer1::ICudaEngine const* engine, int32_t profileIndex) noexcept
{
    std::stringstream ss;
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        std::string const bindingName = engine->getIOTensorName(i);
        nvinfer1::Dims const maxDims
            = engine->getProfileShape(bindingName.c_str(), profileIndex, nvinfer1::OptProfileSelector::kMAX);
        nvinfer1::Dims const minDims
            = engine->getProfileShape(bindingName.c_str(), profileIndex, nvinfer1::OptProfileSelector::kMIN);
        nvinfer1::Dims const optDims
            = engine->getProfileShape(bindingName.c_str(), profileIndex, nvinfer1::OptProfileSelector::kOPT);
        ss << "  " << bindingName << ": MIN=" << dimsToString(minDims) << ", OPT=" << dimsToString(optDims)
           << ", MAX=" << dimsToString(maxDims) << "\n";
    }
    return ss.str();
}

bool engineHasOutputTensor(nvinfer1::ICudaEngine const* engine, char const* tensorName) noexcept
{
    int32_t const nbIO = engine->getNbIOTensors();
    for (int32_t i = 0; i < nbIO; ++i)
    {
        char const* const name = engine->getIOTensorName(i);
        if (std::string_view{name} == tensorName && engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            return true;
        }
    }
    return false;
}

} // namespace trt_edgellm
