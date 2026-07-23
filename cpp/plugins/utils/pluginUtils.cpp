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

#include "pluginUtils.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include <memory>

// Follow the TRT initLibNvInferPlugins(void*, char const*) convention:
// the host application calls this after dlopen to pass its logger,
// so plugin LOG_DEBUG messages respect the application's log level.
extern "C" EDGELLM_PLUGIN_EXPORT bool initEdgellmPlugins(void* logger, char const* /*libNamespace*/)
{
    if (logger)
    {
        auto* appLogger = dynamic_cast<trt_edgellm::logger::EdgeLLMLogger*>(static_cast<nvinfer1::ILogger*>(logger));
        if (appLogger)
        {
            trt_edgellm::gLogger.setLevel(appLogger->getLevel());
        }
    }
    return true;
}

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

void applyThorSMRenumberWAR(int32_t& smVersion)
{
    // Workaround for CUDA12/13 Thor re-numbering. The kernels themselves have version compatibility.
    if (smVersion == 110)
    {
        smVersion = 101;
    }
}

size_t alignTensorSize(size_t size)
{
    return ((size + kDEVICE_ALIGNMENT - 1) / kDEVICE_ALIGNMENT) * kDEVICE_ALIGNMENT;
}

size_t accumulateWorkspaceSize(size_t currentSize, rt::Coords const& shape, DataType dataType)
{
    size_t alignedSize = alignTensorSize(currentSize);
    size_t tensorSizeBytes = rt::utils::getTypeSize(dataType) * static_cast<size_t>(shape.volume());

    return alignedSize + alignTensorSize(tensorSizeBytes);
}

rt::Tensor assignTensorFromWorkspace(std::byte*& workspace, rt::Coords const& shape, DataType dataType)
{
    size_t space = kDEVICE_ALIGNMENT;
    void* workspace_void = workspace;
    check::check(
        workspace != nullptr && std::align(kDEVICE_ALIGNMENT, kDEVICE_ALIGNMENT, workspace_void, space) == workspace,
        "Workspace pointer shall be valid and aligned to device alignment granularity.");

    // Create non-owned tensor instance from the workspace pointer.
    rt::Tensor tensor(workspace, shape, rt::DeviceType::kGPU, dataType);

    // Move the workspace pointer to the next aligned position after this tensor.
    size_t alignedSize = alignTensorSize(tensor.getMemoryCapacity());
    workspace += alignedSize;
    return tensor;
}

} // namespace plugins
} // namespace trt_edgellm
