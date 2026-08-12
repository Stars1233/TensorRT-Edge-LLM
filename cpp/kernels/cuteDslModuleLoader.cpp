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

#include "cuteDslModuleLoader.h"

#include "common/logger.h"

#if defined(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
#include <cstdlib>
#include <cstring>
#endif

namespace trt_edgellm::detail
{

namespace
{

thread_local cudaError_t gCuteDslCudaError{cudaSuccess};

char const* nonNullName(char const* moduleName) noexcept
{
    return moduleName != nullptr ? moduleName : "<unnamed>";
}

} // namespace

void recordCuteDslCudaError(cudaError_t error) noexcept
{
    if (error != cudaSuccess && gCuteDslCudaError == cudaSuccess)
    {
        gCuteDslCudaError = error;
    }
}

void clearCuteDslCudaError() noexcept
{
    gCuteDslCudaError = cudaSuccess;
}

cudaError_t takeCuteDslCudaError() noexcept
{
    cudaError_t const error = gCuteDslCudaError;
    gCuteDslCudaError = cudaSuccess;
    return error;
}

namespace module_loader
{

std::mutex& getGlobalMutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

void logLoaded(char const* moduleName) noexcept
{
    try
    {
        LOG_DEBUG("CuTe DSL kernel module loaded: %s", nonNullName(moduleName));
    }
    catch (...)
    {
    }
}

void logFailure(char const* moduleName, char const* reason) noexcept
{
    try
    {
        LOG_ERROR("CuTe DSL kernel module '%s' failed: %s", nonNullName(moduleName),
            reason != nullptr ? reason : "unknown failure");
    }
    catch (...)
    {
    }
}

void logCudaFailure(char const* moduleName, char const* operation, cudaError_t error) noexcept
{
    try
    {
        LOG_ERROR("CuTe DSL kernel module '%s' failed to %s: %s (%s)", nonNullName(moduleName),
            operation != nullptr ? operation : "complete a CUDA operation", cudaGetErrorName(error),
            cudaGetErrorString(error));
    }
    catch (...)
    {
    }
}

#if defined(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
bool shouldInjectFailure(char const* moduleName) noexcept
{
    char const* const requestedModule = std::getenv("TRT_EDGELLM_TEST_FAIL_CUTEDSL_MODULE");
    return requestedModule != nullptr && moduleName != nullptr && std::strcmp(requestedModule, moduleName) == 0;
}
#endif

} // namespace module_loader

} // namespace trt_edgellm::detail
