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

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "stringUtils.h"

namespace trt_edgellm
{

namespace check
{

/*!
 * @brief Check condition and throw exception if false
 *
 * @param condition Condition to check
 * @param errorMsg Error message to include in exception
 * @throws std::runtime_error If condition is false
 */
inline void check(bool condition, std::string errorMsg)
{
    if (!condition)
    {
        throw std::runtime_error(errorMsg);
    }
}

/*!
 * @brief Internal helper to check CUDA runtime errors
 *
 * @param result CUDA error code
 * @param func Function name string
 * @param file Source file name
 * @param line Source line number
 * @throws std::runtime_error If CUDA error occurred
 */
inline void _checkCuda(cudaError_t result, char const* const func, [[maybe_unused]] char const* const file,
    [[maybe_unused]] int const line)
{
    if (result)
    {
        throw std::runtime_error(format::fmtstr("CUDA runtime error in %s: %s", func, cudaGetErrorString(result)));
    }
}

/*!
 * @brief Internal helper to check CUDA driver API errors
 *
 * @param result CUDA driver error code
 * @param func Function name string
 * @param file Source file name
 * @param line Source line number
 * @throws std::runtime_error If CUDA driver error occurred
 */
inline void _checkCudaDriver(
    CUresult result, char const* const func, [[maybe_unused]] char const* const file, [[maybe_unused]] int const line)
{
    if (result)
    {
        char const* errorName = nullptr;
        if (cuGetErrorName(result, &errorName) != CUDA_SUCCESS)
        {
            errorName = "CUDA driver API error happened, but we failed to get error name.";
        }
        throw std::runtime_error(format::fmtstr("CUDA driver API error in %s: %s", func, errorName));
    }
}

} // namespace check

// Stringify helpers for embedding __LINE__ as a string literal at preprocessor time.
#define TRT_EDGELLM_STRINGIFY_IMPL(x) #x
#define TRT_EDGELLM_STRINGIFY(x) TRT_EDGELLM_STRINGIFY_IMPL(x)

/*!
 * @brief Lazy-message precondition check
 *
 * Throws std::runtime_error if @p cond is false. The @p msg expression is only
 * evaluated on failure, which matters when the message uses string concatenation
 * (e.g. std::to_string, ostringstream) on hot paths. The thrown message is
 * prefixed with __FILE__:__LINE__ to aid debugging.
 *
 * Usage: ELLM_CHECK(ptr != nullptr, "ptr must not be null");
 *        ELLM_CHECK(n > 0, "n must be positive, got " + std::to_string(n));
 */
#define ELLM_CHECK(cond, msg)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            throw std::runtime_error(std::string(__FILE__ ":" TRT_EDGELLM_STRINGIFY(__LINE__) ": ") + (msg));          \
        }                                                                                                              \
    } while (0)

/*!
 * @brief Check CUDA runtime API calls
 *
 * Wraps CUDA runtime API calls and throws exception on error.
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(stat)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        trt_edgellm::check::_checkCuda((stat), #stat, __FILE__, __LINE__);                                             \
    } while (0)

/*!
 * @brief Check CUDA driver API calls
 *
 * Wraps CUDA driver API calls and throws exception on error.
 * Usage: CUDA_DRIVER_CHECK(cuMemAlloc(&dptr, size));
 */
#define CUDA_DRIVER_CHECK(stat)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        trt_edgellm::check::_checkCudaDriver((stat), #stat, __FILE__, __LINE__);                                       \
    } while (0)

} // namespace trt_edgellm
