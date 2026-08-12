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

#include <atomic>
#include <cuda_runtime.h>
#include <exception>
#include <mutex>
#include <type_traits>

namespace trt_edgellm::detail
{

//! Record the first CUDA error reported by a generated CuTe DSL loader or unloader.
void recordCuteDslCudaError(cudaError_t error) noexcept;

//! Clear and consume the current thread's generated-loader CUDA error.
void clearCuteDslCudaError() noexcept;
cudaError_t takeCuteDslCudaError() noexcept;

enum class LazyKernelModuleStatus : unsigned char
{
    kUninitialized,
    kLoaded,
    kPoisoned,
};

//! Process-lifetime state for one generated CuTe DSL kernel module variant.
template <typename Module>
struct LazyKernelModule
{
    Module module{};
    std::once_flag once;
    std::atomic<LazyKernelModuleStatus> status{LazyKernelModuleStatus::kUninitialized};

    LazyKernelModule() = default;
    LazyKernelModule(LazyKernelModule const&) = delete;
    LazyKernelModule& operator=(LazyKernelModule const&) = delete;
};

namespace module_loader
{

class RetryableFailure
{
};

std::mutex& getGlobalMutex() noexcept;
void logLoaded(char const* moduleName) noexcept;
void logFailure(char const* moduleName, char const* reason) noexcept;
void logCudaFailure(char const* moduleName, char const* operation, cudaError_t error) noexcept;

#if defined(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
bool shouldInjectFailure(char const* moduleName) noexcept;
#endif

} // namespace module_loader

//! Load one generated module variant on its first dispatch.
//!
//! Loader and Unloader are the generated `*_Kernel_Module_Load` and
//! `*_Kernel_Module_Unload` functions. A clean failure throws through
//! std::call_once so a later dispatch can retry. If partial-module cleanup
//! fails, the state is poisoned and all later dispatches fail without retry.
//! Successfully loaded modules remain resident for the lifetime of the process.
template <auto Loader, auto Unloader, typename Module>
bool ensureModuleLoaded(LazyKernelModule<Module>& state, char const* moduleName, cudaStream_t stream) noexcept
{
    static_assert(std::is_invocable_v<decltype(Loader), Module*>, "CuTe DSL loader must accept Module*");
    static_assert(std::is_invocable_v<decltype(Unloader), Module*>, "CuTe DSL unloader must accept Module*");

    auto const initialStatus = state.status.load(std::memory_order_acquire);
    if (initialStatus == LazyKernelModuleStatus::kLoaded)
    {
        return true;
    }
    if (initialStatus == LazyKernelModuleStatus::kPoisoned)
    {
        return false;
    }

    try
    {
        std::call_once(state.once, [&state, moduleName, stream] {
            // CUDA library loading mutates process-global driver state. Keep
            // different variants from entering that slow path concurrently.
            std::lock_guard<std::mutex> const lock(module_loader::getGlobalMutex());

            cudaStreamCaptureStatus captureStatus{cudaStreamCaptureStatusNone};
            cudaError_t const captureError = cudaStreamIsCapturing(stream, &captureStatus);
            if (captureError != cudaSuccess)
            {
                module_loader::logCudaFailure(moduleName, "query CUDA stream capture state", captureError);
                throw module_loader::RetryableFailure{};
            }
            if (captureStatus != cudaStreamCaptureStatusNone)
            {
                module_loader::logFailure(
                    moduleName, "first use during CUDA graph capture; execute an uncaptured warmup first");
                throw module_loader::RetryableFailure{};
            }

            cudaError_t const contextError = cudaFree(nullptr);
            if (contextError != cudaSuccess)
            {
                module_loader::logCudaFailure(moduleName, "initialize the CUDA context", contextError);
                throw module_loader::RetryableFailure{};
            }

#if defined(ENABLE_CUTEDSL_MODULE_TEST_HOOK)
            // This branch and its environment-variable implementation do not
            // exist in release builds.
            if (module_loader::shouldInjectFailure(moduleName))
            {
                module_loader::logFailure(moduleName, "failure injected by the CuTe DSL module test hook");
                throw module_loader::RetryableFailure{};
            }
#endif

            clearCuteDslCudaError();
            bool loaderThrew{false};
            try
            {
                Loader(&state.module);
            }
            catch (std::exception const& error)
            {
                loaderThrew = true;
                module_loader::logFailure(moduleName, error.what());
            }
            catch (...)
            {
                loaderThrew = true;
                module_loader::logFailure(moduleName, "generated loader threw an unknown exception");
            }

            cudaError_t const loadError = takeCuteDslCudaError();
            bool const hasModule = state.module.module != nullptr;
            if (!loaderThrew && loadError == cudaSuccess && hasModule)
            {
                state.status.store(LazyKernelModuleStatus::kLoaded, std::memory_order_release);
                module_loader::logLoaded(moduleName);
                return;
            }

            if (loadError != cudaSuccess)
            {
                module_loader::logCudaFailure(moduleName, "load generated module", loadError);
            }
            else if (!loaderThrew && !hasModule)
            {
                module_loader::logFailure(moduleName, "generated loader returned a null module handle");
            }

            bool cleanupFailed{false};
            if (hasModule)
            {
                clearCuteDslCudaError();
                try
                {
                    Unloader(&state.module);
                }
                catch (std::exception const& error)
                {
                    cleanupFailed = true;
                    module_loader::logFailure(moduleName, error.what());
                }
                catch (...)
                {
                    cleanupFailed = true;
                    module_loader::logFailure(moduleName, "generated unloader threw an unknown exception");
                }

                cudaError_t const cleanupError = takeCuteDslCudaError();
                if (cleanupError != cudaSuccess)
                {
                    cleanupFailed = true;
                    module_loader::logCudaFailure(moduleName, "clean up a partially loaded module", cleanupError);
                }
            }

            if (cleanupFailed)
            {
                state.status.store(LazyKernelModuleStatus::kPoisoned, std::memory_order_release);
                module_loader::logFailure(moduleName, "partial-load cleanup failed; module state is poisoned");
                return;
            }

            state.module = Module{};
            throw module_loader::RetryableFailure{};
        });
    }
    catch (module_loader::RetryableFailure const&)
    {
        return false;
    }
    catch (std::exception const& error)
    {
        module_loader::logFailure(moduleName, error.what());
        return false;
    }
    catch (...)
    {
        module_loader::logFailure(moduleName, "unexpected lazy-loader failure");
        return false;
    }

    return state.status.load(std::memory_order_acquire) == LazyKernelModuleStatus::kLoaded;
}

} // namespace trt_edgellm::detail
