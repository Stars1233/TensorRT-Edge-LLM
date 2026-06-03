/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/tensor.h"
#include "runtime/exec/tensorMap.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <string_view>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class EngineExecutor;

//! Owns externalized model-weight tensors loaded from external weight files.
//!
//! Export can move selected ONNX initializers into safetensors external weight
//! files and expose those weights as engine inputs. This manager loads the
//! external weight files once, validates them against the TensorRT engine in a
//! separate engine-aware step, and registers their stable tensor addresses into
//! a TensorMap for execution.
class ExternalWeightManager
{
public:
    //! @brief Default constructor.
    ExternalWeightManager() = default;

    //! Deleted copy to prevent accidental duplication of GPU resources.
    ExternalWeightManager(ExternalWeightManager const&) = delete;
    ExternalWeightManager& operator=(ExternalWeightManager const&) = delete;

    //! Allow move.
    ExternalWeightManager(ExternalWeightManager&&) noexcept = default;
    ExternalWeightManager& operator=(ExternalWeightManager&&) noexcept = default;

    //! Load external weight tensors listed in @p configPath from @p engineDir.
    //!
    //! Missing `external_weight_files` is treated as an empty external-weight set.
    //!
    //! Must be called exactly once before `validateAgainstEngine()`. Calling
    //! `load()` a second time is a programming error and throws.
    //!
    //! @throws std::runtime_error if an external weight file cannot be loaded or
    //! if `load()` has already been called.
    void load(std::filesystem::path const& engineDir, std::filesystem::path const& configPath, cudaStream_t stream);

    //! Validate loaded external weight tensors against @p executor.
    //!
    //! Must be called exactly once after `load()` and before
    //! `registerTensorMapEntries()`. Calling it a second time is a programming
    //! error and throws.
    //!
    //! @throws std::runtime_error if `load()` has not been called first, if
    //! validation has already happened, or if an external weight tensor does
    //! not match the TensorRT engine input it feeds.
    void validateAgainstEngine(EngineExecutor const& executor, std::string_view engineLabel);

    //! Register all loaded weights in @p map using each tensor's safetensors name.
    //!
    //! This is separate from load() because TensorMap is constructed after
    //! SharedResources, matching the other state managers that publish stable
    //! tensor addresses after the runtime builds its engine binding map.
    //!
    //! Must be called exactly once after `validateAgainstEngine()`. Calling it a
    //! second time is a programming error and throws.
    //!
    //! @throws std::runtime_error if validation has not run first or if
    //! registration has already happened.
    void registerTensorMapEntries(TensorMap& map);

    //! Number of externalized tensors currently owned by the manager.
    size_t size() const noexcept
    {
        return mWeights.size();
    }

private:
    std::vector<Tensor> mWeights{};
    //! True once `load()` has run successfully.
    bool mLoaded{false};
    //! True once loaded weights have been validated against the engine.
    bool mValidated{false};
    //! True once entries have been registered into a TensorMap.
    bool mRegistered{false};
};

} // namespace rt
} // namespace trt_edgellm
