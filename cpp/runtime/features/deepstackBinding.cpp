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

#include "runtime/features/deepstackBinding.h"

#include "common/bindingNames.h"

namespace trt_edgellm
{
namespace rt
{

DeepstackBinding::DeepstackBinding(std::vector<Tensor>& realBuffers, Tensor& zeroTarget)
    : mRealBuffers(&realBuffers)
    , mZeroTarget(&zeroTarget)
{
}

void DeepstackBinding::useRealFeatures(TensorMap& map)
{
    int32_t const n = numFeatures();
    for (int32_t idx = 0; idx < n; ++idx)
    {
        map.set(binding_names::formatDeepstackEmbedsName(idx), (*mRealBuffers)[idx]);
    }
    mMode = Mode::kReal;
}

void DeepstackBinding::useZeroTarget(TensorMap& map)
{
    int32_t const n = numFeatures();
    for (int32_t idx = 0; idx < n; ++idx)
    {
        map.set(binding_names::formatDeepstackEmbedsName(idx), *mZeroTarget);
    }
    mMode = Mode::kZero;
}

std::vector<std::string> DeepstackBinding::ownedNames() const
{
    std::vector<std::string> names;
    int32_t const n = numFeatures();
    names.reserve(n);
    for (int32_t idx = 0; idx < n; ++idx)
    {
        names.push_back(binding_names::formatDeepstackEmbedsName(idx));
    }
    return names;
}

std::string DeepstackBinding::currentModeName() const
{
    return mMode == Mode::kReal ? "Real" : "Zero";
}

} // namespace rt
} // namespace trt_edgellm
