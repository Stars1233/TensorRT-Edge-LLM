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

#include "runtime/exec/tensorMap.h"

namespace trt_edgellm
{
namespace rt
{

void TensorMap::set(std::string const& name, Tensor& tensor)
{
    mTensors[name] = &tensor;
}

Tensor* TensorMap::get(std::string const& name) const
{
    auto it = mTensors.find(name);
    return it != mTensors.end() ? it->second : nullptr;
}

bool TensorMap::contains(std::string const& name) const
{
    return mTensors.count(name) > 0;
}

std::vector<std::string> TensorMap::allNames() const
{
    std::vector<std::string> names;
    names.reserve(mTensors.size());
    for (auto const& [name, _] : mTensors)
    {
        names.push_back(name);
    }
    return names;
}

} // namespace rt
} // namespace trt_edgellm
