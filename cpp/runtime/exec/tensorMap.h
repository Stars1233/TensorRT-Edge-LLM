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

#include "common/tensor.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Name-to-Tensor pointer mapping for the TRT binding layer.
 *
 * TensorMap holds non-owning pointers to tensors keyed by name. Callers are
 * responsible for ensuring the referenced tensors outlive the map.
 */
class TensorMap
{
public:
    //! @brief Default constructor
    TensorMap() = default;

    /*!
     * @brief Associate a name with a tensor.
     *
     * If the name already exists it is overwritten with the new pointer.
     *
     * @param name Binding name (e.g. "inputs_embeds")
     * @param tensor Tensor to associate — the map stores a non-owning pointer
     */
    void set(std::string const& name, Tensor& tensor);

    //! @brief Deleted rvalue overload — prevents callers from binding a moved-from
    //! or local temporary whose storage would go out of scope before `bindAll`
    //! dereferences the stored pointer.  Any attempt to pass a `Tensor&&` fails
    //! to compile as a deliberate tripwire.
    void set(std::string const& name, Tensor&& tensor) = delete;

    /*!
     * @brief Retrieve a tensor by name.
     *
     * @param name Binding name to look up
     * @return Pointer to the associated tensor, or nullptr if not found
     */
    Tensor* get(std::string const& name) const;

    /*!
     * @brief Check whether a name is present in the map.
     *
     * @param name Binding name to check
     * @return True if the name exists, false otherwise
     */
    bool contains(std::string const& name) const;

    /*!
     * @brief Return all registered names.
     *
     * @return Vector of all names currently stored in the map
     */
    std::vector<std::string> allNames() const;

    //! @brief Return the number of entries in the map.
    size_t size() const
    {
        return mTensors.size();
    }

private:
    std::unordered_map<std::string, Tensor*> mTensors;
};

} // namespace rt
} // namespace trt_edgellm
