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

#include "runtime/exec/tensorRegistry.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

void TensorRegistry::addTensor(TensorSpec spec)
{
    // Catch spec authoring mistakes at registration time, not at the first bindAll:
    // resolveShape writes into nvinfer1::Dims::d[MAX_DIMS] (fixed-size 8). Today all
    // specs are 5D or fewer, but checking here means a future TensorSpec with too many
    // dims fails loud at construction instead of silently corrupting the stack.
    ELLM_CHECK(spec.shape.size() <= static_cast<size_t>(nvinfer1::Dims::MAX_DIMS),
        "TensorRegistry::addTensor: spec '" + spec.name + "' has " + std::to_string(spec.shape.size())
            + " dims, exceeds nvinfer1::Dims::MAX_DIMS=" + std::to_string(nvinfer1::Dims::MAX_DIMS));
    mSpecs.push_back(std::move(spec));
    mDirty = true;
}

std::vector<TensorSpec> TensorRegistry::expandPerLayer(TensorSpec const& spec) const
{
    std::vector<TensorSpec> result;
    if (spec.perLayer <= 0)
    {
        result.push_back(spec);
        result.back().perLayer = 0;
        return result;
    }
    for (int32_t i = 0; i < spec.perLayer; ++i)
    {
        TensorSpec expanded = spec;
        char buf[256];
        std::snprintf(buf, sizeof(buf), spec.name.c_str(), i);
        expanded.name = buf;
        expanded.perLayer = 0;
        result.push_back(std::move(expanded));
    }
    return result;
}

std::vector<TensorSpec> const& TensorRegistry::allExpandedSpecs() const
{
    if (mDirty)
    {
        mExpandedSpecs.clear();
        mReferencedMembers.clear();
        for (auto const& spec : mSpecs)
        {
            auto expanded = expandPerLayer(spec);
            mExpandedSpecs.insert(mExpandedSpecs.end(), expanded.begin(), expanded.end());
        }
        // `std::hash` has no specialization for pointer-to-member, so dedupe via
        // a linear scan. The working set is small (≤6 LLM symbolic dims).
        for (auto const& spec : mExpandedSpecs)
        {
            for (auto const& dim : spec.shape)
            {
                if (dim.isSymbolic()
                    && std::find(mReferencedMembers.begin(), mReferencedMembers.end(), dim.symbol)
                        == mReferencedMembers.end())
                {
                    mReferencedMembers.push_back(dim.symbol);
                }
            }
        }
        mDirty = false;
    }
    return mExpandedSpecs;
}

std::vector<int64_t InferenceDims::*> const& TensorRegistry::referencedMembers() const
{
    // allExpandedSpecs() populates mReferencedMembers as a side-effect when mDirty.
    (void) allExpandedSpecs();
    return mReferencedMembers;
}

std::vector<std::string> TensorRegistry::allTensorNames() const
{
    std::vector<std::string> names;
    for (auto const& spec : allExpandedSpecs())
    {
        names.push_back(spec.name);
    }
    return names;
}

bool TensorRegistry::contains(std::string_view name) const
{
    for (auto const& spec : allExpandedSpecs())
    {
        if (spec.name == name)
        {
            return true;
        }
    }
    return false;
}

nvinfer1::Dims TensorRegistry::resolveShape(
    std::vector<ShapeDim> const& shape, InferenceDims const& inferenceDims) const
{
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = shape[i].isSymbolic() ? inferenceDims.*(shape[i].symbol) : shape[i].value;
    }
    return dims;
}

bool TensorRegistry::bindAll(
    nvinfer1::IExecutionContext* ctx, TensorMap const& tensorMap, InferenceDims const& inferenceDims) const
{
    bool success = true;
    for (auto const& spec : allExpandedSpecs())
    {
        Tensor* tensor = tensorMap.get(spec.name);
        if (!tensor)
        {
            LOG_ERROR("tensor '%s' not in tensorMap", spec.name.c_str());
            success = false;
            continue;
        }
        if (!ctx->setTensorAddress(spec.name.c_str(), tensor->rawPointer()))
        {
            LOG_ERROR("setTensorAddress failed for '%s'", spec.name.c_str());
            success = false;
            continue;
        }
        if (spec.io == TensorIO::kInput)
        {
            // Resolve the input shape from symbolic dimensions. This is necessary
            // because some tensors (e.g. KV cache) are allocated at maxBatchSize
            // but need their input shape set to activeBatchSize for TRT execution.
            // Tensors with all-fixed shapes are returned as-is.
            auto dims = resolveShape(spec.shape, inferenceDims);
            if (!ctx->setInputShape(spec.name.c_str(), dims))
            {
                LOG_ERROR("setInputShape failed for '%s'", spec.name.c_str());
                success = false;
            }
        }
    }
    return success;
}

} // namespace rt
} // namespace trt_edgellm
