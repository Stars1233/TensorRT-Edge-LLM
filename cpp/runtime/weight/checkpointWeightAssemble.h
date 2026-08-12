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
#include "runtime/weight/checkpointReader.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Some final weights consume another prepared engine input. Prerequisites are
//! materialized first; ordinary bindings are independent of one another.
enum class CheckpointWeightPhase
{
    kPrerequisite,
    kWeight,
};

//! Return the materialization phase owned by this transform recipe.
CheckpointWeightPhase checkpointWeightPhase(nlohmann::json const& binding);

//! Whether this recipe maps and releases bounded source ranges internally.
bool checkpointWeightManagesSourceRegistration(nlohmann::json const& binding);

//! Load and convert one checkpoint binding into its final engine-input layout.
//! Every recipe writes directly into the stable manager-owned output. Identity
//! and cast recipes CUDA-map bounded checkpoint windows; other recipes map only
//! the source tensors for the current binding. This path allocates no temporary
//! CUDA memory and performs no cudaMemcpy host/device transfer.
void loadCheckpointWeight(CheckpointReader& checkpoint, nlohmann::json const& binding,
    std::vector<Tensor> const& preparedWeights, Tensor& output, cudaStream_t stream);

} // namespace rt
} // namespace trt_edgellm
