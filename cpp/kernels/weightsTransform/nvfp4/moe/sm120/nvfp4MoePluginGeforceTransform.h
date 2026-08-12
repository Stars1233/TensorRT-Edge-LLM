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

#include "kernels/weightsTransform/nvfp4/moe/common/nvfp4MoeScaleTransform.h"
#include "kernels/weightsTransform/nvfp4/moe/common/nvfp4MoeWeightTransform.h"

namespace trt_edgellm
{
namespace kernel
{

inline cudaError_t launchNvfp4MoePluginGeforceFc1WeightTransform(uint8_t const* const* upSources,
    uint8_t const* const* gateSources, int32_t count, uint8_t* output, int32_t I, int32_t K, int32_t elementBytes,
    cudaStream_t stream)
{
    return launchNvfp4MoeFc1WeightTransformSourceBatch(
        upSources, gateSources, count, output, I, K, elementBytes, Nvfp4MoeFc1Layout::kConcatenated, stream);
}

inline cudaError_t launchNvfp4MoePluginGeforceScaleTransform(uint8_t const* const* firstSources,
    uint8_t const* const* secondSources, int32_t count, int8_t* output, int32_t rows, int32_t Ksf, cudaStream_t stream)
{
    return launchNvfp4MoeScaleTransformSourceBatch(
        firstSources, secondSources, count, output, rows, Ksf, Nvfp4MoeFc1Layout::kConcatenated, stream);
}

} // namespace kernel
} // namespace trt_edgellm
