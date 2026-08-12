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

#include <cstdint>
#include <cuda_runtime.h>

#include "kernels/weightsTransform/nvfp4/moe/common/nvfp4MoeLayout.h"

namespace trt_edgellm
{
namespace kernel
{

//! Transform non-contiguous checkpoint scales into an NVFP4 MoE MMA layout.
//! ``secondSources`` is null for FC2; otherwise ``layout`` selects the FC1
//! arrangement required by the consuming plugin.
cudaError_t launchNvfp4MoeScaleTransformSourceBatch(uint8_t const* const* firstSources,
    uint8_t const* const* secondSources, int32_t count, int8_t* mmaLayout, int32_t rows, int32_t Ksf,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream);

//! Transform scales while zero-padding to larger plugin M/K dimensions.
cudaError_t launchNvfp4MoeScaleTransformSourceBatchPadded(uint8_t const* const* firstSources,
    uint8_t const* const* secondSources, int32_t count, int8_t* mmaLayout, int32_t sourceRows, int32_t sourceKsf,
    int32_t outputRows, int32_t outputKsf, Nvfp4MoeFc1Layout layout, cudaStream_t stream);

//! Requantize provider-packed FP4 blocks and write their normalized FP8
//! scales directly in the final plugin layout. ``secondWeights`` is null for
//! FC2.
cudaError_t launchNvfp4MoeScaleNormalizeSourceBatchPadded(uint8_t const* const* firstWeights,
    uint8_t const* const* secondWeights, uint8_t const* const* firstScales, uint8_t const* const* secondScales,
    float const* firstAlphas, float const* secondAlphas, int32_t count, int8_t* mmaLayout, int32_t sourceRows,
    int32_t sourceRowBytes, int32_t sourceScaleColumns, int32_t outputRows, int32_t outputScaleColumns,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
