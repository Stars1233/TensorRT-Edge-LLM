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

//! Copy non-contiguous expert matrices directly from CUDA-mapped checkpoint
//! storage into one contiguous output.
cudaError_t launchCheckpointSourceBatchCopy(
    uint8_t const* const* sources, int32_t count, uint8_t* output, int64_t bytesPerSource, cudaStream_t stream);

//! Copy and zero-pad non-contiguous expert matrices into a final plugin layout.
cudaError_t launchCheckpointSourceBatchPaddedCopy(uint8_t const* const* sources, int32_t count, uint8_t* output,
    int32_t sourceRows, int32_t sourceColumns, int32_t outputRows, int32_t outputColumns, cudaStream_t stream);

//! Transform provider FC1 matrices into the selected NVFP4 MoE plugin layout.
cudaError_t launchNvfp4MoeFc1WeightTransformSourceBatch(uint8_t const* const* upSources,
    uint8_t const* const* gateSources, int32_t count, uint8_t* output, int32_t I, int32_t K, int32_t elementBytes,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream);

//! Transform and zero-pad provider FC1 matrices to a larger plugin I.
cudaError_t launchNvfp4MoeFc1WeightTransformSourceBatchPadded(uint8_t const* const* upSources,
    uint8_t const* const* gateSources, int32_t count, uint8_t* output, int32_t sourceI, int32_t outputI, int32_t K,
    int32_t elementBytes, Nvfp4MoeFc1Layout layout, cudaStream_t stream);

//! Normalize provider-packed FP4 blocks into the plugin quantization
//! convention while arranging and padding the final weight buffer.
//! ``secondWeights`` is null for FC2.
cudaError_t launchNvfp4MoeWeightNormalizeSourceBatchPadded(uint8_t const* const* firstWeights,
    uint8_t const* const* secondWeights, uint8_t const* const* firstScales, uint8_t const* const* secondScales,
    float const* firstAlphas, float const* secondAlphas, int32_t count, uint8_t* output, int32_t sourceRows,
    int32_t sourceRowBytes, int32_t sourceScaleColumns, int32_t outputRows, int32_t outputRowBytes,
    Nvfp4MoeFc1Layout layout, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
