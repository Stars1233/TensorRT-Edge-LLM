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

namespace trt_edgellm
{
namespace kernel
{

//! Repack non-contiguous expert scales directly from CUDA-mapped checkpoint
//! storage. ``secondSources`` is null for a single projection.
cudaError_t launchInt4MoeScaleRepackSourceBatch(uint16_t const* const* firstSources,
    uint16_t const* const* secondSources, uint16_t* marlinOutput, int32_t count, int32_t G, int32_t projectionN,
    cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
