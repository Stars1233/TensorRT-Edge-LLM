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

#include "kernels/moe/moePerExpertScaleKernels.h"

namespace trt_edgellm
{
namespace kernel
{

__global__ void applyPerExpertScaleKernel(
    float* topkWeights, int32_t const* topkIds, float const* perExpertScale, int32_t count)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        topkWeights[idx] *= perExpertScale[topkIds[idx]];
    }
}

void applyPerExpertScale(float* topkWeights, int32_t const* topkIds, float const* perExpertScale, int32_t numTokens,
    int32_t topK, cudaStream_t stream)
{
    int32_t const count = numTokens * topK;
    int32_t const threads = 256;
    int32_t const blocks = (count + threads - 1) / threads;
    applyPerExpertScaleKernel<<<blocks, threads, 0, stream>>>(topkWeights, topkIds, perExpertScale, count);
}

} // namespace kernel
} // namespace trt_edgellm
