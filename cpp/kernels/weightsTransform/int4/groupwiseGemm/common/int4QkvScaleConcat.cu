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

#include "int4QkvScaleConcat.h"

#include <algorithm>
#include <cuda_fp16.h>

namespace trt_edgellm
{
namespace kernel
{
namespace
{

__global__ void qkvScaleConcatKernel(__half const* qScales, int32_t qWidth, __half const* kScales, int32_t kWidth,
    __half const* vScales, int32_t vWidth, int32_t numGroups, __half* output)
{
    int32_t const outputWidth = qWidth + kWidth + vWidth;
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const count = static_cast<int64_t>(numGroups) * outputWidth;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride)
    {
        int32_t const group = static_cast<int32_t>(index / outputWidth);
        int32_t column = static_cast<int32_t>(index % outputWidth);
        if (column < qWidth)
        {
            output[index] = qScales[static_cast<int64_t>(group) * qWidth + column];
        }
        else if ((column -= qWidth) < kWidth)
        {
            output[index] = kScales[static_cast<int64_t>(group) * kWidth + column];
        }
        else
        {
            column -= kWidth;
            output[index] = vScales[static_cast<int64_t>(group) * vWidth + column];
        }
    }
}

} // namespace

cudaError_t launchGptqInt4QkvScaleConcat(void const* qScalesGN, int32_t qWidth, void const* kScalesGN, int32_t kWidth,
    void const* vScalesGN, int32_t vWidth, int32_t numGroups, void* output, cudaStream_t stream)
{
    if (qScalesGN == nullptr || kScalesGN == nullptr || vScalesGN == nullptr || output == nullptr || qWidth <= 0
        || kWidth <= 0 || vWidth <= 0 || numGroups <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const count = static_cast<int64_t>(numGroups) * (qWidth + kWidth + vWidth);
    int32_t constexpr threads = 256;
    int32_t const blocks = static_cast<int32_t>(std::min<int64_t>((count + threads - 1) / threads, 65535));
    qkvScaleConcatKernel<<<blocks, threads, 0, stream>>>(static_cast<__half const*>(qScalesGN), qWidth,
        static_cast<__half const*>(kScalesGN), kWidth, static_cast<__half const*>(vScalesGN), vWidth, numGroups,
        static_cast<__half*>(output));
    return cudaGetLastError();
}

} // namespace kernel
} // namespace trt_edgellm
