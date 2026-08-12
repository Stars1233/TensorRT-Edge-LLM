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

/* Adapted from https://github.com/sgl-project/sglang/tree/444bbd866dbb0c58c75ae14df6f7f65685b28c3c
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the SGLang project
 */

#include <cuda_runtime.h>

#include "common/cudaMacros.h"
#include "kernel.h"
#include "marlin_template.h"

#if SUPPORTS_FP8

namespace MARLIN_NAMESPACE_NAME
{

#define INSTANTIATE_NVFP4_BF16(THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8)             \
    template __global__ void                                                                                           \
    Marlin<trt_edgellm::marlin_dtypes::kBFloat16.id(), trt_edgellm::marlin_dtypes::kFE2M1f.id(),                       \
        trt_edgellm::marlin_dtypes::kBFloat16.id(), trt_edgellm::marlin_dtypes::kFE4M3fn.id(), THREADS,                \
        THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, 4, 1, false>(MARLIN_KERNEL_PARAMS)

INSTANTIATE_NVFP4_BF16(256, 1, 8, 8, true);
INSTANTIATE_NVFP4_BF16(128, 1, 8, 4, true);
INSTANTIATE_NVFP4_BF16(256, 1, 8, 8, false);
INSTANTIATE_NVFP4_BF16(128, 1, 8, 4, false);
INSTANTIATE_NVFP4_BF16(256, 2, 16, 4, false);
INSTANTIATE_NVFP4_BF16(128, 2, 8, 4, false);
INSTANTIATE_NVFP4_BF16(256, 3, 16, 4, false);
INSTANTIATE_NVFP4_BF16(128, 3, 8, 4, false);
INSTANTIATE_NVFP4_BF16(256, 4, 16, 4, false);
INSTANTIATE_NVFP4_BF16(128, 4, 8, 4, false);

#undef INSTANTIATE_NVFP4_BF16

} // namespace MARLIN_NAMESPACE_NAME

#endif // SUPPORTS_FP8
