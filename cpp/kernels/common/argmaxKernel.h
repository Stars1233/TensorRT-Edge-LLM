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

/*!
 * \brief Row-wise top-1 (argmax) over a ``[rows, cols]`` row-major tensor.
 *
 * ``outIndices[r] = argmax_c input[r, c]``. Ties resolve to the LOWEST column
 * index, matching ``torch.argmax`` — required by callers that must reproduce a
 * PyTorch greedy reference (RNN-T greedy decode, EAGLE target argmax).
 *
 * One block per row; a warp-shuffle + shared-memory reduction inside the block.
 * The comparison is done in ``float`` (``T`` loads are widened), so the result
 * is independent of reduction order and bit-stable across launch shapes.
 *
 * Instantiated for ``T`` in {``__half``, ``float``}.
 *
 * \param input      Device ``[rows, cols]`` row-major, dtype ``T``.
 * \param rows       Number of rows (blocks launched).
 * \param cols       Row width (reduction extent).
 * \param outIndices Device ``[rows]`` int32, receives the argmax column per row.
 * \param stream     CUDA stream.
 */
template <typename T>
void invokeRowwiseArgmax(T const* input, int32_t rows, int32_t cols, int32_t* outIndices, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
