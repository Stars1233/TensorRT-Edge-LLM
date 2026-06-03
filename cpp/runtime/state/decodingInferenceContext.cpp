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

#include "runtime/state/decodingInferenceContext.h"

namespace trt_edgellm
{
namespace rt
{

void DecodingInferenceContext::initialize(int32_t batchSize, int32_t maxGenLength,
    rt::OptionalInputTensor const& visual, rt::OptionalInputTensors const& deepstack, std::string const& loraName,
    cudaStream_t cudaStream)
{
    systemPrompts.resize(batchSize);
    rawBatchedInputIds.reserve(batchSize);
    tokenIds.resize(batchSize);
    currentGenerateLengths.resize(batchSize, 0);
    effectivePrefillLengths.resize(batchSize, 0);
    finishedStates.resize(batchSize, 0);
    slotStreams.clear();
    slotStreams.resize(batchSize);
    stopStringsPerSlot.clear();
    stopStringsPerSlot.resize(batchSize);

    batchIndexMapping.resize(batchSize);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        batchIndexMapping[i] = i;
    }

    completedBatches.clear();

    visualEmbeddings = visual;
    deepstackFeatures = deepstack;
    generationRound = 0;
    maxGenerateLength = maxGenLength;
    activeBatchSize = batchSize;
    loraWeightsName = loraName;
    stream = cudaStream;
}

} // namespace rt
} // namespace trt_edgellm
