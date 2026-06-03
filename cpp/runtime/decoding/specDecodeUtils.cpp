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

#include "runtime/decoding/specDecodeUtils.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "runtime/config/llmEngineConfig.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace spec_decode_utils
{

char const* isGreedyCompatible(LLMGenerationRequest const& request) noexcept
{
    constexpr float kSamplingEpsilon{1e-3f};
    bool const hasNonGreedySampling
        = (request.topK > 1 || request.topP < 1.0f || std::fabs(request.temperature - 1.0f) > kSamplingEpsilon);
    if (hasNonGreedySampling)
    {
        return "speculative decoding currently supports greedy-compatible sampling only";
    }
    return nullptr;
}

std::unique_ptr<EngineExecutor> loadDraftEngine(std::filesystem::path const& engineDir, DeploymentConfig& deployment)
{
    std::filesystem::path const draftEnginePath = engineDir / "eagle_draft.engine";
    std::unique_ptr<EngineExecutor> draftExecutor;
    try
    {
        draftExecutor = EngineExecutor::createForSpecDecodeDraft(draftEnginePath, deployment);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize draft EngineExecutor: %s", e.what());
        throw std::runtime_error("Failed to initialize draft EngineExecutor: " + std::string(e.what()));
    }
    LOG_INFO("Draft EngineExecutor successfully loaded from %s.", draftEnginePath.c_str());
    validateAgainstEngine(*deployment.draft, *draftExecutor, "draft");
    return draftExecutor;
}

void appendAcceptedTokens(DecodingInferenceContext& context, Tensor& hostAcceptLengths, Tensor& hostAcceptedTokenIds,
    Tensor const& deviceAcceptLength, Tensor const& deviceAcceptedTokenIds, int32_t maxAcceptDepth,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t stream)
{
    int32_t const activeBatchSize = context.activeBatchSize;

    check::check(hostAcceptLengths.reshape({activeBatchSize}), "Tensor reshape failed");
    check::check(hostAcceptedTokenIds.reshape({activeBatchSize, maxAcceptDepth}), "Tensor reshape failed");
    int32_t* hostAcceptLengthsData = hostAcceptLengths.dataPointer<int32_t>();
    int32_t* hostAcceptedTokenIdsData = hostAcceptedTokenIds.dataPointer<int32_t>();

    CUDA_CHECK(cudaMemcpyAsync(hostAcceptLengthsData, deviceAcceptLength.rawPointer(),
        activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(hostAcceptedTokenIdsData, deviceAcceptedTokenIds.rawPointer(),
        activeBatchSize * maxAcceptDepth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int32_t batchIdx = 0; batchIdx < activeBatchSize; ++batchIdx)
    {
        int32_t const acceptLength = hostAcceptLengthsData[batchIdx];
        for (int32_t i = 0; i < acceptLength; i++)
        {
            int32_t const token = hostAcceptedTokenIdsData[batchIdx * maxAcceptDepth + i];
            context.tokenIds[batchIdx].push_back(token);
            context.currentGenerateLengths[batchIdx]++;
            if (token == tokenizer.getEosId())
            {
                break;
            }
        }
    }
}

} // namespace spec_decode_utils
} // namespace rt
} // namespace trt_edgellm
