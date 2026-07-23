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

#include "qwen3omniViTRunner.h"
#include "common/checkMacros.h"
#include <algorithm>

namespace trt_edgellm
{
namespace rt
{

std::tuple<int64_t, int64_t> Qwen3OmniViTRunner::computeVisionSpans(
    rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans)
{
    // Reuse the base flat layout, then fill the fps-derived secondPerGrid. Omni's
    // getMRopePositionIds applies the position_id_per_seconds scale + T-extent advance.
    auto const extent = QwenViTRunner::computeVisionSpans(image, patchBase, spans);
    spans.back().llm.secondPerGrid = static_cast<int64_t>(mConfig.temporalPatchSize / image.fps);
    return extent;
}

std::tuple<int64_t, int64_t> Qwen3OmniViTRunner::getResizedImageSize(
    int64_t numFrames, int64_t height, int64_t width, int64_t maxRatio)
{
    // Omni uses the base 2D per-frame budget, NOT Qwen3-VL's 3D.
    return QwenViTRunner::getResizedImageSize(numFrames, height, width, maxRatio);
}

bool Qwen3OmniViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    // Omni shares Qwen3-VL's ViT config (num_position_embeddings, deepstack), plus position_id_per_seconds for MRoPE.
    if (!Qwen3VLViTRunner::validateExtraConfig(jsonConfig))
    {
        return false;
    }
    auto visionConfig = jsonConfig["vision_config"];
    // Qwen3-Omni video MRoPE temporal interval = position_id_per_seconds * int(temporal_patch_size / fps).
    if (visionConfig.contains("position_id_per_seconds"))
    {
        mPositionIdPerSecond = visionConfig["position_id_per_seconds"].get<int64_t>();
    }
    else if (jsonConfig.contains("position_id_per_seconds"))
    {
        mPositionIdPerSecond = jsonConfig["position_id_per_seconds"].get<int64_t>();
    }
    else
    {
        LOG_ERROR("Qwen3-Omni requires position_id_per_seconds (in vision_config or top-level) for video MRoPE");
        return false;
    }
    return true;
}

void Qwen3OmniViTRunner::getMRopePositionIds(
    std::vector<std::vector<int32_t>> const& batchInputIds, std::vector<VisionSpan> const& spans) noexcept
{
    // Mirrors HF Qwen3-Omni get_rope_index: per-frame temporal step = secondPerGrid * position_id_per_seconds;
    // span advance = st_idx = max(all span positions)+1.
    int64_t* mropePositionIdsPtr = mMropePositionIdsHost.dataPointer<int64_t>();
    int64_t const maxPositionEmbeddings = mMropePositionIdsHost.getShape()[2];
    int64_t totalImageIdx = 0;
    int64_t batchOffset = 0;

    mMropeRopeDeltasPerBatch.clear();
    for (auto const& inputIds : batchInputIds)
    {
        auto start = inputIds.begin();
        auto end = inputIds.end();
        auto it = inputIds.begin();
        int64_t startIdx = 0;
        int64_t remainingStartPos = 0;

        while ((it = std::find(start, end, mConfig.visionStartTokenId)) != end)
        {
            int64_t textLen = it + 1 - start;
            for (int64_t i = 0; i < 3; ++i)
            {
                for (int64_t j = 0; j < textLen; ++j)
                {
                    mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
                }
            }

            LlmVisionBlock const& block = spans[totalImageIdx].llm;
            int64_t const llmGridT = block.llmGridT;
            int64_t const llmGridH = block.llmGridH;
            int64_t const llmGridW = block.llmGridW;
            int64_t const timeInterval = block.secondPerGrid * mPositionIdPerSecond; // fps-aware temporal step
            ++totalImageIdx;

            for (int64_t t = 0; t < llmGridT; ++t)
            {
                for (int64_t h = 0; h < llmGridH; ++h)
                {
                    for (int64_t w = 0; w < llmGridW; ++w)
                    {
                        int64_t idx = remainingStartPos + textLen + t * llmGridH * llmGridW + h * llmGridW + w;
                        mropePositionIdsPtr[batchOffset + 0 * maxPositionEmbeddings + idx]
                            = t * timeInterval + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 1 * maxPositionEmbeddings + idx] = h + textLen + startIdx;
                        mropePositionIdsPtr[batchOffset + 2 * maxPositionEmbeddings + idx] = w + textLen + startIdx;
                    }
                }
            }

            start = it + 1 + llmGridT * llmGridH * llmGridW;
            // Advance counts the temporal extent too (unlike the base/Qwen2.5 spatial-only advance).
            int64_t const advance = std::max(std::max(llmGridH, llmGridW), (llmGridT - 1) * timeInterval + 1);
            startIdx += advance + textLen;
            remainingStartPos = start - inputIds.begin();
        }

        int64_t const maxMropePositionId = startIdx + inputIds.size() - remainingStartPos - 1;
        mMropeRopeDeltasPerBatch.push_back(maxMropePositionId + 1 - inputIds.size());

        int64_t textLen = maxPositionEmbeddings - remainingStartPos;
        for (int64_t i = 0; i < 3; ++i)
        {
            for (int64_t j = 0; j < textLen; ++j)
            {
                mropePositionIdsPtr[batchOffset + i * maxPositionEmbeddings + remainingStartPos + j] = j + startIdx;
            }
        }

        batchOffset += 3 * maxPositionEmbeddings;
    }
}

void Qwen3OmniViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<VisionSpan> const& spans,
    trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    // Qwen3-Omni: flat-only (computeVisionSpans delegates to the base flat layout, so spans never carry timestamps)
    // and every visual pad is filled with a CONSTANT imageTokenId — embeddingLookupMultimodal inserts the visual
    // features at those positions. No incrementing IDs, no <X.X s> markers, no video-triplet expansion.
    size_t spanIdx = 0;
    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids;
        if (i < batchInputIds.size() && !batchInputIds[i].empty())
        {
            ids = batchInputIds[i]; // already tokenized by another runner
        }
        else
        {
            ids = tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        }

        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            if (ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
            {
                ELLM_CHECK(spanIdx < spans.size(),
                    "Pad token found but no matching vision span at index " + std::to_string(spanIdx));
                LlmVisionBlock const& block = spans[spanIdx++].llm;
                for (int64_t k = 0; k < block.numTokens; ++k)
                {
                    newIds.push_back(mConfig.imageTokenId);
                }
            }
            else
            {
                newIds.push_back(ids[j]);
            }
        }

        if (i < batchInputIds.size())
        {
            batchInputIds[i] = std::move(newIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(newIds));
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
