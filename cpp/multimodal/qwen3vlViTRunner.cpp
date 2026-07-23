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

#include "qwen3vlViTRunner.h"
#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "kernels/preprocessKernels/imageUtilKernels.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace trt_edgellm
{
namespace rt
{

bool Qwen3VLViTRunner::validateExtraConfig(nlohmann::json const& jsonConfig)
{
    auto visionConfig = jsonConfig["vision_config"];
    auto numPositionEmbeddings = visionConfig["num_position_embeddings"].get<int64_t>();
    mNumGridPerSide = static_cast<int64_t>(std::sqrt(numPositionEmbeddings));
    auto deepstackIndexes = visionConfig.value("deepstack_visual_indexes", std::vector<int64_t>{});
    mNumDeepstackFeatures = deepstackIndexes.size(); // 0 for Qwen3.5 (no deepstack)
    mConfig.mropeInterleaved = true;                 // Qwen3-VL/3.5/Omni use interleaved MRoPE (architectural)
    return true;
}

bool Qwen3VLViTRunner::allocateExtraBuffers(int64_t maxImageTokens)
{
    bool setTensorAddressStatus{true};
    mFastPosEmbIdx = rt::Tensor(
        {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "Qwen3VLViTRunner::mFastPosEmbIdx");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.rawPointer());

    mFastPosEmbWeight = rt::Tensor(
        {4, mConfig.maxHW}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "Qwen3VLViTRunner::mFastPosEmbWeight");
    setTensorAddressStatus
        &= mVisualContext->setTensorAddress(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.rawPointer());

    for (int64_t i = 0; i < mNumDeepstackFeatures; ++i)
    {
        std::string const deepstackFeatureName = binding_names::formatDeepstackFeaturesName(i);
        mDeepstackFeatures.emplace_back(rt::Tensor({maxImageTokens, mConfig.outHiddenSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kHALF, deepstackFeatureName));
        setTensorAddressStatus
            &= mVisualContext->setTensorAddress(deepstackFeatureName.c_str(), mDeepstackFeatures.back().rawPointer());
    }
    return setTensorAddressStatus;
}

void Qwen3VLViTRunner::buildExtraInputs(
    std::vector<VisionSpan> const& spans, int64_t totalSeqLength, int64_t totalImageTokens, cudaStream_t stream)
{
    check::check(mFastPosEmbIdx.reshape({4, totalSeqLength}), "Tensor reshape failed");
    check::check(mFastPosEmbWeight.reshape({4, totalSeqLength}), "Tensor reshape failed");

    // Each span's gridT*gridH*gridW fast-pos-emb entries are written starting at vit.patchStart.
    for (auto const& s : spans)
    {
        kernel::initFastPosEmbedQwenViT(mFastPosEmbIdx, mFastPosEmbWeight, {s.vit.gridT, s.vit.gridH, s.vit.gridW},
            mConfig.mergeSize, mNumGridPerSide, s.vit.patchStart, stream);
    }

    for (int64_t i = 0; i < mNumDeepstackFeatures; ++i)
    {
        check::check(mDeepstackFeatures[i].reshape({totalImageTokens, mConfig.outHiddenSize}), "Tensor reshape failed");
    }
}

bool Qwen3VLViTRunner::bindExtraInputShapes()
{
    bool setEngineIOStatus{true};
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbIdx, mFastPosEmbIdx.getShape().getTRTDims());
    setEngineIOStatus
        &= mVisualContext->setInputShape(binding_names::kFastPosEmbWeight, mFastPosEmbWeight.getShape().getTRTDims());
    return setEngineIOStatus;
}

std::tuple<int64_t, int64_t> Qwen3VLViTRunner::getResizedImageSize(
    int64_t numFrames, int64_t height, int64_t width, int64_t maxRatio)
{
    // Mirrors HF Qwen3-VL smart_resize: 3D temporal-aware budget (t_bar = ceil(N/TPS)*TPS) — the base 2D body plus
    // numFrames counted into the budget.
    int64_t const factor = mConfig.patchSize * mConfig.mergeSize;
    int64_t const minPixels = mConfig.minImageTokensPerImage * factor * factor;
    int64_t const maxPixels = mConfig.maxImageTokensPerImage * factor * factor;

    // Banker's rounding (round-half-to-even) to match Python's round() used by the HF reference.
    auto roundByFactor = [](int64_t value, int64_t f) -> int64_t {
        int64_t q = value / f;
        int64_t r = value - q * f;
        int64_t twoR = 2 * r;
        if (twoR > f || (twoR == f && (q & 1)))
            ++q;
        return q * f;
    };
    auto floorByFactor
        = [](int64_t value, int64_t f) -> int64_t { return std::floor(static_cast<double>(value) / f) * f; };
    auto ceilByFactor
        = [](int64_t value, int64_t f) -> int64_t { return std::ceil(static_cast<double>(value) / f) * f; };

    ELLM_CHECK(std::max(height, width) / std::min(height, width) <= maxRatio,
        "absolute aspect ratio must be smaller than " + std::to_string(maxRatio) + ", got "
            + std::to_string(std::max(height, width) / std::min(height, width)));

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    int64_t tBar = (numFrames > 1) ? ceilByFactor(numFrames, mConfig.temporalPatchSize) : 1;

    int64_t const budget = tBar * hBar * wBar;
    if (budget > maxPixels)
    {
        double beta = std::sqrt(static_cast<double>(numFrames * height * width) / maxPixels);
        hBar = std::max(factor, floorByFactor(static_cast<int64_t>(height / beta), factor));
        wBar = std::max(factor, floorByFactor(static_cast<int64_t>(width / beta), factor));
    }
    else if (budget < minPixels)
    {
        double beta = std::sqrt(static_cast<double>(minPixels) / (numFrames * height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {hBar, wBar};
}

std::tuple<int64_t, int64_t> Qwen3VLViTRunner::computeVisionSpans(
    rt::imageUtils::ImageData const& image, int64_t patchBase, std::vector<VisionSpan>& spans)
{
    int64_t const height = image.height;
    int64_t const width = image.width;
    ELLM_CHECK(
        height % (mConfig.patchSize * mConfig.mergeSize) == 0 && width % (mConfig.patchSize * mConfig.mergeSize) == 0,
        "Image height or width is not divisible by patchSize * mergeSize = "
            + std::to_string(mConfig.patchSize * mConfig.mergeSize) + " got height: " + std::to_string(height)
            + ", width: " + std::to_string(width));

    int64_t const gridT = (image.frames + mConfig.temporalPatchSize - 1) / mConfig.temporalPatchSize;
    int64_t const gridH = height / mConfig.patchSize; // HF grid_h (patch units)
    int64_t const gridW = width / mConfig.patchSize;  // HF grid_w (patch units)
    int64_t const patchesPerFrame = gridH * gridW;
    int64_t const llmGridH = gridH / mConfig.mergeSize; // HF llm_grid_h
    int64_t const llmGridW = gridW / mConfig.mergeSize; // HF llm_grid_w
    int64_t const tokensPerFrame = llmGridH * llmGridW;

    // One frame -> single flat span (== the per-frame loop at gridT==1); image-vs-video is decided in textPreprocess.
    if (image.frames <= 1)
    {
        VisionSpan span;
        span.llm
            = LlmVisionBlock{tokensPerFrame, /*llmGridT*/ 1, llmGridH, llmGridW}; // secondPerGrid unused (gridT==1)
        span.vit = VitFrameGrid{/*gridT*/ 1, gridH, gridW, patchBase};
        spans.push_back(span);
        return {/*totalSeqLen*/ patchesPerFrame, /*totalGridT*/ 1};
    }

    // Video: per-frame sub-spans, each gridT==1.
    // The <X.X seconds> timestamp marker is a textPreprocess concern (recomputed there from frames/fps).
    int64_t patch = patchBase;
    for (int64_t t = 0; t < gridT; ++t)
    {
        VisionSpan span;
        span.llm = LlmVisionBlock{tokensPerFrame, /*llmGridT*/ 1, llmGridH, llmGridW};
        span.vit = VitFrameGrid{/*gridT*/ 1, gridH, gridW, patch};
        spans.push_back(span);
        patch += patchesPerFrame;
    }
    // gridT sub-spans, each 1 frame of patchesPerFrame patches.
    return {/*totalSeqLen*/ gridT * patchesPerFrame, /*totalGridT*/ gridT};
}

rt::OptionalInputTensors Qwen3VLViTRunner::getDeepstackFeatures()
{
    if (mNumDeepstackFeatures == 0) // Qwen3.5: no deepstack
    {
        return {};
    }
    std::vector<std::reference_wrapper<rt::Tensor const>> refs;
    refs.reserve(mDeepstackFeatures.size());
    for (auto const& tensor : mDeepstackFeatures)
    {
        refs.emplace_back(std::cref(tensor));
    }
    return refs;
}

void Qwen3VLViTRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<VisionSpan> const& spans,
    trt_edgellm::tokenizer::Tokenizer const* tokenizer)
{
    // Pads expand to incrementing IDs (>= vocabSize; embeddingLookupWithImageInsertion). Two paths:
    //   (a) VIDEO: the <|vision_start|><|video_pad|><|vision_end|> triplet -> one timestamped (<X.X s> + vision_start
    //       + pads + vision_end) group per per-frame sub-span. Detected data-driven (next span carries a timestamp).
    //   (b) IMAGE (and any non-video pad): one flat pad run.
    int32_t nextImageTokenId = mConfig.vocabSize;
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

        auto const& imgBuffers = request.requests[i].imageBuffers;
        size_t bufferIdx = 0; // per-request image-buffer cursor (for the video sub-span count)
        std::vector<int32_t> newIds;
        for (size_t j = 0; j < ids.size(); ++j)
        {
            bool const findVideoTriplet = j + 2 < ids.size() && ids[j] == mConfig.visionStartTokenId
                && ids[j + 1] == mConfig.videoTokenId && ids[j + 2] == mConfig.visionEndTokenId;
            if (findVideoTriplet)
            {
                int64_t const tps = mConfig.temporalPatchSize;
                ELLM_CHECK(bufferIdx < imgBuffers.size(),
                    "Video triplet found but no matching image buffer at index " + std::to_string(bufferIdx));
                int64_t const frames = imgBuffers[bufferIdx].frames;
                double const fps = imgBuffers[bufferIdx].fps;
                int64_t const gridT = (frames + tps - 1) / tps;
                for (int64_t t = 0; t < gridT; ++t)
                {
                    ELLM_CHECK(spanIdx < spans.size(),
                        "Pad token found but no matching vision span at index " + std::to_string(spanIdx));
                    LlmVisionBlock const& block = spans[spanIdx++].llm;
                    // Frame-pair midpoint time. Matches HF (frames_indices/video_fps) only when frames are uniformly
                    // sampled from frame 0 and fps is the post-sampling rate (we renumber 0..frames-1, no
                    // frames_indices).
                    int64_t const firstIdx = std::min(t * tps, frames - 1);
                    int64_t const lastIdx = std::min(t * tps + tps - 1, frames - 1);
                    double const ts = static_cast<double>(firstIdx + lastIdx) / 2.0 / fps;
                    char tsBuf[32];
                    std::snprintf(tsBuf, sizeof(tsBuf), "<%.1f seconds>", ts);
                    std::vector<int32_t> tsTokens = tokenizer->encode(std::string(tsBuf));
                    newIds.insert(newIds.end(), tsTokens.begin(), tsTokens.end());

                    newIds.push_back(mConfig.visionStartTokenId);
                    for (int64_t k = 0; k < block.numTokens; ++k)
                    {
                        newIds.push_back(nextImageTokenId++);
                    }
                    newIds.push_back(mConfig.visionEndTokenId);
                }
                ++bufferIdx;
                j += 2; // consumed video_pad + vision_end
            }
            else if (ids[j] == mConfig.imageTokenId || ids[j] == mConfig.videoTokenId)
            {
                ELLM_CHECK(spanIdx < spans.size(),
                    "Pad token found but no matching vision span at index " + std::to_string(spanIdx));
                LlmVisionBlock const& block = spans[spanIdx++].llm;
                for (int64_t k = 0; k < block.numTokens; ++k)
                {
                    newIds.push_back(nextImageTokenId++);
                }
                ++bufferIdx;
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
