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

#include "runtime/state/contextCache/contextCacheDeployment.h"

#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"

#include <algorithm>
#include <string>
#include <vector>

namespace trt_edgellm::rt
{
namespace
{

void validateStateContract(LLMEngineConfig const& config, char const* label)
{
    int32_t const attentionCount = static_cast<int32_t>(
        std::count(config.layerTypes.begin(), config.layerTypes.end(), HybridCacheManager::LayerType::kAttention));
    int32_t const recurrentCount = static_cast<int32_t>(config.layerTypes.size()) - attentionCount;
    ELLM_CHECK(static_cast<int32_t>(config.layerTypes.size()) == config.numAttentionLayers + config.numLinearAttnLayers,
        std::string(label) + " layer_types must describe every stateful attention/recurrent layer.");
    ELLM_CHECK(attentionCount == config.numAttentionLayers,
        std::string(label) + " attention layer count does not match layer_types.");
    ELLM_CHECK(recurrentCount == config.numLinearAttnLayers,
        std::string(label) + " recurrent layer count does not match layer_types.");
    ELLM_CHECK(static_cast<int32_t>(config.kvLayerConfigs.size()) == config.numAttentionLayers,
        std::string(label) + " kv_layer_configs must describe every attention layer.");
    ELLM_CHECK(config.kvSharingDonors.empty()
            || static_cast<int32_t>(config.kvSharingDonors.size()) == config.numAttentionLayers,
        std::string(label) + " kv_sharing_donors must be empty or match the attention layer count.");

    if (config.numAttentionLayers > 0)
    {
        ELLM_CHECK(config.kvCacheDtype == nvinfer1::DataType::kHALF,
            std::string(label)
                + " uses a KV dtype outside the supported non-identity page-table boundary (FP16 KV only).");
        // SWA changes the kernel read mask, not physical retention: context reuse requires a full logical allocation
        // for every attention layer so cached pages remain valid when rebound across requests.
        int64_t const minimumActivePages
            = computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity);
        ELLM_CHECK(config.kvPoolPages >= minimumActivePages && config.kvPoolPages <= kMAX_KV_POOL_PAGES,
            std::string(label) + " has invalid context-cache pool geometry.");
    }

    if (!config.kvSharingDonors.empty())
    {
        for (int32_t recipient = 0; recipient < config.numAttentionLayers; ++recipient)
        {
            int32_t const donor = config.kvSharingDonors[recipient];
            if (donor < 0)
            {
                continue;
            }
            ELLM_CHECK(donor < config.numAttentionLayers && donor != recipient,
                std::string(label) + " has an invalid KV donor index.");
            ELLM_CHECK(
                config.kvSharingDonors[donor] == -1, std::string(label) + " KV donor chains/cycles are not supported.");
            ELLM_CHECK(config.kvLayerConfigs[donor].numKVHeads == config.kvLayerConfigs[recipient].numKVHeads
                    && config.kvLayerConfigs[donor].headDim == config.kvLayerConfigs[recipient].headDim,
                std::string(label) + " KV donor and recipient layouts must match.");
        }
    }

    if (config.numLinearAttnLayers > 0)
    {
        ELLM_CHECK(config.recurrentStateNumHeads > 0 && config.recurrentStateHeadDim > 0
                && config.recurrentStateSize > 0 && config.convDim > 0 && config.convKernel > 0,
            std::string(label) + " has an incomplete recurrent-state schema.");
    }
    ELLM_CHECK(!config.useVisionBidirectionalAttention,
        std::string(label) + " uses vision-bidirectional attention, which is not managed by the context cache.");
}

} // namespace

ContextCacheDeploymentKind validateContextCacheDeployment(DeploymentConfig const& deployment)
{
    ELLM_CHECK(!deployment.base.isDiffusionBackbone,
        "Block Diffusion context reuse is outside the current context-reuse support boundary.");
    validateStateContract(deployment.base, "base engine");

    switch (deployment.base.specDecodeType)
    {
    case SpecDecodeMode::kNONE:
        ELLM_CHECK(
            !deployment.base.isSpecDecodeBase && !deployment.draft.has_value() && !deployment.specConfig.has_value(),
            "Vanilla context-cache deployment requires an llm-role base and no draft/speculative configuration.");
        ELLM_CHECK(deployment.base.numAttentionLayers > 0 || deployment.base.numLinearAttnLayers > 0,
            "Context-cache deployment has no stateful layers.");
        if (deployment.base.numLinearAttnLayers == 0)
        {
            return ContextCacheDeploymentKind::kVanilla;
        }
        if (deployment.base.numAttentionLayers == 0)
        {
            return ContextCacheDeploymentKind::kPureRecurrent;
        }
        return ContextCacheDeploymentKind::kHybrid;

    case SpecDecodeMode::kEAGLE:
    {
        ELLM_CHECK(deployment.base.isSpecDecodeBase && deployment.draft.has_value() && deployment.specConfig.has_value()
                && deployment.draft->specDecodeType == SpecDecodeMode::kEAGLE && !deployment.draft->isSpecDecodeBase,
            "EAGLE context-cache deployment requires matching base-role/draft-role engines and speculative "
            "configuration.");
        validateStateContract(*deployment.draft, "draft engine");
        ELLM_CHECK(deployment.base.numLinearAttnLayers == 0 && deployment.draft->numLinearAttnLayers == 0,
            "Hybrid EAGLE context reuse is outside the current context-reuse support boundary.");
        ELLM_CHECK(deployment.base.numAttentionLayers > 0 && deployment.draft->numAttentionLayers > 0,
            "EAGLE context reuse requires page-backed base and draft KV caches.");
        ELLM_CHECK(deployment.draft->hasOwnKVCache && !deployment.draft->sharesTargetKV,
            "EAGLE context reuse requires a draft engine with its own independent KV cache.");
        ELLM_CHECK(!deployment.base.specTargetLayerIds.empty(),
            "EAGLE context reuse requires an explicit base conditioning-layer contract.");
        {
            std::vector<int32_t> sortedTargetLayers = deployment.base.specTargetLayerIds;
            std::sort(sortedTargetLayers.begin(), sortedTargetLayers.end());
            ELLM_CHECK(sortedTargetLayers.front() >= 0 && sortedTargetLayers.back() < deployment.base.numDecoderLayers
                    && std::adjacent_find(sortedTargetLayers.begin(), sortedTargetLayers.end())
                        == sortedTargetLayers.end(),
                "EAGLE conditioning layer IDs must be unique and within the base decoder-layer range.");
        }
        int64_t const expectedConditioningSize = static_cast<int64_t>(deployment.base.hiddenSize)
            * static_cast<int64_t>(deployment.base.specTargetLayerIds.size());
        ELLM_CHECK(deployment.draft->baseModelHiddenSize == expectedConditioningSize,
            "EAGLE base_model_hidden_size does not match the base hidden size and conditioning-layer count.");
        return ContextCacheDeploymentKind::kEAGLE;
    }

    case SpecDecodeMode::kMTP:
    case SpecDecodeMode::kDFlash:
    case SpecDecodeMode::kGemma4MTP:
    case SpecDecodeMode::kDSpark:
        ELLM_CHECK(false, "Context reuse does not support MTP, DFlash, DSpark, or Gemma4 MTP deployments.");
    }
    ELLM_CHECK(false, "Unknown speculative decoding mode in context-cache deployment validation.");
}

} // namespace trt_edgellm::rt
