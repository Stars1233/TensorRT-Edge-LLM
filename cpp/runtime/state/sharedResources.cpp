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

#include "runtime/state/sharedResources.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

void allocateZeroBuffer(SharedResources& res, int64_t bytes)
{
    res.zeroBuffer = Tensor({bytes}, DeviceType::kGPU, nvinfer1::DataType::kUINT8, "SharedResources::zeroBuffer");
    CUDA_CHECK(cudaMemset(res.zeroBuffer.rawPointer(), 0, res.zeroBuffer.getMemoryCapacity()));
}

std::unique_ptr<SharedResources> SharedResources::createForLLM(
    LLMEngineConfig const& cfg, std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    auto resources = std::make_unique<SharedResources>();

    // Hybrid cache manager (attention KV + Mamba recurrent + conv states).
    // KVLayerConfig list is sized and populated by parseEngineConfig; its length
    // equals the attention-layer count and is indexed by LOCAL attention index.
    rt::KVCacheManager::Config kvCfg{
        /*.numAttentionLayers=*/static_cast<int32_t>(cfg.kvLayerConfigs.size()),
        /*.maxBatchSize=*/cfg.maxSupportedBatchSize,
        /*.maxSequenceLength=*/cfg.maxKVCacheCapacity,
        /*.layerConfigs=*/cfg.kvLayerConfigs,
        /*.kvCacheType=*/cfg.kvCacheDtype,
    };
    rt::MambaCacheManager::Config mambaCfg{
        /*.numRecurrentLayers=*/cfg.numLinearAttnLayers,
        /*.maxBatchSize=*/cfg.maxSupportedBatchSize,
        /*.recurrentStateNumHeads=*/cfg.recurrentStateNumHeads,
        /*.recurrentStateHeadDim=*/cfg.recurrentStateHeadDim,
        /*.recurrentStateSize=*/cfg.recurrentStateSize,
        /*.convDim=*/cfg.convDim,
        /*.convKernel=*/cfg.convKernel,
        /*.maxIntermediateSeqLen=*/0,
        /*.recurrentStateType=*/cfg.recurrentStateDtype,
        /*.convStateType=*/cfg.convStateDtype,
    };
    rt::HybridCacheManager::Config hybridCfg{
        /*.layerTypes=*/cfg.layerTypes,
        /*.kvConfig=*/std::move(kvCfg),
        /*.mambaConfig=*/std::move(mambaCfg),
        /*.maxBatchSize=*/cfg.maxSupportedBatchSize,
    };
    resources->cacheManagers.push_back(std::make_unique<HybridCacheManager>(hybridCfg, stream));

    // RoPE cache
    // For MRope, the cache is stored in PipelineIO (initialized below).
    // For LongRope, we initialize the pool entry manually.
    // For standard / dynamic / NoRope, getOrCreate handles everything.
    if (cfg.ropeConfig.type == RopeType::kLongRope)
    {
        // Pre-populate the RoPE pool entry, then reinitialize with the correct LongRope data.
        rt::Tensor& ropeEntry
            = resources->ropePool.getOrCreate(cfg.ropeConfig, cfg.rotaryDim, cfg.maxKVCacheCapacity, stream);
        // getOrCreate allocated the tensor but initializeRopeCosSinCache fails for LongRope.
        // Re-initialize with the LongRope-specific function.
        check::check(
            cfg.ropeConfig.longRope.has_value() && cfg.ropeConfig.longRope.value().originalMaxPositionEmbeddings != -1,
            "longRope is not set correctly");

        rt::Tensor shortCosSinCache = rt::Tensor({1, cfg.maxKVCacheCapacity, cfg.rotaryDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "LLMInferenceRuntime::shortCosSinCache");
        rt::Tensor longCosSinCache = rt::Tensor({1, cfg.maxKVCacheCapacity, cfg.rotaryDim}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "LLMInferenceRuntime::longCosSinCache");
        bool const initRopeStatus
            = initializeLongRopeCosSinCache(shortCosSinCache, longCosSinCache, cfg.ropeConfig, stream);
        check::check(initRopeStatus, "Failed to initialize long Rope CosSinCache.");
        // Choose short or long based on max KV capacity vs original max position embeddings.
        if (cfg.maxKVCacheCapacity <= cfg.ropeConfig.longRope.value().originalMaxPositionEmbeddings)
        {
            ropeEntry = std::move(shortCosSinCache);
        }
        else
        {
            ropeEntry = std::move(longCosSinCache);
        }
    }
    else if (cfg.ropeConfig.type != RopeType::kMRope)
    {
        // Standard / Dynamic / NoRope — getOrCreate handles initialization.
        resources->ropePool.getOrCreate(cfg.ropeConfig, cfg.rotaryDim, cfg.maxKVCacheCapacity, stream);
    }
    // MRope: handled via PipelineIO::mropeCosSin below.

    // LoRA
    if (cfg.maxSupportedLoraRank > 0)
    {
        resources->loraManager = std::make_unique<LoRAManager>();
        for (auto const& [loraWeightsName, loraWeightsPath] : loraWeightsMap)
        {
            if (loraWeightsPath.empty())
            {
                continue;
            }
            resources->loraManager->loadWeights(loraWeightsName, loraWeightsPath, stream);
        }
    }

    // Externalized model weights are immutable base-engine inputs. The manager
    // is constructed here, but file loading and engine validation are deferred
    // to runtime initialization with the loaded EngineExecutor.
    resources->externalWeightManager = std::make_unique<ExternalWeightManager>();

    // Zero buffer — sized to the largest shape any consumer binds:
    //  * deepstack_embeds_* during non-prefill phases
    //    (batch × seqLen × hiddenSize HALF; seqLen = maxVerifyTreeSize for SpecDecode, else 1)
    //  * attention mask / pos ids on paths that still need a dummy bind
    //  * LoRA dummy weights on resetWeights paths
    // kvcache_start_index is now a registry-bound tensor and does not need a dummy.
    {
        int64_t const maxBatch = cfg.maxSupportedBatchSize;
        int64_t const deepstackSeqLen = cfg.isSpecDecodeBase ? std::max(cfg.maxVerifyTreeSize, 1) : 1;
        int64_t deepstackSize = 0;
        if (cfg.numDeepstackFeatures > 0)
        {
            deepstackSize = maxBatch * deepstackSeqLen * cfg.hiddenSize * static_cast<int64_t>(sizeof(uint16_t));
        }
        int64_t const zeroBufferBytes = std::max(deepstackSize, static_cast<int64_t>(256));
        allocateZeroBuffer(*resources, zeroBufferBytes);
    }

    return resources;
}

std::unique_ptr<SharedResources> SharedResources::createForSpecDecode(DeploymentConfig const& bundle,
    int32_t maxRuntimeBatchSize, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    cudaStream_t stream)
{
    check::check(
        bundle.draft.has_value(), "SharedResources::createForSpecDecode requires DeploymentConfig.draft to be set");
    check::check(bundle.specConfig.has_value(),
        "SharedResources::createForSpecDecode requires DeploymentConfig.specConfig to be set");

    auto resources = std::make_unique<SharedResources>();

    int32_t const maxDraftProposalSize = bundle.specConfig->maxDraftProposalSize;

    // Base hybrid cache manager (index 0). EAGLE3 base is pure-attention but
    // an MTP base can be a hybrid model (e.g. Qwen3.5 MTP, 18 mamba layers),
    // so size the Mamba sub-manager from the base's parsed config.
    // MTP base needs intermediate-state slots sized to the verification
    // budget so the accepted recurrent/conv snapshot can be committed after
    // base verification.
    {
        int32_t const baseMaxIntermediateSeqLen
            = (bundle.base.specDecodeType == SpecDecodeMode::kMTP && bundle.base.numLinearAttnLayers > 0)
            ? bundle.specConfig->maxVerifySize
            : 0;
        rt::KVCacheManager::Config kvCfg{
            /*.numAttentionLayers=*/static_cast<int32_t>(bundle.base.kvLayerConfigs.size()),
            /*.maxBatchSize=*/bundle.base.maxSupportedBatchSize,
            /*.maxSequenceLength=*/bundle.base.maxKVCacheCapacity,
            /*.layerConfigs=*/bundle.base.kvLayerConfigs,
            /*.kvCacheType=*/bundle.base.kvCacheDtype,
        };
        rt::MambaCacheManager::Config mambaCfg{
            /*.numRecurrentLayers=*/bundle.base.numLinearAttnLayers,
            /*.maxBatchSize=*/bundle.base.maxSupportedBatchSize,
            /*.recurrentStateNumHeads=*/bundle.base.recurrentStateNumHeads,
            /*.recurrentStateHeadDim=*/bundle.base.recurrentStateHeadDim,
            /*.recurrentStateSize=*/bundle.base.recurrentStateSize,
            /*.convDim=*/bundle.base.convDim,
            /*.convKernel=*/bundle.base.convKernel,
            /*.maxIntermediateSeqLen=*/baseMaxIntermediateSeqLen,
            /*.recurrentStateType=*/bundle.base.recurrentStateDtype,
            /*.convStateType=*/bundle.base.convStateDtype,
        };
        rt::HybridCacheManager::Config hybridCfg{
            /*.layerTypes=*/bundle.base.layerTypes,
            /*.kvConfig=*/std::move(kvCfg),
            /*.mambaConfig=*/std::move(mambaCfg),
            /*.maxBatchSize=*/bundle.base.maxSupportedBatchSize,
        };
        resources->cacheManagers.push_back(std::make_unique<HybridCacheManager>(hybridCfg, stream));
    }

    // Draft hybrid cache manager (index 1). Same pattern: pure-attention.
    {
        rt::KVCacheManager::Config kvCfg{
            /*.numAttentionLayers=*/static_cast<int32_t>(bundle.draft->kvLayerConfigs.size()),
            /*.maxBatchSize=*/bundle.draft->maxSupportedBatchSize,
            /*.maxSequenceLength=*/bundle.draft->maxKVCacheCapacity,
            /*.layerConfigs=*/bundle.draft->kvLayerConfigs,
            /*.kvCacheType=*/bundle.draft->kvCacheDtype,
        };
        rt::MambaCacheManager::Config mambaCfg{
            /*.numRecurrentLayers=*/0,
            /*.maxBatchSize=*/bundle.draft->maxSupportedBatchSize,
            /*.recurrentStateNumHeads=*/0,
            /*.recurrentStateHeadDim=*/0,
            /*.recurrentStateSize=*/0,
            /*.convDim=*/0,
            /*.convKernel=*/0,
            /*.maxIntermediateSeqLen=*/0,
            /*.recurrentStateType=*/nvinfer1::DataType::kHALF,
            /*.convStateType=*/nvinfer1::DataType::kHALF,
        };
        rt::HybridCacheManager::Config hybridCfg{
            /*.layerTypes=*/bundle.draft->layerTypes,
            /*.kvConfig=*/std::move(kvCfg),
            /*.mambaConfig=*/std::move(mambaCfg),
            /*.maxBatchSize=*/bundle.draft->maxSupportedBatchSize,
        };
        resources->cacheManagers.push_back(std::make_unique<HybridCacheManager>(hybridCfg, stream));
    }

    // RoPE cache (shared — base and draft use same RoPE config)
    if (bundle.base.ropeConfig.type != RopeType::kMRope)
    {
        resources->ropePool.getOrCreate(
            bundle.base.ropeConfig, bundle.base.rotaryDim, bundle.base.maxKVCacheCapacity, stream);
    }

    // LoRA
    if (bundle.base.maxSupportedLoraRank > 0)
    {
        resources->loraManager = std::make_unique<LoRAManager>();
        for (auto const& [loraWeightsName, loraWeightsPath] : loraWeightsMap)
        {
            if (loraWeightsPath.empty())
            {
                continue;
            }
            resources->loraManager->loadWeights(loraWeightsName, loraWeightsPath, stream);
        }
    }

    // Externalized model weights currently apply to the base engine only.
    // The manager is constructed here, but file loading and engine validation
    // are deferred to runtime initialization with the loaded EngineExecutor.
    resources->externalWeightManager = std::make_unique<ExternalWeightManager>();

    // Zero buffer — sized to the largest shape any consumer binds:
    //  * deepstack_embeds_* during non-prefill phases
    //    (maxRuntimeBatchSize × verifySize × hiddenSize HALF)
    //  * attention_pos_id dummy shapes during vanilla fallback on a SpecDecode runtime
    {
        int64_t const maxBatch = maxRuntimeBatchSize;
        int64_t const verifySeqLen = bundle.specConfig->verifySize;
        int64_t deepstackSize = 0;
        if (bundle.base.numDeepstackFeatures > 0)
        {
            deepstackSize = maxBatch * verifySeqLen * bundle.base.hiddenSize * static_cast<int64_t>(sizeof(uint16_t));
        }
        int64_t const attnPosIdSize
            = maxBatch * std::max(maxDraftProposalSize, 1) * static_cast<int64_t>(sizeof(int32_t));
        int64_t const zeroBufferBytes = std::max({deepstackSize, attnPosIdSize, static_cast<int64_t>(256)});
        allocateZeroBuffer(*resources, zeroBufferBytes);
    }

    return resources;
}

} // namespace rt
} // namespace trt_edgellm
