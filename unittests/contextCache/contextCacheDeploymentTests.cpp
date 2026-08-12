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

#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

namespace
{

LLMEngineConfig makeAttentionConfig(int32_t attentionLayers = 4)
{
    LLMEngineConfig config;
    config.hiddenSize = 1024;
    config.numDecoderLayers = attentionLayers;
    config.numAttentionLayers = attentionLayers;
    config.numKVHeads = 8;
    config.headDim = 128;
    config.rotaryDim = 128;
    config.maxSupportedBatchSize = 2;
    config.maxKVCacheCapacity = 512;
    int64_t const minimumActivePages
        = computeMinimumKvPoolPages(config.maxSupportedBatchSize, config.maxKVCacheCapacity);
    ELLM_CHECK(minimumActivePages <= kMAX_KV_POOL_PAGES, "Test KV pool page count must fit int32.");
    config.kvPoolPages = static_cast<int32_t>(minimumActivePages);
    config.kvCacheDtype = nvinfer1::DataType::kHALF;
    config.layerTypes.assign(attentionLayers, HybridCacheManager::LayerType::kAttention);
    config.kvLayerConfigs.assign(attentionLayers, KVLayerConfig{config.numKVHeads, config.headDim});
    return config;
}

LLMEngineConfig makeHybridConfig(int32_t attentionLayers = 2, int32_t recurrentLayers = 2)
{
    LLMEngineConfig config = makeAttentionConfig(attentionLayers);
    config.numDecoderLayers = attentionLayers + recurrentLayers;
    config.numLinearAttnLayers = recurrentLayers;
    config.layerTypes.insert(config.layerTypes.end(), recurrentLayers, HybridCacheManager::LayerType::kMamba);
    config.recurrentStateNumHeads = 16;
    config.recurrentStateHeadDim = 32;
    config.recurrentStateSize = 64;
    config.convDim = 256;
    config.convKernel = 4;
    config.recurrentStateDtype = nvinfer1::DataType::kHALF;
    config.convStateDtype = nvinfer1::DataType::kHALF;
    return config;
}

DeploymentConfig makeEagleDeployment()
{
    DeploymentConfig deployment;
    deployment.base = makeAttentionConfig();
    deployment.base.specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.base.isSpecDecodeBase = true;
    deployment.base.specTargetLayerIds = {0, 2, 3};
    deployment.draft = makeAttentionConfig(/*attentionLayers=*/2);
    deployment.draft->specDecodeType = SpecDecodeMode::kEAGLE;
    deployment.draft->baseModelHiddenSize = deployment.base.hiddenSize * 3;
    deployment.specConfig = SpecDecodeConfig{};
    return deployment;
}

} // namespace

TEST(ContextCacheDeploymentTests, ClassifiesSupportedVanillaHybridAndPureRecurrentDeployments)
{
    DeploymentConfig vanilla{makeAttentionConfig(), std::nullopt, std::nullopt};
    EXPECT_EQ(validateContextCacheDeployment(vanilla), ContextCacheDeploymentKind::kVanilla);

    DeploymentConfig hybrid{makeHybridConfig(), std::nullopt, std::nullopt};
    EXPECT_EQ(validateContextCacheDeployment(hybrid), ContextCacheDeploymentKind::kHybrid);

    LLMEngineConfig recurrent = makeHybridConfig(/*attentionLayers=*/0, /*recurrentLayers=*/3);
    DeploymentConfig pureRecurrent{recurrent, std::nullopt, std::nullopt};
    EXPECT_EQ(validateContextCacheDeployment(pureRecurrent), ContextCacheDeploymentKind::kPureRecurrent);
}

TEST(ContextCacheDeploymentTests, RejectsBlockDiffusion)
{
    DeploymentConfig deployment{makeAttentionConfig(), std::nullopt, std::nullopt};
    deployment.base.isDiffusionBackbone = true;

    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, RejectsUnsupportedDtypeAndVisionAttention)
{
    DeploymentConfig deployment{makeAttentionConfig(), std::nullopt, std::nullopt};

    deployment.base.kvCacheDtype = nvinfer1::DataType::kFP8;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment.base = makeAttentionConfig();
    deployment.base.kvCacheDtype = nvinfer1::DataType::kBF16;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment.base = makeAttentionConfig();
    deployment.base.useVisionBidirectionalAttention = true;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, AllowsNonStatefulDecoderLayers)
{
    DeploymentConfig deployment{makeHybridConfig(), std::nullopt, std::nullopt};
    deployment.base.numDecoderLayers += 2;
    EXPECT_NO_THROW(validateContextCacheDeployment(deployment));
}

TEST(ContextCacheDeploymentTests, RejectsMalformedDeploymentTuples)
{
    DeploymentConfig malformedVanilla{makeAttentionConfig(), std::nullopt, SpecDecodeConfig{}};
    EXPECT_THROW(validateContextCacheDeployment(malformedVanilla), std::runtime_error);

    DeploymentConfig malformedEagle = makeEagleDeployment();
    malformedEagle.draft.reset();
    EXPECT_THROW(validateContextCacheDeployment(malformedEagle), std::runtime_error);

    malformedEagle = makeEagleDeployment();
    malformedEagle.base.isSpecDecodeBase = false;
    EXPECT_THROW(validateContextCacheDeployment(malformedEagle), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, RejectsUnsafeKvDonorGraphsAndLayouts)
{
    DeploymentConfig deployment{makeAttentionConfig(/*attentionLayers=*/3), std::nullopt, std::nullopt};
    deployment.base.kvSharingDonors = {-1, 0, 1};
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment.base.kvSharingDonors = {-1, 0, -1};
    deployment.base.kvLayerConfigs[1].headDim += 16;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, RejectsUnmanagedSpecModesAndHybridEagle)
{
    DeploymentConfig mtp = makeEagleDeployment();
    mtp.base.specDecodeType = SpecDecodeMode::kMTP;
    mtp.draft->specDecodeType = SpecDecodeMode::kMTP;
    EXPECT_THROW(validateContextCacheDeployment(mtp), std::runtime_error);

    DeploymentConfig hybridEagle = makeEagleDeployment();
    hybridEagle.base = makeHybridConfig();
    hybridEagle.base.specDecodeType = SpecDecodeMode::kEAGLE;
    hybridEagle.base.isSpecDecodeBase = true;
    EXPECT_THROW(validateContextCacheDeployment(hybridEagle), std::runtime_error);
}

TEST(ContextCacheDeploymentTests, ClassifiesSupportedEagleAndRejectsInvalidConditioning)
{
    DeploymentConfig deployment = makeEagleDeployment();
    EXPECT_EQ(validateContextCacheDeployment(deployment), ContextCacheDeploymentKind::kEAGLE);

    deployment.base.specTargetLayerIds.clear();
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment = makeEagleDeployment();
    deployment.base.specTargetLayerIds = {0, 2, 2};
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment = makeEagleDeployment();
    deployment.base.specTargetLayerIds = {0, 2, 4};
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);

    deployment = makeEagleDeployment();
    deployment.draft->baseModelHiddenSize -= deployment.base.hiddenSize;
    EXPECT_THROW(validateContextCacheDeployment(deployment), std::runtime_error);
}
