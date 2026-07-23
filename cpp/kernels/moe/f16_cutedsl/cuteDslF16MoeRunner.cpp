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

#if defined(CUTE_DSL_F16_MOE_ENABLED)

#include "cuteDslF16MoeRunner.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "kernels/moe/f16MoeSupportKernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

namespace trt_edgellm
{

namespace detail
{

thread_local cudaError_t gCuteDslF16MoeCudaError{cudaSuccess};

void recordCuteDslF16MoeCudaError(cudaError_t error) noexcept
{
    if (error != cudaSuccess && gCuteDslF16MoeCudaError == cudaSuccess)
    {
        gCuteDslF16MoeCudaError = error;
    }
}

void clearCuteDslF16MoeCudaError() noexcept
{
    gCuteDslF16MoeCudaError = cudaSuccess;
}

cudaError_t getCuteDslF16MoeCudaError() noexcept
{
    return gCuteDslF16MoeCudaError;
}

} // namespace detail

#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
f16_moe_ampere_grouped_fp16_Kernel_Module_t CuteDslF16MoeRunner::sAmpereModule{};
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
f16_moe_blackwell_grouped_fp16_Kernel_Module_t CuteDslF16MoeRunner::sBlackwellModule{};
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
f16_moe_blackwell_geforce_grouped_fp16_Kernel_Module_t CuteDslF16MoeRunner::sBlackwellGeforceModule{};
#endif
CuteDslF16MoeRunner::Variant CuteDslF16MoeRunner::sActiveVariant{CuteDslF16MoeRunner::Variant::kNone};
std::unordered_map<int32_t, int32_t> CuteDslF16MoeRunner::sPersistentBlockCounts;
bool CuteDslF16MoeRunner::sLoaded{false};
std::mutex CuteDslF16MoeRunner::sLoadMutex;

namespace
{

constexpr int32_t kACT_SWIGLU{2};
constexpr int32_t kACT_RELU2{4};
#if !defined(CUTE_DSL_F16_MOE_ARTIFACT_SM)
#error "CUTE_DSL_F16_MOE_ARTIFACT_SM must identify the linked target artifact"
#endif
constexpr int32_t kARTIFACT_SM{CUTE_DSL_F16_MOE_ARTIFACT_SM};
constexpr size_t kDEVICE_ALIGNMENT{256};
constexpr int32_t kMAX_PERSISTENT_BLOCKS{256};
constexpr int32_t kTENSORMAPS_PER_BLOCK{3};
constexpr int32_t kBYTES_PER_TENSORMAP{128};
constexpr size_t kPROBLEM_SHAPE_VALUES_PER_EXPERT{4};
constexpr size_t kSTRIDE_VALUES_PER_EXPERT{6};
constexpr size_t kADDRESS_VALUES_PER_EXPERT{3};

size_t alignUp(size_t value, size_t alignment)
{
    if (alignment == 0 || value > std::numeric_limits<size_t>::max() - (alignment - 1))
    {
        throw std::overflow_error("FP16 MoE workspace alignment overflow");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t checkedAdd(size_t lhs, size_t rhs)
{
    if (lhs > std::numeric_limits<size_t>::max() - rhs)
    {
        throw std::overflow_error("FP16 MoE workspace size addition overflow");
    }
    return lhs + rhs;
}

size_t checkedMultiply(size_t lhs, size_t rhs)
{
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
    {
        throw std::overflow_error("FP16 MoE workspace size multiplication overflow");
    }
    return lhs * rhs;
}

size_t matrixBytes(int32_t rows, int32_t columns)
{
    return checkedMultiply(checkedMultiply(static_cast<size_t>(rows), static_cast<size_t>(columns)), sizeof(__half));
}

struct WorkspaceLayout
{
    size_t expertCounts{};
    size_t expertOffsets{};
    size_t expertWriteOffsets{};
    size_t sortedToExpanded{};
    size_t expandedToSorted{};
    size_t fc1ProblemShapes{};
    size_t fc1Strides{};
    size_t fc1Addresses{};
    size_t fc2ProblemShapes{};
    size_t fc2Strides{};
    size_t fc2Addresses{};
    size_t tensormapWorkspace{};
    size_t gatheredInput{};
    size_t rawFc1Output{};
    size_t activatedFc1Output{};
    size_t routedFc2Output{};
    size_t total{};
};

WorkspaceLayout buildWorkspaceLayout(
    int32_t routedRows, int32_t numExperts, int32_t hiddenSize, int32_t moeInterSize, int32_t activationType)
{
    if (routedRows <= 0 || numExperts <= 0 || hiddenSize <= 0 || moeInterSize <= 0
        || moeInterSize > std::numeric_limits<int32_t>::max() / 2
        || (activationType != kACT_SWIGLU && activationType != kACT_RELU2))
    {
        throw std::invalid_argument("FP16 MoE workspace dimensions must be positive and representable");
    }
    int32_t const fc1N = activationType == kACT_SWIGLU ? 2 * moeInterSize : moeInterSize;

    WorkspaceLayout layout{};
    size_t cursor{};
    auto place = [&cursor](size_t& offset, size_t bytes) {
        offset = cursor;
        cursor = alignUp(checkedAdd(cursor, bytes), kDEVICE_ALIGNMENT);
    };

    size_t const numExpertsSize = static_cast<size_t>(numExperts);
    size_t const problemShapesBytes
        = checkedMultiply(checkedMultiply(numExpertsSize, kPROBLEM_SHAPE_VALUES_PER_EXPERT), sizeof(int32_t));
    size_t const stridesBytes
        = checkedMultiply(checkedMultiply(numExpertsSize, kSTRIDE_VALUES_PER_EXPERT), sizeof(int32_t));
    size_t const addressesBytes
        = checkedMultiply(checkedMultiply(numExpertsSize, kADDRESS_VALUES_PER_EXPERT), sizeof(int64_t));

    place(layout.expertCounts, checkedMultiply(numExpertsSize, sizeof(int32_t)));
    place(layout.expertOffsets, checkedMultiply(checkedAdd(numExpertsSize, 1), sizeof(int32_t)));
    place(layout.expertWriteOffsets, checkedMultiply(numExpertsSize, sizeof(int32_t)));
    place(layout.sortedToExpanded, checkedMultiply(static_cast<size_t>(routedRows), sizeof(int32_t)));
    place(layout.expandedToSorted, checkedMultiply(static_cast<size_t>(routedRows), sizeof(int32_t)));
    place(layout.fc1ProblemShapes, problemShapesBytes);
    place(layout.fc1Strides, stridesBytes);
    place(layout.fc1Addresses, addressesBytes);
    place(layout.fc2ProblemShapes, problemShapesBytes);
    place(layout.fc2Strides, stridesBytes);
    place(layout.fc2Addresses, addressesBytes);
    place(layout.tensormapWorkspace,
        static_cast<size_t>(kMAX_PERSISTENT_BLOCKS) * kTENSORMAPS_PER_BLOCK * kBYTES_PER_TENSORMAP);
    place(layout.gatheredInput, matrixBytes(routedRows, hiddenSize));
    place(layout.rawFc1Output, matrixBytes(routedRows, fc1N));
    place(layout.activatedFc1Output, matrixBytes(routedRows, moeInterSize));
    place(layout.routedFc2Output, matrixBytes(routedRows, hiddenSize));
    layout.total = alignUp(cursor, kDEVICE_ALIGNMENT);
    return layout;
}

void* offsetPointer(void* base, size_t offset)
{
    return static_cast<std::byte*>(base) + offset;
}

bool checkCuda(cudaError_t error, char const* operation) noexcept
{
    if (error == cudaSuccess)
    {
        return true;
    }
    LOG_ERROR("CuteDslF16MoeRunner: %s failed: %s (%s)", operation, cudaGetErrorName(error), cudaGetErrorString(error));
    return false;
}

#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
bool isAmpereSm(int32_t smVersion)
{
    return smVersion == 80 || smVersion == 86 || smVersion == 87 || smVersion == 89;
}
#endif

#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
bool isBlackwellSm(int32_t smVersion)
{
    return smVersion == 100 || smVersion == 101 || smVersion == 103 || smVersion == 110;
}
#endif

#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
bool isBlackwellGeforceSm(int32_t smVersion)
{
    return smVersion == 120 || smVersion == 121;
}
#endif

} // namespace

bool CuteDslF16MoeRunner::canImplement(int32_t hiddenSize, int32_t moeInterSize, int32_t numExperts, int32_t topK,
    int32_t smVersion, int32_t activationType) noexcept
{
    int64_t const fc1N = activationType == kACT_SWIGLU ? 2LL * moeInterSize : moeInterSize;
    bool architectureAvailable{false};
#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
    architectureAvailable = architectureAvailable || isAmpereSm(smVersion);
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
    architectureAvailable = architectureAvailable || isBlackwellSm(smVersion);
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
    architectureAvailable = architectureAvailable || isBlackwellGeforceSm(smVersion);
#endif
    return architectureAvailable && smVersion == kARTIFACT_SM && isSupportedNumExperts(numExperts) && topK > 0
        && topK <= kMaxTopK && hiddenSize > 0 && hiddenSize % kHiddenSizeAlignment == 0 && moeInterSize > 0
        && moeInterSize % kInterSizeAlignment == 0 && fc1N <= std::numeric_limits<int32_t>::max()
        && fc1N % kFc1NAlignment == 0 && (activationType == kACT_SWIGLU || activationType == kACT_RELU2);
}

bool CuteDslF16MoeRunner::loadKernelModules() noexcept
{
    try
    {
        std::lock_guard<std::mutex> lock(sLoadMutex);
        int32_t device{};
        cudaDeviceProp properties{};
        if (!checkCuda(cudaGetDevice(&device), "query current device")
            || !checkCuda(cudaGetDeviceProperties(&properties, device), "query device properties"))
        {
            return false;
        }
        int32_t const smVersion = properties.major * 10 + properties.minor;
        if (smVersion != kARTIFACT_SM)
        {
            LOG_ERROR("CuteDslF16MoeRunner: linked SM%d artifact cannot execute on SM%d", kARTIFACT_SM, smVersion);
            return false;
        }
        Variant requestedVariant{Variant::kNone};
#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
        if (isAmpereSm(smVersion))
        {
            requestedVariant = Variant::kAmpere;
        }
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
        if (isBlackwellSm(smVersion))
        {
            requestedVariant = Variant::kBlackwell;
        }
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
        if (isBlackwellGeforceSm(smVersion))
        {
            requestedVariant = Variant::kBlackwellGeforce;
        }
#endif
        if (requestedVariant == Variant::kNone)
        {
            LOG_ERROR("CuteDslF16MoeRunner: no compiled f16_moe AOT variant matches SM%d", smVersion);
            return false;
        }

        if (sLoaded)
        {
            if (sActiveVariant != requestedVariant)
            {
                LOG_ERROR("CuteDslF16MoeRunner: the linked artifact pack cannot mix architecture families");
                return false;
            }
            sPersistentBlockCounts.try_emplace(
                device, std::min(kMAX_PERSISTENT_BLOCKS, properties.multiProcessorCount));
            return true;
        }

        detail::clearCuteDslF16MoeCudaError();
#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
        if (requestedVariant == Variant::kAmpere)
        {
            f16_moe_ampere_grouped_fp16_Kernel_Module_Load(&sAmpereModule);
            sActiveVariant = Variant::kAmpere;
        }
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
        if (requestedVariant == Variant::kBlackwell)
        {
            f16_moe_blackwell_grouped_fp16_Kernel_Module_Load(&sBlackwellModule);
            sActiveVariant = Variant::kBlackwell;
        }
#endif
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
        if (requestedVariant == Variant::kBlackwellGeforce)
        {
            f16_moe_blackwell_geforce_grouped_fp16_Kernel_Module_Load(&sBlackwellGeforceModule);
            sActiveVariant = Variant::kBlackwellGeforce;
        }
#endif
        cudaError_t const loadError = detail::getCuteDslF16MoeCudaError();
        if (loadError != cudaSuccess)
        {
            LOG_ERROR("CuteDslF16MoeRunner: AOT module load failed: %s (%s)", cudaGetErrorName(loadError),
                cudaGetErrorString(loadError));
            detail::clearCuteDslF16MoeCudaError();
            switch (requestedVariant)
            {
            case Variant::kAmpere:
#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
                if (sAmpereModule.module != nullptr)
                {
                    f16_moe_ampere_grouped_fp16_Kernel_Module_Unload(&sAmpereModule);
                    sAmpereModule = {};
                }
#endif
                break;
            case Variant::kBlackwell:
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
                if (sBlackwellModule.module != nullptr)
                {
                    f16_moe_blackwell_grouped_fp16_Kernel_Module_Unload(&sBlackwellModule);
                    sBlackwellModule = {};
                }
#endif
                break;
            case Variant::kBlackwellGeforce:
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
                if (sBlackwellGeforceModule.module != nullptr)
                {
                    f16_moe_blackwell_geforce_grouped_fp16_Kernel_Module_Unload(&sBlackwellGeforceModule);
                    sBlackwellGeforceModule = {};
                }
#endif
                break;
            case Variant::kNone: break;
            }
            cudaError_t const cleanupError = detail::getCuteDslF16MoeCudaError();
            if (cleanupError != cudaSuccess)
            {
                LOG_ERROR("CuteDslF16MoeRunner: failed-load cleanup failed: %s (%s)", cudaGetErrorName(cleanupError),
                    cudaGetErrorString(cleanupError));
            }
            sActiveVariant = Variant::kNone;
            return false;
        }
        int32_t const persistentBlockCount = std::min(kMAX_PERSISTENT_BLOCKS, properties.multiProcessorCount);
        if (persistentBlockCount <= 0)
        {
            LOG_ERROR("CuteDslF16MoeRunner: CUDA device %d has no multiprocessors", device);
            sActiveVariant = Variant::kNone;
            return false;
        }
        sPersistentBlockCounts.emplace(device, persistentBlockCount);
        sLoaded = true;
        LOG_DEBUG("CuteDslF16MoeRunner: loaded variant %d for SM%d with %d persistent blocks",
            static_cast<int32_t>(sActiveVariant), smVersion, persistentBlockCount);
        return true;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("CuteDslF16MoeRunner module load failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("CuteDslF16MoeRunner module load failed: unknown error");
    }
    return false;
}

int32_t CuteDslF16MoeRunner::getPersistentBlockCount() noexcept
{
    if (!loadKernelModules())
    {
        return 0;
    }
    std::lock_guard<std::mutex> lock(sLoadMutex);
    int32_t device{};
    if (!checkCuda(cudaGetDevice(&device), "query current device for persistent block count"))
    {
        return 0;
    }
    auto const found = sPersistentBlockCounts.find(device);
    return found == sPersistentBlockCounts.end() ? 0 : found->second;
}

size_t CuteDslF16MoeRunner::getWorkspaceSize(int32_t maxRoutedRows, int32_t numExperts, int32_t hiddenSize,
    int32_t moeInterSize, int32_t activationType) noexcept
{
    try
    {
        return buildWorkspaceLayout(maxRoutedRows, numExperts, hiddenSize, moeInterSize, activationType).total;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("CuteDslF16MoeRunner workspace calculation failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("CuteDslF16MoeRunner workspace calculation failed: unknown error");
    }
    return 0;
}

int32_t CuteDslF16MoeRunner::run(CuteDslF16MoeParams const& params, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        if (params.persistentBlockCount <= 0 || params.persistentBlockCount > kMAX_PERSISTENT_BLOCKS)
        {
            LOG_ERROR("CuteDslF16MoeRunner: invalid cached persistent block count %d", params.persistentBlockCount);
            return -1;
        }
        if (workspace == nullptr || params.hiddenStates == nullptr || params.topkIds == nullptr
            || params.topkWeights == nullptr || params.fc1Weights == nullptr || params.fc2Weights == nullptr
            || params.output == nullptr)
        {
            LOG_ERROR("CuteDslF16MoeRunner: null input, output, or workspace pointer");
            return -1;
        }
        int64_t const fc1N64 = params.activationType == kACT_SWIGLU ? 2LL * params.moeInterSize : params.moeInterSize;
        int32_t const smVersion = getSMVersion();
        if (!canImplement(params.hiddenSize, params.moeInterSize, params.numExperts, params.topK, smVersion,
                params.activationType)
            || params.numTokens <= 0 || params.numTokens > std::numeric_limits<int32_t>::max() / params.topK)
        {
            LOG_ERROR("CuteDslF16MoeRunner: invalid launch configuration");
            return -1;
        }

        int32_t const routedRows = params.numTokens * params.topK;
        int32_t const fc1N = static_cast<int32_t>(fc1N64);
        WorkspaceLayout const layout = buildWorkspaceLayout(
            routedRows, params.numExperts, params.hiddenSize, params.moeInterSize, params.activationType);
        auto* const expertCounts = static_cast<int32_t*>(offsetPointer(workspace, layout.expertCounts));
        auto* const expertOffsets = static_cast<int32_t*>(offsetPointer(workspace, layout.expertOffsets));
        auto* const expertWriteOffsets = static_cast<int32_t*>(offsetPointer(workspace, layout.expertWriteOffsets));
        auto* const sortedToExpanded = static_cast<int32_t*>(offsetPointer(workspace, layout.sortedToExpanded));
        auto* const expandedToSorted = static_cast<int32_t*>(offsetPointer(workspace, layout.expandedToSorted));
        auto* const fc1ProblemShapes = static_cast<int32_t*>(offsetPointer(workspace, layout.fc1ProblemShapes));
        auto* const fc1Strides = static_cast<int32_t*>(offsetPointer(workspace, layout.fc1Strides));
        auto* const fc1Addresses = static_cast<int64_t*>(offsetPointer(workspace, layout.fc1Addresses));
        auto* const fc2ProblemShapes = static_cast<int32_t*>(offsetPointer(workspace, layout.fc2ProblemShapes));
        auto* const fc2Strides = static_cast<int32_t*>(offsetPointer(workspace, layout.fc2Strides));
        auto* const fc2Addresses = static_cast<int64_t*>(offsetPointer(workspace, layout.fc2Addresses));
        void* const tensormapWorkspace = offsetPointer(workspace, layout.tensormapWorkspace);
        void* const gatheredInput = offsetPointer(workspace, layout.gatheredInput);
        void* const rawFc1Output = offsetPointer(workspace, layout.rawFc1Output);
        void* const activatedFc1Output = offsetPointer(workspace, layout.activatedFc1Output);
        void* const routedFc2Output = offsetPointer(workspace, layout.routedFc2Output);

        kernel::F16MoeRoutingBuffers const routing{
            expertCounts, expertOffsets, expertWriteOffsets, sortedToExpanded, expandedToSorted};
        kernel::F16MoeGemmMetadata const fc1Metadata{fc1ProblemShapes, fc1Strides, fc1Addresses};
        kernel::F16MoeGemmMetadata const fc2Metadata{fc2ProblemShapes, fc2Strides, fc2Addresses};
        kernel::F16MoeGemmSetup const fc1Setup{
            fc1Metadata, gatheredInput, params.fc1Weights, rawFc1Output, fc1N, params.hiddenSize};
        kernel::F16MoeGemmSetup const fc2Setup{fc2Metadata, activatedFc1Output, params.fc2Weights, routedFc2Output,
            params.hiddenSize, params.moeInterSize};
        if (!checkCuda(kernel::buildF16MoeRoutingAndGemmMetadata(routing, fc1Setup, fc2Setup, params.topkIds,
                           params.numTokens, params.topK, params.numExperts, stream),
                "build routing and grouped-GEMM metadata")
            || !checkCuda(kernel::gatherF16MoeHiddenRows(params.hiddenStates, gatheredInput, sortedToExpanded,
                              routedRows, params.topK, params.hiddenSize, stream),
                "gather hidden rows"))
        {
            return -1;
        }

        auto launchGrouped = [&](kernel::F16MoeGemmMetadata const& metadata, void* input, void const* weights,
                                 void* output, int32_t n, int32_t k, char const* stage) -> bool {
            int32_t result{-1};
            switch (sActiveVariant)
            {
            case Variant::kAmpere:
#if defined(CUTE_DSL_F16_MOE_AMPERE_ENABLED)
                result = cute_dsl_f16_moe_ampere_grouped_fp16_wrapper(&sAmpereModule, input, const_cast<void*>(weights),
                    output, metadata.problemShapes, metadata.strides, metadata.addresses, tensormapWorkspace,
                    routedRows, n, k, params.numExperts, params.persistentBlockCount, stream);
#endif
                break;
            case Variant::kBlackwell:
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_ENABLED)
                result = cute_dsl_f16_moe_blackwell_grouped_fp16_wrapper(&sBlackwellModule, input,
                    const_cast<void*>(weights), output, metadata.problemShapes, metadata.strides, metadata.addresses,
                    tensormapWorkspace, routedRows, n, k, params.numExperts, params.persistentBlockCount, stream);
#endif
                break;
            case Variant::kBlackwellGeforce:
#if defined(CUTE_DSL_F16_MOE_BLACKWELL_GEFORCE_ENABLED)
                result = cute_dsl_f16_moe_blackwell_geforce_grouped_fp16_wrapper(&sBlackwellGeforceModule, input,
                    const_cast<void*>(weights), output, metadata.problemShapes, metadata.strides, metadata.addresses,
                    tensormapWorkspace, routedRows, n, k, params.numExperts, params.persistentBlockCount, stream);
#endif
                break;
            case Variant::kNone: break;
            }
            if (result != 0)
            {
                LOG_ERROR("CuteDslF16MoeRunner: %s AOT wrapper failed with code %d", stage, result);
                return false;
            }
            return checkCuda(cudaGetLastError(), stage);
        };

        if (!launchGrouped(
                fc1Metadata, gatheredInput, params.fc1Weights, rawFc1Output, fc1N, params.hiddenSize, "grouped FC1")
            || !checkCuda(kernel::activateF16Moe(rawFc1Output, activatedFc1Output, routedRows, params.moeInterSize,
                              params.activationType, stream),
                "FC1 activation")
            || !launchGrouped(fc2Metadata, activatedFc1Output, params.fc2Weights, routedFc2Output, params.hiddenSize,
                params.moeInterSize, "grouped FC2")
            || !checkCuda(kernel::scatterF16MoeOutput(routedFc2Output, expandedToSorted, params.topkWeights,
                              params.output, params.numTokens, params.topK, params.hiddenSize, stream),
                "weighted output scatter"))
        {
            return -1;
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        LOG_ERROR("CuteDslF16MoeRunner failed: %s", error.what());
    }
    catch (...)
    {
        LOG_ERROR("CuteDslF16MoeRunner failed: unknown error");
    }
    return -1;
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_F16_MOE_ENABLED)
