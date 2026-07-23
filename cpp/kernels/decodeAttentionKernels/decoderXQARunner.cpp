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

#include "decoderXQARunner.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "xqa_kernel_cubin.h"

#include <algorithm>
#include <cuda.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

using XQADataType = xqa::kernels::Data_type;
using XQAKernelMetaInfo = xqa::kernels::XQAKernelMetaInfo;
using XQAKernelVariant = XQAKernelMetaInfo::XQAKernelVariant;

namespace
{

constexpr uint32_t kHEAD_DIM_512{512U};
constexpr uint32_t kSPLIT_HEAD_DIM_512_CLUSTER_SIZE{2U};
constexpr uint32_t kSPLIT_HEAD_DIM_512_CTA_DIM_Z{2U};
constexpr uint32_t kDEFAULT_XQA_CTA_DIM_Z{2U};
constexpr uint32_t kDEFAULT_XQA_CTA_DIM_X{128U};
constexpr uint32_t kHEAD_DIM_512_CTA_DIM_X{256U};

//! @throws std::runtime_error if datatype is unsupported
XQADataType trtToXqaDataType(nvinfer1::DataType type)
{
    XQADataType xqaType{XQADataType::DATA_TYPE_FP16};
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: xqaType = XQADataType::DATA_TYPE_FP32; break;
    case nvinfer1::DataType::kHALF: xqaType = XQADataType::DATA_TYPE_FP16; break;
    case nvinfer1::DataType::kBF16: xqaType = XQADataType::DATA_TYPE_BF16; break;
    case nvinfer1::DataType::kFP8: xqaType = XQADataType::DATA_TYPE_E4M3; break;
    default: throw std::runtime_error("Unsupported datatype for XQA.");
    }
    return xqaType;
}
struct XQAKernelLoadHashKey
{
    XQADataType data_type;
    XQADataType kv_data_type;
    int32_t sm;
    bool specDecode;
    bool pagedKVCache;

    bool operator==(XQAKernelLoadHashKey const& other) const noexcept
    {
        return data_type == other.data_type && kv_data_type == other.kv_data_type && sm == other.sm
            && specDecode == other.specDecode && pagedKVCache == other.pagedKVCache;
    }
};

struct XQAKernelLoadHasher
{
    size_t operator()(XQAKernelLoadHashKey const& s) const noexcept
    {
        size_t key = s.data_type;
        key <<= 16;
        key ^= s.kv_data_type;
        key <<= 16;
        key ^= s.sm;
        key <<= 4;
        key ^= s.specDecode;
        key <<= 4;
        key ^= s.pagedKVCache;
        return key;
    }
};

struct XQAKernelRuntimeHashKey
{
    XQADataType q_data_type;
    XQADataType kv_data_type;
    int32_t head_size;
    int32_t num_q_heads_per_kv;
    int32_t beam_size;
    bool sliding_window;
    int32_t tokens_per_page;

    bool operator==(XQAKernelRuntimeHashKey const& other) const noexcept
    {
        return q_data_type == other.q_data_type && kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size
            && sliding_window == other.sliding_window && tokens_per_page == other.tokens_per_page;
    }
};

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQALaunchParams const& xqaParams) noexcept
{
    constexpr int32_t kBEAM_SIZE{1};
    int32_t numQHeadPerKV = xqaParams.numQheads / xqaParams.numKVheads;
    return {trtToXqaDataType(xqaParams.dataType), trtToXqaDataType(xqaParams.kvDataType), xqaParams.headSize,
        numQHeadPerKV, kBEAM_SIZE, xqaParams.slidingWinSize > 0, static_cast<int32_t>(xqaParams.kvCache.tokensPerPage)};
}

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParamsSpecDecode(XQALaunchParams const& xqaParams) noexcept
{
    constexpr int32_t kBEAM_SIZE{1};
    constexpr int32_t kQHEAD_PER_KV = 0; // Tree attention kernel supports any ratio of Q/KV heads.
    return {trtToXqaDataType(xqaParams.dataType), trtToXqaDataType(xqaParams.kvDataType), xqaParams.headSize,
        kQHEAD_PER_KV, kBEAM_SIZE, xqaParams.slidingWinSize > 0, static_cast<int32_t>(xqaParams.kvCache.tokensPerPage)};
}

struct XQAKernelRuntimeHasher
{
    size_t operator()(XQAKernelRuntimeHashKey const& s) const noexcept
    {
        size_t key = s.q_data_type;
        key <<= 16;
        key ^= s.kv_data_type;
        key <<= 16;
        key ^= s.head_size;
        key <<= 8;
        key ^= s.num_q_heads_per_kv;
        key <<= 8;
        key ^= s.beam_size;
        key <<= 4;
        key ^= s.sliding_window;
        key <<= 8;
        key ^= s.tokens_per_page;
        return key;
    }
};

struct XQAKernelFuncInfo
{
    uint32_t mSharedMemBytes{0};
    CUfunction mDeviceFunction{0};
    uint32_t mHeadDim{0};
    uint32_t mMTileSize{0};
    uint32_t mSMVersion{0};
    XQAKernelVariant mKernelVariant{XQAKernelMetaInfo::KERNEL_VARIANT_STANDARD};
    bool mRequiresClusterLaunch{false};
    bool mRequiresDistributedSharedMemory{false};
    bool mSlidingWindow{false};
};

struct XQADeviceCapability
{
    uint32_t mMaxSharedMemPerBlockOptin{0};
    bool mSupportsClusterLaunch{false};
    bool mSupportsDistributedSharedMemory{false};
};

struct KernelContiguousKVCache
{
    void* data{nullptr};
    int32_t const* sequenceLengths{nullptr};
    uint32_t capacity{0};
};

struct KernelPagedKVCache
{
    void* data{nullptr};
    int32_t const* pageList{nullptr};
    int32_t const* sequenceLengths{nullptr};
    uint32_t maxNbPagesPerSeq{0};
};

using KernelKVCacheArg = std::variant<KernelContiguousKVCache, KernelPagedKVCache>;

bool usePagedKVCache(XQALaunchParams::KVCache const& kvCache) noexcept
{
    return kvCache.tokensPerPage != 0;
}

bool isPowerOfTwo(uint32_t value) noexcept
{
    return value > 0U && (value & (value - 1U)) == 0U;
}

uint32_t getMaxNbPagesPerSeq(XQALaunchParams::KVCache const& kvCache)
{
    check::check(kvCache.tokensPerPage > 0, "Paged KV cache requires non-zero tokensPerPage.");
    check::check(isPowerOfTwo(kvCache.tokensPerPage), "Paged KV cache tokensPerPage must be a power of 2.");
    check::check(
        kvCache.capacity % kvCache.tokensPerPage == 0U, "Paged KV cache capacity must be divisible by tokensPerPage.");
    return kvCache.capacity / kvCache.tokensPerPage;
}

KernelKVCacheArg makeKernelKVCacheArg(XQALaunchParams::KVCache const& kvCache, bool usePaged)
{
    if (!usePaged)
    {
        return KernelContiguousKVCache{kvCache.data, kvCache.sequence_lengths, kvCache.capacity};
    }

    return KernelPagedKVCache{kvCache.data, kvCache.pageList, kvCache.sequence_lengths, getMaxNbPagesPerSeq(kvCache)};
}

void* getKernelKVCacheArg(KernelKVCacheArg& kvCacheArg)
{
    return std::visit([](auto& kernelKVCache) -> void* { return &kernelKVCache; }, kvCacheArg);
}

XQADeviceCapability getDeviceCapability()
{
    constexpr int32_t kDEVICE_ID{0};
    CUdevice cuDevice{};
    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice, kDEVICE_ID));

    int32_t maxSharedMemPerBlockOptin{0};
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(
        &maxSharedMemPerBlockOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice));
    check::check(maxSharedMemPerBlockOptin > 0, "Failed to get max shared memory per block opt-in.");

    bool supportsClusterLaunch{false};
#if SUPPORTS_CLUSTER_LAUNCH
    int32_t clusterLaunch{0};
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&clusterLaunch, CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, cuDevice));
    supportsClusterLaunch = clusterLaunch != 0;
#endif // SUPPORTS_CLUSTER_LAUNCH

    return {static_cast<uint32_t>(maxSharedMemPerBlockOptin), supportsClusterLaunch, supportsClusterLaunch};
}

bool isKernelSupportedByBuild(XQAKernelMetaInfo const& kernelMeta) noexcept
{
#if SUPPORTS_CLUSTER_LAUNCH
    (void) kernelMeta;
    return true;
#else
    return !kernelMeta.mRequiresClusterLaunch && !kernelMeta.mRequiresDistributedSharedMemory;
#endif // SUPPORTS_CLUSTER_LAUNCH
}

bool isKernelCompatibleWithDevice(
    XQAKernelFuncInfo const& kernelInfo, XQADeviceCapability const& deviceCapability) noexcept
{
    if (kernelInfo.mSharedMemBytes > deviceCapability.mMaxSharedMemPerBlockOptin)
    {
        return false;
    }
    if (kernelInfo.mRequiresClusterLaunch && !deviceCapability.mSupportsClusterLaunch)
    {
        return false;
    }
    if (kernelInfo.mRequiresDistributedSharedMemory && !deviceCapability.mSupportsDistributedSharedMemory)
    {
        return false;
    }
    return true;
}

uint32_t getKernelVariantPriority(XQAKernelVariant variant) noexcept
{
    // Lower priority value is preferred when multiple compatible candidates exist for the same runtime key.
    switch (variant)
    {
    case XQAKernelMetaInfo::KERNEL_VARIANT_STANDARD: return 0U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512: return 0U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_FULL_SMEM_HEAD_DIM512_ROW_MAX_METHOD4: return 1U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_2CTA_HEAD_DIM512: return 2U;
    case XQAKernelMetaInfo::KERNEL_VARIANT_TILED_QKV_STAGING_HEAD_DIM512: return 3U;
    }
    return 3U;
}

uint32_t getCtaDimX(XQAKernelFuncInfo const& kernelInfo) noexcept
{
    return kernelInfo.mHeadDim == kHEAD_DIM_512 && !kernelInfo.mRequiresClusterLaunch ? kHEAD_DIM_512_CTA_DIM_X
                                                                                      : kDEFAULT_XQA_CTA_DIM_X;
}

#if SUPPORTS_CLUSTER_LAUNCH
void launch2CtaHeadDim512ClusterKernel(XQAKernelFuncInfo const& kernelInfo, dim3 const& dimGrid, dim3 const& dimCta,
    cudaStream_t const& stream, void** kernelParams)
{
    CUlaunchAttribute launchAttr{};
    launchAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launchAttr.value.clusterDim.x = kSPLIT_HEAD_DIM_512_CLUSTER_SIZE;
    launchAttr.value.clusterDim.y = 1U;
    launchAttr.value.clusterDim.z = 1U;

    CUlaunchConfig launchConfig{};
    launchConfig.gridDimX = dimGrid.x;
    launchConfig.gridDimY = dimGrid.y;
    launchConfig.gridDimZ = dimGrid.z;
    launchConfig.blockDimX = dimCta.x;
    launchConfig.blockDimY = dimCta.y;
    launchConfig.blockDimZ = dimCta.z;
    launchConfig.sharedMemBytes = kernelInfo.mSharedMemBytes;
    launchConfig.hStream = stream;
    launchConfig.attrs = &launchAttr;
    launchConfig.numAttrs = 1U;

    CUDA_DRIVER_CHECK(cuLaunchKernelEx(&launchConfig, kernelInfo.mDeviceFunction, kernelParams, nullptr));
}
#endif // SUPPORTS_CLUSTER_LAUNCH

bool hasHeadDim512KernelsForSM(
    int32_t smVersion, XQADataType kvDataType, int32_t headRatio, bool usePagedKVCache) noexcept
{
    bool hasDecodeKernel{false};
    bool hasSpecDecodeKernel{false};
    for (auto const& kernelMeta : xqa::kernels::sXqaKernelMetaInfo)
    {
        if (kernelMeta.mSM != static_cast<unsigned int>(smVersion)
            || kernelMeta.mDataType != XQADataType::DATA_TYPE_FP16 || kernelMeta.mKVDataType != kvDataType
            || kernelMeta.mHeadDim != kHEAD_DIM_512 || kernelMeta.mPagedKVCache != usePagedKVCache
            || !isKernelSupportedByBuild(kernelMeta))
        {
            continue;
        }
        if (kernelMeta.mMultiQueryTokens)
        {
            // Spec-decode tree attention kernels support any Q/KV head ratio.
            hasSpecDecodeKernel = true;
        }
        else if (kernelMeta.mNumQHeadsOverKV == static_cast<unsigned int>(headRatio))
        {
            hasDecodeKernel = true;
        }
    }
    return hasDecodeKernel && hasSpecDecodeKernel;
}

class XQAKernelList
{
    using TKernelMetaInfo = xqa::kernels::XQAKernelMetaInfo;

public:
    XQAKernelList(
        XQADataType dataType, XQADataType kvDataType, uint32_t sm, bool specDecode, bool pagedKVCache) noexcept
        : mKernelMeta(nullptr)
        , mKernelMetaCount(0)
        , mSMVersion(sm)
        , mSpecDecode(specDecode)
        , mPagedKVCache(pagedKVCache)
        , mDataType(dataType)
        , mKVDataType(kvDataType)
    {
        mKernelMeta = &(xqa::kernels::sXqaKernelMetaInfo[0]);
        mKernelMetaCount = sizeof(xqa::kernels::sXqaKernelMetaInfo) / sizeof(xqa::kernels::sXqaKernelMetaInfo[0]);
    }

    //! @throws std::runtime_error if a CUDA driver error occurs
    void loadXQAKernels()
    {
        if (mLoaded)
        {
            return;
        }
        XQADeviceCapability const deviceCapability = getDeviceCapability();
        for (int32_t i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mDataType != mDataType || kernelMeta.mKVDataType != mKVDataType
                || kernelMeta.mSM != mSMVersion || kernelMeta.mCubin == nullptr)
            {
                continue;
            }
            // Filter out kernel that irrelevant to this project.
            if (kernelMeta.mPagedKVCache != mPagedKVCache || kernelMeta.mBeamWidth != 1)
            {
                continue;
            }
            if (kernelMeta.mMultiQueryTokens != mSpecDecode)
            {
                continue;
            }
            if (!isKernelSupportedByBuild(kernelMeta))
            {
                continue;
            }
            // load CUmodule
            CUmodule hModule;
            auto findModuleIter = mModules.find(kernelMeta.mCubin);
            if (findModuleIter != mModules.end())
            {
                hModule = findModuleIter->second;
            }
            else
            {
                CUDA_DRIVER_CHECK(cuModuleLoadData(&hModule, kernelMeta.mCubin));
                mModules.insert(std::make_pair(kernelMeta.mCubin, hModule));
            }

            XQAKernelFuncInfo funcInfo{};
            CUDA_DRIVER_CHECK(cuModuleGetFunction(&funcInfo.mDeviceFunction, hModule, kernelMeta.mFuncName));
            funcInfo.mHeadDim = kernelMeta.mHeadDim;
            funcInfo.mMTileSize = kernelMeta.mMTileSize;
            funcInfo.mSMVersion = kernelMeta.mSM;
            funcInfo.mKernelVariant = kernelMeta.mKernelVariant;
            funcInfo.mRequiresClusterLaunch = kernelMeta.mRequiresClusterLaunch;
            funcInfo.mRequiresDistributedSharedMemory = kernelMeta.mRequiresDistributedSharedMemory;
            funcInfo.mSlidingWindow = kernelMeta.mSlidingWindow;

            uint32_t* deviceSmemSize{nullptr};
            size_t dataSize{0};
            CUDA_DRIVER_CHECK(
                cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&deviceSmemSize), &dataSize, hModule, "smemSize"));
            // Use of default stream is inevitable and justified here because it is called during kernel loading phase,
            // not runtime.
            CUDA_CHECK(cudaMemcpy(&funcInfo.mSharedMemBytes, deviceSmemSize, dataSize, cudaMemcpyDeviceToHost));

            if (!isKernelCompatibleWithDevice(funcInfo, deviceCapability))
            {
                continue;
            }

            // Set 46KB threshold here because we have to take static/driver shared memory into consideration.
            // Default value for shared memory is 48KB.
            if (funcInfo.mSharedMemBytes >= 46 * 1024)
            {
                CUDA_DRIVER_CHECK(cuFuncSetAttribute(funcInfo.mDeviceFunction,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.mSharedMemBytes));
            }
            XQAKernelRuntimeHashKey hashKey{kernelMeta.mDataType, kernelMeta.mKVDataType,
                static_cast<int32_t>(kernelMeta.mHeadDim), static_cast<int32_t>(kernelMeta.mNumQHeadsOverKV),
                static_cast<int32_t>(kernelMeta.mBeamWidth), kernelMeta.mSlidingWindow,
                static_cast<int32_t>(kernelMeta.mTokensPerPage)};
            mFunctions[hashKey].push_back(funcInfo);
        }
        mLoaded = true;
    }

    bool hasKernels() const noexcept
    {
        return !mFunctions.empty();
    }

    XQAKernelFuncInfo findKernelFunction(XQAKernelRuntimeHashKey const& key) const
    {
        auto const findIter = mFunctions.find(key);
        if (findIter == mFunctions.end())
        {
            // Return empty function info.
            return XQAKernelFuncInfo{};
        }

        auto const& candidates = findIter->second;
        auto const findBest = std::min_element(candidates.begin(), candidates.end(),
            [](XQAKernelFuncInfo const& lhs, XQAKernelFuncInfo const& rhs) noexcept {
                return getKernelVariantPriority(lhs.mKernelVariant) < getKernelVariantPriority(rhs.mKernelVariant);
            });
        return *findBest;
    }

protected:
    TKernelMetaInfo const* mKernelMeta;
    int32_t mKernelMetaCount;
    uint32_t mSMVersion;
    bool mSpecDecode;
    bool mPagedKVCache;
    XQADataType mDataType;
    XQADataType mKVDataType;
    bool mLoaded{false};
    std::unordered_map<unsigned long long const*, CUmodule> mModules;

    std::unordered_map<XQAKernelRuntimeHashKey, std::vector<XQAKernelFuncInfo>, XQAKernelRuntimeHasher> mFunctions;
};

class XQAKernelLoader
{

public:
    //! @throws std::runtime_error if a CUDA driver error occurs
    XQAKernelList* getXQAKernelList(
        XQADataType dataType, XQADataType kvDataType, int32_t sm, bool specDecode, bool pagedKVCache)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        XQAKernelLoadHashKey hash_key{dataType, kvDataType, sm, specDecode, pagedKVCache};

        auto findIter = mKernels.find(hash_key);
        if (findIter == mKernels.end())
        {
            std::unique_ptr<XQAKernelList> newKernel
                = std::make_unique<XQAKernelList>(dataType, kvDataType, sm, specDecode, pagedKVCache);
            newKernel->loadXQAKernels();
            mKernels.insert(std::make_pair(hash_key, std::move(newKernel)));
            findIter = mKernels.find(hash_key);
        }
        return findIter->second.get();
    }

    static XQAKernelLoader& Get()
    {
        static XQAKernelLoader instance;
        return instance;
    }

private:
    XQAKernelLoader() = default;

    std::unordered_map<XQAKernelLoadHashKey, std::unique_ptr<XQAKernelList> const, XQAKernelLoadHasher> mKernels;
};

//! @throws std::runtime_error if a CUDA driver error occurs
inline XQAKernelList* getXQAKernels(
    XQADataType dataType, XQADataType kvDataType, int32_t sm, bool specDecode, bool pagedKVCache)
{
    return XQAKernelLoader::Get().getXQAKernelList(dataType, kvDataType, sm, specDecode, pagedKVCache);
}

} // namespace

DecoderXQARunner::DecoderXQARunner(nvinfer1::DataType const dataType, nvinfer1::DataType const kvDataType,
    int32_t batchSize, int32_t numQHeads, int32_t numKvHeads, int32_t headSize, int32_t smVersion) noexcept
    : mDataType(dataType)
    , mKVDataType(kvDataType)
    , mBatchSize(batchSize)
    , mNumHeads(numQHeads)
    , mNumKVHeads(numKvHeads)
    , mHeadSize(headSize)
    , mSmVersion(smVersion)
{
}

XQALaunchParams DecoderXQARunner::initXQAParams() noexcept
{
    XQALaunchParams params{};
    params.numQheads = mNumHeads;
    params.numKVheads = mNumKVHeads;
    params.headSize = mHeadSize;
    params.batchSize = mBatchSize;
    params.dataType = mDataType;
    params.kvDataType = mKVDataType;
    params.headGroupSize = params.numQheads / params.numKVheads;

    return params;
}

bool DecoderXQARunner::canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
    DataType dataType, DataType kvDataType, bool usePagedKVCache) noexcept
{
    bool const checkHeadNumbers = numQHeads % numKVHeads == 0;
    bool const checkType = dataType == DataType::kHALF;
    // FP8 XQA cubins exist only for FP8-capable devices (sm89+).
    bool const checkKVType = kvDataType == DataType::kHALF || (kvDataType == DataType::kFP8 && smVersion >= 89);
    std::vector<int32_t> allowedSMVersions{80, 86, 87, 89, 100, 101, 120, 121};
    bool const checkSMVersion
        = std::find(allowedSMVersions.begin(), allowedSMVersions.end(), smVersion) != allowedSMVersions.end();

    // Current kernel list supports
    // (1) Head ratio 1-8 for head_dim {32, 64, 128}
    // (2) Head ratio 16 for head_dim 128 only (NemotronH).
    // (3) Head ratio 2, 4, 6, 8 for head_dim 256
    //     (4/6/8 for Qwen3.5-MoE / Qwen3.5-Omni Thinker+Talker;
    //      2 for Qwen3.5-Omni Talker decode attention — 16 Q heads / 8 KV heads).
    // (4) Head ratio 2, 4, 8, 16 for head_dim 512 where matching cubins are present.
    //     (2 for Gemma4 E4B assistant: 4 Q heads / 2 KV heads;
    //      4 for Gemma4 E4B: 8 Q heads / 2 KV heads;
    //      8 for Gemma4 E2B: 8 Q heads / 1 KV head;
    //      16 for Gemma4 Unified 12B global attention: 16 Q heads / 1 KV head,
    //      generated for SM100/SM120-family only).
    int32_t const headRatio = numQHeads / numKVHeads;
    XQADataType const xqaKVDataType
        = kvDataType == DataType::kFP8 ? XQADataType::DATA_TYPE_E4M3 : XQADataType::DATA_TYPE_FP16;
    bool const checkHeadDim512SM
        = headSize == 512 && hasHeadDim512KernelsForSM(smVersion, xqaKVDataType, headRatio, usePagedKVCache);
    bool const checkQHeadPerKV
        = ((headSize == 32 || headSize == 64 || headSize == 128) && headRatio >= 1 && headRatio <= 8)
        || (headSize == 128 && headRatio == 16)
        || (headSize == 256 && (headRatio == 2 || headRatio == 4 || headRatio == 6 || headRatio == 8))
        || (headSize == 512 && checkHeadDim512SM
            && (headRatio == 2 || headRatio == 4 || headRatio == 8 || headRatio == 16));

    return checkHeadNumbers && checkType && checkKVType && checkSMVersion && checkQHeadPerKV;
}

bool DecoderXQARunner::loadDecodeXQAKernels(
    int32_t smVersion, DataType dataType, DataType kvDataType, bool useSpecDecodeKernels, bool usePagedKVCache)
{
    XQAKernelList* xqaKernelList = getXQAKernels(
        trtToXqaDataType(dataType), trtToXqaDataType(kvDataType), smVersion, useSpecDecodeKernels, usePagedKVCache);
    return xqaKernelList != nullptr && xqaKernelList->hasKernels();
}

void DecoderXQARunner::dispatchXQAKernel(XQALaunchParams& params, cudaStream_t const& stream)
{
    // Check all device pointers are valid.
    check::check(params.output != nullptr && params.qInputPtr != nullptr && params.kvCache.data != nullptr
            && params.kvCache.sequence_lengths != nullptr
            && (!usePagedKVCache(params.kvCache) || params.kvCache.pageList != nullptr),
        "Invalid device pointer passed to kernel dispatch function");

    constexpr bool useSpecDecode = false;
    bool const usePagedKV = usePagedKVCache(params.kvCache);
    auto hashKey = getRuntimeHashKeyFromXQAParams(params);
    XQAKernelList* xqaKernelList = getXQAKernels(
        trtToXqaDataType(mDataType), trtToXqaDataType(mKVDataType), mSmVersion, useSpecDecode, usePagedKV);
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);
    check::check(kernelInfo.mSharedMemBytes != 0, "No available kernel available for the GQA");

    KernelKVCacheArg kernelKVCacheArg = makeKernelKVCacheArg(params.kvCache, usePagedKV);
    void* const kvCacheArg = getKernelKVCacheArg(kernelKVCacheArg);

    void* kernelParamsNoSliding[]
        = {&params.numKVheads, &params.attentionScale, &params.output, &params.qInputPtr, &params.attentionSinks,
            kvCacheArg, &params.batchSize, &params.kScale, &params.vScale, &params.semaphores, &params.scratch};
    void* kernelParamsSliding[] = {&params.numKVheads, &params.slidingWinSize, &params.attentionScale, &params.output,
        &params.qInputPtr, &params.attentionSinks, kvCacheArg, &params.batchSize, &params.kScale, &params.vScale,
        &params.semaphores, &params.scratch};
    void** kernelParams = kernelInfo.mSlidingWindow ? kernelParamsSliding : kernelParamsNoSliding;

    // The multi-block kernel launch is mainly for long sequence.
    // TODO: Add multiple block launch logic. The launch configuration highly depends on usecase and performance
    // context. Current measured workload doesn't get performance gain from multi-block launch.
    bool const useClusterLaunch = kernelInfo.mRequiresClusterLaunch;
    dim3 const dimGrid{useClusterLaunch ? kSPLIT_HEAD_DIM_512_CLUSTER_SIZE : 1U, static_cast<uint32_t>(mNumKVHeads),
        static_cast<uint32_t>(mBatchSize)};
    dim3 const dimCta{
        getCtaDimX(kernelInfo), 1U, useClusterLaunch ? kSPLIT_HEAD_DIM_512_CTA_DIM_Z : kDEFAULT_XQA_CTA_DIM_Z};
#if SUPPORTS_CLUSTER_LAUNCH
    if (useClusterLaunch)
    {
        launch2CtaHeadDim512ClusterKernel(kernelInfo, dimGrid, dimCta, stream, kernelParams);
        return;
    }
#else
    check::check(!useClusterLaunch, "XQA head_dim=512 2CTA cluster kernel is unavailable.");
#endif // SUPPORTS_CLUSTER_LAUNCH
    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.mDeviceFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimCta.x, dimCta.y,
        dimCta.z, kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
}

void DecoderXQARunner::dispatchSpecDecodeXQAKernel(XQALaunchParams& params, cudaStream_t const& stream)
{
    // Check all device pointers are valid.
    check::check(params.output != nullptr && params.qInputPtr != nullptr && params.kvCache.data != nullptr
            && params.kvCache.sequence_lengths != nullptr && params.treeAttnMask != nullptr
            && (!usePagedKVCache(params.kvCache) || params.kvCache.pageList != nullptr),
        "Invalid device pointer passed to kernel dispatch function");

    constexpr bool useSpecDecode = true;
    bool const usePagedKV = usePagedKVCache(params.kvCache);
    auto hashKey = getRuntimeHashKeyFromXQAParamsSpecDecode(params);
    XQAKernelList* xqaKernelList = getXQAKernels(
        trtToXqaDataType(mDataType), trtToXqaDataType(mKVDataType), mSmVersion, useSpecDecode, usePagedKV);
    XQAKernelFuncInfo kernelInfo = xqaKernelList->findKernelFunction(hashKey);
    check::check(kernelInfo.mSharedMemBytes != 0, "No available kernel available for the Spec-DecodeGQA");

    KernelKVCacheArg kernelKVCacheArg = makeKernelKVCacheArg(params.kvCache, usePagedKV);
    void* const kvCacheArg = getKernelKVCacheArg(kernelKVCacheArg);

    void* kernelParamsNoSliding[] = {&params.qSeqLen, &params.numKVheads, &params.headGroupSize, &params.qCuSeqLen,
        &params.attentionScale, &params.output, &params.qInputPtr, &params.treeAttnMask, &params.attentionSinks,
        kvCacheArg, &params.batchSize, &params.kScale, &params.vScale, &params.semaphores, &params.scratch};
    void* kernelParamsSliding[]
        = {&params.qSeqLen, &params.numKVheads, &params.headGroupSize, &params.qCuSeqLen, &params.slidingWinSize,
            &params.attentionScale, &params.output, &params.qInputPtr, &params.treeAttnMask, &params.attentionSinks,
            kvCacheArg, &params.batchSize, &params.kScale, &params.vScale, &params.semaphores, &params.scratch};
    void** kernelParams = kernelInfo.mSlidingWindow ? kernelParamsSliding : kernelParamsNoSliding;
    int32_t const ctaTileY = static_cast<int32_t>(kernelInfo.mMTileSize);
    check::check(ctaTileY > 0, format::fmtstr("Invalid spec-decode ctaTileY %d in XQA kernel metadata.", ctaTileY));
    int32_t const tokenBlockPerGroup = (params.qSeqLen * params.headGroupSize - 1) / ctaTileY + 1;
    bool const useClusterLaunch = kernelInfo.mRequiresClusterLaunch;
    dim3 const dimGrid{useClusterLaunch ? kSPLIT_HEAD_DIM_512_CLUSTER_SIZE : 1U,
        static_cast<uint32_t>(mNumKVHeads * tokenBlockPerGroup), static_cast<uint32_t>(mBatchSize)};
    dim3 const dimCta{
        getCtaDimX(kernelInfo), 1U, useClusterLaunch ? kSPLIT_HEAD_DIM_512_CTA_DIM_Z : kDEFAULT_XQA_CTA_DIM_Z};
#if SUPPORTS_CLUSTER_LAUNCH
    if (useClusterLaunch)
    {
        launch2CtaHeadDim512ClusterKernel(kernelInfo, dimGrid, dimCta, stream, kernelParams);
        return;
    }
#else
    check::check(!useClusterLaunch, "XQA head_dim=512 2CTA cluster kernel is unavailable.");
#endif // SUPPORTS_CLUSTER_LAUNCH
    CUDA_DRIVER_CHECK(cuLaunchKernel(kernelInfo.mDeviceFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimCta.x, dimCta.y,
        dimCta.z, kernelInfo.mSharedMemBytes, stream, kernelParams, nullptr));
}
