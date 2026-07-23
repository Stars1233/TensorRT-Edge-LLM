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

#ifdef CUTE_DSL_FMHA_ENABLED

#include "cuteDslFMHARunner.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "contextFMHARunner.h"

#include <climits>
#include <cmath>

namespace trt_edgellm
{

// =====================================================================
// Static member initialization
// =====================================================================

// LLM (FP16)
fmha_d64_Kernel_Module_t CuteDslFMHARunner::sLLM_d64 = {};
fmha_d128_Kernel_Module_t CuteDslFMHARunner::sLLM_d128 = {};
fmha_d256_Kernel_Module_t CuteDslFMHARunner::sLLM_d256 = {};
fmha_d64_sw_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw = {};
fmha_d128_sw_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw = {};
fmha_d256_sw_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw = {};
// LLM skip-softmax (BLASST, FP16 causal)
fmha_d64_skipsoftmax_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_skipsoftmax = {};
fmha_d128_skipsoftmax_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_skipsoftmax = {};
// LLM (FP8 input, FP16 output)
fmha_d64_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_fp8 = {};
fmha_d128_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_fp8 = {};
fmha_d256_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_fp8 = {};
fmha_d64_sw_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw_fp8 = {};
fmha_d128_sw_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw_fp8 = {};
fmha_d256_sw_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw_fp8 = {};
// LLM paged KV cache (FP16)
fmha_d64_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_paged = {};
fmha_d128_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_paged = {};
fmha_d256_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_paged = {};
fmha_d64_sw_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw_paged = {};
fmha_d128_sw_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw_paged = {};
fmha_d256_sw_paged_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw_paged = {};
// LLM paged KV cache (FP8 input, FP16 output)
fmha_d64_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_paged_fp8 = {};
fmha_d128_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_paged_fp8 = {};
fmha_d256_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_paged_fp8 = {};
fmha_d64_sw_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d64_sw_paged_fp8 = {};
fmha_d128_sw_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d128_sw_paged_fp8 = {};
fmha_d256_sw_paged_fp8_Kernel_Module_t CuteDslFMHARunner::sLLM_d256_sw_paged_fp8 = {};
bool CuteDslFMHARunner::sLLMLoaded = false;
std::mutex CuteDslFMHARunner::sLLMMutex;

// ViT
vit_fmha_d64_Kernel_Module_t CuteDslFMHARunner::sViT_d64 = {};
vit_fmha_d72_Kernel_Module_t CuteDslFMHARunner::sViT_d72 = {};
vit_fmha_d80_Kernel_Module_t CuteDslFMHARunner::sViT_d80 = {};
vit_fmha_d128_Kernel_Module_t CuteDslFMHARunner::sViT_d128 = {};
bool CuteDslFMHARunner::sViTLoaded = false;
std::mutex CuteDslFMHARunner::sViTMutex;

// =====================================================================
// Kernel module loading
// =====================================================================

bool CuteDslFMHARunner::loadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock(sLLMMutex);
    if (sLLMLoaded)
    {
        return true;
    }
    try
    {
        fmha_d64_Kernel_Module_Load(&sLLM_d64);
        fmha_d128_Kernel_Module_Load(&sLLM_d128);
        fmha_d256_Kernel_Module_Load(&sLLM_d256);
        fmha_d64_sw_Kernel_Module_Load(&sLLM_d64_sw);
        fmha_d128_sw_Kernel_Module_Load(&sLLM_d128_sw);
        fmha_d256_sw_Kernel_Module_Load(&sLLM_d256_sw);
        fmha_d64_skipsoftmax_Kernel_Module_Load(&sLLM_d64_skipsoftmax);
        fmha_d128_skipsoftmax_Kernel_Module_Load(&sLLM_d128_skipsoftmax);
        fmha_d64_fp8_Kernel_Module_Load(&sLLM_d64_fp8);
        fmha_d128_fp8_Kernel_Module_Load(&sLLM_d128_fp8);
        fmha_d256_fp8_Kernel_Module_Load(&sLLM_d256_fp8);
        fmha_d64_sw_fp8_Kernel_Module_Load(&sLLM_d64_sw_fp8);
        fmha_d128_sw_fp8_Kernel_Module_Load(&sLLM_d128_sw_fp8);
        fmha_d256_sw_fp8_Kernel_Module_Load(&sLLM_d256_sw_fp8);
        fmha_d64_paged_Kernel_Module_Load(&sLLM_d64_paged);
        fmha_d128_paged_Kernel_Module_Load(&sLLM_d128_paged);
        fmha_d256_paged_Kernel_Module_Load(&sLLM_d256_paged);
        fmha_d64_sw_paged_Kernel_Module_Load(&sLLM_d64_sw_paged);
        fmha_d128_sw_paged_Kernel_Module_Load(&sLLM_d128_sw_paged);
        fmha_d256_sw_paged_Kernel_Module_Load(&sLLM_d256_sw_paged);
        fmha_d64_paged_fp8_Kernel_Module_Load(&sLLM_d64_paged_fp8);
        fmha_d128_paged_fp8_Kernel_Module_Load(&sLLM_d128_paged_fp8);
        fmha_d256_paged_fp8_Kernel_Module_Load(&sLLM_d256_paged_fp8);
        fmha_d64_sw_paged_fp8_Kernel_Module_Load(&sLLM_d64_sw_paged_fp8);
        fmha_d128_sw_paged_fp8_Kernel_Module_Load(&sLLM_d128_sw_paged_fp8);
        fmha_d256_sw_paged_fp8_Kernel_Module_Load(&sLLM_d256_sw_paged_fp8);
        sLLMLoaded = true;
        LOG_DEBUG("CuTe DSL LLM FMHA kernel modules loaded (FP16 + FP8 + paged)");
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load CuTe DSL LLM FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHARunner::unloadLLMKernelModule()
{
    std::lock_guard<std::mutex> lock(sLLMMutex);
    if (sLLMLoaded)
    {
        fmha_d64_Kernel_Module_Unload(&sLLM_d64);
        fmha_d128_Kernel_Module_Unload(&sLLM_d128);
        fmha_d256_Kernel_Module_Unload(&sLLM_d256);
        fmha_d64_sw_Kernel_Module_Unload(&sLLM_d64_sw);
        fmha_d128_sw_Kernel_Module_Unload(&sLLM_d128_sw);
        fmha_d256_sw_Kernel_Module_Unload(&sLLM_d256_sw);
        fmha_d64_skipsoftmax_Kernel_Module_Unload(&sLLM_d64_skipsoftmax);
        fmha_d128_skipsoftmax_Kernel_Module_Unload(&sLLM_d128_skipsoftmax);
        fmha_d64_fp8_Kernel_Module_Unload(&sLLM_d64_fp8);
        fmha_d128_fp8_Kernel_Module_Unload(&sLLM_d128_fp8);
        fmha_d256_fp8_Kernel_Module_Unload(&sLLM_d256_fp8);
        fmha_d64_sw_fp8_Kernel_Module_Unload(&sLLM_d64_sw_fp8);
        fmha_d128_sw_fp8_Kernel_Module_Unload(&sLLM_d128_sw_fp8);
        fmha_d256_sw_fp8_Kernel_Module_Unload(&sLLM_d256_sw_fp8);
        fmha_d64_paged_Kernel_Module_Unload(&sLLM_d64_paged);
        fmha_d128_paged_Kernel_Module_Unload(&sLLM_d128_paged);
        fmha_d256_paged_Kernel_Module_Unload(&sLLM_d256_paged);
        fmha_d64_sw_paged_Kernel_Module_Unload(&sLLM_d64_sw_paged);
        fmha_d128_sw_paged_Kernel_Module_Unload(&sLLM_d128_sw_paged);
        fmha_d256_sw_paged_Kernel_Module_Unload(&sLLM_d256_sw_paged);
        fmha_d64_paged_fp8_Kernel_Module_Unload(&sLLM_d64_paged_fp8);
        fmha_d128_paged_fp8_Kernel_Module_Unload(&sLLM_d128_paged_fp8);
        fmha_d256_paged_fp8_Kernel_Module_Unload(&sLLM_d256_paged_fp8);
        fmha_d64_sw_paged_fp8_Kernel_Module_Unload(&sLLM_d64_sw_paged_fp8);
        fmha_d128_sw_paged_fp8_Kernel_Module_Unload(&sLLM_d128_sw_paged_fp8);
        fmha_d256_sw_paged_fp8_Kernel_Module_Unload(&sLLM_d256_sw_paged_fp8);
        sLLMLoaded = false;
    }
}

bool CuteDslFMHARunner::loadViTKernelModule()
{
    std::lock_guard<std::mutex> lock(sViTMutex);
    if (sViTLoaded)
    {
        return true;
    }
    try
    {
        vit_fmha_d64_Kernel_Module_Load(&sViT_d64);
        vit_fmha_d72_Kernel_Module_Load(&sViT_d72);
        vit_fmha_d80_Kernel_Module_Load(&sViT_d80);
        vit_fmha_d128_Kernel_Module_Load(&sViT_d128);
        sViTLoaded = true;
        LOG_DEBUG("CuTe DSL ViT FMHA kernel modules loaded");
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load CuTe DSL ViT FMHA kernel modules");
        return false;
    }
}

void CuteDslFMHARunner::unloadViTKernelModule()
{
    std::lock_guard<std::mutex> lock(sViTMutex);
    if (sViTLoaded)
    {
        vit_fmha_d64_Kernel_Module_Unload(&sViT_d64);
        vit_fmha_d72_Kernel_Module_Unload(&sViT_d72);
        vit_fmha_d80_Kernel_Module_Unload(&sViT_d80);
        vit_fmha_d128_Kernel_Module_Unload(&sViT_d128);
        sViTLoaded = false;
    }
}

bool CuteDslFMHARunner::canImplement(int32_t headSize, int32_t smVersion)
{
    bool const supportedHeadSize = headSize == 64 || headSize == 128 || headSize == 256;
    return smVersion >= 100 && supportedHeadSize;
}

bool CuteDslFMHARunner::canImplementViT(int32_t headSize, int32_t smVersion)
{
    return (smVersion >= 100) && (headSize == 64 || headSize == 72 || headSize == 80 || headSize == 128);
}

// =====================================================================
// Constructors
// =====================================================================

CuteDslFMHARunner::CuteDslFMHARunner(
    int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize, int32_t seqLenQ, int32_t kvCacheCapacity)
    : mBatchSize(batchSize)
    , mSeqLenQ(seqLenQ)
    , mKVCacheCapacity(kvCacheCapacity)
    , mNumHeadsQ(numQHeads)
    , mNumHeadsK(numKVHeads)
    , mHeadDim(headDim)
{
}

// Macro shared by run() and runFp8() — sets up Q/KV/O/cumSeqlenK tensor structs
// and calls the CuTe DSL wrapper. Relies on local variables: batchSize, seqLenQ,
// numQHeads, numKVHeads, headDim, capacity, qPtr, kvPtr, oPtr, cuKVSeqLens,
// attentionScale, scaleQ, scaleK, scaleV, invScaleO, ret.
// clang-format off
#define CALL_LLM_FMHA(PREFIX, MODULE, WSL)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_q_tensor_t qTensor{};                                                                          \
        qTensor.data = const_cast<void*>(qPtr);                                                                        \
        qTensor.dynamic_shapes[0] = batchSize;                                                                         \
        qTensor.dynamic_shapes[1] = seqLenQ;                                                                           \
        qTensor.dynamic_shapes[2] = numQHeads;                                                                         \
        qTensor.dynamic_shapes[3] = headDim;                                                                           \
        qTensor.dynamic_strides[0] = static_cast<int64_t>(seqLenQ) * numQHeads * headDim;                             \
        qTensor.dynamic_strides[1] = static_cast<int64_t>(numQHeads) * headDim;                                       \
        qTensor.dynamic_strides[2] = static_cast<int64_t>(headDim);                                                    \
                                                                                                                       \
        PREFIX##_Tensor_kv_cache_t kvTensor{};                                                                          \
        kvTensor.data = const_cast<void*>(kvPtr);                                                                       \
        kvTensor.dynamic_shapes[0] = batchSize;                                                                         \
        kvTensor.dynamic_shapes[1] = 2;                                                                                 \
        kvTensor.dynamic_shapes[2] = numKVHeads;                                                                        \
        kvTensor.dynamic_shapes[3] = capacity;                                                                          \
        kvTensor.dynamic_shapes[4] = headDim;                                                                           \
        kvTensor.dynamic_strides[0] = static_cast<int64_t>(2) * numKVHeads * capacity * headDim;                       \
        kvTensor.dynamic_strides[1] = static_cast<int64_t>(numKVHeads) * capacity * headDim;                           \
        kvTensor.dynamic_strides[2] = static_cast<int64_t>(capacity) * headDim;                                        \
        kvTensor.dynamic_strides[3] = static_cast<int64_t>(headDim);                                                    \
                                                                                                                       \
        PREFIX##_Tensor_o_tensor_t oTensor{};                                                                           \
        oTensor.data = oPtr;                                                                                            \
        oTensor.dynamic_shapes[0] = batchSize;                                                                          \
        oTensor.dynamic_shapes[1] = seqLenQ;                                                                            \
        oTensor.dynamic_shapes[2] = numQHeads;                                                                          \
        oTensor.dynamic_shapes[3] = headDim;                                                                            \
        oTensor.dynamic_strides[0] = static_cast<int64_t>(seqLenQ) * numQHeads * headDim;                              \
        oTensor.dynamic_strides[1] = static_cast<int64_t>(numQHeads) * headDim;                                        \
        oTensor.dynamic_strides[2] = static_cast<int64_t>(headDim);                                                     \
                                                                                                                       \
        PREFIX##_Tensor_cum_seqlen_k_t cumSeqlenK{};                                                                     \
        cumSeqlenK.data = const_cast<void*>(static_cast<void const*>(cuKVSeqLens));                                     \
        cumSeqlenK.dynamic_shapes[0] = batchSize + 1;                                                                   \
                                                                                                                       \
        ret = cute_dsl_##PREFIX##_wrapper(                                                                              \
            &(MODULE), &qTensor, &kvTensor, &oTensor, &cumSeqlenK, (WSL), attentionScale,                               \
            scaleQ, scaleK, scaleV, invScaleO, stream);                                                                 \
    } while (0)
// clang-format on

// =====================================================================
// LLM run: batched Q + combined KV cache (FP16 / FP8→FP16 / FP8→FP8)
// =====================================================================

void CuteDslFMHARunner::run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens,
    cudaStream_t stream, float attentionScale, int32_t slidingWindowSize, bool fp8Input, float qScale, float kScale,
    float vScale, bool enableSkipSoftmax)
{
    if (!sLLMLoaded)
    {
        LOG_ERROR("CuTe DSL LLM FMHA kernel module not loaded.");
        return;
    }

    validateAttentionScale(attentionScale);
    float const scaleQ = qScale;
    float const scaleK = kScale;
    float const scaleV = vScale;
    float const invScaleO = 1.0f;

    int32_t const batchSize = mBatchSize;
    int32_t const seqLenQ = mSeqLenQ;
    int32_t const numQHeads = mNumHeadsQ;
    int32_t const numKVHeads = mNumHeadsK;
    int32_t const headDim = mHeadDim;
    int32_t const capacity = mKVCacheCapacity;

    bool const useSlidingWindow = (slidingWindowSize < INT_MAX);

    int32_t ret = -1;
    int32_t constexpr kNoLimit = 1 << 30;
    int32_t const windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNoLimit;

// Dispatch helper: selects the kernel module pair (non-SW / SW) for a given head dim.
#define DISPATCH_HEADD(D, NO_SW_PREFIX, NO_SW_MOD, SW_PREFIX, SW_MOD)                                                  \
    if (headDim == D)                                                                                                  \
    {                                                                                                                  \
        if (useSlidingWindow)                                                                                          \
        {                                                                                                              \
            CALL_LLM_FMHA(SW_PREFIX, SW_MOD, windowSizeLeft);                                                          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            CALL_LLM_FMHA(NO_SW_PREFIX, NO_SW_MOD, windowSizeLeft);                                                    \
        }                                                                                                              \
    }

    if (enableSkipSoftmax)
    {
        if (fp8Input || useSlidingWindow)
        {
            LOG_ERROR("CuTe DSL LLM FMHA: skip-softmax variant is FP16 causal only (fp8Input=%s, sw=%s)",
                fp8Input ? "true" : "false", useSlidingWindow ? "true" : "false");
            return;
        }
        if (headDim == 64)
        {
            CALL_LLM_FMHA(fmha_d64_skipsoftmax, sLLM_d64_skipsoftmax, windowSizeLeft);
        }
        else if (headDim == 128)
        {
            CALL_LLM_FMHA(fmha_d128_skipsoftmax, sLLM_d128_skipsoftmax, windowSizeLeft);
        }
        else
        {
            LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim);
            return;
        }
    }
    else if (fp8Input)
    {
        DISPATCH_HEADD(64, fmha_d64_fp8, sLLM_d64_fp8, fmha_d64_sw_fp8, sLLM_d64_sw_fp8)
        else DISPATCH_HEADD(128, fmha_d128_fp8, sLLM_d128_fp8, fmha_d128_sw_fp8, sLLM_d128_sw_fp8) else DISPATCH_HEADD(
            256, fmha_d256_fp8, sLLM_d256_fp8, fmha_d256_sw_fp8, sLLM_d256_sw_fp8) else
        {
            LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim);
            return;
        }
    }
    else
    {
        DISPATCH_HEADD(64, fmha_d64, sLLM_d64, fmha_d64_sw, sLLM_d64_sw)
        else DISPATCH_HEADD(128, fmha_d128, sLLM_d128, fmha_d128_sw, sLLM_d128_sw) else DISPATCH_HEADD(
            256, fmha_d256, sLLM_d256, fmha_d256_sw, sLLM_d256_sw) else
        {
            LOG_ERROR("CuTe DSL LLM FMHA: unsupported head_dim=%d", headDim);
            return;
        }
    }

#undef DISPATCH_HEADD

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL LLM FMHA kernel (d=%d, sw=%s, fp8in=%s) failed with error code: %d", headDim,
            useSlidingWindow ? "true" : "false", fp8Input ? "true" : "false", ret);
    }

#undef CALL_LLM_FMHA
}

void CuteDslFMHARunner::runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList,
    void* oPtr, int32_t const* cuKVSeqLens, int32_t numPages, int32_t maxPagesPerSeq, int32_t tokensPerPage,
    nvinfer1::DataType kvDataType, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize, bool fp8Input,
    float qScale, float kScale, float vScale)
{
    if (!sLLMLoaded)
    {
        LOG_ERROR("CuTe DSL LLM FMHA kernel module not loaded.");
        return;
    }

    check::check(qPtr != nullptr, "CuTe DSL paged FMHA qPtr must not be null.");
    check::check(pagedKVPoolPtr != nullptr, "CuTe DSL paged FMHA KV pool must not be null.");
    check::check(kvCachePageList != nullptr, "CuTe DSL paged FMHA page list must not be null.");
    check::check(oPtr != nullptr, "CuTe DSL paged FMHA oPtr must not be null.");
    check::check(cuKVSeqLens != nullptr, "CuTe DSL paged FMHA cuKVSeqLens must not be null.");
    check::check(numPages > 0 && maxPagesPerSeq > 0 && tokensPerPage > 0,
        "CuTe DSL paged FMHA requires positive numPages/maxPagesPerSeq/tokensPerPage.");
    // Direct paged CuTe DSL keeps the existing TMA load pipeline: each logical K/V tile maps to one physical page.
    // These AOT variants use tile_N=128, so smaller pages would require stitching one tile from multiple pages.
    check::check(tokensPerPage == 128,
        "CuTe DSL direct paged FMHA requires tokensPerPage == 128 because one K/V TMA tile maps to one page.");
    check::check(mKVCacheCapacity == maxPagesPerSeq * tokensPerPage,
        "CuTe DSL paged FMHA runner capacity must equal maxPagesPerSeq * tokensPerPage.");
    check::check(kvDataType == nvinfer1::DataType::kHALF || kvDataType == nvinfer1::DataType::kFP8,
        "CuTe DSL paged FMHA supports FP16 or FP8 KV cache.");
    check::check((kvDataType == nvinfer1::DataType::kFP8) == fp8Input,
        "CuTe DSL paged FMHA requires fp8Input to match the paged KV cache dtype.");

    float const scaleQ = qScale;
    float const scaleK = kScale;
    float const scaleV = vScale;
    float const invScaleO = 1.0f;

    int32_t const batchSize = mBatchSize;
    int32_t const seqLenQ = mSeqLenQ;
    int32_t const numQHeads = mNumHeadsQ;
    int32_t const numKVHeads = mNumHeadsK;
    int32_t const headDim = mHeadDim;
    bool const useSlidingWindow = (slidingWindowSize < INT_MAX);

    int32_t ret = -1;
    int32_t constexpr kNoLimit = 1 << 30;
    int32_t const windowSizeLeft = useSlidingWindow ? slidingWindowSize : kNoLimit;

    // The physical paged KV pool layout is fixed to NHD [numPages, tokensPerPage, H_kv, D].
    // CuTe DSL still receives logical shape [numPages, H_kv, tokensPerPage, D], mapped through strides below.
    // clang-format off
#define CALL_LLM_FMHA_PAGED(PREFIX, MODULE, WSL)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_q_tensor_t qTensor{};                                                                          \
        qTensor.data = const_cast<void*>(qPtr);                                                                        \
        qTensor.dynamic_shapes[0] = batchSize;                                                                         \
        qTensor.dynamic_shapes[1] = seqLenQ;                                                                           \
        qTensor.dynamic_shapes[2] = numQHeads;                                                                         \
        qTensor.dynamic_shapes[3] = headDim;                                                                           \
        qTensor.dynamic_strides[0] = static_cast<int64_t>(seqLenQ) * numQHeads * headDim;                             \
        qTensor.dynamic_strides[1] = static_cast<int64_t>(numQHeads) * headDim;                                       \
        qTensor.dynamic_strides[2] = static_cast<int64_t>(headDim);                                                    \
                                                                                                                       \
        PREFIX##_Tensor_kv_cache_pool_t kvPoolTensor{};                                                                \
        kvPoolTensor.data = const_cast<void*>(pagedKVPoolPtr);                                                         \
        kvPoolTensor.dynamic_shapes[0] = numPages;                                                                     \
        kvPoolTensor.dynamic_shapes[1] = numKVHeads;                                                                   \
        kvPoolTensor.dynamic_shapes[2] = tokensPerPage;                                                                \
        kvPoolTensor.dynamic_shapes[3] = headDim;                                                                      \
        kvPoolTensor.dynamic_strides[0] = static_cast<int64_t>(numKVHeads) * tokensPerPage * headDim;                  \
        kvPoolTensor.dynamic_strides[1] = static_cast<int64_t>(headDim);                                                \
        kvPoolTensor.dynamic_strides[2] = static_cast<int64_t>(numKVHeads) * headDim;                                  \
                                                                                                                       \
        PREFIX##_Tensor_kv_cache_page_list_t pageListTensor{};                                                         \
        pageListTensor.data = const_cast<int32_t*>(kvCachePageList);                                                   \
        pageListTensor.dynamic_shapes[0] = batchSize;                                                                  \
        pageListTensor.dynamic_shapes[1] = 2;                                                                          \
        pageListTensor.dynamic_shapes[2] = maxPagesPerSeq;                                                             \
        pageListTensor.dynamic_strides[0] = static_cast<int64_t>(2) * maxPagesPerSeq;                                  \
        pageListTensor.dynamic_strides[1] = static_cast<int64_t>(maxPagesPerSeq);                                      \
                                                                                                                       \
        PREFIX##_Tensor_o_tensor_t oTensor{};                                                                          \
        oTensor.data = oPtr;                                                                                           \
        oTensor.dynamic_shapes[0] = batchSize;                                                                         \
        oTensor.dynamic_shapes[1] = seqLenQ;                                                                           \
        oTensor.dynamic_shapes[2] = numQHeads;                                                                         \
        oTensor.dynamic_shapes[3] = headDim;                                                                           \
        oTensor.dynamic_strides[0] = static_cast<int64_t>(seqLenQ) * numQHeads * headDim;                             \
        oTensor.dynamic_strides[1] = static_cast<int64_t>(numQHeads) * headDim;                                       \
        oTensor.dynamic_strides[2] = static_cast<int64_t>(headDim);                                                    \
                                                                                                                       \
        PREFIX##_Tensor_cum_seqlen_k_t cumSeqlenK{};                                                                   \
        cumSeqlenK.data = const_cast<void*>(static_cast<void const*>(cuKVSeqLens));                                    \
        cumSeqlenK.dynamic_shapes[0] = batchSize + 1;                                                                  \
                                                                                                                       \
        ret = cute_dsl_##PREFIX##_wrapper(&(MODULE), &qTensor, &kvPoolTensor, &pageListTensor, &oTensor, &cumSeqlenK,  \
            (WSL), attentionScale, scaleQ, scaleK, scaleV, invScaleO, stream);                                                         \
    } while (0)
    // clang-format on

#define DISPATCH_PAGED_HEADD(D, NO_SW_PREFIX, NO_SW_MOD, SW_PREFIX, SW_MOD)                                            \
    if (headDim == D)                                                                                                  \
    {                                                                                                                  \
        if (useSlidingWindow)                                                                                          \
        {                                                                                                              \
            CALL_LLM_FMHA_PAGED(SW_PREFIX, SW_MOD, windowSizeLeft);                                                    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            CALL_LLM_FMHA_PAGED(NO_SW_PREFIX, NO_SW_MOD, windowSizeLeft);                                              \
        }                                                                                                              \
    }

    if (fp8Input)
    {
        DISPATCH_PAGED_HEADD(64, fmha_d64_paged_fp8, sLLM_d64_paged_fp8, fmha_d64_sw_paged_fp8, sLLM_d64_sw_paged_fp8)
        else DISPATCH_PAGED_HEADD(128, fmha_d128_paged_fp8, sLLM_d128_paged_fp8, fmha_d128_sw_paged_fp8,
            sLLM_d128_sw_paged_fp8) else DISPATCH_PAGED_HEADD(256, fmha_d256_paged_fp8, sLLM_d256_paged_fp8,
            fmha_d256_sw_paged_fp8, sLLM_d256_sw_paged_fp8) else
        {
            LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", headDim);
            return;
        }
    }
    else
    {
        DISPATCH_PAGED_HEADD(64, fmha_d64_paged, sLLM_d64_paged, fmha_d64_sw_paged, sLLM_d64_sw_paged)
        else DISPATCH_PAGED_HEADD(128, fmha_d128_paged, sLLM_d128_paged, fmha_d128_sw_paged,
            sLLM_d128_sw_paged) else DISPATCH_PAGED_HEADD(256, fmha_d256_paged, sLLM_d256_paged, fmha_d256_sw_paged,
            sLLM_d256_sw_paged) else
        {
            LOG_ERROR("CuTe DSL paged LLM FMHA: unsupported head_dim=%d", headDim);
            return;
        }
    }

#undef DISPATCH_PAGED_HEADD
#undef CALL_LLM_FMHA_PAGED

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL paged LLM FMHA kernel (d=%d, sw=%s, fp8in=%s) failed with error code: %d", headDim,
            useSlidingWindow ? "true" : "false", fp8Input ? "true" : "false", ret);
    }
}

// =====================================================================
// ViT run: packed varlen separate Q/K/V
// =====================================================================

void CuteDslFMHARunner::run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
    int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream, float attentionScale)
{
    if (!sViTLoaded)
    {
        LOG_ERROR("CuTe DSL ViT FMHA kernel module not loaded.");
        return;
    }

    validateAttentionScale(attentionScale);
    float const scaleSoftmaxLog2 = attentionScale * static_cast<float>(M_LOG2E);
    float const scaleOutput = 1.0F;

    int32_t const numHeads = mNumHeadsQ;
    int32_t const headDim = mHeadDim;

    // clang-format off
#define CALL_VIT_FMHA(PREFIX, MODULE)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        PREFIX##_Tensor_q_tensor_t qTensor{};                                                                          \
        qTensor.data = const_cast<void*>(qPtr);                                                                        \
        qTensor.dynamic_shapes[0] = totalSeqLen;                                                                       \
        qTensor.dynamic_shapes[1] = numHeads;                                                                          \
        qTensor.dynamic_shapes[2] = headDim;                                                                           \
        qTensor.dynamic_strides[0] = static_cast<int64_t>(numHeads) * headDim;                                        \
        qTensor.dynamic_strides[1] = static_cast<int64_t>(headDim);                                                    \
                                                                                                                       \
        PREFIX##_Tensor_k_tensor_t kTensor{};                                                                           \
        kTensor.data = const_cast<void*>(kPtr);                                                                         \
        kTensor.dynamic_shapes[0] = totalSeqLen;                                                                        \
        kTensor.dynamic_shapes[1] = numHeads;                                                                           \
        kTensor.dynamic_shapes[2] = headDim;                                                                            \
        kTensor.dynamic_strides[0] = static_cast<int64_t>(numHeads) * headDim;                                         \
        kTensor.dynamic_strides[1] = static_cast<int64_t>(headDim);                                                     \
                                                                                                                       \
        PREFIX##_Tensor_v_tensor_t vTensor{};                                                                           \
        vTensor.data = const_cast<void*>(vPtr);                                                                         \
        vTensor.dynamic_shapes[0] = totalSeqLen;                                                                        \
        vTensor.dynamic_shapes[1] = numHeads;                                                                           \
        vTensor.dynamic_shapes[2] = headDim;                                                                            \
        vTensor.dynamic_strides[0] = static_cast<int64_t>(numHeads) * headDim;                                         \
        vTensor.dynamic_strides[1] = static_cast<int64_t>(headDim);                                                     \
                                                                                                                       \
        PREFIX##_Tensor_o_tensor_t oTensor{};                                                                           \
        oTensor.data = oPtr;                                                                                            \
        oTensor.dynamic_shapes[0] = totalSeqLen;                                                                        \
        oTensor.dynamic_shapes[1] = numHeads;                                                                           \
        oTensor.dynamic_shapes[2] = headDim;                                                                            \
        oTensor.dynamic_strides[0] = static_cast<int64_t>(numHeads) * headDim;                                         \
        oTensor.dynamic_strides[1] = static_cast<int64_t>(headDim);                                                     \
                                                                                                                       \
        PREFIX##_Tensor_cu_seqlens_t cuSeqlensTensor{};                                                                 \
        cuSeqlensTensor.data = const_cast<void*>(static_cast<void const*>(cuSeqLens));                                  \
        cuSeqlensTensor.dynamic_shapes[0] = batchSize + 1;                                                              \
                                                                                                                       \
        ret = cute_dsl_##PREFIX##_wrapper(                                                                              \
            &(MODULE), &qTensor, &kTensor, &vTensor, &oTensor, &cuSeqlensTensor, maxSeqLen,                            \
            scaleSoftmaxLog2, attentionScale, scaleOutput, stream);                                                       \
    } while (0)
    // clang-format on

    int32_t ret = -1;

    if (headDim == 64)
    {
        CALL_VIT_FMHA(vit_fmha_d64, sViT_d64);
    }
    else if (headDim == 72)
    {
        CALL_VIT_FMHA(vit_fmha_d72, sViT_d72);
    }
    else if (headDim == 80)
    {
        CALL_VIT_FMHA(vit_fmha_d80, sViT_d80);
    }
    else if (headDim == 128)
    {
        CALL_VIT_FMHA(vit_fmha_d128, sViT_d128);
    }
    else
    {
        LOG_ERROR("CuTe DSL ViT FMHA: unsupported head_dim=%d", headDim);
        return;
    }

#undef CALL_VIT_FMHA

    if (ret != 0)
    {
        LOG_ERROR("CuTe DSL ViT FMHA kernel (d=%d) failed with error code: %d", headDim, ret);
    }
}

} // namespace trt_edgellm

#endif // CUTE_DSL_FMHA_ENABLED
