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

#if defined(CUTE_DSL_FFPA_ENABLED)

#include "cuteDslFFPARunner.h"
#include "common/logger.h"

#include <cmath>
#include <mutex>

namespace trt_edgellm
{

ffpa_d512_causal_Kernel_Module_t CuteDslFFPARunner::sD512CausalModule{};
#ifdef CUTE_DSL_FFPA_VISIONBLOCK_ENABLED
ffpa_d512_causal_visionblock_Kernel_Module_t CuteDslFFPARunner::sD512CausalVisionBlockModule{};
#endif
#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
ffpa_d512_causal_gqa4_Kernel_Module_t CuteDslFFPARunner::sD512CausalGqa4Module{};
#endif
#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
ffpa_d512_causal_gqa8_Kernel_Module_t CuteDslFFPARunner::sD512CausalGqa8Module{};
#endif
#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
ffpa_d512_causal_gqa16_Kernel_Module_t CuteDslFFPARunner::sD512CausalGqa16Module{};
#endif
bool CuteDslFFPARunner::sLoaded{false};
std::mutex CuteDslFFPARunner::sMutex;

bool CuteDslFFPARunner::canImplement(int32_t headDim, int32_t smVersion, int32_t numQHeads, int32_t numKVHeads)
{
    if (headDim != 512)
    {
        return false;
    }

    switch (smVersion)
    {
    case 80:
    case 86:
    case 87:
    case 89:
    case 100:
    case 101:
    case 110:
    case 120:
    case 121: break;
    default: return false;
    }

    // GQA group size check.
    if (numKVHeads <= 0 || numQHeads % numKVHeads != 0)
    {
        return false;
    }

    int32_t const kvGroupSize = numQHeads / numKVHeads;
    switch (kvGroupSize)
    {
    case 1: return true; // MHA — always supported by base kernel
#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
    case 4: return true;
#endif
#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
    case 8: return true;
#endif
#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
    case 16: return true;
#endif
    default: return false;
    }
}

bool CuteDslFFPARunner::canImplementVisionBlock(int32_t headDim, int32_t smVersion)
{
#ifdef CUTE_DSL_FFPA_VISIONBLOCK_ENABLED
    return canImplement(headDim, smVersion);
#else
    (void) headDim;
    (void) smVersion;
    return false;
#endif
}

bool CuteDslFFPARunner::loadKernelModule()
{
    std::lock_guard<std::mutex> lock{sMutex};
    if (sLoaded)
    {
        return true;
    }

    try
    {
        ffpa_d512_causal_Kernel_Module_Load(&sD512CausalModule);
#ifdef CUTE_DSL_FFPA_VISIONBLOCK_ENABLED
        ffpa_d512_causal_visionblock_Kernel_Module_Load(&sD512CausalVisionBlockModule);
#endif
#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
        ffpa_d512_causal_gqa4_Kernel_Module_Load(&sD512CausalGqa4Module);
#endif
#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
        ffpa_d512_causal_gqa8_Kernel_Module_Load(&sD512CausalGqa8Module);
#endif
#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
        ffpa_d512_causal_gqa16_Kernel_Module_Load(&sD512CausalGqa16Module);
#endif
        sLoaded = true;
        LOG_DEBUG("CuTe DSL FFPA d512 causal kernel module(s) loaded");
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load FFPA d512 causal CuTe DSL kernel module.");
        return false;
    }
}

void CuteDslFFPARunner::unloadKernelModule()
{
    std::lock_guard<std::mutex> lock{sMutex};
    if (!sLoaded)
    {
        return;
    }

#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
    ffpa_d512_causal_gqa16_Kernel_Module_Unload(&sD512CausalGqa16Module);
    sD512CausalGqa16Module = {};
#endif
#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
    ffpa_d512_causal_gqa8_Kernel_Module_Unload(&sD512CausalGqa8Module);
    sD512CausalGqa8Module = {};
#endif
#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
    ffpa_d512_causal_gqa4_Kernel_Module_Unload(&sD512CausalGqa4Module);
    sD512CausalGqa4Module = {};
#endif
    ffpa_d512_causal_Kernel_Module_Unload(&sD512CausalModule);
    sD512CausalModule = {};
#ifdef CUTE_DSL_FFPA_VISIONBLOCK_ENABLED
    ffpa_d512_causal_visionblock_Kernel_Module_Unload(&sD512CausalVisionBlockModule);
    sD512CausalVisionBlockModule = {};
#endif
    sLoaded = false;
}

int CuteDslFFPARunner::run(CuteDslFFPAParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("FFPA CuTe DSL kernel module is not loaded.");
        return -1;
    }

    if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.o == nullptr)
    {
        LOG_ERROR("FFPA CuTe DSL kernel received a null tensor pointer.");
        return -1;
    }

    if (params.cuSeqLenQ == nullptr || params.cuSeqLenK == nullptr)
    {
        LOG_ERROR("FFPA CuTe DSL kernel requires cuSeqLenQ/cuSeqLenK (batchSize + 1) int32 device tensors.");
        return -1;
    }

    bool const useVisionBlock = params.blockBegin != nullptr || params.blockEnd != nullptr;
    if (useVisionBlock && (params.blockBegin == nullptr || params.blockEnd == nullptr))
    {
        LOG_ERROR("FFPA CuTe DSL kernel requires blockBegin and blockEnd to be both set or both null.");
        return -1;
    }
#ifndef CUTE_DSL_FFPA_VISIONBLOCK_ENABLED
    if (useVisionBlock)
    {
        LOG_ERROR(
            "FFPA vision-block overlay requested but the ffpa_d512_causal_visionblock AOT variant is not compiled "
            "into this build.");
        return -1;
    }
#endif

    if (params.batchSize <= 0 || params.seqlenQ <= 0 || params.seqlenK <= 0 || params.numQHeads <= 0
        || params.numKVHeads <= 0 || params.headDim <= 0)
    {
        LOG_ERROR("FFPA CuTe DSL kernel received invalid tensor dimensions.");
        return -1;
    }

    if (params.headDim != 512)
    {
        LOG_ERROR("FFPA CuTe DSL kernel only supports head dimension 512.");
        return -1;
    }

    if (params.numQHeads % params.numKVHeads != 0)
    {
        LOG_ERROR("FFPA d512 causal CuTe DSL kernel requires numQHeads (%d) to be divisible by numKVHeads (%d).",
            params.numQHeads, params.numKVHeads);
        return -1;
    }

    // Strides for Q/O (indexed by numQHeads) and K/V (indexed by numKVHeads).
    int64_t const qStrideBatch
        = static_cast<int64_t>(params.seqlenQ) * static_cast<int64_t>(params.numQHeads) * params.headDim;
    int64_t const qStrideSeq = static_cast<int64_t>(params.numQHeads) * params.headDim;
    int64_t const kStrideBatch
        = static_cast<int64_t>(params.seqlenK) * static_cast<int64_t>(params.numKVHeads) * params.headDim;
    int64_t const kStrideSeq = static_cast<int64_t>(params.numKVHeads) * params.headDim;
    float const softmaxScale
        = params.softmaxScale > 0.0F ? params.softmaxScale : 1.0F / std::sqrt(static_cast<float>(params.headDim));

#ifdef CUTE_DSL_FFPA_VISIONBLOCK_ENABLED
    if (useVisionBlock)
    {
        // Vision-block overlay variant: identical tensor marshalling plus the
        // two [batchSize, seqlenQ] int32 block-interval tensors (compact row
        // stride seqlenQ; element stride statically 1).
        ffpa_d512_causal_visionblock_Tensor_mQ_t qTensor{};
        qTensor.data = const_cast<void*>(params.q);
        qTensor.dynamic_shapes[0] = params.batchSize;
        qTensor.dynamic_shapes[1] = params.seqlenQ;
        qTensor.dynamic_shapes[2] = params.numQHeads;
        qTensor.dynamic_strides[0] = qStrideBatch;
        qTensor.dynamic_strides[1] = qStrideSeq;

        ffpa_d512_causal_visionblock_Tensor_mK_t kTensor{};
        kTensor.data = const_cast<void*>(params.k);
        kTensor.dynamic_shapes[0] = params.batchSize;
        kTensor.dynamic_shapes[1] = params.seqlenK;
        kTensor.dynamic_shapes[2] = params.numKVHeads;
        kTensor.dynamic_strides[0] = kStrideBatch;
        kTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_visionblock_Tensor_mV_t vTensor{};
        vTensor.data = const_cast<void*>(params.v);
        vTensor.dynamic_shapes[0] = params.batchSize;
        vTensor.dynamic_shapes[1] = params.seqlenK;
        vTensor.dynamic_shapes[2] = params.numKVHeads;
        vTensor.dynamic_strides[0] = kStrideBatch;
        vTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_visionblock_Tensor_mO_t oTensor{};
        oTensor.data = params.o;
        oTensor.dynamic_shapes[0] = params.batchSize;
        oTensor.dynamic_shapes[1] = params.seqlenQ;
        oTensor.dynamic_shapes[2] = params.numQHeads;
        oTensor.dynamic_strides[0] = qStrideBatch;
        oTensor.dynamic_strides[1] = qStrideSeq;

        ffpa_d512_causal_visionblock_Tensor_mCuSeqLenQ_t cuSeqLenQTensor{};
        cuSeqLenQTensor.data = const_cast<int32_t*>(params.cuSeqLenQ);
        cuSeqLenQTensor.dynamic_shapes[0] = params.batchSize + 1;

        ffpa_d512_causal_visionblock_Tensor_mCuSeqLenK_t cuSeqLenKTensor{};
        cuSeqLenKTensor.data = const_cast<int32_t*>(params.cuSeqLenK);
        cuSeqLenKTensor.dynamic_shapes[0] = params.batchSize + 1;

        ffpa_d512_causal_visionblock_Tensor_mBlockBegin_t blockBeginTensor{};
        blockBeginTensor.data = const_cast<int32_t*>(params.blockBegin);
        blockBeginTensor.dynamic_shapes[0] = params.batchSize;
        blockBeginTensor.dynamic_shapes[1] = params.seqlenQ;
        blockBeginTensor.dynamic_strides[0] = params.seqlenQ;

        ffpa_d512_causal_visionblock_Tensor_mBlockEnd_t blockEndTensor{};
        blockEndTensor.data = const_cast<int32_t*>(params.blockEnd);
        blockEndTensor.dynamic_shapes[0] = params.batchSize;
        blockEndTensor.dynamic_shapes[1] = params.seqlenQ;
        blockEndTensor.dynamic_strides[0] = params.seqlenQ;

        return cute_dsl_ffpa_d512_causal_visionblock_wrapper(&sD512CausalVisionBlockModule, &qTensor, &kTensor,
            &vTensor, &oTensor, &cuSeqLenQTensor, &cuSeqLenKTensor, &blockBeginTensor, &blockEndTensor, softmaxScale,
            params.numKVHeads, stream);
    }
#endif

    int32_t const kvGroupSize = params.numQHeads / params.numKVHeads;

#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
    if (kvGroupSize == 16)
    {
        ffpa_d512_causal_gqa16_Tensor_mQ_t qTensor{};
        qTensor.data = const_cast<void*>(params.q);
        qTensor.dynamic_shapes[0] = params.batchSize;
        qTensor.dynamic_shapes[1] = params.seqlenQ;
        qTensor.dynamic_shapes[2] = params.numQHeads;
        qTensor.dynamic_strides[0] = qStrideBatch;
        qTensor.dynamic_strides[1] = qStrideSeq;

        ffpa_d512_causal_gqa16_Tensor_mK_t kTensor{};
        kTensor.data = const_cast<void*>(params.k);
        kTensor.dynamic_shapes[0] = params.batchSize;
        kTensor.dynamic_shapes[1] = params.seqlenK;
        kTensor.dynamic_shapes[2] = params.numKVHeads;
        kTensor.dynamic_strides[0] = kStrideBatch;
        kTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_gqa16_Tensor_mV_t vTensor{};
        vTensor.data = const_cast<void*>(params.v);
        vTensor.dynamic_shapes[0] = params.batchSize;
        vTensor.dynamic_shapes[1] = params.seqlenK;
        vTensor.dynamic_shapes[2] = params.numKVHeads;
        vTensor.dynamic_strides[0] = kStrideBatch;
        vTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_gqa16_Tensor_mO_t oTensor{};
        oTensor.data = params.o;
        oTensor.dynamic_shapes[0] = params.batchSize;
        oTensor.dynamic_shapes[1] = params.seqlenQ;
        oTensor.dynamic_shapes[2] = params.numQHeads;
        oTensor.dynamic_strides[0] = qStrideBatch;
        oTensor.dynamic_strides[1] = qStrideSeq;

        // (batchSize + 1) int32 cumulative sequence lengths; stride is statically 1.
        ffpa_d512_causal_gqa16_Tensor_mCuSeqLenQ_t cuSeqLenQTensor{};
        cuSeqLenQTensor.data = const_cast<int32_t*>(params.cuSeqLenQ);
        cuSeqLenQTensor.dynamic_shapes[0] = params.batchSize + 1;

        ffpa_d512_causal_gqa16_Tensor_mCuSeqLenK_t cuSeqLenKTensor{};
        cuSeqLenKTensor.data = const_cast<int32_t*>(params.cuSeqLenK);
        cuSeqLenKTensor.dynamic_shapes[0] = params.batchSize + 1;

        return cute_dsl_ffpa_d512_causal_gqa16_wrapper(&sD512CausalGqa16Module, &qTensor, &kTensor, &vTensor, &oTensor,
            &cuSeqLenQTensor, &cuSeqLenKTensor, softmaxScale, params.numKVHeads, stream);
    }
#endif

#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
    if (kvGroupSize == 8)
    {
        ffpa_d512_causal_gqa8_Tensor_mQ_t qTensor{};
        qTensor.data = const_cast<void*>(params.q);
        qTensor.dynamic_shapes[0] = params.batchSize;
        qTensor.dynamic_shapes[1] = params.seqlenQ;
        qTensor.dynamic_shapes[2] = params.numQHeads;
        qTensor.dynamic_strides[0] = qStrideBatch;
        qTensor.dynamic_strides[1] = qStrideSeq;

        ffpa_d512_causal_gqa8_Tensor_mK_t kTensor{};
        kTensor.data = const_cast<void*>(params.k);
        kTensor.dynamic_shapes[0] = params.batchSize;
        kTensor.dynamic_shapes[1] = params.seqlenK;
        kTensor.dynamic_shapes[2] = params.numKVHeads;
        kTensor.dynamic_strides[0] = kStrideBatch;
        kTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_gqa8_Tensor_mV_t vTensor{};
        vTensor.data = const_cast<void*>(params.v);
        vTensor.dynamic_shapes[0] = params.batchSize;
        vTensor.dynamic_shapes[1] = params.seqlenK;
        vTensor.dynamic_shapes[2] = params.numKVHeads;
        vTensor.dynamic_strides[0] = kStrideBatch;
        vTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_gqa8_Tensor_mO_t oTensor{};
        oTensor.data = params.o;
        oTensor.dynamic_shapes[0] = params.batchSize;
        oTensor.dynamic_shapes[1] = params.seqlenQ;
        oTensor.dynamic_shapes[2] = params.numQHeads;
        oTensor.dynamic_strides[0] = qStrideBatch;
        oTensor.dynamic_strides[1] = qStrideSeq;

        // (batchSize + 1) int32 cumulative sequence lengths; stride is statically 1.
        ffpa_d512_causal_gqa8_Tensor_mCuSeqLenQ_t cuSeqLenQTensor{};
        cuSeqLenQTensor.data = const_cast<int32_t*>(params.cuSeqLenQ);
        cuSeqLenQTensor.dynamic_shapes[0] = params.batchSize + 1;

        ffpa_d512_causal_gqa8_Tensor_mCuSeqLenK_t cuSeqLenKTensor{};
        cuSeqLenKTensor.data = const_cast<int32_t*>(params.cuSeqLenK);
        cuSeqLenKTensor.dynamic_shapes[0] = params.batchSize + 1;

        return cute_dsl_ffpa_d512_causal_gqa8_wrapper(&sD512CausalGqa8Module, &qTensor, &kTensor, &vTensor, &oTensor,
            &cuSeqLenQTensor, &cuSeqLenKTensor, softmaxScale, params.numKVHeads, stream);
    }
#endif

#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
    if (kvGroupSize == 4)
    {
        ffpa_d512_causal_gqa4_Tensor_mQ_t qTensor{};
        qTensor.data = const_cast<void*>(params.q);
        qTensor.dynamic_shapes[0] = params.batchSize;
        qTensor.dynamic_shapes[1] = params.seqlenQ;
        qTensor.dynamic_shapes[2] = params.numQHeads;
        qTensor.dynamic_strides[0] = qStrideBatch;
        qTensor.dynamic_strides[1] = qStrideSeq;

        ffpa_d512_causal_gqa4_Tensor_mK_t kTensor{};
        kTensor.data = const_cast<void*>(params.k);
        kTensor.dynamic_shapes[0] = params.batchSize;
        kTensor.dynamic_shapes[1] = params.seqlenK;
        kTensor.dynamic_shapes[2] = params.numKVHeads;
        kTensor.dynamic_strides[0] = kStrideBatch;
        kTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_gqa4_Tensor_mV_t vTensor{};
        vTensor.data = const_cast<void*>(params.v);
        vTensor.dynamic_shapes[0] = params.batchSize;
        vTensor.dynamic_shapes[1] = params.seqlenK;
        vTensor.dynamic_shapes[2] = params.numKVHeads;
        vTensor.dynamic_strides[0] = kStrideBatch;
        vTensor.dynamic_strides[1] = kStrideSeq;

        ffpa_d512_causal_gqa4_Tensor_mO_t oTensor{};
        oTensor.data = params.o;
        oTensor.dynamic_shapes[0] = params.batchSize;
        oTensor.dynamic_shapes[1] = params.seqlenQ;
        oTensor.dynamic_shapes[2] = params.numQHeads;
        oTensor.dynamic_strides[0] = qStrideBatch;
        oTensor.dynamic_strides[1] = qStrideSeq;

        // (batchSize + 1) int32 cumulative sequence lengths; stride is statically 1.
        ffpa_d512_causal_gqa4_Tensor_mCuSeqLenQ_t cuSeqLenQTensor{};
        cuSeqLenQTensor.data = const_cast<int32_t*>(params.cuSeqLenQ);
        cuSeqLenQTensor.dynamic_shapes[0] = params.batchSize + 1;

        ffpa_d512_causal_gqa4_Tensor_mCuSeqLenK_t cuSeqLenKTensor{};
        cuSeqLenKTensor.data = const_cast<int32_t*>(params.cuSeqLenK);
        cuSeqLenKTensor.dynamic_shapes[0] = params.batchSize + 1;

        return cute_dsl_ffpa_d512_causal_gqa4_wrapper(&sD512CausalGqa4Module, &qTensor, &kTensor, &vTensor, &oTensor,
            &cuSeqLenQTensor, &cuSeqLenKTensor, softmaxScale, params.numKVHeads, stream);
    }
#endif

    // Only MHA (kvGroupSize == 1) falls through to the base kernel.
    // Other group sizes without a specialized variant are unsupported.
    if (kvGroupSize != 1)
    {
        LOG_ERROR(
            "FFPA d512 causal CuTe DSL kernel: GQA group size %d is not supported "
            "(no compiled variant). Supported: 1 (MHA)%s%s%s.",
            kvGroupSize,
#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
            ", 4"
#else
            ""
#endif
            ,
#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
            ", 8"
#else
            ""
#endif
            ,
#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
            ", 16"
#else
            ""
#endif
        );
        return -1;
    }

    // MHA (kvGroupSize == 1) uses the base kernel.
    // The H-stride is statically D and the D-stride is statically 1, so neither
    // is passed across the ABI.
    ffpa_d512_causal_Tensor_mQ_t qTensor{};
    qTensor.data = const_cast<void*>(params.q);
    qTensor.dynamic_shapes[0] = params.batchSize;
    qTensor.dynamic_shapes[1] = params.seqlenQ;
    qTensor.dynamic_shapes[2] = params.numQHeads;
    qTensor.dynamic_strides[0] = qStrideBatch;
    qTensor.dynamic_strides[1] = qStrideSeq;

    ffpa_d512_causal_Tensor_mK_t kTensor{};
    kTensor.data = const_cast<void*>(params.k);
    kTensor.dynamic_shapes[0] = params.batchSize;
    kTensor.dynamic_shapes[1] = params.seqlenK;
    kTensor.dynamic_shapes[2] = params.numKVHeads;
    kTensor.dynamic_strides[0] = kStrideBatch;
    kTensor.dynamic_strides[1] = kStrideSeq;

    ffpa_d512_causal_Tensor_mV_t vTensor{};
    vTensor.data = const_cast<void*>(params.v);
    vTensor.dynamic_shapes[0] = params.batchSize;
    vTensor.dynamic_shapes[1] = params.seqlenK;
    vTensor.dynamic_shapes[2] = params.numKVHeads;
    vTensor.dynamic_strides[0] = kStrideBatch;
    vTensor.dynamic_strides[1] = kStrideSeq;

    ffpa_d512_causal_Tensor_mO_t oTensor{};
    oTensor.data = params.o;
    oTensor.dynamic_shapes[0] = params.batchSize;
    oTensor.dynamic_shapes[1] = params.seqlenQ;
    oTensor.dynamic_shapes[2] = params.numQHeads;
    oTensor.dynamic_strides[0] = qStrideBatch;
    oTensor.dynamic_strides[1] = qStrideSeq;

    // (batchSize + 1) int32 cumulative sequence lengths; stride is statically 1.
    ffpa_d512_causal_Tensor_mCuSeqLenQ_t cuSeqLenQTensor{};
    cuSeqLenQTensor.data = const_cast<int32_t*>(params.cuSeqLenQ);
    cuSeqLenQTensor.dynamic_shapes[0] = params.batchSize + 1;

    ffpa_d512_causal_Tensor_mCuSeqLenK_t cuSeqLenKTensor{};
    cuSeqLenKTensor.data = const_cast<int32_t*>(params.cuSeqLenK);
    cuSeqLenKTensor.dynamic_shapes[0] = params.batchSize + 1;

    return cute_dsl_ffpa_d512_causal_wrapper(&sD512CausalModule, &qTensor, &kTensor, &vTensor, &oTensor,
        &cuSeqLenQTensor, &cuSeqLenKTensor, softmaxScale, params.numKVHeads, stream);
}

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_FFPA_ENABLED)
