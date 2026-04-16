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

#pragma once

#include "cutedsl_fmha_all.h"

#include <climits>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{

/**
 * @brief Unified runner for CuTe DSL compiled FMHA kernels (Blackwell SM100+).
 *
 * Supports two execution modes via separate AOT-compiled kernel variants:
 *
 * 1. LLM prefill/chunked-prefill: batched Q [B,S_q,H_q,D] + combined KV cache
 *    [B,2,H_kv,Cap,D] with causal masking and optional sliding window.
 *
 * 2. ViT: packed varlen separate Q/K/V [total_S,H,D] with cu_seqlens [B+1]
 *    for ragged batching, bidirectional attention (no causal mask).
 *
 * Each mode has its own kernel modules and run() overload.
 */
class CuteDslFMHARunner
{
public:
    CuteDslFMHARunner(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize = 0,
        int32_t seqLenQ = 0, int32_t kvCacheCapacity = 0);

    ~CuteDslFMHARunner() = default;
    CuteDslFMHARunner(CuteDslFMHARunner const&) = delete;
    CuteDslFMHARunner& operator=(CuteDslFMHARunner const&) = delete;

    static bool canImplement(int32_t headSize, int32_t smVersion);
    static bool canImplementViT(int32_t headSize, int32_t smVersion);

    // ---- LLM kernel loading ----
    static bool loadLLMKernelModule();
    static void unloadLLMKernelModule();

    // ---- ViT kernel loading ----
    static bool loadViTKernelModule();
    static void unloadViTKernelModule();

    /**
     * @brief LLM FMHA: batched Q + combined KV cache with causal masking.
     *
     * @param qPtr Query [B, S_q, H_q, D]
     * @param kvPtr Combined KV cache [B, 2, H_kv, Cap, D]
     * @param oPtr Output [B, S_q, H_q, D]
     * @param cuKVSeqLens Cumulative KV sequence lengths [B+1]
     * @param stream CUDA stream
     * @param slidingWindowSize Sliding window size (INT_MAX = disabled)
     */
    void run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens, cudaStream_t stream,
        int32_t slidingWindowSize = INT_MAX);

    /**
     * @brief ViT FMHA: packed varlen separate Q/K/V, bidirectional.
     *
     * @param qPtr  Query  [total_S, H, D]
     * @param kPtr  Key    [total_S, H, D]
     * @param vPtr  Value  [total_S, H, D]
     * @param oPtr  Output [total_S, H, D]
     * @param cuSeqLens Cumulative sequence lengths [B+1]
     * @param totalSeqLen Sum of all sequence lengths
     * @param maxSeqLen Longest individual sequence length
     * @param batchSize Number of sequences
     * @param stream CUDA stream
     */
    void run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
        int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream);

private:
    int32_t mBatchSize{};
    int32_t mSeqLenQ{};
    int32_t mKVCacheCapacity{};
    int32_t mNumHeadsQ{};
    int32_t mNumHeadsK{};
    int32_t mHeadDim{};

    // LLM kernel modules
    static fmha_d64_Kernel_Module_t sLLM_d64;
    static fmha_d128_Kernel_Module_t sLLM_d128;
    static fmha_d64_sw_Kernel_Module_t sLLM_d64_sw;
    static fmha_d128_sw_Kernel_Module_t sLLM_d128_sw;
    static bool sLLMLoaded;
    static std::mutex sLLMMutex;

    // ViT kernel modules
    static vit_fmha_d64_Kernel_Module_t sViT_d64;
    static vit_fmha_d72_Kernel_Module_t sViT_d72;
    static vit_fmha_d80_Kernel_Module_t sViT_d80;
    static vit_fmha_d128_Kernel_Module_t sViT_d128;
    static bool sViTLoaded;
    static std::mutex sViTMutex;
};

} // namespace trt_edgellm
