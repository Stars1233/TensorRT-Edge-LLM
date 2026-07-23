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

#include "cutedsl_all.h"

#include <NvInferRuntime.h>
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
     * Output is always FP16. Selects kernel variant based on fp8Input:
     *   - fp8Input=false → FP16 kernels (all scales ignored)
     *   - fp8Input=true  → FP8-input / FP16-output kernels
     *
     * @param qPtr Query [B, S_q, H_q, D]
     * @param kvPtr Combined KV cache [B, 2, H_kv, Cap, D]
     * @param oPtr Output [B, S_q, H_q, D] (always FP16)
     * @param cuKVSeqLens Cumulative KV sequence lengths [B+1]
     * @param stream CUDA stream
     * @param attentionScale Model-defined multiplier applied to QK^T before softmax. For FP8 input, the effective
     *        softmax scale is attentionScale * qScale * kScale.
     * @param slidingWindowSize Sliding window size (INT_MAX = disabled)
     * @param fp8Input Whether Q/KV are FP8 E4M3
     * @param qScale Q dequant scale (quant→orig), ignored when fp8Input=false
     * @param kScale K dequant scale (quant→orig), ignored when fp8Input=false
     * @param vScale V dequant scale (quant→orig), applied to the attention output and ignored when fp8Input=false
     * @param enableSkipSoftmax Dispatch the skip-softmax (BLASST) kernel variant, which skips the P*V GEMM of
     *        KV tiles whose contribution is negligible (threshold baked at export). Approximate — outputs may
     *        deviate from dense by up to the calibrated accuracy gate. FP16 causal only: incompatible with
     *        fp8Input and slidingWindowSize.
     */
    void run(void const* qPtr, void const* kvPtr, void* oPtr, int32_t const* cuKVSeqLens, cudaStream_t stream,
        float attentionScale, int32_t slidingWindowSize = INT_MAX, bool fp8Input = false, float qScale = 1.0F,
        float kScale = 1.0F, float vScale = 1.0F, bool enableSkipSoftmax = false);

    /**
     * @brief LLM FMHA over a paged KV cache.
     *
     * Dispatches a dedicated CuTe DSL AOT variant that reads K/V directly from
     * a paged pool using kvCachePageList. The logical descriptor shape is
     * [numPages, H_kv, tokensPerPage, D], while the physical pool is fixed to
     * NHD [numPages, tokensPerPage, H_kv, D]. This path maps one logical K/V
     * TMA tile to one physical page. The current CuTe DSL variants use a K/V
     * tile width of 128 tokens, so tokensPerPage must be 128 to avoid
     * multi-page tile stitching or a gather workspace.
     *
     * @param qPtr Query [B, S_q, H_q, D]
     * @param pagedKVPoolPtr Paged KV pool [numPages, tokensPerPage, H_kv, D]
     * @param kvCachePageList Page table [B, 2, maxPagesPerSeq], K pages then V pages
     * @param oPtr Output [B, S_q, H_q, D] (always FP16)
     * @param cuKVSeqLens Cumulative KV sequence lengths [B+1]
     * @param numPages Number of pages in the paged KV pool
     * @param maxPagesPerSeq Max logical pages per sequence
     * @param tokensPerPage Number of tokens per page
     * @param kvDataType Paged KV cache dtype (FP16 or FP8)
     * @param stream CUDA stream
     * @param slidingWindowSize Sliding window size (INT_MAX = disabled)
     * @param fp8Input Whether Q/KV are FP8 E4M3
     * @param qScale Q dequant scale, ignored when fp8Input=false
     * @param kScale K dequant scale, ignored when fp8Input=false
     * @param vScale V dequant scale, ignored when fp8Input=false
     */
    void runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList, void* oPtr,
        int32_t const* cuKVSeqLens, int32_t numPages, int32_t maxPagesPerSeq, int32_t tokensPerPage,
        nvinfer1::DataType kvDataType, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize = INT_MAX,
        bool fp8Input = false, float qScale = 1.0f, float kScale = 1.0f, float vScale = 1.0f);

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
     * @param attentionScale Absolute multiplier applied to QK^T before softmax
     */
    void run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
        int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream, float attentionScale);

private:
    int32_t mBatchSize{};
    int32_t mSeqLenQ{};
    int32_t mKVCacheCapacity{};
    int32_t mNumHeadsQ{};
    int32_t mNumHeadsK{};
    int32_t mHeadDim{};

    // LLM kernel modules (FP16)
    static fmha_d64_Kernel_Module_t sLLM_d64;
    static fmha_d128_Kernel_Module_t sLLM_d128;
    static fmha_d256_Kernel_Module_t sLLM_d256;
    static fmha_d64_sw_Kernel_Module_t sLLM_d64_sw;
    static fmha_d128_sw_Kernel_Module_t sLLM_d128_sw;
    static fmha_d256_sw_Kernel_Module_t sLLM_d256_sw;

    // LLM skip-softmax (BLASST) kernel modules (FP16, causal, no sliding window)
    static fmha_d64_skipsoftmax_Kernel_Module_t sLLM_d64_skipsoftmax;
    static fmha_d128_skipsoftmax_Kernel_Module_t sLLM_d128_skipsoftmax;

    // LLM kernel modules (FP8 input, FP16 output)
    static fmha_d64_fp8_Kernel_Module_t sLLM_d64_fp8;
    static fmha_d128_fp8_Kernel_Module_t sLLM_d128_fp8;
    static fmha_d256_fp8_Kernel_Module_t sLLM_d256_fp8;
    static fmha_d64_sw_fp8_Kernel_Module_t sLLM_d64_sw_fp8;
    static fmha_d128_sw_fp8_Kernel_Module_t sLLM_d128_sw_fp8;
    static fmha_d256_sw_fp8_Kernel_Module_t sLLM_d256_sw_fp8;

    // LLM paged KV cache kernel modules (FP16)
    static fmha_d64_paged_Kernel_Module_t sLLM_d64_paged;
    static fmha_d128_paged_Kernel_Module_t sLLM_d128_paged;
    static fmha_d256_paged_Kernel_Module_t sLLM_d256_paged;
    static fmha_d64_sw_paged_Kernel_Module_t sLLM_d64_sw_paged;
    static fmha_d128_sw_paged_Kernel_Module_t sLLM_d128_sw_paged;
    static fmha_d256_sw_paged_Kernel_Module_t sLLM_d256_sw_paged;

    // LLM paged KV cache kernel modules (FP8 input, FP16 output)
    static fmha_d64_paged_fp8_Kernel_Module_t sLLM_d64_paged_fp8;
    static fmha_d128_paged_fp8_Kernel_Module_t sLLM_d128_paged_fp8;
    static fmha_d256_paged_fp8_Kernel_Module_t sLLM_d256_paged_fp8;
    static fmha_d64_sw_paged_fp8_Kernel_Module_t sLLM_d64_sw_paged_fp8;
    static fmha_d128_sw_paged_fp8_Kernel_Module_t sLLM_d128_sw_paged_fp8;
    static fmha_d256_sw_paged_fp8_Kernel_Module_t sLLM_d256_sw_paged_fp8;

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
