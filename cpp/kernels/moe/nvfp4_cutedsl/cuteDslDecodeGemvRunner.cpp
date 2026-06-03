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

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED

#include "kernels/moe/nvfp4_cutedsl/cuteDslDecodeGemvRunner.h"
#include "common/logger.h"

#include <cstring>

namespace trt_edgellm
{

// Static member definitions.
gemv_up_none_Kernel_Module_t CuteDslDecodeGemvRunner::sModUpNone{};
gemv_up_swiglu_Kernel_Module_t CuteDslDecodeGemvRunner::sModUpSwiglu{};
gemv_dn_relu2_Kernel_Module_t CuteDslDecodeGemvRunner::sModDnRelu2{};
gemv_dn_silu_Kernel_Module_t CuteDslDecodeGemvRunner::sModDnSilu{};
gemv_dn_none_Kernel_Module_t CuteDslDecodeGemvRunner::sModDnNone{};

bool CuteDslDecodeGemvRunner::sLoaded{false};
std::mutex CuteDslDecodeGemvRunner::sLoadMutex{};

bool CuteDslDecodeGemvRunner::loadKernelModules()
{
    std::lock_guard<std::mutex> lock(sLoadMutex);
    if (sLoaded)
    {
        return true;
    }

    gemv_up_none_Kernel_Module_Load(&sModUpNone);
    gemv_up_swiglu_Kernel_Module_Load(&sModUpSwiglu);
    gemv_dn_relu2_Kernel_Module_Load(&sModDnRelu2);
    gemv_dn_silu_Kernel_Module_Load(&sModDnSilu);
    gemv_dn_none_Kernel_Module_Load(&sModDnNone);
    sLoaded = true;
    LOG_DEBUG("CuteDslDecodeGemvRunner: loaded 5 AOT kernel modules");
    return true;
}

void CuteDslDecodeGemvRunner::unloadKernelModules()
{
    std::lock_guard<std::mutex> lock(sLoadMutex);
    if (!sLoaded)
    {
        return;
    }

    gemv_up_none_Kernel_Module_Unload(&sModUpNone);
    gemv_up_swiglu_Kernel_Module_Unload(&sModUpSwiglu);
    gemv_dn_relu2_Kernel_Module_Unload(&sModDnRelu2);
    gemv_dn_silu_Kernel_Module_Unload(&sModDnSilu);
    gemv_dn_none_Kernel_Module_Unload(&sModDnNone);
    sLoaded = false;
}

size_t CuteDslDecodeGemvRunner::getWorkspaceSize(int32_t numTokens, int32_t topK, int32_t moeInterSize)
{
    // One intermediate buffer [numTokens * topK, moeInterSize] fp16.
    // Both gated and non-gated use the same size — the fused up_swiglu kernel
    // produces a single [nT*topK, I] output (SiLU(gate) * up already applied).
    return static_cast<size_t>(numTokens) * static_cast<size_t>(topK) * static_cast<size_t>(moeInterSize)
        * sizeof(__half);
}

int32_t CuteDslDecodeGemvRunner::run(CuteDslDecodeGemvParams const& params, void* workspace, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner::run: kernel modules not loaded");
        return -1;
    }
    return params.isGated ? runGated(params, workspace, stream) : runNonGated(params, workspace, stream);
}

// ============================================================================
// Non-gated pipeline: up_none → dn_{relu2|silu}
// ============================================================================

int32_t CuteDslDecodeGemvRunner::runNonGated(
    CuteDslDecodeGemvParams const& params, void* workspace, cudaStream_t stream)
{
    int32_t const H = params.hiddenSize;
    int32_t const I = params.moeInterSize;
    int32_t const nT = params.numTokens;
    int32_t const topK = params.topK;

    __half* interBuf = static_cast<__half*>(workspace);
    size_t const interBytes = static_cast<size_t>(nT) * topK * I * sizeof(__half);

    // Zero intermediate buffer (down_proj uses atomicAdd).
    cudaError_t err = cudaMemsetAsync(interBuf, 0, interBytes, stream);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: cudaMemsetAsync failed: %s", cudaGetErrorString(err));
        return -1;
    }

    // Up-proj: input [nT, H] → intermediate [nT*topK, I].
    // Kernel: activation=none, gated=false, output_atomic=false.
    // input2 is unused (not gated); pass same pointer.
    cute_dsl_gemv_up_none_wrapper(&sModUpNone,
        const_cast<void*>(static_cast<void const*>(params.hiddenStates)),  // input
        const_cast<void*>(static_cast<void const*>(params.hiddenStates)),  // input2 (unused)
        const_cast<void*>(params.upWeights),                               // weights [E, H/2, I]
        const_cast<void*>(params.upScales),                                // scales [E, ceil(H/16), I] fp8
        const_cast<void*>(static_cast<void const*>(params.upGlobalScale)), // global_scale [E] fp32
        const_cast<void*>(static_cast<void const*>(params.topkIds)),       // expert_ids
        const_cast<void*>(static_cast<void const*>(params.topkWeights)),   // gate_scores (unused by up_none)
        static_cast<void*>(interBuf),                                      // output [nT*topK, I]
        H, I, topK, nT, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: up_proj launch failed: %s", cudaGetErrorString(err));
        return -1;
    }

    // Zero output buffer (down_proj uses atomicAdd to accumulate topK experts).
    size_t const outBytes = static_cast<size_t>(nT) * H * sizeof(__half);
    err = cudaMemsetAsync(params.output, 0, outBytes, stream);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: output memset failed: %s", cudaGetErrorString(err));
        return -1;
    }

    // Down-proj: intermediate [nT*topK, I] → output [nT, H] (atomic accumulation).
    // Kernel: activation={relu2|silu}, gated=false, output_atomic=true.
    // input2 is unused (not gated); pass same pointer.
    if (params.activationKind == 0) // ReLU2
    {
        cute_dsl_gemv_dn_relu2_wrapper(&sModDnRelu2,
            static_cast<void*>(interBuf),                                        // input [nT*topK, I]
            static_cast<void*>(interBuf),                                        // input2 (unused)
            const_cast<void*>(params.downWeights),                               // weights [E, I/2, H]
            const_cast<void*>(params.downScales),                                // scales [E, ceil(I/16), H] fp8
            const_cast<void*>(static_cast<void const*>(params.downGlobalScale)), // global_scale [E] fp32
            const_cast<void*>(static_cast<void const*>(params.topkIds)),         // expert_ids
            const_cast<void*>(static_cast<void const*>(params.topkWeights)),     // gate_scores
            static_cast<void*>(params.output),                                   // output [nT, H]
            I, H, topK, nT, stream);
    }
    else // SiLU
    {
        cute_dsl_gemv_dn_silu_wrapper(&sModDnSilu, static_cast<void*>(interBuf), static_cast<void*>(interBuf),
            const_cast<void*>(params.downWeights),
            const_cast<void*>(params.downScales),                                // scales fp8
            const_cast<void*>(static_cast<void const*>(params.downGlobalScale)), // global_scale fp32
            const_cast<void*>(static_cast<void const*>(params.topkIds)),
            const_cast<void*>(static_cast<void const*>(params.topkWeights)), static_cast<void*>(params.output), I, H,
            topK, nT, stream);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: down_proj launch failed: %s", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// ============================================================================
// Gated pipeline: up_swiglu → dn_none
// ============================================================================

int32_t CuteDslDecodeGemvRunner::runGated(CuteDslDecodeGemvParams const& params, void* workspace, cudaStream_t stream)
{
    int32_t const H = params.hiddenSize;
    int32_t const I = params.moeInterSize;
    int32_t const nT = params.numTokens;
    int32_t const topK = params.topK;

    __half* interBuf = static_cast<__half*>(workspace);
    size_t const interBytes = static_cast<size_t>(nT) * topK * I * sizeof(__half);

    // Zero intermediate buffer.
    cudaError_t err = cudaMemsetAsync(interBuf, 0, interBytes, stream);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: gated workspace memset failed: %s", cudaGetErrorString(err));
        return -1;
    }

    // Fused gate+up with SwiGLU: input [nT, H] × interleaved_weights [E, H/2, 2*I]
    //   → intermediate [nT*topK, I] = SiLU(gate_proj) * up_proj.
    // Kernel: NvFP4MoeGemvSwigluKernel, output_atomic=false (direct store).
    // N arg = I (output dim); kernel internally uses N_full = 2*I.
    cute_dsl_gemv_up_swiglu_wrapper(&sModUpSwiglu,
        const_cast<void*>(static_cast<void const*>(params.hiddenStates)),  // input [nT, H]
        const_cast<void*>(static_cast<void const*>(params.hiddenStates)),  // input2 (unused)
        const_cast<void*>(params.upWeights),                               // weights [E, H/2, 2*I]
        const_cast<void*>(params.upScales),                                // scales (atom-layout, covers 2*I)
        const_cast<void*>(static_cast<void const*>(params.upGlobalScale)), // global_scale [E]
        const_cast<void*>(static_cast<void const*>(params.topkIds)),
        const_cast<void*>(static_cast<void const*>(params.topkWeights)), // unused by kernel
        static_cast<void*>(interBuf),                                    // output [nT*topK, I]
        H, I, topK, nT, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: up_swiglu launch failed: %s", cudaGetErrorString(err));
        return -1;
    }

    // Zero output buffer (down_proj uses atomicAdd to accumulate topK experts).
    size_t const outBytes = static_cast<size_t>(nT) * H * sizeof(__half);
    err = cudaMemsetAsync(params.output, 0, outBytes, stream);
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: output memset failed: %s", cudaGetErrorString(err));
        return -1;
    }

    // Down-proj: intermediate [nT*topK, I] → output [nT, H] (atomic accumulation).
    // Kernel: activation=none, gated=false, output_atomic=true.
    cute_dsl_gemv_dn_none_wrapper(&sModDnNone,
        static_cast<void*>(interBuf),                                        // input [nT*topK, I]
        static_cast<void*>(interBuf),                                        // input2 (unused)
        const_cast<void*>(params.downWeights),                               // weights [E, I/2, H]
        const_cast<void*>(params.downScales),                                // scales fp8
        const_cast<void*>(static_cast<void const*>(params.downGlobalScale)), // global_scale fp32
        const_cast<void*>(static_cast<void const*>(params.topkIds)),         // expert_ids
        const_cast<void*>(static_cast<void const*>(params.topkWeights)),     // gate_scores
        static_cast<void*>(params.output),                                   // output [nT, H]
        I, H, topK, nT, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        LOG_ERROR("CuteDslDecodeGemvRunner: dn_none launch failed: %s", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_NVFP4_MOE_ENABLED
