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

// Implementation of the contiguous grouped GEMM runner.
//
// Dispatches to activation-specific AOT kernels (identity/relu2).
// Each activation is compiled as a separate kernel binary because activation
// is a compile-time constant in the CuteDSL epilogue.

#include "NvFP4MoEContiguousGemmRunner.h"

#include "common/checkMacros.h"
#include <stdexcept>
#include <string>

namespace trt_edgellm
{
namespace kernel
{
namespace nvfp4_moe
{

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
nvfp4_moe_fc1_relu2_n128_bf16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sRelu2N128_bf16{};
nvfp4_moe_fc1_relu2_n256_bf16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sRelu2N256_bf16{};
nvfp4_moe_fc1_swiglu_n128_bf16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sSwigluN128_bf16{};
nvfp4_moe_fc1_swiglu_n256_bf16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sSwigluN256_bf16{};
nvfp4_moe_fc1_relu2_n128_fp16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sRelu2N128_fp16{};
nvfp4_moe_fc1_relu2_n256_fp16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sRelu2N256_fp16{};
nvfp4_moe_fc1_swiglu_n128_fp16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sSwigluN128_fp16{};
nvfp4_moe_fc1_swiglu_n256_fp16_Kernel_Module_t NvFP4MoEContiguousGemmRunner::sSwigluN256_fp16{};
#endif
bool NvFP4MoEContiguousGemmRunner::sLoaded = false;
std::mutex NvFP4MoEContiguousGemmRunner::sMutex{};

NvFP4MoEContiguousGemmRunner::NvFP4MoEContiguousGemmRunner(int32_t numLocalExperts, int32_t topK, int32_t n, int32_t k,
    int32_t tileSize, Activation activation, OutputDType outDtype)
    : mNumLocalExperts(numLocalExperts)
    , mTopK(topK)
    , mN(n)
    , mK(k)
    , mTileSize(tileSize)
    , mActivation(activation)
    , mOutDtype(outDtype)
{
}

bool NvFP4MoEContiguousGemmRunner::loadKernelModules()
{
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    std::lock_guard<std::mutex> lock(sMutex);
    if (sLoaded)
        return true;
    nvfp4_moe_fc1_relu2_n128_bf16_Kernel_Module_Load(&sRelu2N128_bf16);
    nvfp4_moe_fc1_relu2_n256_bf16_Kernel_Module_Load(&sRelu2N256_bf16);
    nvfp4_moe_fc1_swiglu_n128_bf16_Kernel_Module_Load(&sSwigluN128_bf16);
    nvfp4_moe_fc1_swiglu_n256_bf16_Kernel_Module_Load(&sSwigluN256_bf16);
    nvfp4_moe_fc1_relu2_n128_fp16_Kernel_Module_Load(&sRelu2N128_fp16);
    nvfp4_moe_fc1_relu2_n256_fp16_Kernel_Module_Load(&sRelu2N256_fp16);
    nvfp4_moe_fc1_swiglu_n128_fp16_Kernel_Module_Load(&sSwigluN128_fp16);
    nvfp4_moe_fc1_swiglu_n256_fp16_Kernel_Module_Load(&sSwigluN256_fp16);
    sLoaded = true;
    return true;
#else
    return false;
#endif
}

void NvFP4MoEContiguousGemmRunner::unloadKernelModules()
{
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    std::lock_guard<std::mutex> lock(sMutex);
    if (!sLoaded)
        return;
    nvfp4_moe_fc1_relu2_n128_bf16_Kernel_Module_Unload(&sRelu2N128_bf16);
    nvfp4_moe_fc1_relu2_n256_bf16_Kernel_Module_Unload(&sRelu2N256_bf16);
    nvfp4_moe_fc1_swiglu_n128_bf16_Kernel_Module_Unload(&sSwigluN128_bf16);
    nvfp4_moe_fc1_swiglu_n256_bf16_Kernel_Module_Unload(&sSwigluN256_bf16);
    nvfp4_moe_fc1_relu2_n128_fp16_Kernel_Module_Unload(&sRelu2N128_fp16);
    nvfp4_moe_fc1_relu2_n256_fp16_Kernel_Module_Unload(&sRelu2N256_fp16);
    nvfp4_moe_fc1_swiglu_n128_fp16_Kernel_Module_Unload(&sSwigluN128_fp16);
    nvfp4_moe_fc1_swiglu_n256_fp16_Kernel_Module_Unload(&sSwigluN256_fp16);
    sLoaded = false;
#endif
}

int32_t NvFP4MoEContiguousGemmRunner::selectTactic(int64_t n, int64_t k)
{
    if (k % 32 == 0 && n % 32 == 0)
        return 128;
    return 0;
}

void NvFP4MoEContiguousGemmRunner::run(void const* gatheredFP4, void const* weight, void const* gatheredSF,
    void const* weightSF, void* output, void const* alpha, MoELayout const& layout, int64_t permutedM,
    cudaStream_t stream)
{
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    ELLM_CHECK(sLoaded, "kernel modules not loaded");

    int32_t const tactic = selectTactic(mN, mK);
    ELLM_CHECK(tactic > 0, "no valid tactic for N=" + std::to_string(mN) + " K=" + std::to_string(mK));

    bool const isSwiglu = (mActivation == Activation::kSwiglu);
    int64_t const n = static_cast<int64_t>(mN);
    int64_t const n_out = isSwiglu ? n / 2 : n;
    int64_t const k = static_cast<int64_t>(mK);
    int64_t const l = static_cast<int64_t>(mNumLocalExperts);

    void* a = const_cast<void*>(gatheredFP4);
    void* b = const_cast<void*>(weight);
    void* asf = const_cast<void*>(gatheredSF);
    void* bsf = const_cast<void*>(weightSF);
    void* alp = const_cast<void*>(alpha);
    void* tg = layout.tileIdxToGroupIdx;
    void* tm = layout.tileIdxToMnLimit;
    void* nt = layout.numNonExitingTiles;

    bool const isFP16 = (mOutDtype == OutputDType::kFP16);
    if (mActivation == Activation::kRelu2)
    {
        if (tactic == 128)
        {
            if (isFP16)
            {
                cute_dsl_nvfp4_moe_fc1_relu2_n128_fp16_wrapper(
                    &sRelu2N128_fp16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
            else
            {
                cute_dsl_nvfp4_moe_fc1_relu2_n128_bf16_wrapper(
                    &sRelu2N128_bf16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
        }
        else
        {
            if (isFP16)
            {
                cute_dsl_nvfp4_moe_fc1_relu2_n256_fp16_wrapper(
                    &sRelu2N256_fp16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
            else
            {
                cute_dsl_nvfp4_moe_fc1_relu2_n256_bf16_wrapper(
                    &sRelu2N256_bf16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
        }
    }
    else if (mActivation == Activation::kSwiglu)
    {
        if (tactic == 128)
        {
            if (isFP16)
            {
                cute_dsl_nvfp4_moe_fc1_swiglu_n128_fp16_wrapper(
                    &sSwigluN128_fp16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
            else
            {
                cute_dsl_nvfp4_moe_fc1_swiglu_n128_bf16_wrapper(
                    &sSwigluN128_bf16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
        }
        else
        {
            if (isFP16)
            {
                cute_dsl_nvfp4_moe_fc1_swiglu_n256_fp16_wrapper(
                    &sSwigluN256_fp16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
            else
            {
                cute_dsl_nvfp4_moe_fc1_swiglu_n256_bf16_wrapper(
                    &sSwigluN256_bf16, a, b, asf, bsf, output, alp, tg, tm, nt, permutedM, n, n_out, k, l, stream);
            }
        }
    }
    else
    {
        throw std::runtime_error(
            "NvFP4MoEContiguousGemmRunner: activation not supported (only relu2 and swiglu are compiled)");
    }
#else
    (void) gatheredFP4;
    (void) weight;
    (void) gatheredSF;
    (void) weightSF;
    (void) output;
    (void) alpha;
    (void) layout;
    (void) permutedM;
    (void) stream;
    throw std::runtime_error("NvFP4MoEContiguousGemmRunner: decomposed AOT kernels not enabled");
#endif
}

} // namespace nvfp4_moe
} // namespace kernel
} // namespace trt_edgellm
