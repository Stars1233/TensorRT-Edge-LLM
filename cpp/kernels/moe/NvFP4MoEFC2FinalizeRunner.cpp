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

// Implementation of the FC2 finalize runner.
//
// Dispatches to the AOT-compiled FC2 finalize kernel which performs:
// - FP4xFP4 blockscaled grouped GEMM
// - Per-expert alpha scaling and per-token router weight scaling
// - Atomic scatter-reduce to the output buffer
//
// The ABI matches the CuTe tensor export from aot_wrapper():
//   cute_dsl_fc2_finalize_n{128|256}_wrapper(
//       module*,
//       void* a, void* b, void* a_sf, void* b_sf, void* c,
//       void* alpha,
//       void* tile_idx_to_group_idx, void* tile_idx_to_mn_limit,
//       void* permuted_idx_to_expanded_idx, void* num_non_exiting_tiles,
//       void* token_final_scales,
//       int64_t m, int64_t n, int64_t k, int64_t l,
//       int64_t num_tokens, int64_t top_k,
//       stream)

#include "NvFP4MoEFC2FinalizeRunner.h"

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
nvfp4_moe_fc2_n128_bf16_Kernel_Module_t NvFP4MoEFC2FinalizeRunner::sFC2N128_bf16{};
nvfp4_moe_fc2_n256_bf16_Kernel_Module_t NvFP4MoEFC2FinalizeRunner::sFC2N256_bf16{};
nvfp4_moe_fc2_n128_fp16_Kernel_Module_t NvFP4MoEFC2FinalizeRunner::sFC2N128_fp16{};
nvfp4_moe_fc2_n256_fp16_Kernel_Module_t NvFP4MoEFC2FinalizeRunner::sFC2N256_fp16{};
#endif
bool NvFP4MoEFC2FinalizeRunner::sLoaded = false;
std::mutex NvFP4MoEFC2FinalizeRunner::sMutex{};

NvFP4MoEFC2FinalizeRunner::NvFP4MoEFC2FinalizeRunner(
    int32_t numLocalExperts, int32_t topK, int32_t n, int32_t k, OutputDType outDtype)
    : mNumLocalExperts(numLocalExperts)
    , mTopK(topK)
    , mN(n)
    , mK(k)
    , mOutDtype(outDtype)
{
}

bool NvFP4MoEFC2FinalizeRunner::loadKernelModules()
{
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    std::lock_guard<std::mutex> lock(sMutex);
    if (sLoaded)
        return true;

    nvfp4_moe_fc2_n128_bf16_Kernel_Module_Load(&sFC2N128_bf16);
    nvfp4_moe_fc2_n256_bf16_Kernel_Module_Load(&sFC2N256_bf16);
    nvfp4_moe_fc2_n128_fp16_Kernel_Module_Load(&sFC2N128_fp16);
    nvfp4_moe_fc2_n256_fp16_Kernel_Module_Load(&sFC2N256_fp16);
    sLoaded = true;
    return true;
#else
    return false;
#endif
}

void NvFP4MoEFC2FinalizeRunner::unloadKernelModules()
{
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    std::lock_guard<std::mutex> lock(sMutex);
    if (!sLoaded)
        return;
    nvfp4_moe_fc2_n128_bf16_Kernel_Module_Unload(&sFC2N128_bf16);
    nvfp4_moe_fc2_n256_bf16_Kernel_Module_Unload(&sFC2N256_bf16);
    nvfp4_moe_fc2_n128_fp16_Kernel_Module_Unload(&sFC2N128_fp16);
    nvfp4_moe_fc2_n256_fp16_Kernel_Module_Unload(&sFC2N256_fp16);
    sLoaded = false;
#endif
}

int32_t NvFP4MoEFC2FinalizeRunner::selectTactic(int64_t n, int64_t k)
{
    if (k % 32 == 0 && n % 32 == 0)
    {
        return 128; // n128 always valid and preferred
    }
    return 0;
}

void NvFP4MoEFC2FinalizeRunner::run(void const* inputFP4, void const* weight, void const* inputSF, void const* weightSF,
    void* output, void const* alpha, MoELayout const& layout, void const* tokenFinalScales, int64_t permutedM,
    int64_t numTokens, cudaStream_t stream)
{
#ifdef CUTE_DSL_NVFP4_MOE_ENABLED
    ELLM_CHECK(sLoaded, "kernel modules not loaded");

    int32_t const tactic = selectTactic(mN, mK);
    ELLM_CHECK(tactic > 0, "no valid tactic for N=" + std::to_string(mN) + " K=" + std::to_string(mK));

    int64_t const n = static_cast<int64_t>(mN);
    int64_t const k = static_cast<int64_t>(mK);
    int64_t const l = static_cast<int64_t>(mNumLocalExperts);
    int64_t const topK = static_cast<int64_t>(mTopK);

    bool const isFP16 = (mOutDtype == OutputDType::kFP16);
    if (tactic == 128)
    {
        if (isFP16)
        {
            cute_dsl_nvfp4_moe_fc2_n128_fp16_wrapper(&sFC2N128_fp16, const_cast<void*>(inputFP4),
                const_cast<void*>(weight), const_cast<void*>(inputSF), const_cast<void*>(weightSF), output,
                const_cast<void*>(alpha), layout.tileIdxToGroupIdx, layout.tileIdxToMnLimit,
                layout.permutedIdxToExpandedIdx, layout.numNonExitingTiles, const_cast<void*>(tokenFinalScales),
                permutedM, n, k, l, numTokens, topK, stream);
        }
        else
        {
            cute_dsl_nvfp4_moe_fc2_n128_bf16_wrapper(&sFC2N128_bf16, const_cast<void*>(inputFP4),
                const_cast<void*>(weight), const_cast<void*>(inputSF), const_cast<void*>(weightSF), output,
                const_cast<void*>(alpha), layout.tileIdxToGroupIdx, layout.tileIdxToMnLimit,
                layout.permutedIdxToExpandedIdx, layout.numNonExitingTiles, const_cast<void*>(tokenFinalScales),
                permutedM, n, k, l, numTokens, topK, stream);
        }
    }
    else
    {
        if (isFP16)
        {
            cute_dsl_nvfp4_moe_fc2_n256_fp16_wrapper(&sFC2N256_fp16, const_cast<void*>(inputFP4),
                const_cast<void*>(weight), const_cast<void*>(inputSF), const_cast<void*>(weightSF), output,
                const_cast<void*>(alpha), layout.tileIdxToGroupIdx, layout.tileIdxToMnLimit,
                layout.permutedIdxToExpandedIdx, layout.numNonExitingTiles, const_cast<void*>(tokenFinalScales),
                permutedM, n, k, l, numTokens, topK, stream);
        }
        else
        {
            cute_dsl_nvfp4_moe_fc2_n256_bf16_wrapper(&sFC2N256_bf16, const_cast<void*>(inputFP4),
                const_cast<void*>(weight), const_cast<void*>(inputSF), const_cast<void*>(weightSF), output,
                const_cast<void*>(alpha), layout.tileIdxToGroupIdx, layout.tileIdxToMnLimit,
                layout.permutedIdxToExpandedIdx, layout.numNonExitingTiles, const_cast<void*>(tokenFinalScales),
                permutedM, n, k, l, numTokens, topK, stream);
        }
    }
#else
    (void) inputFP4;
    (void) weight;
    (void) inputSF;
    (void) weightSF;
    (void) output;
    (void) alpha;
    (void) layout;
    (void) tokenFinalScales;
    (void) permutedM;
    (void) numTokens;
    (void) stream;
    throw std::runtime_error("NvFP4MoEFC2FinalizeRunner: decomposed AOT kernels not enabled");
#endif
}

} // namespace nvfp4_moe
} // namespace kernel
} // namespace trt_edgellm
