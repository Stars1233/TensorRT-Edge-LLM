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

#include <cuda.h>
#if defined(TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT)
#include <cuda_runtime.h>
#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
static inline cudaError_t cudaLibraryUnload(cudaLibrary_t lib)
{
    CUresult r = cuLibraryUnload(lib);
    return static_cast<cudaError_t>(r);
}
#endif // CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
#endif // TRT_EDGELLM_CUDA_LIBRARY_T_COMPAT

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "cutedsl_all.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{

/**
 * @brief Runner for CuTe DSL compiled GEMM kernels, replacing cuBLAS for Talker MLP.
 *
 * Provides FP16 GEMM with FP32 accumulation: C = A @ B^T
 * where A is [M, K], B is [N, K] (row-major), C is [M, N].
 *
 * Multiple architecture variants are compiled AOT and selected at runtime
 * based on GPU SM version:
 *   - Ampere (SM 80/86/87): cp.async + MmaF16BF16Op
 *   - Blackwell datacenter (SM 100/101/103/110): tcgen05 + TMA
 *   - Blackwell GeForce (SM 120/121): WGMMA + TMA
 */
class CuteDslGemmRunner
{
public:
    /**
     * @brief Check if CuTe DSL GEMM can run on this GPU.
     * @param smVersion GPU SM version (e.g. 87, 100, 121)
     * @return true if a compiled GEMM variant exists for this SM
     */
    static bool canImplement(int32_t smVersion);

    /// Load the kernel module (thread-safe, idempotent).
    static bool loadKernelModule();

    /// Unload the kernel module.
    static void unloadKernelModule();

    /**
     * @brief Execute GEMM: C[M,N] = A[M,K] @ B[N,K]^T
     *
     * All tensors are FP16, row-major. Accumulation is FP32.
     *
     * @param aPtr Input A [M, K] (FP16, K contiguous)
     * @param bPtr Weight B [N, K] (FP16, K contiguous)
     * @param cPtr Output C [M, N] (FP16, N contiguous)
     * @param M Number of rows in A / C
     * @param N Number of rows in B / columns in C
     * @param K Inner dimension
     * @param stream CUDA stream
     * @return true on success, false if kernel module not loaded or variant unavailable.
     */
    static bool run(
        void const* aPtr, void const* bPtr, void* cPtr, int32_t M, int32_t N, int32_t K, cudaStream_t stream);

    /**
     * @brief Execute GEMM with an FP32 C output: C[M,N] = A[M,K] @ B[N,K]^T.
     *
     * A and B are FP16 (tensor-core operands) and accumulation is FP32 — the same
     * MMA path as run() — but C is written in FP32 (the acc->C cast is a no-op since
     * tcgen05 already accumulates in FP32). The Blackwell ABI does not encode the C
     * element type (the tensor struct's data is void*), so this differs from run()
     * only in the dispatched AOT variant; both share the dispatch3d packing helper.
     *
     * Used by the parakeet online GPU fbank mel GEMM, whose output spans ~17 orders
     * of magnitude and must not lose small values to FP16 storage underflow before
     * the downstream natural log.
     *
     * Dispatches to the gemm_blackwell_small_fp16in_fp32out AOT variant (small 64x128
     * tile — suitable for the small M=nMel of the mel projection). Blackwell DC only;
     * returns false on other architectures or if the variant was not compiled.
     *
     * @param cPtr Output C [M, N] (FP32, N contiguous)
     * @return true on success, false if the module / FP32-out variant is unavailable.
     */
    static bool runFp16inFp32out(
        void const* aPtr, void const* bPtr, void* cPtr, int32_t M, int32_t N, int32_t K, cudaStream_t stream);

    /**
     * @brief Execute fused GEMM + bias + SiLU: C = SiLU(A @ B^T + bias)
     *
     * Uses AOT-compiled fused epilogue kernel on all architectures (Ampere, Blackwell DC,
     * BW GeForce). Falls back to plain GEMM + separate CUDA kernel if the fused variant
     * is not compiled for the current arch.
     *
     * @param biasPtr Bias vector [N] (FP16)
     * @return true on success
     */
    static bool runBiasSiLU(void const* aPtr, void const* bPtr, void* cPtr, void const* biasPtr, int32_t M, int32_t N,
        int32_t K, cudaStream_t stream);

    /**
     * @brief Execute fused GEMM + bias: C = A @ B^T + bias
     *
     * Uses AOT-compiled fused epilogue kernel on all architectures. Falls back to
     * plain GEMM + separate CUDA kernel if the fused variant is not compiled.
     *
     * @param biasPtr Bias vector [N] (FP16)
     * @return true on success
     */
    static bool runBias(void const* aPtr, void const* bPtr, void* cPtr, void const* biasPtr, int32_t M, int32_t N,
        int32_t K, cudaStream_t stream);

private:
    // Kernel modules for each architecture variant.
    // Only one will be loaded at runtime based on detected SM.

#ifdef CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED
    static gemm_ampere_decode_fp16_Kernel_Module_t sAmpereDecodeModule;
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_SMALL_PREFILL_ENABLED
    static gemm_ampere_small_prefill_fp16_Kernel_Module_t sAmpereSmallPrefillModule;
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_PREFILL_ENABLED
    static gemm_ampere_medium_prefill_fp16_Kernel_Module_t sAmpereMediumPrefillModule;
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_LARGE_PREFILL_ENABLED
    static gemm_ampere_large_prefill_fp16_Kernel_Module_t sAmpereLargePrefillModule;
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED
    static gemm_ampere_medium_bias_silu_fp16_Kernel_Module_t sAmpereMediumBiasSiLUModule;
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_ENABLED
    static gemm_ampere_medium_bias_fp16_Kernel_Module_t sAmpereMediumBiasModule;
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_ENABLED
    static gemm_blackwell_fp16_Kernel_Module_t sBlackwellModule;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_SILU_ENABLED
    static gemm_blackwell_bias_silu_fp16_Kernel_Module_t sBlackwellBiasSiLUModule;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_ENABLED
    static gemm_blackwell_bias_fp16_Kernel_Module_t sBlackwellBiasModule;
#endif

// Blackwell DC small-tile (tile=64x128 cluster=(1,2)) variants for M <= 512.
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_ENABLED
    static gemm_blackwell_small_fp16_Kernel_Module_t sBlackwellSmallModule;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_SILU_ENABLED
    static gemm_blackwell_small_bias_silu_fp16_Kernel_Module_t sBlackwellSmallBiasSiLUModule;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_ENABLED
    static gemm_blackwell_small_bias_fp16_Kernel_Module_t sBlackwellSmallBiasModule;
#endif

// Blackwell DC small-tile FP16-in/FP32-out variant (64x128 tile, FP32 C
// epilogue) for the parakeet mel GEMM. See runFp16inFp32out.
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED
    static gemm_blackwell_small_fp16in_fp32out_Kernel_Module_t sBlackwellSmallFp16inFp32outModule;
#endif

// Blackwell DC 2-CTA variants (tile=256x256 cluster=(2,1) use_2cta=True)
// for low-SM-count GPUs (Thor) at M >= sBlackwell2ctaMinM.
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_ENABLED
    static gemm_blackwell_2cta_fp16_Kernel_Module_t sBlackwell2ctaModule;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_SILU_ENABLED
    static gemm_blackwell_2cta_bias_silu_fp16_Kernel_Module_t sBlackwell2ctaBiasSiLUModule;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_ENABLED
    static gemm_blackwell_2cta_bias_fp16_Kernel_Module_t sBlackwell2ctaBiasModule;
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED
    static gemm_bw_geforce_fp16_Kernel_Module_t sBlackwellGeforceModule;
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_SMALL_ENABLED
    static gemm_bw_geforce_small_fp16_Kernel_Module_t sBlackwellGeforceSmallModule;
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_SILU_ENABLED
    static gemm_bw_geforce_bias_silu_fp16_Kernel_Module_t sBlackwellGeforceBiasSiLUModule;
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_ENABLED
    static gemm_bw_geforce_bias_fp16_Kernel_Module_t sBlackwellGeforceBiasModule;
#endif

    enum class Variant : int32_t
    {
        kNone = 0,
        kAmpere = 1,
        kBlackwell = 3,
        kBlackwellGeforce = 4,
    };

    static bool sLoaded;
    static int32_t sActiveVariant;
    // M threshold below which Blackwell GEMM dispatches to the small (64x128)
    // tile variant. Captured at loadKernelModule() time from device SM count
    // since the optimal small-vs-default cutover scales with available SMs:
    // B100/sm100 (144 SMs) wants threshold ~512; Thor/sm110 (20 SMs) wants ~64.
    static int32_t sBlackwellSmallTileMaxM;
    // Min M to dispatch to 2-CTA tile (256x256, cluster (2,1)). Set only on
    // low-SM-count GPUs (e.g. Thor) where larger arithmetic intensity per MMA
    // op beats CTA-count parallelism. 0 disables 2-CTA dispatch entirely.
    static int32_t sBlackwell2ctaMinM;
    static std::mutex sMutex;
};

} // namespace trt_edgellm
