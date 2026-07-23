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

#ifdef CUTE_DSL_GEMM_ENABLED

#include "cuteDslGemmRunner.h"

#include "common/logger.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace trt_edgellm
{

// Static member initialization
bool CuteDslGemmRunner::sLoaded = false;
int32_t CuteDslGemmRunner::sActiveVariant = 0;
int32_t CuteDslGemmRunner::sBlackwellSmallTileMaxM = 0;
int32_t CuteDslGemmRunner::sBlackwell2ctaMinM = 0;
std::mutex CuteDslGemmRunner::sMutex;

#ifdef CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED
gemm_ampere_decode_fp16_Kernel_Module_t CuteDslGemmRunner::sAmpereDecodeModule = {};
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_SMALL_PREFILL_ENABLED
gemm_ampere_small_prefill_fp16_Kernel_Module_t CuteDslGemmRunner::sAmpereSmallPrefillModule = {};
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_PREFILL_ENABLED
gemm_ampere_medium_prefill_fp16_Kernel_Module_t CuteDslGemmRunner::sAmpereMediumPrefillModule = {};
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_LARGE_PREFILL_ENABLED
gemm_ampere_large_prefill_fp16_Kernel_Module_t CuteDslGemmRunner::sAmpereLargePrefillModule = {};
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED
gemm_ampere_medium_bias_silu_fp16_Kernel_Module_t CuteDslGemmRunner::sAmpereMediumBiasSiLUModule = {};
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_ENABLED
gemm_ampere_medium_bias_fp16_Kernel_Module_t CuteDslGemmRunner::sAmpereMediumBiasModule = {};
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_ENABLED
gemm_blackwell_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_SILU_ENABLED
gemm_blackwell_bias_silu_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellBiasSiLUModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_ENABLED
gemm_blackwell_bias_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellBiasModule = {};
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_ENABLED
gemm_blackwell_small_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellSmallModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_SILU_ENABLED
gemm_blackwell_small_bias_silu_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellSmallBiasSiLUModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_ENABLED
gemm_blackwell_small_bias_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellSmallBiasModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED
gemm_blackwell_small_fp16in_fp32out_Kernel_Module_t CuteDslGemmRunner::sBlackwellSmallFp16inFp32outModule = {};
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_ENABLED
gemm_blackwell_2cta_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwell2ctaModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_SILU_ENABLED
gemm_blackwell_2cta_bias_silu_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwell2ctaBiasSiLUModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_ENABLED
gemm_blackwell_2cta_bias_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwell2ctaBiasModule = {};
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED
gemm_bw_geforce_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellGeforceModule = {};
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_SMALL_ENABLED
gemm_bw_geforce_small_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellGeforceSmallModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_SILU_ENABLED
gemm_bw_geforce_bias_silu_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellGeforceBiasSiLUModule = {};
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_ENABLED
gemm_bw_geforce_bias_fp16_Kernel_Module_t CuteDslGemmRunner::sBlackwellGeforceBiasModule = {};
#endif

bool CuteDslGemmRunner::canImplement(int32_t smVersion)
{
    (void) smVersion;
#ifdef CUTE_DSL_GEMM_AMPERE_ENABLED
    if (smVersion >= 80 && smVersion < 90)
    {
        return true;
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_ENABLED
    if (smVersion >= 100 && smVersion < 120)
    {
        return true;
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED
    if (smVersion >= 120)
    {
        return true;
    }
#endif

    return false;
}

bool CuteDslGemmRunner::loadKernelModule()
{
    std::lock_guard<std::mutex> lock(sMutex);
    if (sLoaded)
    {
        return true;
    }

    int device = 0;
    cudaDeviceProp props{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&props, device) != cudaSuccess)
    {
        LOG_ERROR("CuteDslGemmRunner: Failed to query CUDA device properties");
        return false;
    }
    int32_t const smVersion = props.major * 10 + props.minor;

    LOG_DEBUG("CuteDslGemmRunner: GPU SM%d detected", smVersion);

#ifdef CUTE_DSL_GEMM_AMPERE_ENABLED
    if (smVersion >= 80 && smVersion < 90)
    {
        // Ampere has four GEMM variants, dispatched by M and N:
        //   - decode:         M == 1
        //   - small prefill:  1 < M < 128
        //   - medium prefill: M >= 128 (default for large M)
        //   - large prefill:  384 <= M <= 640 AND N >= 1536 (larger tile, better for M~512)
#ifdef CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED
        gemm_ampere_decode_fp16_Kernel_Module_Load(&sAmpereDecodeModule);
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_SMALL_PREFILL_ENABLED
        gemm_ampere_small_prefill_fp16_Kernel_Module_Load(&sAmpereSmallPrefillModule);
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_PREFILL_ENABLED
        gemm_ampere_medium_prefill_fp16_Kernel_Module_Load(&sAmpereMediumPrefillModule);
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_LARGE_PREFILL_ENABLED
        gemm_ampere_large_prefill_fp16_Kernel_Module_Load(&sAmpereLargePrefillModule);
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED
        gemm_ampere_medium_bias_silu_fp16_Kernel_Module_Load(&sAmpereMediumBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_ENABLED
        gemm_ampere_medium_bias_fp16_Kernel_Module_Load(&sAmpereMediumBiasModule);
#endif
        sActiveVariant = static_cast<int32_t>(Variant::kAmpere);
        sLoaded = true;
        LOG_INFO("CuteDslGemmRunner: Ampere GEMM module(s) loaded for SM%d", smVersion);
        return true;
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_ENABLED
    if (smVersion >= 100 && smVersion < 120)
    {
        gemm_blackwell_fp16_Kernel_Module_Load(&sBlackwellModule);
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_SILU_ENABLED
        gemm_blackwell_bias_silu_fp16_Kernel_Module_Load(&sBlackwellBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_ENABLED
        gemm_blackwell_bias_fp16_Kernel_Module_Load(&sBlackwellBiasModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_ENABLED
        gemm_blackwell_small_fp16_Kernel_Module_Load(&sBlackwellSmallModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_SILU_ENABLED
        gemm_blackwell_small_bias_silu_fp16_Kernel_Module_Load(&sBlackwellSmallBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_ENABLED
        gemm_blackwell_small_bias_fp16_Kernel_Module_Load(&sBlackwellSmallBiasModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED
        gemm_blackwell_small_fp16in_fp32out_Kernel_Module_Load(&sBlackwellSmallFp16inFp32outModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_ENABLED
        gemm_blackwell_2cta_fp16_Kernel_Module_Load(&sBlackwell2ctaModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_SILU_ENABLED
        gemm_blackwell_2cta_bias_silu_fp16_Kernel_Module_Load(&sBlackwell2ctaBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_ENABLED
        gemm_blackwell_2cta_bias_fp16_Kernel_Module_Load(&sBlackwell2ctaBiasModule);
#endif
        // Pick small/default cutover from SM count. Each small CTA covers a
        // 64x128 output sub-tile; default covers 128x128. Empirically, small
        // wins while #CTAs / SM stays under ~3-4 waves; beyond that, the
        // larger default tile amortizes MMA overhead better.
        sBlackwellSmallTileMaxM = std::max(64, props.multiProcessorCount * 4);
        // 2-CTA (256x256) tile beats the 1-CTA default on low-SM-count GPUs
        // where each CTA needs more arithmetic intensity to fill SMs. Two
        // gates: (a) M must clear sBlackwell2ctaMinM, and (b) for M between
        // [MinM, 2*MinM) the 2-CTA tile only wins if N >= 2048 (otherwise
        // the 256-wide N-tile is too granular and 1-CTA default tile wins).
        // Disabled on B100/H100-class (high SM count -> default already
        // saturates compute).
        sBlackwell2ctaMinM = (props.multiProcessorCount <= 32) ? 256 : 0;
        sActiveVariant = static_cast<int32_t>(Variant::kBlackwell);
        sLoaded = true;
        LOG_INFO(
            "CuteDslGemmRunner: Blackwell GEMM module loaded for SM%d (SMs=%d, "
            "small_tile_max_M=%d, 2cta_min_M=%d)",
            smVersion, props.multiProcessorCount, sBlackwellSmallTileMaxM, sBlackwell2ctaMinM);
        return true;
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED
    if (smVersion >= 120)
    {
        gemm_bw_geforce_fp16_Kernel_Module_Load(&sBlackwellGeforceModule);
#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_SMALL_ENABLED
        gemm_bw_geforce_small_fp16_Kernel_Module_Load(&sBlackwellGeforceSmallModule);
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_SILU_ENABLED
        gemm_bw_geforce_bias_silu_fp16_Kernel_Module_Load(&sBlackwellGeforceBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_ENABLED
        gemm_bw_geforce_bias_fp16_Kernel_Module_Load(&sBlackwellGeforceBiasModule);
#endif
        sActiveVariant = static_cast<int32_t>(Variant::kBlackwellGeforce);
        sLoaded = true;
        LOG_INFO("CuteDslGemmRunner: Blackwell GeForce GEMM module(s) loaded for SM%d", smVersion);
        return true;
    }
#endif

    if (smVersion >= 90 && smVersion < 100)
    {
        LOG_ERROR(
            "CuteDslGemmRunner: Hopper (SM%d) is not supported. "
            "CuTe DSL GEMM variants are available for Ampere (SM80-89), "
            "Blackwell DC (SM100-110), and Blackwell GeForce (SM120+).",
            smVersion);
    }
    else
    {
        LOG_ERROR("CuteDslGemmRunner: No GEMM variant compiled for SM%d", smVersion);
    }
    return false;
}

void CuteDslGemmRunner::unloadKernelModule()
{
    std::lock_guard<std::mutex> lock(sMutex);
    if (!sLoaded)
    {
        return;
    }

#ifdef CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        gemm_ampere_decode_fp16_Kernel_Module_Unload(&sAmpereDecodeModule);
    }
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_SMALL_PREFILL_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        gemm_ampere_small_prefill_fp16_Kernel_Module_Unload(&sAmpereSmallPrefillModule);
    }
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_PREFILL_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        gemm_ampere_medium_prefill_fp16_Kernel_Module_Unload(&sAmpereMediumPrefillModule);
    }
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_LARGE_PREFILL_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        gemm_ampere_large_prefill_fp16_Kernel_Module_Unload(&sAmpereLargePrefillModule);
    }
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        gemm_ampere_medium_bias_silu_fp16_Kernel_Module_Unload(&sAmpereMediumBiasSiLUModule);
    }
#endif
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        gemm_ampere_medium_bias_fp16_Kernel_Module_Unload(&sAmpereMediumBiasModule);
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kBlackwell))
    {
        gemm_blackwell_fp16_Kernel_Module_Unload(&sBlackwellModule);
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_SILU_ENABLED
        gemm_blackwell_bias_silu_fp16_Kernel_Module_Unload(&sBlackwellBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_ENABLED
        gemm_blackwell_bias_fp16_Kernel_Module_Unload(&sBlackwellBiasModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_ENABLED
        gemm_blackwell_small_fp16_Kernel_Module_Unload(&sBlackwellSmallModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_SILU_ENABLED
        gemm_blackwell_small_bias_silu_fp16_Kernel_Module_Unload(&sBlackwellSmallBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_ENABLED
        gemm_blackwell_small_bias_fp16_Kernel_Module_Unload(&sBlackwellSmallBiasModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED
        gemm_blackwell_small_fp16in_fp32out_Kernel_Module_Unload(&sBlackwellSmallFp16inFp32outModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_ENABLED
        gemm_blackwell_2cta_fp16_Kernel_Module_Unload(&sBlackwell2ctaModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_SILU_ENABLED
        gemm_blackwell_2cta_bias_silu_fp16_Kernel_Module_Unload(&sBlackwell2ctaBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_ENABLED
        gemm_blackwell_2cta_bias_fp16_Kernel_Module_Unload(&sBlackwell2ctaBiasModule);
#endif
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kBlackwellGeforce))
    {
        gemm_bw_geforce_fp16_Kernel_Module_Unload(&sBlackwellGeforceModule);
#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_SMALL_ENABLED
        gemm_bw_geforce_small_fp16_Kernel_Module_Unload(&sBlackwellGeforceSmallModule);
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_SILU_ENABLED
        gemm_bw_geforce_bias_silu_fp16_Kernel_Module_Unload(&sBlackwellGeforceBiasSiLUModule);
#endif
#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_ENABLED
        gemm_bw_geforce_bias_fp16_Kernel_Module_Unload(&sBlackwellGeforceBiasModule);
#endif
    }
#endif

    sLoaded = false;
    sActiveVariant = 0;
}

namespace
{

// Helper to dispatch a Blackwell/BW-GeForce 3D plain-GEMM kernel (a, b, c).
// 3D tensor ABI: (mode0, mode1, L=1) with strides (mode1, 1, mode0*mode1).
// Bias-free twin of dispatch3dFused in the fused-epilogue section below.
template <typename ModuleT, typename TensorA, typename TensorB, typename TensorC, typename WrapFn>
bool dispatch3d(ModuleT& mod, WrapFn wrapperFn, void const* aPtr, void const* bPtr, void* cPtr, int32_t M, int32_t N,
    int32_t K, cudaStream_t stream)
{
    TensorA tensorA{};
    tensorA.data = const_cast<void*>(aPtr);
    tensorA.dynamic_shapes[0] = M;
    tensorA.dynamic_shapes[1] = K;
    tensorA.dynamic_shapes[2] = 1;
    tensorA.dynamic_strides[0] = K;
    tensorA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

    TensorB tensorB{};
    tensorB.data = const_cast<void*>(bPtr);
    tensorB.dynamic_shapes[0] = N;
    tensorB.dynamic_shapes[1] = K;
    tensorB.dynamic_shapes[2] = 1;
    tensorB.dynamic_strides[0] = K;
    tensorB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

    TensorC tensorC{};
    tensorC.data = cPtr;
    tensorC.dynamic_shapes[0] = M;
    tensorC.dynamic_shapes[1] = N;
    tensorC.dynamic_shapes[2] = 1;
    tensorC.dynamic_strides[0] = N;
    tensorC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

    wrapperFn(&mod, &tensorA, &tensorB, &tensorC, stream);
    return true;
}

} // namespace

bool CuteDslGemmRunner::run(
    void const* aPtr, void const* bPtr, void* cPtr, int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuteDslGemmRunner: Kernel module not loaded. Call loadKernelModule() first.");
        return false;
    }

    // The AOT-exported wrapper function signature depends on the kernel variant.
    // Each variant exports: cute_dsl_<prefix>_wrapper(&module, &tensor_a, &tensor_b, &tensor_c, stream)
    // where tensor structs contain {data, dynamic_shapes[], dynamic_strides[]}.
    //
    // The exact struct layout is defined in the generated header (e.g., gemm_ampere_fp16.h).
    //
    // ABI summary:
    //   - Ampere exports 2D tensors:
    //       A: [M, K], B: [N, K], C: [M, N]
    //   - Blackwell / Blackwell GeForce export 3D tensors with batch L=1:
    //       A: [M, K, 1], B: [N, K, 1], C: [M, N, 1]
    //     Their logical row-major layout is produced from physical storage
    //     (1, mode0, mode1).transpose(1, 2, 0), so element strides are:
    //       A: (K, 1, M*K)
    //       B: (K, 1, N*K)
    //       C: (N, 1, M*N)
    //     The generated ABI stores the non-unit strides for dims 0 and 2.

    switch (sActiveVariant)
    {
#ifdef CUTE_DSL_GEMM_AMPERE_ENABLED
    case static_cast<int32_t>(Variant::kAmpere):
    {
        // Dispatch strategy (see build_cutedsl.py header for documentation):
        //   M==1        → decode
        //   M=2-95      → small_prefill
        //   M=96-383    → medium_prefill (Split-K handled externally)
        //   M=384-640   → large_prefill (if N>=1536)
        //   M>640       → medium_prefill
        (void) N;
        bool const useDecode = (M == 1);
        bool const useSmallPrefill = (M > 1 && M < 96);
        [[maybe_unused]] bool const useLargePrefill = (M >= 384 && M <= 640 && N >= 1536);

#ifdef CUTE_DSL_GEMM_AMPERE_DECODE_ENABLED
        if (useDecode)
        {
            gemm_ampere_decode_fp16_Tensor_mA_t tensorA{};
            tensorA.data = const_cast<void*>(aPtr);
            tensorA.dynamic_shapes[0] = M;
            tensorA.dynamic_shapes[1] = K;
            tensorA.dynamic_strides[0] = K;

            gemm_ampere_decode_fp16_Tensor_mB_t tensorB{};
            tensorB.data = const_cast<void*>(bPtr);
            tensorB.dynamic_shapes[0] = N;
            tensorB.dynamic_shapes[1] = K;
            tensorB.dynamic_strides[0] = K;

            gemm_ampere_decode_fp16_Tensor_mC_t tensorC{};
            tensorC.data = cPtr;
            tensorC.dynamic_shapes[0] = M;
            tensorC.dynamic_shapes[1] = N;
            tensorC.dynamic_strides[0] = N;

            cute_dsl_gemm_ampere_decode_fp16_wrapper(&sAmpereDecodeModule, &tensorA, &tensorB, &tensorC, stream);
            return true;
        }
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_SMALL_PREFILL_ENABLED
        if (useSmallPrefill)
        {
            gemm_ampere_small_prefill_fp16_Tensor_mA_t tensorA{};
            tensorA.data = const_cast<void*>(aPtr);
            tensorA.dynamic_shapes[0] = M;
            tensorA.dynamic_shapes[1] = K;
            tensorA.dynamic_strides[0] = K;

            gemm_ampere_small_prefill_fp16_Tensor_mB_t tensorB{};
            tensorB.data = const_cast<void*>(bPtr);
            tensorB.dynamic_shapes[0] = N;
            tensorB.dynamic_shapes[1] = K;
            tensorB.dynamic_strides[0] = K;

            gemm_ampere_small_prefill_fp16_Tensor_mC_t tensorC{};
            tensorC.data = cPtr;
            tensorC.dynamic_shapes[0] = M;
            tensorC.dynamic_shapes[1] = N;
            tensorC.dynamic_strides[0] = N;

            cute_dsl_gemm_ampere_small_prefill_fp16_wrapper(
                &sAmpereSmallPrefillModule, &tensorA, &tensorB, &tensorC, stream);
            return true;
        }
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_LARGE_PREFILL_ENABLED
        if (useLargePrefill)
        {
            gemm_ampere_large_prefill_fp16_Tensor_mA_t tensorA{};
            tensorA.data = const_cast<void*>(aPtr);
            tensorA.dynamic_shapes[0] = M;
            tensorA.dynamic_shapes[1] = K;
            tensorA.dynamic_strides[0] = K;

            gemm_ampere_large_prefill_fp16_Tensor_mB_t tensorB{};
            tensorB.data = const_cast<void*>(bPtr);
            tensorB.dynamic_shapes[0] = N;
            tensorB.dynamic_shapes[1] = K;
            tensorB.dynamic_strides[0] = K;

            gemm_ampere_large_prefill_fp16_Tensor_mC_t tensorC{};
            tensorC.data = cPtr;
            tensorC.dynamic_shapes[0] = M;
            tensorC.dynamic_shapes[1] = N;
            tensorC.dynamic_strides[0] = N;

            cute_dsl_gemm_ampere_large_prefill_fp16_wrapper(
                &sAmpereLargePrefillModule, &tensorA, &tensorB, &tensorC, stream);
            return true;
        }
#endif

#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_PREFILL_ENABLED
        {
            gemm_ampere_medium_prefill_fp16_Tensor_mA_t tensorA{};
            tensorA.data = const_cast<void*>(aPtr);
            tensorA.dynamic_shapes[0] = M;
            tensorA.dynamic_shapes[1] = K;
            tensorA.dynamic_strides[0] = K;

            gemm_ampere_medium_prefill_fp16_Tensor_mB_t tensorB{};
            tensorB.data = const_cast<void*>(bPtr);
            tensorB.dynamic_shapes[0] = N;
            tensorB.dynamic_shapes[1] = K;
            tensorB.dynamic_strides[0] = K;

            gemm_ampere_medium_prefill_fp16_Tensor_mC_t tensorC{};
            tensorC.data = cPtr;
            tensorC.dynamic_shapes[0] = M;
            tensorC.dynamic_shapes[1] = N;
            tensorC.dynamic_strides[0] = N;

            cute_dsl_gemm_ampere_medium_prefill_fp16_wrapper(
                &sAmpereMediumPrefillModule, &tensorA, &tensorB, &tensorC, stream);
            return true;
        }
#else
        LOG_ERROR("CuteDslGemmRunner: Ampere M=%d requires medium-prefill variant, but it is not compiled", M);
        return false;
#endif
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_ENABLED
    case static_cast<int32_t>(Variant::kBlackwell):
    {
        // Tile dispatch: small (64x128, cluster (1,2)) wins while there are
        // few enough M-tiles to fit in a small number of waves; default
        // (128x128, cluster (1,1)) wins once #CTAs >> #SMs. Threshold scales
        // with SM count (sBlackwellSmallTileMaxM = max(64, 4 * SMs)) to cover
        // both B100 (~144 SMs -> ~576) and Thor (~20 SMs -> 80).
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_ENABLED
        if (M <= sBlackwellSmallTileMaxM)
        {
            return dispatch3d<gemm_blackwell_small_fp16_Kernel_Module_t, gemm_blackwell_small_fp16_Tensor_a_t,
                gemm_blackwell_small_fp16_Tensor_b_t, gemm_blackwell_small_fp16_Tensor_c_t>(
                sBlackwellSmallModule, cute_dsl_gemm_blackwell_small_fp16_wrapper, aPtr, bPtr, cPtr, M, N, K, stream);
        }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_ENABLED
        if (sBlackwell2ctaMinM > 0 && M >= sBlackwell2ctaMinM && (M >= 2 * sBlackwell2ctaMinM || N >= 2048))
        {
            gemm_blackwell_2cta_fp16_Tensor_a_t tensorA{};
            tensorA.data = const_cast<void*>(aPtr);
            tensorA.dynamic_shapes[0] = M;
            tensorA.dynamic_shapes[1] = K;
            tensorA.dynamic_shapes[2] = 1;
            tensorA.dynamic_strides[0] = K;
            tensorA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

            gemm_blackwell_2cta_fp16_Tensor_b_t tensorB{};
            tensorB.data = const_cast<void*>(bPtr);
            tensorB.dynamic_shapes[0] = N;
            tensorB.dynamic_shapes[1] = K;
            tensorB.dynamic_shapes[2] = 1;
            tensorB.dynamic_strides[0] = K;
            tensorB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

            gemm_blackwell_2cta_fp16_Tensor_c_t tensorC{};
            tensorC.data = cPtr;
            tensorC.dynamic_shapes[0] = M;
            tensorC.dynamic_shapes[1] = N;
            tensorC.dynamic_shapes[2] = 1;
            tensorC.dynamic_strides[0] = N;
            tensorC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

            cute_dsl_gemm_blackwell_2cta_fp16_wrapper(&sBlackwell2ctaModule, &tensorA, &tensorB, &tensorC, stream);
            return true;
        }
#endif

        gemm_blackwell_fp16_Tensor_a_t tensorA{};
        tensorA.data = const_cast<void*>(aPtr);
        tensorA.dynamic_shapes[0] = M;
        tensorA.dynamic_shapes[1] = K;
        tensorA.dynamic_shapes[2] = 1;
        tensorA.dynamic_strides[0] = K;
        tensorA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

        gemm_blackwell_fp16_Tensor_b_t tensorB{};
        tensorB.data = const_cast<void*>(bPtr);
        tensorB.dynamic_shapes[0] = N;
        tensorB.dynamic_shapes[1] = K;
        tensorB.dynamic_shapes[2] = 1;
        tensorB.dynamic_strides[0] = K;
        tensorB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

        gemm_blackwell_fp16_Tensor_c_t tensorC{};
        tensorC.data = cPtr;
        tensorC.dynamic_shapes[0] = M;
        tensorC.dynamic_shapes[1] = N;
        tensorC.dynamic_shapes[2] = 1;
        tensorC.dynamic_strides[0] = N;
        tensorC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

        cute_dsl_gemm_blackwell_fp16_wrapper(&sBlackwellModule, &tensorA, &tensorB, &tensorC, stream);
        return true;
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_ENABLED
    case static_cast<int32_t>(Variant::kBlackwellGeforce):
    {
#ifdef CUTE_DSL_GEMM_BLACKWELL_GEFORCE_SMALL_ENABLED
        if (M <= 64)
        {
            gemm_bw_geforce_small_fp16_Tensor_a_t tensorA{};
            tensorA.data = const_cast<void*>(aPtr);
            tensorA.dynamic_shapes[0] = M;
            tensorA.dynamic_shapes[1] = K;
            tensorA.dynamic_shapes[2] = 1;
            tensorA.dynamic_strides[0] = K;
            tensorA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

            gemm_bw_geforce_small_fp16_Tensor_b_t tensorB{};
            tensorB.data = const_cast<void*>(bPtr);
            tensorB.dynamic_shapes[0] = N;
            tensorB.dynamic_shapes[1] = K;
            tensorB.dynamic_shapes[2] = 1;
            tensorB.dynamic_strides[0] = K;
            tensorB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

            gemm_bw_geforce_small_fp16_Tensor_c_t tensorC{};
            tensorC.data = cPtr;
            tensorC.dynamic_shapes[0] = M;
            tensorC.dynamic_shapes[1] = N;
            tensorC.dynamic_shapes[2] = 1;
            tensorC.dynamic_strides[0] = N;
            tensorC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

            cute_dsl_gemm_bw_geforce_small_fp16_wrapper(
                &sBlackwellGeforceSmallModule, &tensorA, &tensorB, &tensorC, stream);
            return true;
        }
#endif
        gemm_bw_geforce_fp16_Tensor_a_t tensorA{};
        tensorA.data = const_cast<void*>(aPtr);
        tensorA.dynamic_shapes[0] = M;
        tensorA.dynamic_shapes[1] = K;
        tensorA.dynamic_shapes[2] = 1;
        tensorA.dynamic_strides[0] = K;
        tensorA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

        gemm_bw_geforce_fp16_Tensor_b_t tensorB{};
        tensorB.data = const_cast<void*>(bPtr);
        tensorB.dynamic_shapes[0] = N;
        tensorB.dynamic_shapes[1] = K;
        tensorB.dynamic_shapes[2] = 1;
        tensorB.dynamic_strides[0] = K;
        tensorB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

        gemm_bw_geforce_fp16_Tensor_c_t tensorC{};
        tensorC.data = cPtr;
        tensorC.dynamic_shapes[0] = M;
        tensorC.dynamic_shapes[1] = N;
        tensorC.dynamic_shapes[2] = 1;
        tensorC.dynamic_strides[0] = N;
        tensorC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

        cute_dsl_gemm_bw_geforce_fp16_wrapper(&sBlackwellGeforceModule, &tensorA, &tensorB, &tensorC, stream);
        return true;
    }
#endif

    default: LOG_ERROR("CuteDslGemmRunner: No active variant (variant=%d)", sActiveVariant); return false;
    }
}

bool CuteDslGemmRunner::runFp16inFp32out(
    void const* aPtr, void const* bPtr, void* cPtr, int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuteDslGemmRunner: Kernel module not loaded. Call loadKernelModule() first.");
        return false;
    }

    // FP32-C-output path: a single AOT variant with the same Blackwell 3D L=1 row-major
    // ABI as run()'s FP16-out variants (the ABI does not encode the C element type —
    // data is void* — so only the dispatched module and symbols differ). No M-based
    // tile dispatch: the small (64x128) tile is the only FP32-out variant and covers
    // the parakeet mel GEMM's fixed M=nMel. Shares the dispatch3d helper with run().
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_FP16IN_FP32OUT_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kBlackwell))
    {
        return dispatch3d<gemm_blackwell_small_fp16in_fp32out_Kernel_Module_t,
            gemm_blackwell_small_fp16in_fp32out_Tensor_a_t, gemm_blackwell_small_fp16in_fp32out_Tensor_b_t,
            gemm_blackwell_small_fp16in_fp32out_Tensor_c_t>(sBlackwellSmallFp16inFp32outModule,
            cute_dsl_gemm_blackwell_small_fp16in_fp32out_wrapper, aPtr, bPtr, cPtr, M, N, K, stream);
    }
#endif
    LOG_ERROR(
        "CuteDslGemmRunner::runFp16inFp32out: no FP16-in/FP32-out Blackwell variant compiled / active "
        "(variant=%d). Build the gemm_blackwell_small_fp16in_fp32out AOT variant and run on Blackwell DC.",
        sActiveVariant);
    return false;
}

// ---------------------------------------------------------------------------
// Fused epilogue dispatch: GEMM + bias + optional SiLU
// ---------------------------------------------------------------------------

namespace
{

// Helper to dispatch an Ampere 2D fused kernel (mA, mB, mC, mBias).
// The AOT-exported fused kernels accept a 4th 1D bias tensor with dynamic shape[0]=N.
template <typename ModuleT, typename TensorA, typename TensorB, typename TensorC, typename TensorBias, typename WrapFn>
bool dispatchAmpere2dFused(ModuleT& mod, WrapFn wrapperFn, void const* aPtr, void const* bPtr, void* cPtr,
    void const* biasPtr, int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
    TensorA tensorA{};
    tensorA.data = const_cast<void*>(aPtr);
    tensorA.dynamic_shapes[0] = M;
    tensorA.dynamic_shapes[1] = K;
    tensorA.dynamic_strides[0] = K;

    TensorB tensorB{};
    tensorB.data = const_cast<void*>(bPtr);
    tensorB.dynamic_shapes[0] = N;
    tensorB.dynamic_shapes[1] = K;
    tensorB.dynamic_strides[0] = K;

    TensorC tensorC{};
    tensorC.data = cPtr;
    tensorC.dynamic_shapes[0] = M;
    tensorC.dynamic_shapes[1] = N;
    tensorC.dynamic_strides[0] = N;

    TensorBias tensorBias{};
    tensorBias.data = const_cast<void*>(biasPtr);
    tensorBias.dynamic_shapes[0] = N;

    wrapperFn(&mod, &tensorA, &tensorB, &tensorC, stream, &tensorBias);
    return true;
}

// Helper to dispatch a Blackwell/BW-GeForce 3D fused kernel (a, b, c, mBias).
// 3D tensor ABI: (mode0, mode1, L=1) with strides (mode1, 1, mode0*mode1).
template <typename ModuleT, typename TensorA, typename TensorB, typename TensorC, typename TensorBias, typename WrapFn>
bool dispatch3dFused(ModuleT& mod, WrapFn wrapperFn, void const* aPtr, void const* bPtr, void* cPtr,
    void const* biasPtr, int32_t M, int32_t N, int32_t K, cudaStream_t stream)
{
    TensorA tensorA{};
    tensorA.data = const_cast<void*>(aPtr);
    tensorA.dynamic_shapes[0] = M;
    tensorA.dynamic_shapes[1] = K;
    tensorA.dynamic_shapes[2] = 1;
    tensorA.dynamic_strides[0] = K;
    tensorA.dynamic_strides[1] = static_cast<int64_t>(M) * K;

    TensorB tensorB{};
    tensorB.data = const_cast<void*>(bPtr);
    tensorB.dynamic_shapes[0] = N;
    tensorB.dynamic_shapes[1] = K;
    tensorB.dynamic_shapes[2] = 1;
    tensorB.dynamic_strides[0] = K;
    tensorB.dynamic_strides[1] = static_cast<int64_t>(N) * K;

    TensorC tensorC{};
    tensorC.data = cPtr;
    tensorC.dynamic_shapes[0] = M;
    tensorC.dynamic_shapes[1] = N;
    tensorC.dynamic_shapes[2] = 1;
    tensorC.dynamic_strides[0] = N;
    tensorC.dynamic_strides[1] = static_cast<int64_t>(M) * N;

    TensorBias tensorBias{};
    tensorBias.data = const_cast<void*>(biasPtr);
    tensorBias.dynamic_shapes[0] = N;

    wrapperFn(&mod, &tensorA, &tensorB, &tensorC, stream, &tensorBias);
    return true;
}

} // namespace

bool CuteDslGemmRunner::runBiasSiLU(void const* aPtr, void const* bPtr, void* cPtr, void const* biasPtr, int32_t M,
    int32_t N, int32_t K, cudaStream_t stream)
{
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_SILU_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        return dispatchAmpere2dFused<gemm_ampere_medium_bias_silu_fp16_Kernel_Module_t,
            gemm_ampere_medium_bias_silu_fp16_Tensor_mA_t, gemm_ampere_medium_bias_silu_fp16_Tensor_mB_t,
            gemm_ampere_medium_bias_silu_fp16_Tensor_mC_t, gemm_ampere_medium_bias_silu_fp16_Tensor_mBias_t>(
            sAmpereMediumBiasSiLUModule, cute_dsl_gemm_ampere_medium_bias_silu_fp16_wrapper, aPtr, bPtr, cPtr, biasPtr,
            M, N, K, stream);
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_SILU_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kBlackwell))
    {
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_SILU_ENABLED
        if (M <= sBlackwellSmallTileMaxM)
        {
            return dispatch3dFused<gemm_blackwell_small_bias_silu_fp16_Kernel_Module_t,
                gemm_blackwell_small_bias_silu_fp16_Tensor_a_t, gemm_blackwell_small_bias_silu_fp16_Tensor_b_t,
                gemm_blackwell_small_bias_silu_fp16_Tensor_c_t, gemm_blackwell_small_bias_silu_fp16_Tensor_mBias_t>(
                sBlackwellSmallBiasSiLUModule, cute_dsl_gemm_blackwell_small_bias_silu_fp16_wrapper, aPtr, bPtr, cPtr,
                biasPtr, M, N, K, stream);
        }
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_SILU_ENABLED
        if (sBlackwell2ctaMinM > 0 && M >= sBlackwell2ctaMinM && (M >= 2 * sBlackwell2ctaMinM || N >= 2048))
        {
            return dispatch3dFused<gemm_blackwell_2cta_bias_silu_fp16_Kernel_Module_t,
                gemm_blackwell_2cta_bias_silu_fp16_Tensor_a_t, gemm_blackwell_2cta_bias_silu_fp16_Tensor_b_t,
                gemm_blackwell_2cta_bias_silu_fp16_Tensor_c_t, gemm_blackwell_2cta_bias_silu_fp16_Tensor_mBias_t>(
                sBlackwell2ctaBiasSiLUModule, cute_dsl_gemm_blackwell_2cta_bias_silu_fp16_wrapper, aPtr, bPtr, cPtr,
                biasPtr, M, N, K, stream);
        }
#endif
        return dispatch3dFused<gemm_blackwell_bias_silu_fp16_Kernel_Module_t, gemm_blackwell_bias_silu_fp16_Tensor_a_t,
            gemm_blackwell_bias_silu_fp16_Tensor_b_t, gemm_blackwell_bias_silu_fp16_Tensor_c_t,
            gemm_blackwell_bias_silu_fp16_Tensor_mBias_t>(sBlackwellBiasSiLUModule,
            cute_dsl_gemm_blackwell_bias_silu_fp16_wrapper, aPtr, bPtr, cPtr, biasPtr, M, N, K, stream);
    }
#endif

#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_SILU_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kBlackwellGeforce))
    {
        return dispatch3dFused<gemm_bw_geforce_bias_silu_fp16_Kernel_Module_t,
            gemm_bw_geforce_bias_silu_fp16_Tensor_a_t, gemm_bw_geforce_bias_silu_fp16_Tensor_b_t,
            gemm_bw_geforce_bias_silu_fp16_Tensor_c_t, gemm_bw_geforce_bias_silu_fp16_Tensor_mBias_t>(
            sBlackwellGeforceBiasSiLUModule, cute_dsl_gemm_bw_geforce_bias_silu_fp16_wrapper, aPtr, bPtr, cPtr, biasPtr,
            M, N, K, stream);
    }
#endif

    LOG_ERROR("CuteDslGemmRunner: No fused bias+SiLU variant compiled for variant=%d", sActiveVariant);
    return false;
}

bool CuteDslGemmRunner::runBias(void const* aPtr, void const* bPtr, void* cPtr, void const* biasPtr, int32_t M,
    int32_t N, int32_t K, cudaStream_t stream)
{
#ifdef CUTE_DSL_GEMM_AMPERE_MEDIUM_BIAS_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kAmpere))
    {
        return dispatchAmpere2dFused<gemm_ampere_medium_bias_fp16_Kernel_Module_t,
            gemm_ampere_medium_bias_fp16_Tensor_mA_t, gemm_ampere_medium_bias_fp16_Tensor_mB_t,
            gemm_ampere_medium_bias_fp16_Tensor_mC_t, gemm_ampere_medium_bias_fp16_Tensor_mBias_t>(
            sAmpereMediumBiasModule, cute_dsl_gemm_ampere_medium_bias_fp16_wrapper, aPtr, bPtr, cPtr, biasPtr, M, N, K,
            stream);
    }
#endif

#ifdef CUTE_DSL_GEMM_BLACKWELL_BIAS_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kBlackwell))
    {
#ifdef CUTE_DSL_GEMM_BLACKWELL_SMALL_BIAS_ENABLED
        if (M <= sBlackwellSmallTileMaxM)
        {
            return dispatch3dFused<gemm_blackwell_small_bias_fp16_Kernel_Module_t,
                gemm_blackwell_small_bias_fp16_Tensor_a_t, gemm_blackwell_small_bias_fp16_Tensor_b_t,
                gemm_blackwell_small_bias_fp16_Tensor_c_t, gemm_blackwell_small_bias_fp16_Tensor_mBias_t>(
                sBlackwellSmallBiasModule, cute_dsl_gemm_blackwell_small_bias_fp16_wrapper, aPtr, bPtr, cPtr, biasPtr,
                M, N, K, stream);
        }
#endif
#ifdef CUTE_DSL_GEMM_BLACKWELL_2CTA_BIAS_ENABLED
        if (sBlackwell2ctaMinM > 0 && M >= sBlackwell2ctaMinM && (M >= 2 * sBlackwell2ctaMinM || N >= 2048))
        {
            return dispatch3dFused<gemm_blackwell_2cta_bias_fp16_Kernel_Module_t,
                gemm_blackwell_2cta_bias_fp16_Tensor_a_t, gemm_blackwell_2cta_bias_fp16_Tensor_b_t,
                gemm_blackwell_2cta_bias_fp16_Tensor_c_t, gemm_blackwell_2cta_bias_fp16_Tensor_mBias_t>(
                sBlackwell2ctaBiasModule, cute_dsl_gemm_blackwell_2cta_bias_fp16_wrapper, aPtr, bPtr, cPtr, biasPtr, M,
                N, K, stream);
        }
#endif
        return dispatch3dFused<gemm_blackwell_bias_fp16_Kernel_Module_t, gemm_blackwell_bias_fp16_Tensor_a_t,
            gemm_blackwell_bias_fp16_Tensor_b_t, gemm_blackwell_bias_fp16_Tensor_c_t,
            gemm_blackwell_bias_fp16_Tensor_mBias_t>(sBlackwellBiasModule, cute_dsl_gemm_blackwell_bias_fp16_wrapper,
            aPtr, bPtr, cPtr, biasPtr, M, N, K, stream);
    }
#endif

#ifdef CUTE_DSL_GEMM_BW_GEFORCE_BIAS_ENABLED
    if (sActiveVariant == static_cast<int32_t>(Variant::kBlackwellGeforce))
    {
        return dispatch3dFused<gemm_bw_geforce_bias_fp16_Kernel_Module_t, gemm_bw_geforce_bias_fp16_Tensor_a_t,
            gemm_bw_geforce_bias_fp16_Tensor_b_t, gemm_bw_geforce_bias_fp16_Tensor_c_t,
            gemm_bw_geforce_bias_fp16_Tensor_mBias_t>(sBlackwellGeforceBiasModule,
            cute_dsl_gemm_bw_geforce_bias_fp16_wrapper, aPtr, bPtr, cPtr, biasPtr, M, N, K, stream);
    }
#endif

    LOG_ERROR("CuteDslGemmRunner: No fused bias variant compiled for variant=%d", sActiveVariant);
    return false;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_GEMM_ENABLED
