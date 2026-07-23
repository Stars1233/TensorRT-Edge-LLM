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

#if defined(CUTE_DSL_FFPA_ENABLED)

#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 12.8+ defines cudaLibrary_t / cudaLibraryUnload in <cuda_runtime.h>;
// older toolkits still require the cuLibrary* driver API.  Provide a typedef
// and a forward declaration so the generated AOT header (which references both
// the type and the unload function) compiles against CUDA 12.0-12.7.  The
// matching cudaLibrary* function symbol is supplied at link time by
// cpp/kernels/gdnKernels/cutedsl_cuda_runtime_library_shim.c (weak definition,
// linked through trt_edgellm_cutedsl_cudart_shim).
#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library);
#endif

#include "cutedsl_all.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{

//! Parameters for the generated FFPA CuTe DSL kernel.
struct CuteDslFFPAParams
{
    void const* q{nullptr};
    void const* k{nullptr};
    void const* v{nullptr};
    void* o{nullptr};
    //! Device pointers to (batchSize + 1) int32 cumulative sequence lengths
    //! carrying the logical per-batch valid lengths (seqlenQ/seqlenK stay the
    //! physical padded extents).  Per batch b the valid lengths are
    //! cuSeqLenQ[b+1] - cuSeqLenQ[b] and cuSeqLenK[b+1] - cuSeqLenK[b]; the
    //! causal mask is bottom-right aligned with offset seqlenK_b - seqlenQ_b
    //! (0 for plain prefill, KV prefix length for chunked prefill).
    int32_t const* cuSeqLenQ{nullptr};
    int32_t const* cuSeqLenK{nullptr};
    //! Optional device pointers to [batchSize, seqlenQ] int32 vision-block
    //! interval tensors (Gemma4 Unified).  Per query row q they carry an
    //! extra allowed KV interval [blockBegin[q], blockEnd[q]] so image
    //! blocks attend bidirectionally; text rows hold the -1/-1 sentinel
    //! (empty interval).  Both null (the default) selects the plain causal
    //! kernel — behaviour is exactly unchanged for non-vision models.  Both
    //! must be set (or both null); mixed null/non-null is rejected.
    int32_t const* blockBegin{nullptr};
    int32_t const* blockEnd{nullptr};
    int32_t batchSize{0};
    int32_t seqlenQ{0};
    int32_t seqlenK{0};
    int32_t numQHeads{0};
    int32_t numKVHeads{0};
    int32_t headDim{0};
    float softmaxScale{0.0F};
};

//! Runner for the generated FP16 FFPA d=512 causal kernel.
//! Supports MHA and GQA dynamically via numKVHeads runtime argument.
//! When params.blockBegin/blockEnd are set, dispatches the vision-block
//! overlay variant (CUTE_DSL_FFPA_VISIONBLOCK_ENABLED) instead.
class CuteDslFFPARunner
{
public:
    //! Returns true if the given configuration is supported. Checks head dimension,
    //! SM version, and GQA group size (numQHeads / numKVHeads). Group size 1 (MHA) is
    //! always supported; group sizes 4, 8, and 16 require CUTE_DSL_FFPA_GQA4_ENABLED /
    //! CUTE_DSL_FFPA_GQA8_ENABLED / CUTE_DSL_FFPA_GQA16_ENABLED respectively.
    static bool canImplement(int32_t headDim, int32_t smVersion, int32_t numQHeads = 1, int32_t numKVHeads = 1);

    //! Whether the vision-block overlay variant was compiled into this build
    //! (requires the ffpa_d512_causal_visionblock AOT artifact).
    static bool canImplementVisionBlock(int32_t headDim, int32_t smVersion);

    static bool loadKernelModule();

    static void unloadKernelModule();

    static int run(CuteDslFFPAParams const& params, cudaStream_t stream);

private:
    static ffpa_d512_causal_Kernel_Module_t sD512CausalModule;
#ifdef CUTE_DSL_FFPA_VISIONBLOCK_ENABLED
    static ffpa_d512_causal_visionblock_Kernel_Module_t sD512CausalVisionBlockModule;
#endif
#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
    static ffpa_d512_causal_gqa4_Kernel_Module_t sD512CausalGqa4Module;
#endif
#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
    static ffpa_d512_causal_gqa8_Kernel_Module_t sD512CausalGqa8Module;
#endif
#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
    static ffpa_d512_causal_gqa16_Kernel_Module_t sD512CausalGqa16Module;
#endif
    static bool sLoaded;
    static std::mutex sMutex;
};

} // namespace trt_edgellm

#endif // defined(CUTE_DSL_FFPA_ENABLED)
