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

#pragma once

#ifdef CUTE_DSL_NVFP4_MOE_ENABLED

#include "cutedsl_nvfp4_moe_all.h"

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

namespace trt_edgellm
{

//! NVFP4 block scale group size (K elements per scale).
inline constexpr int32_t kDecodeGemvSfVecSize = 16;

//! Per-launch parameters for CuteDslDecodeGemvRunner.
//!
//! W4A16 NVFP4 format — the CuTe DSL decode GEMV kernel uses N-major layouts:
//!   weights       \c [E, K, N/2] uint8 packed FP4 E2M1x2 (N-major, adjacent N nibbles per byte)
//!   scales        Marlin atom-layout FP8 E4M3 block scales (shared with prefill GEMM)
//!   global_scale  \c [E] FP32 per-expert global scale
//! Dequant: \c weight_f32 = fp4_to_f32(nibble) * fp8_to_f32(block_scale) * global_scale
//!
//! Weight and scale layout is shared with the N-major prefill GEMM kernel —
//! no weight or scale duplication needed between prefill and decode paths.
struct CuteDslDecodeGemvParams
{
    int32_t numTokens{};    //!< B*S tokens.
    int32_t numExperts{};   //!< E total experts.
    int32_t topK{};         //!< Top-K experts per token.
    int32_t hiddenSize{};   //!< H (input/output dim for non-gated; MoE hidden dim).
    int32_t moeInterSize{}; //!< I (intermediate dim per expert).

    // Hidden-state input (device).
    __half const* hiddenStates{}; //!< FP16 \c [numTokens, H] row-major.
    int32_t const* topkIds{};     //!< int32 \c [numTokens * topK].
    float const* topkWeights{};   //!< fp32  \c [numTokens * topK] renormalized gate scores.

    //! Up weights.
    //!   Non-gated: \c [E, H, I/2] uint8 N-major.
    //!   Gated (SwiGLU): \c [E, H, I] uint8 N-major interleaved gate+up
    //!     (cols [0,I) = gate, cols [I,2*I) = up, each pair packed). Shared with prefill GEMM.
    void const* upWeights{};
    //! FP8 E4M3 block scales in Marlin atom layout (shared with prefill GEMM).
    //!   Non-gated: covers N=I. Gated: covers N=2*I (full interleaved width).
    void const* upScales{};
    //! FP32 per-expert global scale \c [E].
    float const* upGlobalScale{};

    // Down weights: \c [E, I, H/2] uint8 N-major.
    void const* downWeights{};
    //! FP8 E4M3 block scales in Marlin atom layout (shared with prefill GEMM).
    void const* downScales{};
    //! FP32 per-expert global scale \c [E].
    float const* downGlobalScale{};

    // Output (device).
    __half* output{}; //!< FP16 \c [numTokens, H] row-major.

    //! 0 = ReLU2 (Nemotron), 1 = SiLU (Mixtral/LLaMA).
    int32_t activationKind{0};

    //! True for gated MoE (LLaMA/Qwen3 SwiGLU: gate_proj + up_proj + SiLU-gated down_proj).
    bool isGated{false};
};

//! Runner for the CuTe DSL NVFP4 MoE decode GEMV kernels (AOT cubins).
//!
//! Wraps kernel variants exported from moe-cutedsl/:
//!   \c gemv_up_none    — up-projection, no activation, direct store
//!   \c gemv_up_swiglu  — fused gate+up with SwiGLU epilogue, direct store
//!   \c gemv_dn_relu2   — down-projection, ReLU2 activation, atomic output
//!   \c gemv_dn_silu    — down-projection, SiLU activation, atomic output
//!   \c gemv_dn_none    — down-projection, no activation, atomic output
//!
//! Pipeline configurations:
//!   Non-gated ReLU2 (Nemotron): up_none → dn_relu2
//!   Non-gated SiLU  (Mixtral):  up_none → dn_silu
//!   Gated SwiGLU    (LLaMA):    up_swiglu → dn_none
//!
//! The gated pipeline reads interleaved FC1 weights [E, H, I] shared with
//! the prefill GEMM kernel — no weight duplication needed.
//!
//! Module load/unload is process-global (static) matching the CuteDslNvfp4MoeRunner pattern.
//! Per-launch state is passed through CuteDslDecodeGemvParams.
class CuteDslDecodeGemvRunner
{
public:
    CuteDslDecodeGemvRunner() = default;
    ~CuteDslDecodeGemvRunner() = default;
    CuteDslDecodeGemvRunner(CuteDslDecodeGemvRunner const&) = delete;
    CuteDslDecodeGemvRunner& operator=(CuteDslDecodeGemvRunner const&) = delete;

    //! Load all four AOT kernel modules. Safe to call multiple times (idempotent).
    static bool loadKernelModules();
    static void unloadKernelModules();

    //! Workspace size for the intermediate buffer between up-proj and down-proj.
    //! Both gated and non-gated: \c numTokens * topK * moeInterSize * sizeof(__half).
    //! (Gated no longer needs 2x — the fused up_swiglu kernel produces one output.)
    static size_t getWorkspaceSize(int32_t numTokens, int32_t topK, int32_t moeInterSize);

    //! Execute the full decode GEMV pipeline (up + down, or gate + up + gated-down).
    //! \c workspace must be at least getWorkspaceSize() bytes.
    //! Returns 0 on success.
    int32_t run(CuteDslDecodeGemvParams const& params, void* workspace, cudaStream_t stream);

private:
    //! Non-gated pipeline: up_none → dn_{relu2|silu}.
    int32_t runNonGated(CuteDslDecodeGemvParams const& params, void* workspace, cudaStream_t stream);

    //! Gated pipeline: up_swiglu → dn_none.
    int32_t runGated(CuteDslDecodeGemvParams const& params, void* workspace, cudaStream_t stream);

    // Static AOT kernel module handles.
    static gemv_up_none_Kernel_Module_t sModUpNone;
    static gemv_up_swiglu_Kernel_Module_t sModUpSwiglu;
    static gemv_dn_relu2_Kernel_Module_t sModDnRelu2;
    static gemv_dn_silu_Kernel_Module_t sModDnSilu;
    static gemv_dn_none_Kernel_Module_t sModDnNone;

    static bool sLoaded;
    static std::mutex sLoadMutex;
};

} // namespace trt_edgellm

#endif // CUTE_DSL_NVFP4_MOE_ENABLED
