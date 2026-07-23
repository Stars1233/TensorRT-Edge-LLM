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

#ifdef CUTE_DSL_GDN_ENABLED

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

#include "cutedsl_all.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{

/** Device pointers and dimensions; runner fills generated tensor structs from these. */
struct GDNParams
{
    void* q{};
    void* k{};
    void* v{};
    void* a{};
    void* b{};
    void* A_log{};
    void* dt_bias{};
    void* h0_source{};
    void* context_lengths{}; ///< [N] int32 — valid length per batch (decode / prefill; unused in MTP)
    void* cu_seqlens{};      ///< [N+1] int32 — prefix-sum of context_lengths (Blackwell prefill)
    void* h0_scratch{};      ///< [N, hv, k, v] f32 — pre-allocated scratch for h0_out (Blackwell prefill);
                             ///<   must be provided by caller (e.g. plugin workspace).
    void* o{};

    // MTP (multi-token) decode fields — used only when use_mtp == true.
    void* intermediate_states{}; ///< [N, seq_len, HV, K, V] FP32 — per-step h cache for rollback.
                                 ///<   Must be non-null when use_mtp == true.
    bool use_mtp{false};         ///< true → MTP decode path (any seq_len).

    // DDTree decode fields — used only when use_ddtree == true.
    void* tree_parent_ids{};    ///< [N, seq_len] int32 — parent node index per verify token.
    void* tree_depths{};        ///< [N, seq_len] int32 — depth per verify token.
    bool use_ddtree{false};     ///< true → DDTree decode path; never falls back to linear MTP.
    void* ddtree_qk_scales{};   ///< [N, seq_len, H, 2] FP32 — split-v precomputed q/k scales.
    void* ddtree_gate_values{}; ///< [N, seq_len, HV, 2] FP32 — split-v precomputed g/beta.

    int32_t n{};
    int32_t seq_len{};
    int32_t h{};
    int32_t hv{};
    int32_t k_dim{};
    int32_t v_dim{};
    int32_t smVersion{}; // GPU SM version for dispatch (e.g. 87, 110)
};

/** Loads AOT .o, fills tensor structs from GDNParams, calls generated wrapper.
 *
 *  Dispatch table (evaluated in order):
 *    use_ddtree == true           → runDecodeTree()      (DDTree: parent/depth-driven tree state)
 *    use_mtp == true              → runDecodeMTP()       (MTP: any seq_len)
 *    seq_len == 1                 → runDecode()          (single-token decode)
 *    seq_len > 1 && SM >= 100     → runPrefillBlackwell() (Blackwell prefill)
 *    seq_len > 1                  → runPrefill()          (sequential prefill)
 *
 *  MTP note: all batch items process seq_len (T) draft tokens uniformly.
 *  context_lengths is not used in MTP mode.
 */
class CuteDslGDNRunner
{
public:
    CuteDslGDNRunner() = default;
    ~CuteDslGDNRunner() = default;
    CuteDslGDNRunner(CuteDslGDNRunner const&) = delete;
    CuteDslGDNRunner& operator=(CuteDslGDNRunner const&) = delete;

    static bool canImplement(int32_t kDim, int32_t vDim, int32_t smVersion);

    static bool loadKernelModules();
    static void unloadKernelModules();

    /** Run GDN kernel. See class-level dispatch table. */
    int run(GDNParams const& params, cudaStream_t stream);

    /** Run DDTree decode with parent/depth-driven recurrent state checkpoints. */
    int runDecodeTree(GDNParams const& params, cudaStream_t stream);

private:
    int runDecodeTreeSplitVPrecomputed(GDNParams const& params, cudaStream_t stream);
    int runDecode(GDNParams const& params, cudaStream_t stream);
    int runPrefill(GDNParams const& params, cudaStream_t stream);
    int runPrefillBlackwell(GDNParams const& params, cudaStream_t stream);
    int runDecodeMTP(GDNParams const& params, cudaStream_t stream);

    static gdn_decode_Kernel_Module_t sDecodeModule;
    static gdn_prefill_Kernel_Module_t sPrefillModule;
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    static gdn_prefill_blackwell_Kernel_Module_t sBlackwellPrefillModule;
#endif
    static gdn_decode_mtp_cache_Kernel_Module_t sMTPDecodeCacheModule;
    static gdn_decode_tree_split_v_precomputed_Kernel_Module_t sDecodeTreeSplitVPrecomputedModule;
    static bool sLoaded;
};

} // namespace trt_edgellm

#endif // CUTE_DSL_GDN_ENABLED
