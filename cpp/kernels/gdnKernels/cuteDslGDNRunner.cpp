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

#ifdef CUTE_DSL_GDN_ENABLED

#include "cuteDslGDNRunner.h"
#include "gdnKernelUtils.cuh"

#include "common/logger.h"

#include <cmath>
#include <cstring>
#include <mutex>

namespace trt_edgellm
{

gdn_decode_Kernel_Module_t CuteDslGDNRunner::sDecodeModule = {};
gdn_prefill_Kernel_Module_t CuteDslGDNRunner::sPrefillModule = {};
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
gdn_prefill_blackwell_Kernel_Module_t CuteDslGDNRunner::sBlackwellPrefillModule = {};
#endif
gdn_decode_mtp_cache_Kernel_Module_t CuteDslGDNRunner::sMTPDecodeCacheModule = {};
gdn_decode_tree_split_v_precomputed_Kernel_Module_t CuteDslGDNRunner::sDecodeTreeSplitVPrecomputedModule = {};
bool CuteDslGDNRunner::sLoaded = false;

static std::mutex sGDNMutex;

#define SET_4D_TENSOR(tensor, data_ptr, dim0, dim1, dim2, dim3)                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_shapes[2] = (dim2);                                                                           \
        (tensor).dynamic_shapes[3] = (dim3);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1) * (dim2) * (dim3);                                    \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(dim2) * (dim3);                                             \
        (tensor).dynamic_strides[2] = static_cast<int64_t>(dim3);                                                      \
    } while (0)

#define SET_3D_TENSOR(tensor, data_ptr, dim0, dim1, dim2)                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_shapes[2] = (dim2);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1) * (dim2);                                             \
        (tensor).dynamic_strides[1] = static_cast<int64_t>(dim2);                                                      \
    } while (0)

#define SET_2D_TENSOR(tensor, data_ptr, dim0, dim1)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
        (tensor).dynamic_shapes[1] = (dim1);                                                                           \
        (tensor).dynamic_strides[0] = static_cast<int64_t>(dim1);                                                      \
    } while (0)

#define SET_1D_TENSOR(tensor, data_ptr, dim0)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        (tensor).data = (data_ptr);                                                                                    \
        (tensor).dynamic_shapes[0] = (dim0);                                                                           \
    } while (0)

bool CuteDslGDNRunner::canImplement(int32_t kDim, int32_t vDim, int32_t smVersion)
{
    return (smVersion >= 80) && (kDim == 128) && (vDim == 128);
}

bool CuteDslGDNRunner::loadKernelModules()
{
    std::lock_guard<std::mutex> lock(sGDNMutex);
    if (sLoaded)
    {
        return true;
    }
    try
    {
        gdn_decode_Kernel_Module_Load(&sDecodeModule);
        gdn_prefill_Kernel_Module_Load(&sPrefillModule);
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
        gdn_prefill_blackwell_Kernel_Module_Load(&sBlackwellPrefillModule);
#endif
        gdn_decode_mtp_cache_Kernel_Module_Load(&sMTPDecodeCacheModule);
        gdn_decode_tree_split_v_precomputed_Kernel_Module_Load(&sDecodeTreeSplitVPrecomputedModule);
        LOG_DEBUG(
            "CuTe DSL GDN kernel modules loaded (decode + prefill + MTP + DDTree"
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
            " + blackwell"
#endif
            ")");
        sLoaded = true;
        return true;
    }
    catch (...)
    {
        LOG_ERROR("Failed to load CuTe DSL GDN kernel modules");
        return false;
    }
}

void CuteDslGDNRunner::unloadKernelModules()
{
    std::lock_guard<std::mutex> lock(sGDNMutex);
    if (sLoaded)
    {
        gdn_decode_Kernel_Module_Unload(&sDecodeModule);
        gdn_prefill_Kernel_Module_Unload(&sPrefillModule);
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
        gdn_prefill_blackwell_Kernel_Module_Unload(&sBlackwellPrefillModule);
#endif
        gdn_decode_mtp_cache_Kernel_Module_Unload(&sMTPDecodeCacheModule);
        gdn_decode_tree_split_v_precomputed_Kernel_Module_Unload(&sDecodeTreeSplitVPrecomputedModule);
        sLoaded = false;
    }
}

int CuteDslGDNRunner::run(GDNParams const& params, cudaStream_t stream)
{
    if (params.use_ddtree)
    {
        // DDTree verifies many tree nodes from one committed recurrent state.
        // The tree kernels keep h0_source read-only and emit per-node checkpoints for accept/commit.
        return runDecodeTree(params, stream);
    }
    // MTP decode takes priority: handles any seq_len for speculative-decoding verification.
    if (params.use_mtp)
    {
        return runDecodeMTP(params, stream);
    }
    if (params.seq_len == 1)
    {
        return runDecode(params, stream);
    }
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    if (params.smVersion >= 100)
    {
        return runPrefillBlackwell(params, stream);
    }
#endif
    return runPrefill(params, stream);
}

int CuteDslGDNRunner::runDecodeTree(GDNParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL GDN DDTree decode kernel module not loaded.");
        return -1;
    }

    if (params.seq_len < 1)
    {
        LOG_ERROR("CuTe DSL GDN DDTree decode requires seq_len >= 1, got %d", params.seq_len);
        return -1;
    }
    constexpr int32_t kDDTreeGDNMaxSeqLen = 128;
    if (params.seq_len > kDDTreeGDNMaxSeqLen)
    {
        LOG_ERROR("CuTe DSL GDN DDTree decode supports seq_len <= %d, got %d", kDDTreeGDNMaxSeqLen, params.seq_len);
        return -1;
    }
    if (params.k_dim != 128 || params.v_dim != 128)
    {
        LOG_ERROR(
            "CuTe DSL GDN DDTree decode requires k_dim=v_dim=128, got k_dim=%d v_dim=%d", params.k_dim, params.v_dim);
        return -1;
    }
    if (params.h <= 0 || params.hv <= 0 || params.hv < params.h || (params.hv % params.h) != 0)
    {
        LOG_ERROR("CuTe DSL GDN DDTree decode requires hv >= h and hv %% h == 0, got h=%d hv=%d", params.h, params.hv);
        return -1;
    }
    if (params.tree_parent_ids == nullptr || params.tree_depths == nullptr)
    {
        LOG_ERROR("CuTe DSL GDN DDTree decode requires non-null tree_parent_ids and tree_depths.");
        return -1;
    }
    if (params.intermediate_states == nullptr)
    {
        LOG_ERROR("CuTe DSL GDN DDTree decode requires intermediate_states output for rollback state checkpoints.");
        return -1;
    }

    if (params.ddtree_qk_scales == nullptr || params.ddtree_gate_values == nullptr)
    {
        LOG_ERROR("CuTe DSL GDN DDTree decode requires precomputed split-v workspace buffers.");
        return -1;
    }

    return runDecodeTreeSplitVPrecomputed(params, stream);
}

int CuteDslGDNRunner::runDecodeTreeSplitVPrecomputed(GDNParams const& params, cudaStream_t stream)
{
    int32_t const n = params.n;
    int32_t const seqLen = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    launchGdnDDTreePrecompute(params.q, params.k, params.a, params.b, params.A_log, params.dt_bias,
        params.ddtree_qk_scales, params.ddtree_gate_values, n, seqLen, h, hv, k, stream);

    gdn_decode_tree_split_v_precomputed_Tensor_h0_source_t h0Tensor{};
    h0Tensor.data = params.h0_source;
    h0Tensor.dynamic_shapes[0] = n;
    h0Tensor.dynamic_shapes[1] = hv;
    h0Tensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    gdn_decode_tree_split_v_precomputed_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seqLen, h, k);

    gdn_decode_tree_split_v_precomputed_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seqLen, h, k);

    gdn_decode_tree_split_v_precomputed_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seqLen, hv, v);

    gdn_decode_tree_split_v_precomputed_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seqLen, hv, v);

    gdn_decode_tree_split_v_precomputed_Tensor_intermediate_states_t intermTensor{};
    intermTensor.data = params.intermediate_states;
    intermTensor.dynamic_shapes[0] = n;
    intermTensor.dynamic_shapes[1] = seqLen;
    intermTensor.dynamic_shapes[2] = hv;
    intermTensor.dynamic_strides[0] = static_cast<int64_t>(seqLen) * hv * k * v;
    intermTensor.dynamic_strides[1] = static_cast<int64_t>(hv) * k * v;
    intermTensor.dynamic_strides[2] = static_cast<int64_t>(k) * v;
    intermTensor.dynamic_strides[3] = static_cast<int64_t>(v);

    gdn_decode_tree_split_v_precomputed_Tensor_tree_parent_ids_t parentTensor{};
    SET_2D_TENSOR(parentTensor, params.tree_parent_ids, n, seqLen);

    gdn_decode_tree_split_v_precomputed_Tensor_tree_depths_t depthTensor{};
    SET_2D_TENSOR(depthTensor, params.tree_depths, n, seqLen);

    gdn_decode_tree_split_v_precomputed_Tensor_qk_scales_t qkScalesTensor{};
    SET_4D_TENSOR(qkScalesTensor, params.ddtree_qk_scales, n, seqLen, h, 2);

    gdn_decode_tree_split_v_precomputed_Tensor_gate_values_t gateValuesTensor{};
    SET_4D_TENSOR(gateValuesTensor, params.ddtree_gate_values, n, seqLen, hv, 2);

    cute_dsl_gdn_decode_tree_split_v_precomputed_wrapper(&sDecodeTreeSplitVPrecomputedModule, &h0Tensor, &qTensor,
        &kTensor, &vTensor, &oTensor, &intermTensor, &parentTensor, &depthTensor, &qkScalesTensor, &gateValuesTensor,
        seqLen, stream);
    return 0;
}

int CuteDslGDNRunner::runDecode(GDNParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL GDN decode kernel module not loaded.");
        return -1;
    }
    int32_t const n = params.n;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    if (params.seq_len != 1)
    {
        LOG_ERROR("CuTe DSL GDN decode requires seq_len == 1, got %d", params.seq_len);
        return -1;
    }

    gdn_decode_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, 1, h, k);

    gdn_decode_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, 1, h, k);

    gdn_decode_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, 1, hv, v);

    gdn_decode_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, 1, hv);

    gdn_decode_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, 1, hv);

    gdn_decode_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_decode_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    gdn_decode_Tensor_h0_source_t h0_sourceTensor{};
    h0_sourceTensor.data = params.h0_source;
    h0_sourceTensor.dynamic_shapes[0] = n;
    h0_sourceTensor.dynamic_shapes[1] = params.hv;
    h0_sourceTensor.dynamic_strides[0] = static_cast<int64_t>(params.hv) * params.k_dim * params.v_dim;

    gdn_decode_Tensor_context_lengths_t contextLengthsTensor{};
    SET_1D_TENSOR(contextLengthsTensor, params.context_lengths, n);

    gdn_decode_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, 1, hv, v);

    cute_dsl_gdn_decode_wrapper(&sDecodeModule, &qTensor, &kTensor, &vTensor, &aTensor, &bTensor, &A_logTensor,
        &dt_biasTensor, &h0_sourceTensor, &contextLengthsTensor, &oTensor, stream);

    return 0;
}

int CuteDslGDNRunner::runPrefill(GDNParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL GDN prefill kernel module not loaded.");
        return -1;
    }
    int32_t const n = params.n;
    int32_t const seq_len = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    gdn_prefill_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seq_len, h, k);

    gdn_prefill_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seq_len, h, k);

    gdn_prefill_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seq_len, hv, v);

    gdn_prefill_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seq_len, hv);

    gdn_prefill_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seq_len, hv);

    gdn_prefill_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_prefill_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    gdn_prefill_Tensor_h0_source_t h0_sourceTensor{};
    h0_sourceTensor.data = params.h0_source;
    h0_sourceTensor.dynamic_shapes[0] = n;
    h0_sourceTensor.dynamic_shapes[1] = params.hv;
    h0_sourceTensor.dynamic_strides[0] = static_cast<int64_t>(params.hv) * params.k_dim * params.v_dim;

    gdn_prefill_Tensor_context_lengths_t contextLengthsTensor{};
    SET_1D_TENSOR(contextLengthsTensor, params.context_lengths, n);

    gdn_prefill_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seq_len, hv, v);

    cute_dsl_gdn_prefill_wrapper(&sPrefillModule, &qTensor, &kTensor, &vTensor, &aTensor, &bTensor, &A_logTensor,
        &dt_biasTensor, &h0_sourceTensor, &contextLengthsTensor, &oTensor, seq_len, stream);

    return 0;
}

int CuteDslGDNRunner::runPrefillBlackwell(GDNParams const& params, cudaStream_t stream)
{
#ifdef CUTE_DSL_GDN_BLACKWELL_ENABLED
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL GDN Blackwell prefill kernel module not loaded.");
        return -1;
    }
    int32_t const n = params.n;
    int32_t const seq_len = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    // L2-normalize Q and K in-place.  The Blackwell kernel expects pre-normalized
    // Q/K so that QK^T, KK^T, state update, and output paths are all consistent.
    launchGdnL2NormQK(params.q, params.k, n, seq_len, h, k, stream);

    // Set up tensor structs for fused Blackwell kernel (g/beta computed inline in warp 7)
    gdn_prefill_blackwell_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seq_len, h, k);

    gdn_prefill_blackwell_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seq_len, h, k);

    gdn_prefill_blackwell_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seq_len, hv, v);

    gdn_prefill_blackwell_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seq_len, hv);

    gdn_prefill_blackwell_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seq_len, hv);

    // A_log and dt_bias are constant-shape tensors in the Blackwell kernel — data pointer only.
    gdn_prefill_blackwell_Tensor_A_log_t A_logTensor{};
    A_logTensor.data = params.A_log;

    gdn_prefill_blackwell_Tensor_dt_bias_t dt_biasTensor{};
    dt_biasTensor.data = params.dt_bias;

    // The Blackwell MMA computes state = V^T * K in V-major (d_v, d_k) order,
    // while sequential/decode kernels use K-major (d_k, d_v).  We transpose
    // h0 before and after the kernel to bridge this convention mismatch.
    // The kernel is not in-place safe, so we ping-pong between h0_source and h0_scratch.
    size_t const h0ScratchBytes = static_cast<size_t>(n) * hv * k * v * sizeof(float);
    if (params.h0_scratch == nullptr)
    {
        LOG_WARNING(
            "GDN Blackwell prefill: h0_scratch not provided in GDNParams — "
            "caller must allocate [n=%d, hv=%d, k=%d, v=%d] f32 scratch and set params.h0_scratch.",
            n, hv, k, v);
        return -1;
    }

    int32_t const numStateBlocks = n * hv;

    // Step 1: Transpose initial state from K-major to V-major into scratch buffer.
    launchGdnStateTranspose(params.h0_source, params.h0_scratch, numStateBlocks, k, stream);

    gdn_prefill_blackwell_Tensor_h0_in_t h0InTensor{};
    h0InTensor.data = params.h0_scratch; // V-major initial state
    h0InTensor.dynamic_shapes[0] = n;
    h0InTensor.dynamic_shapes[1] = hv;
    h0InTensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    gdn_prefill_blackwell_Tensor_h0_out_t h0OutTensor{};
    h0OutTensor.data = params.h0_source; // V-major output written here
    h0OutTensor.dynamic_shapes[0] = n;
    h0OutTensor.dynamic_shapes[1] = hv;
    h0OutTensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    // cu_seqlens [N+1]: prefix-sum of context_lengths, computed by the plugin before launch.
    gdn_prefill_blackwell_Tensor_cu_seqlens_t cuSeqLensTensor{};
    SET_1D_TENSOR(cuSeqLensTensor, params.cu_seqlens, n + 1);

    gdn_prefill_blackwell_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seq_len, hv, v);

    // Step 2: Run the Blackwell prefill kernel.
    cute_dsl_gdn_prefill_blackwell_wrapper(&sBlackwellPrefillModule, &qTensor, &kTensor, &vTensor, &aTensor, &bTensor,
        &A_logTensor, &dt_biasTensor, &h0InTensor, &h0OutTensor, &oTensor, &cuSeqLensTensor, stream);

    // Step 3: Transpose V-major output state back to K-major into scratch.
    launchGdnStateTranspose(params.h0_source, params.h0_scratch, numStateBlocks, k, stream);

    // Step 4: Copy K-major state from scratch back to h0_source (stream-ordered).
    cudaMemcpyAsync(params.h0_source, params.h0_scratch, h0ScratchBytes, cudaMemcpyDeviceToDevice, stream);
    return 0;
#else
    LOG_ERROR("Blackwell GDN prefill not compiled in this build.");
    return -1;
#endif
}

int CuteDslGDNRunner::runDecodeMTP(GDNParams const& params, cudaStream_t stream)
{
    if (!sLoaded)
    {
        LOG_ERROR("CuTe DSL GDN MTP decode kernel module not loaded.");
        return -1;
    }

    int32_t const n = params.n;
    int32_t const seq_len = params.seq_len;
    int32_t const h = params.h;
    int32_t const hv = params.hv;
    int32_t const k = params.k_dim;
    int32_t const v = params.v_dim;

    if (seq_len < 1)
    {
        LOG_ERROR("CuTe DSL GDN MTP decode requires seq_len >= 1, got %d", seq_len);
        return -1;
    }

    // Must match T_MAX_AOT in gdn_decode_mtp.py.
    constexpr int32_t kMTPMaxSeqLen = 16;
    if (seq_len > kMTPMaxSeqLen)
    {
        LOG_ERROR("CuTe DSL GDN MTP decode requires seq_len <= %d, got %d", kMTPMaxSeqLen, seq_len);
        return -1;
    }

    // q / k : [n, seq_len, h, k]
    gdn_decode_mtp_cache_Tensor_q_t qTensor{};
    SET_4D_TENSOR(qTensor, params.q, n, seq_len, h, k);

    gdn_decode_mtp_cache_Tensor_k_t kTensor{};
    SET_4D_TENSOR(kTensor, params.k, n, seq_len, h, k);

    // v : [n, seq_len, hv, v]
    gdn_decode_mtp_cache_Tensor_v_t vTensor{};
    SET_4D_TENSOR(vTensor, params.v, n, seq_len, hv, v);

    // a / b : [n, seq_len, hv]
    gdn_decode_mtp_cache_Tensor_a_t aTensor{};
    SET_3D_TENSOR(aTensor, params.a, n, seq_len, hv);

    gdn_decode_mtp_cache_Tensor_b_t bTensor{};
    SET_3D_TENSOR(bTensor, params.b, n, seq_len, hv);

    // A_log / dt_bias : [hv]
    gdn_decode_mtp_cache_Tensor_A_log_t A_logTensor{};
    SET_1D_TENSOR(A_logTensor, params.A_log, hv);

    gdn_decode_mtp_cache_Tensor_dt_bias_t dt_biasTensor{};
    SET_1D_TENSOR(dt_biasTensor, params.dt_bias, hv);

    // h0_source : [n, hv, k, v]  (batch-dense, in-place updated)
    gdn_decode_mtp_cache_Tensor_h0_source_t h0Tensor{};
    h0Tensor.data = params.h0_source;
    h0Tensor.dynamic_shapes[0] = n;
    h0Tensor.dynamic_shapes[1] = hv;
    h0Tensor.dynamic_strides[0] = static_cast<int64_t>(hv) * k * v;

    // o : [n, seq_len, hv, v]
    gdn_decode_mtp_cache_Tensor_o_t oTensor{};
    SET_4D_TENSOR(oTensor, params.o, n, seq_len, hv, v);

    if (params.intermediate_states == nullptr)
    {
        LOG_ERROR("CuTe DSL GDN MTP: intermediate_states is null.");
        return -1;
    }

    // intermediate_states : [n, seq_len, hv, k, v]
    gdn_decode_mtp_cache_Tensor_intermediate_states_t intermTensor{};
    intermTensor.data = params.intermediate_states;
    intermTensor.dynamic_shapes[0] = n;
    intermTensor.dynamic_shapes[1] = seq_len;
    intermTensor.dynamic_shapes[2] = hv;
    intermTensor.dynamic_strides[0] = static_cast<int64_t>(seq_len) * hv * k * v;
    intermTensor.dynamic_strides[1] = static_cast<int64_t>(hv) * k * v;
    intermTensor.dynamic_strides[2] = static_cast<int64_t>(k) * v;
    intermTensor.dynamic_strides[3] = static_cast<int64_t>(v);

    cute_dsl_gdn_decode_mtp_cache_wrapper(&sMTPDecodeCacheModule, &h0Tensor, &qTensor, &kTensor, &vTensor, &aTensor,
        &bTensor, &A_logTensor, &dt_biasTensor, &oTensor, &intermTensor, seq_len, stream);

    return 0;
}

} // namespace trt_edgellm

#endif // CUTE_DSL_GDN_ENABLED
