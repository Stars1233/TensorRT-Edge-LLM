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

// Batched MTP State Scatter Kernels
//
// After MTP speculative decoding verification, the base model's recurrent
// (GDN) and conv1d states must be updated to the last accepted step.
// During verify, the MTP kernel (with cache ON, state update ON) writes:
//   - h0_out = state after ALL T steps (wrong when only L < T tokens accepted)
//   - intermediate_states[t] = state snapshot after step t
//
// This kernel copies the correct snapshot back to the main state pool, batched
// across all GDN/conv layers in a single launch:
//   dst[layer][batch, :] = src[layer][batch, accepted_step, :]
//
// Per-layer pointers come from a device-resident MtpLayerInfo array (same
// pattern as KVLayerInfo in HybridCacheManager).  Recurrent vs conv field
// selection is compile-time via the StateKind template parameter.
//
// Design follows SGLang's fused_mamba_state_scatter_with_mask (Triton),
// adapted to CUDA using edge-llm's DVec<T> vectorized load/store.
//
// Grid: (batchSize, numLayers, ceil(vecCount / blockDim.x))
//   - blockIdx.y = layer index → MtpLayerInfo lookup
//   - Early exit for skip (acceptLength <= 0) and all-accept (acceptLength >= verifyTreeSize)
//   - DVec<float> = 8 floats (256-bit), DVec<half> = 8 halves (128-bit)

#include "mtpStateScatterKernels.h"

#include "kernels/common/vectorizedTypes.cuh"

#include "common/checkMacros.h"
#include <stdexcept>

namespace trt_edgellm
{
namespace kernel
{

namespace
{

enum class StateKind
{
    Recurrent,
    Conv
};

// Each thread copies one DVec (8 elements) for one (layer, batch, vecIdx).
template <typename T, StateKind Kind>
__global__ void mtpStateScatterKernel(MtpLayerInfo const* __restrict__ layerInfos, // [numLayers]
    int32_t const* __restrict__ acceptLengths,                                     // [batchSize]
    int32_t verifyTreeSize,
    int32_t vecCount) // stateElements / DVec<T>::vec_size
{
    int32_t const b = blockIdx.x;     // batch index
    int32_t const layer = blockIdx.y; // layer index

    // Convert 1-based accept length to 0-based step index.
    int32_t const step = acceptLengths[b] - 1;

    // Skip invalid batch items or all-T-tokens-accepted (h0 already correct).
    if (step < 0 || step >= verifyTreeSize - 1)
    {
        return;
    }

    constexpr int32_t kVecSize = DVec<T>::vec_size;

    int32_t const vecIdx = blockIdx.z * blockDim.x + threadIdx.x;
    if (vecIdx >= vecCount)
    {
        return;
    }

    auto const& info = layerInfos[layer];
    void* dstRaw;
    void const* srcRaw;
    if constexpr (Kind == StateKind::Recurrent)
    {
        dstRaw = info.recurrentDst;
        srcRaw = info.recurrentSrc;
    }
    else
    {
        dstRaw = info.convDst;
        srcRaw = info.convSrc;
    }
    auto* const dst = static_cast<T*>(dstRaw);
    auto const* const src = static_cast<T const*>(srcRaw);

    // dst layout: [batchSize, stateElements]
    int64_t const stateElems = static_cast<int64_t>(vecCount) * kVecSize;
    int64_t const dstScalar = static_cast<int64_t>(b) * stateElems + static_cast<int64_t>(vecIdx) * kVecSize;

    // src layout: [batchSize, verifyTreeSize, stateElements]
    int64_t const srcScalar
        = (static_cast<int64_t>(b) * verifyTreeSize + step) * stateElems + static_cast<int64_t>(vecIdx) * kVecSize;

    DVec<T> v;
    v.load(src + srcScalar);
    v.store(dst + dstScalar);
}

// Each thread copies one DVec (8 elements) for one (layer, batch, vecIdx).
// Unlike linear MTP, DDTree cannot derive the source step from acceptLength.
// It must use the last verified tree node id in the accepted path.
template <typename T, StateKind Kind>
__global__ void mtpAcceptedTreeStateScatterKernel(MtpLayerInfo const* __restrict__ layerInfos, // [numLayers]
    int32_t const* __restrict__ acceptedStateNodeIds, // [batchSize, maxAcceptLen]
    int32_t const* __restrict__ acceptLengths,        // [batchSize]
    int32_t maxAcceptLen, int32_t verifyTreeSize,
    int32_t vecCount) // stateElements / DVec<T>::vec_size
{
    int32_t const b = blockIdx.x;     // batch index
    int32_t const layer = blockIdx.y; // layer index

    int32_t acceptedCount = acceptLengths[b];
    if (acceptedCount > maxAcceptLen)
    {
        acceptedCount = maxAcceptLen;
    }

    int32_t nodeId = -1;
    for (int32_t i = acceptedCount - 1; i >= 0; --i)
    {
        int32_t const candidate = acceptedStateNodeIds[b * maxAcceptLen + i];
        if (candidate >= 0)
        {
            nodeId = candidate;
            break;
        }
    }

    if (nodeId < 0 || nodeId >= verifyTreeSize)
    {
        return;
    }

    constexpr int32_t kVecSize = DVec<T>::vec_size;

    int32_t const vecIdx = blockIdx.z * blockDim.x + threadIdx.x;
    if (vecIdx >= vecCount)
    {
        return;
    }

    auto const& info = layerInfos[layer];
    void* dstRaw;
    void const* srcRaw;
    if constexpr (Kind == StateKind::Recurrent)
    {
        dstRaw = info.recurrentDst;
        srcRaw = info.recurrentSrc;
    }
    else
    {
        dstRaw = info.convDst;
        srcRaw = info.convSrc;
    }
    auto* const dst = static_cast<T*>(dstRaw);
    auto const* const src = static_cast<T const*>(srcRaw);

    // dst layout: [batchSize, stateElements]
    int64_t const stateElems = static_cast<int64_t>(vecCount) * kVecSize;
    int64_t const dstScalar = static_cast<int64_t>(b) * stateElems + static_cast<int64_t>(vecIdx) * kVecSize;

    // src layout: [batchSize, verifyTreeSize, stateElements]
    int64_t const srcScalar
        = (static_cast<int64_t>(b) * verifyTreeSize + nodeId) * stateElems + static_cast<int64_t>(vecIdx) * kVecSize;

    DVec<T> v;
    v.load(src + srcScalar);
    v.store(dst + dstScalar);
}

template <typename T, StateKind Kind>
void launchScatter(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptLengths, cudaStream_t stream)
{
    if (numLayers == 0 || activeBatchSize == 0 || stateElements == 0)
    {
        return;
    }

    constexpr int32_t kVecSize = DVec<T>::vec_size;
    ELLM_CHECK(stateElements % kVecSize == 0, "stateElements must be divisible by DVec vec_size (8)");

    int32_t const vecCount = stateElements / kVecSize;

    // 256 threads per block — good default for memory-bound vectorized copy.
    constexpr int32_t kThreads = 256;

    // Grid: (batch, layer, ceil(vecCount / kThreads))
    int32_t const zBlocks = (vecCount + kThreads - 1) / kThreads;
    dim3 const grid(activeBatchSize, numLayers, zBlocks);
    dim3 const block(kThreads);

    mtpStateScatterKernel<T, Kind>
        <<<grid, block, 0, stream>>>(deviceLayerInfos, acceptLengths, verifyTreeSize, vecCount);
}

template <typename T, StateKind Kind>
void launchAcceptedTreeScatter(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptedStateNodeIds, int32_t maxAcceptLen,
    int32_t const* acceptLengths, cudaStream_t stream)
{
    if (numLayers == 0 || activeBatchSize == 0 || stateElements == 0 || maxAcceptLen == 0)
    {
        return;
    }

    constexpr int32_t kVecSize = DVec<T>::vec_size;
    ELLM_CHECK(stateElements % kVecSize == 0, "stateElements must be divisible by DVec vec_size (8)");

    int32_t const vecCount = stateElements / kVecSize;

    // 256 threads per block — good default for memory-bound vectorized copy.
    constexpr int32_t kThreads = 256;

    // Grid: (batch, layer, ceil(vecCount / kThreads))
    int32_t const zBlocks = (vecCount + kThreads - 1) / kThreads;
    dim3 const grid(activeBatchSize, numLayers, zBlocks);
    dim3 const block(kThreads);

    mtpAcceptedTreeStateScatterKernel<T, Kind><<<grid, block, 0, stream>>>(
        deviceLayerInfos, acceptedStateNodeIds, acceptLengths, maxAcceptLen, verifyTreeSize, vecCount);
}

} // anonymous namespace

void mtpScatterRecurrentStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptLengths, cudaStream_t stream)
{
    launchScatter<float, StateKind::Recurrent>(
        deviceLayerInfos, numLayers, activeBatchSize, verifyTreeSize, stateElements, acceptLengths, stream);
}

void mtpScatterConvStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptLengths, cudaStream_t stream)
{
    launchScatter<half, StateKind::Conv>(
        deviceLayerInfos, numLayers, activeBatchSize, verifyTreeSize, stateElements, acceptLengths, stream);
}

void mtpScatterAcceptedTreeRecurrentStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers,
    int32_t activeBatchSize, int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptedStateNodeIds,
    int32_t maxAcceptLen, int32_t const* acceptLengths, cudaStream_t stream)
{
    launchAcceptedTreeScatter<float, StateKind::Recurrent>(deviceLayerInfos, numLayers, activeBatchSize, verifyTreeSize,
        stateElements, acceptedStateNodeIds, maxAcceptLen, acceptLengths, stream);
}

void mtpScatterAcceptedTreeConvStates(MtpLayerInfo const* deviceLayerInfos, int32_t numLayers, int32_t activeBatchSize,
    int32_t verifyTreeSize, int32_t stateElements, int32_t const* acceptedStateNodeIds, int32_t maxAcceptLen,
    int32_t const* acceptLengths, cudaStream_t stream)
{
    launchAcceptedTreeScatter<half, StateKind::Conv>(deviceLayerInfos, numLayers, activeBatchSize, verifyTreeSize,
        stateElements, acceptedStateNodeIds, maxAcceptLen, acceptLengths, stream);
}

} // namespace kernel
} // namespace trt_edgellm
