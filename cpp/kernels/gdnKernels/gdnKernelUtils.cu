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

#include "gdnKernelUtils.cuh"

#include <cuda_fp16.h>

#include <cmath>

namespace trt_edgellm
{

/**
 * Single-thread prefix-sum kernel: converts context_lengths[N] to cu_seqlens[N+1].
 *
 * cu_seqlens[0] = 0
 * cu_seqlens[i+1] = cu_seqlens[i] + context_lengths[i]
 */
__global__ void gdnCalCuSeqLensKernel(int32_t const* context_lengths, // [N]
    int32_t* cu_seqlens,                                              // [N+1]  output
    int32_t batchSize)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cu_seqlens[0] = 0;
        int32_t running = 0;
        for (int32_t i = 0; i < batchSize; ++i)
        {
            running += context_lengths[i];
            cu_seqlens[i + 1] = running;
        }
    }
}

void launchGdnCalCuSeqLens(void const* context_lengths, void* cu_seqlens, int32_t batchSize, cudaStream_t stream)
{
    gdnCalCuSeqLensKernel<<<1, 1, 0, stream>>>(
        static_cast<int32_t const*>(context_lengths), static_cast<int32_t*>(cu_seqlens), batchSize);
}

/**
 * L2-normalize Q and K along the head dimension (last axis) in-place.
 *
 * Input layout: (N * T * H, D) where D = head_dim (e.g. 128).
 * Each row (token-head pair) is divided by its L2 norm + eps.
 * One warp per row, warp-level reduction for the norm.
 */
__global__ void gdnL2NormQKKernel(half* data, // [numRows, headDim]
    int32_t numRows, int32_t headDim)
{
    int32_t const row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int32_t const lane = threadIdx.x % 32;
    if (row >= numRows)
        return;

    half* rowPtr = data + static_cast<int64_t>(row) * headDim;

    // Pass 1: compute sum of squares
    float sumSq = 0.0f;
    for (int32_t d = lane; d < headDim; d += 32)
    {
        float val = __half2float(rowPtr[d]);
        sumSq += val * val;
    }
    // Warp reduction
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        sumSq += __shfl_xor_sync(0xFFFFFFFF, sumSq, offset);
    }
    float invNorm = rsqrtf(sumSq + 1e-6f);

    // Pass 2: normalize in-place
    for (int32_t d = lane; d < headDim; d += 32)
    {
        float val = __half2float(rowPtr[d]);
        rowPtr[d] = __float2half(val * invNorm);
    }
}

void launchGdnL2NormQK(void* q, void* k, int32_t n, int32_t seqLen, int32_t h, int32_t headDim, cudaStream_t stream)
{
    int32_t const numRowsQ = n * seqLen * h;
    int32_t const warpsPerBlock = 8;
    int32_t const threadsPerBlock = warpsPerBlock * 32;
    int32_t const numBlocksQ = (numRowsQ + warpsPerBlock - 1) / warpsPerBlock;
    int32_t const numBlocksK = numBlocksQ; // same shape

    gdnL2NormQKKernel<<<numBlocksQ, threadsPerBlock, 0, stream>>>(static_cast<half*>(q), numRowsQ, headDim);
    gdnL2NormQKKernel<<<numBlocksK, threadsPerBlock, 0, stream>>>(static_cast<half*>(k), numRowsQ, headDim);
}

__device__ __forceinline__ float gdnSoftplus(float x)
{
    constexpr float kSoftplusThreshold{20.0F};
    if (x <= kSoftplusThreshold)
    {
        return logf(1.0F + expf(x));
    }
    return x;
}

__global__ void gdnDDTreePrecomputeKernel(half const* __restrict__ q, half const* __restrict__ k,
    half const* __restrict__ a, half const* __restrict__ b, float const* __restrict__ a_log,
    half const* __restrict__ dtBias, float* __restrict__ qkScales, float* __restrict__ gateValues, int32_t seqLen,
    int32_t h, int32_t hv, int32_t headDim, int32_t totalQKRows, int32_t totalGateRows, float qScale)
{
    (void) seqLen;
    constexpr int32_t kWarpSize{32};
    int32_t const lane = threadIdx.x % kWarpSize;
    int32_t const warpIdx = threadIdx.x / kWarpSize;
    int32_t const warpsPerBlock = blockDim.x / kWarpSize;
    int32_t const row = blockIdx.x * warpsPerBlock + warpIdx;

    if (row < totalQKRows)
    {
        int32_t const headIdx = row % h;
        int32_t const tokenBatch = row / h;
        int64_t const qkOffset = (static_cast<int64_t>(tokenBatch) * h + headIdx) * headDim;

        float qSum = 0.0F;
        float kSum = 0.0F;
        for (int32_t d = lane; d < headDim; d += kWarpSize)
        {
            float const qVal = __half2float(q[qkOffset + d]);
            float const kVal = __half2float(k[qkOffset + d]);
            qSum += qVal * qVal;
            kSum += kVal * kVal;
        }

        for (int32_t offset = kWarpSize / 2; offset > 0; offset >>= 1)
        {
            qSum += __shfl_xor_sync(0xFFFFFFFFU, qSum, offset);
            kSum += __shfl_xor_sync(0xFFFFFFFFU, kSum, offset);
        }

        if (lane == 0)
        {
            qkScales[row * 2] = rsqrtf(qSum + 1.0e-6F) * qScale;
            qkScales[row * 2 + 1] = rsqrtf(kSum + 1.0e-6F);
        }
    }

    if (row < totalGateRows && lane == 0)
    {
        int32_t const hvIdx = row % hv;
        int32_t const tokenBatch = row / hv;
        int64_t const gateOffset = static_cast<int64_t>(tokenBatch) * hv + hvIdx;
        float const x = __half2float(a[gateOffset]) + __half2float(dtBias[hvIdx]);
        float const sp = gdnSoftplus(x);
        float const g = expf(-expf(a_log[hvIdx]) * sp);
        float const beta = 1.0F / (1.0F + expf(-__half2float(b[gateOffset])));
        gateValues[row * 2] = g;
        gateValues[row * 2 + 1] = beta;
    }
}

void launchGdnDDTreePrecompute(void const* q, void const* k, void const* a, void const* b, void const* a_log,
    void const* dtBias, void* qkScales, void* gateValues, int32_t n, int32_t seqLen, int32_t h, int32_t hv,
    int32_t headDim, cudaStream_t stream)
{
    int32_t const totalQKRows = n * seqLen * h;
    int32_t const totalGateRows = n * seqLen * hv;
    int32_t const numBlocks = totalQKRows > totalGateRows ? totalQKRows : totalGateRows;
    if (numBlocks <= 0)
    {
        return;
    }
    constexpr int32_t kThreadsPerBlock{256};
    constexpr int32_t kWarpsPerBlock{kThreadsPerBlock / 32};
    float const qScale = 1.0F / std::sqrt(static_cast<float>(headDim));
    dim3 const grid((numBlocks + kWarpsPerBlock - 1) / kWarpsPerBlock);
    gdnDDTreePrecomputeKernel<<<grid, kThreadsPerBlock, 0, stream>>>(static_cast<half const*>(q),
        static_cast<half const*>(k), static_cast<half const*>(a), static_cast<half const*>(b),
        static_cast<float const*>(a_log), static_cast<half const*>(dtBias), static_cast<float*>(qkScales),
        static_cast<float*>(gateValues), seqLen, h, hv, headDim, totalQKRows, totalGateRows, qScale);
}

/**
 * Transpose the last two dims of a batch of square float32 matrices.
 *
 * Layout: src/dst are contiguous (numBlocks, dim, dim) float32.
 * One block handles one (dim, dim) matrix using 32x32 shared-memory tiles
 * with +1 padding to avoid bank conflicts.
 *
 * Grid: (numTilesPerMatrix, numBlocks)  where numTilesPerMatrix = (dim/32)^2.
 * Block: (32, 8)  — 8 rows per thread to amortize tile overhead.
 */
__global__ void gdnStateTransposeKernel(
    float const* __restrict__ src, float* __restrict__ dst, int32_t numBlocks, int32_t dim)
{
    // Shared tile with bank-conflict padding
    __shared__ float tile[32][33];

    int32_t const tilesPerRow = dim / 32;
    int32_t const tileIdx = blockIdx.x;            // which 32x32 tile within the matrix
    int32_t const matIdx = blockIdx.y;             // which matrix
    int32_t const tileRow = tileIdx / tilesPerRow; // tile row in the matrix
    int32_t const tileCol = tileIdx % tilesPerRow; // tile col in the matrix

    int64_t const matOffset = static_cast<int64_t>(matIdx) * dim * dim;

    // Read tile from src[matIdx, tileRow*32.., tileCol*32..]
    int32_t const baseRow = tileRow * 32;
    int32_t const baseCol = tileCol * 32;

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const r = baseRow + threadIdx.y + j;
        int32_t const c = baseCol + threadIdx.x;
        if (r < dim && c < dim)
        {
            tile[threadIdx.y + j][threadIdx.x] = src[matOffset + static_cast<int64_t>(r) * dim + c];
        }
    }
    __syncthreads();

    // Write transposed tile to dst[matIdx, tileCol*32.., tileRow*32..]
    int32_t const dstBaseRow = baseCol; // transposed
    int32_t const dstBaseCol = baseRow;

    for (int32_t j = 0; j < 32; j += 8)
    {
        int32_t const r = dstBaseRow + threadIdx.y + j;
        int32_t const c = dstBaseCol + threadIdx.x;
        if (r < dim && c < dim)
        {
            dst[matOffset + static_cast<int64_t>(r) * dim + c] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void launchGdnStateTranspose(void const* src, void* dst, int32_t numBlocks, int32_t dim, cudaStream_t stream)
{
    int32_t const tilesPerRow = (dim + 31) / 32;
    int32_t const tilesPerMatrix = tilesPerRow * tilesPerRow;
    dim3 grid(tilesPerMatrix, numBlocks);
    dim3 block(32, 8);
    gdnStateTransposeKernel<<<grid, block, 0, stream>>>(
        static_cast<float const*>(src), static_cast<float*>(dst), numBlocks, dim);
}

} // namespace trt_edgellm
