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

#if defined(CUTE_DSL_FFPA_ENABLED)

#include "kernels/contextAttentionKernels/cuteDslFFPARunner.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "contextAttnReference.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace nvinfer1;
using namespace trt_edgellm;

namespace
{

bool isFfpaSupportedSm(int32_t smVersion)
{
    return CuteDslFFPARunner::canImplement(512, smVersion, 1, 1);
}

void initializeFp16(rt::Tensor& tensor, int32_t seed)
{
    std::vector<__half> host(static_cast<size_t>(tensor.getShape().volume()));
    for (size_t index = 0; index < host.size(); ++index)
    {
        int32_t const value = (static_cast<int32_t>((index * 17 + seed * 13) % 67) - 33);
        host[index] = __float2half(static_cast<float>(value) * 0.0078125F);
    }
    copyHostToDevice(tensor, host);
}

void initializeNormalFp16(rt::Tensor& tensor, uint32_t seed)
{
    std::vector<__half> host(static_cast<size_t>(tensor.getShape().volume()));
    std::mt19937 rng{seed};
    // N(0, 1/sqrt(D)) keeps QK^T row variance unit-ish after the 1/sqrt(D) scale,
    // exercising the realistic softmax dynamic range rather than near-zero scores.
    std::normal_distribution<float> dist{0.0F, 1.0F / std::sqrt(512.0F)};
    for (size_t index = 0; index < host.size(); ++index)
    {
        host[index] = __float2half(dist(rng));
    }
    copyHostToDevice(tensor, host);
}

//! Build a (B+1,) int32 device tensor of cumulative sequence lengths.
rt::Tensor makeCuSeqLens(std::vector<int32_t> const& lens)
{
    std::vector<int32_t> cu(lens.size() + 1, 0);
    for (size_t i = 0; i < lens.size(); ++i)
    {
        cu[i + 1] = cu[i] + lens[i];
    }
    rt::Tensor tensor(rt::Coords{static_cast<int32_t>(cu.size())}, rt::DeviceType::kGPU, DataType::kINT32);
    copyHostToDevice(tensor, cu);
    return tensor;
}

//! Poison padding rows (>= lens[b]) of a [B, S, H, D] fp16 tensor with random
//! near-fp16-max garbage — models the token-0 / PLE-0 large-magnitude values
//! that padding positions carry.  When nanPoison is set, fill
//! with NaN (first padding row: Inf) instead — padding rows legitimately hold
//! NaN/Inf mid-network once fp16 overflow of garbage embeddings has occurred
//! in non-attention layers, and the kernel must not leak them into valid rows
//! (IEEE 0 x NaN = NaN defeats score-only masking).
void poisonPaddingRows(rt::Tensor& tensor, std::vector<int32_t> const& lens, uint32_t seed, bool nanPoison = false)
{
    auto host = copyDeviceToHost<__half>(tensor);
    auto const& shape = tensor.getShape();
    int64_t const batch = shape[0];
    int64_t const seqLen = shape[1];
    int64_t const rowElems = shape[2] * shape[3];
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> dist{-60000.0F, 60000.0F};
    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t s = lens[static_cast<size_t>(b)]; s < seqLen; ++s)
        {
            int64_t const base = (b * seqLen + s) * rowElems;
            for (int64_t e = 0; e < rowElems; ++e)
            {
                float value = dist(rng);
                if (nanPoison)
                {
                    value = (s == lens[static_cast<size_t>(b)]) ? std::numeric_limits<float>::infinity()
                                                                : std::numeric_limits<float>::quiet_NaN();
                }
                host[static_cast<size_t>(base + e)] = __float2half(value);
            }
        }
    }
    copyHostToDevice(tensor, host);
}

void expectFp16Close(rt::Tensor const& actual, rt::Tensor const& expected, std::string const& label)
{
    ASSERT_EQ(actual.getShape().volume(), expected.getShape().volume()) << label;
    auto const actualHost = copyDeviceToHost<__half>(actual);
    auto const expectedHost = copyDeviceToHost<__half>(expected);
    auto const& shape = actual.getShape();

    bool nanDetected = false;
    int64_t closeWithin1e3 = 0;
    int64_t const totalElements = static_cast<int64_t>(actualHost.size());

    for (int64_t idx = 0; idx < totalElements; ++idx)
    {
        float const actualValue = __half2float(actualHost[static_cast<size_t>(idx)]);
        float const expectedValue = __half2float(expectedHost[static_cast<size_t>(idx)]);

        ASSERT_FALSE(std::isnan(actualValue)) << label << " produced NaN at " << formatTensorIndex(shape, idx);
        ASSERT_FALSE(std::isinf(actualValue)) << label << " produced Inf at " << formatTensorIndex(shape, idx);

        ASSERT_TRUE(isclose(actualHost[static_cast<size_t>(idx)], expectedHost[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
            << label << " mismatch at index=" << formatTensorIndex(shape, idx) << " flat_index=" << idx
            << " expected=" << expectedValue << " actual=" << actualValue;

        if (isclose(actualHost[static_cast<size_t>(idx)], expectedHost[static_cast<size_t>(idx)], 1e-3F, 1e-3F))
        {
            ++closeWithin1e3;
        }
        nanDetected = nanDetected || std::isnan(actualValue);
    }

    float const passRate1e3 = static_cast<float>(closeWithin1e3) / static_cast<float>(totalElements);
    EXPECT_GT(passRate1e3, 0.9F) << label << " pass-rate at 1e-3 = " << passRate1e3;
    EXPECT_FALSE(nanDetected) << label;
}

struct ShapeParam
{
    int32_t batchSize;
    int32_t seqLen;
    int32_t numQHeads;
    int32_t numKVHeads; // == numQHeads for MHA; < numQHeads for GQA/MQA (must divide numQHeads)
    bool useNormalInit;
    char const* name;
    float attentionScale{1.0F / std::sqrt(512.0F)};
};

void runAccuracyCase(ShapeParam const& p, cudaStream_t stream)
{
    int32_t constexpr kHeadDim = 512;
    // Q / O carry the full Q-head count; K / V carry the (possibly smaller) KV-head
    // count.  The FP32 BSHD reference maps q_head -> kv_head as q_head * Hkv / Hq,
    // which matches the kernel's q_head // (Hq / Hkv) when Hq % Hkv == 0.
    rt::Coords const qShape{p.batchSize, p.seqLen, p.numQHeads, kHeadDim};
    rt::Coords const kvShape{p.batchSize, p.seqLen, p.numKVHeads, kHeadDim};
    rt::Tensor q(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor k(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor v(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor output(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor outputReference(qShape, rt::DeviceType::kGPU, DataType::kHALF);

    if (p.useNormalInit)
    {
        initializeNormalFp16(q, 11);
        initializeNormalFp16(k, 29);
        initializeNormalFp16(v, 47);
    }
    else
    {
        initializeFp16(q, 11);
        initializeFp16(k, 29);
        initializeFp16(v, 47);
    }

    rt::launchFmhaReferenceBshd(q, k, v, outputReference, true, p.attentionScale, stream);

    // Uniform (dense) cumulative lengths: varlen masking degenerates to the
    // padded extents, preserving the original dense-kernel semantics.
    rt::Tensor cuSeqLens = makeCuSeqLens(std::vector<int32_t>(p.batchSize, p.seqLen));

    CuteDslFFPAParams params;
    params.q = q.rawPointer();
    params.k = k.rawPointer();
    params.v = v.rawPointer();
    params.o = output.rawPointer();
    params.cuSeqLenQ = cuSeqLens.dataPointer<int32_t>();
    params.cuSeqLenK = cuSeqLens.dataPointer<int32_t>();
    params.batchSize = p.batchSize;
    params.seqlenQ = p.seqLen;
    params.seqlenK = p.seqLen;
    params.numQHeads = p.numQHeads;
    params.numKVHeads = p.numKVHeads;
    params.headDim = kHeadDim;
    params.softmaxScale = p.attentionScale;

    ASSERT_EQ(CuteDslFFPARunner::run(params, stream), 0) << p.name;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    expectFp16Close(output, outputReference, p.name);
}

class CuteDslFFPABase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mSmVersion = getSMVersion();
        if (!isFfpaSupportedSm(mSmVersion))
        {
            GTEST_SKIP() << "FFPA d512 causal CuTe DSL kernel is not supported on SM" << mSmVersion;
        }
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&mStream));
        if (!CuteDslFFPARunner::loadKernelModule())
        {
            CUDA_CHECK(cudaStreamDestroy(mStream));
            mStream = nullptr;
            GTEST_SKIP() << "Failed to load FFPA d512 causal CuTe DSL kernel module";
        }
    }

    void TearDown() override
    {
        CuteDslFFPARunner::unloadKernelModule();
        if (mStream != nullptr)
        {
            CUDA_CHECK(cudaStreamDestroy(mStream));
        }
    }

    cudaStream_t mStream{nullptr};
    int32_t mSmVersion{0};
};

class CuteDslFFPAAccuracySweep : public CuteDslFFPABase, public ::testing::WithParamInterface<ShapeParam>
{
};

TEST_P(CuteDslFFPAAccuracySweep, Causal)
{
    auto const& p = GetParam();
    int32_t constexpr kHeadDim = 512;
    if (!CuteDslFFPARunner::canImplement(kHeadDim, mSmVersion, p.numQHeads, p.numKVHeads))
    {
        GTEST_SKIP() << p.name << ": GQA group size " << (p.numQHeads / p.numKVHeads) << " not supported on this build";
    }
    runAccuracyCase(p, mStream);
}

INSTANTIATE_TEST_SUITE_P(FP16Causal, CuteDslFFPAAccuracySweep,
    ::testing::Values(
        // --- MHA (numKVHeads == numQHeads) ---
        ShapeParam{1, 16, 1, 1, /*useNormalInit=*/false, "B1_S16_H1"},
        ShapeParam{2, 24, 2, 2, /*useNormalInit=*/false, "B2_S24_H2"},
        ShapeParam{1, 16, 1, 1, /*useNormalInit=*/false, "identity_scale", 1.0F},
        ShapeParam{1, 16, 1, 1, /*useNormalInit=*/false, "custom_scale", 0.37F},
        ShapeParam{1, 8, 1, 1, /*useNormalInit=*/true, "sub_Bc"},
        ShapeParam{1, 16, 4, 4, /*useNormalInit=*/true, "eq_Bc"},
        ShapeParam{1, 64, 8, 8, /*useNormalInit=*/true, "eq_Br"},
        ShapeParam{1, 128, 8, 8, /*useNormalInit=*/true, "multi_block_M"},
        ShapeParam{1, 130, 4, 4, /*useNormalInit=*/true, "multi_block_M_unaligned"},
        ShapeParam{1, 1024, 8, 8, /*useNormalInit=*/true, "llm_1k"},
        ShapeParam{1, 2048, 8, 8, /*useNormalInit=*/true, "llm_2k"},
        ShapeParam{4, 256, 4, 4, /*useNormalInit=*/true, "batch"},
        // --- GQA group size 4 (the primary target: H_q / H_kv == 4) ---
        ShapeParam{1, 128, 8, 2, /*useNormalInit=*/true, "gqa_g4_H8_KV2"},
        ShapeParam{1, 1024, 8, 2, /*useNormalInit=*/true, "gqa_g4_H8_KV2_1k"},
        ShapeParam{2, 256, 8, 2, /*useNormalInit=*/true, "gqa_g4_H8_KV2_batch"},
        // --- Other GQA / MQA group sizes for coverage ---
        ShapeParam{1, 128, 8, 4, /*useNormalInit=*/true, "gqa_g2_H8_KV4"},
        ShapeParam{1, 128, 8, 1, /*useNormalInit=*/true, "gemma_mqa_identity", 1.0F},
        ShapeParam{1, 128, 8, 1, /*useNormalInit=*/true, "gemma_mqa_custom", 0.37F},
        ShapeParam{1, 256, 4, 1, /*useNormalInit=*/true, "mqa_g4_H4_KV1"},
        ShapeParam{1, 1024, 8, 1, /*useNormalInit=*/true, "mqa_H8_KV1_1k"},
        // Gemma4 Unified 12B global-attention layers use Hq=16, Hkv=1.
        ShapeParam{1, 128, 16, 1, /*useNormalInit=*/true, "gemma4_mqa_g16_H16_KV1"},
        // Br/Bc-unaligned lengths exercising the KV tail block.
        ShapeParam{1, 45, 16, 1, /*useNormalInit=*/true, "gemma4_g16_S45_tail"},
        ShapeParam{1, 282, 16, 1, /*useNormalInit=*/true, "gemma4_g16_S282_tail"},
        ShapeParam{1, 45, 8, 1, /*useNormalInit=*/true, "gqa8_S45_tail_control"}),
    [](::testing::TestParamInfo<ShapeParam> const& info) { return std::string{info.param.name}; });

class CuteDslFFPACausalProperty : public CuteDslFFPABase, public ::testing::WithParamInterface<float>
{
protected:
    // Run the causal kernel for a B=1, H=numHeads prefix [0..seqLen) drawn from
    // the leading rows of a single full-length input buffer. The runner re-derives
    // strides from seqlenK, so reusing the full-S input pointer with a shorter
    // seqlenQ/seqlenK correctly addresses the contiguous leading-row prefix.
    rt::Tensor runPrefix(rt::Tensor const& fullQ, rt::Tensor const& fullK, rt::Tensor const& fullV, int32_t seqLen,
        int32_t numHeads, float attentionScale)
    {
        int32_t constexpr kHeadDim = 512;
        rt::Coords const outShape{1, seqLen, numHeads, kHeadDim};
        rt::Tensor output(outShape, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor cuSeqLens = makeCuSeqLens({seqLen});

        CuteDslFFPAParams params;
        params.q = fullQ.rawPointer();
        params.k = fullK.rawPointer();
        params.v = fullV.rawPointer();
        params.o = output.rawPointer();
        params.cuSeqLenQ = cuSeqLens.dataPointer<int32_t>();
        params.cuSeqLenK = cuSeqLens.dataPointer<int32_t>();
        params.batchSize = 1;
        params.seqlenQ = seqLen;
        params.seqlenK = seqLen;
        params.numQHeads = numHeads;
        params.numKVHeads = numHeads;
        params.headDim = kHeadDim;
        params.softmaxScale = attentionScale;

        EXPECT_EQ(CuteDslFFPARunner::run(params, mStream), 0) << "prefix S=" << seqLen;
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        CUDA_CHECK(cudaGetLastError());
        return output;
    }
};

// Prefix invariance: causal attention's output at rows [0..k] depends only on
// Q[:k+1], K[:k+1], V[:k+1].  Running the kernel at S=16, 64, 128 on the same
// leading-row inputs must yield identical outputs for the rows they share.
// Picking 16/64/128 crosses the Br=64 tile boundary so we exercise both
// single-block and multi-block Q traversal.
TEST_P(CuteDslFFPACausalProperty, PrefixEquivalence)
{
    int32_t constexpr kHeadDim = 512;
    int32_t constexpr kNumHeads = 2;
    int32_t constexpr kFullSeq = 128;

    rt::Coords const fullShape{1, kFullSeq, kNumHeads, kHeadDim};
    rt::Tensor fullQ(fullShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor fullK(fullShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor fullV(fullShape, rt::DeviceType::kGPU, DataType::kHALF);

    initializeNormalFp16(fullQ, 101);
    initializeNormalFp16(fullK, 211);
    initializeNormalFp16(fullV, 307);

    float const attentionScale = GetParam();
    rt::Tensor out128 = runPrefix(fullQ, fullK, fullV, 128, kNumHeads, attentionScale);
    rt::Tensor out64 = runPrefix(fullQ, fullK, fullV, 64, kNumHeads, attentionScale);
    rt::Tensor out16 = runPrefix(fullQ, fullK, fullV, 16, kNumHeads, attentionScale);

    auto const host128 = copyDeviceToHost<__half>(out128);
    auto const host64 = copyDeviceToHost<__half>(out64);
    auto const host16 = copyDeviceToHost<__half>(out16);

    int64_t const rowStride = static_cast<int64_t>(kNumHeads) * kHeadDim;

    // Rows [0..15] must match across all three (sub-Bc prefix; single Q block in all).
    for (int64_t row = 0; row < 16; ++row)
    {
        for (int64_t col = 0; col < rowStride; ++col)
        {
            int64_t const idx = row * rowStride + col;
            float const v128 = __half2float(host128[static_cast<size_t>(idx)]);
            float const v64 = __half2float(host64[static_cast<size_t>(idx)]);
            float const v16 = __half2float(host16[static_cast<size_t>(idx)]);
            ASSERT_TRUE(isclose(host128[static_cast<size_t>(idx)], host64[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                << "rows[0..15] S=128 vs S=64 diverge at flat_index=" << idx << " row=" << row << " S=128=" << v128
                << " S=64=" << v64;
            ASSERT_TRUE(isclose(host64[static_cast<size_t>(idx)], host16[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                << "rows[0..15] S=64 vs S=16 diverge at flat_index=" << idx << " row=" << row << " S=64=" << v64
                << " S=16=" << v16;
        }
    }

    // Rows [0..63] must match between S=64 and S=128 (crosses the Br=64 boundary).
    for (int64_t row = 0; row < 64; ++row)
    {
        for (int64_t col = 0; col < rowStride; ++col)
        {
            int64_t const idx = row * rowStride + col;
            float const v128 = __half2float(host128[static_cast<size_t>(idx)]);
            float const v64 = __half2float(host64[static_cast<size_t>(idx)]);
            ASSERT_TRUE(isclose(host128[static_cast<size_t>(idx)], host64[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                << "rows[0..63] S=128 vs S=64 diverge at flat_index=" << idx << " row=" << row << " S=128=" << v128
                << " S=64=" << v64;
        }
    }
}

class CuteDslFFPAVarlen : public CuteDslFFPABase
{
protected:
    static int32_t constexpr kHeadDim = 512;

    //! Copy batch b's leading `rows` rows out of a padded [B, S, H, D] host image
    //! into a fresh [1, rows, H, D] device tensor.
    static rt::Tensor sliceBatchPrefix(
        std::vector<__half> const& host, rt::Coords const& shape, int32_t b, int32_t rows)
    {
        int64_t const rowElems = shape[2] * shape[3];
        rt::Tensor out(rt::Coords{1, rows, shape[2], shape[3]}, rt::DeviceType::kGPU, DataType::kHALF);
        std::vector<__half> slice(static_cast<size_t>(rows * rowElems));
        int64_t const base = static_cast<int64_t>(b) * shape[1] * rowElems;
        std::copy_n(host.begin() + base, slice.size(), slice.begin());
        copyHostToDevice(out, slice);
        return out;
    }
};

// Ragged BS=3 right-padded batch with near-fp16-max garbage in the padding
// positions — the Gemma4 BS>1 MMLU prefill shape.  The dense
// FFPA kernel attends the poisoned padding and corrupts the batch; with
// per-batch cu_seqlens the valid rows must match the per-batch FP32 reference,
// padding rows must stay bounded, and nothing may be NaN/Inf.
TEST_F(CuteDslFFPAVarlen, RaggedBatchPoisonedPadding)
{
    int32_t constexpr kBatch = 3;
    std::vector<int32_t> const lens{257, 130, 64};
    int32_t const seqLen = *std::max_element(lens.begin(), lens.end());
    int32_t constexpr kNumQHeads = 8; // Gemma4-E2B: Hq=8, Hkv=1 (MQA)
    int32_t constexpr kNumKVHeads = 1;

    rt::Coords const qShape{kBatch, seqLen, kNumQHeads, kHeadDim};
    rt::Coords const kvShape{kBatch, seqLen, kNumKVHeads, kHeadDim};
    rt::Tensor q(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor k(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor v(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor output(qShape, rt::DeviceType::kGPU, DataType::kHALF);

    initializeNormalFp16(q, 101);
    initializeNormalFp16(k, 211);
    initializeNormalFp16(v, 307);
    poisonPaddingRows(q, lens, 41);
    poisonPaddingRows(k, lens, 43);
    poisonPaddingRows(v, lens, 47);

    rt::Tensor cuSeqLens = makeCuSeqLens(lens);

    CuteDslFFPAParams params;
    params.q = q.rawPointer();
    params.k = k.rawPointer();
    params.v = v.rawPointer();
    params.o = output.rawPointer();
    params.cuSeqLenQ = cuSeqLens.dataPointer<int32_t>();
    params.cuSeqLenK = cuSeqLens.dataPointer<int32_t>();
    params.batchSize = kBatch;
    params.seqlenQ = seqLen;
    params.seqlenK = seqLen;
    params.numQHeads = kNumQHeads;
    params.numKVHeads = kNumKVHeads;
    params.headDim = kHeadDim;
    params.softmaxScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

    ASSERT_EQ(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    auto const qHost = copyDeviceToHost<__half>(q);
    auto const kHost = copyDeviceToHost<__half>(k);
    auto const vHost = copyDeviceToHost<__half>(v);
    auto const oHost = copyDeviceToHost<__half>(output);

    int64_t const qRowElems = static_cast<int64_t>(kNumQHeads) * kHeadDim;
    for (int32_t b = 0; b < kBatch; ++b)
    {
        // Per-batch FP32 reference over the valid prefix only.
        rt::Tensor qRef = sliceBatchPrefix(qHost, qShape, b, lens[static_cast<size_t>(b)]);
        rt::Tensor kRef = sliceBatchPrefix(kHost, kvShape, b, lens[static_cast<size_t>(b)]);
        rt::Tensor vRef = sliceBatchPrefix(vHost, kvShape, b, lens[static_cast<size_t>(b)]);
        rt::Tensor oRef(
            rt::Coords{1, lens[static_cast<size_t>(b)], kNumQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::launchFmhaReferenceBshd(qRef, kRef, vRef, oRef, true, params.softmaxScale, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        auto const oRefHost = copyDeviceToHost<__half>(oRef);

        for (int64_t s = 0; s < lens[static_cast<size_t>(b)]; ++s)
        {
            for (int64_t e = 0; e < qRowElems; ++e)
            {
                int64_t const idx = (static_cast<int64_t>(b) * seqLen + s) * qRowElems + e;
                float const actual = __half2float(oHost[static_cast<size_t>(idx)]);
                float const expected = __half2float(oRefHost[static_cast<size_t>(s * qRowElems + e)]);
                ASSERT_FALSE(std::isnan(actual)) << "NaN at b=" << b << " s=" << s << " e=" << e;
                ASSERT_TRUE(isclose(
                    oHost[static_cast<size_t>(idx)], oRefHost[static_cast<size_t>(s * qRowElems + e)], 1e-2F, 1e-2F))
                    << "valid-row mismatch at b=" << b << " s=" << s << " e=" << e << " expected=" << expected
                    << " actual=" << actual;
            }
        }

        // Padding rows: bounded (never amplified garbage / NaN / Inf).
        for (int64_t s = lens[static_cast<size_t>(b)]; s < seqLen; ++s)
        {
            for (int64_t e = 0; e < qRowElems; ++e)
            {
                int64_t const idx = (static_cast<int64_t>(b) * seqLen + s) * qRowElems + e;
                float const actual = __half2float(oHost[static_cast<size_t>(idx)]);
                ASSERT_FALSE(std::isnan(actual)) << "padding NaN at b=" << b << " s=" << s;
                ASSERT_FALSE(std::isinf(actual)) << "padding Inf at b=" << b << " s=" << s;
                ASSERT_LT(std::abs(actual), 1.0F)
                    << "padding row not bounded at b=" << b << " s=" << s << " value=" << actual;
            }
        }
    }
}

// NaN-poisoned padding: padding K/V rows can legitimately contain NaN/Inf at
// inference time (fp16 overflow of garbage padding
// embeddings in the FFN turns pad rows NaN from layer 1 on).  Score masking
// alone is not NaN-safe — BMM2 computes P(0) x V(NaN) = NaN and poisons the
// whole boundary q-tile — so the boundary-tile K/V loads zero-fill logical
// positions >= seqlen_k_b.  Valid rows must stay exact and finite; padding
// rows are don't-care (their own Q is NaN).
TEST_F(CuteDslFFPAVarlen, NaNPoisonedPaddingDoesNotLeak)
{
    int32_t constexpr kBatch = 2;
    std::vector<int32_t> const lens{200, 77};
    int32_t const seqLen = *std::max_element(lens.begin(), lens.end());
    int32_t constexpr kNumQHeads = 8;
    int32_t constexpr kNumKVHeads = 1;

    rt::Coords const qShape{kBatch, seqLen, kNumQHeads, kHeadDim};
    rt::Coords const kvShape{kBatch, seqLen, kNumKVHeads, kHeadDim};
    rt::Tensor q(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor k(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor v(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor output(qShape, rt::DeviceType::kGPU, DataType::kHALF);

    initializeNormalFp16(q, 811);
    initializeNormalFp16(k, 823);
    initializeNormalFp16(v, 827);
    poisonPaddingRows(q, lens, 0, /*nanPoison=*/true);
    poisonPaddingRows(k, lens, 0, /*nanPoison=*/true);
    poisonPaddingRows(v, lens, 0, /*nanPoison=*/true);

    rt::Tensor cuSeqLens = makeCuSeqLens(lens);

    CuteDslFFPAParams params;
    params.q = q.rawPointer();
    params.k = k.rawPointer();
    params.v = v.rawPointer();
    params.o = output.rawPointer();
    params.cuSeqLenQ = cuSeqLens.dataPointer<int32_t>();
    params.cuSeqLenK = cuSeqLens.dataPointer<int32_t>();
    params.batchSize = kBatch;
    params.seqlenQ = seqLen;
    params.seqlenK = seqLen;
    params.numQHeads = kNumQHeads;
    params.numKVHeads = kNumKVHeads;
    params.headDim = kHeadDim;
    params.softmaxScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

    ASSERT_EQ(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    auto const qHost = copyDeviceToHost<__half>(q);
    auto const kHost = copyDeviceToHost<__half>(k);
    auto const vHost = copyDeviceToHost<__half>(v);
    auto const oHost = copyDeviceToHost<__half>(output);
    int64_t const qRowElems = static_cast<int64_t>(kNumQHeads) * kHeadDim;

    for (int32_t b = 0; b < kBatch; ++b)
    {
        rt::Tensor qRef = sliceBatchPrefix(qHost, qShape, b, lens[static_cast<size_t>(b)]);
        rt::Tensor kRef = sliceBatchPrefix(kHost, kvShape, b, lens[static_cast<size_t>(b)]);
        rt::Tensor vRef = sliceBatchPrefix(vHost, kvShape, b, lens[static_cast<size_t>(b)]);
        rt::Tensor oRef(
            rt::Coords{1, lens[static_cast<size_t>(b)], kNumQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::launchFmhaReferenceBshd(qRef, kRef, vRef, oRef, true, params.softmaxScale, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        auto const oRefHost = copyDeviceToHost<__half>(oRef);

        for (int64_t s = 0; s < lens[static_cast<size_t>(b)]; ++s)
        {
            for (int64_t e = 0; e < qRowElems; ++e)
            {
                int64_t const idx = (static_cast<int64_t>(b) * seqLen + s) * qRowElems + e;
                float const actual = __half2float(oHost[static_cast<size_t>(idx)]);
                ASSERT_FALSE(std::isnan(actual)) << "NaN leaked into valid row b=" << b << " s=" << s << " e=" << e;
                ASSERT_FALSE(std::isinf(actual)) << "Inf leaked into valid row b=" << b << " s=" << s << " e=" << e;
                ASSERT_TRUE(isclose(
                    oHost[static_cast<size_t>(idx)], oRefHost[static_cast<size_t>(s * qRowElems + e)], 1e-2F, 1e-2F))
                    << "valid-row mismatch at b=" << b << " s=" << s << " e=" << e;
            }
        }
    }
}

// Chunked prefill: per-batch Q chunk shorter than the KV context
// (seqlen_q_b < seqlen_k_b), bottom-right causal offset = KV prefix length,
// including a Bc-misaligned offset (63).  Rows of the chunk must match the
// corresponding suffix rows of a full-sequence FP32 reference.
TEST_F(CuteDslFFPAVarlen, ChunkedPrefill)
{
    int32_t constexpr kBatch = 2;
    std::vector<int32_t> const chunkLens{64, 37};
    std::vector<int32_t> const kvLens{192, 100}; // offsets 128 and 63
    int32_t const seqLenQ = *std::max_element(chunkLens.begin(), chunkLens.end());
    int32_t const seqLenK = *std::max_element(kvLens.begin(), kvLens.end());
    int32_t constexpr kNumQHeads = 8;
    int32_t constexpr kNumKVHeads = 1;

    // Full-length Q per batch: the chunk rows are the last chunkLens[b] rows of
    // the kvLens[b]-long query history.
    rt::Coords const fullQShape{kBatch, seqLenK, kNumQHeads, kHeadDim};
    rt::Coords const kvShape{kBatch, seqLenK, kNumKVHeads, kHeadDim};
    rt::Tensor fullQ(fullQShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor k(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor v(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
    initializeNormalFp16(fullQ, 501);
    initializeNormalFp16(k, 601);
    initializeNormalFp16(v, 701);
    poisonPaddingRows(k, kvLens, 53);
    poisonPaddingRows(v, kvLens, 59);

    auto const fullQHost = copyDeviceToHost<__half>(fullQ);
    auto const kHost = copyDeviceToHost<__half>(k);
    auto const vHost = copyDeviceToHost<__half>(v);

    // Chunk Q input [B, seqLenQ, Hq, D]: rows [kvLen-chunk, kvLen) of fullQ,
    // right-padded with poison.
    rt::Coords const qShape{kBatch, seqLenQ, kNumQHeads, kHeadDim};
    rt::Tensor q(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    int64_t const qRowElems = static_cast<int64_t>(kNumQHeads) * kHeadDim;
    {
        std::vector<__half> qHost(static_cast<size_t>(qShape.volume()));
        for (int32_t b = 0; b < kBatch; ++b)
        {
            int64_t const srcBase = (static_cast<int64_t>(b) * seqLenK + kvLens[static_cast<size_t>(b)]
                                        - chunkLens[static_cast<size_t>(b)])
                * qRowElems;
            int64_t const dstBase = static_cast<int64_t>(b) * seqLenQ * qRowElems;
            std::copy_n(fullQHost.begin() + srcBase, static_cast<size_t>(chunkLens[static_cast<size_t>(b)] * qRowElems),
                qHost.begin() + dstBase);
        }
        copyHostToDevice(q, qHost);
        poisonPaddingRows(q, chunkLens, 61);
    }

    rt::Tensor output(qShape, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor cuQSeqLens = makeCuSeqLens(chunkLens);
    rt::Tensor cuKVSeqLens = makeCuSeqLens(kvLens);

    CuteDslFFPAParams params;
    params.q = q.rawPointer();
    params.k = k.rawPointer();
    params.v = v.rawPointer();
    params.o = output.rawPointer();
    params.cuSeqLenQ = cuQSeqLens.dataPointer<int32_t>();
    params.cuSeqLenK = cuKVSeqLens.dataPointer<int32_t>();
    params.batchSize = kBatch;
    params.seqlenQ = seqLenQ;
    params.seqlenK = seqLenK;
    params.numQHeads = kNumQHeads;
    params.numKVHeads = kNumKVHeads;
    params.headDim = kHeadDim;
    params.softmaxScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

    ASSERT_EQ(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    auto const oHost = copyDeviceToHost<__half>(output);

    for (int32_t b = 0; b < kBatch; ++b)
    {
        int32_t const kvLen = kvLens[static_cast<size_t>(b)];
        int32_t const chunk = chunkLens[static_cast<size_t>(b)];
        // Full-sequence reference: top-left causal over the whole kvLen-long
        // sequence; its last `chunk` rows equal bottom-right-offset chunked
        // attention of the chunk against the full KV context.
        rt::Tensor qRef = sliceBatchPrefix(fullQHost, fullQShape, b, kvLen);
        rt::Tensor kRef = sliceBatchPrefix(kHost, kvShape, b, kvLen);
        rt::Tensor vRef = sliceBatchPrefix(vHost, kvShape, b, kvLen);
        rt::Tensor oRef(rt::Coords{1, kvLen, kNumQHeads, kHeadDim}, rt::DeviceType::kGPU, DataType::kHALF);
        rt::launchFmhaReferenceBshd(qRef, kRef, vRef, oRef, true, params.softmaxScale, mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        auto const oRefHost = copyDeviceToHost<__half>(oRef);

        for (int64_t s = 0; s < chunk; ++s)
        {
            for (int64_t e = 0; e < qRowElems; ++e)
            {
                int64_t const idx = (static_cast<int64_t>(b) * seqLenQ + s) * qRowElems + e;
                int64_t const refIdx = (static_cast<int64_t>(kvLen - chunk) + s) * qRowElems + e;
                float const actual = __half2float(oHost[static_cast<size_t>(idx)]);
                float const expected = __half2float(oRefHost[static_cast<size_t>(refIdx)]);
                ASSERT_FALSE(std::isnan(actual)) << "NaN at b=" << b << " s=" << s << " e=" << e;
                ASSERT_TRUE(
                    isclose(oHost[static_cast<size_t>(idx)], oRefHost[static_cast<size_t>(refIdx)], 1e-2F, 1e-2F))
                    << "chunk-row mismatch at b=" << b << " s=" << s << " e=" << e << " expected=" << expected
                    << " actual=" << actual;
            }
        }
    }
}
INSTANTIATE_TEST_SUITE_P(AttentionScale, CuteDslFFPACausalProperty, ::testing::Values(1.0F, 0.37F),
    [](::testing::TestParamInfo<float> const& info) { return info.param == 1.0F ? "Identity" : "Custom"; });

//! Expand [B, S] vision-block IDs (-1 = text, >= 0 = block id per contiguous
//! image run) into per-row [blockBegin, blockEnd] interval tensors with the
//! -1/-1 sentinel for text and padding rows — the same expansion the
//! AttentionPlugin performs on the vision_block_ids input.
std::pair<std::vector<int32_t>, std::vector<int32_t>> buildBlockRanges(
    std::vector<int32_t> const& blockIds, int32_t batchSize, int32_t seqLen, std::vector<int32_t> const& lens)
{
    std::vector<int32_t> begin(blockIds.size(), -1);
    std::vector<int32_t> end(blockIds.size(), -1);
    for (int32_t b = 0; b < batchSize; ++b)
    {
        int32_t const len = lens[static_cast<size_t>(b)];
        int32_t s = 0;
        while (s < len)
        {
            int32_t const id = blockIds[static_cast<size_t>(b) * seqLen + s];
            int32_t e = s;
            while (e + 1 < len && blockIds[static_cast<size_t>(b) * seqLen + e + 1] == id)
            {
                ++e;
            }
            if (id >= 0)
            {
                for (int32_t i = s; i <= e; ++i)
                {
                    begin[static_cast<size_t>(b) * seqLen + i] = s;
                    end[static_cast<size_t>(b) * seqLen + i] = e;
                }
            }
            s = e + 1;
        }
    }
    return {begin, end};
}

class CuteDslFFPAVisionBlock : public CuteDslFFPABase
{
protected:
    static int32_t constexpr kHeadDim = 512;

    void SetUp() override
    {
        CuteDslFFPABase::SetUp();
        if (IsSkipped())
        {
            return;
        }
        if (!CuteDslFFPARunner::canImplementVisionBlock(kHeadDim, mSmVersion))
        {
            GTEST_SKIP() << "ffpa_d512_causal_visionblock AOT variant is not compiled into this build";
        }
    }

    //! Independent FP32 host reference for causal + vision-block overlay:
    //! allow(q, k) = (k <= q) OR (blockBegin[q] >= 0 AND blockBegin[q] <= k <=
    //! blockEnd[q]), restricted to the per-batch valid prefix.  Padding rows
    //! are left untouched (the kernel only guarantees boundedness there).
    static std::vector<__half> computeReference(std::vector<__half> const& q, std::vector<__half> const& k,
        std::vector<__half> const& v, std::vector<int32_t> const& blockBegin, std::vector<int32_t> const& blockEnd,
        std::vector<int32_t> const& lens, int32_t batchSize, int32_t seqLen, int32_t numQHeads, int32_t numKVHeads)
    {
        std::vector<__half> out(q.size(), __float2half(0.0F));
        float const scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
        int32_t const groupSize = numQHeads / numKVHeads;
        auto qIdx = [&](int32_t b, int32_t s, int32_t h, int32_t d) {
            return ((static_cast<size_t>(b) * seqLen + s) * numQHeads + h) * kHeadDim + d;
        };
        auto kvIdx = [&](int32_t b, int32_t s, int32_t h, int32_t d) {
            return ((static_cast<size_t>(b) * seqLen + s) * numKVHeads + h) * kHeadDim + d;
        };
        for (int32_t b = 0; b < batchSize; ++b)
        {
            int32_t const len = lens[static_cast<size_t>(b)];
            for (int32_t query = 0; query < len; ++query)
            {
                int32_t const bBegin = blockBegin[static_cast<size_t>(b) * seqLen + query];
                int32_t const bEnd = blockEnd[static_cast<size_t>(b) * seqLen + query];
                for (int32_t qHead = 0; qHead < numQHeads; ++qHead)
                {
                    int32_t const kvHead = qHead / groupSize;
                    std::vector<float> logits(static_cast<size_t>(len), -INFINITY);
                    float maxLogit = -INFINITY;
                    for (int32_t key = 0; key < len; ++key)
                    {
                        bool const causal = key <= query;
                        bool const inBlock = bBegin >= 0 && key >= bBegin && key <= bEnd;
                        if (!causal && !inBlock)
                        {
                            continue;
                        }
                        float dot = 0.0F;
                        for (int32_t d = 0; d < kHeadDim; ++d)
                        {
                            dot += __half2float(q[qIdx(b, query, qHead, d)])
                                * __half2float(k[kvIdx(b, key, kvHead, d)]);
                        }
                        logits[static_cast<size_t>(key)] = dot * scale;
                        maxLogit = std::max(maxLogit, logits[static_cast<size_t>(key)]);
                    }
                    float denominator = 0.0F;
                    for (float& logit : logits)
                    {
                        logit = std::isfinite(logit) ? std::exp(logit - maxLogit) : 0.0F;
                        denominator += logit;
                    }
                    for (int32_t d = 0; d < kHeadDim; ++d)
                    {
                        float value = 0.0F;
                        for (int32_t key = 0; key < len; ++key)
                        {
                            value += logits[static_cast<size_t>(key)] * __half2float(v[kvIdx(b, key, kvHead, d)]);
                        }
                        out[qIdx(b, query, qHead, d)] = __float2half(value / denominator);
                    }
                }
            }
        }
        return out;
    }

    //! Run the overlay kernel for the given vision-block IDs and compare the
    //! valid rows against the FP32 host reference; padding rows must stay
    //! bounded and NaN/Inf-free.
    void runAndCheck(std::vector<int32_t> const& blockIds, std::vector<int32_t> const& lens, int32_t seqLen,
        int32_t numQHeads, int32_t numKVHeads, bool poisonPadding, std::string const& label)
    {
        int32_t const batchSize = static_cast<int32_t>(lens.size());
        rt::Coords const qShape{batchSize, seqLen, numQHeads, kHeadDim};
        rt::Coords const kvShape{batchSize, seqLen, numKVHeads, kHeadDim};
        rt::Tensor q(qShape, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor k(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor v(kvShape, rt::DeviceType::kGPU, DataType::kHALF);
        rt::Tensor output(qShape, rt::DeviceType::kGPU, DataType::kHALF);
        initializeNormalFp16(q, 811);
        initializeNormalFp16(k, 823);
        initializeNormalFp16(v, 827);
        if (poisonPadding)
        {
            poisonPaddingRows(q, lens, 71);
            poisonPaddingRows(k, lens, 73);
            poisonPaddingRows(v, lens, 79);
        }

        auto const [blockBegin, blockEnd] = buildBlockRanges(blockIds, batchSize, seqLen, lens);
        rt::Tensor blockBeginTensor(rt::Coords{batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
        rt::Tensor blockEndTensor(rt::Coords{batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
        copyHostToDevice(blockBeginTensor, blockBegin);
        copyHostToDevice(blockEndTensor, blockEnd);
        rt::Tensor cuSeqLens = makeCuSeqLens(lens);

        CuteDslFFPAParams params;
        params.q = q.rawPointer();
        params.k = k.rawPointer();
        params.v = v.rawPointer();
        params.o = output.rawPointer();
        params.cuSeqLenQ = cuSeqLens.dataPointer<int32_t>();
        params.cuSeqLenK = cuSeqLens.dataPointer<int32_t>();
        params.blockBegin = blockBeginTensor.dataPointer<int32_t>();
        params.blockEnd = blockEndTensor.dataPointer<int32_t>();
        params.batchSize = batchSize;
        params.seqlenQ = seqLen;
        params.seqlenK = seqLen;
        params.numQHeads = numQHeads;
        params.numKVHeads = numKVHeads;
        params.headDim = kHeadDim;
        params.softmaxScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

        ASSERT_EQ(CuteDslFFPARunner::run(params, mStream), 0) << label;
        CUDA_CHECK(cudaStreamSynchronize(mStream));
        CUDA_CHECK(cudaGetLastError());

        auto const qHost = copyDeviceToHost<__half>(q);
        auto const kHost = copyDeviceToHost<__half>(k);
        auto const vHost = copyDeviceToHost<__half>(v);
        auto const oHost = copyDeviceToHost<__half>(output);
        auto const reference = computeReference(
            qHost, kHost, vHost, blockBegin, blockEnd, lens, batchSize, seqLen, numQHeads, numKVHeads);

        int64_t const rowElems = static_cast<int64_t>(numQHeads) * kHeadDim;
        for (int32_t b = 0; b < batchSize; ++b)
        {
            for (int64_t s = 0; s < lens[static_cast<size_t>(b)]; ++s)
            {
                for (int64_t e = 0; e < rowElems; ++e)
                {
                    int64_t const idx = (static_cast<int64_t>(b) * seqLen + s) * rowElems + e;
                    float const actual = __half2float(oHost[static_cast<size_t>(idx)]);
                    float const expected = __half2float(reference[static_cast<size_t>(idx)]);
                    ASSERT_FALSE(std::isnan(actual)) << label << " NaN at b=" << b << " s=" << s << " e=" << e;
                    ASSERT_TRUE(
                        isclose(oHost[static_cast<size_t>(idx)], reference[static_cast<size_t>(idx)], 1e-2F, 1e-2F))
                        << label << " mismatch at b=" << b << " s=" << s << " e=" << e << " expected=" << expected
                        << " actual=" << actual;
                }
            }
            for (int64_t s = lens[static_cast<size_t>(b)]; s < seqLen; ++s)
            {
                for (int64_t e = 0; e < rowElems; ++e)
                {
                    int64_t const idx = (static_cast<int64_t>(b) * seqLen + s) * rowElems + e;
                    float const actual = __half2float(oHost[static_cast<size_t>(idx)]);
                    ASSERT_FALSE(std::isnan(actual)) << label << " padding NaN at b=" << b << " s=" << s;
                    ASSERT_FALSE(std::isinf(actual)) << label << " padding Inf at b=" << b << " s=" << s;
                    ASSERT_LT(std::abs(actual), 1.0F)
                        << label << " padding row not bounded at b=" << b << " s=" << s << " value=" << actual;
                }
            }
        }
    }
};

// One block in the middle of the sequence, crossing both the Br=64 Q-tile
// boundary and several Bc=16 KV tiles.  Rows 70..149 must attend keys up to
// 149 (beyond their causal diagonal); text rows before/after stay causal.
TEST_F(CuteDslFFPAVisionBlock, BlockMidSequence)
{
    int32_t constexpr kSeqLen = 192;
    std::vector<int32_t> blockIds(kSeqLen, -1);
    for (int32_t s = 70; s < 150; ++s)
    {
        blockIds[static_cast<size_t>(s)] = 0;
    }
    runAndCheck(blockIds, {kSeqLen}, kSeqLen, 4, 1, /*poisonPadding=*/false, "block_mid_sequence");
}

// A block that runs to the last token: blockEnd == seqlen - 1 exercises the
// KV-bound clamp and the topmost (physically partial) KV tile.
TEST_F(CuteDslFFPAVisionBlock, BlockAtSequenceEnd)
{
    int32_t constexpr kSeqLen = 160;
    std::vector<int32_t> blockIds(kSeqLen, -1);
    for (int32_t s = 100; s < kSeqLen; ++s)
    {
        blockIds[static_cast<size_t>(s)] = 0;
    }
    runAndCheck(blockIds, {kSeqLen}, kSeqLen, 4, 1, /*poisonPadding=*/false, "block_at_sequence_end");
}

// Two disjoint blocks with the Gemma4 Unified 12B global-layer head shape
// (Hq=16, Hkv=1).  Rows of block 0 must not attend block 1 and vice versa.
TEST_F(CuteDslFFPAVisionBlock, TwoBlocksGemma4HeadShape)
{
    int32_t constexpr kSeqLen = 128;
    std::vector<int32_t> blockIds(kSeqLen, -1);
    for (int32_t s = 8; s < 40; ++s)
    {
        blockIds[static_cast<size_t>(s)] = 0;
    }
    for (int32_t s = 80; s < 112; ++s)
    {
        blockIds[static_cast<size_t>(s)] = 1;
    }
    runAndCheck(blockIds, {kSeqLen}, kSeqLen, 16, 1, /*poisonPadding=*/false, "two_blocks_gemma4");
}

// Ragged BS=2 right-padded batch with poisoned padding and different block
// layouts per batch — the overlay must stay per-batch correct under varlen
// masking (reusing the ragged-padding test structure).
TEST_F(CuteDslFFPAVisionBlock, RaggedBatchWithBlocks)
{
    int32_t constexpr kSeqLen = 192;
    std::vector<int32_t> const lens{192, 130};
    std::vector<int32_t> blockIds(static_cast<size_t>(2) * kSeqLen, -1);
    for (int32_t s = 70; s < 150; ++s)
    {
        blockIds[static_cast<size_t>(s)] = 0; // batch 0
    }
    for (int32_t s = 20; s < 60; ++s)
    {
        blockIds[static_cast<size_t>(kSeqLen) + s] = 0; // batch 1
    }
    for (int32_t s = 100; s < 130; ++s)
    {
        blockIds[static_cast<size_t>(kSeqLen) + s] = 1; // batch 1, ends at len-1
    }
    runAndCheck(blockIds, lens, kSeqLen, 4, 1, /*poisonPadding=*/true, "ragged_batch_with_blocks");
}

// Text-only inputs through the overlay kernel (all -1/-1 sentinel intervals)
// must reproduce the plain causal FP32 reference exactly like the plain
// kernel does — the overlay must be a strict superset feature.
TEST_F(CuteDslFFPAVisionBlock, SentinelDegeneratesToCausal)
{
    int32_t constexpr kSeqLen = 130; // Br/Bc-unaligned on purpose
    std::vector<int32_t> const blockIds(kSeqLen, -1);
    runAndCheck(blockIds, {kSeqLen}, kSeqLen, 4, 2, /*poisonPadding=*/false, "sentinel_degenerates_to_causal");
}

// Gemma4-12B Unified head shape (Hq=16, Hkv=1) at Br/Bc-unaligned lengths.
TEST_F(CuteDslFFPAVisionBlock, SentinelCausalG16UnalignedS45)
{
    int32_t constexpr kSeqLen = 45;
    std::vector<int32_t> const blockIds(kSeqLen, -1);
    runAndCheck(blockIds, {kSeqLen}, kSeqLen, 16, 1, /*poisonPadding=*/false, "sentinel_g16_s45");
}

TEST_F(CuteDslFFPAVisionBlock, SentinelCausalG16UnalignedS283)
{
    int32_t constexpr kSeqLen = 283;
    std::vector<int32_t> const blockIds(kSeqLen, -1);
    runAndCheck(blockIds, {kSeqLen}, kSeqLen, 16, 1, /*poisonPadding=*/false, "sentinel_g16_s283");
}

TEST_F(CuteDslFFPAVisionBlock, ImageSpanG16UnalignedS282)
{
    int32_t constexpr kSeqLen = 282; // mirrors the 12B probe: boi@4, image 5..270, eoi@271
    std::vector<int32_t> blockIds(kSeqLen, -1);
    for (int32_t s = 5; s <= 270; ++s)
    {
        blockIds[static_cast<size_t>(s)] = 0;
    }
    runAndCheck(blockIds, {kSeqLen}, kSeqLen, 16, 1, /*poisonPadding=*/false, "image_span_g16_s282");
}

class CuteDslFFPANegativePath : public CuteDslFFPABase
{
};

// Runtime guard: blockBegin/blockEnd must be both set or both null.
TEST_F(CuteDslFFPANegativePath, RejectsMixedNullBlockRangePointers)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = dummy;
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.cuSeqLenQ = static_cast<int32_t const*>(dummy);
    params.cuSeqLenK = static_cast<int32_t const*>(dummy);
    params.blockBegin = static_cast<int32_t const*>(dummy);
    params.blockEnd = nullptr; // intentionally mixed
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 1;
    params.numKVHeads = 1;
    params.headDim = 512;
    params.softmaxScale = 1.0F / std::sqrt(512.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

// Runtime guard: params.headDim != 512 must be rejected with a non-zero return code,
// even though canImplement() filters this at the API entry.
TEST_F(CuteDslFFPANegativePath, RejectsUnsupportedHeadDim)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = dummy;
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.cuSeqLenQ = static_cast<int32_t const*>(dummy);
    params.cuSeqLenK = static_cast<int32_t const*>(dummy);
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 1;
    params.numKVHeads = 1;
    params.headDim = 128; // not supported — the kernel is D=512 only
    params.softmaxScale = 1.0F / std::sqrt(128.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

// Runtime guard: GQA is supported (num_kv_heads is a runtime kernel argument), but
// the Q heads must partition evenly across the K/V heads.  numQHeads % numKVHeads != 0
// (here 8 % 3) has no valid group size and must be rejected before any kernel launch.
// Note: numQHeads != numKVHeads on its own is *not* an error anymore — see the GQA
// cases in CuteDslFFPAAccuracySweep, which exercise the divisible path end-to-end.
TEST_F(CuteDslFFPANegativePath, RejectsIndivisibleKvHeads)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = dummy;
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.cuSeqLenQ = static_cast<int32_t const*>(dummy);
    params.cuSeqLenK = static_cast<int32_t const*>(dummy);
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 8;
    params.numKVHeads = 3; // 8 % 3 != 0 — no integer GQA group size
    params.headDim = 512;
    params.softmaxScale = 1.0F / std::sqrt(512.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

// Runtime guard: a null Q pointer must be rejected before any kernel launch.
TEST_F(CuteDslFFPANegativePath, RejectsNullPointers)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = nullptr; // intentionally null
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.cuSeqLenQ = static_cast<int32_t const*>(dummy);
    params.cuSeqLenK = static_cast<int32_t const*>(dummy);
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 1;
    params.numKVHeads = 1;
    params.headDim = 512;
    params.softmaxScale = 1.0F / std::sqrt(512.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

// Runtime guard: the varlen kernel requires (batchSize + 1) int32 cu_seqlen
// device tensors; null pointers must be rejected before any kernel launch.
TEST_F(CuteDslFFPANegativePath, RejectsNullCuSeqLens)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = dummy;
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.cuSeqLenQ = nullptr; // intentionally null
    params.cuSeqLenK = nullptr;
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 1;
    params.numKVHeads = 1;
    params.headDim = 512;
    params.softmaxScale = 1.0F / std::sqrt(512.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

TEST(CuteDslFFPARunnerStaticTest, CanImplementSupportedHeadDimAndSMs)
{
    // MHA (default numQHeads=1, numKVHeads=1)
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 80));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 86));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 87));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 89));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 100));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 101));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 110));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 120));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, 121));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(128, 100));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, 75));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, 90));
}

TEST(CuteDslFFPARunnerStaticTest, CanImplementGQAGroupSizes)
{
    int32_t constexpr kSM = 100;

    // MHA always supported
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 8, 8));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 1, 1));

    // GQA4: Hq=8, Hkv=2 (Gemma4 E4B)
#if defined(CUTE_DSL_FFPA_GQA4_ENABLED)
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 8, 2));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 4, 1));
#else
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 8, 2));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 4, 1));
#endif

    // GQA8: Hq=8, Hkv=1 (Gemma4 E2B)
#if defined(CUTE_DSL_FFPA_GQA8_ENABLED)
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 8, 1));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 16, 2));
#else
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 8, 1));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 16, 2));
#endif

    // GQA16: Hq=16, Hkv=1 (Gemma4 Unified 12B global attention)
#if defined(CUTE_DSL_FFPA_GQA16_ENABLED)
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 16, 1));
    EXPECT_TRUE(CuteDslFFPARunner::canImplement(512, kSM, 32, 2));
#else
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 16, 1));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 32, 2));
#endif

    // Unsupported group sizes (2, 3)
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 8, 4));  // group=2
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 12, 4)); // group=3

    // Invalid: indivisible
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 8, 3));
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 7, 2));

    // Invalid: zero/negative KV heads
    EXPECT_FALSE(CuteDslFFPARunner::canImplement(512, kSM, 8, 0));
}

// Runtime guard: unsupported GQA group size (e.g. group=2) must be rejected by run().
TEST_F(CuteDslFFPANegativePath, RejectsUnsupportedGQAGroupSize)
{
    void* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, 16));

    CuteDslFFPAParams params;
    params.q = dummy;
    params.k = dummy;
    params.v = dummy;
    params.o = dummy;
    params.cuSeqLenQ = static_cast<int32_t const*>(dummy);
    params.cuSeqLenK = static_cast<int32_t const*>(dummy);
    params.batchSize = 1;
    params.seqlenQ = 16;
    params.seqlenK = 16;
    params.numQHeads = 8;
    params.numKVHeads = 4; // group size 2 — no compiled variant
    params.headDim = 512;
    params.softmaxScale = 1.0F / std::sqrt(512.0F);

    EXPECT_NE(CuteDslFFPARunner::run(params, mStream), 0);
    CUDA_CHECK(cudaStreamSynchronize(mStream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dummy));
}

} // namespace

#endif // defined(CUTE_DSL_FFPA_ENABLED)
