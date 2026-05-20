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

/// Unit tests for CuTe DSL SSD (Mamba2 chunk scan) prefill kernel.
/// Verifies correctness by comparing against the serial invokeSelectiveStateUpdatePrefill reference.

#ifdef CUTE_DSL_SSD_ENABLED

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/mamba/cuteDslSSDRunner.h"
#include "kernels/mamba/selectiveStateUpdate.h"

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// =============================================================================
// CPU Reference (token-by-token, same as invokeSelectiveStateUpdatePrefill)
// =============================================================================

float softplus(float x)
{
    return std::log(1.f + std::exp(x));
}

float thresholdedSoftplus(float x)
{
    constexpr float threshold = 20.f;
    return (x <= threshold) ? softplus(x) : x;
}

/// CPU reference for SSD prefill (sequential scan, matches selectiveStateUpdatePrefill).
void ssdPrefillReference(int32_t batch, int32_t seqLen, int32_t nheads, int32_t dim, int32_t dstate, int32_t ngroups,
    std::vector<half> const& x,       // [batch, seqLen, nheads, dim]
    std::vector<half> const& dt,      // [batch, seqLen, nheads]
    std::vector<float> const& A,      // [nheads]
    std::vector<half> const& B,       // [batch, seqLen, ngroups, dstate]
    std::vector<half> const& C,       // [batch, seqLen, ngroups, dstate]
    std::vector<float> const& D,      // [nheads]
    std::vector<float> const& dtBias, // [nheads]
    bool dtSoftplus,
    std::vector<float>& stateRef,                        // [batch, nheads, dim, dstate] -- in/out
    std::vector<half>& outputRef,                        // [batch, seqLen, nheads, dim]
    std::vector<int32_t> const* contextLengths = nullptr // [batch] (optional, nullptr = uniform = seqLen)
)
{
    int32_t const headsPerGroup = nheads / ngroups;

    for (int32_t b = 0; b < batch; ++b)
    {
        int32_t const cl = contextLengths ? (*contextLengths)[b] : seqLen;
        for (int32_t t = 0; t < seqLen; ++t)
        {
            // Padding region: state stays unchanged, output writes 0.
            // Mirrors selectiveStateUpdateMultiStepReferenceFp32 in
            // mambaSelectiveStateUpdateTests.cu:191-197.
            if (t >= cl)
            {
                for (int32_t h = 0; h < nheads; ++h)
                    for (int32_t d = 0; d < dim; ++d)
                        outputRef[((b * seqLen + t) * nheads + h) * dim + d] = __float2half(0.f);
                continue;
            }
            for (int32_t h = 0; h < nheads; ++h)
            {
                int32_t const g = h / headsPerGroup;
                float dtVal = __half2float(dt[((b * seqLen + t) * nheads) + h]) + dtBias[h];
                if (dtSoftplus)
                    dtVal = thresholdedSoftplus(dtVal);
                float const aVal = A[h];
                float const dA = std::exp(aVal * dtVal);
                float const dVal = D[h];

                for (int32_t d = 0; d < dim; ++d)
                {
                    float const xVal = __half2float(x[((b * seqLen + t) * nheads + h) * dim + d]);
                    float outVal = dVal * xVal;

                    for (int32_t ds = 0; ds < dstate; ++ds)
                    {
                        float const bVal = __half2float(B[((b * seqLen + t) * ngroups + g) * dstate + ds]);
                        float const cVal = __half2float(C[((b * seqLen + t) * ngroups + g) * dstate + ds]);

                        int64_t const sIdx
                            = static_cast<int64_t>(b) * nheads * dim * dstate + h * dim * dstate + d * dstate + ds;
                        float const newState = dA * stateRef[sIdx] + dtVal * bVal * xVal;
                        stateRef[sIdx] = newState;
                        outVal += newState * cVal;
                    }
                    outputRef[((b * seqLen + t) * nheads + h) * dim + d] = __float2half(outVal);
                }
            }
        }
    }
}

// =============================================================================
// Test fixture
// =============================================================================

struct SsdCuteDslTestConfig
{
    int32_t batch;
    int32_t seqLen;
    int32_t nheads;
    int32_t dim;
    int32_t dstate;
    int32_t ngroups;
    std::vector<int32_t> contextLengths{}; // empty = uniform (cl[b] == seqLen for all)
    bool useNonzeroInitState{false};       // when true, initialize state with deterministic random values
};

class SsdCuteDslTest : public ::testing::TestWithParam<SsdCuteDslTestConfig>
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(CuteDslSSDRunner::loadKernelModules()) << "Failed to load SSD CuTe DSL kernel modules";
    }
};

TEST_P(SsdCuteDslTest, CorrectnessVsSerialReference)
{
    auto const& cfg = GetParam();
    int32_t const batch = cfg.batch;
    int32_t const seqLen = cfg.seqLen;
    int32_t const nheads = cfg.nheads;
    int32_t const dim = cfg.dim;
    int32_t const dstate = cfg.dstate;
    int32_t const ngroups = cfg.ngroups;

    if (!CuteDslSSDRunner::canImplement(dim, dstate, 80))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate;
    }

    // ssd_prefill_d128_n128 cubin fails to load on sm_87 (root cause TBD).
    {
        int sm_major{}, sm_minor{};
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, 0);
        cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, 0);
        if (sm_major == 8 && sm_minor == 7 && dim == 128 && dstate == 128)
        {
            GTEST_SKIP() << "ssd_prefill_d128_n128 cubin load fails on sm_87";
        }
    }

    // Allocate host data
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * seqLen * nheads * dim;
    size_t const dtSize = batch * seqLen * nheads;
    size_t const bSize = batch * seqLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;
    size_t const outSize = xSize;

    std::vector<half> xHost(xSize), dtHost(dtSize), bHost(bSize), cHost(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateHost(stateSize, 0.f);
    std::vector<half> outHost(outSize, __float2half(0.f));

    for (auto& v : xHost)
        v = __float2half(normal(rng));
    for (auto& v : dtHost)
        v = __float2half(uniform(rng));
    for (auto& v : bHost)
        v = __float2half(normal(rng));
    for (auto& v : cHost)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;
    // Initial SSM state: zeros by default, deterministic small values when test exercises has_init_states.
    if (cfg.useNonzeroInitState)
    {
        std::normal_distribution<float> stateNormal(0.f, 0.1f);
        for (auto& v : stateHost)
            v = stateNormal(rng);
    }

    // Optional per-batch context_lengths
    std::vector<int32_t> const* contextLengthsPtr = cfg.contextLengths.empty() ? nullptr : &cfg.contextLengths;

    // CPU reference (refState is the initial state; mutated to final by reference scan).
    std::vector<float> refState = stateHost;
    std::vector<half> refOut(outSize, __float2half(0.f));
    ssdPrefillReference(batch, seqLen, nheads, dim, dstate, ngroups, xHost, dtHost, aHost, bHost, cHost, dHost,
        dtBiasHost, true, refState, refOut, contextLengthsPtr);

    // SM80 wrapper now takes fp16 D / dt_bias / state (matches plugin contract).
    std::vector<half> dHostFp16(nheads), dtBiasHostFp16(nheads), stateHostFp16(stateSize);
    for (size_t i = 0; i < nheads; ++i)
        dHostFp16[i] = __float2half(dHost[i]);
    for (size_t i = 0; i < nheads; ++i)
        dtBiasHostFp16[i] = __float2half(dtBiasHost[i]);
    for (size_t i = 0; i < stateSize; ++i)
        stateHostFp16[i] = __float2half(stateHost[i]);

    // GPU: allocate and copy
    void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutput;
    CUDA_CHECK(cudaMalloc(&dX, xSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDt, dtSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOutput, outSize * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dX, xHost.data(), xSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDt, dtHost.data(), dtSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, dHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dState, stateHostFp16.data(), stateSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOutput, 0, outSize * sizeof(half)));

    // Allocate workspace for chunk scan intermediates
    size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
    void* dWorkspace = nullptr;
    if (wsSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&dWorkspace, wsSize));
        CUDA_CHECK(cudaMemset(dWorkspace, 0, wsSize));
    }

    // Run CuteDSL kernel
    SSDParams params{};
    params.x = dX;
    params.dt = dDt;
    params.A = dA;
    params.B = dB;
    params.C = dC;
    params.D = dD;
    params.dt_bias = dDtBias;
    params.z = nullptr;
    params.state = dState;
    params.output = dOutput;
    params.workspace = dWorkspace;
    params.batch = batch;
    params.seq_len = seqLen;
    params.nheads = nheads;
    params.dim = dim;
    params.dstate = dstate;
    params.ngroups = ngroups;
    params.smVersion = 80;
    params.dt_softplus = true;
    params.has_D = true;
    params.has_z = false;
    // Tests that exercise non-zero initial states need the runner to dispatch
    // to the Blackwell variant that reads init state at chunk 0.
    params.has_init_states = cfg.useNonzeroInitState;

    // Optional context_lengths device tensor
    void* dContextLengths = nullptr;
    if (contextLengthsPtr != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dContextLengths, batch * sizeof(int32_t)));
        CUDA_CHECK(
            cudaMemcpy(dContextLengths, contextLengthsPtr->data(), batch * sizeof(int32_t), cudaMemcpyHostToDevice));
        params.context_lengths = dContextLengths;
    }

    CuteDslSSDRunner runner;
    ASSERT_EQ(runner.run(params, nullptr), 0) << "CuteDslSSDRunner::run failed";
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and compare
    std::vector<half> gpuOut(outSize);
    CUDA_CHECK(cudaMemcpy(gpuOut.data(), dOutput, outSize * sizeof(half), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < outSize; ++i)
    {
        float const a = __half2float(gpuOut[i]);
        float const b = __half2float(refOut[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "Relative error " << relErr << " exceeds threshold. "
                             << "maxDiff=" << maxDiff << " refMax=" << refMax;

    // Cleanup
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dDt));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dDtBias));
    CUDA_CHECK(cudaFree(dState));
    CUDA_CHECK(cudaFree(dOutput));
    if (dWorkspace)
    {
        CUDA_CHECK(cudaFree(dWorkspace));
    }
    if (dContextLengths)
    {
        CUDA_CHECK(cudaFree(dContextLengths));
    }
}

// SM80 test configurations — all D×N combos: {128,64} × {128,64}
INSTANTIATE_TEST_SUITE_P(SsdCuteDslSM80, SsdCuteDslTest,
    ::testing::Values(
        // batch, seqLen, nheads, dim, dstate, ngroups
        // D=128, N=128
        SsdCuteDslTestConfig{1, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 1024, 8, 128, 128, 1},
        SsdCuteDslTestConfig{4, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{4, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 256, 64, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 64, 128, 128, 8},
        // D=64, N=128
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 128, 1},
        // D=128, N=64
        SsdCuteDslTestConfig{1, 128, 8, 128, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 128, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 128, 64, 1},
        // D=64, N=64
        SsdCuteDslTestConfig{1, 128, 8, 64, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 64, 1},
        // Varlen: explicit context_lengths
        SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1, {256}},               // batch=1 cl==seqLen sanity
        SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1, {200}},                // batch=1 partial
        SsdCuteDslTestConfig{4, 256, 8, 64, 128, 1, {100, 256, 200, 256}}, // batch>1 mixed
        SsdCuteDslTestConfig{4, 384, 8, 64, 128, 1, {100, 200, 250, 384}}, // mixed + partial last chunk
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {32, 96}},              // small mixed
        SsdCuteDslTestConfig{8, 128, 8, 128, 128, 1, {1, 64, 128, 128, 100, 128, 50, 128}},
        // Nonzero initial state: exercises has_init_states=True kernel path on SM80. Reference
        // scan is seeded with the same state, so output must match. Validates chunked prefill /
        // continuous batching on Ampere fallback.
        SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1, {}, true},                  // D=128 N=128, init nonzero
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1, {}, true},                   // multi-chunk, init nonzero
        SsdCuteDslTestConfig{4, 256, 8, 64, 128, 1, {100, 256, 200, 256}, true}, // mixed varlen + init
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {32, 96}, true}),             // small mixed + init
    [](testing::TestParamInfo<SsdCuteDslTestConfig> const& info) {
        auto const& c = info.param;
        std::string name = "b" + std::to_string(c.batch) + "_s" + std::to_string(c.seqLen) + "_h"
            + std::to_string(c.nheads) + "_d" + std::to_string(c.dim) + "_ds" + std::to_string(c.dstate) + "_g"
            + std::to_string(c.ngroups);
        if (!c.contextLengths.empty())
        {
            name += "_vl";
            for (int32_t cl : c.contextLengths)
                name += std::to_string(cl) + "x";
            name.pop_back();
        }
        if (c.useNonzeroInitState)
            name += "_initstate";
        return name;
    });

#ifdef CUTE_DSL_SSD_BLACKWELL_ENABLED

// =============================================================================
// Blackwell test fixture (SM100+, dim=64, dstate=128)
// =============================================================================
// The Blackwell kernel is a single persistent kernel that takes pre-computed
// cumsum_delta and dt_processed as inputs (unlike the SM80 kernel which does
// cumsum internally). This test verifies end-to-end correctness via
// CuteDslSSDRunner::run() with smVersion=100.

class SsdCuteDslBlackwellTest : public ::testing::TestWithParam<SsdCuteDslTestConfig>
{
protected:
    void SetUp() override
    {
        // Check that GPU actually supports SM100+
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 10)
        {
            GTEST_SKIP() << "Blackwell tests require SM100+ GPU (got SM" << prop.major << prop.minor << ")";
        }
        ASSERT_TRUE(CuteDslSSDRunner::loadKernelModules()) << "Failed to load SSD CuTe DSL kernel modules";
    }
};

TEST_P(SsdCuteDslBlackwellTest, CorrectnessVsSerialReference)
{
    auto const& cfg = GetParam();
    int32_t const batch = cfg.batch;
    int32_t const seqLen = cfg.seqLen;
    int32_t const nheads = cfg.nheads;
    int32_t const dim = cfg.dim;
    int32_t const dstate = cfg.dstate;
    int32_t const ngroups = cfg.ngroups;

    if (!CuteDslSSDRunner::canImplement(dim, dstate, 100))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate << " for SM100";
    }

    // Both Blackwell native (dim=64) and SM80 fallback (dim=128) wrappers take fp16
    // D/state/dt_bias (matches plugin kIN_D_IDX / kIN_DT_BIAS_IDX / kIN_STATE_IDX = kHALF).
    // Allocate host data
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * seqLen * nheads * dim;
    size_t const dtSize = batch * seqLen * nheads;
    size_t const bSize = batch * seqLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;
    size_t const outSize = xSize;

    std::vector<half> xHost(xSize), dtHost(dtSize), bHost(bSize), cHost(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateHost(stateSize, 0.f);
    std::vector<half> outHost(outSize, __float2half(0.f));

    for (auto& v : xHost)
        v = __float2half(normal(rng));
    for (auto& v : dtHost)
        v = __float2half(uniform(rng));
    for (auto& v : bHost)
        v = __float2half(normal(rng));
    for (auto& v : cHost)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;
    // Initial SSM state: zeros by default, deterministic small values when test exercises has_init_states.
    if (cfg.useNonzeroInitState)
    {
        std::normal_distribution<float> stateNormal(0.f, 0.1f);
        for (auto& v : stateHost)
            v = stateNormal(rng);
    }

    // Optional per-batch context_lengths
    std::vector<int32_t> const* contextLengthsPtr = cfg.contextLengths.empty() ? nullptr : &cfg.contextLengths;

    // CPU reference (refState is the initial state; mutated to final by reference scan).
    std::vector<float> refState = stateHost;
    std::vector<half> refOut(outSize, __float2half(0.f));
    ssdPrefillReference(batch, seqLen, nheads, dim, dstate, ngroups, xHost, dtHost, aHost, bHost, cHost, dHost,
        dtBiasHost, true, refState, refOut, contextLengthsPtr);

    std::vector<half> dHostFp16(nheads), dtBiasHostFp16(nheads), stateHostFp16(stateSize);
    for (size_t i = 0; i < nheads; ++i)
        dHostFp16[i] = __float2half(dHost[i]);
    for (size_t i = 0; i < nheads; ++i)
        dtBiasHostFp16[i] = __float2half(dtBiasHost[i]);
    for (size_t i = 0; i < stateSize; ++i)
        stateHostFp16[i] = __float2half(stateHost[i]);

    // GPU: allocate and copy
    void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutput;
    CUDA_CHECK(cudaMalloc(&dX, xSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDt, dtSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, bSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOutput, outSize * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(dX, xHost.data(), xSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDt, dtHost.data(), dtSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, cHost.data(), bSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dD, dHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dState, stateHostFp16.data(), stateSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOutput, 0, outSize * sizeof(half)));

    size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
    void* dWorkspace = nullptr;
    if (wsSize > 0)
    {
        CUDA_CHECK(cudaMalloc(&dWorkspace, wsSize));
        CUDA_CHECK(cudaMemset(dWorkspace, 0, wsSize));
    }

    SSDParams params{};
    params.x = dX;
    params.dt = dDt;
    params.A = dA;
    params.B = dB;
    params.C = dC;
    params.D = dD;
    params.dt_bias = dDtBias;
    params.z = nullptr;
    params.state = dState;
    params.output = dOutput;
    params.workspace = dWorkspace;
    params.batch = batch;
    params.seq_len = seqLen;
    params.nheads = nheads;
    params.dim = dim;
    params.dstate = dstate;
    params.ngroups = ngroups;
    params.smVersion = 100; // Force Blackwell path
    params.dt_softplus = true;
    params.has_D = true;
    params.has_z = false;
    // Tests with non-zero initial state dispatch to the init-states variant.
    params.has_init_states = cfg.useNonzeroInitState;

    // Optional context_lengths device tensor
    void* dContextLengths = nullptr;
    if (contextLengthsPtr != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dContextLengths, batch * sizeof(int32_t)));
        CUDA_CHECK(
            cudaMemcpy(dContextLengths, contextLengthsPtr->data(), batch * sizeof(int32_t), cudaMemcpyHostToDevice));
        params.context_lengths = dContextLengths;
    }

    CuteDslSSDRunner runner;
    ASSERT_EQ(runner.run(params, nullptr), 0) << "CuteDslSSDRunner::run (Blackwell) failed";
    CUDA_CHECK(cudaDeviceSynchronize());

    // Kernel writes y directly to params.output [B, S, H, D] (no transpose adapter).
    std::vector<half> gpuOut(outSize);
    CUDA_CHECK(cudaMemcpy(gpuOut.data(), dOutput, outSize * sizeof(half), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f;
    float refMax = 0.f;
    for (size_t i = 0; i < outSize; ++i)
    {
        float const a = __half2float(gpuOut[i]);
        float const b = __half2float(refOut[i]);
        maxDiff = std::max(maxDiff, std::abs(a - b));
        refMax = std::max(refMax, std::abs(b));
    }
    float const relErr = maxDiff / (refMax + 1e-8f);

    EXPECT_LT(relErr, 0.05f) << "Relative error " << relErr << " exceeds threshold. "
                             << "maxDiff=" << maxDiff << " refMax=" << refMax;

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dDt));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dDtBias));
    CUDA_CHECK(cudaFree(dState));
    CUDA_CHECK(cudaFree(dOutput));
    if (dWorkspace)
    {
        CUDA_CHECK(cudaFree(dWorkspace));
    }
    if (dContextLengths)
    {
        CUDA_CHECK(cudaFree(dContextLengths));
    }
}

// =============================================================================
// Chunked prefill simulation -- exercises has_init_states correctness end-to-end.
// Splits a single seq of length 2*chunkLen into two consecutive runner calls; the
// second call's initial state is the first call's final output state. The combined
// output must match a single-shot run of the full sequence.
// =============================================================================
TEST(SsdCuteDslBlackwellChunkedPrefill, StateCarriesAcrossCalls)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 10)
    {
        GTEST_SKIP() << "Blackwell tests require SM100+ GPU";
    }
    ASSERT_TRUE(CuteDslSSDRunner::loadKernelModules());

    int32_t const batch = 1;
    int32_t const chunkLen = 256;
    int32_t const totalLen = chunkLen * 2; // 512 -- exercises multi-chunk per call
    int32_t const nheads = 8, dim = 64, dstate = 128, ngroups = 1;
    if (!CuteDslSSDRunner::canImplement(dim, dstate, 100))
    {
        GTEST_SKIP() << "CuteDslSSDRunner cannot implement dim=" << dim << " dstate=" << dstate;
    }

    std::mt19937 rng(123);
    std::normal_distribution<float> normal(0.f, 0.5f);
    std::uniform_real_distribution<float> uniform(0.1f, 0.6f);

    size_t const xSize = batch * totalLen * nheads * dim;
    size_t const dtSize = batch * totalLen * nheads;
    size_t const bSize = batch * totalLen * ngroups * dstate;
    size_t const stateSize = batch * nheads * dim * dstate;

    std::vector<half> xFull(xSize), dtFull(dtSize), bFull(bSize), cFull(bSize);
    std::vector<float> aHost(nheads), dHost(nheads), dtBiasHost(nheads);
    std::vector<float> stateZero(stateSize, 0.f);

    for (auto& v : xFull)
        v = __float2half(normal(rng));
    for (auto& v : dtFull)
        v = __float2half(uniform(rng));
    for (auto& v : bFull)
        v = __float2half(normal(rng));
    for (auto& v : cFull)
        v = __float2half(normal(rng));
    for (auto& v : aHost)
        v = -(uniform(rng) + 0.5f);
    for (auto& v : dHost)
        v = normal(rng) * 0.1f;
    for (auto& v : dtBiasHost)
        v = normal(rng) * 0.1f;

    // Blackwell wrapper takes fp16 D / dt_bias / state (matches plugin contract).
    std::vector<half> dHostFp16(nheads), dtBiasHostFp16(nheads);
    for (size_t i = 0; i < nheads; ++i)
        dHostFp16[i] = __float2half(dHost[i]);
    for (size_t i = 0; i < nheads; ++i)
        dtBiasHostFp16[i] = __float2half(dtBiasHost[i]);

    auto runOnce
        = [&](int32_t seqLen, std::vector<half> const& x, std::vector<half> const& dt, std::vector<half> const& Bv,
              std::vector<half> const& Cv, std::vector<float>& stateInOut, std::vector<half>& outBuf) {
              size_t const n_x = batch * seqLen * nheads * dim;
              size_t const n_dt = batch * seqLen * nheads;
              size_t const n_bc = batch * seqLen * ngroups * dstate;

              // Convert state float->half for input; convert back to float on output.
              std::vector<half> stateFp16In(stateSize);
              for (size_t i = 0; i < stateSize; ++i)
                  stateFp16In[i] = __float2half(stateInOut[i]);

              void *dX, *dDt, *dA, *dB, *dC, *dD, *dDtBias, *dState, *dOutBuf;
              CUDA_CHECK(cudaMalloc(&dX, n_x * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dDt, n_dt * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dA, nheads * sizeof(float)));
              CUDA_CHECK(cudaMalloc(&dB, n_bc * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dC, n_bc * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dD, nheads * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dDtBias, nheads * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dState, stateSize * sizeof(half)));
              CUDA_CHECK(cudaMalloc(&dOutBuf, n_x * sizeof(half)));
              CUDA_CHECK(cudaMemcpy(dX, x.data(), n_x * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dDt, dt.data(), n_dt * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dA, aHost.data(), nheads * sizeof(float), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dB, Bv.data(), n_bc * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dC, Cv.data(), n_bc * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dD, dHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dDtBias, dtBiasHostFp16.data(), nheads * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemcpy(dState, stateFp16In.data(), stateSize * sizeof(half), cudaMemcpyHostToDevice));
              CUDA_CHECK(cudaMemset(dOutBuf, 0, n_x * sizeof(half)));

              size_t const wsSize = CuteDslSSDRunner::getWorkspaceSize(batch, seqLen, nheads, dim, dstate, ngroups);
              void* dWs = nullptr;
              if (wsSize > 0)
              {
                  CUDA_CHECK(cudaMalloc(&dWs, wsSize));
                  CUDA_CHECK(cudaMemset(dWs, 0, wsSize));
              }

              SSDParams params{};
              params.x = dX;
              params.dt = dDt;
              params.A = dA;
              params.B = dB;
              params.C = dC;
              params.D = dD;
              params.dt_bias = dDtBias;
              params.z = nullptr;
              params.state = dState;
              params.output = dOutBuf;
              params.workspace = dWs;
              params.batch = batch;
              params.seq_len = seqLen;
              params.nheads = nheads;
              params.dim = dim;
              params.dstate = dstate;
              params.ngroups = ngroups;
              params.smVersion = 100;
              params.dt_softplus = true;
              params.has_D = true;
              params.has_z = false;
              // ChunkedPrefill exercises the carry-over path: each call reads its
              // initial state from the prior call's output state.
              params.has_init_states = true;

              CuteDslSSDRunner runner;
              ASSERT_EQ(runner.run(params, nullptr), 0) << "runner.run failed";
              CUDA_CHECK(cudaDeviceSynchronize());

              outBuf.resize(n_x);
              CUDA_CHECK(cudaMemcpy(outBuf.data(), dOutBuf, n_x * sizeof(half), cudaMemcpyDeviceToHost));
              std::vector<half> stateFp16Out(stateSize);
              CUDA_CHECK(cudaMemcpy(stateFp16Out.data(), dState, stateSize * sizeof(half), cudaMemcpyDeviceToHost));
              for (size_t i = 0; i < stateSize; ++i)
                  stateInOut[i] = __half2float(stateFp16Out[i]);

              CUDA_CHECK(cudaFree(dX));
              CUDA_CHECK(cudaFree(dDt));
              CUDA_CHECK(cudaFree(dA));
              CUDA_CHECK(cudaFree(dB));
              CUDA_CHECK(cudaFree(dC));
              CUDA_CHECK(cudaFree(dD));
              CUDA_CHECK(cudaFree(dDtBias));
              CUDA_CHECK(cudaFree(dState));
              CUDA_CHECK(cudaFree(dOutBuf));
              if (dWs)
                  CUDA_CHECK(cudaFree(dWs));
          };

    // 1. One-shot reference: full seq in single runner call.
    std::vector<float> stateOneShot = stateZero;
    std::vector<half> outOneShot;
    runOnce(totalLen, xFull, dtFull, bFull, cFull, stateOneShot, outOneShot);

    // 2. Two-call: split [0,chunkLen) and [chunkLen,totalLen). Second call uses first call's final state.
    auto sliceTokens = [&](std::vector<half> const& src, int32_t start, int32_t len, int32_t per_token_len) {
        std::vector<half> out(len * per_token_len);
        std::copy(src.begin() + static_cast<size_t>(start) * per_token_len,
            src.begin() + static_cast<size_t>(start + len) * per_token_len, out.begin());
        return out;
    };
    int32_t const x_per_token = nheads * dim;
    int32_t const dt_per_token = nheads;
    int32_t const bc_per_token = ngroups * dstate;

    std::vector<half> x1 = sliceTokens(xFull, 0, chunkLen, x_per_token);
    std::vector<half> dt1 = sliceTokens(dtFull, 0, chunkLen, dt_per_token);
    std::vector<half> b1 = sliceTokens(bFull, 0, chunkLen, bc_per_token);
    std::vector<half> c1 = sliceTokens(cFull, 0, chunkLen, bc_per_token);
    std::vector<half> x2 = sliceTokens(xFull, chunkLen, chunkLen, x_per_token);
    std::vector<half> dt2 = sliceTokens(dtFull, chunkLen, chunkLen, dt_per_token);
    std::vector<half> b2 = sliceTokens(bFull, chunkLen, chunkLen, bc_per_token);
    std::vector<half> c2 = sliceTokens(cFull, chunkLen, chunkLen, bc_per_token);

    std::vector<float> stateChunked = stateZero;
    std::vector<half> out1, out2;
    runOnce(chunkLen, x1, dt1, b1, c1, stateChunked, out1);
    runOnce(chunkLen, x2, dt2, b2, c2, stateChunked, out2);

    // Reconstruct full output: out1 ++ out2.
    std::vector<half> outChunked(xSize);
    std::copy(out1.begin(), out1.end(), outChunked.begin());
    std::copy(out2.begin(), out2.end(), outChunked.begin() + out1.size());

    // Compare one-shot vs chunked. Tolerances allow for fp16 rounding accumulated through the scan.
    float maxDiffY = 0.f, refMaxY = 0.f;
    for (size_t i = 0; i < outChunked.size(); ++i)
    {
        float a = __half2float(outChunked[i]);
        float b = __half2float(outOneShot[i]);
        maxDiffY = std::max(maxDiffY, std::abs(a - b));
        refMaxY = std::max(refMaxY, std::abs(b));
    }
    float relErrY = maxDiffY / (refMaxY + 1e-8f);
    EXPECT_LT(relErrY, 0.05f) << "chunked-prefill output diverges from one-shot. relErr=" << relErrY
                              << " maxDiff=" << maxDiffY;

    float maxDiffS = 0.f, refMaxS = 0.f;
    for (size_t i = 0; i < stateChunked.size(); ++i)
    {
        maxDiffS = std::max(maxDiffS, std::abs(stateChunked[i] - stateOneShot[i]));
        refMaxS = std::max(refMaxS, std::abs(stateOneShot[i]));
    }
    float relErrS = maxDiffS / (refMaxS + 1e-8f);
    EXPECT_LT(relErrS, 0.05f) << "chunked-prefill final state diverges from one-shot. relErr=" << relErrS
                              << " maxDiff=" << maxDiffS;
}

// Blackwell test configurations: D=64 (Blackwell TMA kernel) + D=128/N=64 (SM80 fallback)
INSTANTIATE_TEST_SUITE_P(SsdCuteDslBlackwell, SsdCuteDslBlackwellTest,
    ::testing::Values(
        // batch, seqLen, nheads, dim, dstate, ngroups
        // D=64, N=128: Blackwell persistent kernel (native)
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 1024, 8, 64, 128, 1},
        SsdCuteDslTestConfig{4, 128, 8, 64, 128, 1}, SsdCuteDslTestConfig{1, 256, 64, 64, 128, 1},
        SsdCuteDslTestConfig{1, 256, 64, 64, 128, 8},
        // D=64, N=64: Blackwell persistent kernel (native)
        SsdCuteDslTestConfig{1, 128, 8, 64, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 64, 64, 1},
        SsdCuteDslTestConfig{1, 512, 8, 64, 64, 1}, SsdCuteDslTestConfig{4, 128, 8, 64, 64, 1},
        // D=128, N=128: SM80 cp.async kernel running on Blackwell GPU (fallback)
        SsdCuteDslTestConfig{1, 128, 8, 128, 128, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 128, 1},
        SsdCuteDslTestConfig{1, 1024, 8, 128, 128, 1},
        // D=128, N=64: SM80 fallback
        SsdCuteDslTestConfig{1, 128, 8, 128, 64, 1}, SsdCuteDslTestConfig{1, 256, 8, 128, 64, 1},
        // Varlen: explicit context_lengths (Blackwell varlen path target)
        SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1, {256}},                // batch=1 cl==seqLen sanity
        SsdCuteDslTestConfig{1, 256, 8, 64, 128, 1, {200}},                // batch=1 partial
        SsdCuteDslTestConfig{4, 256, 8, 64, 128, 1, {100, 256, 200, 256}}, // batch>1 mixed
        SsdCuteDslTestConfig{4, 384, 8, 64, 128, 1, {100, 200, 250, 384}}, // mixed + partial last chunk
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {32, 96}},              // small mixed
        SsdCuteDslTestConfig{8, 128, 8, 64, 128, 1, {1, 64, 128, 128, 100, 128, 50, 128}},
        // Padding-heavy varlen ports of mambaSelectiveStateUpdateTests Padding_*
        // (scaled to seqLen >= 128 so CuTeDSL Blackwell path is exercised).
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {40, 128}},            // port of Padding_MixedContextLengths
        SsdCuteDslTestConfig{2, 128, 8, 64, 64, 1, {24, 56}},             // port of Padding_AllShorter
        SsdCuteDslTestConfig{2, 256, 8, 64, 128, 1, {40, 256}},           // partial seq + full seq
        SsdCuteDslTestConfig{4, 512, 8, 64, 128, 1, {77, 200, 333, 512}}, // mixed partials, seqLen=512
        SsdCuteDslTestConfig{2, 1024, 8, 64, 128, 1, {500, 1024}},        // large seqLen mixed
        // Nonzero initial state: exercises has_init_states=True kernel path. Reference scan
        // also seeds from this state, so output must match exactly.
        SsdCuteDslTestConfig{1, 128, 8, 64, 128, 1, {}, true}, // batch=1 init_state != 0, single chunk
        SsdCuteDslTestConfig{1, 512, 8, 64, 128, 1, {}, true}, // batch=1 init_state != 0, multi-chunk
        SsdCuteDslTestConfig{
            4, 256, 8, 64, 128, 1, {100, 256, 200, 256}, true},       // batch>1 mixed cl + nonzero init_state
        SsdCuteDslTestConfig{2, 256, 8, 64, 64, 1, {200, 256}, true}, // partial cl + nonzero init_state (D=64 N=64)
        // Long-seq coverage at production batch shapes (bs <= 8).
        SsdCuteDslTestConfig{4, 2048, 8, 64, 128, 1}, SsdCuteDslTestConfig{4, 4096, 8, 64, 128, 1},
        SsdCuteDslTestConfig{8, 1024, 8, 64, 128, 1}, SsdCuteDslTestConfig{8, 2048, 8, 64, 128, 1},
        SsdCuteDslTestConfig{8, 4096, 8, 64, 128, 1}),
    [](testing::TestParamInfo<SsdCuteDslTestConfig> const& info) {
        auto const& c = info.param;
        std::string name = "b" + std::to_string(c.batch) + "_s" + std::to_string(c.seqLen) + "_h"
            + std::to_string(c.nheads) + "_d" + std::to_string(c.dim) + "_ds" + std::to_string(c.dstate) + "_g"
            + std::to_string(c.ngroups);
        if (!c.contextLengths.empty())
        {
            name += "_vl";
            for (int32_t cl : c.contextLengths)
                name += std::to_string(cl) + "x";
            name.pop_back();
        }
        if (c.useNonzeroInitState)
            name += "_initstate";
        return name;
    });

#endif // CUTE_DSL_SSD_BLACKWELL_ENABLED

} // anonymous namespace

#endif // CUTE_DSL_SSD_ENABLED
