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

#include "common/cudaUtils.h"
#include "kernels/speculative/eagleUtilKernels.h"
#include "testUtils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::kernel;
using namespace nvinfer1;

// ============================================================================
// Test 1: prepareEaglePrefillInputs
// Description: Given sequenceContextLengths as input, compute selectIndices=sequenceLength-1 for each batch
// Now supports per-batch variable sequence lengths
// ============================================================================
TEST(EagleKernels, PrepareEaglePrefillInputs)
{
    cudaStream_t stream = nullptr;

    struct TestCase
    {
        int32_t sequenceContextLength;
        int64_t expectedSelectIndex;
    };

    // Define test cases with clear input-output relationship
    std::vector<TestCase> testCases
        = {{128, 127}, {64, 63}, {96, 95}, {112, 111}, {80, 79}, {100, 99}, {120, 119}, {88, 87}};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        // Extract input and expected data for current batch size
        std::vector<int32_t> inputSequenceContextLengths(batchSize);
        std::vector<int64_t> expectedSelectIndices(batchSize);
        for (int32_t b = 0; b < batchSize; b++)
        {
            inputSequenceContextLengths[b] = testCases[b].sequenceContextLength;
            expectedSelectIndices[b] = testCases[b].expectedSelectIndex;
        }

        auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);

        // Directly populate sequenceContextLengths as input
        copyHostToDevice<int32_t>(sequenceContextLengthsDevice, inputSequenceContextLengths);

        prepareEaglePrefillInputs(sequenceContextLengthsDevice, selectIndicesDevice, stream);

        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectIndicesDevice);

        for (int32_t b = 0; b < batchSize; b++)
        {
            EXPECT_EQ(actualSelectIndices[b], expectedSelectIndices[b]);
        }
    }
}

// ============================================================================
// Test 2: initializeDraftTreeTables
// Description: Initialize draft tree with root + level1 tokens, translate to full vocab
// Format: [root(score=0, parent=-1), level1_tokens(score=logProb, parent=0), empty(-inf, -5)]
// ============================================================================
TEST(EagleKernels, InitializeDraftTreeTables)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t tableLength = 21; // 1 + 4 + 16

    std::vector<int32_t> inputSelectedIndices = {
        10, 20, 30, 40, // Batch 0
        15, 25, 35, 45, // Batch 1
        12, 22, 32, 42, // Batch 2
        18, 28, 38, 48, // Batch 3
        11, 21, 31, 41, // Batch 4
        13, 23, 33, 43, // Batch 5
        14, 24, 34, 44, // Batch 6
        16, 26, 36, 46  // Batch 7
    };

    std::vector<float> inputLogProbs = {
        -0.5f, -0.7f, -0.9f, -1.1f,     // Batch 0
        -0.6f, -0.8f, -1.0f, -1.2f,     // Batch 1
        -0.55f, -0.75f, -0.95f, -1.15f, // Batch 2
        -0.65f, -0.85f, -1.05f, -1.25f, // Batch 3
        -0.52f, -0.72f, -0.92f, -1.12f, // Batch 4
        -0.58f, -0.78f, -0.98f, -1.18f, // Batch 5
        -0.62f, -0.82f, -1.02f, -1.22f, // Batch 6
        -0.68f, -0.88f, -1.08f, -1.28f  // Batch 7
    };

    std::vector<int32_t> inputRootTokens = {5000, 6000, 7000, 8000, 5500, 6500, 7500, 8500};

    // Vocab mapping: full_id = draft_id + vocabMapping[draft_id]
    std::vector<int32_t> inputVocabMapping(1000);
    for (int i = 0; i < 1000; i++)
    {
        inputVocabMapping[i] = i * 100;
    }

    // Expected: Batch 0 full data (others follow same pattern)
    // Root: 5000, Level1: 10+1000=1010, 20+2000=2020, 30+3000=3030, 40+4000=4040
    std::vector<int32_t> expectedIdsBatch0
        = {5000, 1010, 2020, 3030, 4040, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Note: the parent of the root node is -1, the parent of the level 1 nodes is 0, the parent of the empty nodes is
    // -5. It's pre-defined in the kernel with below constants. constexpr int32_t kROOT_NODE_PREDECESSOR{-1}; constexpr
    // int32_t kEMPTY_NODE_PREDECESSOR{-5};
    std::vector<int32_t> expectedParentsBatch0
        = {-1, 0, 0, 0, 0, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto selectedIndicesDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto logProbsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto rootTokensDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto vocabMappingDevice = rt::Tensor({1000}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftIdFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto draftParentFullTableDevice = rt::Tensor({batchSize, tableLength}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(logProbsDevice.rawPointer(), inputLogProbs.data(), batchSize * draftTopK * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rootTokensDevice.rawPointer(), inputRootTokens.data(), batchSize * sizeof(int32_t),
            cudaMemcpyHostToDevice));
        copyHostToDevice<int32_t>(vocabMappingDevice, inputVocabMapping);

        initializeDraftTreeTables(selectedIndicesDevice, logProbsDevice, rootTokensDevice, vocabMappingDevice,
            draftIdFullTableDevice, draftScoreFullTableDevice, draftParentFullTableDevice, draftTopK, stream);

        auto const actualIds = copyDeviceToHost<int32_t>(draftIdFullTableDevice);
        auto const actualScores = copyDeviceToHost<float>(draftScoreFullTableDevice);
        auto const actualParents = copyDeviceToHost<int32_t>(draftParentFullTableDevice);

        // Verify batch 0 in detail
        for (int32_t i = 0; i < tableLength; i++)
        {
            EXPECT_EQ(actualIds[i], expectedIdsBatch0[i]);
            EXPECT_EQ(actualParents[i], expectedParentsBatch0[i]);
            if (i == 0)
            {
                EXPECT_FLOAT_EQ(actualScores[i], 0.0f);
            }
            else if (i <= draftTopK)
            {
                EXPECT_FLOAT_EQ(actualScores[i], inputLogProbs[i - 1]);
            }
            else
            {
                EXPECT_TRUE(std::isinf(actualScores[i]) && actualScores[i] < 0);
            }
        }

        // Verify all batches have correct structure
        for (int32_t b = 0; b < batchSize; b++)
        {
            int offset = b * tableLength;
            EXPECT_EQ(actualIds[offset], inputRootTokens[b]);
            EXPECT_EQ(actualParents[offset], -1);
            for (int i = 1; i <= draftTopK; i++)
            {
                EXPECT_EQ(actualParents[offset + i], 0);
            }
        }
    }
}

// ============================================================================
// Test 3: assembleInitialDraftTreeInput
// Description: Assemble first round draft tree input from full table
// This test simulates a multi-step scenario (draftingStep=6) but only tests initial round
// The full table size follows: 1 + topK + (step-1) * topK²
// ============================================================================
TEST(EagleKernels, AssembleInitialDraftTreeInput)
{
    cudaStream_t stream = nullptr;

    // Simulate production config: topK=10, step=6 (scaled down for testing)
    // Use topK=4, step=4 to keep test data manageable
    // fullTableLength = 1 + topK + (step-1) * topK² = 1 + 4 + 3*16 = 53
    int32_t draftTopK = 4;
    int32_t draftingStep = 4; // Simulate multi-step scenario
    int32_t fullTableLength = 1 + draftTopK + (draftingStep - 1) * draftTopK * draftTopK;
    // fullTableLength = 1 + 4 + 3*16 = 53

    int32_t paddedDraftTreeSize = 12;
    int32_t draftHiddenDim = 256;

    // Input: Full table with complete structure (root + all levels)
    // Each batch has fullTableLength=53 elements
    // Structure: [root, level1(4), level2(16), level3(16), level4(16)]
    //
    // CRITICAL: Data must be unique at each global index to distinguish correct vs wrong offset
    // Use pattern: globalIndex * 10000 + batch * 100 + localPos
    std::vector<int32_t> inputDraftIdFullTable;

    for (int b = 0; b < 8; b++)
    {
        for (int localPos = 0; localPos < fullTableLength; localPos++)
        {
            int32_t globalIdx = b * fullTableLength + localPos;
            // Each element encodes its global position uniquely
            // This ensures different offsets give completely different values
            int32_t value = globalIdx * 100 + b * 10 + localPos;
            inputDraftIdFullTable.push_back(value);
        }
    }

    std::vector<half> inputHiddenStates(8 * draftHiddenDim);
    for (int b = 0; b < 8; b++)
    {
        for (int d = 0; d < draftHiddenDim; d++)
        {
            inputHiddenStates[b * draftHiddenDim + d] = __float2half(1.0f + b * 0.1f);
        }
    }

    // Expected outputs for multiple batches
    // Initial round reads positions [1:4] (level 1) from each batch's fullTable
    // Each value = globalIdx * 100 + batch * 10 + localPos
    //
    // Batch 0: reads fullTable[globalIdx 1-4]
    //   fullTable[1] = 1*100 + 0*10 + 1 = 101
    //   fullTable[2] = 2*100 + 0*10 + 2 = 202
    //   fullTable[3] = 3*100 + 0*10 + 3 = 303
    //   fullTable[4] = 4*100 + 0*10 + 4 = 404
    std::vector<int32_t> expectedInputIdsBatch0 = {101, 202, 303, 404, 0, 0, 0, 0, 0, 0, 0, 0};

    // Batch 1: reads fullTable[globalIdx 54-57] (offset = 1*53 + [1:4])
    //   fullTable[54] = 54*100 + 1*10 + 1 = 5411
    //   fullTable[55] = 55*100 + 1*10 + 2 = 5512
    //   fullTable[56] = 56*100 + 1*10 + 3 = 5613
    //   fullTable[57] = 57*100 + 1*10 + 4 = 5714
    std::vector<int32_t> expectedInputIdsBatch1 = {5411, 5512, 5613, 5714, 0, 0, 0, 0, 0, 0, 0, 0};

    // Batch 2: reads fullTable[globalIdx 107-110] (offset = 2*53 + [1:4])
    //   fullTable[107] = 107*100 + 2*10 + 1 = 10721
    //   fullTable[108] = 108*100 + 2*10 + 2 = 10822
    //   fullTable[109] = 109*100 + 2*10 + 3 = 10923
    //   fullTable[110] = 110*100 + 2*10 + 4 = 11024
    std::vector<int32_t> expectedInputIdsBatch2 = {10721, 10822, 10923, 11024, 0, 0, 0, 0, 0, 0, 0, 0};

    // If kernel uses WRONG formula (21 instead of 53):
    //   Batch 1 would read fullTable[22-25]:
    //     [22] = 22*100 + 0*10 + 22 = 2222 ✗ (not 5411)
    //     [23] = 23*100 + 0*10 + 23 = 2323 ✗ (not 5512)
    //   Test would FAIL! ✅

    int32_t expectedTreeLength = 4;

    std::cout << "Testing with fullTableLength=" << fullTableLength << " (simulates draftingStep=" << draftingStep
              << ")\n";
    std::cout << "Kernel's wrong formula would give: " << (1 + draftTopK + draftTopK * draftTopK) << "\n";

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftIdFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenStatesOutputDevice
            = rt::Tensor({batchSize, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto inputIdsDevice = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenStatesInputDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);

        CUDA_CHECK(cudaMemcpy(draftIdFullTableDevice.rawPointer(), inputDraftIdFullTable.data(),
            batchSize * fullTableLength * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftHiddenStatesOutputDevice.rawPointer(), inputHiddenStates.data(),
            batchSize * draftHiddenDim * sizeof(half), cudaMemcpyHostToDevice));

        assembleInitialDraftTreeInput(draftIdFullTableDevice, draftHiddenStatesOutputDevice, inputIdsDevice,
            draftHiddenStatesInputDevice, draftTreeLengthDevice, draftTreeMaskDevice, draftTopK, stream);

        auto const actualInputIds = copyDeviceToHost<int32_t>(inputIdsDevice);
        auto const actualTreeLength = copyDeviceToHost<int32_t>(draftTreeLengthDevice);
        auto const actualMask = copyDeviceToHost<int8_t>(draftTreeMaskDevice);

        // ========== Verify Batch 0 ==========
        for (int i = 0; i < paddedDraftTreeSize; i++)
        {
            EXPECT_EQ(actualInputIds[i], expectedInputIdsBatch0[i]) << "Batch 0, position " << i << " mismatch";
        }
        EXPECT_EQ(actualTreeLength[0], expectedTreeLength);

        // Verify Batch 0 mask: diagonal only for first topK tokens
        for (int i = 0; i < paddedDraftTreeSize; i++)
        {
            for (int j = 0; j < paddedDraftTreeSize; j++)
            {
                int8_t expected = (i < draftTopK && i == j) ? 1 : 0;
                EXPECT_EQ(actualMask[i * paddedDraftTreeSize + j], expected)
                    << "Batch 0 mask mismatch at [" << i << "][" << j << "]";
            }
        }

        // ========== CRITICAL: Verify Batch 1 (if batchSize >= 2) ==========
        // This tests that kernel uses fullTableLength parameter for offset calculation
        // With fullTableLength=53 (step=4), Batch 1 should read from globalIdx [54:57]
        // If kernel used wrong formula (1+topK+topK²=21), it would read from globalIdx [22:25]
        if (batchSize >= 2)
        {
            // Kernel line 218: tableOffset = batchIdx * fullTableLength + i + 1
            // For Batch 1, i=0:
            //   CORRECT: tableOffset = 1*53 + 0 + 1 = 54 → fullTable[54] = 5411 ✓
            //   WRONG:   tableOffset = 1*21 + 0 + 1 = 22 → fullTable[22] = 2222 ✗
            // These values are COMPLETELY DIFFERENT, so bug will be caught!

            for (int i = 0; i < draftTopK; i++) // Only check first topK positions
            {
                int32_t batchOffset = 1 * paddedDraftTreeSize;
                int32_t actual = actualInputIds[batchOffset + i];
                int32_t expected = expectedInputIdsBatch1[i];

                // Calculate what we would get if kernel used wrong formula
                int32_t wrongOffset = 1 * 21 + i + 1;                          // Wrong formula result
                int32_t wrongValue = wrongOffset * 100 + 0 * 10 + wrongOffset; // Batch 0's data

                EXPECT_EQ(actual, expected)
                    << "Batch 1, position " << i << " CRITICAL MISMATCH"
                    << "\n  Expected: " << expected << " (from correct offset " << (1 * fullTableLength + i + 1) << ")"
                    << "\n  If kernel used WRONG formula (21), would get: " << wrongValue << " (from wrong offset "
                    << wrongOffset << ")"
                    << "\n  Actual: " << actual << "\n  This verifies kernel uses fullTableLength=" << fullTableLength
                    << ", NOT hardcoded formula " << (1 + draftTopK + draftTopK * draftTopK);
            }
            EXPECT_EQ(actualTreeLength[1], expectedTreeLength);

            // Verify Batch 1 mask
            for (int i = 0; i < paddedDraftTreeSize; i++)
            {
                for (int j = 0; j < paddedDraftTreeSize; j++)
                {
                    int8_t expected = (i < draftTopK && i == j) ? 1 : 0;
                    int32_t maskOffset = 1 * paddedDraftTreeSize * paddedDraftTreeSize + i * paddedDraftTreeSize + j;
                    EXPECT_EQ(actualMask[maskOffset], expected)
                        << "Batch 1 mask mismatch at [" << i << "][" << j << "]";
                }
            }
        }

        // ========== Verify Batch 2 (if batchSize >= 4) ==========
        if (batchSize >= 4)
        {
            for (int i = 0; i < paddedDraftTreeSize; i++)
            {
                int32_t batchOffset = 2 * paddedDraftTreeSize;
                EXPECT_EQ(actualInputIds[batchOffset + i], expectedInputIdsBatch2[i])
                    << "Batch 2, position " << i << " mismatch";
            }
            EXPECT_EQ(actualTreeLength[2], expectedTreeLength);
        }

        // Verify tree lengths for all batches
        for (int32_t b = 0; b < batchSize; b++)
        {
            EXPECT_EQ(actualTreeLength[b], expectedTreeLength) << "Batch " << b << " tree length mismatch";
        }

        // Print success message for multi-batch tests
        if (batchSize >= 2)
        {
            std::cout << "✓ Multi-batch test passed (batchSize=" << batchSize
                      << "): fullTableLength=" << fullTableLength << " (step=" << draftingStep
                      << ") correctly used, not wrong formula " << (1 + draftTopK + draftTopK * draftTopK) << "\n";
        }
    }
}

// ============================================================================
// Test 4: assembleInitialIntermediateData
// Description: Copy logProbs to scores; set parents to [1,2,3,4] (first level indices)
// ============================================================================
TEST(EagleKernels, AssembleInitialIntermediateData)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;

    std::vector<float> inputLogProbs = {
        -0.5f, -0.7f, -0.9f, -1.1f,     // Batch 0
        -0.6f, -0.8f, -1.0f, -1.2f,     // Batch 1
        -0.55f, -0.75f, -0.95f, -1.15f, // Batch 2
        -0.65f, -0.85f, -1.05f, -1.25f, // Batch 3
        -0.52f, -0.72f, -0.92f, -1.12f, // Batch 4
        -0.58f, -0.78f, -0.98f, -1.18f, // Batch 5
        -0.62f, -0.82f, -1.02f, -1.22f, // Batch 6
        -0.68f, -0.88f, -1.08f, -1.28f  // Batch 7
    };

    // Expected: parents = [1,2,3,4], scores = logProbs
    std::vector<int32_t> expectedParents = {1, 2, 3, 4};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto logProbsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateParentsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto intermediateScoresDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);

        CUDA_CHECK(cudaMemcpy(logProbsDevice.rawPointer(), inputLogProbs.data(), batchSize * draftTopK * sizeof(float),
            cudaMemcpyHostToDevice));

        assembleInitialIntermediateData(
            logProbsDevice, intermediateParentsDevice, intermediateScoresDevice, draftTopK, stream);

        auto const actualParents = copyDeviceToHost<int32_t>(intermediateParentsDevice);
        auto const actualScores = copyDeviceToHost<float>(intermediateScoresDevice);

        // Verify batch 0
        for (int i = 0; i < draftTopK; i++)
        {
            EXPECT_EQ(actualParents[i], expectedParents[i]);
            EXPECT_FLOAT_EQ(actualScores[i], inputLogProbs[i]);
        }

        // Verify all batches have correct parents
        for (int32_t b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < draftTopK; i++)
            {
                EXPECT_EQ(actualParents[b * draftTopK + i], expectedParents[i]);
            }
        }
    }
}

// ============================================================================
// Test 5: computeCuScoresAndTranslateToken
// Description: Translate draft tokens to full vocab; compute cumulative scores
// score[i] = intermediateScores[i/topK] + logProbs[i]
// ============================================================================
TEST(EagleKernels, ComputeCuScoresAndTranslateToken)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;

    // Input: 16 candidate tokens per batch
    std::vector<int32_t> inputSelectedIndices(8 * 16);
    std::vector<float> inputLogProbs(8 * 16);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < 16; i++)
        {
            inputSelectedIndices[b * 16 + i] = 10 + b * 10 + i;
            inputLogProbs[b * 16 + i] = -0.3f - i * 0.1f;
        }
    }

    std::vector<float> inputIntermediateScores = {
        -0.5f, -0.7f, -0.9f, -1.1f,     // Batch 0
        -0.6f, -0.8f, -1.0f, -1.2f,     // Batch 1
        -0.55f, -0.75f, -0.95f, -1.15f, // Batch 2
        -0.65f, -0.85f, -1.05f, -1.25f, // Batch 3
        -0.52f, -0.72f, -0.92f, -1.12f, // Batch 4
        -0.58f, -0.78f, -0.98f, -1.18f, // Batch 5
        -0.62f, -0.82f, -1.02f, -1.22f, // Batch 6
        -0.68f, -0.88f, -1.08f, -1.28f  // Batch 7
    };

    std::vector<int32_t> inputVocabMapping(1000);
    for (int i = 0; i < 1000; i++)
    {
        inputVocabMapping[i] = i * 100;
    }

    // Expected batch 0: draft_id=10 -> full_id=10+1000=1010
    // score[0] = intermediateScores[0] + logProbs[0] = -0.5 + (-0.3) = -0.8
    int32_t expectedId0 = 10 + 1000;        // 1010
    float expectedScore0 = -0.5f + (-0.3f); // -0.8

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto selectedIndicesDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto logProbsDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateScoresDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto vocabMappingDevice = rt::Tensor({1000}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftIdTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kFLOAT);

        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * 16 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            logProbsDevice.rawPointer(), inputLogProbs.data(), batchSize * 16 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(intermediateScoresDevice.rawPointer(), inputIntermediateScores.data(),
            batchSize * draftTopK * sizeof(float), cudaMemcpyHostToDevice));
        copyHostToDevice<int32_t>(vocabMappingDevice, inputVocabMapping);

        computeCuScoresAndTranslateToken(selectedIndicesDevice, logProbsDevice, intermediateScoresDevice,
            vocabMappingDevice, draftIdTableDevice, draftScoreTableDevice, draftTopK, stream);

        auto const actualIds = copyDeviceToHost<int32_t>(draftIdTableDevice);
        auto const actualScores = copyDeviceToHost<float>(draftScoreTableDevice);

        // Verify batch 0
        EXPECT_EQ(actualIds[0], expectedId0);
        EXPECT_FLOAT_EQ(actualScores[0], expectedScore0);

        // Verify score computation for all batches (first 4 entries use parent 0)
        for (int32_t b = 0; b < batchSize; b++)
        {
            float parentScore = inputIntermediateScores[b * draftTopK];
            for (int i = 0; i < 4; i++)
            {
                float expected = parentScore + inputLogProbs[b * 16 + i];
                EXPECT_FLOAT_EQ(actualScores[b * 16 + i], expected);
            }
        }
    }
}

// ============================================================================
// Test 6: prepareEagleAcceptDecodeTokenInputs
// Description: Create causal mask and position indices for accepted tokens
// mask[i] = 0b00..011..1 (i+1 ones), positions = startIndex + [0,1,2,..]
// Now supports per-batch variable accepted token numbers
// ============================================================================
TEST(EagleKernels, PrepareEagleAcceptDecodeTokenInputs)
{
    cudaStream_t stream = nullptr;
    int32_t maxAcceptedTokenNum = 8; // Max across all batches

    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 300, 400, 150, 250, 350, 450};
    std::vector<int32_t> inputAcceptedTokenNums = {5, 3, 6, 4, 5, 7, 4, 8};

    // Expected for batch 0: 5 tokens
    std::vector<int32_t> expectedMaskBatch0 = {1, 3, 7, 15, 31};
    std::vector<int32_t> expectedPositionsBatch0 = {100, 101, 102, 103, 104};
    int64_t expectedSelectIndexBatch0 = 4; // Last token
    // Context length now uses maxAcceptedTokenNum (padded length) instead of actual acceptedTokenNum
    // This is required for correct XQA attention context K range calculation with padding
    int32_t expectedContextLenBatch0 = 100 + maxAcceptedTokenNum; // 108

    // Expected for batch 1: 3 tokens
    int64_t expectedSelectIndexBatch1 = 2;
    int32_t expectedContextLenBatch1 = 200 + maxAcceptedTokenNum; // 208

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto acceptedTokenNumsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto packedMaskDevice = rt::Tensor({batchSize, maxAcceptedTokenNum, 1}, rt::DeviceType::kGPU, DataType::kINT32);
        auto positionsDevice = rt::Tensor({batchSize, maxAcceptedTokenNum}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
        auto contextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(sequenceStartIndicesDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(acceptedTokenNumsDevice.rawPointer(), inputAcceptedTokenNums.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleAcceptDecodeTokenInputs(sequenceStartIndicesDevice, acceptedTokenNumsDevice, packedMaskDevice,
            positionsDevice, selectIndicesDevice, contextLengthsDevice, stream);

        auto const actualMask = copyDeviceToHost<int32_t>(packedMaskDevice);
        auto const actualPositions = copyDeviceToHost<int32_t>(positionsDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectIndicesDevice);
        auto const actualContextLengths = copyDeviceToHost<int32_t>(contextLengthsDevice);

        // Verify batch 0
        for (int i = 0; i < inputAcceptedTokenNums[0]; i++)
        {
            EXPECT_EQ(actualMask[i], expectedMaskBatch0[i]);
            EXPECT_EQ(actualPositions[i], expectedPositionsBatch0[i]);
        }
        EXPECT_EQ(actualSelectIndices[0], expectedSelectIndexBatch0);
        EXPECT_EQ(actualContextLengths[0], expectedContextLenBatch0);

        // Verify batch 1 if available
        if (batchSize >= 2)
        {
            EXPECT_EQ(actualSelectIndices[1], expectedSelectIndexBatch1);
            EXPECT_EQ(actualContextLengths[1], expectedContextLenBatch1);
        }

        // Verify all batches have correct select indices and context lengths
        for (int32_t b = 0; b < batchSize; b++)
        {
            EXPECT_EQ(actualSelectIndices[b], inputAcceptedTokenNums[b] - 1);
            // Context length uses maxAcceptedTokenNum (padded length) for correct XQA attention range
            EXPECT_EQ(actualContextLengths[b], inputSequenceStartIndices[b] + maxAcceptedTokenNum);
        }
    }
}

// ============================================================================
// Test 7: constructVerificationDraftTree
// Description: Build verification tree from selected nodes in full table
// Track parent chain to build attention mask
// Large-scale test with deeper tree (4 layers) and batch size 2
// ============================================================================
TEST(EagleKernels, ConstructVerificationDraftTree)
{
    cudaStream_t stream = nullptr;
    int32_t const batchSize = 2;
    int32_t const fullTableLength = 50; // Larger table: 1 root + 4 L1 + 16 L2 + 29 L3
    int32_t const verifyTreeSize = 12;  // Larger verification tree

    // ========== Batch 0: Construct a 4-layer tree with specific structure ==========
    // Tree structure for Batch 0:
    // Layer 0 (root):     0
    // Layer 1 (4 nodes):  1, 2, 3, 4 (all children of 0)
    // Layer 2 (16 nodes): 5-20 (4 children per L1 node)
    // Layer 3 (29 nodes): 21-49 (varies, testing asymmetric tree)
    std::vector<int32_t> inputDraftIdFullTableBatch0(fullTableLength);
    std::vector<int32_t> inputDraftParentFullTableBatch0(fullTableLength);

    // Root node
    inputDraftIdFullTableBatch0[0] = 1000;
    inputDraftParentFullTableBatch0[0] = -1;

    // Layer 1: nodes 1-4, all children of root (0)
    for (int i = 1; i <= 4; i++)
    {
        inputDraftIdFullTableBatch0[i] = 1000 + i * 100;
        inputDraftParentFullTableBatch0[i] = 0;
    }

    // Layer 2: nodes 5-20, 4 children per L1 node
    for (int i = 5; i <= 20; i++)
    {
        inputDraftIdFullTableBatch0[i] = 1000 + i * 100;
        inputDraftParentFullTableBatch0[i] = 1 + (i - 5) / 4; // Parent in range [1,4]
    }

    // Layer 3: nodes 21-49, children of L2 nodes (asymmetric distribution)
    for (int i = 21; i < fullTableLength; i++)
    {
        inputDraftIdFullTableBatch0[i] = 1000 + i * 100;
        // Distribute among L2 nodes: more children to early L2 nodes
        inputDraftParentFullTableBatch0[i] = 5 + (i - 21) % 16;
    }

    // Select a path through the tree for verification (Batch 0):
    // Path: 0 -> 2 -> 7 -> 11 -> 27, and additional branches
    // Selected: [0(root), 2(L1), 6(L2), 7(L2), 11(L2), 14(L2), 27(L3), 28(L3), 32(L3), 35(L3), 40(L3), 45(L3)]
    std::vector<int32_t> inputSelectedIndicesBatch0 = {0, 2, 6, 7, 11, 14, 27, 28, 32, 35, 40, 45};

    // Expected token IDs for Batch 0
    std::vector<int32_t> expectedIdsBatch0(verifyTreeSize);
    for (int i = 0; i < verifyTreeSize; i++)
    {
        expectedIdsBatch0[i] = inputDraftIdFullTableBatch0[inputSelectedIndicesBatch0[i]];
    }

    // Expected attention mask for Batch 0
    // Kernel logic: For each token, sequentially match its parent chain against prior tokens in verify tree
    // If direct parent is not found, all ancestors become unreachable
    //
    // Parent chains (computed):
    // node 0: [] (root)
    // node 2: [0]
    // node 6: [1, 0] - node 1 NOT in verify tree → only self
    // node 7: [1, 0] - node 1 NOT in verify tree → only self
    // node 11: [2, 0] - node 2 IS in verify tree (Token 1) → can reach root
    // node 14: [3, 0] - node 3 NOT in verify tree → only self
    // node 27: [11, 2, 0] - all in verify tree → full chain
    // node 28: [12, 2, 0] - node 12 NOT in verify tree → only self
    // node 32: [16, 3, 0] - node 16 NOT in verify tree → only self
    // node 35: [19, 4, 0] - node 19 NOT in verify tree → only self
    // node 40: [8, 1, 0] - node 8 NOT in verify tree → only self
    // node 45: [13, 3, 0] - node 13 NOT in verify tree → only self
    std::vector<std::vector<int8_t>> expectedMaskBatch0(verifyTreeSize, std::vector<int8_t>(verifyTreeSize, 0));
    // Token 0 (node 0): attends to [0]
    expectedMaskBatch0[0][0] = 1;
    // Token 1 (node 2): attends to [0, 1]
    expectedMaskBatch0[1][0] = 1;
    expectedMaskBatch0[1][1] = 1;
    // Token 2 (node 6): attends to [2]
    expectedMaskBatch0[2][2] = 1;
    // Token 3 (node 7): attends to [3]
    expectedMaskBatch0[3][3] = 1;
    // Token 4 (node 11): attends to [0, 1, 4]
    expectedMaskBatch0[4][0] = 1;
    expectedMaskBatch0[4][1] = 1;
    expectedMaskBatch0[4][4] = 1;
    // Token 5 (node 14): attends to [5]
    expectedMaskBatch0[5][5] = 1;
    // Token 6 (node 27): attends to [0, 1, 4, 6]
    expectedMaskBatch0[6][0] = 1;
    expectedMaskBatch0[6][1] = 1;
    expectedMaskBatch0[6][4] = 1;
    expectedMaskBatch0[6][6] = 1;
    // Token 7 (node 28): attends to [7]
    expectedMaskBatch0[7][7] = 1;
    // Token 8 (node 32): attends to [8]
    expectedMaskBatch0[8][8] = 1;
    // Token 9 (node 35): attends to [9]
    expectedMaskBatch0[9][9] = 1;
    // Token 10 (node 40): attends to [10]
    expectedMaskBatch0[10][10] = 1;
    // Token 11 (node 45): attends to [11]
    expectedMaskBatch0[11][11] = 1;

    // ========== Batch 1: Different tree structure with deeper paths ==========
    std::vector<int32_t> inputDraftIdFullTableBatch1(fullTableLength);
    std::vector<int32_t> inputDraftParentFullTableBatch1(fullTableLength);

    // Root node
    inputDraftIdFullTableBatch1[0] = 2000;
    inputDraftParentFullTableBatch1[0] = -1;

    // Layer 1: nodes 1-4
    for (int i = 1; i <= 4; i++)
    {
        inputDraftIdFullTableBatch1[i] = 2000 + i * 50;
        inputDraftParentFullTableBatch1[i] = 0;
    }

    // Layer 2: nodes 5-20 (focus more children on node 3)
    for (int i = 5; i <= 20; i++)
    {
        inputDraftIdFullTableBatch1[i] = 2000 + i * 50;
        // Bias towards node 3: nodes 5-12 are children of node 3
        if (i <= 12)
        {
            inputDraftParentFullTableBatch1[i] = 3;
        }
        else
        {
            inputDraftParentFullTableBatch1[i] = 1 + (i - 13) % 3;
        }
    }

    // Layer 3: nodes 21-49, children of L2 nodes (creating longer chains)
    for (int i = 21; i < fullTableLength; i++)
    {
        inputDraftIdFullTableBatch1[i] = 2000 + i * 50;
        // Create deeper paths through nodes 5-12
        inputDraftParentFullTableBatch1[i] = 5 + (i - 21) % 8;
    }

    // Select a deep path for verification (Batch 1):
    // Path focusing on deep chains: 0 -> 3 -> 5 -> 6 -> 8 -> 21 -> 25 -> 29 -> 33 -> 37 -> 41 -> 48
    std::vector<int32_t> inputSelectedIndicesBatch1 = {0, 3, 5, 6, 8, 21, 25, 29, 33, 37, 41, 48};

    // Expected token IDs for Batch 1
    std::vector<int32_t> expectedIdsBatch1(verifyTreeSize);
    for (int i = 0; i < verifyTreeSize; i++)
    {
        expectedIdsBatch1[i] = inputDraftIdFullTableBatch1[inputSelectedIndicesBatch1[i]];
    }

    // Expected attention mask for Batch 1
    // Parent chains (computed):
    // node 0: [] (root)
    // node 3: [0]
    // node 5: [3, 0] - full chain in verify tree
    // node 6: [3, 0] - full chain in verify tree
    // node 8: [3, 0] - full chain in verify tree
    // node 21: [5, 3, 0] - full chain in verify tree
    // node 25: [9, 3, 0] - node 9 NOT in verify tree → only self
    // node 29: [5, 3, 0] - full chain in verify tree
    // node 33: [9, 3, 0] - node 9 NOT in verify tree → only self
    // node 37: [5, 3, 0] - full chain in verify tree
    // node 41: [9, 3, 0] - node 9 NOT in verify tree → only self
    // node 48: [8, 3, 0] - full chain in verify tree (node 8 is Token 4)
    std::vector<std::vector<int8_t>> expectedMaskBatch1(verifyTreeSize, std::vector<int8_t>(verifyTreeSize, 0));
    // Token 0 (node 0): attends to [0]
    expectedMaskBatch1[0][0] = 1;
    // Token 1 (node 3): attends to [0, 1]
    expectedMaskBatch1[1][0] = 1;
    expectedMaskBatch1[1][1] = 1;
    // Token 2 (node 5): attends to [0, 1, 2]
    expectedMaskBatch1[2][0] = 1;
    expectedMaskBatch1[2][1] = 1;
    expectedMaskBatch1[2][2] = 1;
    // Token 3 (node 6): attends to [0, 1, 3]
    expectedMaskBatch1[3][0] = 1;
    expectedMaskBatch1[3][1] = 1;
    expectedMaskBatch1[3][3] = 1;
    // Token 4 (node 8): attends to [0, 1, 4]
    expectedMaskBatch1[4][0] = 1;
    expectedMaskBatch1[4][1] = 1;
    expectedMaskBatch1[4][4] = 1;
    // Token 5 (node 21): attends to [0, 1, 2, 5]
    expectedMaskBatch1[5][0] = 1;
    expectedMaskBatch1[5][1] = 1;
    expectedMaskBatch1[5][2] = 1;
    expectedMaskBatch1[5][5] = 1;
    // Token 6 (node 25): attends to [6]
    expectedMaskBatch1[6][6] = 1;
    // Token 7 (node 29): attends to [0, 1, 2, 7]
    expectedMaskBatch1[7][0] = 1;
    expectedMaskBatch1[7][1] = 1;
    expectedMaskBatch1[7][2] = 1;
    expectedMaskBatch1[7][7] = 1;
    // Token 8 (node 33): attends to [8]
    expectedMaskBatch1[8][8] = 1;
    // Token 9 (node 37): attends to [0, 1, 2, 9]
    expectedMaskBatch1[9][0] = 1;
    expectedMaskBatch1[9][1] = 1;
    expectedMaskBatch1[9][2] = 1;
    expectedMaskBatch1[9][9] = 1;
    // Token 10 (node 41): attends to [10]
    expectedMaskBatch1[10][10] = 1;
    // Token 11 (node 48): attends to [0, 1, 4, 11]
    expectedMaskBatch1[11][0] = 1;
    expectedMaskBatch1[11][1] = 1;
    expectedMaskBatch1[11][4] = 1;
    expectedMaskBatch1[11][11] = 1;

    // ========== Prepare combined input for both batches ==========
    std::vector<int32_t> inputDraftIdFullTable(batchSize * fullTableLength);
    std::vector<int32_t> inputDraftParentFullTable(batchSize * fullTableLength);
    std::vector<int32_t> inputSelectedIndices(batchSize * verifyTreeSize);

    // Copy batch 0 data
    std::copy(inputDraftIdFullTableBatch0.begin(), inputDraftIdFullTableBatch0.end(), inputDraftIdFullTable.begin());
    std::copy(inputDraftParentFullTableBatch0.begin(), inputDraftParentFullTableBatch0.end(),
        inputDraftParentFullTable.begin());
    std::copy(inputSelectedIndicesBatch0.begin(), inputSelectedIndicesBatch0.end(), inputSelectedIndices.begin());

    // Copy batch 1 data
    std::copy(inputDraftIdFullTableBatch1.begin(), inputDraftIdFullTableBatch1.end(),
        inputDraftIdFullTable.begin() + fullTableLength);
    std::copy(inputDraftParentFullTableBatch1.begin(), inputDraftParentFullTableBatch1.end(),
        inputDraftParentFullTable.begin() + fullTableLength);
    std::copy(inputSelectedIndicesBatch1.begin(), inputSelectedIndicesBatch1.end(),
        inputSelectedIndices.begin() + verifyTreeSize);

    // ========== Execute kernel ==========
    auto draftIdFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftParentFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
    auto selectedIndicesDevice = rt::Tensor({batchSize, verifyTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto inputIdsDevice = rt::Tensor({batchSize, verifyTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTreeMaskDevice
        = rt::Tensor({batchSize, verifyTreeSize, verifyTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);

    copyHostToDevice<int32_t>(draftIdFullTableDevice, inputDraftIdFullTable);
    copyHostToDevice<int32_t>(draftParentFullTableDevice, inputDraftParentFullTable);
    copyHostToDevice<int32_t>(selectedIndicesDevice, inputSelectedIndices);

    constructVerificationDraftTree(draftIdFullTableDevice, draftParentFullTableDevice, selectedIndicesDevice,
        inputIdsDevice, draftTreeMaskDevice, stream);

    auto const actualIds = copyDeviceToHost<int32_t>(inputIdsDevice);
    auto const actualMask = copyDeviceToHost<int8_t>(draftTreeMaskDevice);

    // ========== Comprehensive verification for Batch 0 ==========
    // Verify token IDs for batch 0
    for (int i = 0; i < verifyTreeSize; i++)
    {
        EXPECT_EQ(actualIds[i], expectedIdsBatch0[i]) << "Batch 0 token ID mismatch at position " << i;
    }

    // Verify attention mask for batch 0
    for (int i = 0; i < verifyTreeSize; i++)
    {
        for (int j = 0; j < verifyTreeSize; j++)
        {
            int maskIdx = i * verifyTreeSize + j;
            EXPECT_EQ(actualMask[maskIdx], expectedMaskBatch0[i][j])
                << "Batch 0 mask mismatch at position [" << i << "][" << j << "]";
        }
    }

    // ========== Comprehensive verification for Batch 1 ==========
    // Verify token IDs for batch 1
    for (int i = 0; i < verifyTreeSize; i++)
    {
        int actualIdx = verifyTreeSize + i;
        EXPECT_EQ(actualIds[actualIdx], expectedIdsBatch1[i]) << "Batch 1 token ID mismatch at position " << i;
    }

    // Verify attention mask for batch 1
    for (int i = 0; i < verifyTreeSize; i++)
    {
        for (int j = 0; j < verifyTreeSize; j++)
        {
            int maskIdx = verifyTreeSize * verifyTreeSize + i * verifyTreeSize + j;
            EXPECT_EQ(actualMask[maskIdx], expectedMaskBatch1[i][j])
                << "Batch 1 mask mismatch at position [" << i << "][" << j << "]";
        }
    }
}

// ============================================================================
// Test 8: eagleBaseCommitKVCacheAndAssembleHiddenState
// Description: Test inplace compaction of accepted tokens from stride=draftTreeSize to stride=maxDepth
// Key test: Verify multi-batch scenario where Batch 1+ needs to move ALL tokens including position 0
// ============================================================================
TEST(EagleKernels, EagleBaseCommitKVCacheAndAssembleHiddenState)
{
    cudaStream_t stream = nullptr;
    int32_t numLayers = 2;
    int32_t numKVHead = 4;
    int32_t headDim = 128;
    int32_t maxSeqLen = 2048;
    int32_t maxDepth = 6;       // After compaction, stride will be maxDepth
    int32_t draftTreeSize = 10; // Input stride is draftTreeSize
    int32_t baseHiddenDim = 512;

    // Test with 2 batches to verify compaction with stride change
    int32_t batchSize = 2;
    int32_t maxBatchSize = 2;

    // Batch 0: accept positions [0, 3, 7] (length=3)
    // Batch 1: accept positions [0, 2, 5] (length=3)
    std::vector<int32_t> inputAcceptedIndices = {
        0, 3, 7, -1, -1, -1, // Batch 0
        0, 2, 5, -1, -1, -1  // Batch 1
    };
    std::vector<int32_t> inputAcceptLengths = {3, 3};
    std::vector<int32_t> inputKvCacheLengths = {256, 256};

    // Setup input hidden states with unique markers for each batch and position
    // Use simple pattern: batch_id * 1000 + position_id * 100
    std::vector<half> inputHiddenState(batchSize * draftTreeSize * baseHiddenDim, __float2half(0.0f));

    // Batch 0: Input layout with stride=draftTreeSize=10
    //   Position 0: marker = 0 + 0*100 = 0
    //   Position 3: marker = 0 + 3*100 = 300
    //   Position 7: marker = 0 + 7*100 = 700
    for (int d = 0; d < baseHiddenDim; d++)
    {
        inputHiddenState[0 * draftTreeSize * baseHiddenDim + 0 * baseHiddenDim + d] = __float2half(0.0f);
        inputHiddenState[0 * draftTreeSize * baseHiddenDim + 3 * baseHiddenDim + d] = __float2half(300.0f);
        inputHiddenState[0 * draftTreeSize * baseHiddenDim + 7 * baseHiddenDim + d] = __float2half(700.0f);
    }

    // Batch 1: Input layout with stride=draftTreeSize=10, starting at offset=10*baseHiddenDim
    //   Position 0: marker = 1000 + 0*100 = 1000
    //   Position 2: marker = 1000 + 2*100 = 1200
    //   Position 5: marker = 1000 + 5*100 = 1500
    for (int d = 0; d < baseHiddenDim; d++)
    {
        inputHiddenState[1 * draftTreeSize * baseHiddenDim + 0 * baseHiddenDim + d] = __float2half(1000.0f);
        inputHiddenState[1 * draftTreeSize * baseHiddenDim + 2 * baseHiddenDim + d] = __float2half(1200.0f);
        inputHiddenState[1 * draftTreeSize * baseHiddenDim + 5 * baseHiddenDim + d] = __float2half(1500.0f);
    }

    // Create device tensors
    auto acceptedIndicesDevice = rt::Tensor({batchSize, maxDepth}, rt::DeviceType::kGPU, DataType::kINT32);
    auto acceptLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto kvCacheLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto kvCacheDevice = rt::Tensor(
        {numLayers, maxBatchSize, 2, numKVHead, maxSeqLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    auto hiddenStateDevice
        = rt::Tensor({batchSize, draftTreeSize, baseHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);

    std::vector<half> kvCache(numLayers * maxBatchSize * 2 * numKVHead * maxSeqLen * headDim, __float2half(1.0f));

    // Copy data to device
    copyHostToDevice<int32_t>(acceptedIndicesDevice, inputAcceptedIndices);
    copyHostToDevice<int32_t>(acceptLengthsDevice, inputAcceptLengths);
    copyHostToDevice<int32_t>(kvCacheLengthsDevice, inputKvCacheLengths);
    copyHostToDevice<half>(kvCacheDevice, kvCache);
    copyHostToDevice<half>(hiddenStateDevice, inputHiddenState);

    // Execute kernel - this should compact inplace from stride=10 to stride=maxDepth
    eagleBaseCommitKVCacheAndAssembleHiddenState(
        acceptedIndicesDevice, acceptLengthsDevice, kvCacheLengthsDevice, kvCacheDevice, hiddenStateDevice, stream);

    // Read back results
    auto actualHiddenState = copyDeviceToHost<half>(hiddenStateDevice);

    // ========== Verify Batch 0 ==========
    // After compaction, Batch 0 should be at offset 0 with compacted layout
    // Expected: [0.0, 300.0, 700.0] at positions [0, 1, 2]
    std::vector<float> expectedBatch0 = {0.0f, 300.0f, 700.0f};
    for (int i = 0; i < 3; i++)
    {
        // Batch 0 starts at offset 0, compacted with stride=maxDepth (not draftTreeSize anymore)
        float actual = __half2float(actualHiddenState[0 * draftTreeSize * baseHiddenDim + i * baseHiddenDim]);
        EXPECT_TRUE(isclose(actual, expectedBatch0[i], 1e-3f, 1e-3f))
            << "Batch 0, position " << i << ": expected " << expectedBatch0[i] << ", got " << actual;
    }

    // ========== Verify Batch 1 (CRITICAL TEST) ==========
    // After compaction, Batch 1 data should move from offset 10*dim to offset maxDepth*dim
    // This tests that ALL tokens including position 0 are moved correctly
    // Expected: [1000.0, 1200.0, 1500.0] at NEW offset (maxDepth * baseHiddenDim)
    std::vector<float> expectedBatch1 = {1000.0f, 1200.0f, 1500.0f};

    // NEW: Batch 1 compacted data should start at maxDepth tokens from beginning
    // NOT at draftTreeSize tokens (which was the old stride)
    int32_t batch1OutputOffset = maxDepth * baseHiddenDim; // New compacted offset

    for (int i = 0; i < 3; i++)
    {
        // CRITICAL: Read from NEW compacted location, not old location
        float actual = __half2float(actualHiddenState[batch1OutputOffset + i * baseHiddenDim]);
        EXPECT_TRUE(isclose(actual, expectedBatch1[i], 1e-3f, 1e-3f))
            << "Batch 1, position " << i << ": expected " << expectedBatch1[i] << ", got " << actual
            << " (reading from compacted offset " << batch1OutputOffset / baseHiddenDim << ")";
    }

    // Additional verification: old Batch 1 location should be overwritten or unchanged
    // (positions beyond maxDepth*batchSize are undefined after compaction)

    std::cout << "✓ Inplace compaction test passed: stride changed from " << draftTreeSize << " to " << maxDepth
              << " successfully\n";
    std::cout << "✓ Batch 1 position 0 correctly moved from offset " << draftTreeSize << " to offset " << maxDepth
              << "\n";
}

// ============================================================================
// Test 9: prepareEagleDraftProposalInputs
// Description: Pack tree mask, compute positions, prepare select indices
// selectIndices = [treeLen-selectTokenLen : treeLen]
// ============================================================================
TEST(EagleKernels, PrepareEagleDraftProposalInputs)
{
    cudaStream_t stream = nullptr;
    int32_t paddedDraftTreeSize = 12;
    int32_t selectTokenLength = 4;
    int32_t packedTreeMaskLen = divUp(paddedDraftTreeSize, 32);

    std::vector<int32_t> inputDraftTreeLength = {8, 9, 10, 11, 8, 9, 10, 11};
    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 300, 400, 150, 250, 350, 450};

    // Simple causal mask
    std::vector<int8_t> inputDraftTreeMask(8 * paddedDraftTreeSize * paddedDraftTreeSize, 0);
    for (int b = 0; b < 8; b++)
    {
        int treeLen = inputDraftTreeLength[b];
        for (int i = 0; i < treeLen; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                inputDraftTreeMask[b * paddedDraftTreeSize * paddedDraftTreeSize + i * paddedDraftTreeSize + j] = 1;
            }
        }
    }

    // Expected batch 0: contextLen=112, selectIndices=[4,5,6,7]
    int32_t expectedContextLenBatch0 = 112;
    std::vector<int64_t> expectedSelectIndicesBatch0 = {4, 5, 6, 7};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto packedDraftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
        auto tensorPositionIndicesDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectTokenIndicesDevice = rt::Tensor({batchSize, selectTokenLength}, // 2D tensor
            rt::DeviceType::kGPU, DataType::kINT64);
        auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(draftTreeMaskDevice.rawPointer(), inputDraftTreeMask.data(),
            batchSize * paddedDraftTreeSize * paddedDraftTreeSize * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftTreeLengthDevice.rawPointer(), inputDraftTreeLength.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sequenceStartIndicesDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleDraftProposalInputs(draftTreeMaskDevice, draftTreeLengthDevice, sequenceStartIndicesDevice,
            packedDraftTreeMaskDevice, tensorPositionIndicesDevice, selectTokenIndicesDevice,
            sequenceContextLengthsDevice, stream);

        auto const actualContextLengths = copyDeviceToHost<int32_t>(sequenceContextLengthsDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectTokenIndicesDevice);

        // Verify batch 0
        EXPECT_EQ(actualContextLengths[0], expectedContextLenBatch0);
        for (int i = 0; i < selectTokenLength; i++)
        {
            EXPECT_EQ(actualSelectIndices[i], expectedSelectIndicesBatch0[i]);
        }
    }
}

// ============================================================================
// Test 10: assembleDraftTreeInput (subsequent rounds)
// Description: Assemble draft tree input for round > 0
// Select tokens from topK x topK candidates, build tree mask from parent
// ============================================================================
TEST(EagleKernels, AssembleDraftTreeInput)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t paddedDraftTreeSize = 12;
    int32_t draftHiddenDim = 256;
    int32_t round = 1;

    std::vector<int32_t> inputDraftIdTable(8 * 16);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < 16; i++)
        {
            inputDraftIdTable[b * 16 + i] = 200 + b * 100 + i;
        }
    }

    // Select: [0,5,10,15] for batch 0
    std::vector<int32_t> inputSelectedIndices = {
        0, 5, 10, 15, // Batch 0
        1, 6, 11, 14, // Batch 1
        2, 7, 9, 13,  // Batch 2
        3, 4, 8, 12,  // Batch 3
        0, 5, 10, 15, // Batch 4
        1, 6, 11, 14, // Batch 5
        2, 7, 9, 13,  // Batch 6
        3, 4, 8, 12   // Batch 7
    };

    std::vector<half> inputHiddenOutput(8 * draftTopK * draftHiddenDim, __float2half(2.0f));

    // Expected batch 0: inputIds[4:7] = [200,205,210,215]
    std::vector<int32_t> expectedInputIdsRound1Batch0 = {200, 205, 210, 215};
    int32_t expectedTreeLength = 8; // 4 + 4

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftIdTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenOutputDevice
            = rt::Tensor({batchSize * draftTopK, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto selectedIndicesDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto inputIdsDevice = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftHiddenStatesInputDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, draftHiddenDim}, rt::DeviceType::kGPU, DataType::kHALF);
        auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);

        std::vector<int32_t> initialTreeLength(batchSize, draftTopK);
        copyHostToDevice<int32_t>(draftTreeLengthDevice, initialTreeLength);

        CUDA_CHECK(cudaMemcpy(draftIdTableDevice.rawPointer(), inputDraftIdTable.data(),
            batchSize * 16 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftHiddenOutputDevice.rawPointer(), inputHiddenOutput.data(),
            batchSize * draftTopK * draftHiddenDim * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));

        assembleDraftTreeInput(draftIdTableDevice, draftHiddenOutputDevice, selectedIndicesDevice, inputIdsDevice,
            draftHiddenStatesInputDevice, draftTreeLengthDevice, draftTreeMaskDevice, draftTopK, round, stream);

        auto const actualInputIds = copyDeviceToHost<int32_t>(inputIdsDevice);
        auto const actualTreeLength = copyDeviceToHost<int32_t>(draftTreeLengthDevice);

        // Verify batch 0
        EXPECT_EQ(actualTreeLength[0], expectedTreeLength);
        for (int i = 0; i < draftTopK; i++)
        {
            int pos = round * draftTopK + i;
            EXPECT_EQ(actualInputIds[pos], expectedInputIdsRound1Batch0[i]);
        }
    }
}

// ============================================================================
// Test 11: assembleIntermediateData (subsequent rounds)
// Description: Copy cuLogProbs to scores; compute parent pointers in full table
// parent = startOffset + selectedIndices[i]
// ============================================================================
TEST(EagleKernels, AssembleIntermediateData)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t round = 1;

    std::vector<float> inputCuLogProbs = {
        -1.5f, -1.7f, -1.9f, -2.1f,     // Batch 0
        -1.6f, -1.8f, -2.0f, -2.2f,     // Batch 1
        -1.55f, -1.75f, -1.95f, -2.15f, // Batch 2
        -1.65f, -1.85f, -2.05f, -2.25f, // Batch 3
        -1.5f, -1.7f, -1.9f, -2.1f,     // Batch 4
        -1.6f, -1.8f, -2.0f, -2.2f,     // Batch 5
        -1.55f, -1.75f, -1.95f, -2.15f, // Batch 6
        -1.65f, -1.85f, -2.05f, -2.25f  // Batch 7
    };

    std::vector<int32_t> inputSelectedIndices = {
        0, 5, 10, 15, // Batch 0
        1, 6, 11, 14, // Batch 1
        2, 7, 9, 13,  // Batch 2
        3, 4, 8, 12,  // Batch 3
        0, 5, 10, 15, // Batch 4
        1, 6, 11, 14, // Batch 5
        2, 7, 9, 13,  // Batch 6
        3, 4, 8, 12   // Batch 7
    };

    // Expected batch 0: startOffset = 1 + 4 + 16*0 = 5
    // parents = [5+0, 5+5, 5+10, 5+15] = [5, 10, 15, 20]
    std::vector<int32_t> expectedParentsBatch0 = {5, 10, 15, 20};

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto cuLogProbsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto selectedIndicesDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto intermediateScoresDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateParentsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(cuLogProbsDevice.rawPointer(), inputCuLogProbs.data(),
            batchSize * draftTopK * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(selectedIndicesDevice.rawPointer(), inputSelectedIndices.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));

        assembleIntermediateData(cuLogProbsDevice, selectedIndicesDevice, intermediateScoresDevice,
            intermediateParentsDevice, draftTopK, round, stream);

        auto const actualScores = copyDeviceToHost<float>(intermediateScoresDevice);
        auto const actualParents = copyDeviceToHost<int32_t>(intermediateParentsDevice);

        // Verify batch 0
        for (int i = 0; i < draftTopK; i++)
        {
            EXPECT_FLOAT_EQ(actualScores[i], inputCuLogProbs[i]);
            EXPECT_EQ(actualParents[i], expectedParentsBatch0[i]);
        }
    }
}

// ============================================================================
// Test 12: updateDraftTreeFullTables
// Description: Update full table at specific offset with draft results
// dstOffset = 1 + topK + topK^2 * round
// ============================================================================
TEST(EagleKernels, UpdateDraftTreeFullTables)
{
    cudaStream_t stream = nullptr;
    int32_t draftTopK = 4;
    int32_t round = 0;
    int32_t fullTableLength = 21;

    std::vector<int32_t> inputDraftIdTable(8 * 16);
    std::vector<float> inputDraftScoreTable(8 * 16);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < 16; i++)
        {
            inputDraftIdTable[b * 16 + i] = 300 + b * 100 + i;
            inputDraftScoreTable[b * 16 + i] = -1.0f - b * 0.1f - i * 0.05f;
        }
    }

    std::vector<int32_t> inputIntermediateParents(8 * draftTopK);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < draftTopK; i++)
        {
            inputIntermediateParents[b * draftTopK + i] = 5 + i;
        }
    }

    // Expected batch 0: write to offset 5-20 (round 0)
    int32_t dstOffset = 5;

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto draftIdTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreTableDevice = rt::Tensor({batchSize, 16}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto intermediateParentsDevice = rt::Tensor({batchSize, draftTopK}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftIdFullTableDevice = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);
        auto draftScoreFullTableDevice
            = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kFLOAT);
        auto draftParentFullTableDevice
            = rt::Tensor({batchSize, fullTableLength}, rt::DeviceType::kGPU, DataType::kINT32);

        std::vector<int32_t> init_ids(batchSize * fullTableLength, 0);
        std::vector<float> init_scores(batchSize * fullTableLength, -INFINITY);
        std::vector<int32_t> init_parents(batchSize * fullTableLength, -5);

        CUDA_CHECK(cudaMemcpy(draftIdTableDevice.rawPointer(), inputDraftIdTable.data(),
            batchSize * 16 * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(draftScoreTableDevice.rawPointer(), inputDraftScoreTable.data(),
            batchSize * 16 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(intermediateParentsDevice.rawPointer(), inputIntermediateParents.data(),
            batchSize * draftTopK * sizeof(int32_t), cudaMemcpyHostToDevice));
        copyHostToDevice<int32_t>(draftIdFullTableDevice, init_ids);
        copyHostToDevice<float>(draftScoreFullTableDevice, init_scores);
        copyHostToDevice<int32_t>(draftParentFullTableDevice, init_parents);

        updateDraftTreeFullTables(draftIdTableDevice, draftScoreTableDevice, intermediateParentsDevice,
            draftIdFullTableDevice, draftScoreFullTableDevice, draftParentFullTableDevice, draftTopK, round, stream);

        auto const actualIds = copyDeviceToHost<int32_t>(draftIdFullTableDevice);
        auto const actualScores = copyDeviceToHost<float>(draftScoreFullTableDevice);
        auto const actualParents = copyDeviceToHost<int32_t>(draftParentFullTableDevice);

        // Verify batch 0 first few entries
        EXPECT_EQ(actualIds[dstOffset], inputDraftIdTable[0]);
        EXPECT_FLOAT_EQ(actualScores[dstOffset], inputDraftScoreTable[0]);
        EXPECT_EQ(actualParents[dstOffset], inputIntermediateParents[0]);
    }
}

// ============================================================================
// Test 13: prepareEagleBaseTreeDecodingInputs
// Description: Prepare inputs for base model tree decoding (similar to draft proposal)
// ============================================================================
TEST(EagleKernels, PrepareEagleBaseTreeDecodingInputs)
{
    cudaStream_t stream = nullptr;
    int32_t treeSize = 12;
    int32_t packedTreeMaskLen = divUp(treeSize, 32);

    std::vector<int8_t> inputBaseTreeDecodingMask(8 * treeSize * treeSize, 0);
    for (int b = 0; b < 8; b++)
    {
        for (int i = 0; i < treeSize; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                inputBaseTreeDecodingMask[b * treeSize * treeSize + i * treeSize + j] = 1;
            }
        }
    }

    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 300, 400, 150, 250, 350, 450};

    // Expected batch 0: positions=[100,101,...,111], contextLen=112
    std::vector<int32_t> expectedPositionsBatch0(treeSize);
    for (int i = 0; i < treeSize; i++)
    {
        expectedPositionsBatch0[i] = 100 + i;
    }
    int32_t expectedContextLenBatch0 = 112;

    for (int32_t batchSize : {1, 2, 4, 8})
    {
        auto baseTreeDecodingMaskDevice
            = rt::Tensor({batchSize, treeSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto packedBaseTreeDecodingMaskDevice
            = rt::Tensor({batchSize, treeSize, packedTreeMaskLen}, rt::DeviceType::kGPU, DataType::kINT32);
        auto tensorPositionIndicesDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectTokenIndicesDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT64);
        auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemcpy(baseTreeDecodingMaskDevice.rawPointer(), inputBaseTreeDecodingMask.data(),
            batchSize * treeSize * treeSize * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sequenceStartIndicesDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleBaseTreeDecodingInputs(baseTreeDecodingMaskDevice, sequenceStartIndicesDevice,
            packedBaseTreeDecodingMaskDevice, tensorPositionIndicesDevice, selectTokenIndicesDevice,
            sequenceContextLengthsDevice, stream);

        auto const actualPositions = copyDeviceToHost<int32_t>(tensorPositionIndicesDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectTokenIndicesDevice);
        auto const actualContextLengths = copyDeviceToHost<int32_t>(sequenceContextLengthsDevice);

        // Verify batch 0
        EXPECT_EQ(actualContextLengths[0], expectedContextLenBatch0);
        for (int i = 0; i < treeSize; i++)
        {
            EXPECT_EQ(actualPositions[i], expectedPositionsBatch0[i]);
            EXPECT_EQ(actualSelectIndices[i], i);
        }
    }
}

// ============================================================================
// TRT Native kernel tests
// ============================================================================

// ============================================================================
// Test: prepareEaglePrefillInputsTrtNative
// Description: Generate causal attention mask [batch, 1, inputSeqLen, presentLen]
// and sequential position IDs [batch, inputSeqLen] starting from pastLength.
// The mask is lower-triangular for current input tokens and fully attends to
// all past positions.
// ============================================================================
TEST(EagleKernels, PrepareEaglePrefillInputsTrtNative)
{
    cudaStream_t stream = nullptr;

    for (int32_t batchSize : {1, 2, 4})
    {
        int32_t const inputSeqLen = 8;
        // Each batch has a different past length (KV cache length)
        // contextLengths[b] = pastLength[b] + inputSeqLen
        std::vector<int32_t> pastLengths = {0, 10, 5, 20};
        std::vector<int32_t> inputContextLengths(batchSize);
        for (int32_t b = 0; b < batchSize; b++)
        {
            inputContextLengths[b] = pastLengths[b] + inputSeqLen;
        }
        int32_t maxPresentLength = 0;
        for (int32_t b = 0; b < batchSize; b++)
        {
            maxPresentLength = std::max(maxPresentLength, inputContextLengths[b]);
        }

        auto contextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto maskDevice
            = rt::Tensor({batchSize, 1, inputSeqLen, maxPresentLength}, rt::DeviceType::kGPU, DataType::kBOOL);
        auto posIdsDevice = rt::Tensor({batchSize, inputSeqLen}, rt::DeviceType::kGPU, DataType::kINT32);

        // Zero-initialize the mask to avoid stale data
        CUDA_CHECK(cudaMemset(maskDevice.rawPointer(), 0, maskDevice.getMemoryCapacity()));

        copyHostToDevice<int32_t>(contextLengthsDevice, inputContextLengths);

        prepareEaglePrefillInputsTrtNative(contextLengthsDevice, maskDevice, posIdsDevice, stream);

        // Copy results back (use uint8_t instead of bool because std::vector<bool> is a bit-packed proxy)
        auto const actualMask = copyDeviceToHost<uint8_t>(maskDevice);
        auto const actualPosIds = copyDeviceToHost<int32_t>(posIdsDevice);

        for (int32_t b = 0; b < batchSize; b++)
        {
            int32_t const pastLen = pastLengths[b];

            for (int32_t q = 0; q < inputSeqLen; q++)
            {
                // Verify position ID = pastLength + q
                EXPECT_EQ(actualPosIds[b * inputSeqLen + q], pastLen + q)
                    << "Batch " << b << ", query " << q << " position ID mismatch";

                int32_t const maskRowOffset = b * inputSeqLen * maxPresentLength + q * maxPresentLength;

                // All past positions should be attended to
                for (int32_t kv = 0; kv < pastLen; kv++)
                {
                    EXPECT_TRUE(actualMask[maskRowOffset + kv])
                        << "Batch " << b << ", q=" << q << ", kv=" << kv << " should attend to past";
                }

                // Causal mask within current input: token q attends to positions [pastLen .. pastLen+q]
                for (int32_t kv = 0; kv < inputSeqLen; kv++)
                {
                    bool expected = (kv <= q);
                    EXPECT_EQ(actualMask[maskRowOffset + pastLen + kv], expected)
                        << "Batch " << b << ", q=" << q << ", kv(input)=" << kv << " causal mask mismatch";
                }
            }
        }
    }
}

// ============================================================================
// Test: prepareEagleDraftProposalInputsTrtNative
// Description: Convert tree attention mask from [batch, treeSize, treeSize] Int8
// to [batch, 1, treeSize, presentLen] Bool, where past positions are all true
// and tree positions follow the original tree mask. Also verify position IDs
// are computed as sequenceStartIndex + (number of attended nodes - 1).
// ============================================================================
TEST(EagleKernels, PrepareEagleDraftProposalInputsTrtNative)
{
    cudaStream_t stream = nullptr;

    int32_t const paddedDraftTreeSize = 8;
    int32_t const selectTokenLength = 4;

    // Tree mask: a simple chain tree [0->1->2->3] plus branching [0->4, 1->5]
    // Actual tree size = 6 (tokens 6,7 are padding)
    // clang-format off
    std::vector<int8_t> inputDraftTreeMaskBatch0 = {
        1,0,0,0,0,0,0,0,  // token 0 attends to [0]
        1,1,0,0,0,0,0,0,  // token 1 attends to [0,1]
        1,1,1,0,0,0,0,0,  // token 2 attends to [0,1,2]
        1,1,1,1,0,0,0,0,  // token 3 attends to [0,1,2,3]
        1,0,0,0,1,0,0,0,  // token 4 attends to [0,4] (branch from 0)
        1,1,0,0,0,1,0,0,  // token 5 attends to [0,1,5] (branch from 1)
        0,0,0,0,0,0,0,0,  // padding
        0,0,0,0,0,0,0,0,  // padding
    };
    // clang-format on

    std::vector<int32_t> inputDraftTreeLength = {6, 6, 6, 6};
    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 50, 150};

    // Expected position IDs for batch 0 (seqStart=100):
    // token 0: attends to 1 node => pos = 100 + 1 - 1 = 100
    // token 1: attends to 2 nodes => pos = 100 + 2 - 1 = 101
    // token 2: attends to 3 nodes => pos = 100 + 3 - 1 = 102
    // token 3: attends to 4 nodes => pos = 100 + 4 - 1 = 103
    // token 4: attends to 2 nodes => pos = 100 + 2 - 1 = 101
    // token 5: attends to 3 nodes => pos = 100 + 3 - 1 = 102
    // token 6,7: padding => pos = 0
    std::vector<int32_t> expectedPositionsBatch0 = {100, 101, 102, 103, 101, 102, 0, 0};

    for (int32_t batchSize : {1, 2, 4})
    {
        int32_t maxPresentLength = 0;
        for (int32_t b = 0; b < batchSize; b++)
        {
            int32_t presentLen = inputSequenceStartIndices[b] + paddedDraftTreeSize;
            maxPresentLength = std::max(maxPresentLength, presentLen);
        }

        // Replicate batch 0 mask for all batches
        std::vector<int8_t> inputDraftTreeMask;
        for (int32_t b = 0; b < batchSize; b++)
        {
            inputDraftTreeMask.insert(
                inputDraftTreeMask.end(), inputDraftTreeMaskBatch0.begin(), inputDraftTreeMaskBatch0.end());
        }

        auto draftTreeMaskDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        auto draftTreeLengthDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto sequenceStartIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto trtNativeMaskDevice
            = rt::Tensor({batchSize, 1, paddedDraftTreeSize, maxPresentLength}, rt::DeviceType::kGPU, DataType::kBOOL);
        auto positionIndicesDevice
            = rt::Tensor({batchSize, paddedDraftTreeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectTokenIndicesDevice
            = rt::Tensor({batchSize, selectTokenLength}, rt::DeviceType::kGPU, DataType::kINT64);
        auto sequenceContextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemset(trtNativeMaskDevice.rawPointer(), 0, trtNativeMaskDevice.getMemoryCapacity()));
        copyHostToDevice<int8_t>(draftTreeMaskDevice, inputDraftTreeMask);
        CUDA_CHECK(cudaMemcpy(draftTreeLengthDevice.rawPointer(), inputDraftTreeLength.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sequenceStartIndicesDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleDraftProposalInputsTrtNative(draftTreeMaskDevice, draftTreeLengthDevice, sequenceStartIndicesDevice,
            trtNativeMaskDevice, positionIndicesDevice, selectTokenIndicesDevice, sequenceContextLengthsDevice, stream);

        // Copy results
        auto const actualMask = copyDeviceToHost<uint8_t>(trtNativeMaskDevice);
        auto const actualPositions = copyDeviceToHost<int32_t>(positionIndicesDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectTokenIndicesDevice);
        auto const actualContextLengths = copyDeviceToHost<int32_t>(sequenceContextLengthsDevice);

        // Verify batch 0
        int32_t const seqStart0 = inputSequenceStartIndices[0];

        // Check position indices
        for (int32_t i = 0; i < paddedDraftTreeSize; i++)
        {
            EXPECT_EQ(actualPositions[i], expectedPositionsBatch0[i]) << "Batch 0, token " << i << " position mismatch";
        }

        // Check mask: past positions should all be true for valid tokens
        for (int32_t q = 0; q < 6; q++) // Only valid tokens
        {
            int32_t const maskRowOffset = q * maxPresentLength;

            // All past positions should be attended to
            for (int32_t kv = 0; kv < seqStart0; kv++)
            {
                EXPECT_TRUE(actualMask[maskRowOffset + kv])
                    << "Batch 0, q=" << q << ", kv=" << kv << " should attend to past";
            }

            // Tree mask region: match original tree mask
            for (int32_t treeKv = 0; treeKv <= q; treeKv++)
            {
                bool expected = (inputDraftTreeMaskBatch0[q * paddedDraftTreeSize + treeKv] != 0);
                EXPECT_EQ(actualMask[maskRowOffset + seqStart0 + treeKv], expected)
                    << "Batch 0, q=" << q << ", treeKv=" << treeKv << " tree mask mismatch";
            }
        }

        // Check padding tokens (6,7) have all-false mask
        for (int32_t q = 6; q < paddedDraftTreeSize; q++)
        {
            int32_t const maskRowOffset = q * maxPresentLength;
            for (int32_t kv = 0; kv < maxPresentLength; kv++)
            {
                EXPECT_FALSE(actualMask[maskRowOffset + kv])
                    << "Batch 0, padding token q=" << q << ", kv=" << kv << " should be false";
            }
        }

        // Check context lengths: seqStart + paddedDraftTreeSize
        EXPECT_EQ(actualContextLengths[0], seqStart0 + paddedDraftTreeSize);

        // Check select token indices: should be last selectTokenLength tokens of actual tree
        for (int32_t i = 0; i < selectTokenLength; i++)
        {
            EXPECT_EQ(actualSelectIndices[i], 6 - selectTokenLength + i)
                << "Batch 0, select index " << i << " mismatch";
        }

        // Verify batch 1 (if available) to test different sequence start
        if (batchSize >= 2)
        {
            int32_t const seqStart1 = inputSequenceStartIndices[1];
            int32_t const batch1MaskOffset = paddedDraftTreeSize * maxPresentLength;

            // Token 0 of batch 1: attend to all seqStart1 past positions
            for (int32_t kv = 0; kv < seqStart1; kv++)
            {
                EXPECT_TRUE(actualMask[batch1MaskOffset + kv]) << "Batch 1, q=0, kv=" << kv << " should attend to past";
            }
            // Token 0 attends to itself in tree region
            EXPECT_TRUE(actualMask[batch1MaskOffset + seqStart1]);
            // Token 0 does NOT attend to token 1 in tree region
            EXPECT_FALSE(actualMask[batch1MaskOffset + seqStart1 + 1]);

            EXPECT_EQ(actualContextLengths[1], seqStart1 + paddedDraftTreeSize);
        }
    }
}

// ============================================================================
// Test: prepareEagleAcceptDecodeTokenInputsTrtNative
// Description: Generate causal mask [batch, 1, maxAcceptedTokenNum, presentLen]
// for accepted tokens. Valid tokens get causal attention, padding tokens get
// all-false mask. Position IDs are sequential from sequenceStartIndex.
// ============================================================================
TEST(EagleKernels, PrepareEagleAcceptDecodeTokenInputsTrtNative)
{
    cudaStream_t stream = nullptr;

    int32_t const maxAcceptedTokenNum = 8;
    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 50, 150};
    std::vector<int32_t> inputAcceptedTokenNums = {5, 3, 7, 4};

    for (int32_t batchSize : {1, 2, 4})
    {
        int32_t maxPresentLength = 0;
        for (int32_t b = 0; b < batchSize; b++)
        {
            int32_t presentLen = inputSequenceStartIndices[b] + inputAcceptedTokenNums[b];
            maxPresentLength = std::max(maxPresentLength, presentLen);
        }

        auto seqStartDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto acceptNumsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto maskDevice
            = rt::Tensor({batchSize, 1, maxAcceptedTokenNum, maxPresentLength}, rt::DeviceType::kGPU, DataType::kBOOL);
        auto posIdsDevice = rt::Tensor({batchSize, maxAcceptedTokenNum}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectIndicesDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT64);
        auto contextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemset(maskDevice.rawPointer(), 0, maskDevice.getMemoryCapacity()));
        CUDA_CHECK(cudaMemcpy(seqStartDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(acceptNumsDevice.rawPointer(), inputAcceptedTokenNums.data(), batchSize * sizeof(int32_t),
            cudaMemcpyHostToDevice));

        prepareEagleAcceptDecodeTokenInputsTrtNative(seqStartDevice, acceptNumsDevice, maskDevice, posIdsDevice,
            selectIndicesDevice, contextLengthsDevice, stream);

        auto const actualMask = copyDeviceToHost<uint8_t>(maskDevice);
        auto const actualPosIds = copyDeviceToHost<int32_t>(posIdsDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectIndicesDevice);
        auto const actualContextLengths = copyDeviceToHost<int32_t>(contextLengthsDevice);

        for (int32_t b = 0; b < batchSize; b++)
        {
            int32_t const seqStart = inputSequenceStartIndices[b];
            int32_t const acceptNum = inputAcceptedTokenNums[b];
            int32_t const batchMaskOffset = b * maxAcceptedTokenNum * maxPresentLength;

            for (int32_t q = 0; q < maxAcceptedTokenNum; q++)
            {
                int32_t const rowOffset = batchMaskOffset + q * maxPresentLength;

                if (q < acceptNum)
                {
                    // Valid token: position ID = seqStart + q
                    EXPECT_EQ(actualPosIds[b * maxAcceptedTokenNum + q], seqStart + q)
                        << "Batch " << b << ", token " << q << " position ID mismatch";

                    // Attend to all past positions
                    for (int32_t kv = 0; kv < seqStart; kv++)
                    {
                        EXPECT_TRUE(actualMask[rowOffset + kv])
                            << "Batch " << b << ", q=" << q << ", kv=" << kv << " should attend to past";
                    }

                    // Causal mask within accepted tokens: attend to [0..q]
                    for (int32_t acceptKv = 0; acceptKv <= q; acceptKv++)
                    {
                        EXPECT_TRUE(actualMask[rowOffset + seqStart + acceptKv])
                            << "Batch " << b << ", q=" << q << ", acceptKv=" << acceptKv << " should attend";
                    }

                    // Mask future positions within accepted tokens
                    for (int32_t acceptKv = q + 1; acceptKv < acceptNum; acceptKv++)
                    {
                        EXPECT_FALSE(actualMask[rowOffset + seqStart + acceptKv])
                            << "Batch " << b << ", q=" << q << ", acceptKv=" << acceptKv << " should NOT attend";
                    }
                }
                else
                {
                    // Padding token: all-false mask
                    for (int32_t kv = 0; kv < maxPresentLength; kv++)
                    {
                        EXPECT_FALSE(actualMask[rowOffset + kv])
                            << "Batch " << b << ", padding q=" << q << ", kv=" << kv << " should be false";
                    }
                }
            }

            // Select index: last accepted token (acceptNum - 1)
            EXPECT_EQ(actualSelectIndices[b], acceptNum - 1) << "Batch " << b << " select index mismatch";

            // Context length: seqStart + maxAcceptedTokenNum
            EXPECT_EQ(actualContextLengths[b], seqStart + maxAcceptedTokenNum)
                << "Batch " << b << " context length mismatch";
        }
    }
}

// ============================================================================
// Test: prepareEagleBaseTreeDecodingInputsTrtNative
// Description: Convert tree attention mask to TRT native boolean mask for base
// model tree decoding. Same semantics as draft proposal but without draft tree
// length (all tokens are valid). Verifies mask, positions, select indices, and
// context lengths.
// ============================================================================
TEST(EagleKernels, PrepareEagleBaseTreeDecodingInputsTrtNative)
{
    cudaStream_t stream = nullptr;

    int32_t const treeSize = 8;

    // Tree mask: full tree (same structure as draft proposal test)
    // clang-format off
    std::vector<int8_t> inputTreeMaskBatch0 = {
        1,0,0,0,0,0,0,0,  // token 0: [0]
        1,1,0,0,0,0,0,0,  // token 1: [0,1]
        1,1,1,0,0,0,0,0,  // token 2: [0,1,2]
        1,1,1,1,0,0,0,0,  // token 3: [0,1,2,3]
        1,0,0,0,1,0,0,0,  // token 4: [0,4]
        1,1,0,0,0,1,0,0,  // token 5: [0,1,5]
        1,0,0,0,1,0,1,0,  // token 6: [0,4,6]
        1,1,0,0,0,1,0,1,  // token 7: [0,1,5,7]
    };
    // clang-format on

    std::vector<int32_t> inputSequenceStartIndices = {100, 200, 50, 150};

    // Expected positions for batch 0 (seqStart=100):
    // token 0: 1 attended => 100
    // token 1: 2 attended => 101
    // token 2: 3 attended => 102
    // token 3: 4 attended => 103
    // token 4: 2 attended => 101
    // token 5: 3 attended => 102
    // token 6: 3 attended => 102
    // token 7: 4 attended => 103
    std::vector<int32_t> expectedPositionsBatch0 = {100, 101, 102, 103, 101, 102, 102, 103};

    for (int32_t batchSize : {1, 2, 4})
    {
        int32_t maxPresentLength = 0;
        for (int32_t b = 0; b < batchSize; b++)
        {
            int32_t presentLen = inputSequenceStartIndices[b] + treeSize;
            maxPresentLength = std::max(maxPresentLength, presentLen);
        }

        std::vector<int8_t> inputTreeMask;
        for (int32_t b = 0; b < batchSize; b++)
        {
            inputTreeMask.insert(inputTreeMask.end(), inputTreeMaskBatch0.begin(), inputTreeMaskBatch0.end());
        }

        auto treeMaskDevice = rt::Tensor({batchSize, treeSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT8);
        auto seqStartDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto maskDevice = rt::Tensor({batchSize, 1, treeSize, maxPresentLength}, rt::DeviceType::kGPU, DataType::kBOOL);
        auto posIdsDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT32);
        auto selectIndicesDevice = rt::Tensor({batchSize, treeSize}, rt::DeviceType::kGPU, DataType::kINT64);
        auto contextLengthsDevice = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);

        CUDA_CHECK(cudaMemset(maskDevice.rawPointer(), 0, maskDevice.getMemoryCapacity()));
        CUDA_CHECK(cudaMemcpy(treeMaskDevice.rawPointer(), inputTreeMask.data(),
            batchSize * treeSize * treeSize * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(seqStartDevice.rawPointer(), inputSequenceStartIndices.data(),
            batchSize * sizeof(int32_t), cudaMemcpyHostToDevice));

        prepareEagleBaseTreeDecodingInputsTrtNative(treeMaskDevice, seqStartDevice, maskDevice, posIdsDevice,
            selectIndicesDevice, contextLengthsDevice, stream);

        auto const actualMask = copyDeviceToHost<uint8_t>(maskDevice);
        auto const actualPositions = copyDeviceToHost<int32_t>(posIdsDevice);
        auto const actualSelectIndices = copyDeviceToHost<int64_t>(selectIndicesDevice);
        auto const actualContextLengths = copyDeviceToHost<int32_t>(contextLengthsDevice);

        // Verify batch 0
        int32_t const seqStart0 = inputSequenceStartIndices[0];

        for (int32_t i = 0; i < treeSize; i++)
        {
            EXPECT_EQ(actualPositions[i], expectedPositionsBatch0[i]) << "Batch 0, token " << i << " position mismatch";
        }

        // Verify mask for batch 0
        for (int32_t q = 0; q < treeSize; q++)
        {
            int32_t const maskRowOffset = q * maxPresentLength;

            // Past positions: all attended
            for (int32_t kv = 0; kv < seqStart0; kv++)
            {
                EXPECT_TRUE(actualMask[maskRowOffset + kv])
                    << "Batch 0, q=" << q << ", kv=" << kv << " should attend to past";
            }

            // Tree region: follow original tree mask
            for (int32_t treeKv = 0; treeKv <= q; treeKv++)
            {
                bool expected = (inputTreeMaskBatch0[q * treeSize + treeKv] != 0);
                EXPECT_EQ(actualMask[maskRowOffset + seqStart0 + treeKv], expected)
                    << "Batch 0, q=" << q << ", treeKv=" << treeKv << " tree mask mismatch";
            }

            // Future tree positions: all false
            for (int32_t treeKv = q + 1; treeKv < treeSize; treeKv++)
            {
                EXPECT_FALSE(actualMask[maskRowOffset + seqStart0 + treeKv])
                    << "Batch 0, q=" << q << ", future treeKv=" << treeKv << " should be false";
            }
        }

        // Check context length and select indices for all batches
        for (int32_t b = 0; b < batchSize; b++)
        {
            EXPECT_EQ(actualContextLengths[b], inputSequenceStartIndices[b] + treeSize)
                << "Batch " << b << " context length mismatch";

            // All tree tokens are selected (select indices = [0, 1, ..., treeSize-1])
            for (int32_t i = 0; i < treeSize; i++)
            {
                EXPECT_EQ(actualSelectIndices[b * treeSize + i], i)
                    << "Batch " << b << ", select index " << i << " mismatch";
            }
        }
    }
}
