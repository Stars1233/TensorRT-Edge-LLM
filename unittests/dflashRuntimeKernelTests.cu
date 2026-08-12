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
#include "kernels/speculative/dflashRuntimeKernels.h"
#include "runtime/state/kvPageTable.h"
#include "testUtils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

TEST(DFlashRuntimeKernels, CheckPageTableIdentityAcceptsIdentityRows)
{
    cudaStream_t stream{nullptr};
    int32_t const maxBatch = 3;
    int32_t const maxPagesPerSeq = 4;
    rt::KVPageTable pageTable(maxBatch, maxPagesPerSeq, maxBatch * maxPagesPerSeq);
    pageTable.setIdentity();
    pageTable.upload(stream);

    for (int32_t b = 0; b < maxBatch; ++b)
    {
        EXPECT_NO_THROW(kernel::checkDFlashPageTableIdentity(pageTable.hostRow(b), b, maxPagesPerSeq));
    }
}

TEST(DFlashRuntimeKernels, CheckPageTableIdentityRejectsScrambledRow)
{
    int32_t const maxPagesPerSeq = 4;
    // Slot 1's identity range would be [4, 8); scramble one entry.
    std::vector<int32_t> const scrambledRow = {4, 5, 9, 7};
    EXPECT_THROW(
        kernel::checkDFlashPageTableIdentity(scrambledRow.data(), /*slot=*/1, maxPagesPerSeq), std::runtime_error);
}

TEST(DFlashRuntimeKernels, CheckPageTableIdentityRejectsUnallocatedTail)
{
    int32_t const maxPagesPerSeq = 4;
    // Slot 0's identity range is [0, 4); a short row with an unallocated tail is not identity either.
    std::vector<int32_t> const shortRow = {0, 1, -1, -1};
    EXPECT_THROW(kernel::checkDFlashPageTableIdentity(shortRow.data(), /*slot=*/0, maxPagesPerSeq), std::runtime_error);
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityAcceptsNonPageAlignedCapacity)
{
    // maxKVCacheCapacity=4000 (not a multiple of 128) -> capPadded=4096. This must not throw.
    EXPECT_NO_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/4000, /*kvCapacity=*/4096));
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityAcceptsExactlyPageAlignedCapacity)
{
    EXPECT_NO_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/4096, /*kvCapacity=*/4096));
}

TEST(DFlashRuntimeKernels, CheckRopeCapacityRejectsSeqLenExceedingCap)
{
    // A rope cache sized past the KV pool's padded capacity indicates a genuine mismatch.
    EXPECT_THROW(kernel::checkDFlashRopeCapacity(/*cosSinSeqLen=*/5000, /*kvCapacity=*/4096), std::runtime_error);
}

TEST(DFlashRuntimeKernels, BuildLinearVerifyInputsUsesDraftStrideForBatchRows)
{
    cudaStream_t stream = nullptr;
    constexpr int32_t batchSize = 2;
    constexpr int32_t dflashBlockSize = 4;
    constexpr int32_t proposalLen = dflashBlockSize - 1;
    constexpr int32_t verifySize = proposalLen + 1;

    auto lastAcceptedTokens = rt::Tensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto draftTokenIds = rt::Tensor({batchSize, dflashBlockSize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto verifyTokenIds = rt::Tensor({batchSize, verifySize}, rt::DeviceType::kGPU, DataType::kINT32);
    auto verifyTreeMask = rt::Tensor({batchSize, verifySize, verifySize}, rt::DeviceType::kGPU, DataType::kINT8);

    copyHostToDevice<int32_t>(lastAcceptedTokens, {10, 20});
    // DFlash draft output at position 0 predicts the current token (t_last), not the next token.
    // Real draft proposals start at position 1 — consistent with DDTree which skips depthIdx==0.
    // Layout per batch row: [<pos0: unused t_last prediction>, pos1, pos2, pos3]
    copyHostToDevice<int32_t>(draftTokenIds, {999, 101, 102, 103, 999, 201, 202, 203});

    kernel::launchDFlashBuildLinearVerifyInputs(lastAcceptedTokens.dataPointer<int32_t>(),
        draftTokenIds.dataPointer<int32_t>(), verifyTokenIds.dataPointer<int32_t>(),
        verifyTreeMask.dataPointer<int8_t>(), batchSize, proposalLen, dflashBlockSize, verifySize, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(copyDeviceToHost<int32_t>(verifyTokenIds), (std::vector<int32_t>{10, 101, 102, 103, 20, 201, 202, 203}));

    std::vector<int8_t> expectedMask;
    expectedMask.reserve(static_cast<size_t>(batchSize) * verifySize * verifySize);
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (int32_t rowIdx = 0; rowIdx < verifySize; ++rowIdx)
        {
            for (int32_t colIdx = 0; colIdx < verifySize; ++colIdx)
            {
                expectedMask.push_back(colIdx <= rowIdx ? int8_t{1} : int8_t{0});
            }
        }
    }
    EXPECT_EQ(copyDeviceToHost<int8_t>(verifyTreeMask), expectedMask);
}
