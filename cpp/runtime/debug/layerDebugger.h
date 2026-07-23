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

#include "common/tensor.h"
#include "runtime/hybridCacheManager.h"

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! @brief Per-request debug dumper for 4-layer numeric validation.
//!
//! When both environment variables below are set, the LLM runtime dumps, for
//! every inference round (round 0 = prefill, round r = decode step r), the
//! last-token logits, the per-layer combined KV cache (only the layers named
//! in the env var), the per-sequence valid lengths, and the per-round generated
//! token ids. Everything is buffered across rounds and written as a single
//! safetensors file at request end so the comparison tool can reconcile it
//! against the PyTorch golden.
//!
//!   EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS  number of leading decoder layers k
//!                                       -> dumps layers 0..k-1.
//!   EDGELLM_DUMP_LOGITS_KVCACHE_DIR     output directory for the dump file.
//!
//! The two variables are XOR-coupled: setting exactly one is an error.
//!
//! Per-layer tensors are dumped full-length over the active-batch prefix (no truncation here);
//! the comparison tool slices each sequence to its valid length in PyTorch using the dumped
//! context_lengths.
//!
//! Safetensors layout (single file, all rounds):
//!   round_{r}.logits               [activeBatch, vocab]                          (native dtype)
//!   round_{r}.layer_{i}.kv         [activeBatch, 2, kvHeads, maxSeqLen, headDim] (native dtype)
//!                                  dim 1: 0 = key, 1 = value
//!   round_{r}.context_lengths      [activeBatch]                                 int32
//!   round_{r}.generated_token_ids  [activeBatch]                                 int32
//!
//! Scope (POC): base model + vanilla decoding only. Hooked from
//! ``runBaseModelPrefill`` (round 0) and ``VanillaDecoder::decodeStep``.
//!
//! Optionally also drives teacher-forcing: when ``EDGELLM_FORCE_TOKENS_FILE`` is set the
//! dumper overrides each step's sampled token with the golden's (see applyForcedTokens()),
//! so the run follows the golden token-for-token. Only ever active alongside a dump.
class LayerDebugger
{
public:
    //! @brief Build a dumper from the environment, or return nullptr when disabled.
    //! @return A dumper if both env vars are set; nullptr if neither is set.
    //! @throws std::runtime_error if exactly one of the two env vars is set (XOR
    //!         violation) or the layer spec is empty / malformed.
    static std::unique_ptr<LayerDebugger> fromEnv();

    //! @brief Accumulate one round's tensors into the in-memory buffer.
    //!
    //! Synchronises @p stream first, so the KV cache and logits are final.
    //! @param cacheManager       Base-model KV cache manager.
    //! @param logits             Device logits tensor [activeBatch, vocab].
    //! @param validLengths       Per-sequence valid KV/sequence length this round.
    //! @param generatedTokenIds  Host int32 [activeBatch] tokens sampled this round
    //!                           (may be nullptr to skip).
    //! @param activeBatchSize    Number of active sequences this round.
    //! @param stream             CUDA stream.
    void dumpRound(HybridCacheManager& cacheManager, Tensor const& logits, std::vector<int32_t> const& validLengths,
        int32_t const* generatedTokenIds, int32_t activeBatchSize, cudaStream_t stream);

    //! @brief Write all buffered rounds to a single safetensors file.
    //! @param stream CUDA stream (forwarded to the safetensors writer).
    void flush(cudaStream_t stream);

    //! @brief Teacher-forcing: overwrite each active sequence's sampled token with the forced
    //! one for this step. No-op unless ``EDGELLM_FORCE_TOKENS_FILE`` was set at construction.
    //!
    //! Call *after* ``dumpRound`` so the dump still records the model's own sampled token; the
    //! forced token (if any) is what the caller then commits. This decouples the numeric
    //! comparison from greedy argmax stability — a near-tie argmax flip no longer diverges the
    //! two sides, while the dump still surfaces where the runtime *would* have diverged.
    //! @param genLengths      Per-sequence count of tokens generated so far (== the index to force).
    //! @param tokenIds        Host array [activeBatchSize] of sampled tokens, overwritten in place.
    //! @param activeBatchSize Number of active sequences.
    void applyForcedTokens(std::vector<int32_t> const& genLengths, int32_t* tokenIds, int32_t activeBatchSize);

private:
    LayerDebugger(std::set<int32_t> layers, std::string dir, std::vector<std::vector<int32_t>> forcedTokens);

    //! @brief Read per-sequence forced token ids from ``EDGELLM_FORCE_TOKENS_FILE`` (one line per
    //! sequence, whitespace-separated ids), or an empty vector when the env var is unset. Emits a
    //! warning when forcing is enabled, since it overrides the model's own sampled tokens.
    static std::vector<std::vector<int32_t>> readForcedTokensFromEnv();

    std::set<int32_t> mLayers;                       //!< Absolute decoder-layer indices to dump KV for.
    std::string mDir;                                //!< Output directory.
    int32_t mRequestIdx{0};                          //!< Per-process request index (unique filenames).
    int32_t mRound{0};                               //!< Next round index to assign.
    std::vector<Tensor> mTensors;                    //!< Accumulated tensors across rounds.
    std::vector<std::vector<int32_t>> mForcedTokens; //!< Teacher-forcing tokens (empty = disabled).
};

} // namespace rt
} // namespace trt_edgellm
