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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Typed symbolic dimension values resolved per inference step.
//!
//! Fields have NO in-class defaults — every construction site must set every
//! field explicitly. The canonical construction path is an `LLMEngineConfig`
//! recipe method (`prefillDims`, `decodeDims`, `specVerifyDims`,
//! `proposalDims`, `acceptDims`, `resetDims`). Direct construction (aggregate
//! or designated initializers) is supported for unit tests.
//!
//! The nine fields are the complete set of symbolic dims used by LLM and SpecDecode
//! draft engines. Fixed-shape tensor dims do not appear here.
struct InferenceDims
{
    int64_t batch;     //!< Active batch size
    int64_t seqLen;    //!< Work-unit length for this step (prompt / proposal / accept / 1)
    int64_t kvLen;     //!< KV cache capacity (usually LLMEngineConfig::maxKVCacheCapacity)
    int64_t selectLen; //!< last_token_ids select count (1 except for SpecDecode verification)
    //! Effective sequence length for the SpecDecode attention_mask / attention_pos_id
    //! tensors. This is decoupled from `seqLen` because the base engine's
    //! attention plugin treats a "small" mask shape ([B, 1, 1]) as a signal to
    //! use standard causal attention, while a proposal-shaped mask triggers
    //! proposal attention and reads the buffer contents as a bit-packed mask.
    //!
    //! Set to 1 for prefill / decode / reset (engine applies standard causal
    //! attention and ignores the dummy mask buffer); set to the effective proposal
    //! size for verify / proposal / accept (engine applies proposal attention
    //! using the prepared bit-packed mask).
    int64_t attnMaskSeqLen;
    int64_t ropeBatch;     //!< RoPE broadcast dim (1 for non-MRope; batch for MRope)
    int64_t packedMaskLen; //!< divUp(attnMaskSeqLen, 32) for SpecDecode masks; else 1
    //! Shape length for `kvcache_start_index`. Zero is the engine's sentinel
    //! for "initial prefill of an empty KV cache"; `batch` means "use these
    //! per-batch start offsets" for chunked prefill, decode, verify, and
    //! accept. Zero is a legitimate, engine-meaningful value for this dim.
    int64_t startIndexLen;
    //! Shape length for `spec_verify_phase_marker`. Zero means normal
    //! prefill/decode; one means speculative verification. The plugin reads
    //! this shape, not the marker payload.
    int64_t specVerifyPhaseLen;
};

//! Tripwires: if `InferenceDims` gains, loses, or reorders a field, these asserts fire
//! and force a visit to `kDimNames` (diagnostics), `toString` (formatting), and
//! the six recipe methods on `LLMEngineConfig`. The size assert catches add/remove;
//! the per-field `offsetof` asserts catch reorders (which would otherwise silently
//! change the meaning of every positional aggregate init). Note: these do NOT catch
//! "short" aggregate inits (omitting trailing fields) — the policy is that production
//! construction goes through recipe methods, which always set every field.
static_assert(sizeof(InferenceDims) == 9 * sizeof(int64_t),
    "InferenceDims layout changed: update kDimNames, toString(), kZeroAllowedMembers, and every recipe "
    "method in LLMEngineConfig (prefillDims / decodeDims / specVerifyDims / "
    "proposalDims / acceptDims / resetDims).");
static_assert(offsetof(InferenceDims, batch) == 0 * sizeof(int64_t), "InferenceDims::batch reordered");
static_assert(offsetof(InferenceDims, seqLen) == 1 * sizeof(int64_t), "InferenceDims::seqLen reordered");
static_assert(offsetof(InferenceDims, kvLen) == 2 * sizeof(int64_t), "InferenceDims::kvLen reordered");
static_assert(offsetof(InferenceDims, selectLen) == 3 * sizeof(int64_t), "InferenceDims::selectLen reordered");
static_assert(
    offsetof(InferenceDims, attnMaskSeqLen) == 4 * sizeof(int64_t), "InferenceDims::attnMaskSeqLen reordered");
static_assert(offsetof(InferenceDims, ropeBatch) == 5 * sizeof(int64_t), "InferenceDims::ropeBatch reordered");
static_assert(offsetof(InferenceDims, packedMaskLen) == 6 * sizeof(int64_t), "InferenceDims::packedMaskLen reordered");
static_assert(offsetof(InferenceDims, startIndexLen) == 7 * sizeof(int64_t), "InferenceDims::startIndexLen reordered");
static_assert(
    offsetof(InferenceDims, specVerifyPhaseLen) == 8 * sizeof(int64_t), "InferenceDims::specVerifyPhaseLen reordered");

namespace detail
{
//! Compile-time mapping from member pointer to human-readable name.
//!
//! Diagnostics-only (log messages, assertions). Never used on the bind-time
//! hot path. Linear-scan over the small field set is fine for that use.
//!
//! The array size is derived from `InferenceDims` layout so that an extra or
//! missing entry here fails to compile alongside the `sizeof(InferenceDims)`
//! `static_assert` above.
inline constexpr std::array<std::pair<int64_t InferenceDims::*, std::string_view>,
    sizeof(InferenceDims) / sizeof(int64_t)>
    kDimNames = {{
        {&InferenceDims::batch, "batch"},
        {&InferenceDims::seqLen, "seq_len"},
        {&InferenceDims::kvLen, "kv_len"},
        {&InferenceDims::selectLen, "select_len"},
        {&InferenceDims::attnMaskSeqLen, "attn_seq_len"},
        {&InferenceDims::ropeBatch, "rope_batch"},
        {&InferenceDims::packedMaskLen, "packed_mask_len"},
        {&InferenceDims::startIndexLen, "start_index_len"},
        {&InferenceDims::specVerifyPhaseLen, "spec_verify_phase_len"},
    }};

//! Members where `0` is a legitimate engine-meaningful value (not a recipe
//! bypass). `firstInvalidMember` excludes these from the `> 0` positivity
//! check. Keep this set as small as possible — default validation should be
//! strict, and most dims (batch, seqLen, kvLen, etc.) must be > 0.
inline constexpr std::array<int64_t InferenceDims::*, 2> kZeroAllowedMembers{
    &InferenceDims::startIndexLen,
    &InferenceDims::specVerifyPhaseLen,
};
} // namespace detail

//! Return the human-readable name for a `InferenceDims` member pointer.
//!
//! Returns an empty `string_view` if the pointer is not a member of
//! `InferenceDims` (should not happen in practice — the pointer type restricts
//! the input at compile time).
//!
//! @param member Pointer-to-member — e.g. `&InferenceDims::batch`
//! @return Name corresponding to the member (e.g. "batch")
std::string_view dimName(int64_t InferenceDims::* member);

//! Format a `InferenceDims` as a human-readable string.
//!
//! Returns an owning `std::string`, not a `string_view`, because the formatted
//! output has no persistent backing storage.
//!
//! @param dims The value to format
//! @return A string like `{batch=4, seq_len=128, kv_len=4096, ...}`
std::string toString(InferenceDims const& dims);

//! Return the first referenced member whose value is <= 0, or nullptr if all
//! referenced members are positive.
//!
//! Used by `EngineExecutor::prepare` to catch callers who bypassed a recipe method
//! and left some fields unset. Pure function: no TRT or CUDA dependency, so
//! it is directly unit-testable without an `IExecutionContext`.
//!
//! The scope of this check is restricted to registry-referenced members so
//! that engines which do not use every dim (e.g. a model without a packed
//! attention mask) are not forced to set values they do not consume.
//!
//! @param dims The values to validate
//! @param referenced The set of members referenced by the registry — typically
//!                   returned by `TensorRegistry::referencedMembers()`
//! @return The first member-pointer with an invalid value, or nullptr on success
int64_t InferenceDims::* firstInvalidMember(
    InferenceDims const& dims, std::vector<int64_t InferenceDims::*> const& referenced);

//! A single dimension in a tensor shape — either fixed or symbolic.
//!
//! When @c symbol is non-null the dimension is symbolic and its value is
//! resolved by dereferencing `inferenceDims.*symbol` at bind time. Otherwise the
//! fixed @c value is used as-is.
struct ShapeDim
{
    int64_t InferenceDims::* symbol{nullptr}; //!< Non-null ⇒ symbolic; pointer-to-member of InferenceDims
    int64_t value{0};                         //!< Fixed value when symbol is null

    //! @brief Return true when this dimension is symbolic.
    bool isSymbolic() const noexcept
    {
        return symbol != nullptr;
    }
};

//! Helper: create a symbolic ShapeDim from a pointer-to-member.
inline ShapeDim sym(int64_t InferenceDims::* p) noexcept
{
    return ShapeDim{p, 0};
}

//! Helper: create a fixed ShapeDim.
inline ShapeDim fixed(int64_t v) noexcept
{
    return ShapeDim{nullptr, v};
}

} // namespace rt
} // namespace trt_edgellm
