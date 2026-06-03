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

#include "common/tensor.h"
#include "runtime/exec/tensorMap.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Encapsulates the two-mode binding for Qwen3-VL / Qwen3-Omni deepstack
//! engine inputs.
//!
//! The engine graph elementwise-adds `deepstack_embeds_%d` to `hidden_states`
//! inside every decoder-layer forward, so *something* must be bound each call
//! even when the request has no vision features. This class owns that swap:
//!
//!   * `useRealFeatures(map)` binds `io.deepstackEmbeds[i]` (real per-request
//!     features, populated by the embedding preprocessor on prefill).
//!   * `useZeroTarget(map)` binds a shared zero buffer owned by
//!     `SharedResources::zeroBuffer`. The buffer is sized to the worst-case
//!     resolved shape (`{maxBatch, maxDeepstackSeqLen, hiddenSize}` HALF) so
//!     TRT's read falls within the allocation regardless of the per-step
//!     `batch` / `seqLen` resolved from `InferenceDims`. Zero contents make the
//!     engine's `hidden_states + deepstack` elementwise add a no-op.
//!
//! The spec runtime speaks intent (verbs), never tensor names. Name templating
//! (`deepstack_embeds_0`, `_1`, ...) stays inside this class.
class DeepstackBinding
{
public:
    //! Construct, capturing references to the per-request real-feature buffers
    //! and the shared zero target tensor. Both references must outlive every
    //! `useRealFeatures` / `useZeroTarget` call.
    DeepstackBinding(std::vector<Tensor>& realBuffers, Tensor& zeroTarget);

    //! Bind each deepstack_embeds_%d entry to the corresponding real-feature
    //! buffer. Call before base prefill.
    void useRealFeatures(TensorMap& map);

    //! Bind every deepstack_embeds_%d entry to the shared zero target tensor.
    //! Call before every non-prefill engine execute on the base side
    //! (vanilla decode, tree verify, CUDA-graph capture).
    void useZeroTarget(TensorMap& map);

    //! Enumerate every binding name this feature owns. Used by TensorMap
    //! validation to assert: every map entry is covered by either the
    //! TensorRegistry, a MutableBinding, or LoRA.
    std::vector<std::string> ownedNames() const;

    //! Diagnostic: current mode as a human-readable string.
    std::string currentModeName() const;

    //! Number of deepstack features (== `cfg.numDeepstackFeatures`).
    int32_t numFeatures() const noexcept
    {
        return static_cast<int32_t>(mRealBuffers->size());
    }

private:
    enum class Mode
    {
        kReal,
        kZero,
    };

    std::vector<Tensor>* mRealBuffers; //!< Owned by PipelineIO; one tensor per feature.
    Tensor* mZeroTarget;               //!< Owned by SharedResources::zeroBuffer.
    Mode mMode{Mode::kZero};           //!< Initial state matches `buildTensorMap`'s initial bind.
};

} // namespace rt
} // namespace trt_edgellm
