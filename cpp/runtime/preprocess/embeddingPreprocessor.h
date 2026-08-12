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
#include "runtime/config/llmEngineConfig.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/state/pipelineIO.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace rt
{

//! Reusable preprocessor that wraps all embedding-lookup kernel calls.
//!
//! Given token IDs (and optional multimodal embeddings) it writes dense
//! vectors into `PipelineIO::inputsEmbeds` (and deepstack slots when
//! applicable).  The class is intentionally stateless beyond the
//! configuration references so that it can be shared across prefill /
//! decode / system-prompt-cache paths.
class EmbeddingPreprocessor
{
public:
    //! Construct with the embedding table data and engine configuration.
    //!
    //! Both references must outlive the preprocessor.
    EmbeddingPreprocessor(EmbeddingData const& embedding, LLMEngineConfig const& config);

    //! Embed token IDs into dense vectors, optionally inserting multimodal embeddings.
    //!
    //! All requests go through the single `kernel::embeddingLookup`. For a vision and/or audio
    //! request the image/audio embeddings are inserted at the imageTokenId / audioTokenId
    //! positions; for a pure-text request the same kernel performs a plain table lookup.
    //!
    //! The multimodal indices are generated on-device directly from the GPU token IDs (no host
    //! round-trip: zero D2H/H2D), cached in `mMultimodalIndices`, and reused by a subsequent
    //! `prepareDeepstack`/`assembleDeepstack` for the same tokens.
    //!
    //! @param tokenIds     GPU tensor of token IDs [batchSize, seqLen].
    //! @param visionEmbeds Optional vision (image) embeddings.
    //! @param audioEmbeds  Optional audio embeddings.
    //! @param io           Pipeline I/O – `inputsEmbeds` is written.
    //! @param stream       CUDA stream for execution.
    void embed(Tensor const& tokenIds, OptionalInputTensor visionEmbeds, OptionalInputTensor audioEmbeds,
        PipelineIO& io, cudaStream_t stream);

    //! Assemble deepstack features at image placeholder positions.
    //!
    //! For each feature in @p features, calls `kernel::assembleDeepstackEmbedding`
    //! and writes into the corresponding slot in `io.deepstackEmbeds`.
    //!
    //! @param tokenIds   GPU tensor of token IDs [batchSize, seqLen].
    //! @param features   Vector of deepstack feature tensors from the vision runner.
    //! @param io         Pipeline I/O – `deepstackEmbeds[i]` is written.
    //! @param stream     CUDA stream for execution.
    //! @return Vector of const references suitable for engine binding.
    OptionalInputTensors assembleDeepstack(
        Tensor const& tokenIds, OptionalInputTensors const& features, PipelineIO& io, cudaStream_t stream);

    //! Prepare deepstack slots for the current step.
    //!
    //! Encapsulates the "config has deepstack, features present or missing?" policy so the
    //! runtime does not need to gate on `numDeepstackFeatures`:
    //!   - no-op when `mConfig.numDeepstackFeatures == 0` (non-VLM engine);
    //!   - assembles real features via `assembleDeepstack` when @p features is non-empty;
    //!   - zero-fills `io.deepstackEmbeds[idx]` otherwise (text-only request on a VLM
    //!     engine so the engine reads known-zero bytes).
    //!
    //! @param tokenIds   GPU tensor of token IDs [batchSize, seqLen].
    //! @param features   Vector of deepstack feature tensors from the vision runner (may be empty).
    //! @param io         Pipeline I/O – `io.deepstackEmbeds[idx]` is written or zeroed.
    //! @param stream     CUDA stream for execution.
    void prepareDeepstack(
        Tensor const& tokenIds, OptionalInputTensors const& features, PipelineIO& io, cudaStream_t stream);

private:
    EmbeddingData const& mEmbedding;
    LLMEngineConfig mConfig;

    //! Multimodal indices for the prefill embedding lookup, computed once per prefill in `embed()`
    //! from the host token IDs and reused by `assembleDeepstack()` for the same tokens (deepstack
    //! always follows `embed()` in the base prefill). Persisted for async kernel lifetime.
    Tensor mMultimodalIndices;
};

} // namespace rt
} // namespace trt_edgellm
