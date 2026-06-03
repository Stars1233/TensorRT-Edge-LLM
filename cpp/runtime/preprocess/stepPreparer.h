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
#include "runtime/config/inferencePhase.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/pipelineIO.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace rt
{

//! Prepares per-step sequence metadata (selectTokenIndices, contextLengths).
//!
//! Extracted from the former `LLMEngineRunner::executePrefillStep()` and
//! `vanillaDecodingStepPrepareInputs()` / `vanillaDecodingStepBindTensors()`
//! methods. The class is stateless beyond configuration and a small host
//! scratch buffer for selectTokenIndices computation.
//!
//! Binding management is NOT this class's concern:
//!   * `kvcache_start_index` is a static registry binding whose per-phase
//!     shape comes from `InferenceDims::startIndexLen`.
//!   * `deepstack_embeds_*` bindings are owned by `DeepstackBinding` (the
//!     runtime calls `useRealFeatures` / `useZeroBroadcast` directly).
class StepPreparer
{
public:
    //! Construct with the engine configuration.
    explicit StepPreparer(LLMEngineConfig const& config);

    //! Prepare per-step sequence metadata (`selectTokenIndices`,
    //! `contextLengths`) for the given phase. Does not modify `tensorMap`.
    //!
    //! Fills PipelineIO:
    //!   - selectTokenIndices: prefill = ctxLen-1, decode = 0
    //!   - contextLengths: prefill = H2D copy from hostContextLengths,
    //!                     decode  = KV lengths (plugin) or zeros (native) + 1
    //!
    //! @param phase       Inference phase (Prefill or Decode).
    //! @param batchSize   Active batch size for this step.
    //! @param kvCache     KV cache to query for lengths.
    //! @param io          Pipeline I/O — selectTokenIndices & contextLengths are written.
    //! @param stream      CUDA stream for async operations.
    void prepare(
        InferencePhase phase, int32_t batchSize, HybridCacheManager& kvCache, PipelineIO& io, cudaStream_t stream);

private:
    LLMEngineConfig mConfig;

    //! Host-side scratch for computing selectTokenIndices before H2D copy.
    Tensor mHostSelectTokenIndices;
};

} // namespace rt
} // namespace trt_edgellm
