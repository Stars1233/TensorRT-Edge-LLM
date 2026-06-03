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
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/exec/tensorRegistry.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Thin TRT wrapper with prepare/execute split.
 *
 * EngineExecutor owns a TRT runtime, engine, and execution context. It replaces both
 * LLMEngineRunner and EagleDraftEngineRunner with a single model-agnostic
 * wrapper (~300 LOC).
 *
 *  - @c prepare() sets the optimization profile and delegates binding to
 *    TensorRegistry::bindAll().
 *  - @c execute() replays a cached CUDA graph when the binding state matches,
 *    otherwise falls back to enqueueV3.
 *  - @c captureGraph() captures a new CUDA graph for the current bindings.
 *
 * EngineExecutor knows nothing about models, phases, or features.
 */
class EngineExecutor
{
public:
    //! @brief Destructor — destroys all captured CUDA graphs.
    ~EngineExecutor() noexcept;

    //! Build an EngineExecutor for a vanilla single-engine LLM or a SpecDecode
    //! base engine. The factory builds the TensorRegistry internally via
    //! `buildRegistryForLLM(cfg)`.
    static std::unique_ptr<EngineExecutor> createForLLM(std::filesystem::path const& enginePath,
        LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim = std::nullopt);

    //! Build an EngineExecutor for the SpecDecode draft engine. The factory builds the
    //! TensorRegistry internally via `buildRegistryForSpecDecodeDraft(bundle)`.
    static std::unique_ptr<EngineExecutor> createForSpecDecodeDraft(
        std::filesystem::path const& enginePath, DeploymentConfig const& bundle);

    EngineExecutor(EngineExecutor const&) = delete;
    EngineExecutor& operator=(EngineExecutor const&) = delete;

    /*!
     * @brief Switch optimization profile, resolve shapes, bind all tensors.
     *
     * @param profileIndex TRT optimization profile index
     * @param dims Symbolic dimension values for this step
     * @param map Name-to-tensor mapping
     * @param stream CUDA stream for the async profile switch
     * @return True on success
     */
    bool prepare(int32_t profileIndex, InferenceDims const& dims, TensorMap const& map, cudaStream_t stream);

    /*!
     * @brief Execute inference.
     *
     * Replays a cached CUDA graph if one matches the current bindings,
     * otherwise falls back to enqueueV3.
     *
     * @param stream CUDA stream
     * @return True on success
     */
    bool execute(cudaStream_t stream);

    /*!
     * @brief Capture a CUDA graph for the current binding state (after prepare()).
     *
     * Performs a warmup enqueue, then captures via cudaStreamBeginCapture.
     * The captured graph is keyed by a binding hash with full snapshot
     * verification.
     *
     * @param stream CUDA stream (must not be the default stream)
     * @return True if capture succeeded
     */
    bool captureGraph(cudaStream_t stream);

    /*!
     * @brief Query required device memory for the execution context.
     * @return Required memory size in bytes
     */
    int64_t getRequiredContextMemorySize() const;

    /*!
     * @brief Provide shared device memory for the execution context.
     *
     * @param sharedMem Tensor whose memory will back the TRT context
     * @return True on success
     */
    bool setContextMemory(Tensor& sharedMem);

    //! @brief Return the number of I/O tensors in the engine.
    int32_t getNumIOTensors() const;

    //! @brief Return the name of the i-th I/O tensor.
    char const* getIOTensorName(int32_t index) const;

    //! @brief Return the data type of a named binding.
    nvinfer1::DataType getBindingDataType(char const* name) const;

    //! @brief Return a profile shape (min/opt/max) for a named binding.
    nvinfer1::Dims getProfileShape(char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const;

    //! @brief Access the underlying TRT engine for generic introspection.
    nvinfer1::ICudaEngine const& getEngine() const noexcept;

    //! @brief Snapshot of all binding addresses and shapes — used for graph-cache verification.
    struct BindingSnapshot
    {
        std::vector<std::pair<uintptr_t, nvinfer1::Dims>> bindings;

        bool operator==(BindingSnapshot const& rhs) const noexcept;
    };

private:
    /*!
     * @brief Construct a EngineExecutor from a serialized TRT engine file.
     *
     * Reads the engine, creates an IRuntime, deserializes the engine,
     * and creates an IExecutionContext with USER_MANAGED allocation.
     *
     * Private — use `createForLLM` / `createForSpecDecodeDraft` factories.
     *
     * @param enginePath Path to the serialized TRT engine file
     * @param registry TensorRegistry describing the binding layout
     * @throws std::runtime_error On I/O or deserialization failure
     */
    EngineExecutor(std::filesystem::path const& enginePath, TensorRegistry registry);

    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    TensorRegistry mRegistry;

    //! A captured CUDA graph together with its binding snapshot for verification.
    struct CapturedGraph
    {
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
        BindingSnapshot snapshot;
    };

    //! Graph cache keyed by a hash of all binding addresses + shapes.
    std::unordered_map<size_t, CapturedGraph> mGraphs;

    //! Hash all current binding addresses and shapes into a single key.
    size_t computeBindingHash() const;

    //! Build a full snapshot of the current binding state.
    BindingSnapshot snapshotBindings() const;
};

} // namespace rt
} // namespace trt_edgellm
