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

#include <common/tensor.h>
#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Per-layer KV head configuration for heterogeneous models.
struct KVLayerConfig
{
    int32_t numKVHeads{}; //!< Number of key-value heads for this layer
    int32_t headDim{};    //!< Head dimension for this layer
};

//! Per-layer KV cache manager that supports heterogeneous head configurations across layers.
//! Each attention layer gets its own independently-sized tensor with shape
//! [maxBatchSize, 2, numKVHeads_i, maxSequenceLength, headDim_i].
//! This replaces the monolithic LinearKVCache allocation when layers have different numKVHeads or headDim.
class KVCacheManager
{
public:
    //! \cond INTERNAL
    /*!
     * @brief Configuration for per-layer KV cache manager
     *
     * Defines the dimensions and capacity of the KV cache for each attention layer.
     */
    struct Config
    {
        int32_t numAttentionLayers{};            //!< Number of attention layers needing KV cache
        int32_t maxBatchSize{};                  //!< Maximum batch size
        int32_t maxSequenceLength{};             //!< Maximum sequence length
        std::vector<KVLayerConfig> layerConfigs; //!< Per-layer head config (size == numAttentionLayers)
        nvinfer1::DataType kvCacheType{};        //!< Storage dtype for KV cache (kHALF or kFP8)
    };
    //! \endcond

    //! @brief Default constructor
    KVCacheManager() noexcept = default;

    /*!
     * @brief Construct and initialize per-layer KV cache
     *
     * Allocates one device tensor per attention layer. Once allocated, memory won't be reallocated.
     * Determines whether all layers share the same numKVHeads and headDim (uniform mode).
     *
     * @param config Cache configuration with per-layer configs
     * @param stream CUDA stream for allocation
     * @throws std::runtime_error if config is invalid or data type is unsupported
     */
    KVCacheManager(Config const& config, cudaStream_t stream);

    //! @brief Destructor
    ~KVCacheManager() noexcept;

    //! @brief Deleted copy constructor to avoid large data copy
    KVCacheManager(KVCacheManager const&) = delete;

    //! @brief Deleted copy assignment to avoid large data copy
    //! @return Reference to this
    KVCacheManager& operator=(KVCacheManager const&) = delete;

    //! @brief Move constructor
    KVCacheManager(KVCacheManager&&) noexcept;

    //! @brief Move assignment operator
    //! @return Reference to this
    KVCacheManager& operator=(KVCacheManager&&) noexcept;

    //! Get the combined KVCache for the given attention layer.
    //! @param attnLayerIdx The index of the attention layer.
    //! @return A reference to the tensor with shape [maxBatchSize, 2, numKVHeads_i, maxSequenceLength, headDim_i].
    rt::Tensor& getCombinedKVCache(int32_t attnLayerIdx) noexcept;

    //! Get the layer configuration for the given attention layer.
    //! @param attnLayerIdx The index of the attention layer.
    //! @return The KVLayerConfig for this layer.
    KVLayerConfig const& getLayerConfig(int32_t attnLayerIdx) const noexcept;

    //! @brief Get the number of attention layers
    //! @return Number of attention layers
    int32_t numLayers() const noexcept;

    //! @brief Check if all layers have the same numKVHeads and headDim
    //! @return True if all layers are uniform
    bool isUniform() const noexcept;

    //! @brief Get cache configuration
    //! @return Cache configuration
    Config const& getConfig() const noexcept;

private:
    Config mConfig{};                     //!< Cache configuration
    std::vector<rt::Tensor> mLayerCaches; //!< Per-layer KV cache tensors on device
    bool mIsUniform{true};                //!< True if all layers share the same numKVHeads and headDim
};

} // namespace rt
} // namespace trt_edgellm
