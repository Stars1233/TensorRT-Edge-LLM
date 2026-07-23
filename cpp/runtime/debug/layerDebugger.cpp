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

#include "runtime/debug/layerDebugger.h"

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

namespace
{
constexpr char const* kLayersEnv = "EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS";
constexpr char const* kDirEnv = "EDGELLM_DUMP_LOGITS_KVCACHE_DIR";

//! Trim ASCII whitespace from both ends.
std::string trim(std::string const& s)
{
    size_t const b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
    {
        return "";
    }
    size_t const e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

//! Parse the layer count "k" (a single positive integer) into the leading-layer set {0..k-1}.
//! The only producer (few-layer-validation.sh) dumps the first N decoder layers; selecting
//! arbitrary / non-leading layers is intentionally unsupported -- dump the prefix and pick the
//! layers of interest at analysis time (disk is cheap; a simpler parser is worth more).
std::set<int32_t> parseLeadingLayers(std::string const& spec)
{
    int32_t const k = std::stoi(trim(spec));
    if (k < 1)
    {
        throw std::runtime_error(std::string(kLayersEnv) + ": expected a positive layer count, got '" + spec + "'.");
    }
    std::set<int32_t> out;
    for (int32_t i = 0; i < k; ++i)
    {
        out.insert(i);
    }
    return out;
}

//! Copy the first @p activeBatchSize rows (batch is the outermost dim) of a device tensor into a
//! fresh host tensor of the same inner shape. Used for fixed-size recurrent / conv state, which —
//! unlike the KV cache — has no sequence dimension, so the active batch is a contiguous prefix.
Tensor copyBatchPrefix(Tensor const& src, int32_t activeBatchSize, std::string const& name)
{
    Coords const s = src.getShape();
    std::vector<int64_t> dims;
    dims.reserve(s.getNumDims());
    dims.push_back(activeBatchSize);
    int64_t inner = 1;
    for (int32_t d = 1; d < s.getNumDims(); ++d)
    {
        dims.push_back(s[d]);
        inner *= s[d];
    }
    nvinfer1::DataType const dtype = src.getDataType();
    Tensor host(Coords(dims), DeviceType::kCPU, dtype, name);
    size_t const bytes = static_cast<size_t>(activeBatchSize) * inner * utils::getTypeSize(dtype);
    CUDA_CHECK(cudaMemcpy(host.rawPointer(), src.rawPointer(), bytes, cudaMemcpyDeviceToHost));
    return host;
}
} // namespace

LayerDebugger::LayerDebugger(std::set<int32_t> layers, std::string dir, std::vector<std::vector<int32_t>> forcedTokens)
    : mLayers(std::move(layers))
    , mDir(std::move(dir))
    , mForcedTokens(std::move(forcedTokens))
{
    static std::atomic<int32_t> sRequestCounter{0};
    mRequestIdx = sRequestCounter++;
}

std::unique_ptr<LayerDebugger> LayerDebugger::fromEnv()
{
    char const* layersEnv = std::getenv(kLayersEnv);
    char const* dirEnv = std::getenv(kDirEnv);
    bool const hasLayers = layersEnv != nullptr && *layersEnv != '\0';
    bool const hasDir = dirEnv != nullptr && *dirEnv != '\0';

    if (!hasLayers && !hasDir)
    {
        return nullptr;
    }
    if (hasLayers != hasDir)
    {
        throw std::runtime_error(std::string(kLayersEnv) + " and " + kDirEnv
            + " must be set together (set both to enable layer dumping, or neither to disable).");
    }

    std::set<int32_t> layers = parseLeadingLayers(layersEnv);
    std::filesystem::create_directories(dirEnv);
    std::vector<std::vector<int32_t>> forcedTokens = readForcedTokensFromEnv();
    LOG_INFO("LayerDebugger enabled: %zu layer(s) -> %s", layers.size(), dirEnv);
    return std::unique_ptr<LayerDebugger>(new LayerDebugger(std::move(layers), dirEnv, std::move(forcedTokens)));
}

void LayerDebugger::dumpRound(HybridCacheManager& cacheManager, Tensor const& logits,
    std::vector<int32_t> const& validLengths, int32_t const* generatedTokenIds, int32_t activeBatchSize,
    cudaStream_t stream)
{
    // The KV cache / logits are produced asynchronously on this stream; synchronise
    // so the device-side data is final before we copy it out.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t const r = mRound;
    std::string const prefix = "round_" + std::to_string(r) + ".";

    // ---- logits: [activeBatch, vocab] (last-token only) ----
    {
        Coords const shape = logits.getShape();
        int64_t const vocab = shape[shape.getNumDims() - 1];
        nvinfer1::DataType const dtype = logits.getDataType();
        Tensor logitsHost({activeBatchSize, vocab}, DeviceType::kCPU, dtype, prefix + "logits");
        size_t const bytes = static_cast<size_t>(activeBatchSize) * vocab * utils::getTypeSize(dtype);
        CUDA_CHECK(cudaMemcpy(logitsHost.rawPointer(), logits.rawPointer(), bytes, cudaMemcpyDeviceToHost));
        mTensors.push_back(std::move(logitsHost));
    }

    // ---- per-layer state, by layer type ----
    // Every per-layer tensor is dumped as-is over the active-batch prefix (a plain contiguous
    // copy); no sequence-length truncation happens here. The comparison tool slices each sequence
    // to its valid length in PyTorch (using the dumped context_lengths), keeping this side simple.
    //   Attention layers   -> full combined KV cache [activeBatch, 2, kvHeads, maxSeqLen, headDim].
    //   Mamba / Gated-DeltaNet -> fixed-size recurrent + conv state (no sequence dim).
    int32_t const numLayers = cacheManager.numLayers();
    for (int32_t layer : mLayers)
    {
        if (layer < 0 || layer >= numLayers)
        {
            LOG_WARNING("LayerDebugger: layer %d out of range (model has %d layers); skipping.", layer, numLayers);
            continue;
        }
        std::string const lp = prefix + "layer_" + std::to_string(layer) + ".";
        if (cacheManager.getLayerType(layer) != HybridCacheManager::LayerType::kAttention)
        {
            // Mamba / Gated DeltaNet: fixed-size recurrent state [B, heads, headDim, stateSize] and
            // conv state [B, convDim, convKernel].
            mTensors.push_back(
                copyBatchPrefix(cacheManager.getRecurrentState(layer), activeBatchSize, lp + "recurrent_state"));
            mTensors.push_back(copyBatchPrefix(cacheManager.getConvState(layer), activeBatchSize, lp + "conv_state"));
            continue;
        }
        // Attention: combined KV cache [maxBatch, 2, kvHeads, maxSeqLen, headDim], contiguous.
        mTensors.push_back(copyBatchPrefix(cacheManager.getCombinedKVCache(layer), activeBatchSize, lp + "kv"));
    }

    // ---- per-sequence valid lengths ----
    {
        Tensor ctxLenHost({static_cast<int64_t>(activeBatchSize)}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
            prefix + "context_lengths");
        int32_t* p = ctxLenHost.dataPointer<int32_t>();
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            p[i] = validLengths.at(i);
        }
        mTensors.push_back(std::move(ctxLenHost));
    }

    // ---- per-round generated token ids (for cross-side greedy divergence detection) ----
    if (generatedTokenIds != nullptr)
    {
        Tensor genTokHost({static_cast<int64_t>(activeBatchSize)}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
            prefix + "generated_token_ids");
        std::copy(generatedTokenIds, generatedTokenIds + activeBatchSize, genTokHost.dataPointer<int32_t>());
        mTensors.push_back(std::move(genTokHost));
    }

    ++mRound;
}

void LayerDebugger::flush(cudaStream_t stream)
{
    if (mTensors.empty())
    {
        LOG_WARNING("LayerDebugger: no rounds were dumped; nothing to write.");
        return;
    }
    std::string const stem = mRequestIdx == 0 ? "edgellm_dump" : "edgellm_dump_" + std::to_string(mRequestIdx);
    std::filesystem::path const path = std::filesystem::path(mDir) / (stem + ".safetensors");
    if (safetensors::saveSafetensors(path, mTensors, stream))
    {
        LOG_INFO("LayerDebugger: wrote %zu tensors across %d round(s) to %s", mTensors.size(), mRound,
            path.string().c_str());
    }
    else
    {
        LOG_ERROR("LayerDebugger: failed to write %s", path.string().c_str());
    }
}

std::vector<std::vector<int32_t>> LayerDebugger::readForcedTokensFromEnv()
{
    char const* path = std::getenv("EDGELLM_FORCE_TOKENS_FILE");
    if (path == nullptr || *path == '\0')
    {
        return {};
    }
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error(std::string("EDGELLM_FORCE_TOKENS_FILE could not be opened: ") + path);
    }
    std::vector<std::vector<int32_t>> forced;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::vector<int32_t> row;
        int32_t tok;
        while (ss >> tok)
        {
            row.push_back(tok);
        }
        forced.push_back(std::move(row));
    }
    LOG_WARNING(
        "Teacher forcing ON: the decode loop will OVERRIDE its own sampled tokens with the "
        "forced tokens for %zu sequence(s) from %s (numeric-validation debug; not for production).",
        forced.size(), path);
    return forced;
}

void LayerDebugger::applyForcedTokens(
    std::vector<int32_t> const& genLengths, int32_t* tokenIds, int32_t activeBatchSize)
{
    if (mForcedTokens.empty())
    {
        return;
    }
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // The token produced this step is generated-token index genLengths[i] (0 at prefill).
        int32_t const idx = genLengths.at(i);
        if (i < static_cast<int32_t>(mForcedTokens.size()) && idx >= 0
            && idx < static_cast<int32_t>(mForcedTokens[i].size()))
        {
            tokenIds[i] = mForcedTokens.at(i).at(idx);
        }
    }
}

} // namespace rt
} // namespace trt_edgellm
