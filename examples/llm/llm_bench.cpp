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

#include "benchLogger.h"
#include "benchRunner.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/layerProfiler.h"
#include "runtime/legacy/eagleDraftEngineRunner.h"
#include "runtime/legacy/llmEngineRunner.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace trt_edgellm;
using rt::Json;

// ==================== Type Definitions ====================

enum ProfileBenchOptionId : int
{
    HELP = 801,
    ENGINE_DIR = 802,
    DEBUG = 803,
    BATCH_SIZE = 804,
    INPUT_LEN = 805,
    ITERATIONS = 806,
    WARMUP = 807,
    MODE = 812,
    REUSE_KV_LEN = 813,
    PAST_KV_LEN = 814,
    VERIFY_TREE_SIZE = 815,
    DRAFT_TREE_SIZE = 816,
    IMAGE_SIZE = 818,
    OSL = 827,
    OUTPUT_DIR = 828,
    SEED = 829,
    NO_CUDA_GRAPH = 830,
    EXTRACT_LAYER_INFO = 831,
    ACCEPT_RATE = 832,
    DRAFT_STEP = 833,
    PROFILE = 834
};

struct ProfileBenchArgs
{
    bool help{false};
    std::string engineDir;
    bool debug{false};
    int32_t batchSize{1};
    int32_t inputLen{-1}; // Input sequence length per batch (required for prefill modes)
    int32_t iterations{10};
    int32_t warmup{3};
    bool noProfile{true};   // Layer profiling is disabled by default; --profile enables it.
    std::string outputDir;  // Directory to dump output CSV files (layer profiling and E2E timing)
    int32_t imageHeight{0}; // Image height in pixels (required for visual mode)
    int32_t imageWidth{0};  // Image width in pixels (required for visual mode)

    // Mode parameter - no default
    BenchMode mode{BenchMode::kNONE};

    // KV cache parameters - no defaults for required params
    int32_t reuseKVLen{0}; // For prefill: reused KV cache length per batch
    int32_t pastKVLen{-1}; // For decode/verify/draft: past KV cache length per batch (required)

    // Speculative decoding parameters - no defaults
    int32_t verifyTreeSize{-1}; // For spec_verify
    int32_t draftTreeSize{-1};  // For spec_draft_proposal/spec_draft_prefill

    int32_t osl{1};        // Output sequence length (LLM OSL per batch, default: 1)
    int32_t acceptRate{5}; // Avg accepted tokens per spec-decode iteration (default: 5)
    int32_t draftStep{6};  // Number of drafting steps per spec-decode iteration (default: 6)

    // Random seed for reproducibility
    uint64_t seed{0};

    // CUDA graph is enabled by default for decode/EAGLE E2E timing.
    bool noCudaGraph{false};

    // Metadata extraction flags - disabled by default for performance.
    // Set via --extractLayerInfo <comma-separated list>.
    ExtractLayerInfo extractLayerInfo;

    //! Convert to BenchOutputParams for CSV/log output functions
    BenchOutputParams toOutputParams() const
    {
        BenchOutputParams p;
        p.mode = mode;
        p.batchSize = batchSize;
        p.inputLen = inputLen;
        p.pastKVLen = pastKVLen;
        p.verifyTreeSize = verifyTreeSize;
        p.draftTreeSize = draftTreeSize;
        p.osl = osl;
        p.imageHeight = imageHeight;
        p.imageWidth = imageWidth;
        p.reuseKVLen = reuseKVLen;
        p.iterations = iterations;
        p.acceptRate = acceptRate;
        p.draftStep = draftStep;
        return p;
    }
};

// ==================== printUsage ====================

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName << " --engineDir <dir> --mode <mode> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Required Options:" << std::endl;
    std::cerr << "  --engineDir               TensorRT engine directory path. Required." << std::endl;
    std::cerr << "  --mode                    Benchmarking mode. Required. One of:" << std::endl;
    std::cerr << "                              prefill           - LLM prefill phase" << std::endl;
    std::cerr << "                              decode            - LLM decode phase" << std::endl;
    std::cerr
        << "                              spec_verify       - Speculative decoding base model verification (EAGLE/MTP)"
        << std::endl;
    std::cerr << "                              spec_draft_proposal - Speculative decoding draft proposal (EAGLE/MTP)"
              << std::endl;
    std::cerr << "                              spec_draft_prefill - Speculative decoding draft prefill (EAGLE/MTP)"
              << std::endl;
    std::cerr << "                              visual            - Visual encoder" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Mode-Specific Required Options:" << std::endl;
    std::cerr << "  For prefill mode:" << std::endl;
    std::cerr << "    --inputLen              Input sequence length. Required." << std::endl;
    std::cerr << "    --reuseKVLen            Reused KV cache length. Optional, default=0." << std::endl;
    std::cerr << "  For decode mode:" << std::endl;
    std::cerr << "    --pastKVLen             Past KV cache length. Required." << std::endl;
    std::cerr << "  For spec_verify mode:" << std::endl;
    std::cerr << "    --verifyTreeSize        Verify tree size. Required." << std::endl;
    std::cerr << "    --pastKVLen             Past KV cache length. Required." << std::endl;
    std::cerr << "  For spec_draft_proposal mode:" << std::endl;
    std::cerr << "    --draftTreeSize         Draft tree size. Required." << std::endl;
    std::cerr << "    --pastKVLen             Past KV cache length. Required." << std::endl;
    std::cerr << "  For spec_draft_prefill mode:" << std::endl;
    std::cerr << "    --inputLen              Input sequence length. Required." << std::endl;
    std::cerr << "    --reuseKVLen            Reused KV cache length. Optional, default=0." << std::endl;
    std::cerr << "  For visual mode:" << std::endl;
    std::cerr << "    --engineDir             Visual encoder engine directory. Required." << std::endl;
    std::cerr << "    --imageSize             Image dimensions as HxW (e.g., 896x448). Required." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Common Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --debug                   Use debug mode (verbose logging)" << std::endl;
    std::cerr << "  --batchSize               Batch size. Default = 1" << std::endl;
    std::cerr << "  --iterations              Number of profiling iterations (after warmup). Default = 10" << std::endl;
    std::cerr << "  --warmup                  Number of warmup iterations. Default = 3" << std::endl;
    std::cerr << "  --osl                     Output sequence length for decode E2E timing. Default = 1." << std::endl;
    std::cerr << "                            osl=1: E2E runs --iterations times." << std::endl;
    std::cerr << "                            osl>1: E2E runs full sequence decode once." << std::endl;
    std::cerr << "  --profile                 Enable per-layer profiling. Disabled by default." << std::endl;
    std::cerr << "  --outputDir               Directory to dump output CSV files (layer profiling and E2E timing)"
              << std::endl;
    std::cerr << "  --seed                    Random seed for reproducible data. Default = 0" << std::endl;
    std::cerr << "  --noCudaGraph             Disable CUDA graph capture for decode/EAGLE E2E timing." << std::endl;
    std::cerr << "                            Capture is enabled by default and falls back to non-graph on failure."
              << std::endl;
    std::cerr << "  --extractLayerInfo <opts>  Comma-separated list of layer info to extract:" << std::endl;
    std::cerr << "                              all, shapes, onnx_ops, tactics, data_types" << std::endl;
    std::cerr << "                            Implies --profile." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Speculative Decoding Options:" << std::endl;
    std::cerr << "  --acceptRate              Avg accepted tokens per spec-decode iteration (default: 5)." << std::endl;
    std::cerr << "                            verify_steps = ceil((osl-1) / acceptRate)." << std::endl;
    std::cerr << "  --draftStep               Number of drafting steps per spec-decode iteration (default: 6)."
              << std::endl;
    std::cerr << "                            draft_calls = verify_steps * (draftStep-1)." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  # Prefill mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./engines --mode prefill --inputLen 128" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Decode mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./engines --mode decode --pastKVLen 128" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Spec-decode verify mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./engines --mode spec_verify --verifyTreeSize 60 --pastKVLen 128"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Spec-decode draft proposal mode" << std::endl;
    std::cerr << "  " << programName
              << " --engineDir ./engines --mode spec_draft_proposal --draftTreeSize 60 --pastKVLen 128" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Visual encoder mode" << std::endl;
    std::cerr << "  " << programName << " --engineDir ./visual_engines --mode visual --imageSize 896x448" << std::endl;
}

// ==================== parseArgs + validateArgs ====================

bool parseArgs(ProfileBenchArgs& args, int argc, char* argv[])
{
    static struct option options[] = {{"help", no_argument, 0, ProfileBenchOptionId::HELP},
        {"engineDir", required_argument, 0, ProfileBenchOptionId::ENGINE_DIR},
        {"debug", no_argument, 0, ProfileBenchOptionId::DEBUG},
        {"batchSize", required_argument, 0, ProfileBenchOptionId::BATCH_SIZE},
        {"inputLen", required_argument, 0, ProfileBenchOptionId::INPUT_LEN},
        {"iterations", required_argument, 0, ProfileBenchOptionId::ITERATIONS},
        {"warmup", required_argument, 0, ProfileBenchOptionId::WARMUP},
        {"mode", required_argument, 0, ProfileBenchOptionId::MODE},
        {"reuseKVLen", required_argument, 0, ProfileBenchOptionId::REUSE_KV_LEN},
        {"pastKVLen", required_argument, 0, ProfileBenchOptionId::PAST_KV_LEN},
        {"verifyTreeSize", required_argument, 0, ProfileBenchOptionId::VERIFY_TREE_SIZE},
        {"draftTreeSize", required_argument, 0, ProfileBenchOptionId::DRAFT_TREE_SIZE},
        {"profile", no_argument, 0, ProfileBenchOptionId::PROFILE},
        {"outputDir", required_argument, 0, ProfileBenchOptionId::OUTPUT_DIR},
        {"osl", required_argument, 0, ProfileBenchOptionId::OSL},
        {"seed", required_argument, 0, ProfileBenchOptionId::SEED},
        {"imageSize", required_argument, 0, ProfileBenchOptionId::IMAGE_SIZE},
        {"noCudaGraph", no_argument, 0, ProfileBenchOptionId::NO_CUDA_GRAPH},
        {"extractLayerInfo", required_argument, 0, ProfileBenchOptionId::EXTRACT_LAYER_INFO},
        {"acceptRate", required_argument, 0, ProfileBenchOptionId::ACCEPT_RATE},
        {"draftStep", required_argument, 0, ProfileBenchOptionId::DRAFT_STEP}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        try
        {
            switch (opt)
            {
            case ProfileBenchOptionId::HELP: args.help = true; return true;
            case ProfileBenchOptionId::ENGINE_DIR: args.engineDir = optarg; break;
            case ProfileBenchOptionId::DEBUG: args.debug = true; break;
            case ProfileBenchOptionId::BATCH_SIZE:
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::INPUT_LEN:
                args.inputLen = std::stoi(optarg);
                if (args.inputLen <= 0)
                {
                    LOG_ERROR("Invalid inputLen: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::ITERATIONS:
                args.iterations = std::stoi(optarg);
                if (args.iterations <= 0)
                {
                    LOG_ERROR("Invalid iterations: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::WARMUP:
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup: must be non-negative");
                    return false;
                }
                break;
            case ProfileBenchOptionId::MODE:
            {
                std::string modeStr = optarg;
                if (modeStr == "prefill")
                {
                    args.mode = BenchMode::kPREFILL;
                }
                else if (modeStr == "decode")
                {
                    args.mode = BenchMode::kDECODE;
                }
                else if (modeStr == "spec_verify" || modeStr == "eagle_verify")
                {
                    args.mode = BenchMode::kEAGLE_VERIFY;
                }
                else if (modeStr == "spec_draft_proposal" || modeStr == "eagle_draft_proposal")
                {
                    args.mode = BenchMode::kEAGLE_DRAFT_PROPOSAL;
                }
                else if (modeStr == "spec_draft_prefill" || modeStr == "eagle_draft_prefill")
                {
                    args.mode = BenchMode::kEAGLE_DRAFT_PREFILL;
                }
                else if (modeStr == "visual")
                {
                    args.mode = BenchMode::kVISUAL;
                }
                else
                {
                    LOG_ERROR("Invalid mode: %s", optarg);
                    return false;
                }
                break;
            }
            case ProfileBenchOptionId::REUSE_KV_LEN:
                args.reuseKVLen = std::stoi(optarg);
                if (args.reuseKVLen < 0)
                {
                    LOG_ERROR("Invalid reuseKVLen: must be non-negative");
                    return false;
                }
                break;
            case ProfileBenchOptionId::PAST_KV_LEN:
                args.pastKVLen = std::stoi(optarg);
                if (args.pastKVLen < 0)
                {
                    LOG_ERROR("Invalid pastKVLen: must be non-negative");
                    return false;
                }
                break;
            case ProfileBenchOptionId::VERIFY_TREE_SIZE:
                args.verifyTreeSize = std::stoi(optarg);
                if (args.verifyTreeSize <= 0)
                {
                    LOG_ERROR("Invalid verifyTreeSize: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::DRAFT_TREE_SIZE:
                args.draftTreeSize = std::stoi(optarg);
                if (args.draftTreeSize <= 0)
                {
                    LOG_ERROR("Invalid draftTreeSize: must be positive");
                    return false;
                }
                break;
            case ProfileBenchOptionId::IMAGE_SIZE:
            {
                // Parse "HxW" format (e.g., "896x448")
                std::string sizeStr = optarg;
                size_t xPos = sizeStr.find('x');
                if (xPos == std::string::npos)
                    xPos = sizeStr.find('X');
                if (xPos != std::string::npos)
                {
                    args.imageHeight = std::stoi(sizeStr.substr(0, xPos));
                    args.imageWidth = std::stoi(sizeStr.substr(xPos + 1));
                }
                else
                {
                    args.imageHeight = std::stoi(sizeStr);
                    args.imageWidth = args.imageHeight;
                }
                if (args.imageWidth <= 0 || args.imageHeight <= 0)
                {
                    LOG_ERROR("Invalid imageSize: height and width must be positive");
                    return false;
                }
                break;
            }
            case ProfileBenchOptionId::PROFILE: args.noProfile = false; break;
            case ProfileBenchOptionId::OUTPUT_DIR: args.outputDir = optarg; break;
            case ProfileBenchOptionId::OSL:
                args.osl = std::stoi(optarg);
                if (args.osl < 1)
                {
                    LOG_ERROR("Invalid osl: must be >= 1");
                    return false;
                }
                break;
            case ProfileBenchOptionId::SEED: args.seed = std::stoull(optarg); break;
            case ProfileBenchOptionId::NO_CUDA_GRAPH: args.noCudaGraph = true; break;
            case ProfileBenchOptionId::EXTRACT_LAYER_INFO:
            {
                std::string val = optarg;
                std::istringstream stream(val);
                std::string token;
                while (std::getline(stream, token, ','))
                {
                    if (token == "all")
                    {
                        args.extractLayerInfo.shapes = true;
                        args.extractLayerInfo.onnxOps = true;
                        args.extractLayerInfo.tactics = true;
                        args.extractLayerInfo.dataTypes = true;
                    }
                    else if (token == "shapes")
                        args.extractLayerInfo.shapes = true;
                    else if (token == "onnx_ops")
                        args.extractLayerInfo.onnxOps = true;
                    else if (token == "tactics")
                        args.extractLayerInfo.tactics = true;
                    else if (token == "data_types")
                        args.extractLayerInfo.dataTypes = true;
                    else
                    {
                        LOG_ERROR(
                            "Unknown --extractLayerInfo value: '%s'. Valid: all,shapes,onnx_ops,tactics,data_types",
                            token.c_str());
                        return false;
                    }
                }
                break;
            }
            case ProfileBenchOptionId::ACCEPT_RATE: args.acceptRate = std::stoi(optarg); break;
            case ProfileBenchOptionId::DRAFT_STEP: args.draftStep = std::stoi(optarg); break;
            default: return false;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to parse argument: %s", e.what());
            return false;
        }
    }

    // Layer metadata is only emitted through the layer profiling CSV.
    if (args.extractLayerInfo.any())
    {
        args.noProfile = false;
    }

    return true;
}

bool validateArgs(ProfileBenchArgs const& args)
{
    if (args.engineDir.empty())
    {
        LOG_ERROR("--engineDir is required");
        return false;
    }

    if (args.mode == BenchMode::kNONE)
    {
        LOG_ERROR("--mode is required. Use --help for available modes.");
        return false;
    }

    // Mode-specific validation
    switch (args.mode)
    {
    case BenchMode::kPREFILL:
        if (args.inputLen < 0)
        {
            LOG_ERROR("--inputLen is required for prefill mode");
            return false;
        }
        break;
    case BenchMode::kDECODE:
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for decode mode");
            return false;
        }
        break;
    case BenchMode::kEAGLE_VERIFY:
        if (args.verifyTreeSize < 0)
        {
            LOG_ERROR("--verifyTreeSize is required for spec_verify mode");
            return false;
        }
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for spec_verify mode");
            return false;
        }
        break;
    case BenchMode::kEAGLE_DRAFT_PROPOSAL:
        if (args.draftTreeSize < 0)
        {
            LOG_ERROR("--draftTreeSize is required for spec_draft_proposal mode");
            return false;
        }
        if (args.pastKVLen < 0)
        {
            LOG_ERROR("--pastKVLen is required for spec_draft_proposal mode");
            return false;
        }
        break;
    case BenchMode::kEAGLE_DRAFT_PREFILL:
        if (args.inputLen < 0)
        {
            LOG_ERROR("--inputLen is required for spec_draft_prefill mode");
            return false;
        }
        break;
    case BenchMode::kVISUAL:
        if (args.imageHeight <= 0 || args.imageWidth <= 0)
        {
            LOG_ERROR("--imageSize is required for visual mode (e.g., --imageSize 896x448)");
            return false;
        }
        break;
    default: LOG_ERROR("Unknown mode"); return false;
    }

    return true;
}

// ==================== main ====================

int main(int argc, char** argv)
{
    // ===== Phase 0: Parse & Setup =====
    ProfileBenchArgs args;
    if ((argc < 2) || (!parseArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (!validateArgs(args))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    LOG_INFO("=== LLM Profile Benchmark ===");
    LOG_INFO("Mode: %s", modeToString(args.mode).c_str());

    auto pluginHandles = loadEdgellmPluginLib();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Layer profiling is opt-in; E2E timing is the default benchmark path.
    if (!args.noProfile)
    {
        layerProfiler::LayerProfiler::getInstance().setEnabled(true);
        LOG_INFO("Layer profiling enabled. E2E CUDA graph timing will be skipped.");
    }
    else
    {
        LOG_INFO("Layer profiling disabled from startup. E2E timing only.");
    }

    // Layer metadata (ONNX ops, shapes, tactics, data types) extracted from engine inspector
    std::map<std::string, LayerMetadata> layerMetadata;

    // Create layer info directory if specified
    if (!args.outputDir.empty())
    {
        std::filesystem::create_directories(args.outputDir);
        LOG_INFO("Output CSV files will be saved to: %s", args.outputDir.c_str());
    }

    // Variables for benchmark results
    std::vector<KernelTimes> timesPerIter;
    timesPerIter.reserve(args.iterations);

    // E2E timing result (set by each mode's E2E timing section)
    float e2eTimeMsResult = 0.0f;

    // Ordered collection to accumulate layer timings across iterations (preserves model layer order)
    OrderedLayerTimings layerTimings;

    // ===== Phase 1: Initialize Runner =====
    std::unique_ptr<rt::LLMEngineRunner> runner;
    std::unique_ptr<rt::EagleDraftEngineRunner> draftRunner;
    std::unique_ptr<rt::MultimodalRunner> visualRunner;
    rt::Tensor contextMemory; // Shared context memory for TRT execution (kUSER_MANAGED)

    int32_t hiddenSize = 0;
    int32_t vocabSize = 0;
    int32_t eagleHiddenDim = 0;
    int32_t baseHiddenSize = 0;
    nvinfer1::DataType dtype = nvinfer1::DataType::kHALF;
    nvinfer1::DataType logitsDtype = nvinfer1::DataType::kHALF;
    // Standalone engine for dtype queries and layer metadata extraction. Destroyed after use.
    std::unique_ptr<nvinfer1::ICudaEngine> standaloneEngine;
    int64_t imageTokens = 0;

    if (args.mode == BenchMode::kVISUAL)
    {
        if (args.imageHeight <= 0 || args.imageWidth <= 0)
        {
            LOG_ERROR("--imageSize is required for visual mode (e.g., --imageSize 1024x512)");
            return EXIT_FAILURE;
        }

        try
        {
            // Read maxSequenceLength from sibling LLM config (needed for MRoPE buffer allocation).
            int32_t maxSeqLen = 4096; // fallback default
            for (auto const& siblingDir : {"llm", "base"})
            {
                auto llmConfigPath = std::filesystem::path(args.engineDir).parent_path() / siblingDir / "config.json";
                if (std::filesystem::exists(llmConfigPath))
                {
                    std::ifstream f(llmConfigPath);
                    if (f.is_open())
                    {
                        auto cfg = Json::parse(f);
                        if (cfg.contains("builder_config") && cfg["builder_config"].contains("max_kv_cache_capacity"))
                        {
                            maxSeqLen = cfg["builder_config"]["max_kv_cache_capacity"].get<int32_t>();
                            LOG_INFO("Read maxSequenceLength=%d from %s", maxSeqLen, llmConfigPath.string().c_str());
                        }
                    }
                    break;
                }
            }
            visualRunner = rt::MultimodalRunner::create(args.engineDir, args.batchSize, maxSeqLen, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to create MultimodalRunner: %s", e.what());
            return EXIT_FAILURE;
        }

        int64_t memSize = visualRunner->getRequiredContextMemorySize();
        contextMemory
            = rt::Tensor(rt::Coords{memSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "context_memory");
        visualRunner->setContextMemory(contextMemory);

        // Construct a fake request with dummy images for benchmarking.
        rt::LLMGenerationRequest dummyRequest;
        for (int32_t b = 0; b < args.batchSize; ++b)
        {
            rt::LLMGenerationRequest::Request req;
            rt::Tensor fakeImage({static_cast<int64_t>(args.imageHeight), static_cast<int64_t>(args.imageWidth), 3},
                rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "fake");
            std::memset(fakeImage.rawPointer(), 128, static_cast<size_t>(args.imageHeight) * args.imageWidth * 3);
            req.imageBuffers.emplace_back(std::move(fakeImage));
            dummyRequest.requests.push_back(std::move(req));
        }

        std::vector<std::vector<int32_t>> unusedInputIds;
        rt::Tensor unusedRope;
        if (!visualRunner->preprocess(dummyRequest, unusedInputIds, nullptr, unusedRope, stream, true))
        {
            LOG_ERROR("Failed to prepare dummy visual inputs for %dx%d", args.imageHeight, args.imageWidth);
            return EXIT_FAILURE;
        }

        imageTokens = visualRunner->getOutputEmbedding().getShape()[0];
        LOG_INFO("Image Size: %dx%d -> %ld image tokens (batch=%d)", args.imageHeight, args.imageWidth, imageTokens,
            args.batchSize);

        standaloneEngine = loadStandaloneEngine(std::filesystem::path(args.engineDir) / "visual.engine");
    }
    else if (args.mode == BenchMode::kPREFILL || args.mode == BenchMode::kDECODE
        || args.mode == BenchMode::kEAGLE_VERIFY)
    {
        // Use llm.engine for standard VLM, eagle_base.engine for EAGLE (auto-detect by file presence)
        std::filesystem::path enginePath = std::filesystem::path(args.engineDir) / "llm.engine";
        if (!std::filesystem::exists(enginePath))
        {
            enginePath = std::filesystem::path(args.engineDir) / "eagle_base.engine";
        }
        if (!std::filesystem::exists(enginePath))
        {
            LOG_ERROR("Engine not found (tried llm.engine and eagle_base.engine in %s)", args.engineDir.c_str());
            return EXIT_FAILURE;
        }

        // Use config.json for standard VLM, base_config.json for EAGLE
        std::filesystem::path configPath = std::filesystem::path(args.engineDir) / "config.json";
        if (!std::filesystem::exists(configPath))
        {
            configPath = std::filesystem::path(args.engineDir) / "base_config.json";
        }
        std::unordered_map<std::string, std::string> loraMap;

        try
        {
            runner = std::make_unique<rt::LLMEngineRunner>(enginePath, configPath, loraMap, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to create LLMEngineRunner: %s", e.what());
            return EXIT_FAILURE;
        }

        // Allocate shared context memory (required for kUSER_MANAGED strategy)
        int64_t memSize = runner->getRequiredContextMemorySize();
        contextMemory
            = rt::Tensor(rt::Coords{memSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "context_memory");
        runner->setContextMemory(contextMemory);

        auto engineConfig = runner->getEngineConfig();
        hiddenSize = engineConfig.hiddenSize;
        vocabSize = engineConfig.vocabSize;
        eagleHiddenDim = engineConfig.outputHiddenDim;

        standaloneEngine = loadStandaloneEngine(enginePath);
        dtype = standaloneEngine->getTensorDataType(binding_names::kInputsEmbeds);
        logitsDtype = standaloneEngine->getTensorDataType(binding_names::kLogits);
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL || args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
    {
        std::filesystem::path enginePath = std::filesystem::path(args.engineDir) / "eagle_draft.engine";
        std::filesystem::path configPath = std::filesystem::path(args.engineDir) / "draft_config.json";
        if (!std::filesystem::exists(enginePath))
        {
            LOG_ERROR("Eagle draft engine not found at %s", enginePath.string().c_str());
            return EXIT_FAILURE;
        }

        try
        {
            draftRunner = std::make_unique<rt::EagleDraftEngineRunner>(enginePath, configPath, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to create EagleDraftEngineRunner: %s", e.what());
            return EXIT_FAILURE;
        }

        // Allocate shared context memory (required for kUSER_MANAGED strategy)
        int64_t memSize = draftRunner->getRequiredContextMemorySize();
        contextMemory
            = rt::Tensor(rt::Coords{memSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "context_memory");
        draftRunner->setContextMemory(contextMemory);

        auto engineConfig = draftRunner->getDraftEngineConfig();
        hiddenSize = engineConfig.draftModelHiddenDim;
        vocabSize = engineConfig.draftModelVocabSize;
        baseHiddenSize = engineConfig.baseModelHiddenDim;
        standaloneEngine = loadStandaloneEngine(enginePath);
        dtype = standaloneEngine->getTensorDataType(binding_names::kInputsEmbeds);
    }

    // ===== Phase 2: Log Config =====
    if (runner)
    {
        logLlmEngineConfig(runner->getEngineConfig());
    }
    else if (draftRunner)
    {
        logDraftEngineConfig(draftRunner->getDraftEngineConfig());
    }

    logBenchConfig(args.toOutputParams(), imageTokens);

    // ===== Phase 3: Prepare Tensors + Define Lambdas =====
    std::function<void()> resetState;
    std::function<bool()> step;
    std::string modeName;

    // Tensors at main scope (lifetime must span all phases).
    rt::Tensor reuseKVCacheLengths;
    if (args.mode != BenchMode::kVISUAL)
    {
        reuseKVCacheLengths = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "reuse_kv_lengths");
    }
    rt::Tensor prefillInputs;
    rt::Tensor contextLengths;
    rt::Tensor logits;
    rt::Tensor decodeInputs;
    rt::Tensor verifyInputs;
    rt::Tensor verifyMask;
    rt::Tensor verifyLogits;
    rt::Tensor verifyHiddenStates;
    rt::Tensor draftInputs;
    rt::Tensor draftMask;
    rt::Tensor draftBaseHiddenStates;
    rt::Tensor draftModelHiddenStates;
    rt::Tensor draftTreeLength;
    rt::Tensor draftLogits;
    rt::Tensor draftOutputHiddenStates;
    rt::Tensor inputsEmbeds;
    rt::Tensor baseModelHiddenStates;
    rt::Tensor draftPrefillDraftModelHiddenStates;
    rt::Tensor draftPrefillContextLengths;
    rt::Tensor outputLogits;
    rt::Tensor outputHiddenStates;

    // Deepstack tensors for prefill mode
    std::vector<rt::Tensor> deepstackEmbedTensors;
    rt::OptionalInputTensors deepstackEmbeds;
    std::unique_ptr<rt::Tensor> outputHiddenStatesPtr;
    rt::OptionalOutputTensor outputHiddenStatesOpt = std::nullopt;

    // Vectors for KV cache reset
    std::vector<int32_t> reuseKVLenVec;
    std::vector<int32_t> pastKVLenVec;

    // E2E-specific lambdas/values (mode-specific, used in Phase 7)
    std::function<void(int32_t)> postStep = [](int32_t) {};
    std::function<bool()> captureGraph = []() { return false; };
    int32_t decodeSteps = 1;
    bool useSequentialE2E = false;

    if (args.mode == BenchMode::kVISUAL)
    {
        modeName = "Visual Encoder";

        step = [&]() { return visualRunner->infer(stream); };
        resetState = []() {};
    }
    else if (args.mode == BenchMode::kPREFILL)
    {
        modeName = "Prefill";
        LOG_INFO("Prefill mode: InputLen=%d, ReuseKVLen=%d", args.inputLen, args.reuseKVLen);

        // For MRoPE (VLM), reshape RoPE cache to match batchSize before prefill.
        // MRoPE allocates with maxSupportedBatchSize capacity (see llmEngineRunner.cpp RopeType::kMRope),
        // then initializes as {1, maxKV, rotaryDim}. Reshape to {batchSize, ...} is safe.
        // In real inference, multimodal preprocess handles this; in benchmark we do it manually.
        auto engineCfg = runner->getEngineConfig();
        if (args.batchSize > 1 && engineCfg.ropeConfig.type == rt::RopeType::kMRope)
        {
            auto& ropeCache = runner->getRopeCosSinCacheTensor();
            check::check(ropeCache.reshape({args.batchSize, engineCfg.maxKVCacheCapacity, engineCfg.rotaryDim}),
                "MRoPE RopeCosSinCache reshape failed");
        }

        prefillInputs = rt::Tensor(
            rt::Coords{args.batchSize, args.inputLen, hiddenSize}, rt::DeviceType::kGPU, dtype, "prefill_input");
        fillRandomData(prefillInputs, -1.0f, 1.0f, dtype, args.seed);

        contextLengths = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "context_lengths");
        std::vector<int32_t> contextLengthsHost(args.batchSize, args.reuseKVLen + args.inputLen);
        std::memcpy(
            contextLengths.rawPointer(), contextLengthsHost.data(), contextLengthsHost.size() * sizeof(int32_t));

        logits = rt::Tensor(rt::Coords{args.batchSize, vocabSize}, rt::DeviceType::kGPU, logitsDtype, "logits");

        reuseKVLenVec.assign(args.batchSize, args.reuseKVLen);

        // Allocate deepstack embeds if needed (zero tensors for benchmarking without multimodal input)
        auto engineConfig = runner->getEngineConfig();
        if (engineConfig.numDeepstackFeatures > 0)
        {
            LOG_INFO("Allocating %d deepstack embed tensors (zero-filled for benchmarking)",
                engineConfig.numDeepstackFeatures);
            for (int32_t i = 0; i < engineConfig.numDeepstackFeatures; ++i)
            {
                deepstackEmbedTensors.emplace_back(rt::Coords{args.batchSize, args.inputLen, hiddenSize},
                    rt::DeviceType::kGPU, dtype, "deepstack_embed_" + std::to_string(i));
                // Zero-fill the tensor
                CUDA_CHECK(cudaMemsetAsync(deepstackEmbedTensors.back().rawPointer(), 0,
                    deepstackEmbedTensors.back().getShape().volume() * rt::utils::getTypeSize(dtype), stream));
            }
            for (auto const& t : deepstackEmbedTensors)
            {
                deepstackEmbeds.push_back(std::cref(t));
            }
        }

        // Allocate output hidden states if Eagle is enabled
        if (engineConfig.enableEagleSpecDecode)
        {
            outputHiddenStatesPtr
                = std::make_unique<rt::Tensor>(rt::Coords{args.batchSize, args.inputLen, engineConfig.outputHiddenDim},
                    rt::DeviceType::kGPU, dtype, "output_hidden_states");
            outputHiddenStatesOpt = std::ref(*outputHiddenStatesPtr);
        }

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), reuseKVLenVec.data(), reuseKVLenVec.size() * sizeof(int32_t));
            runner->getCacheManager().resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&]() {
            return runner->executePrefillStep(
                prefillInputs, contextLengths, deepstackEmbeds, logits, outputHiddenStatesOpt, stream);
        };
    }
    else if (args.mode == BenchMode::kDECODE)
    {
        modeName = "Decode";
        LOG_INFO("Decode mode: PastKVLen=%d", args.pastKVLen);

        int32_t osl = args.osl;
        int32_t decodeTokens = (osl > 1) ? (osl - 1) : 1;
        LOG_INFO("OSL=%d: will run %d decode steps for E2E timing", osl, decodeTokens);
        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        decodeInputs
            = rt::Tensor(rt::Coords{args.batchSize, 1, hiddenSize}, rt::DeviceType::kGPU, dtype, "decode_input");
        fillRandomData(decodeInputs, -1.0f, 1.0f, dtype, args.seed);

        logits = rt::Tensor(rt::Coords{args.batchSize, vocabSize}, rt::DeviceType::kGPU, logitsDtype, "logits");

        pastKVLenVec.assign(args.batchSize, args.pastKVLen);

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            runner->getCacheManager().resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&]() { return runner->executeVanillaDecodingStep(decodeInputs, logits, std::nullopt, stream); };
        captureGraph = [&]() { return runner->captureVanillaDecodingCudaGraph(decodeInputs, logits, {}, stream); };

        // E2E config: osl=1 uses repeated single-step timing, osl>1 uses sequential full-decode timing
        if (osl > 1)
        {
            useSequentialE2E = true;
            decodeSteps = decodeTokens;
            postStep = [](int32_t) {};
        }
    }
    else if (args.mode == BenchMode::kEAGLE_VERIFY)
    {
        modeName = "Spec Verify";
        LOG_INFO("Spec Verify mode: VerifyTreeSize=%d, PastKVLen=%d", args.verifyTreeSize, args.pastKVLen);

        pastKVLenVec.assign(args.batchSize, args.pastKVLen);

        verifyInputs = rt::Tensor(
            rt::Coords{args.batchSize, args.verifyTreeSize, hiddenSize}, rt::DeviceType::kGPU, dtype, "verify_input");
        fillRandomData(verifyInputs, -1.0f, 1.0f, dtype, args.seed);

        verifyMask = rt::Tensor(rt::Coords{args.batchSize, args.verifyTreeSize, args.verifyTreeSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "verify_mask");
        fillInt8(verifyMask, 1);

        int32_t selectTokenSize = args.batchSize * args.verifyTreeSize;
        verifyLogits = rt::Tensor(
            rt::Coords{selectTokenSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "verify_logits");
        verifyHiddenStates = rt::Tensor(
            rt::Coords{selectTokenSize, eagleHiddenDim}, rt::DeviceType::kGPU, dtype, "verify_hidden_states");

        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            runner->getCacheManager().resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&]() {
            return runner->executeEagleBaseTreeDecodingStep(
                verifyInputs, verifyMask, verifyLogits, verifyHiddenStates, stream);
        };
        captureGraph = [&]() {
            return runner->captureEagleBaseTreeDecodingCudaGraph(
                verifyInputs, verifyMask, verifyLogits, verifyHiddenStates, "", stream);
        };

        // E2E config: osl=1 uses repeated single-step timing, osl>1 uses sequential full-decode timing
        if (args.osl > 1)
        {
            useSequentialE2E = true;
            decodeSteps = (args.osl - 1 + args.acceptRate - 1) / args.acceptRate;
            postStep = [&](int32_t) { runner->getCacheManager().commitSequenceLength(args.acceptRate, stream); };
        }
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PROPOSAL)
    {
        modeName = "Spec Draft";
        LOG_INFO("Spec Draft mode: DraftTreeSize=%d, PastKVLen=%d", args.draftTreeSize, args.pastKVLen);

        pastKVLenVec.assign(args.batchSize, args.pastKVLen);

        draftInputs = rt::Tensor(
            rt::Coords{args.batchSize, args.draftTreeSize, hiddenSize}, rt::DeviceType::kGPU, dtype, "draft_input");
        fillRandomData(draftInputs, -1.0f, 1.0f, dtype, args.seed);

        draftMask = rt::Tensor(rt::Coords{args.batchSize, args.draftTreeSize, args.draftTreeSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "draft_mask");
        fillInt8(draftMask, 1);

        draftBaseHiddenStates = rt::Tensor(rt::Coords{args.batchSize, args.draftTreeSize, baseHiddenSize},
            rt::DeviceType::kGPU, dtype, "draft_base_hidden_zeros");
        fillRandomData(draftBaseHiddenStates, 0.0f, 0.0f, dtype, args.seed);

        draftModelHiddenStates = rt::Tensor(rt::Coords{args.batchSize, args.draftTreeSize, hiddenSize},
            rt::DeviceType::kGPU, dtype, "draft_hidden_input");
        fillRandomData(draftModelHiddenStates, -1.0f, 1.0f, dtype, args.seed);

        draftTreeLength = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "draft_tree_len");
        fillInt32(draftTreeLength, args.draftTreeSize);

        int32_t numSelectedTokens = args.draftTreeSize;
        draftLogits = rt::Tensor(rt::Coords{args.batchSize, numSelectedTokens, vocabSize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kFLOAT, "draft_logits");
        draftOutputHiddenStates = rt::Tensor(rt::Coords{args.batchSize, numSelectedTokens, hiddenSize},
            rt::DeviceType::kGPU, dtype, "draft_hidden_output");

        LOG_INFO(args.noCudaGraph ? "CUDA graph disabled; using non-CUDA-graph execution" : "CUDA graph enabled");

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), pastKVLenVec.data(), pastKVLenVec.size() * sizeof(int32_t));
            draftRunner->getCacheManager().resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&]() {
            return draftRunner->executeEagleDraftProposalStep(draftInputs, draftBaseHiddenStates,
                draftModelHiddenStates, draftTreeLength, draftMask, draftLogits, draftOutputHiddenStates, stream);
        };
        captureGraph = [&]() {
            return draftRunner->captureEagleDraftProposalCudaGraph(draftInputs, draftBaseHiddenStates,
                draftModelHiddenStates, draftTreeLength, draftMask, draftLogits, draftOutputHiddenStates, stream);
        };

        // E2E config: osl=1 uses repeated single-step timing, osl>1 uses sequential full-decode timing
        if (args.osl > 1)
        {
            useSequentialE2E = true;
            int32_t eagleIterations = (args.osl - 1 + args.acceptRate - 1) / args.acceptRate;
            decodeSteps = eagleIterations * (args.draftStep - 1);
            int32_t draftStepsPerIter = args.draftStep - 1;
            postStep = [&, draftStepsPerIter](int32_t t) {
                if ((t + 1) % draftStepsPerIter == 0)
                {
                    draftRunner->getCacheManager().commitSequenceLength(args.acceptRate, stream);
                }
            };
        }
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
    {
        modeName = "Spec Draft Prefill";
        LOG_INFO("Spec Draft Prefill mode: InputLen=%d, ReuseKVLen=%d", args.inputLen, args.reuseKVLen);

        reuseKVLenVec.assign(args.batchSize, args.reuseKVLen);

        inputsEmbeds = rt::Tensor(
            rt::Coords{args.batchSize, args.inputLen, hiddenSize}, rt::DeviceType::kGPU, dtype, "input_embeds");
        fillRandomData(inputsEmbeds, -1.0f, 1.0f, dtype, args.seed);

        baseModelHiddenStates = rt::Tensor(rt::Coords{args.batchSize, args.inputLen, baseHiddenSize},
            rt::DeviceType::kGPU, dtype, "base_hidden_states");
        fillRandomData(baseModelHiddenStates, -1.0f, 1.0f, dtype, args.seed);

        draftPrefillDraftModelHiddenStates = rt::Tensor(
            rt::Coords{args.batchSize, args.inputLen, hiddenSize}, rt::DeviceType::kGPU, dtype, "draft_hidden_input");
        // Set to zeros as per API documentation
        fillRandomData(draftPrefillDraftModelHiddenStates, 0.0f, 0.0f, dtype, args.seed);

        draftPrefillContextLengths = rt::Tensor(
            rt::Coords{args.batchSize}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "context_lengths");
        std::vector<int32_t> contextLengthsHost(args.batchSize, args.reuseKVLen + args.inputLen);
        std::memcpy(draftPrefillContextLengths.rawPointer(), contextLengthsHost.data(),
            contextLengthsHost.size() * sizeof(int32_t));

        outputLogits = rt::Tensor(
            rt::Coords{args.batchSize, vocabSize}, rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "output_logits");
        outputHiddenStates
            = rt::Tensor(rt::Coords{args.batchSize, hiddenSize}, rt::DeviceType::kGPU, dtype, "output_hidden");

        // RoPE cos/sin cache from draft runner (values don't affect benchmark performance)
        rt::Tensor& ropeCache = draftRunner->getRopeCosSinCacheTensor();

        resetState = [&]() {
            std::memcpy(reuseKVCacheLengths.rawPointer(), reuseKVLenVec.data(), reuseKVLenVec.size() * sizeof(int32_t));
            draftRunner->getCacheManager().resetForNewSequences(reuseKVCacheLengths, stream);
        };
        step = [&]() {
            return draftRunner->executeEaglePrefillStep(inputsEmbeds, baseModelHiddenStates,
                draftPrefillDraftModelHiddenStates, draftPrefillContextLengths, outputLogits, outputHiddenStates,
                ropeCache, stream);
        };
    }

    // ===== Phase 4: Warmup =====
    if (runWarmupLoop(modeName, args.warmup, resetState, step, stream) != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    // ===== Phase 5: Extract Layer Metadata =====
    if (args.extractLayerInfo.any())
    {
        layerMetadata = extractLayerMetadata(standaloneEngine.get(), args.extractLayerInfo);
        standaloneEngine.reset(); // Free standalone engine memory
    }

    // ===== Phase 6: Layer Profiling =====
    if (!args.noProfile)
    {
        if (runLayerProfilingLoop(modeName, args.iterations, !args.outputDir.empty(), resetState, step, layerMetadata,
                timesPerIter, layerTimings, stream)
            != EXIT_SUCCESS)
        {
            return EXIT_FAILURE;
        }

        // Write layer profiling CSV
        if (!args.outputDir.empty() && !layerTimings.empty())
        {
            auto outParams = args.toOutputParams();
            writeLayerInfoCsv(
                layerTimings, buildLayerCsvPath(args.outputDir, outParams), outParams, imageTokens, layerMetadata);
        }

        // A TensorRT profiler attached to an execution context cannot be fully detached in this benchmark process.
        // Keep --profile isolated to non-CUDA-graph layer profiling so E2E CUDA graph timing stays uncontaminated.
        logResultsSummary(args.toOutputParams(), timesPerIter, e2eTimeMsResult, imageTokens);
        if (!args.outputDir.empty())
        {
            LOG_INFO("Output CSV files saved to: %s", args.outputDir.c_str());
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return EXIT_SUCCESS;
    }

    // ===== Phase 7: E2E Timing =====
    // When osl=1 (default): all modes use runRepeatedE2ETiming (single-step, multiple iterations)
    // When osl>1: decode modes use runSequentialE2ETiming (full sequence decode, one run)
    // Determine numTokens for E2E CSV based on mode
    int32_t e2eNumTokens = 1;
    if (args.mode == BenchMode::kVISUAL)
    {
        e2eTimeMsResult = runRepeatedE2ETiming("Visual Encoder", args.iterations, resetState, step, stream);
        e2eNumTokens = 1;
    }
    else if (args.mode == BenchMode::kPREFILL)
    {
        e2eTimeMsResult = runRepeatedE2ETiming("Prefill", args.iterations, resetState, step, stream);
        e2eNumTokens = args.inputLen;
    }
    else if (args.mode == BenchMode::kEAGLE_DRAFT_PREFILL)
    {
        e2eTimeMsResult = runRepeatedE2ETiming("Spec Draft Prefill", args.iterations, resetState, step, stream);
        e2eNumTokens = args.inputLen;
    }
    else if (useSequentialE2E)
    {
        // osl > 1: full sequence decode, single run
        e2eTimeMsResult = runSequentialE2ETiming(
            modeName, decodeSteps, resetState, step, postStep, !args.noCudaGraph, captureGraph, stream);
        e2eNumTokens = decodeSteps;
    }
    else
    {
        // osl=1: single-step, multiple iterations (decode/spec-decode modes)
        e2eTimeMsResult = runRepeatedE2ETiming(
            modeName, args.iterations, resetState, step, stream, !args.noCudaGraph, captureGraph);
        e2eNumTokens = 1;
    }

    if (e2eTimeMsResult < 0)
    {
        return EXIT_FAILURE;
    }

    if (!args.outputDir.empty())
    {
        auto outParams = args.toOutputParams();
        writeE2ECsv(buildE2ECsvPath(args.outputDir, outParams), outParams, e2eTimeMsResult, e2eNumTokens, imageTokens);
    }

    // ===== Phase 8: Results Summary =====
    logResultsSummary(args.toOutputParams(), args.noProfile ? std::vector<KernelTimes>{} : timesPerIter,
        e2eTimeMsResult, imageTokens);

    if (!args.outputDir.empty())
    {
        LOG_INFO("Output CSV files saved to: %s", args.outputDir.c_str());
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
