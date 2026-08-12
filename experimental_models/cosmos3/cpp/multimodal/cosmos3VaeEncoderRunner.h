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

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace cosmos3
{

//! \brief Runs the Wan VAE encoder engine over the frame-0 conditioning image to produce the GEN
//! conditioning latent. The Wan VAE decoder is intentionally not built (video pixels are out of scope).
class Cosmos3VaeEncoderRunner
{
public:
    //! \param engineDir Directory containing vae_encoder.engine and config.json.
    //! \param stream CUDA stream.
    Cosmos3VaeEncoderRunner(std::string const& engineDir, cudaStream_t stream);
    ~Cosmos3VaeEncoderRunner() noexcept = default;

    int64_t getRequiredContextMemorySize() const;
    bool setContextMemory(rt::Tensor& sharedContextMemory);

    //! \brief Encode a preprocessed pixel tensor (FLOAT32, normalized) to the conditioning latent.
    //! \param pixelValues Device tensor, shape [B, 3, t_in, H, W].
    //! \return Device tensor holding cond_latent [B, latentChannel, t, h, w]. Owned by this runner.
    rt::Tensor& encode(rt::Tensor const& pixelValues, cudaStream_t stream);

    //! \brief Latent shape [B, C, t, h, w] produced by the last encode().
    std::vector<int64_t> const& latentShape() const noexcept
    {
        return mLatentShape;
    }

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    rt::Tensor mCondLatentDevice;
    std::vector<int64_t> mLatentShape;
};

} // namespace cosmos3
} // namespace trt_edgellm
