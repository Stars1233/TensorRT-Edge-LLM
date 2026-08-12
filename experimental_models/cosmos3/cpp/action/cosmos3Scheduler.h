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

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace trt_edgellm
{
namespace cosmos3
{

//! \brief Flow-matching UniPC multistep scheduler for the Cosmos3 policy denoising loop.
//!
//! Ports the diffusers `UniPCMultistepScheduler` configured for flow matching (use_flow_sigmas,
//! prediction_type "flow_prediction") with a resolution-dependent timestep `shift` (flow_shift).
//!
//! The scheduler is device-resident: only the per-step scalar coefficients are computed on host;
//! every vector operation (x0 conversion, UniC corrector, UniP predictor) runs as one fused
//! elementwise CUDA kernel over the packed [video-latent ⧺ action-latent] state, so the state
//! never leaves the GPU during the denoising loop.
//!
//! The model is trained with a velocity / flow parameterization; with predictX0 the scheduler converts
//! the model output to a data (x0) prediction before the UniPC update, matching the reference pipeline.
//! When the component contract supplies an explicit timestep schedule (`reference_timesteps`), it is
//! used verbatim; otherwise initialize() implements the diffusers flow-shift schedule for any
//! step count.
class Cosmos3Scheduler
{
public:
    struct Config
    {
        int32_t numTrainTimesteps{1000};
        float shift{5.0F};      //!< flow_shift
        int32_t solverOrder{2}; //!< UniPC solver order (capacity); per-step order is clamped.
        bool predictX0{true};   //!< data-prediction form (UniPC default).
    };

    explicit Cosmos3Scheduler(Config const& config);

    //! \brief Initialize the flow-shift timestep/sigma schedule for `numSteps` inference steps.
    void initialize(int32_t numSteps);

    int32_t numInferenceSteps() const noexcept
    {
        return static_cast<int32_t>(mTimesteps.size());
    }

    float timestepAt(int32_t stepIdx) const
    {
        return mTimesteps.at(static_cast<size_t>(stepIdx));
    }

    //! \brief Allocate the device history buffers for a state of `numElems` floats (once).
    void prepare(int64_t numElems);

    void reset();

    //! \brief Advance the device-resident state by one UniPC step (corrector + predictor).
    //! \param modelOutput Device FLOAT32 model velocity prediction, `numElems` floats.
    //! \param sample      Device FLOAT32 packed state, updated in place.
    void step(rt::Tensor const& modelOutput, rt::Tensor& sample, int32_t stepIdx, cudaStream_t stream);

private:
    //! \brief Host-side scalar coefficients of one UniBH combine:
    //! out = cX*x + cM0*m0 + cMPrev*mPrev + cMT*modelT (mPrev/modelT terms optional).
    struct UniBHCoeffs
    {
        float cX{0.0F};
        float cM0{0.0F};
        float cMPrev{0.0F};
        float cMT{0.0F};
        bool useMPrev{false};
        bool useMT{false};
    };

    UniBHCoeffs computeUniBHCoeffs(int32_t order, int32_t idxT, int32_t idxS0, int32_t k, bool corrector) const;

    Config mConfig;
    std::vector<float> mTimesteps; //!< Descending timesteps, one per inference step.
    std::vector<float> mSigmas;    //!< sigmas[0..numSteps]; sigmas[numSteps] = 0.

    int64_t mNumElems{0};
    int64_t mCapacityElems{0};       //!< allocated element capacity; prepare() reshapes within it.
    std::vector<rt::Tensor> mX0Ring; //!< 3-slot device ring of x0 predictions (slot = stepIdx % 3).
    rt::Tensor mLastSample;          //!< Device sample handed to the previous predictor (UniC last_sample).
    rt::Tensor mScratch;             //!< Device scratch for the corrector output.
    int32_t mHistCount{0};           //!< Number of stored x0 predictions (capped at solverOrder).
    bool mHaveLastSample{false};
    int32_t mLowerOrderNums{0}; //!< Multistep warmup counter.
    int32_t mThisOrder{0};      //!< Order used by the previous predictor (consumed by the corrector).
};

} // namespace cosmos3
} // namespace trt_edgellm
