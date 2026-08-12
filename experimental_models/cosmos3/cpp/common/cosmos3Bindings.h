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

#include <string>

//! Cosmos3 (experimental) tensor binding names. These mirror the ONNX I/O emitted by
//! tensorrt_edgellm.scripts.export_cosmos3. Names that already exist in the core
//! binding_names namespace (rope_rotary_cos_sin, attention_pos_id) are reused from there and not
//! redefined here, to keep a single source of truth for the shared bindings.
namespace trt_edgellm
{
namespace cosmos3
{
namespace binding_names
{

//! @name GEN diffusion expert (one denoising step) bindings
//! @{

//! Noisy video latent input. Shape: [batch, latent_channel, t, h, w] (FLOAT32).
inline constexpr char const* kVideoLatent = "video_latent";

//! Noisy action latent input. Shape: [batch, action_len, max_action_dim] (FLOAT32).
//! action_len is dynamic: 0 for the video-only warmup step, action_chunk_size for action steps.
inline constexpr char const* kActionLatent = "action_latent";

//! Diffusion timestep (scaled by timestep_scale in-graph). Shape: [batch] (FLOAT32).
inline constexpr char const* kTimestep = "timestep";

//! Per-video-token noisy mask (0 => clean conditioning token, no timestep added).
//! Shape: [batch, num_video_tokens, 1] (FLOAT32).
inline constexpr char const* kTokenNoisyMask = "token_noisy_mask";

//! Per-action-token noisy mask. Shape: [batch, action_len, 1] (FLOAT32).
inline constexpr char const* kActionNoisyMask = "action_noisy_mask";

//! Predicted video velocity/output. Shape: [batch, latent_channel, t, h, w] (FLOAT32).
inline constexpr char const* kVideoPred = "video_pred";

//! Predicted action velocity/output. Shape: [batch, action_len, max_action_dim] (FLOAT32).
inline constexpr char const* kActionPred = "action_pred";

//! Per-layer frozen UND key/value conditioning inputs (cross-attention context).
//! Shape: [batch, und_len, num_kv_heads, head_dim] (FLOAT16).
//! @return Binding name like "und_k_layer00".
inline std::string formatUndKName(int32_t layerIdx)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "und_k_layer%02d", layerIdx);
    return std::string(buf);
}

//! @return Binding name like "und_v_layer00".
inline std::string formatUndVName(int32_t layerIdx)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "und_v_layer%02d", layerIdx);
    return std::string(buf);
}

//! @}

//! @name UND / text tower (policy prefill) bindings
//! @{

//! Token ids input. Shape: [batch, und_len] (INT32).
inline constexpr char const* kInputIds = "input_ids";

//! Per-layer K/V outputs emitted by the UND-prefill engine (consumed as GEN und_k/und_v inputs).
//! Reuses the formatUndKName/formatUndVName layout above.

//! @}

//! @name Wan VAE encoder (frame-0 conditioning) bindings
//! @{

//! Conditioning image input (preprocessed/normalized). Shape: [batch, 3, t_in, H, W] (FLOAT32).
inline constexpr char const* kVaeImage = "pixel_values";

//! Conditioning latent output. Shape: [batch, latent_channel, t, h, w] (FLOAT32).
inline constexpr char const* kCondLatent = "cond_latent";

//! @}

} // namespace binding_names
} // namespace cosmos3
} // namespace trt_edgellm
