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

#include <tuple>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

/*! \brief Past trajectory point (x, y, z) for Alpamayo 1 input */
using PastTrajectoryPoint = std::tuple<float, float, float>;

//! Chat-template / tokenizer strings for trajectory history blocks (pads replaced after tokenization).
inline constexpr char kTrajHistoryStartStr[] = "<|traj_history_start|>";
inline constexpr char kTrajHistoryPadStr[] = "<|traj_history|>";
inline constexpr char kTrajHistoryEndStr[] = "<|traj_history_end|>";

/*! \brief Future trajectory waypoint (accel, kappa) produced by action/diffusion head */
using FutureTrajectoryPoint = std::pair<float, float>;

} // namespace rt
} // namespace trt_edgellm
