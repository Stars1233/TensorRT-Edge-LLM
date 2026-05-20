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

#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenEncoder.h"
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace action_utils
{

//! Encode past trajectory points (x,y,z) to binned trajectory token IDs for Alpamayo-style input.
std::vector<tokenizer::Rank> trajectoryToTokenIds(std::vector<PastTrajectoryPoint> const& trajectoryHistory,
    tokenizer::Rank numTrajTokens, tokenizer::Rank trajTokenStart);

} // namespace action_utils
} // namespace rt
} // namespace trt_edgellm
