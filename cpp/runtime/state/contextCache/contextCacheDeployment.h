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

#include "runtime/config/deploymentConfig.h"

#include <cstdint>

namespace trt_edgellm::rt
{

//! Deployment-level execution family admitted by the context-cache coordinator.
enum class ContextCacheDeploymentKind : uint8_t
{
    kVanilla,
    kHybrid,
    kPureRecurrent,
    kEAGLE,
};

//! Validate the logical deployment contract and return its execution family.
//!
//! This is an early configuration gate. Request-level policy such as greedy-only EAGLE, media bypass, and
//! full-hidden-output bypass is selected before admission by the runtime. Physical binding compatibility is validated
//! separately after the engines are loaded.
//! @throws std::runtime_error when a requested cache-enabled deployment is outside the supported matrix; callers must
//! fail initialization rather than silently changing the configured execution mode
ContextCacheDeploymentKind validateContextCacheDeployment(DeploymentConfig const& deployment);

} // namespace trt_edgellm::rt
