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

namespace trt_edgellm
{
namespace rt
{
namespace dflash_utils
{

//! Return the DFlash draft block horizon used by runtime draft forward.
//!
//! DeploymentConfig resolves explicit/user-supplied DFlash horizons before the
//! decoder is created, so the runtime reads one consolidated value.
int32_t runtimeBlockSize(DeploymentConfig const& deployment);

//! DFlash uses a linear-tree proposal when draftingTopK == 1 and a branching
//! DDTree proposal when draftingTopK > 1.
bool shouldUseDDTree(DeploymentConfig const& deployment);

} // namespace dflash_utils
} // namespace rt
} // namespace trt_edgellm
