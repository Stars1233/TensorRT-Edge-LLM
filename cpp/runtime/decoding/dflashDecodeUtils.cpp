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

#include "runtime/decoding/dflashDecodeUtils.h"

#include "common/checkMacros.h"

namespace trt_edgellm
{
namespace rt
{
namespace dflash_utils
{

int32_t runtimeBlockSize(DeploymentConfig const& deployment)
{
    ELLM_CHECK(deployment.specConfig.has_value(), "DFlash runtime block size requires specConfig.");
    ELLM_CHECK(
        deployment.specConfig->dflashBlockSize > 0, "DFlash runtime block size requires resolved dflashBlockSize.");
    return deployment.specConfig->dflashBlockSize;
}

bool shouldUseDDTree(DeploymentConfig const& deployment)
{
    if (deployment.specDecodeMode() != SpecDecodeMode::kDFlash || !deployment.specConfig.has_value())
    {
        return false;
    }
    return deployment.specConfig->draftingTopK > 1;
}

} // namespace dflash_utils
} // namespace rt
} // namespace trt_edgellm
