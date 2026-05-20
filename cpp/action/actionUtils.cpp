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

#include "action/actionUtils.h"
#include <cmath>
#include <tuple>

namespace trt_edgellm
{
namespace rt
{
namespace action_utils
{

std::vector<tokenizer::Rank> trajectoryToTokenIds(std::vector<PastTrajectoryPoint> const& trajectoryHistory,
    tokenizer::Rank numTrajTokens, tokenizer::Rank trajTokenStart)
{
    std::vector<tokenizer::Rank> tokenIds;
    std::tuple<float, float, float> point0, point;
    auto const ego_xyz_max = std::make_tuple(4.0f, 4.0f, 10.0f);
    auto const ego_xyz_min = std::make_tuple(-4.0f, -4.0f, -10.0f);

    auto clamp = [numTrajTokens](
                     int value) { return value < 0 ? 0 : (value >= numTrajTokens - 1 ? numTrajTokens - 1 : value); };

    auto tokenizePoint = [&](std::tuple<float, float, float> const& pt) {
        int const bx = clamp(static_cast<int>(std::round((std::get<0>(pt) - std::get<0>(ego_xyz_min))
                           / (std::get<0>(ego_xyz_max) - std::get<0>(ego_xyz_min)) * (numTrajTokens - 1))))
            + trajTokenStart;
        int const by = clamp(static_cast<int>(std::round((std::get<1>(pt) - std::get<1>(ego_xyz_min))
                           / (std::get<1>(ego_xyz_max) - std::get<1>(ego_xyz_min)) * (numTrajTokens - 1))))
            + trajTokenStart;
        int const bz = clamp(static_cast<int>(std::round((std::get<2>(pt) - std::get<2>(ego_xyz_min))
                           / (std::get<2>(ego_xyz_max) - std::get<2>(ego_xyz_min)) * (numTrajTokens - 1))))
            + trajTokenStart;
        tokenIds.push_back(static_cast<tokenizer::Rank>(bx));
        tokenIds.push_back(static_cast<tokenizer::Rank>(by));
        tokenIds.push_back(static_cast<tokenizer::Rank>(bz));
    };

    for (size_t i = 0; i < trajectoryHistory.size(); ++i)
    {
        auto const& item = trajectoryHistory[i];
        if (i == 0)
        {
            point0 = item;
            point = item;
        }
        else
        {
            point = std::make_tuple(std::get<0>(item) - std::get<0>(point0), std::get<1>(item) - std::get<1>(point0),
                std::get<2>(item) - std::get<2>(point0));
            point0 = item;
        }
        tokenizePoint(point);
    }

    return tokenIds;
}

} // namespace action_utils
} // namespace rt
} // namespace trt_edgellm
