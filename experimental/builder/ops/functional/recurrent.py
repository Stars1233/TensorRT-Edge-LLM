# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Recurrent and stateful sequence operations."""

from typing import Optional, Sequence, Tuple

from ..tensor import Tensor
from ._operation import operation, supports_operation_attribute


def causal_conv1d(
    hidden_states: Tensor,
    weight: Tensor,
    bias: Tensor,
    conv_state: Tensor,
    context_lengths: Tensor,
    groups: int,
    padding: int,
    spec_metadata: Sequence[Tensor] = (),
    use_ddtree: bool = False,
    use_intermediate: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Run stateful depthwise causal convolution."""
    use_intermediate = use_intermediate or bool(spec_metadata)
    modern_abi = supports_operation_attribute("causal_conv1d", "use_ddtree")
    if use_ddtree and not modern_abi:
        raise RuntimeError(
            "loaded causal_conv1d operation does not support DDTree inputs")
    if modern_abi and use_intermediate and not spec_metadata:
        raise RuntimeError(
            "modern causal_conv1d MTP ABI requires phase metadata")
    attributes = {
        "stride": 1,
        "padding": padding,
        "dilation": 1,
        "groups": groups,
        "use_mtp": int(use_intermediate and not use_ddtree),
    }
    inputs = [hidden_states, weight, bias, conv_state, context_lengths]
    if modern_abi:
        attributes["use_ddtree"] = int(use_ddtree)
        inputs.extend(spec_metadata)
    result = operation("causal_conv1d",
                       inputs,
                       output_count=3 if use_intermediate else 2,
                       **attributes)
    hidden_states, conv_state = result[:2]
    intermediate = result[2] if use_intermediate else None
    return hidden_states, conv_state, intermediate


def gated_delta_net(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    state: Tensor,
    context_lengths: Tensor,
    key_head_dim: int,
    value_head_dim: int,
    spec_metadata: Sequence[Tensor] = (),
    use_ddtree: bool = False,
    use_intermediate: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Run recurrent Gated DeltaNet attention."""
    use_intermediate = use_intermediate or bool(spec_metadata)
    modern_abi = supports_operation_attribute("gated_delta_net", "use_ddtree")
    if use_ddtree and not modern_abi:
        raise RuntimeError(
            "loaded gated_delta_net operation does not support DDTree inputs")
    if modern_abi and use_intermediate and not spec_metadata:
        raise RuntimeError(
            "modern gated_delta_net MTP ABI requires phase metadata")
    attributes = {
        "k_dim": key_head_dim,
        "v_dim": value_head_dim,
        "use_mtp": int(use_intermediate and not use_ddtree),
    }
    inputs = [q, k, v, a, b, a_log, dt_bias, state, context_lengths]
    if modern_abi:
        attributes["use_ddtree"] = int(use_ddtree)
        inputs.extend(spec_metadata)
    result = operation("gated_delta_net",
                       inputs,
                       output_count=3 if use_intermediate else 2,
                       **attributes)
    hidden_states, state = result[:2]
    intermediate = result[2] if use_intermediate else None
    return hidden_states, state, intermediate


def update_ssm_state(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    dt: Tensor,
    dt_bias: Tensor,
    state: Tensor,
    context_lengths: Tensor,
    state_start_index: Tensor,
    dim: int,
    dstate: int,
    nheads: int,
    ngroups: int,
) -> Tuple[Tensor, Tensor]:
    """Run selective state update."""
    return operation("update_ssm_state", [
        x, a, b, c, d, dt, dt_bias, state, context_lengths, state_start_index
    ],
                     output_count=2,
                     dim=dim,
                     dstate=dstate,
                     nheads=nheads,
                     ngroups=ngroups,
                     dt_softplus=1,
                     chunk_size=1,
                     time_step_limit=[0.0, float("inf")])
