# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Vocabulary-reduction utilities for ``tensorrt_edgellm``.

The package separates two concerns:

* ``selection`` builds a reduced vocabulary map from task data.
* ``onnx_export`` applies an existing map to the model during ONNX export.
"""

from .onnx_export import (apply_reduced_vocab, apply_reduced_vocab_from_dir,
                          copy_reduced_vocab_artifacts, load_reduced_vocab_map,
                          should_apply_reduced_vocab_before_repacking)
from .selection import (extract_d2t_required_tokens, get_special_tokens,
                        get_vocab_size, input_aware_filter,
                        input_frequency_filter, reduce_vocab_size)

__all__ = [
    "apply_reduced_vocab",
    "apply_reduced_vocab_from_dir",
    "copy_reduced_vocab_artifacts",
    "extract_d2t_required_tokens",
    "get_special_tokens",
    "get_vocab_size",
    "input_aware_filter",
    "input_frequency_filter",
    "load_reduced_vocab_map",
    "reduce_vocab_size",
    "should_apply_reduced_vocab_before_repacking",
]
