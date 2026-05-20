# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Command-line script for exporting the Alpamayo action expert to ONNX.

Usage:
    tensorrt-edgellm-export-action --model_dir /path/to/alpamayo --output_dir /path/to/output
    tensorrt-edgellm-export-action --model_dir /path/to/alpamayo --output_dir /path/to/output --device cuda:0
"""

import argparse
import sys
import traceback

from tensorrt_edgellm.onnx_export.action_export import action_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Alpamayo action expert to ONNX format")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the Alpamayo-R1 model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=
        "Path to save the exported ONNX model (model.onnx and config.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load the model on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16"],
        help="Export dtype (default: fp16)",
    )
    parser.add_argument(
        "--max_kv_cache_capacity",
        type=int,
        default=4096,
        help=
        "Maximum KV cache capacity. Must match maxKVCacheCapacity that the LLM engine is built with (default: 4096)",
    )

    args = parser.parse_args()

    try:
        action_export(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            device=args.device,
            dtype=args.dtype,
            max_kv_cache_capacity=args.max_kv_cache_capacity,
        )
        print("Action expert export completed successfully!")
    except Exception as e:
        print(f"Error during action expert export: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
