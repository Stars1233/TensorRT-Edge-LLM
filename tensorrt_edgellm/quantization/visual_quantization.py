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

import importlib
import math
from fractions import Fraction

import modelopt.torch.quantization as mtq
from datasets import (concatenate_datasets, get_dataset_config_names,
                      load_dataset)
from PIL import Image
from torch.utils.data import Dataset
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
except (ImportError, ModuleNotFoundError):
    Qwen3_5VisionModel = None

try:
    Qwen3OmniVisionEncoder = importlib.import_module(
        "transformers.models.qwen3_omni.modeling_qwen3_omni"
    ).Qwen3OmniVisionEncoder
except (ImportError, AttributeError):
    Qwen3OmniVisionEncoder = None

from ..visual_models.internvl3_model import InternVLVisionModel
from ..visual_models.phi4mm_model import Phi4MMVisionModel
from .quantization_utils import quantize_model


def resize_image_to_nearest_multiple(image, multiple):
    w, h = image.size
    # Candidate unit counts around floor/round/ceil
    w_div = w / multiple
    h_div = h / multiple
    w_candidates = {max(1, math.floor(w_div)), max(1, math.ceil(w_div))}
    h_candidates = {max(1, math.floor(h_div)), max(1, math.ceil(h_div))}

    orig_ratio = Fraction(w, h) if h else Fraction(1, 1)
    best = None
    best_cost = float("inf")
    best_delta = float("inf")
    for mu in w_candidates:
        for nu in h_candidates:
            ratio = Fraction(mu, nu)
            cost = abs(ratio - orig_ratio)
            new_w = mu * multiple
            new_h = nu * multiple
            delta = abs(new_w - w) + abs(new_h - h)
            if cost < best_cost or (cost == best_cost and delta < best_delta):
                best_cost = cost
                best_delta = delta
                best = (new_w, new_h)
    new_w, new_h = best if best is not None else (w, h)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image


class CalibrationDataset(Dataset):

    def __init__(self, data, preprocess_fn):
        self.data = data
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_data = self.data[idx]
        return self.preprocess_fn(raw_data)


def get_visual_calib_dataloader(
    model,
    processor,
    dataset_dir="lmms-lab/MMMU",
    block_size=448,
):
    # There are 2 possible dataset: lmms-lab/MMMU and MMMU/MMMU. The first one does not have any configs.
    # https://huggingface.co/datasets/lmms-lab/MMMU
    # https://huggingface.co/datasets/MMMU/MMMU

    if "lmms-lab/MMMU" in dataset_dir:
        # Default use MMMU_DEV. It's recommended to use your own dataset for calibration.
        dataset = load_dataset(dataset_dir, split="dev")
    elif "MMMU" in dataset_dir:
        dataset_configs = get_dataset_config_names(dataset_dir)
        dataset = concatenate_datasets([
            load_dataset(dataset_dir, config, split="dev")
            for config in dataset_configs
        ])
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_dir}."
        )

    def _collect_images(data, resize_multiple=None):
        """
        If `resize_multiple` is not None, images will be resized to the nearest multiple of `resize_multiple`.
        For models like InternVL/Phi-4MM, the inputs are expected to be aligned to vision block size.
        """
        image_inputs = []
        for key, value in data.items():
            if "image" in key and isinstance(value, Image.Image):
                if resize_multiple is not None:
                    value = resize_image_to_nearest_multiple(
                        value, resize_multiple)
                image_inputs.append(value.convert("RGB"))
        return image_inputs

    if isinstance(model, InternVLVisionModel):

        def preprocess_fn(data):
            image_inputs = _collect_images(data, resize_multiple=block_size)
            inputs = processor(images=image_inputs, return_tensors="pt")
            return {"pixel_values": inputs["pixel_values"].to(model.dtype)}

    elif isinstance(model, Phi4MMVisionModel):

        def preprocess_fn(data):
            image_inputs = _collect_images(data, resize_multiple=block_size)
            inputs = processor(images=image_inputs, )["input_image_embeds"][0]
            return {"pixel_values": inputs.to(model.dtype)}

    else:  # Qwen VIT series

        def preprocess_fn(data):
            image_inputs = _collect_images(data)
            inputs = processor(
                text="",
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            return {
                "hidden_states": inputs["pixel_values"].to(model.dtype),
                "grid_thw": inputs["image_grid_thw"],
            }

    return CalibrationDataset(dataset, preprocess_fn)


def quantize_visual(model, precision, processor, dataset_dir="lmms-lab/MMMU"):
    supported_model_types = [
        Qwen3VLVisionModel,
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2VisionTransformerPretrainedModel,
        InternVLVisionModel,
        Phi4MMVisionModel,
    ]
    if Qwen3_5VisionModel is not None:
        supported_model_types.insert(0, Qwen3_5VisionModel)
    if Qwen3OmniVisionEncoder is not None:
        supported_model_types.append(Qwen3OmniVisionEncoder)
    assert isinstance(model, tuple(supported_model_types)), \
        f"Invalid model type {type(model)}"
    assert precision in [
        "fp8"
    ], f"Only fp8(W8A8) is supported for visual model. You passed an unsupported precision: {precision}."
    assert "MMMU" in dataset_dir, f"Unsupported dataset name or local repo directory: {dataset_dir}."

    quant_config = mtq.FP8_DEFAULT_CFG.copy()

    # (Optional) Uncomment the following lines to enable FP8 MHA for static shape VIT, dynamic shape FP8 MHA fusion is not supported in TensorRT yet.
    # For Qwen VIT series, MHA is implemented in custom plugin, and FP8 MHA is not supported yet.
    # quant_config["quant_cfg"]["*[qkv]_bmm_quantizer"] = {
    #     "num_bits": (4, 3),
    #     "axis": None
    # }
    # quant_config["quant_cfg"]["*softmax_quantizer"] = {
    #     "num_bits": (4, 3),
    #     "axis": None
    # }

    # Disable Conv to avoid accuracy degradation
    quant_config["quant_cfg"]["nn.Conv3d"] = {"*": {"enable": False}}
    quant_config["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    # Determine block size: prefer config.vision_config.image_size, fallback to vision_model.crop_size, else 448
    block_size = 448
    vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
    if vision_cfg is not None and hasattr(vision_cfg, "image_size"):
        # Get the block size of InternVL3
        img_size = getattr(vision_cfg, "image_size", 448)
        # image_size can be int or [H, W]; prefer the first dimension if list/tuple
        block_size = int(img_size[0]) if isinstance(img_size,
                                                    (list,
                                                     tuple)) else int(img_size)
    else:
        # Get the block size of Phi-4MM
        block_size = getattr(getattr(model, "vision_model", None), "crop_size",
                             448)
    data_loader = get_visual_calib_dataloader(model, processor, dataset_dir,
                                              block_size)
    quantized_model = quantize_model(model, quant_config, data_loader)
    return quantized_model
