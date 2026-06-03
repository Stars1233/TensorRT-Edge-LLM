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
"""Phi-4-Multimodal loading helpers shared by LoRA merge and quantization."""

import json
import os
import sys
import types
from importlib import util

import torch
import torch.nn as nn


def patch_phi4mm_lora_merge_cache() -> None:
    import transformers.cache_utils as cache_utils

    if not hasattr(cache_utils, "SlidingWindowCache"):
        from transformers.cache_utils import StaticCache

        class SlidingWindowCache(StaticCache):
            pass

        cache_utils.SlidingWindowCache = SlidingWindowCache


def patch_phi4mm_peft_generation(model: torch.nn.Module) -> None:
    inner_model = getattr(model, "model", None)
    if inner_model is None or hasattr(inner_model,
                                      "prepare_inputs_for_generation"):
        return

    def _prepare_inputs_for_generation(self, *args, **kwargs):
        return None

    inner_model.prepare_inputs_for_generation = (
        _prepare_inputs_for_generation.__get__(inner_model, type(inner_model)))


def load_phi4mm_module(model_dir: str):
    package_name = "tensorrt_edgellm_phi4mm_merge"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [model_dir]
        sys.modules[package_name] = pkg

    cfg_path = os.path.join(model_dir, "configuration_phi4mm.py")
    if os.path.exists(cfg_path):
        cfg_name = f"{package_name}.configuration_phi4mm"
        cfg_spec = util.spec_from_file_location(cfg_name, cfg_path)
        cfg_mod = util.module_from_spec(cfg_spec)
        cfg_mod.__package__ = package_name
        sys.modules[cfg_name] = cfg_mod
        sys.modules["configuration_phi4mm"] = cfg_mod
        assert cfg_spec is not None and cfg_spec.loader is not None
        cfg_spec.loader.exec_module(cfg_mod)

    module_name = f"{package_name}.modeling_phi4mm"
    mdl_path = os.path.join(model_dir, "modeling_phi4mm.py")
    spec = util.spec_from_file_location(module_name, mdl_path)
    module = util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[module_name] = module
    sys.modules["modeling_phi4mm"] = module
    assert spec is not None and spec.loader is not None

    patch_phi4mm_lora_merge_cache()
    spec.loader.exec_module(module)

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    has_builtin_lora = (config_dict.get("vision_lora") is not None
                        or config_dict.get("speech_lora") is not None)

    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        tied_keys = getattr(value, "_tied_weights_keys", None)
        if isinstance(tied_keys, list):
            value._tied_weights_keys = {key: key for key in tied_keys}

    if (not has_builtin_lora and hasattr(module, "Phi4MMForCausalLM")
            and hasattr(module, "Phi4MMModel")
            and hasattr(module, "Phi4MMPreTrainedModel")):

        def _phi4mm_init_without_builtin_lora(self, config):
            module.Phi4MMPreTrainedModel.__init__(self, config)
            self.model = module.Phi4MMModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size,
                                     config.vocab_size,
                                     bias=False)
            self.post_init()

        module.Phi4MMForCausalLM.__init__ = _phi4mm_init_without_builtin_lora

    if hasattr(module, "Phi4MMModel"):

        def _prepare_inputs_for_generation(self, *args, **kwargs):
            return None

        module.Phi4MMModel.prepare_inputs_for_generation = (
            _prepare_inputs_for_generation)

    if hasattr(module, "Phi4MMImageAudioEmbedding"):

        def _image_audio_embedding_init_vision_only(self, config, **kwargs):
            nn.Module.__init__(self)
            self.vocab_size = config.vocab_size
            self.image_input_id = kwargs.get("image_input_id", -1)
            self.audio_input_id = kwargs.get("audio_input_id", -10000)
            assert self.image_input_id != self.audio_input_id

            self.image_embd_layer_kwargs = kwargs["image_embd_layer"]
            self.image_embed = module.Phi4MMImageEmbedding(
                config, **self.image_embd_layer_kwargs)
            self.audio_embd_layer_kwargs = kwargs.get("audio_embd_layer", {})
            self.audio_embed = None
            self.input_image_embeds = None
            self.image_sizes = None
            self.image_attention_mask = None
            self.input_audio_embeds = None
            self.audio_embed_sizes = None

        module.Phi4MMImageAudioEmbedding.__init__ = (
            _image_audio_embedding_init_vision_only)

    return module


def load_phi4mm_model(model_dir: str,
                      torch_dtype,
                      patch_peft_generation: bool = False):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config._attn_implementation = "eager"
    config._attn_implementation_internal = "eager"
    module = load_phi4mm_module(model_dir)
    model = module.Phi4MMForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    if getattr(model.lm_head.weight, "is_meta", False):
        embed_weight = model.model.embed_tokens.weight
        model.lm_head.weight = nn.Parameter(embed_weight.detach().clone(),
                                            requires_grad=False)
    if patch_peft_generation:
        patch_phi4mm_peft_generation(model)
    return model
