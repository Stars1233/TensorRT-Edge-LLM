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
"""Llama checkpoint-direct graph aligned with Transformers."""

from typing import Dict

import tensorrt as trt

from ...ops import (BuildContext, DecoderAttention, DecoderLayer, DecoderModel,
                    Linear, NetworkModule)
from ...ops import functional as F


class LlamaAttention(DecoderAttention):
    """Llama attention policy and model-family extension point."""

    def __init__(self,
                 ctx,
                 prefix,
                 layer_index=None,
                 sliding_window_size=None):
        super().__init__(ctx, prefix, layer_index, sliding_window_size=-1)


class LlamaDecoderLayer(DecoderLayer):
    """Llama decoder layer composed from shared primitive modules."""

    attention_class = LlamaAttention


class LlamaModel(DecoderModel):
    """Llama decoder stack and family extension point."""

    layer_class = LlamaDecoderLayer


class LlamaForCausalLM(NetworkModule):
    """Llama engine component and its explicit runtime I/O contract."""

    def __init__(self, ctx: BuildContext) -> None:
        super().__init__(ctx)
        self.model = LlamaModel(ctx)
        lm_head = ("lm_head" if ctx.weights.has("lm_head.weight")
                   or ctx.weights.has("lm_head.qweight") else
                   "model.embed_tokens")
        self.lm_head = Linear(ctx, lm_head)

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        io = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, -1, cfg.hidden_size)),
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.head_dim))
                for index in range(cfg.num_hidden_layers)
            ],
            "rope_rotary_cos_sin":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, -1, cfg.rotary_dim)),
            "context_lengths":
            self.add_input("context_lengths", trt.int32, (-1, )),
            "kvcache_start_index":
            self.add_input("kvcache_start_index", trt.int32, (-1, )),
            "kv_page_table":
            self.add_input("kv_page_table", trt.int32, (-1, 2, -1)),
            "last_token_ids":
            self.add_input("last_token_ids", trt.int64,
                           (-1, -1) if cfg.engine_role == "base" else (-1, 1)),
        }
        if cfg.engine_role == "base":
            io["attention_pos_id"] = self.add_input("attention_pos_id",
                                                    trt.int32, (-1, -1))
            io["attention_mask"] = self.add_input("attention_mask", trt.int32,
                                                  (-1, -1, -1))
        else:
            io["attention_pos_id"] = None
            io["attention_mask"] = None
        return io

    def forward(self,
                inputs_embeds,
                past_key_values,
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                kv_page_table,
                last_token_ids,
                attention_pos_id=None,
                attention_mask=None):
        outputs = {}
        hidden_states, present_key_values, all_hidden_states = self.model(
            inputs_embeds,
            past_key_values,
            rope_rotary_cos_sin,
            context_lengths,
            kvcache_start_index,
            kv_page_table,
            attention_mask=attention_mask,
            attention_pos_id=attention_pos_id)
        selected = F.gather_last_tokens(hidden_states, last_token_ids)
        outputs["logits"] = F.cast(self.lm_head(selected), trt.float32)
        if self.cfg.engine_role == "base":
            outputs["hidden_states"] = F.hidden_state_feedback(
                hidden_states, all_hidden_states, self.cfg)
        for index, present in enumerate(present_key_values):
            outputs[f"present_key_values_{index}"] = present
        return outputs
