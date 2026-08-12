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

import json

import transformers

from tensorrt_edgellm.chat_template import process_chat_template


class _DiffusionGemmaLikeTokenizer:

    chat_template = "fake"
    bos_token = "<bos>"

    def apply_chat_template(self, messages, **kwargs):
        del kwargs["tokenize"]
        add_generation_prompt = kwargs["add_generation_prompt"]
        enable_thinking = kwargs.get("enable_thinking", False)

        output = self.bos_token
        loop_messages = messages
        if enable_thinking or messages[0]["role"] == "system":
            output += "<|turn>system\n"
            if enable_thinking:
                output += "<|think|>\n"
            if messages[0]["role"] == "system":
                output += messages[0]["content"].strip()
                loop_messages = messages[1:]
            output += "<turn|>\n"

        for message in loop_messages:
            role = "model" if message["role"] == "assistant" else message[
                "role"]
            content = message["content"]
            if isinstance(content, list):
                content = next(
                    item.get("text", "") for item in content
                    if item.get("type") == "text")
            output += f"<|turn>{role}\n"
            output += content.strip()
            output += "<turn|>\n"

        if add_generation_prompt:
            output += "<|turn>model\n"
        return output


def test_process_chat_template_handles_global_bos_and_trim(
        monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
        }))

    tokenizer = _DiffusionGemmaLikeTokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: tokenizer)
    process_chat_template(str(model_dir), str(out_dir))

    data = json.loads((out_dir / "processed_chat_template.json").read_text())
    assert data["prompt_prefix"] == "<bos>"
    assert data["roles"]["system"]["prefix"] == "<|turn>system\n"
    assert data["roles"]["user"]["prefix"] == "<|turn>user\n"
    assert data["trim_content"] is True
    assert data["generation_prompt"] == "<|turn>model\n"
