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

import json

from experimental.server.tool_chat_template import (ToolChatTemplateFormatter,
                                                    needs_tool_chat_template)


def test_needs_tool_template():
    assert needs_tool_chat_template([{
        "role": "user",
        "content": "hi"
    }],
                                    tools=[{
                                        "type": "function"
                                    }])
    assert needs_tool_chat_template([{
        "role": "user",
        "content": "hi"
    }],
                                    tool_choice="required")
    assert needs_tool_chat_template([{
        "role": "assistant",
        "tool_calls": []
    }, {
        "role": "tool",
        "content": "42"
    }])
    assert not needs_tool_chat_template([{"role": "user", "content": "hi"}])


class _RecordingTemplateOwner:

    def __init__(self):
        self.kwargs = None
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return json.dumps(
            {
                "messages": messages,
                "tools": kwargs["tools"],
                "tool_choice": kwargs.get("tool_choice"),
                "add_generation_prompt": kwargs["add_generation_prompt"],
            },
            sort_keys=True)


def test_formats_tool_template():
    owner = _RecordingTemplateOwner()
    formatter = ToolChatTemplateFormatter([], template_owner=owner)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object"
            },
        },
    }]
    prompt = formatter.format(
        [{
            "role":
            "assistant",
            "content":
            None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}",
                },
            }],
        }, {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": {
                "temperature": 22
            },
        }],
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {
                "name": "get_weather"
            },
        },
    )

    formatted = json.loads(prompt)
    tool_call = formatted["messages"][0]["tool_calls"][0]
    tool_message = formatted["messages"][1]
    assert formatted["tools"] == tools
    assert formatted["tool_choice"] == {
        "type": "function",
        "function": {
            "name": "get_weather"
        },
    }
    assert formatted["add_generation_prompt"] is True
    assert tool_call["function"]["arguments"] == {"city": "Paris"}
    assert tool_message["content"] == '{"temperature": 22}'
