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
"""Tests for OpenAI-compatible logit_bias request validation."""

import math

import pytest

from experimental.server.engine import (_MAX_LOGIT_BIAS_TOKENS, LLM,
                                        SamplingParams, _normalize_logit_bias,
                                        _validate_logit_bias_spec_decode)

_INT32_MAX = 2**31 - 1


def test_normalize_logit_bias_accepts_integer_like_keys_and_boundaries():
    assert _normalize_logit_bias(None) == {}
    assert _normalize_logit_bias({
        "1": -100,
        2: 0.5,
        "3": 100.0,
        str(_INT32_MAX): 0,
    }) == {
        1: -100.0,
        2: 0.5,
        3: 100.0,
        _INT32_MAX: 0.0,
    }


def test_normalize_logit_bias_rejects_too_many_entries():
    too_many_entries = {str(i): 0.0 for i in range(_MAX_LOGIT_BIAS_TOKENS + 1)}

    with pytest.raises(ValueError, match="max is"):
        _normalize_logit_bias(too_many_entries)


@pytest.mark.parametrize("token_id", [True, 1.25, object(), "not-an-int"])
def test_normalize_logit_bias_rejects_non_integer_token_ids(token_id):
    with pytest.raises(ValueError, match="not an integer"):
        _normalize_logit_bias({token_id: 0.0})


@pytest.mark.parametrize(
    "token_id",
    [-1, "-1", _INT32_MAX + 1, str(_INT32_MAX + 1)])
def test_normalize_logit_bias_rejects_token_ids_outside_nonnegative_int32(
        token_id):
    with pytest.raises(ValueError, match="token ID"):
        _normalize_logit_bias({token_id: 0.0})


@pytest.mark.parametrize("bias", [
    True, "1.0", math.nan, math.inf, -100.1, 100.1,
    pytest.param(10**400, id="overflowing-int")
])
def test_normalize_logit_bias_rejects_invalid_bias_values(bias):
    with pytest.raises(ValueError):
        _normalize_logit_bias({"1": bias})


@pytest.mark.parametrize(
    "logit_bias,disable_spec_decode,has_draft_model",
    [
        ({
            1: 1.0
        }, False, False),
        ({
            1: 1.0
        }, True, True),
        ({}, False, True),
    ],
)
def test_validate_logit_bias_spec_decode_accepts_compatible_states(
        logit_bias, disable_spec_decode, has_draft_model):
    _validate_logit_bias_spec_decode(
        logit_bias,
        disable_spec_decode=disable_spec_decode,
        has_draft_model=has_draft_model,
    )


def test_validate_logit_bias_spec_decode_rejects_active_spec_decode():
    with pytest.raises(ValueError, match="disable_spec_decode"):
        _validate_logit_bias_spec_decode(
            {1: 1.0},
            disable_spec_decode=False,
            has_draft_model=True,
        )


def test_hlapi_generate_rejects_logit_bias_with_active_spec_decode():

    class FakeRuntime:

        @staticmethod
        def has_draft_model():
            return True

    llm = object.__new__(LLM)
    llm._runtime = FakeRuntime()

    with pytest.raises(ValueError, match="disable_spec_decode"):
        llm.generate("hello", SamplingParams(logit_bias={1: 1.0}))


@pytest.mark.parametrize("stream", [False, True])
def test_api_rejects_logit_bias_with_active_spec_decode(stream):
    from fastapi.testclient import TestClient

    from experimental.server.api_server import _create_app

    class FakeLLM:
        _model_id = "test-model"
        has_draft_model = True

    response = TestClient(_create_app(FakeLLM())).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hello"
            }],
            "logit_bias": {
                "1": 1.0
            },
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert "disable_spec_decode" in response.json()["error"]


@pytest.mark.parametrize("stream", [False, True])
def test_api_rejects_overflowing_logit_bias(stream):
    from fastapi.testclient import TestClient

    from experimental.server.api_server import _create_app

    class FakeLLM:
        _model_id = "test-model"
        has_draft_model = False

    response = TestClient(_create_app(FakeLLM())).post(
        "/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": "hello"
            }],
            "logit_bias": {
                "1": 10**400
            },
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert "logit_bias" in response.json()["error"]
