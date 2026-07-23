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
"""
End-to-end tests for the experimental Python server (pybind11 runtime).

Tests the full pipeline: build pybind extension -> load TRT engine via
Python API -> run inference -> validate output.
"""
import logging
import os
import shlex
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig


def test_build_pybind(env_config: EnvironmentConfig,
                      remote_config: Optional[RemoteConfig],
                      test_logger: logging.Logger):
    """Build the pybind extension by reconfiguring the existing build directory.

    Uses subdirectory mode (BUILD_PYTHON_BINDINGS=ON) so all cmake
    variables (CUDA, TRT, toolchain) are inherited from the prior
    test_build_project configuration.
    """
    build_dir = env_config.build_dir

    repo_root = '.' if remote_config else env_config.llm_sdk_dir
    pybind_venv = 'venv/server'
    install_cmd = (f'cd {shlex.quote(repo_root)}'
                   f' && python3 -m venv {pybind_venv}'
                   f' && {pybind_venv}/bin/pip install -q'
                   ' -r requirements-server.txt')
    result = run_command(cmd=['bash', '-c', install_cmd],
                         remote_config=remote_config,
                         timeout=120,
                         logger=test_logger)
    if not result['success']:
        pytest.fail(
            f"Failed to install server dependencies: {result.get('error')}")

    pybind_python = f'{repo_root}/{pybind_venv}/bin/python'
    pybind11_dir_expr = (
        f'$({shlex.quote(pybind_python)}'
        ' -c "import pybind11; print(pybind11.get_cmake_dir())")')
    build_cmd = (f'PYBIND11_DIR={pybind11_dir_expr}'
                 f' && cd {build_dir}'
                 f' && cmake .. -DBUILD_PYTHON_BINDINGS=ON'
                 f' -Dpybind11_DIR=$PYBIND11_DIR'
                 f' && make -j$(nproc) _edgellm_runtime')

    with timer_context("Building pybind extension", test_logger):
        result = run_command(cmd=['bash', '-c', build_cmd],
                             remote_config=remote_config,
                             timeout=600,
                             logger=test_logger)

    if not result['success']:
        pytest.fail(f"Pybind build failed: {result.get('error')}")

    pybind_output_dir = f'{build_dir}/pybind'
    result = run_command(
        cmd=['bash', '-c', f'ls {pybind_output_dir}/*_edgellm_runtime*.so'],
        remote_config=remote_config,
        timeout=10,
        logger=test_logger)
    if not result['success']:
        pytest.fail(
            f"_edgellm_runtime.so not found in {pybind_output_dir} after build"
        )


class TestServerPipeline:
    """E2E tests for inference via the experimental Python server API."""

    def test_server_inference(self, test_param: str,
                              executable_files: Dict[str, str],
                              remote_config: Optional[RemoteConfig],
                              test_logger: logging.Logger,
                              env_config: EnvironmentConfig) -> None:
        """Test inference using the pybind Python runtime directly."""
        is_vlm = "-mnit" in test_param
        model_type = ModelType.VLM if is_vlm else ModelType.LLM
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        multimodal_engine_dir = config.get_visual_engine_dir(
        ) if is_vlm else ""
        test_logger.info("Using engine dir: %s", engine_dir)
        if multimodal_engine_dir:
            test_logger.info("Using visual engine dir: %s",
                             multimodal_engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        script = f"""\
import sys, os
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
multimodal_engine_dir = {multimodal_engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, multimodal_engine_dir, {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.temperature = 0.7
request.top_p = 0.9
request.top_k = 50
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.enable_thinking = False
request.disable_spec_decode = False

response = runtime.handle_request(request)
text = response.output_texts[0] if response.output_texts else ''
ids = response.output_ids[0] if response.output_ids else []
print(f'OUTPUT_TEXT_LEN={{len(text)}}')
print(f'OUTPUT_IDS_LEN={{len(ids)}}')
print(f'OUTPUT_TEXT={{text[:200]}}')
assert len(text) > 0, 'Empty output text'
assert len(ids) > 0, 'Empty output token ids'
print('SERVER_INFERENCE_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server inference for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server inference failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'SERVER_INFERENCE_PASSED' not in output:
            pytest.fail(
                f"Server inference did not produce expected output. Output:\n{output}"
            )

    def test_server_inference_with_audio(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Audio pybind path: in-memory PCM -> AudioData -> Request.audio_buffers -> runtime."""
        is_asr = "-asr" in test_param
        is_omni = "-omni" in test_param
        if not (is_asr or is_omni):
            pytest.skip(
                "audio test requires test_param with '-asr' or '-omni'")
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        audio_engine_dir = (getattr(config, "get_audio_engine_dir",
                                    lambda: "")()
                            or os.environ.get("AUDIO_ENCODER_ENGINE_DIR", ""))
        if not audio_engine_dir:
            pytest.skip("AUDIO_ENCODER_ENGINE_DIR not set")
        test_wav = (getattr(config, "get_audio_test_wav", lambda: "")()
                    or os.environ.get("AUDIO_TEST_WAV", ""))
        if not test_wav:
            pytest.skip("AUDIO_TEST_WAV not set")

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        script = f"""\
import sys, os, importlib.util
sys.path.insert(0, {pybind_build_dir!r})
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
with open({test_wav!r}, 'rb') as _f: _audio_bytes = _f.read()
runtime = rt.LLMRuntime({engine_dir!r}, {audio_engine_dir!r}, {{}})
runtime.capture_decoding_cuda_graph()
request = rt.LLMGenerationRequest()
msg = rt.Message(); msg.role = 'user'; msg.contents = [rt.MessageContent('audio', '')]
req = rt.Request(messages=[msg]); req.image_buffers = []
req.audio_buffers = [rt.load_audio_buffer_from_bytes(_audio_bytes)]
request.requests = [req]; request.temperature = 1.0; request.top_p = 1.0; request.top_k = 50
request.max_generate_length = 128; request.apply_chat_template = True; request.add_generation_prompt = True
response = runtime.handle_request(request)
ids = response.output_ids[0] if response.output_ids else []
assert len(ids) > 0
print('SERVER_INFERENCE_WITH_AUDIO_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = {
            "LD_LIBRARY_PATH":
            f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
        } if env_config.trt_package_dir else None
        with timer_context(f"Server audio inference for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result[
                'success'] or 'SERVER_INFERENCE_WITH_AUDIO_PASSED' not in result.get(
                    'output', ''):
            pytest.fail(f"audio inference failed:\n{result.get('output', '')}")

    def test_server_streaming(self, test_param: str,
                              executable_files: Dict[str, str],
                              remote_config: Optional[RemoteConfig],
                              test_logger: logging.Logger,
                              env_config: EnvironmentConfig) -> None:
        """Test streaming inference using StreamChannel via pybind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Using engine dir: %s", engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Count from 1 to 10."
        max_tokens = 128

        script = f"""\
import sys, os, threading
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 0.7
request.top_p = 0.9
request.top_k = 50
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.enable_thinking = False
request.disable_spec_decode = False

def run_inference():
    runtime.handle_request(request)

worker = threading.Thread(target=run_inference, daemon=True)
worker.start()

chunks = []
while True:
    chunk = channel.wait_pop(timeout_ms=500)
    if chunk is None:
        if channel.is_finished() or channel.is_cancelled():
            break
        continue
    chunks.append(chunk)
    if chunk.finished:
        break

worker.join(timeout=10)

total_text = ''.join(c.text for c in chunks)
total_ids = sum(len(c.token_ids) for c in chunks)
print(f'STREAM_CHUNKS={{len(chunks)}}')
print(f'STREAM_TEXT_LEN={{len(total_text)}}')
print(f'STREAM_IDS={{total_ids}}')
print(f'STREAM_TEXT={{total_text[:200]}}')
assert len(chunks) > 1, f'Expected multiple chunks, got {{len(chunks)}}'
assert len(total_text) > 0, 'Empty streamed text'
assert any(c.finished for c in chunks), 'No terminal chunk received'
print('SERVER_STREAMING_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server streaming for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server streaming failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'SERVER_STREAMING_PASSED' not in output:
            pytest.fail(
                f"Server streaming did not produce expected output. Output:\n{output}"
            )

    def test_server_inference_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test non-streaming inference returns per-token logprobs via pybind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Server inference with logprobs: engine=%s",
                         engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3

        script = f"""\
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
sys.modules['_edgellm_runtime'] = rt  # so api_server/engine reuse this exact module
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, '', {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.num_logprobs = {num_logprobs}

response = runtime.handle_request(request)
ids = response.output_ids[0] if response.output_ids else []
lps = response.logprobs[0] if response.logprobs else []
print(f'OUTPUT_IDS_LEN={{len(ids)}}')
print(f'LOGPROBS_STEPS={{len(lps)}}')
assert len(ids) > 0, 'Empty output ids'
assert len(lps) == len(ids), f'logprobs steps {{len(lps)}} != token count {{len(ids)}}'
for step in lps:
    assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
    for e in step:
        assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
        assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
        assert isinstance(e.piece, bytes), f'piece should be bytes, got {{type(e.piece)}}'

# Also exercise the OpenAI-compatible formatting layer on the same raw response.
from experimental.server import api_server
obj = api_server._format_logprobs(response)
assert obj is not None and 'content' in obj, f'missing content: {{obj}}'
oc = obj['content']
assert len(oc) == len(ids), f'content {{len(oc)}} != token count {{len(ids)}}'
for c in oc:
    assert set(c) >= {{'token', 'token_id', 'bytes', 'logprob', 'top_logprobs'}}, f'missing keys: {{sorted(c)}}'
    assert isinstance(c['token'], str) and isinstance(c['token_id'], int) and isinstance(c['bytes'], list)
    assert c['logprob'] is None or c['logprob'] <= 0.0, f'bad chosen logprob {{c["logprob"]}}'
    top = c['top_logprobs']
    assert len(top) == {num_logprobs}, f'expected {num_logprobs} top_logprobs, got {{len(top)}}'
    for t in top:
        assert set(t) >= {{'token', 'token_id', 'bytes', 'logprob'}}, f'missing top keys: {{sorted(t)}}'
        assert isinstance(t['token'], str) and isinstance(t['bytes'], list) and t['logprob'] <= 0.0
print('SERVER_INFERENCE_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server inference with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server inference with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'SERVER_INFERENCE_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server inference logprobs output:\n{result.get('output', '')}"
            )

    def test_server_streaming_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test streaming inference delivers per-token logprobs on each chunk via pybind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Server streaming with logprobs: engine=%s",
                         engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3

        script = f"""\
import sys, os, threading
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.num_logprobs = {num_logprobs}

def run_inference():
    runtime.handle_request(request)

worker = threading.Thread(target=run_inference, daemon=True)
worker.start()

chunks = []
while True:
    chunk = channel.wait_pop(timeout_ms=500)
    if chunk is None:
        if channel.is_finished() or channel.is_cancelled():
            break
        continue
    chunks.append(chunk)
    if chunk.finished:
        break

worker.join(timeout=10)

total_tokens = sum(len(c.token_ids) for c in chunks)
total_lp_steps = sum(len(c.logprobs) for c in chunks)
print(f'STREAM_TOTAL_TOKENS={{total_tokens}}')
print(f'STREAM_TOTAL_LP_STEPS={{total_lp_steps}}')
assert total_tokens > 0, 'No tokens received'
assert total_lp_steps == total_tokens, f'logprob steps {{total_lp_steps}} != token count {{total_tokens}}'
for chunk in chunks:
    for step in chunk.logprobs:
        assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
        for e in step:
            assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
            assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
            assert isinstance(e.piece, bytes), f'piece should be bytes, got {{type(e.piece)}}'
print('SERVER_STREAMING_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server streaming with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server streaming with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'SERVER_STREAMING_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server streaming logprobs output:\n{result.get('output', '')}"
            )

    def test_server_inference_with_stop(self, test_param: str,
                                        executable_files: Dict[str, str],
                                        remote_config: Optional[RemoteConfig],
                                        test_logger: logging.Logger,
                                        env_config: EnvironmentConfig) -> None:
        """Non-streaming pybind path: Request.stop_strings + response.finish_reasons round-trip."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"

        script = f"""\
import sys, os
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

runtime = rt.LLMRuntime({engine_dir!r}, '', {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
req.stop_strings = [{stop!r}]
request.requests = [req]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = 128
request.apply_chat_template = True
request.add_generation_prompt = True

response = runtime.handle_request(request)
text = response.output_texts[0] if response.output_texts else ''
reasons = list(response.finish_reasons) if response.finish_reasons else []
print(f'OUTPUT_TEXT={{text!r}}')
print(f'FINISH_REASON={{reasons[0] if reasons else None}}')
assert {stop!r} not in text, f'Stop string leaked into output: {{text!r}}'
assert len(reasons) == 1, f'Expected 1 finish_reason, got {{len(reasons)}}'
assert reasons[0] == rt.FinishReason.STOP_WORDS, f'Expected STOP_WORDS, got {{reasons[0]}}'
print('SERVER_INFERENCE_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"Server inference with stop for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server inference with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'SERVER_INFERENCE_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server inference with stop did not produce expected output. Output:\n{result.get('output', '')}"
            )

    def test_server_streaming_with_stop(self, test_param: str,
                                        executable_files: Dict[str, str],
                                        remote_config: Optional[RemoteConfig],
                                        test_logger: logging.Logger,
                                        env_config: EnvironmentConfig) -> None:
        """Streaming pybind path: stop string trims chunk text, terminal chunk has STOP_WORDS reason."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"

        script = f"""\
import sys, os, threading
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

runtime = rt.LLMRuntime({engine_dir!r}, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
req.stop_strings = [{stop!r}]
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = 128
request.apply_chat_template = True
request.add_generation_prompt = True

threading.Thread(target=lambda: runtime.handle_request(request), daemon=True).start()
chunks = []
while True:
    c = channel.wait_pop(timeout_ms=500)
    if c is None:
        if channel.is_finished() or channel.is_cancelled(): break
        continue
    chunks.append(c)
    if c.finished: break

text = ''.join(c.text for c in chunks)
terminal = next((c for c in chunks if c.finished), None)
print(f'STREAM_TEXT={{text!r}}')
print(f'TERMINAL_REASON={{terminal.reason if terminal else None}}')
assert {stop!r} not in text, f'Stop string leaked into streamed text: {{text!r}}'
assert terminal is not None, 'No terminal chunk'
assert terminal.reason == rt.FinishReason.STOP_WORDS, f'Expected STOP_WORDS, got {{terminal.reason}}'
print('SERVER_STREAMING_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"Server streaming with stop for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server streaming with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'SERVER_STREAMING_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"Server streaming with stop did not produce expected output. Output:\n{result.get('output', '')}"
            )

    def test_server_inference_length_finish_reason(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Verify response.finish_reasons reports LENGTH when max_generate_length hit (no stops set)."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Write a long detailed essay about transformer neural networks."

        script = f"""\
import sys, os
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

runtime = rt.LLMRuntime({engine_dir!r}, '', {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.temperature = 0.0
request.top_p = 1.0
request.top_k = 1
request.max_generate_length = 8  # tiny → should hit LENGTH
request.apply_chat_template = True
request.add_generation_prompt = True

response = runtime.handle_request(request)
reasons = list(response.finish_reasons) if response.finish_reasons else []
print(f'FINISH_REASON={{reasons[0] if reasons else None}}')
assert len(reasons) == 1
assert reasons[0] == rt.FinishReason.LENGTH, f'Expected LENGTH, got {{reasons[0]}}'
print('SERVER_INFERENCE_LENGTH_REASON_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"Server inference length-reason for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Length-reason test failed: {result.get('error', 'Unknown')}")
        if 'SERVER_INFERENCE_LENGTH_REASON_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"Length-reason test did not produce expected output. Output:\n{result.get('output', '')}"
            )


class TestHLAPI:
    """E2E tests for the high-level LLM Python API with pre-built engines."""

    @staticmethod
    def _build_hlapi_env_setup(trt_package_dir: str = "") -> str:
        """Return inline script preamble that sets up sys.path and LD_LIBRARY_PATH."""
        parts = [
            "import sys, os",
            "sys.path.insert(0, os.getcwd())",
        ]
        if trt_package_dir:
            parts.append("os.environ.setdefault('LD_LIBRARY_PATH', '')")
            parts.append(
                f"os.environ['LD_LIBRARY_PATH'] += ':{trt_package_dir}/lib'")
        return "\n".join(parts)

    def test_hlapi_generate(self, test_param: str, executable_files: Dict[str,
                                                                          str],
                            remote_config: Optional[RemoteConfig],
                            test_logger: logging.Logger,
                            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate() with a pre-built engine directory."""
        is_vlm = "-mnit" in test_param
        model_type = ModelType.VLM if is_vlm else ModelType.LLM
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        visual_engine_dir = config.get_visual_engine_dir() if is_vlm else ""
        test_logger.info("HLAPI generate: engine=%s visual=%s", engine_dir,
                         visual_engine_dir or "(none)")

        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r}, visual_engine_dir={visual_engine_dir!r})
outputs = llm.generate(
    [{prompt!r}],
    SamplingParams(temperature=0.7, max_tokens={max_tokens}),
)
text = outputs[0].text
ids = outputs[0].token_ids
print(f'HLAPI_TEXT_LEN={{len(text)}}')
print(f'HLAPI_IDS_LEN={{len(ids)}}')
print(f'HLAPI_TEXT={{text[:200]}}')
assert len(text) > 0, 'Empty output text'
assert len(ids) > 0, 'Empty output token ids'
print('HLAPI_GENERATE_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI generate for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI generate failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'HLAPI_GENERATE_PASSED' not in output:
            pytest.fail(
                f"HLAPI generate did not produce expected output. Output:\n{output}"
            )

    def test_hlapi_generate_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate() returns per-token logprobs when num_logprobs > 0."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("HLAPI generate with logprobs: engine=%s", engine_dir)

        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
outputs = llm.generate(
    [[{{"role": "user", "content": {prompt!r}}}]],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens={max_tokens},
                   num_logprobs={num_logprobs}),
)
out = outputs[0]
ids = out.token_ids
lps = out.logprobs
print(f'HLAPI_IDS_LEN={{len(ids)}}')
print(f'HLAPI_LOGPROBS_STEPS={{len(lps)}}')
assert len(ids) > 0, 'Empty output token ids'
assert len(lps) == len(ids), f'logprobs steps {{len(lps)}} != token count {{len(ids)}}'
for step in lps:
    assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
    for e in step:
        assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
        assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
        assert isinstance(e.token, str), f'token should be str, got {{type(e.token)}}'
        assert isinstance(e.bytes, list), f'bytes should be list, got {{type(e.bytes)}}'
print('HLAPI_GENERATE_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI generate with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI generate with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'HLAPI_GENERATE_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI generate logprobs output:\n{result.get('output', '')}")

    def test_hlapi_generate_with_audio(self, test_param: str,
                                       executable_files: Dict[str, str],
                                       remote_config: Optional[RemoteConfig],
                                       test_logger: logging.Logger,
                                       env_config: EnvironmentConfig) -> None:
        """HLAPI audio path: OpenAI input_audio.data base64 wav -> transcription."""
        is_asr = "-asr" in test_param
        is_omni = "-omni" in test_param
        if not (is_asr or is_omni):
            pytest.skip(
                "audio HLAPI test requires '-asr' or '-omni' test_param")
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        audio_engine_dir = (getattr(config, "get_audio_engine_dir",
                                    lambda: "")()
                            or os.environ.get("AUDIO_ENCODER_ENGINE_DIR", ""))
        if not audio_engine_dir:
            pytest.skip("AUDIO_ENCODER_ENGINE_DIR not set")
        test_wav = (getattr(config, "get_audio_test_wav", lambda: "")()
                    or os.environ.get("AUDIO_TEST_WAV", ""))
        if not test_wav:
            pytest.skip("AUDIO_TEST_WAV not set")

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")
        script = f"""\
{setup}
import base64
with open({test_wav!r}, 'rb') as f:
    wav_b64 = base64.b64encode(f.read()).decode()
from experimental.server import LLM, SamplingParams
llm = LLM(engine_dir={engine_dir!r}, visual_engine_dir={audio_engine_dir!r})
messages = [{{'role': 'user', 'content': [
    {{'type': 'input_audio', 'input_audio': {{'data': wav_b64, 'format': 'wav'}}}}
]}}]
outputs = llm.generate([messages], SamplingParams(temperature=1.0, max_tokens=128))
assert len(outputs[0].token_ids) > 0
print('HLAPI_GENERATE_WITH_AUDIO_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = {
            "LD_LIBRARY_PATH":
            f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
        } if env_config.trt_package_dir else None
        with timer_context(f"HLAPI audio generate for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result[
                'success'] or 'HLAPI_GENERATE_WITH_AUDIO_PASSED' not in result.get(
                    'output', ''):
            pytest.fail(
                f"HLAPI audio generate failed:\n{result.get('output', '')}")

    def test_hlapi_generate_with_logit_bias(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Validate non-streaming HLAPI logit_bias behavior.

        Runs generation in a subprocess against a real engine. The +100 case
        selects a non-special tokenizer ID and verifies that deterministic
        generation returns it for both the prefill-sampled token and a vanilla
        decode token. The -100 case first records the baseline greedy token,
        then verifies biasing that token suppresses it. When a draft model is
        present, speculative decoding is explicitly disabled because combining
        it with logit bias is rejected.
        """
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("HLAPI logit_bias: engine=%s", engine_dir)

        prompt = "Complete this sentence with one short word: NVIDIA makes"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
import json
import os
from experimental.server import LLM, SamplingParams

engine_dir = {engine_dir!r}

def pick_positive_bias_target_id(engine_dir):
    tokenizer_path = os.path.join(engine_dir, 'tokenizer.json')
    with open(tokenizer_path, encoding='utf-8') as f:
        tokenizer = json.load(f)

    special_ids = set()
    for token in tokenizer.get('added_tokens', []):
        token_id = token.get('id')
        if token.get('special') and isinstance(token_id, int):
            special_ids.add(token_id)
    for token_id, token in tokenizer.get('added_tokens_decoder', {{}}).items():
        if token.get('special'):
            try:
                special_ids.add(int(token_id))
            except ValueError:
                pass

    vocab = tokenizer.get('model', {{}}).get('vocab', {{}})
    preferred_pieces = (
        ' NVIDIA', 'NVIDIA', ' hello', 'Hello', ' the', 'The',
        ' answer', 'Answer', ' cat', 'cat', '!', '.',
        'ĠNVIDIA', 'Ġhello', 'Ġthe', 'Ġanswer', 'Ġcat',
        '▁NVIDIA', '▁hello', '▁the', '▁answer', '▁cat',
    )
    for piece in preferred_pieces:
        token_id = vocab.get(piece)
        if isinstance(token_id, int) and token_id not in special_ids:
            return token_id

    vocab_items = (
        (piece, token_id) for piece, token_id in vocab.items()
        if isinstance(token_id, int)
    )
    for piece, token_id in sorted(vocab_items, key=lambda item: item[1]):
        if (
            token_id not in special_ids
            and piece
            and not piece.startswith(('<', '[', '{{'))
        ):
            return token_id

    raise RuntimeError('Could not find a non-special token ID for logit_bias')

def generate_ids(llm, *, max_tokens=1, logit_bias=None):
    outputs = llm.generate(
        [{prompt!r}],
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_tokens=max_tokens,
            disable_spec_decode=llm.has_draft_model,
            logit_bias=logit_bias or {{}},
        ),
    )
    ids = outputs[0].token_ids
    assert ids, 'Expected at least one generated token id'
    return ids

llm = LLM(engine_dir=engine_dir)

target_token_id = pick_positive_bias_target_id(engine_dir)
forced_token_count = 2
positive_token_ids = generate_ids(
    llm,
    max_tokens=forced_token_count,
    logit_bias={{target_token_id: 100.0}},
)
print(f'HLAPI_POSITIVE_TARGET_ID={{target_token_id}}')
print(f'HLAPI_POSITIVE_TOKEN_IDS={{positive_token_ids}}')
assert len(positive_token_ids) == forced_token_count, (
    f'Expected {{forced_token_count}} generated tokens, got {{positive_token_ids}}'
)
assert all(token_id == target_token_id for token_id in positive_token_ids), (
    f'Expected +100 logit_bias to force {{target_token_id}} for prefill and decode, '
    f'got {{positive_token_ids}}'
)

baseline_token_id = generate_ids(llm)[0]
negative_token_id = generate_ids(
    llm, logit_bias={{baseline_token_id: -100.0}}
)[0]
print(f'HLAPI_NEGATIVE_BANNED_ID={{baseline_token_id}}')
print(f'HLAPI_NEGATIVE_TOKEN_ID={{negative_token_id}}')
assert negative_token_id != baseline_token_id, (
    f'Expected -100 logit_bias to suppress {{baseline_token_id}}, got {{negative_token_id}}'
)
print('HLAPI_GENERATE_WITH_LOGIT_BIAS_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HLAPI logit_bias for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI logit_bias failed: {result.get('error', 'Unknown')}")
        if 'HLAPI_GENERATE_WITH_LOGIT_BIAS_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"HLAPI logit_bias output:\n{result.get('output', '')}")

    def test_hlapi_streaming(self, test_param: str,
                             executable_files: Dict[str, str],
                             remote_config: Optional[RemoteConfig],
                             test_logger: logging.Logger,
                             env_config: EnvironmentConfig) -> None:
        """Test LLM.generate_stream() with a pre-built engine directory."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("HLAPI streaming: engine=%s", engine_dir)

        prompt = "Count from 1 to 10."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.7, max_tokens={max_tokens}),
))
total_text = ''.join(c.text for c in chunks)
print(f'HLAPI_STREAM_CHUNKS={{len(chunks)}}')
print(f'HLAPI_STREAM_TEXT_LEN={{len(total_text)}}')
print(f'HLAPI_STREAM_TEXT={{total_text[:200]}}')
assert len(chunks) > 1, f'Expected multiple chunks, got {{len(chunks)}}'
assert len(total_text) > 0, 'Empty streamed text'
assert any(c.finished for c in chunks), 'No terminal chunk received'
print('HLAPI_STREAMING_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI streaming for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI streaming failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'HLAPI_STREAMING_PASSED' not in output:
            pytest.fail(
                f"HLAPI streaming did not produce expected output. Output:\n{output}"
            )

    def test_hlapi_streaming_with_logprobs(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate_stream() delivers per-token logprobs matching token count."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        test_logger.info("HLAPI streaming with logprobs: engine=%s",
                         engine_dir)

        prompt = "Count from 1 to 5."
        max_tokens = 32
        num_logprobs = 3
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens={max_tokens},
                   num_logprobs={num_logprobs}),
))
total_tokens = sum(len(c.token_ids) for c in chunks)
total_lp_steps = sum(len(c.logprobs) for c in chunks)
print(f'HLAPI_STREAM_TOTAL_TOKENS={{total_tokens}}')
print(f'HLAPI_STREAM_TOTAL_LP_STEPS={{total_lp_steps}}')
assert total_tokens > 0, 'No tokens received'
assert total_lp_steps == total_tokens, f'logprob steps {{total_lp_steps}} != token count {{total_tokens}}'
for chunk in chunks:
    for step in chunk.logprobs:
        assert len(step) == {num_logprobs}, f'Expected {num_logprobs} entries per step, got {{len(step)}}'
        for e in step:
            assert isinstance(e.token_id, int) and e.token_id >= 0, f'Bad token_id {{e.token_id}}'
            assert e.logprob <= 0.0, f'logprob should be <= 0, got {{e.logprob}}'
            assert isinstance(e.token, str), f'token should be str, got {{type(e.token)}}'
            assert isinstance(e.bytes, list), f'bytes should be list, got {{type(e.bytes)}}'

# Also exercise the SSE formatting layer on the same engine: streaming
# choices[0].logprobs must be the OpenAI object shape {{"content": [...]}} with
# nested top_logprobs, mirroring the non-streaming _format_logprobs.
import json as _json
from experimental.server import api_server
from experimental.server.tool_calling import validate_tool_request
_msgs = [{{"role": "user", "content": {prompt!r}}}]
_tc = validate_tool_request(_msgs, None, None)
lp_objs = []
for sse in api_server._generate_stream_sse(
        llm, _msgs,
        SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens={max_tokens},
                       num_logprobs={num_logprobs}),
        'chatcmpl-test', False, tool_config=_tc):
    if not sse.startswith('data: ') or sse.strip() == 'data: [DONE]':
        continue
    _choice = _json.loads(sse[len('data: '):].strip())['choices'][0]
    assert 'logprobs' not in _choice.get('delta', {{}}), 'logprobs must not live inside delta'
    if _choice.get('logprobs') is not None:
        lp_objs.append(_choice['logprobs'])
assert lp_objs, 'no logprobs found in any SSE chunk'
for lp in lp_objs:
    assert isinstance(lp, dict) and 'content' in lp, f'SSE logprobs not OpenAI-shaped: {{lp}}'
    for c in lp['content']:
        assert set(c) >= {{'token', 'token_id', 'bytes', 'logprob', 'top_logprobs'}}, f'missing keys: {{sorted(c)}}'
        assert len(c['top_logprobs']) == {num_logprobs}, f'expected {num_logprobs} top_logprobs, got {{len(c["top_logprobs"])}}'
print('HLAPI_STREAMING_LOGPROBS_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI streaming with logprobs for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI streaming with logprobs failed: {result.get('error', 'Unknown')}"
            )

        if 'HLAPI_STREAMING_LOGPROBS_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI streaming logprobs output:\n{result.get('output', '')}"
            )

    def test_hlapi_generate_with_stop(self, test_param: str,
                                      executable_files: Dict[str, str],
                                      remote_config: Optional[RemoteConfig],
                                      test_logger: logging.Logger,
                                      env_config: EnvironmentConfig) -> None:
        """HLAPI non-streaming: SamplingParams(stop=[...]) trims output, finish_reason == 'stop'."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
outputs = llm.generate(
    [{prompt!r}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=128, stop=[{stop!r}]),
)
text = outputs[0].text
reason = outputs[0].finish_reason
print(f'HLAPI_TEXT={{text!r}}')
print(f'HLAPI_REASON={{reason}}')
assert {stop!r} not in text, f'Stop string leaked into output: {{text!r}}'
assert reason == 'stop', f'Expected reason=stop, got {{reason}}'
print('HLAPI_GENERATE_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HLAPI generate with stop for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI generate with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_GENERATE_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI generate with stop output:\n{result.get('output', '')}"
            )

    def test_hlapi_streaming_with_stop(self, test_param: str,
                                       executable_files: Dict[str, str],
                                       remote_config: Optional[RemoteConfig],
                                       test_logger: logging.Logger,
                                       env_config: EnvironmentConfig) -> None:
        """HLAPI streaming: stop string trimmed from chunks, last chunk reason == 'stop'."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        prompt = "List three colors, separated by commas. End your list with '###'."
        stop = "###"
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=128, stop=[{stop!r}]),
))
text = ''.join(c.text for c in chunks)
terminal_reason = next((c.finish_reason for c in chunks if c.finished), None)
print(f'HLAPI_STREAM_TEXT={{text!r}}')
print(f'HLAPI_TERMINAL_REASON={{terminal_reason}}')
assert {stop!r} not in text, f'Stop string leaked: {{text!r}}'
assert terminal_reason == 'stop', f'Expected reason=stop, got {{terminal_reason}}'
print('HLAPI_STREAMING_WITH_STOP_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"HLAPI streaming with stop for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI streaming with stop failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_STREAMING_WITH_STOP_PASSED' not in result.get('output', ''):
            pytest.fail(
                f"HLAPI streaming with stop output:\n{result.get('output', '')}"
            )

    def test_hlapi_generate_length_finish_reason(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """Verify HLAPI non-streaming reports finish_reason='length' on max_tokens hit."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        prompt = "Write a long detailed essay about transformer neural networks."
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
outputs = llm.generate(
    [{prompt!r}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=8),
)
print(f'HLAPI_REASON={{outputs[0].finish_reason}}')
assert outputs[0].finish_reason == 'length', f'Expected length, got {{outputs[0].finish_reason}}'
print('HLAPI_GENERATE_LENGTH_REASON_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(f"HLAPI length-reason for {config.model_name}",
                           test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI length-reason failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_GENERATE_LENGTH_REASON_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"HLAPI length-reason output:\n{result.get('output', '')}")

    def test_hlapi_streaming_length_finish_reason(
            self, test_param: str, executable_files: Dict[str, str],
            remote_config: Optional[RemoteConfig], test_logger: logging.Logger,
            env_config: EnvironmentConfig) -> None:
        """HLAPI streaming: terminal chunk reports finish_reason='length' on max_tokens hit."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)
        engine_dir = config.get_llm_engine_dir()
        prompt = "Write a long detailed essay about transformer neural networks."
        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.0, top_p=1.0, top_k=1, max_tokens=8),
))
terminal_reason = next((c.finish_reason for c in chunks if c.finished), None)
print(f'HLAPI_TERMINAL_REASON={{terminal_reason}}')
assert terminal_reason == 'length', f'Expected length, got {{terminal_reason}}'
print('HLAPI_STREAMING_LENGTH_REASON_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']
        env_vars = None
        if env_config.trt_package_dir:
            env_vars = {
                "LD_LIBRARY_PATH":
                f"$LD_LIBRARY_PATH:{env_config.trt_package_dir}/lib"
            }

        with timer_context(
                f"HLAPI streaming length-reason for {config.model_name}",
                test_logger):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)
        if not result['success']:
            pytest.fail(
                f"HLAPI streaming length-reason failed: {result.get('error', 'Unknown')}"
            )
        if 'HLAPI_STREAMING_LENGTH_REASON_PASSED' not in result.get(
                'output', ''):
            pytest.fail(
                f"HLAPI streaming length-reason output:\n{result.get('output', '')}"
            )
