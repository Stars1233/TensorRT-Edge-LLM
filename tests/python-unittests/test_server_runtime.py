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

import threading
import time
from types import SimpleNamespace

from experimental.server.engine import LLM


class _ConcurrentRuntime:

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def handle_request(self, request):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        time.sleep(0.02)

        with self.lock:
            self.active -= 1
        return request


def test_runtime_requests_are_serialized():
    runtime = _ConcurrentRuntime()
    llm = SimpleNamespace(_runtime=runtime)
    results = []

    threads = [
        threading.Thread(target=lambda request=request: results.append(
            LLM._handle_request(llm, request))) for request in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == list(range(8))
    assert runtime.max_active == 1
