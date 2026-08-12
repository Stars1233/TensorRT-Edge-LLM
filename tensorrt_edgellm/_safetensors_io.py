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
"""Thin wrapper around ``safetensors.torch.save_file`` that keeps exported
files world-readable.

safetensors >= 0.8 writes atomically via a private temp file (``mkstemp``,
mode ``0600``) followed by a rename, so the final file ends up owner-read-only
regardless of the process umask. Exported artifacts such as
``embedding.safetensors`` and the external weight files live in a shared cache
and are later read by other CI jobs and edge boards running as a *different*
user; ``0600`` makes those reads fail with "Permission denied". Restoring the
usual ``0644`` after the save keeps the export cache portable. (safetensors 0.7
respected the umask and produced ``0644`` directly.)
"""

import os

from safetensors.torch import save_file as _save_file


def save_file(tensors, filename, metadata=None):
    """Save tensors like ``safetensors.torch.save_file`` but leave the file
    group/other-readable (``0644``) so cross-job/cross-user consumers can read
    it."""
    _save_file(tensors, filename, metadata=metadata)
    try:
        os.chmod(filename, 0o644)
    except OSError:
        # Best effort: a non-owner or a filesystem without chmod support should
        # not fail the export just because the mode could not be relaxed.
        pass
