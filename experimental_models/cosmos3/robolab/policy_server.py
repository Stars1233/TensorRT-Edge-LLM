#!/usr/bin/env python3
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
"""Edge-LLM Cosmos3 policy server (HTTP + JSON).

Exposes the RoboLab observation -> action-chunk contract over HTTP so a RoboLab
``InferenceClient`` (Isaac Lab / Isaac Sim, x86-only) can drive the Edge-LLM
Cosmos3 policy running on a remote target.

Contract (all JSON):
  POST /infer
    request : {"image": <base64-PNG> | <path>, "instruction": <str>,
               "domain": <str, optional>, "steps": <int, optional>}
    response: {"action": [[...8 floats] x32], "shape": [1,32,8],
               "dtype": "float32", "domain": "...", "num_inference_steps": N,
               "finite": bool, "meta": {...}}
  GET  /metadata -> {"action_chunk_size":32, "raw_action_dim":8, "domain":..., ...}
  GET  /healthz  -> {"status":"ok"}
  POST /reset    -> {"status":"reset successful"}  (stateless; provided for parity)

The server is a thin wrapper around the ``cosmos3_policy_inference`` binary: it
writes the image to a temp PNG, runs the CLI with ``--output <tmp>.json`` (the
default JSON I/O interface), and returns the parsed JSON action. No heavy Python
deps beyond the standard library + Pillow (image decode) are required.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("cosmos3.robolab.server")

ACTION_CHUNK_SIZE = 32
RAW_ACTION_DIM = 8


class Cosmos3PolicyBackend:
    """Runs the Edge-LLM Cosmos3 policy CLI and returns the JSON action dict."""

    def __init__(
        self,
        binary: str,
        engine_dir: str,
        domain: str = "droid_lerobot",
        steps: int = 4,
        extra_env: dict | None = None,
    ) -> None:
        self.binary = binary
        self.engine_dir = engine_dir
        self.domain = domain
        self.steps = steps
        self.extra_env = extra_env or {}
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"cosmos3_policy_inference binary not found: {binary}")
        if not os.path.isdir(engine_dir):
            raise FileNotFoundError(f"engine dir not found: {engine_dir}")

    def _decode_image_to_png(self, image_field, tmpdir: str) -> str:
        """Materialize the request image to a PNG path.

        Accepts either an existing file path, a base64-encoded PNG/JPG, or a
        base64-encoded raw HxWx3 uint8 array (data-URI style not required).
        """
        # Case 1: an existing path on the server host.
        if isinstance(image_field, str) and os.path.isfile(image_field):
            return image_field
        # Case 2: base64 string (optionally a data URI).
        if isinstance(image_field, str):
            b64 = image_field.split(
                ",", 1)[1] if image_field.startswith("data:") else image_field
            raw = base64.b64decode(b64)
            try:
                from PIL import Image

                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                # Fall back to treating it as a raw PNG blob written to disk.
                path = os.path.join(tmpdir, "obs.png")
                with open(path, "wb") as fh:
                    fh.write(raw)
                return path
            path = os.path.join(tmpdir, "obs.png")
            img.save(path)
            return path
        # Case 3: a nested list / array (HxWx3 uint8).
        try:
            import numpy as np
            from PIL import Image

            arr = np.asarray(image_field, dtype=np.uint8)
            path = os.path.join(tmpdir, "obs.png")
            Image.fromarray(arr).save(path)
            return path
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Unsupported 'image' field: {exc!r}") from exc

    def infer(self, request: dict) -> dict:
        instruction = request.get("instruction") or request.get("prompt")
        if not instruction:
            raise ValueError("request missing 'instruction'")
        domain = request.get("domain", self.domain)
        steps = int(request.get("steps", self.steps))
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._decode_image_to_png(request["image"], tmpdir)
            out_path = os.path.join(tmpdir, "action.json")
            cmd = [
                self.binary,
                "--image",
                image_path,
                "--prompt",
                instruction,
                "--domain",
                domain,
                "--steps",
                str(steps),
                "--engineDir",
                self.engine_dir,
                "--output",
                out_path,
            ]
            env = dict(os.environ)
            env.update(self.extra_env)
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            latency = time.perf_counter() - t0
            if proc.returncode != 0 or not os.path.isfile(out_path):
                raise RuntimeError(
                    f"cosmos3_policy_inference failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-2000:]}")
            with open(out_path) as fh:
                result = json.load(fh)
        result.setdefault("meta", {})["server_latency_s"] = latency
        return result


class _Handler(BaseHTTPRequestHandler):
    backend: Cosmos3PolicyBackend  # set on the server class

    def log_message(self, fmt, *args):  # keep the stdlib server quiet-ish
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/healthz", "/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path.rstrip("/") == "/metadata":
            self._send_json(
                200,
                {
                    "policy": "edge-cosmos3",
                    "action_chunk_size": ACTION_CHUNK_SIZE,
                    "raw_action_dim": RAW_ACTION_DIM,
                    "action_shape": [ACTION_CHUNK_SIZE, RAW_ACTION_DIM],
                    "domain": self.server.backend.domain,
                    "num_inference_steps": self.server.backend.steps,
                },
            )
        else:
            self._send_json(404, {
                "type": "error",
                "message": f"no route {self.path}"
            })

    def do_POST(self):  # noqa: N802
        route = self.path.rstrip("/")
        try:
            req = self._read_json()
        except Exception as exc:
            self._send_json(400, {
                "type": "error",
                "message": f"bad JSON: {exc!r}"
            })
            return
        if route == "/reset":
            # Stateless server; reset is a no-op provided for client parity.
            self._send_json(200, {"status": "reset successful"})
            return
        if route == "/infer":
            try:
                result = self.server.backend.infer(req)
            except Exception as exc:
                logger.exception("inference failed")
                self._send_json(500, {"type": "error", "message": str(exc)})
                return
            self._send_json(200, result)
            return
        self._send_json(404, {
            "type": "error",
            "message": f"no route {self.path}"
        })


def serve(backend: Cosmos3PolicyBackend, host: str,
          port: int) -> ThreadingHTTPServer:
    handler = _Handler
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.backend = backend  # attach for the handler
    logger.info(
        "Cosmos3 policy server listening on http://%s:%d (engine_dir=%s)",
        host, port, backend.engine_dir)
    return httpd


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Edge-LLM Cosmos3 RoboLab policy server (HTTP+JSON)")
    ap.add_argument("--binary",
                    required=True,
                    help="path to cosmos3_policy_inference")
    ap.add_argument(
        "--engine-dir",
        required=True,
        help="Cosmos3 engine dir (und_prefill/, vae_encoder/, gen/, ...)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--domain", default="droid_lerobot")
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    backend = Cosmos3PolicyBackend(args.binary, args.engine_dir, args.domain,
                                   args.steps)
    httpd = serve(backend, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
