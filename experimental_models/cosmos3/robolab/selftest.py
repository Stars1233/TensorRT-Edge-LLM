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
"""Local mock self-test for the Cosmos3 RoboLab policy server + client.

Runs WITHOUT Isaac Sim. It starts the HTTP+JSON policy server in a background
thread, fabricates a RoboLab-style observation (a fake ``obs["policy"]`` dict
with a camera frame + proprio), drives it through ``Cosmos3JointposClient``,
and checks the JSON round-trip: server call -> action chunk (32, 8) -> one 8-D
action per step, exercising the open-loop chunk cache.

Usage:
  python -m experimental_models.cosmos3.robolab.selftest \
      --binary <.../cosmos3_policy_inference> --engine-dir <ENGINE_DIR> \
      [--steps 4] [--rollout-steps 10]
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

import numpy as np

from .cosmos3_client import (ACTION_CHUNK_SIZE, RAW_ACTION_DIM,
                             Cosmos3JointposClient)
from .policy_server import Cosmos3PolicyBackend, serve

logger = logging.getLogger("cosmos3.robolab.selftest")


def _fake_observation(seed: int = 0) -> dict:
    """A RoboLab-style observation dict (no Isaac Sim required).

    High-variance natural-ish RGB frame + 7-DoF joint proprio + gripper scalar,
    laid out like ``obs["policy"]`` (batched, env-0 indexed by the client).
    """
    rng = np.random.default_rng(seed)
    h, w = 180, 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 2] = 255 * (1 - yy / h)
    img[..., 0] = 40 + 180 * (xx / w)
    img += (255 * np.exp(-(((xx - 260)**2 + (yy - 30)**2) /
                           (2 * 25.0**2))))[..., None]
    img += rng.normal(0, 15, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return {
        "policy": {
            # batched (num_envs, H, W, 3): the client uses env 0.
            "external_cam": img[None],
            "arm_joint_pos": rng.uniform(-1, 1, size=(7, )).astype(np.float32),
            "gripper_pos": np.asarray([0.3], dtype=np.float32),
        }
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(
        description=
        "Cosmos3 RoboLab server+client local mock self-test (no Isaac Sim)")
    ap.add_argument("--binary",
                    required=True,
                    help="path to cosmos3_policy_inference")
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--open-loop-horizon", type=int, default=8)
    args = ap.parse_args()

    instruction = "Pick up the banana and place it in the bowl."

    # 1) Start the policy server in a background thread.
    backend = Cosmos3PolicyBackend(args.binary, args.engine_dir,
                                   "droid_lerobot", args.steps)
    httpd = serve(backend, args.host, args.port)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    ok = True
    try:
        # 2) Connect the RoboLab client subclass.
        client = Cosmos3JointposClient(
            remote_host=args.host,
            remote_port=args.port,
            open_loop_horizon=args.open_loop_horizon,
        )
        logger.info("server metadata: %s", client.metadata())

        # 3) Drive a fake rollout (JSON round-trip through server + chunking).
        obs = _fake_observation()
        server_calls_seen = 0
        for step in range(args.rollout_steps):
            result = client.infer(obs, instruction)
            action = result["action"]
            assert action.shape == (
                RAW_ACTION_DIM, ), f"bad action shape {action.shape}"
            assert np.isfinite(action).all(), "non-finite action returned"
            assert action[-1] in (0.0, 1.0), "gripper not binarized"
            if result["used_server_call"]:
                server_calls_seen += 1
                logger.info(
                    "step %d: SERVER CALL latency=%.2fs action[:3]=%s gripper=%.0f",
                    step,
                    result["chunk_latency_s"],
                    np.round(action[:3], 4).tolist(),
                    action[-1],
                )
            else:
                logger.info("step %d: cached (open-loop) action[:3]=%s", step,
                            np.round(action[:3], 4).tolist())

        # The cached chunk must have shape (32, 8) and cover open_loop_horizon steps.
        assert client.pred_action_chunk.shape == (ACTION_CHUNK_SIZE,
                                                  RAW_ACTION_DIM)
        expected_calls = (args.rollout_steps + args.open_loop_horizon -
                          1) // args.open_loop_horizon
        assert server_calls_seen == expected_calls, f"expected {expected_calls} server calls, got {server_calls_seen}"

        # 4) Reset round-trip.
        status = client.reset()
        logger.info("reset -> %s", status)

        print(
            "SELFTEST PASS: JSON round-trip OK | "
            f"action_chunk={client.pred_action_chunk is None} "
            f"server_calls={server_calls_seen} rollout_steps={args.rollout_steps} "
            f"chunk_shape=({ACTION_CHUNK_SIZE},{RAW_ACTION_DIM})")
    except Exception:
        ok = False
        logger.exception("SELFTEST FAILED")
    finally:
        httpd.shutdown()
        httpd.server_close()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
