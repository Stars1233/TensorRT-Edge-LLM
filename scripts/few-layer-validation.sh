#!/usr/bin/env bash
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
#
# End-to-end few-layer numeric validation.
#
# Runs the whole smoke test for one model:
#   1. PyTorch golden            -> golden.safetensors
#   2. EdgeLLM export (N layers) -> export/llm/model.onnx
#   3. EdgeLLM build             -> engine/llm.engine
#   4. EdgeLLM inference (+dump)  -> edgellm_dump/edgellm_dump.safetensors
#   5. compare golden vs edgellm -> PASS/FAIL (this script's exit code)
#
# Everything is parameterized so it is portable (no per-user absolute paths) and
# CI-friendly. Paths are resolved relative to the repo, intermediates go in a temp
# workdir, and max-generate-length / batch-size are read from the input JSON so the
# golden and the runtime stay aligned.
#
# Usage:
#   scripts/few-layer-validation.sh --model PATH [options]
#
# Options (--model is required; the rest are optional):
#   --model PATH        Checkpoint dir or HF repo id to validate (REQUIRED).
#   --num-layers N      Number of leading decoder layers to validate (default: 4).
#   --input-file JSON   Requests JSON, llm_inference format
#                       (default: <repo>/tests/test_cases/ragged_batch.json).
#   --build-dir DIR     CMake build dir with the llm_build/llm_inference binaries and
#                       the plugin .so (default: <repo>/build).
#   --python BIN        Python used for the golden + ONNX export (default: python3,
#                       i.e. the active venv; needs torch + transformers + the TRT wheel).
#   --workdir DIR       Where to put intermediates (default: a fresh mktemp dir).
#   --keep              Do not delete the workdir on exit (for debugging).
#   --cos X             Min cosine similarity threshold (default: 0.99).
#   --atol X            allclose atol (default: 2e-2).
#   --rtol X            allclose rtol (default: 2e-2).
#   --max-input-len N   Engine maxInputLen (default: 128). Engine build parameters can
#                       affect accuracy; raise for longer prompts (grows the dump).
#   --max-kv-cache-capacity N
#                       Engine maxKVCacheCapacity (default: 256). Drives the KV-cache
#                       sequence dim, so the dump size scales with it (see Dump size).
#   --verbose           Print per-tensor cos / max_abs for every round (passed to the comparison).
#   -h | --help         Show this help.
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (this script lives in <repo>/scripts/).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Make the repo importable for the golden/export Python without requiring an
# editable install (`pip install -e .`). CI runs the golden as a script, so its
# sys.path[0] is tests/, not the repo root -- without this the quantized golden's
# `import tensorrt_edgellm` would fail. Harmless when the package is installed.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL=""
NUM_LAYERS=4
INPUT_FILE="${REPO}/tests/test_cases/ragged_batch.json"
BUILD_DIR="${REPO}/build"
PYBIN="python3"
WORKDIR=""
KEEP=0
COS=0.99
ATOL=2e-2
RTOL=2e-2
MAX_INPUT_LEN=128
MAX_KV_CACHE_CAPACITY=256
VERBOSE=0

# Print the leading doc block (from the title down to the first non-comment line).
usage() { awk '/^# End-to-end/{p=1} p&&!/^#/{exit} p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --num-layers)  NUM_LAYERS="$2"; shift 2 ;;
    --input-file)  INPUT_FILE="$2"; shift 2 ;;
    --build-dir)   BUILD_DIR="$2"; shift 2 ;;
    --python)      PYBIN="$2"; shift 2 ;;
    --workdir)     WORKDIR="$2"; shift 2 ;;
    --keep)        KEEP=1; shift ;;
    --cos)         COS="$2"; shift 2 ;;
    --atol)        ATOL="$2"; shift 2 ;;
    --rtol)        RTOL="$2"; shift 2 ;;
    --max-input-len)          MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-kv-cache-capacity)  MAX_KV_CACHE_CAPACITY="$2"; shift 2 ;;
    --verbose)     VERBOSE=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

LLM_BUILD="${BUILD_DIR}/examples/llm/llm_build"
LLM_INFER="${BUILD_DIR}/examples/llm/llm_inference"
PLUGIN="${BUILD_DIR}/libNvInfer_edgellm_plugin.so"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
fail() { echo "[few-layer] ERROR: $*" >&2; exit 2; }
[[ -n "${MODEL}" ]] || fail "--model is required: pass a checkpoint dir or HF repo id"
[[ -e "${MODEL}" || "${MODEL}" == */* ]] || fail "model '${MODEL}' not found (pass --model)"
[[ -f "${INPUT_FILE}" ]] || fail "input file not found: ${INPUT_FILE}"
[[ -x "${LLM_BUILD}" ]] || fail "llm_build not found at ${LLM_BUILD} (build first, or pass --build-dir)"
[[ -x "${LLM_INFER}" ]] || fail "llm_inference not found at ${LLM_INFER} (build first, or pass --build-dir)"
[[ -f "${PLUGIN}" ]] || fail "plugin not found at ${PLUGIN}"

# Read batch size and max generate length from the input JSON so both sides agree.
read -r BATCH_SIZE MAX_GEN < <("${PYBIN}" - "${INPUT_FILE}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(int(c.get("batch_size", len(c["requests"]))), int(c.get("max_generate_length", 6)))
PY
)
[[ -n "${BATCH_SIZE}" && -n "${MAX_GEN}" ]] \
  || fail "could not parse batch_size / max_generate_length from ${INPUT_FILE}"
DECODE_ROUNDS=$(( MAX_GEN - 1 ))

# Workdir (temp by default; cleaned up unless --keep).
if [[ -z "${WORKDIR}" ]]; then WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/few-layer-XXXXXX")"; fi
mkdir -p "${WORKDIR}"
cleanup() { if [[ "${KEEP}" -eq 0 ]]; then rm -r "${WORKDIR}"; fi; }
trap cleanup EXIT

GOLDEN="${WORKDIR}/golden.safetensors"
EXPORT_DIR="${WORKDIR}/export"
ENGINE_DIR="${WORKDIR}/engine"
DUMP_DIR="${WORKDIR}/edgellm_dump"

echo "[few-layer] repo        : ${REPO}"
echo "[few-layer] model       : ${MODEL}"
echo "[few-layer] num-layers  : ${NUM_LAYERS}  (dump layers 0-$(( NUM_LAYERS - 1 )))"
echo "[few-layer] input       : ${INPUT_FILE}  (batch=${BATCH_SIZE}, max_gen=${MAX_GEN})"
echo "[few-layer] python      : ${PYBIN}"
echo "[few-layer] workdir     : ${WORKDIR}"
echo

# ---------------------------------------------------------------------------
# 1. PyTorch golden
# ---------------------------------------------------------------------------
echo "[few-layer] === 1/5 PyTorch golden ==="
"${PYBIN}" "${REPO}/tests/golden_layer_dump.py" \
  --ckpt "${MODEL}" \
  --input-file "${INPUT_FILE}" \
  --num-layers "${NUM_LAYERS}" \
  --decode-rounds "${DECODE_ROUNDS}" \
  --out "${GOLDEN}"

# Teacher forcing: extract the golden's per-sequence tokens so EdgeLLM replays the exact same
# sequence (decouples the numeric comparison from greedy argmax stability). One line per sequence.
FORCED_TOKENS="${WORKDIR}/forced_tokens.txt"
"${PYBIN}" - "${GOLDEN}" "${FORCED_TOKENS}" <<'PY'
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt") as f:
    rows = f.get_tensor("generated_ids").tolist()
with open(sys.argv[2], "w") as out:
    out.write("\n".join(" ".join(str(t) for t in row) for row in rows) + "\n")
PY

# ---------------------------------------------------------------------------
# 2. EdgeLLM export (first N decoder layers)
# ---------------------------------------------------------------------------
echo "[few-layer] === 2/5 EdgeLLM export (${NUM_LAYERS} layers) ==="
"${PYBIN}" -m tensorrt_edgellm.scripts.export \
  "${MODEL}" "${EXPORT_DIR}" \
  --num-decoder-layer "${NUM_LAYERS}"

# ---------------------------------------------------------------------------
# 3. EdgeLLM build
# ---------------------------------------------------------------------------
# maxInputLen / maxKVCacheCapacity default tight (--max-input-len / --max-kv-cache-capacity):
# this validation uses short, fixed prompts (a few dozen tokens) and the dumper now copies the
# full KV cache ([..., maxSeqLen, ...]) rather than truncating to the valid length, so maxSeqLen
# == maxKVCacheCapacity directly drives the dump size. A small cap keeps the dump at a few hundred
# MB instead of multiple GB; the comparison slices each sequence to its real length in PyTorch.
# Engine build parameters can affect accuracy, so they are overridable for debugging.
echo "[few-layer] === 3/5 EdgeLLM build ==="
EDGELLM_PLUGIN_PATH="${PLUGIN}" "${LLM_BUILD}" \
  --onnxDir "${EXPORT_DIR}/llm" \
  --engineDir "${ENGINE_DIR}" \
  --maxBatchSize "${BATCH_SIZE}" \
  --maxInputLen "${MAX_INPUT_LEN}" \
  --maxKVCacheCapacity "${MAX_KV_CACHE_CAPACITY}"

# ---------------------------------------------------------------------------
# 4. EdgeLLM inference with the per-layer dump (greedy via top_k=1 in the JSON,
#    fixed length via EDGELLM_IGNORE_EOS).
# ---------------------------------------------------------------------------
echo "[few-layer] === 4/5 EdgeLLM inference + dump ==="
EDGELLM_PLUGIN_PATH="${PLUGIN}" \
EDGELLM_IGNORE_EOS=1 \
EDGELLM_FORCE_TOKENS_FILE="${FORCED_TOKENS}" \
EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS="${NUM_LAYERS}" \
EDGELLM_DUMP_LOGITS_KVCACHE_DIR="${DUMP_DIR}" \
"${LLM_INFER}" \
  --engineDir "${ENGINE_DIR}" \
  --inputFile "${INPUT_FILE}" \
  --outputFile "${WORKDIR}/out.json" \
  --maxGenerateLength "${MAX_GEN}"

# ---------------------------------------------------------------------------
# 5. Compare
# ---------------------------------------------------------------------------
echo "[few-layer] === 5/5 compare golden vs EdgeLLM ==="
# Pass --verbose through to the comparison only when requested on this script.
COMPARE_VERBOSE=()
[[ "${VERBOSE}" -eq 1 ]] && COMPARE_VERBOSE=(--verbose)
# Wrap in `if` so `set -e` does not abort before we capture the comparison's exit code.
if "${PYBIN}" "${REPO}/tests/compare_layer_dumps.py" \
  --golden "${GOLDEN}" \
  --edgellm "${DUMP_DIR}/edgellm_dump.safetensors" \
  --teacher-forced \
  "${COMPARE_VERBOSE[@]}" \
  --cos "${COS}" --atol "${ATOL}" --rtol "${RTOL}"; then
  RC=0
else
  RC=$?
fi

echo
if [[ "${RC}" -eq 0 ]]; then
  echo "[few-layer] RESULT: PASS (model=${MODEL}, layers=${NUM_LAYERS})"
else
  echo "[few-layer] RESULT: FAIL (model=${MODEL}, layers=${NUM_LAYERS})"
fi
exit "${RC}"
