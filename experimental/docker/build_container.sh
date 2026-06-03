#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

: "${CUDA_CTK_VERSION:=13.0}"
: "${CUTE_DSL_ARTIFACT_TAG:=sm_110}"
: "${CUTE_DSL_ARCH:=aarch64}"
: "${CUTE_DSL_GPU_ARCH:=${CUTE_DSL_ARTIFACT_TAG}}"
: "${CUTE_DSL_KERNELS:=ALL}"
: "${EXPERIMENTAL_DOCKER_IMAGE:=tensorrt-edge-llm:experimental}"

cuda_major="${CUDA_CTK_VERSION%%.*}"
prebuilt_dir="${repo_root}/kernelSrcs/cuteDSLPrebuilt"
tarball_name="cutedsl_${CUTE_DSL_ARCH}_${CUTE_DSL_ARTIFACT_TAG}_cuda${cuda_major}.tar.gz"
tarball_path="${prebuilt_dir}/${tarball_name}"

case "${cuda_major}" in
    13)
        : "${CUTE_DSL_PACKAGE:=nvidia-cutlass-dsl[cu13]==4.5.0}"
        : "${CUTE_DSL_CUPY_PACKAGE:=cupy-cuda13x==13.6.0}"
        ;;
    12)
        : "${CUTE_DSL_PACKAGE:=nvidia-cutlass-dsl[cu12]==4.5.0}"
        : "${CUTE_DSL_CUPY_PACKAGE:=cupy-cuda12x==12.3.0}"
        ;;
    *)
        echo "Unsupported CUDA_CTK_VERSION=${CUDA_CTK_VERSION}; set CUTE_DSL_PACKAGE and CUTE_DSL_CUPY_PACKAGE explicitly." >&2
        exit 1
        ;;
esac

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "$1 is required to prepare the CuteDSL artifact." >&2
        exit 1
    fi
}

build_cutedsl_tarball() {
    require_tool python3
    require_tool tar
    require_tool sha256sum

    local cache_dir="${HOME:-/tmp}/.cache/tensorrt-edge-llm"
    local venv_dir="${CUTE_DSL_VENV:-${cache_dir}/cutedsl-venv-cuda${cuda_major}}"
    local output_dir
    output_dir="$(mktemp -d)"

    cleanup() {
        rm -rf "${output_dir}"
    }
    trap cleanup RETURN

    echo "No CuteDSL tarball found at ${tarball_path}"
    echo "Building CuteDSL artifact before docker build."

    mkdir -p "${cache_dir}" "${prebuilt_dir}"
    python3 -m venv "${venv_dir}"
    "${venv_dir}/bin/pip" install -q --upgrade pip wheel
    "${venv_dir}/bin/pip" install -q "${CUTE_DSL_PACKAGE}" "${CUTE_DSL_CUPY_PACKAGE}"

    "${venv_dir}/bin/python" kernelSrcs/build_cutedsl.py \
        --kernels "${CUTE_DSL_KERNELS}" \
        --gpu_arch "${CUTE_DSL_GPU_ARCH}" \
        --arch "${CUTE_DSL_ARCH}" \
        --output_dir "${output_dir}" \
        --clean

    local artifact_dir="${output_dir}/${CUTE_DSL_ARCH}/${CUTE_DSL_ARTIFACT_TAG}"
    test -f "${artifact_dir}/metadata.json"
    test -f "${artifact_dir}/libcutedsl_${CUTE_DSL_ARCH}.a"
    test -f "${artifact_dir}/include/cutedsl_all.h"

    tar -C "${output_dir}/${CUTE_DSL_ARCH}" -czf "${tarball_path}" "${CUTE_DSL_ARTIFACT_TAG}"
    (cd "${prebuilt_dir}" && sha256sum "${tarball_name}" > "${tarball_name}.sha256")
    echo "Wrote ${tarball_path}"
}

cd "${repo_root}"

if [[ ! -f "${tarball_path}" ]]; then
    build_cutedsl_tarball
elif [[ -f "${tarball_path}.sha256" ]]; then
    (cd "${prebuilt_dir}" && sha256sum -c "${tarball_name}.sha256")
fi

if [[ "$#" -eq 0 ]]; then
    set -- \
        --network=host \
        --shm-size=8g \
        -f experimental/docker/Dockerfile \
        -t "${EXPERIMENTAL_DOCKER_IMAGE}" \
        .
fi

require_tool docker
docker build "$@"
