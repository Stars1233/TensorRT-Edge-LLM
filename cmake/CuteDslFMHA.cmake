# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# CuTe DSL FMHA: link prebuilt static libraries from cuteDSLArtifact/{arch}/
#
# Prebuilt artifacts are generated offline by: python
# kernelSrcs/fmha_cutedsl_blackwell/build_static_lib.py and checked into the
# repo. No Python, CUTLASS DSL, or Blackwell GPU is needed at CMake build time.
#
# Usage: include(cmake/CuteDslFMHA.cmake) then call: cute_dsl_fmha_setup(
# TARGETS      target1 target2 ...   # get define + include path only
# LINK_TARGETS target3 target4 ...   # get define + include path + link ) This
# function is safe to include and call multiple times (e.g. once from
# cpp/CMakeLists.txt for the plugin and once from the root for unit tests).

# ---------------------------------------------------------------------------
# cute_dsl_fmha_setup()
#
# TARGETS      — get CUTE_DSL_FMHA_ENABLED define + include path for headers
# LINK_TARGETS — same as TARGETS, plus link libcutedsl_fmha_*.a
# (libcuda_dialect_runtime_static.a is merged into the same archive)
# ---------------------------------------------------------------------------
function(cute_dsl_fmha_setup)
  cmake_parse_arguments(ARG "" "" "TARGETS;LINK_TARGETS" ${ARGN})

  # Detect host CPU architecture
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_arch "aarch64")
  else()
    set(_arch "x86_64")
  endif()

  # Use CMAKE_SOURCE_DIR so this function is callable from any subdirectory.
  set(_artifact_dir
      "${CMAKE_SOURCE_DIR}/cpp/kernels/contextAttentionKernels/cuteDSLArtifact/${_arch}"
  )
  set(_static_lib "${_artifact_dir}/libcutedsl_fmha_${_arch}.a")
  set(_inc_dir "${_artifact_dir}/include")

  # Validate prebuilt artifacts exist
  if(NOT EXISTS "${_static_lib}")
    message(
      FATAL_ERROR
        "Prebuilt CuTe DSL FMHA library not found:\n"
        "  ${_static_lib}\n"
        "Generate it with:\n"
        "  python kernelSrcs/fmha_cutedsl_blackwell/build_static_lib.py --arch ${_arch}\n"
        "then commit the resulting ${_arch}/ directory.")
  endif()

  if(NOT EXISTS "${_inc_dir}/cutedsl_fmha_all.h")
    message(
      FATAL_ERROR "Prebuilt CuTe DSL FMHA headers not found in ${_inc_dir}/\n"
                  "Re-run build_static_lib.py to regenerate artifacts.")
  endif()

  # TARGETS: compile definition + include path only
  foreach(_tgt ${ARG_TARGETS})
    target_compile_definitions(${_tgt} PRIVATE CUTE_DSL_FMHA_ENABLED)
    target_include_directories(${_tgt} PRIVATE "${_inc_dir}")
  endforeach()

  # LINK_TARGETS: compile definition + include path + link the single archive.
  # The objects from libcuda_dialect_runtime_static.a (_cuda_* wrappers) are
  # merged into the same .a by build_static_lib.py — no separate runtime lib
  # needed.
  foreach(_tgt ${ARG_LINK_TARGETS})
    target_compile_definitions(${_tgt} PRIVATE CUTE_DSL_FMHA_ENABLED)
    target_include_directories(${_tgt} PRIVATE "${_inc_dir}")
    target_link_libraries(${_tgt} PRIVATE "${_static_lib}")
  endforeach()

  if(ARG_TARGETS OR ARG_LINK_TARGETS)
    message(
      STATUS "CuTe DSL FMHA: using prebuilt ${_arch} artifacts (${_static_lib})"
    )
  endif()
endfunction()
