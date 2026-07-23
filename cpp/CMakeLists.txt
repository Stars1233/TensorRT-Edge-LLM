# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier:
# LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated -fPIC"
)

# Define additional CUDA flags for kernel compilation.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -t 0 -fPIC")
set(CUDA_PTXAS_FLAGS "-warn-lmem-usage -warn-double-usage -warn-spills")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo -res-usage --use_fast_math -Xptxas='${CUDA_PTXAS_FLAGS}'"
)

set(CMAKE_CUDA_FLAGS_DEBUG "-g")

# Hide internal C++ symbols; only EDGELLM_PLUGIN_EXPORT entry points stay
# visible. Prevents duplicate-symbol collisions when our libraries are loaded
# together.
function(edgellm_set_hidden_visibility target)
  set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden
                                             VISIBILITY_INLINES_HIDDEN ON)
endfunction()

# ------------------------------------------------------------------------------
# FMHA Cubin Conditional Compilation Exclude SM architectures that are not in
# CMAKE_CUDA_ARCHITECTURES to reduce binary size. Only cubins for the target
# architectures will be compiled.
# ------------------------------------------------------------------------------
set(FMHA_ALL_SM_VERSIONS
    80
    86
    87
    89
    100
    101
    120
    121)
set(FMHA_EXCLUDE_DEFINITIONS "")

# Build a list of required FMHA SM versions based on target architectures. Some
# SM versions reuse cubins from other SM versions at runtime. Strip the
# accelerated/virtual/real suffixes (e.g. 100a, 101a, 110a, 90-real) so the
# IN_LIST checks below compare against plain numeric SM versions.
set(REQUIRED_FMHA_SM_VERSIONS "")
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
  string(REGEX REPLACE "[^0-9].*$" "" arch_num "${arch}")
  list(APPEND REQUIRED_FMHA_SM_VERSIONS ${arch_num})
endforeach()
list(REMOVE_DUPLICATES REQUIRED_FMHA_SM_VERSIONS)

# SM 110 uses SM 101 FMHA cubins at runtime
if(110 IN_LIST REQUIRED_FMHA_SM_VERSIONS AND NOT 101 IN_LIST
                                             REQUIRED_FMHA_SM_VERSIONS)
  list(APPEND REQUIRED_FMHA_SM_VERSIONS 101)
endif()

# Exclude SM versions that are not required
foreach(SM_VER ${FMHA_ALL_SM_VERSIONS})
  if(NOT SM_VER IN_LIST REQUIRED_FMHA_SM_VERSIONS)
    list(APPEND FMHA_EXCLUDE_DEFINITIONS "EXCLUDE_SM_${SM_VER}")
  endif()
endforeach()

if(FMHA_EXCLUDE_DEFINITIONS)
  message(
    STATUS
      "FMHA Kernels: Excluding SM architectures: ${FMHA_EXCLUDE_DEFINITIONS}")
else()
  message(STATUS "FMHA Kernels: Compiling cubins for all SM architectures")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/XQACubins.cmake)
edgellm_xqa_cubins_setup()

# Generate Core C++ static library for LLM-SDK
file(GLOB_RECURSE COMMON_CPP_SRCS "common/*.cpp")
file(GLOB_RECURSE KERNELS_CPP_SRCS "kernels/*.cpp")
edgellm_xqa_add_generated_sources(KERNELS_CPP_SRCS)
file(GLOB_RECURSE KERNELS_CU_SRCS "kernels/*.cu")
# Exclude MoE Marlin ops/sm80 - compiled only via include from moeMarlin.cu
list(FILTER KERNELS_CU_SRCS EXCLUDE REGEX
     "marlin_moe_wna16/(ops|sm80_kernel_float16_u4_float16)\\.cu$")
file(GLOB_RECURSE SAMPLER_CPP_SRCS "sampler/*.cpp")
file(GLOB_RECURSE SAMPLER_CU_SRCS "sampler/*.cu")
file(GLOB_RECURSE MULTIMODAL_CPP_SRCS "multimodal/*.cpp")
file(GLOB_RECURSE ACTION_CPP_SRCS "action/*.cpp")
file(GLOB_RECURSE RUNTIME_CPP_SRCS "runtime/*.cpp")
file(GLOB_RECURSE PROFILER_CPP_SRCS "profiling/*.cpp")

# Enable relocatable device code for MoE Marlin kernels
file(GLOB_RECURSE MOE_MARLIN_CU_SRCS "kernels/moe/moe_marlin/*.cu")
set_source_files_properties(${MOE_MARLIN_CU_SRCS} PROPERTIES COMPILE_FLAGS
                                                             "-rdc=true")

add_library(edgellmKernels STATIC ${KERNELS_CPP_SRCS} ${KERNELS_CU_SRCS}
                                  ${COMMON_CPP_SRCS})
target_include_directories(edgellmKernels PRIVATE ${COMMON_INCLUDE_DIRS})
edgellm_xqa_configure_target(edgellmKernels)
# Apply FMHA SM exclusion definitions
target_compile_definitions(edgellmKernels PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
# Enable separable compilation for MoE Marlin kernel templates
set_target_properties(edgellmKernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
edgellm_set_hidden_visibility(edgellmKernels)

file(GLOB_RECURSE TOKENIZER_CPP_SRCS "tokenizer/*.cpp")

add_library(
  edgellmCore STATIC
  ${COMMON_CPP_SRCS}
  ${COMMON_CU_SRCS}
  ${KERNELS_CPP_SRCS}
  ${KERNELS_CU_SRCS}
  ${SAMPLER_CPP_SRCS}
  ${SAMPLER_CU_SRCS}
  ${MULTIMODAL_CPP_SRCS}
  ${ACTION_CPP_SRCS}
  ${RUNTIME_CPP_SRCS}
  ${PROFILER_CPP_SRCS}
  ${TOKENIZER_CPP_SRCS})
target_include_directories(edgellmCore PRIVATE ${COMMON_INCLUDE_DIRS})
edgellm_xqa_configure_target(edgellmCore)
target_link_libraries(edgellmCore PRIVATE dl)
# Apply FMHA SM exclusion definitions
target_compile_definitions(edgellmCore PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
# Enable separable compilation for MoE Marlin kernel templates
set_target_properties(edgellmCore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
edgellm_set_hidden_visibility(edgellmCore)

# Generate builder C++ static library
file(GLOB_RECURSE BUILDER_CPP_SRCS "builder/*.cpp")
add_library(edgellmBuilder STATIC ${BUILDER_CPP_SRCS} ${COMMON_CPP_SRCS})
target_include_directories(edgellmBuilder PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                                  ${COMMON_INCLUDE_DIRS})
target_link_libraries(edgellmBuilder PRIVATE ${NV_ONNX_PARSER_LIB} dl)
edgellm_set_hidden_visibility(edgellmBuilder)

# Generate the TensorRT plugin shared libraries
file(GLOB_RECURSE PLUGIN_CPP_SRCS "plugins/*.cpp")
add_library(NvInfer_edgellm_plugin SHARED ${PLUGIN_CPP_SRCS})
set_target_properties(NvInfer_edgellm_plugin PROPERTIES VERSION 1.0 SOVERSION 1)
set_target_properties(NvInfer_edgellm_plugin PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                        ${CMAKE_BINARY_DIR})
target_include_directories(NvInfer_edgellm_plugin
                           PRIVATE ${COMMON_INCLUDE_DIRS})
# Apply FMHA SM exclusion definitions
target_compile_definitions(NvInfer_edgellm_plugin
                           PRIVATE ${FMHA_EXCLUDE_DEFINITIONS})
target_link_libraries(
  NvInfer_edgellm_plugin PRIVATE commonLibraryExt edgellmKernels
                                 ${CUDA_DRIVER_LIB} ${CUDART_LIB})
add_cross_build_link_options(NvInfer_edgellm_plugin)
edgellm_set_hidden_visibility(NvInfer_edgellm_plugin)
# Also drop symbols pulled in from static archives (edgellmKernels, CuTe DSL).
target_link_options(NvInfer_edgellm_plugin PRIVATE "-Wl,--exclude-libs,ALL")

# cmake-format: off
# CuTe DSL unified kernels: link local prebuilt static library. Artifacts are
# generated by: python kernelSrcs/build_cutedsl.py --gpu_arch <sm_NN>. Set
# ENABLE_CUTE_DSL to enable one of:
#   fmha             -- FMHA Blackwell attention
#   gdn              -- gated delta net (Mamba2-style hybrid SSM)
#   f16_moe          -- target-specific FP16 grouped MoE for Ampere,
#                       Blackwell DC, and Blackwell GeForce
#   nvfp4_moe        -- SM110 (Thor) split FC1/FC2 NVFP4 MoE; consumed by the
#                       SM110-only ``Nvfp4MoePlugin``
#   nvfp4_fused_moe  -- SM12x (consumer Blackwell / Spark GB10) fused NVFP4
#                       MoE; consumed by ``NvFP4MoEPluginGeforce``
# or a semicolon-separated list (e.g. fmha;gdn), or ALL, or OFF. See the
# top-level CMakeLists.txt for the canonical option list.
# cmake-format: on
if(NOT ENABLE_CUTE_DSL STREQUAL "OFF")
  include(${CMAKE_SOURCE_DIR}/cmake/CuteDsl.cmake)
  cute_dsl_setup(LINK_TARGETS edgellmKernels edgellmCore NvInfer_edgellm_plugin)
endif()
