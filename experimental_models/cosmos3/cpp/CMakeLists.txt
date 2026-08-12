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

# Cosmos3 experimental runtime: links the core edgellm libraries (edgellmCore,
# edgellmBuilder) and only adds Cosmos3-specific code (policy runner, host UniPC
# scheduler, VAE-encoder runner, runtime orchestrator, component builder).

set(COSMOS3_CPP_SRCS
    action/cosmos3Kernels.cu action/cosmos3PolicyRunner.cpp
    action/cosmos3Scheduler.cpp builder/cosmos3Builder.cpp
    multimodal/cosmos3VaeEncoderRunner.cpp runtime/cosmos3Runtime.cpp)

add_library(cosmos3Experimental STATIC ${COSMOS3_CPP_SRCS})

# A static library only needs the core headers to compile; it does not link the
# core libraries (the executables do). Avoid a build-order dependency on the
# edgellmCore / edgellmBuilder targets so this library compiles independently of
# a core rebuild.
target_include_directories(
  cosmos3Experimental
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${COMMON_INCLUDE_DIRS}
          ${ONNX_PARSER_INCLUDE_DIR})
