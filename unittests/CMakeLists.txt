# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

add_executable(dflashBenchDDTree dflashBenchDDTree.cu)
target_include_directories(dflashBenchDDTree PRIVATE ${COMMON_INCLUDE_DIRS}
                                                     ${CMAKE_SOURCE_DIR})
target_link_libraries(
  dflashBenchDDTree PRIVATE gtest gtest_main edgellmCore commonLibraryExt
                            ${CUDA_DRIVER_LINK_LIB})
set_target_properties(dflashBenchDDTree PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                   ON)
add_cross_build_link_options(dflashBenchDDTree)

add_test(NAME dflashBenchDDTree COMMAND dflashBenchDDTree)

add_executable(benchLoggerTest benchLoggerTest.cpp)
target_include_directories(
  benchLoggerTest PRIVATE ${COMMON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}
                          ${CMAKE_SOURCE_DIR}/examples/utils)
target_link_libraries(benchLoggerTest PRIVATE gtest gtest_main exampleUtils
                                              edgellmCore commonLibraryExt)
add_cross_build_link_options(benchLoggerTest)

add_test(NAME benchLoggerTest COMMAND benchLoggerTest)
