#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2019 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////


SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${QMCPACK_UNIT_TEST_DIR})

SET(SRC_DIR cpu)
SET(UTEST_EXE test_${SRC_DIR})
SET(UTEST_NAME deterministic-unit_test_${SRC_DIR})

ADD_EXECUTABLE(${UTEST_EXE} test_aligned_allocator.cpp test_e2iphi.cpp)
TARGET_LINK_LIBRARIES(${UTEST_EXE} platform_device catch_main)

ADD_UNIT_TEST(${UTEST_NAME} "${QMCPACK_UNIT_TEST_DIR}/${UTEST_EXE}")
