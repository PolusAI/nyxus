cmake_minimum_required(VERSION 3.12)

set (CMAKE_CXX_STANDARD 17)

message(STATUS "Build tests")
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(CMAKE_EXE_LINKER_FLAGS "-v")

set(TEST_SRC
        testAll.cc
		testDummy.h
		test_gabor.cc
		test_gabor.h
		test_download_data.cc
		test_download_data.h
	)

add_executable(runAllTests ${TEST_SRC})

target_link_libraries(runAllTests stdc++fs)

target_include_directories(runAllTests PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_directories(runAllTests PUBLIC ${GTEST_LIBRARY_PATH})
target_link_libraries(runAllTests gtest ${runAllTests_LIBRARIES})
target_link_libraries(runAllTests gtest_main ${runAllTests_LIBRARIES})