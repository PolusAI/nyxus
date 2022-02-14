message(STATUS "Build tests")
set(CMAKE_VERBOSE_MAKEFILE ON)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(CMAKE_EXE_LINKER_FLAGS "-v")

set(TEST_SRC
        testAll.cc
		testDummy.h
	)

add_executable(runAllTests ${TEST_SRC})


target_include_directories(runAllTests PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_directories(runAllTests PUBLIC ${GTEST_LIBRARY_PATH})
target_link_libraries(runAllTests gtest ${runAllTests_LIBRARIES})