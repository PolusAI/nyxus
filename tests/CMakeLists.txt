cmake_minimum_required(VERSION 3.20)
set (CMAKE_CXX_STANDARD 20)
message(STATUS "Build tests")
#==== Compiler Options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/src/nyx)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

foreach(SOURCE_FILE ${SOURCE})
    list(APPEND TEST_SOURCE_FILES "../${SOURCE_FILE}")
endforeach()

set(TEST_SRC
	test_all.cc
	test_gabor_regression.cc
	${TEST_SOURCE_FILES}
)
# NOTE: individual test_*.h headers are intentionally NOT listed here. They are pulled in via #include
# (relative to test_all.cc), so listing them adds nothing to the build and only creates reconfigure
# breakage every time a header is renamed during the vetting migration (e.g. the long-stale
# test_pixel_intensity_features.h entry, renamed away in Wave 4, that blocked reconfigure). Only real
# compile units belong here.
add_executable(runAllTests ${TEST_SRC})

if(USEGPU)
	include_directories("${CUDA_INCLUDE_DIRS}")
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		target_compile_options(runAllTests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>)
	endif()

	set_target_properties(runAllTests PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

endif()

target_include_directories (runAllTests PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_directories (runAllTests PUBLIC ${GTEST_LIBRARY_PATH})
target_link_libraries (runAllTests PUBLIC gtest ${Nyxus_LIBRARIES})