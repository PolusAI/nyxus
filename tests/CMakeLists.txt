cmake_minimum_required(VERSION 3.20)
set (CMAKE_CXX_STANDARD 17)
message(STATUS "Build tests")
#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR}/src/nyx)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

foreach(SOURCE_FILE ${SOURCE})
    list(APPEND TEST_SOURCE_FILES "../${SOURCE_FILE}")
endforeach()

set(TEST_SRC
	test_all.cc
	test_pixel_intensity_features.h
	test_data.h
	test_gabor.cc
	test_gabor.h
	test_initialization.h
	${TEST_SOURCE_FILES}
)
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
target_link_libraries(runAllTests PRIVATE -static-libgcc -static-libstdc++)
target_link_libraries(runAllTests PRIVATE stdc++fs)
