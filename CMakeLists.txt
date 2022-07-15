cmake_minimum_required(VERSION 3.20)
project(AxleNyxus)

find_package(CUDA)
if(CUDA_FOUND)
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR} 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nyx/gpu/find_cuda_gpu.c
        CMAKE_FLAGS 
            -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
            -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
        RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
	message("---------${CUDA_TOOLKIT_INCLUDE}--------")
    message("${RUN_OUTPUT_VAR}") # Display number of GPUs found
    # COMPILE_RESULT_VAR is TRUE when compile succeeds
    # RUN_RESULT_VAR is zero when a GPU is found
    if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
		option(USEGPU "Use GPU" ON) 
    else()
		option(USEGPU "Use GPU" OFF) 
    endif()
else()
	option(USEGPU "Use GPU" OFF) 
endif(CUDA_FOUND)

if(USEGPU)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CXX_COMPILER})
    enable_language("CUDA")
    add_definitions(-DUSE_GPU)

    include(FindCUDA/select_compute_arch)
	CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
	string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
	string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
	SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
	set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
	message("Found GPU with compute capability ${CUDA_ARCH_LIST}.")
endif()

#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- ")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

#== GTest 
option(RUN_GTEST "Downloads google unit test API and runs google test scripts to test Nyxus" OFF)
set(RUN_GTEST ON)

#==== Pybind11
add_subdirectory(lib/pybind11)

include_directories(
	lib/pybind11/include/
)

find_package(OpenMP COMPONENTS C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake-modules")

#==== Source files
set(SOURCE
	src/nyx/features/basic_morphology.cpp
	src/nyx/features/caliper_feret.cpp
	src/nyx/features/caliper_martin.cpp
	src/nyx/features/caliper_nassenstein.cpp
	src/nyx/features/chords.cpp
	src/nyx/features/chords_nontriv.cpp
	src/nyx/features/circle.cpp
	src/nyx/features/contour.cpp
	src/nyx/features/convex_hull_nontriv.cpp
	src/nyx/features/ellipse_fitting.cpp
	src/nyx/features/erosion_pixels.cpp
	src/nyx/features/euler_number.cpp
	src/nyx/features/extrema.cpp
	src/nyx/features/fractal_dim.cpp
	src/nyx/features/gabor.cpp
	src/nyx/features/gabor_nontriv.cpp
	src/nyx/features/geo_len_thickness.cpp
	src/nyx/features/glcm.cpp
	src/nyx/features/glcm_nontriv.cpp
	src/nyx/features/gldm.cpp
	src/nyx/features/glrlm.cpp
	src/nyx/features/glszm.cpp
	src/nyx/features/hexagonality_polygonality.cpp
	src/nyx/features/image_matrix.cpp
	src/nyx/features/image_matrix_nontriv.cpp
	src/nyx/features/image_moments.cpp
	src/nyx/features/image_moments_nontriv.cpp
	src/nyx/features/intensity.cpp
	src/nyx/features/neighbors.cpp
	src/nyx/features/ngtdm.cpp
	src/nyx/features/pixel_intensity.cpp
	src/nyx/features/radial_distribution.cpp
	src/nyx/features/roi_label.cpp
	src/nyx/features/roi_radius.cpp
	src/nyx/features/rotation.cpp
	src/nyx/features/specfunc.cpp
	src/nyx/features/zernike.cpp
	src/nyx/features/zernike_nontriv.cpp
	src/nyx/helpers/timing.cpp
	src/nyx/common_stats.cpp
	src/nyx/dirs_and_files.cpp
	src/nyx/environment.cpp
	src/nyx/environment_basic.cpp
	src/nyx/feature_method.cpp
	src/nyx/feature_mgr.cpp
	src/nyx/feature_mgr_init.cpp
	src/nyx/features_calc_workflow.cpp
	src/nyx/featureset.cpp
	src/nyx/globals.cpp
	src/nyx/image_loader.cpp
	src/nyx/output_2_buffer.cpp
	src/nyx/output_2_csv.cpp
	src/nyx/parallel.cpp
	src/nyx/phase1.cpp
	src/nyx/phase2.cpp
	src/nyx/phase3.cpp
	src/nyx/pixel_feed.cpp
	src/nyx/reduce_by_feature.cpp
	src/nyx/reduce_trivial_rois.cpp
	src/nyx/roi_cache.cpp
	src/nyx/roi_cache_basic.cpp
	src/nyx/scan_fastloader_way.cpp
)


# CLI
if(BUILD_CLI)
	add_executable(nyxus ${SOURCE} src/nyx/main_nyxus.cpp)
	add_executable(nyxushie src/nyx/dirs_and_files.cpp src/nyx/environment_basic.cpp src/nyx/image_loader1x.cpp src/nyx/main_nyxushie.cpp src/nyx/nested_roi.cpp src/nyx/python/nested_roi_py.cpp src/nyx/roi_cache_basic.cpp)
endif()


# Python bindings.
pybind11_add_module(backend 
	${SOURCE}
	src/nyx/python/new_bindings_py.cpp
	src/nyx/python/nested_roi_py.cpp
	src/nyx/nested_roi.cpp
	src/nyx/image_loader1x.cpp
)
#== Install OMEZarr PreReqs
option(WITH_OMEZARR "Build with OMEZarr Support" ON)
if(WITH_OMEZARR)
	if(UNIX)
		execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/build_scripts/install_prereq_linux.sh)
	endif()
	if(WIN32)
		execute_process(COMMAND cmd /C ${CMAKE_SOURCE_DIR}/build_scripts/install_prereq_win.bat)
	endif()
	set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/local_install)
	find_package(Boost REQUIRED)
	if(Boost_FOUND)
		include_directories(${Boost_INCLUDE_DIR})
	endif()
	find_package(BLOSC REQUIRED)
	include_directories(${BLOSC_INCLUDE_DIR})
	target_compile_definitions(backend PRIVATE -DWITH_BLOSC)
	target_link_libraries(backend PRIVATE ${BLOSC_LIBRARIES})
	if(BUILD_CLI)
		target_compile_definitions(nyxus PRIVATE -DWITH_BLOSC)
		target_compile_definitions(nyxushie PRIVATE -DWITH_BLOSC) 	
		target_link_libraries(nyxus PRIVATE ${BLOSC_LIBRARIES})	
		target_link_libraries(nyxushie PRIVATE ${BLOSC_LIBRARIES})	
	endif()
endif()

find_package(FastLoader)
if(FastLoader_FOUND)
	list(APPEND Nyxus_LIBRARIES ${FastLoader_LIBRARIES})
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FastLoader_CXX_FLAGS}")
	include_directories(${FastLoader_INCLUDE_DIR})
else()
	message(FATAL_ERROR "Unable to find FastLoader which is required for Nyxus.")
endif()

if(USEGPU)
	set(GPU_SOURCE_FILES
		src/nyx/gpu/gpu_helpers.cu
		src/nyx/gpu/image_moments.cu
        src/nyx/gpu/gabor.cu
	)
	add_library(nyxus_gpu ${GPU_SOURCE_FILES})
	target_compile_options(nyxus_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>)

    INCLUDE_DIRECTORIES(/usr/local/cuda-11.0/samples/common/inc  
		${CUDA_TOOLKIT_INCLUDE})
	target_link_libraries(backend PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
	set_target_properties(backend PROPERTIES CUDA_ARCHITECTURES "35;37;50;72")

	target_link_libraries(backend PRIVATE nyxus_gpu)
	if(BUILD_CLI)
		target_link_libraries(nyxus PRIVATE nyxus_gpu)
		target_link_libraries(nyxus PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
		set_target_properties(nyxus PROPERTIES CUDA_ARCHITECTURES "35;37;50;72")
	endif()
endif()

#== Install OMEZarr PreReqs
option(WITH_OMEZARR "Build with OMEZarr Support" ON)
if(WITH_OMEZARR)
	if(UNIX)
		execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/build_scripts/install_prereq_linux.sh)
	endif()
	if(WIN32)
		execute_process(COMMAND cmd /C ${CMAKE_SOURCE_DIR}/build_scripts/install_prereq_win.bat)
	endif()
	set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/local_install)
	find_package(Boost REQUIRED)
	if(Boost_FOUND)
		include_directories(${Boost_INCLUDE_DIR})
	endif()
	find_package(BLOSC REQUIRED)
	include_directories(${BLOSC_INCLUDE_DIR})
	target_compile_definitions(backend PRIVATE -DWITH_BLOSC)
	target_link_libraries(backend PRIVATE ${BLOSC_LIBRARIES})
	if(BUILD_CLI)
		target_compile_definitions(nyxus PRIVATE -DWITH_BLOSC)
		target_compile_definitions(nyxushie PRIVATE -DWITH_BLOSC) 	
		target_link_libraries(nyxus PRIVATE ${BLOSC_LIBRARIES})	
		target_link_libraries(nyxushie PRIVATE ${BLOSC_LIBRARIES})	
	endif()
endif()

find_package(FastLoader)
if(FastLoader_FOUND)
	list(APPEND Nyxus_LIBRARIES ${FastLoader_LIBRARIES})
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FastLoader_CXX_FLAGS}")
	include_directories(${FastLoader_INCLUDE_DIR})
else()
	message(FATAL_ERROR "Unable to find FastLoader which is required for Nyxus.")
endif()

if(USEGPU)
	target_link_libraries(backend PRIVATE nyxus_gpu)
       	set_target_properties(backend PROPERTIES CUDA_ARCHITECTURES "35;37;50;72")
	if(BUILD_CLI)
		target_link_libraries(nyxus PRIVATE nyxus_gpu)
        	set_target_properties(nyxus PROPERTIES CUDA_ARCHITECTURES "35;37;50;72")
	endif()
endif()

target_compile_definitions(backend PRIVATE WITH_PYTHON_H)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
target_compile_definitions(backend PRIVATE VERSION_INFO=${VERSION_INFO})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")

	target_link_libraries(backend PRIVATE stdc++fs)
	target_link_libraries(backend PRIVATE -static-libgcc -static-libstdc++)
	
	if(BUILD_CLI)
		target_link_libraries(nyxus PRIVATE -static-libgcc -static-libstdc++)
		target_link_libraries(nyxus PRIVATE stdc++fs)
		target_link_libraries(nyxushie PRIVATE -static-libgcc -static-libstdc++)
		target_link_libraries(nyxushie PRIVATE stdc++fs)
	endif()
endif()

target_link_libraries(backend PRIVATE ${Nyxus_LIBRARIES})

if (APPLE)
	set_target_properties(backend PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()


if(BUILD_CLI)
	target_link_libraries(nyxus PRIVATE ${Nyxus_LIBRARIES})
	target_link_libraries(nyxushie PRIVATE ${Nyxus_LIBRARIES})
endif()

## Running Tests
if (RUN_GTEST)
	# Download and unpack googletest at configure time
	configure_file(tests/CMakeLists.txt.gtest googletest-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)
	execute_process(COMMAND ${CMAKE_COMMAND} --build .
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)

	# Prevent GoogleTest from overriding our compiler/linker options
	# when building with Visual Studio
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

	# Add googletest directly to our build. This adds
	# the following targets: gtest, gtest_main, gmock
	# and gmock_main
	add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
			${CMAKE_BINARY_DIR}/googletest-build)

	# The gtest/gmock targets carry header search path
	# dependencies automatically when using CMake 2.8.11 or
	# later. Otherwise we have to add them here ourselves.
	if (CMAKE_VERSION VERSION_LESS 2.8.11)
		include_directories("${gtest_SOURCE_DIR}/include")
	endif ()
	#	enable_testing()
	add_subdirectory(tests)
	message(STATUS "GTEST downloaded and imported")
endif (RUN_GTEST)
