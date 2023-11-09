cmake_minimum_required(VERSION 3.20)
project(AxleNyxus)
set(CMAKE_VERBOSE_MAKEFILE ON)
include(GNUInstallDirs)
include(CheckLanguage)

# this is a workaround for GitHub Action for wheelbuiling
if(DEFINED ENV{NYXUS_DEP_DIR})
	set(CMAKE_PREFIX_PATH $ENV{NYXUS_DEP_DIR})
	link_directories($ENV{NYXUS_DEP_DIR}/${CMAKE_INSTALL_LIBDIR})
endif()

# Build Options
option(USE_Z5 "Support OMEZarr" OFF)
option(USE_DCMTK "Use DCMTK" OFF)
option(USE_ARROW "Use Arrow" OFF)
option(ALLEXTRAS "Building with all I/O support" OFF)
option(NOEXTRAS "Building with Only Tiff and Pybind11(For Python package)" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake-modules")

if(NOEXTRAS)
	set(USE_Z5 OFF)
	set(USE_DCMTK OFF)
	set(USE_ARROW OFF) 
endif()

if(ALLEXTRAS)
	set(USE_Z5 ON)
	set(USE_DCMTK ON)
	set(USE_ARROW ON) 
endif()

set(Z5_REQUESTED ${USE_Z5})
set(ARROW_REQUESTED ${USE_ARROW})
set(DCMTK_REQUESTED ${USE_DCMTK})

option(USEGPU "Use GPU" ON)
set(CUDA_REQUESTED ${USEGPU})
check_language(CUDA)
find_package(CUDAToolkit)
if (USEGPU)
	if (NOT(CMAKE_CUDA_COMPILER) OR NOT(CUDAToolkit_FOUND))
		set(USEGPU OFF)
	endif()
endif()

if(USEGPU) 
	enable_language(CUDA)
	set_target_properties("CUDA Toolkit Version {CUDAToolkit_VERSION}")
	if (CUDAToolkit_VERSION_MAJOR STREQUAL "10")
		set(CUDA_ARCH_LIST "35;37;50;72;75")
	elseif (CUDAToolkit_VERSION_MAJOR STREQUAL "11")
		if(CUDAToolkit_VERSION_MINOR STREQUAL "0")
			set(CUDA_ARCH_LIST "35;37;50;72;75")
		elseif (CUDAToolkit_VERSION_MINOR STREQUAL "1" OR CUDAToolkit_VERSION_MINOR STREQUAL "2")
			set(CUDA_ARCH_LIST "35;37;50;72;75;80;86")
		else () # for now, we assume CUDA 11.2+ supports all these archs.
			set(CUDA_ARCH_LIST "35;37;50;72;75;80;86")
		endif()
	elseif (CUDAToolkit_VERSION_MAJOR STREQUAL "12")
		set(CUDA_ARCH_LIST "52;60;75;80;86;89;90")
	else() # some old CUDA version (<10)
		set(CUDA_ARCH_LIST "50")
	endif()
else()
	if(CUDA_REQUESTED)
		message(WARNING "CUDA Toolkit not found. USEGPU flag will be ignored.")
	endif()
endif()

#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USEGPU)
    enable_language("CUDA")
	SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
	set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
	message("Building with compute capability for ${CUDA_ARCH_LIST}.")
    add_definitions(-DUSE_GPU)
	if (CUDAToolkit_VERSION_MAJOR STREQUAL "10")
		set(CMAKE_CUDA_STANDARD 14)
	elseif (CUDAToolkit_VERSION_MAJOR STREQUAL "11" OR CUDAToolkit_VERSION_MAJOR STREQUAL "12")
		set(CMAKE_CUDA_STANDARD 17)
	endif()
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)	
endif()

if(USE_Z5)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		#since xtensor does not built with GCC 7.5 and lower
		if(NOT(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.5)) 
			set(USE_Z5 OFF)
			message(WARNING "OMEZarr support is not available with GCC 7.5 and older compiler.")
		endif()
	endif()
	find_package(ZLIB)
	if (NOT ZLIB_FOUND)
		set(USE_Z5 OFF)
		message(WARNING "zlib not found. Blosc and hereby OMEZarr support will not be available.")
	endif()
	find_package(BLOSC)
	if(NOT BLOSC_FOUND)
		set(USE_Z5 OFF)
		message(WARNING "Blosc not found. OMEZarr support will not be available.")
	endif()
	find_package(Boost)
	if(NOT Boost_FOUND)
		set(USE_Z5 OFF)
		message(WARNING "Boost not found. OMEZarr support will not be available.")
	endif()

	find_package(nlohmann_json)
	if(NOT nlohmann_json_FOUND)
		set(USE_Z5 OFF)
		message(WARNING "nlohmann_json not found. OMEZarr support will not be available.")
	endif()

	find_file(Z5 "z5/z5.hxx")
	if(NOT Z5)
		set(USE_Z5 OFF)
		message(WARNING "Z5 not found. OMEZarr support will not be available.")
	endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- /EHsc /bigobj")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

#== GTest 
option(RUN_GTEST "Downloads google unit test API and runs google test scripts to test Nyxus" OFF)

#==== Pybind11
find_package(pybind11 CONFIG REQUIRED)

#==== Source files
set(SOURCE
	src/nyx/arrow_output_stream.cpp
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
	src/nyx/features/gldzm.cpp
	src/nyx/features/glrlm.cpp
	src/nyx/features/glszm.cpp
	src/nyx/features/hexagonality_polygonality.cpp
	src/nyx/features/image_matrix.cpp
	src/nyx/features/image_matrix_nontriv.cpp
	src/nyx/features/image_moments.cpp
	src/nyx/features/image_moments_nontriv.cpp
	src/nyx/features/intensity.cpp
	src/nyx/features/neighbors.cpp
	src/nyx/features/ngldm.cpp
	src/nyx/features/ngtdm.cpp
	src/nyx/features/pixel.cpp
	src/nyx/features/radial_distribution.cpp
	src/nyx/features/roi_radius.cpp
	src/nyx/features/rotation.cpp
	src/nyx/features/specfunc.cpp
	src/nyx/features/zernike.cpp
	src/nyx/features/zernike_nontriv.cpp
	src/nyx/helpers/timing.cpp
	src/nyx/cli_fpimage_options.cpp
	src/nyx/cli_gabor_options.cpp
	src/nyx/cli_nested_roi_options.cpp
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
	src/nyx/output_writers.cpp
	src/nyx/parallel.cpp
	src/nyx/phase1.cpp
	src/nyx/phase2.cpp
	src/nyx/phase3.cpp
	src/nyx/pixel_feed.cpp
	src/nyx/reduce_by_feature.cpp
	src/nyx/reduce_trivial_rois.cpp
	src/nyx/roi_blacklist.cpp
	src/nyx/roi_cache.cpp
	src/nyx/roi_cache_basic.cpp
	src/nyx/scan_fastloader_way.cpp
)


# CLI
if(BUILD_CLI)
	add_executable(nyxus ${SOURCE} src/nyx/main_nyxus.cpp)
endif()

# Python bindings.
if(BUILD_LIB)
	pybind11_add_module(backend 
		${SOURCE}
		src/nyx/python/nested_roi_py.cpp
		src/nyx/python/new_bindings_py.cpp
		src/nyx/image_loader1x.cpp
		src/nyx/nested_roi.cpp
	)
endif()

#== Required for OMETiff  
find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	list(APPEND Nyxus_LIBRARIES ${TIFF_LIBRARIES})
	include_directories (${TIFF_INCLUDE_DIR})
endif (TIFF_FOUND)

#== Required for OMEZarr 
if(USE_Z5)
	add_definitions(-DOMEZARR_SUPPORT)
	if(BLOSC_FOUND)
		add_definitions(-DWITH_BLOSC)
		list(APPEND Nyxus_LIBRARIES ${BLOSC_LIBRARIES})
	endif()

	if(Boost_FOUND)
		include_directories(${Boost_INCLUDE_DIR})
	endif()

	if (ZLIB_FOUND)
		list(APPEND Nyxus_LIBRARIES ${ZLIB_LIBRARIES} )
		include_directories (${ZLIB_INCLUDE_DIR})
	endif()
	
	if(nlohmann_json_FOUND)
		include_directories(${nlohmann_json_INCLUDE_DIR})
	endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	find_package(Threads QUIET)
	if (Threads_FOUND)
		if (CMAKE_USE_PTHREADS_INIT)
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
		endif (CMAKE_USE_PTHREADS_INIT)
		list(APPEND Nyxus_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
	else ()
		message(STATUS "Unable to find threads. Nyxus must have a threading library i.e. pthreads.")
	endif ()
endif()

if(USE_DCMTK)
	#== Required for DICOM
	find_package(DCMTK)
	if(DCMTK_FOUND)
		find_package(ZLIB)
		if (NOT ZLIB_FOUND)
			set(USE_DCMTK OFF)
			message(WARNING "zlib not found. Dicom support will not be available.")
		endif()
		find_package(fmjpeg2k)
		if(NOT fmjpeg2k_FOUND)
			message(WARNING "fmjpeg2k not found. JPEG2K support will not be available.")
		endif()
	else()
		set(USE_DCMTK OFF)
	endif()
endif()

if(USE_DCMTK)
	if(DCMTK_FOUND)
		include_directories(${DCMTK_INCLUDE_DIRS})
		add_definitions(-DDICOM_SUPPORT)
		foreach(LIB ${DCMTK_LIBRARIES})
			FIND_LIBRARY(FOUND_LIB_${LIB} ${LIB})
			LIST(APPEND Nyxus_LIBRARIES ${FOUND_LIB_${LIB}})
		endforeach(LIB)
	endif()

	if (ZLIB_FOUND)
		list(APPEND Nyxus_LIBRARIES ${ZLIB_LIBRARIES} )
		include_directories (${ZLIB_INCLUDE_DIR})
	endif()

	if(fmjpeg2k_FOUND)
		add_definitions(-DJPEG2K_SUPPORT)
		foreach(LIB ${FMJPEG2K_LIBRARIES})
			FIND_LIBRARY(FOUND_LIB_${LIB} ${LIB})
			LIST(APPEND Nyxus_LIBRARIES ${FOUND_LIB_${LIB}})
		endforeach(LIB)
	endif()
endif()

if(USEGPU)
	include_directories("${CUDAToolkit_INCLUDE_DIRS}")
	set(GPU_SOURCE_FILES
        src/nyx/gpu/gabor.cu
		src/nyx/gpu/gpu_helpers.cu
		src/nyx/gpu/geomoments_central.cu
		src/nyx/gpu/geomoments_hu.cu
		src/nyx/gpu/geomoments_main.cu
		src/nyx/gpu/geomoments_norm.cu
		src/nyx/gpu/geomoments_origin.cu
		src/nyx/gpu/geomoments_raw.cu
		src/nyx/gpu/geomoments_weighting.cu
	)
	add_library(nyxus_gpu ${GPU_SOURCE_FILES})

	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		target_compile_options(nyxus_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fPIC>)
	endif()
	list(APPEND Nyxus_LIBRARIES CUDA::cudart)
	list(APPEND Nyxus_LIBRARIES CUDA::cufft)
	list(APPEND Nyxus_LIBRARIES nyxus_gpu)

	if(BUILD_LIB)
		set_target_properties(backend PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
	endif()

	if(BUILD_CLI)
		set_target_properties(nyxus PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
	endif()
endif()

if(USE_ARROW)
	option(PARQUET_LINK_SHARED "Link to the Parquet shared library" ON)
	# Look for installed packages the system
	find_package(Arrow)
	if (Arrow_FOUND) 
		# Look for installed packages the system
		find_package(Parquet)
		if (Parquet_FOUND)
			if (BUILD_LIB)
				# Find Python
				find_package(Python REQUIRED COMPONENTS Interpreter)
				# Execute the Python script to find pyarrow include path
				execute_process(
					COMMAND ${Python_EXECUTABLE} -c "import pyarrow as pa; print(pa.get_include())"
					OUTPUT_VARIABLE PATH_OUTPUT
					OUTPUT_STRIP_TRAILING_WHITESPACE
				)
				# Store the path in a CMake variable
				set(PYARROW_INCLUDE_PATH "${PATH_OUTPUT}" CACHE INTERNAL "Path to pyarrow include directory")
				# Execute the Python script to find pyarrow library path
				execute_process(
					COMMAND ${Python_EXECUTABLE} -c "import pyarrow as pa; print(pa.get_library_dirs()[0])"
					OUTPUT_VARIABLE LIB_PATH_OUTPUT
					OUTPUT_STRIP_TRAILING_WHITESPACE
				)
				# Store the path in a CMake variable
				set(PYARROW_LIB_PATH "${LIB_PATH_OUTPUT}" CACHE INTERNAL "Path to pyarrow library directory")
				if (PYARROW_INCLUDE_PATH STREQUAL "" OR PYARROW_LIB_PATH STREQUAL "")
					message(STATUS "Pyarrow library not found. The build will continue without Arrow support.")
					set(USE_ARROW OFF)
				endif()

				# Print the path
				message(STATUS "PyArrow Include Path: ${PYARROW_INCLUDE_PATH}")
				message(STATUS "PyArrow Library Path: ${PYARROW_LIB_PATH}")
			endif()
		
		else()
			message(WARNING "The Parquet library was not found. The build will continue without Arrow support.")
			set(USE_ARROW OFF)
		endif()

	else ()
		message(WARNING "The Arrow library was not found. The build will continue without Arrow support.")
		set(USE_ARROW OFF)
	endif()
endif()

if(USE_ARROW)
	add_definitions(-DUSE_ARROW)
	list(APPEND Nyxus_LIBRARIES arrow_shared)

	if(PARQUET_LINK_SHARED)
		list(APPEND Nyxus_LIBRARIES parquet_shared)
	else()
		list(APPEND Nyxus_LIBRARIES parquet_static)
	endif()
	if (BUILD_LIB)
		# add pyarrow include path
		include_directories(${PYARROW_INCLUDE_PATH})
		#add pyarrow library
		list(APPEND Nyxus_LIBRARIES arrow_python)
		target_link_directories(backend PRIVATE ${PYARROW_LIB_PATH})
	endif()
endif()

if(BUILD_LIB)
	target_compile_definitions(backend PRIVATE WITH_PYTHON_H)
	# VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
	target_compile_definitions(backend PRIVATE VERSION_INFO=${VERSION_INFO})
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
	list(APPEND Nyxus_LIBRARIES stdc++fs)
endif()

if(BUILD_LIB)
	target_link_libraries(backend PRIVATE ${Nyxus_LIBRARIES})
endif()

if (APPLE AND BUILD_LIB)
	set_target_properties(backend PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

if(BUILD_CLI)
	target_link_libraries(nyxus PRIVATE ${Nyxus_LIBRARIES})
endif()

message(STATUS "Nyxus Dependencies Summary")
message(STATUS "================================")
message(STATUS "Support | Requested | Found")
message(STATUS "================================")
message(STATUS "Z5      | ${Z5_REQUESTED} | ${USE_Z5}")
message(STATUS "Arrow   | ${ARROW_REQUESTED} | ${USE_ARROW}")
message(STATUS "Dicom   | ${DCMTK_REQUESTED} | ${USE_DCMTK}")
message(STATUS "CUDA    | ${CUDA_REQUESTED} | ${USEGPU}")
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
