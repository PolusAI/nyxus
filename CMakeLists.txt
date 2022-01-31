cmake_minimum_required(VERSION 3.12)
project(SensemakerPythonLibrary)

#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- ")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

#==== Pybind11
find_package(Python COMPONENTS Interpreter Development)
add_subdirectory(lib/pybind11)

if (Python_FOUND OR Python_INCLUDE_DIRS)
	add_compile_definitions(WITH_PYTHON_H)
endif()


include_directories(
	lib/pybind11/include/
	${Python_INCLUDE_DIRS}
)

message(STATUS "PYTHON HEADERS: ${Python_INCLUDE_DIRS}")

find_package(OpenMP COMPONENTS C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake-modules")

#==== Include directories. Note: libtiff (required by FastLoader) is 
#	the only dependency needing manual preparation - unzip, cmake-install 
#	it, and put its headers in lib/libtiff/include, and 
#	its library - in lib/libtiff/lib
find_package(FastLoader)

if(FastLoader_FOUND)
	list(APPEND Nyxus_LIBRARIES ${FastLoader_LIBRARIES})
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FastLoader_CXX_FLAGS}")
	include_directories(${FastLoader_INCLUDE_DIR})
else()
	message(FATAL_ERROR "Unable to find FastLoader which is required for Nyxus.")
endif()

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
	src/nyx/features/gldm.cpp
	src/nyx/features/glrlm.cpp
	src/nyx/features/glszm.cpp
	src/nyx/features/hexagonality_polygonality.cpp
	src/nyx/features/histogram.cpp
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
	src/nyx/helpers/timing.cpp
	src/nyx/common_stats.cpp
	src/nyx/dirs_and_files.cpp
	src/nyx/environment.cpp
	src/nyx/feature_method.cpp
	src/nyx/feature_mgr.cpp
	src/nyx/feature_mgr_init.cpp
	src/nyx/features_calc_workflow.cpp
	src/nyx/featureset.cpp
	src/nyx/globals.cpp
	src/nyx/image_loader.cpp
	src/nyx/main.cpp
	src/nyx/output_2_buffer.cpp
	src/nyx/output_2_csv.cpp
	src/nyx/parallel.cpp
	src/nyx/phase1.cpp
	src/nyx/phase2.cpp
	src/nyx/phase3.cpp
	src/nyx/reduce_by_feature.cpp
	src/nyx/reduce_trivial_rois.cpp
	src/nyx/roi_cache.cpp
	src/nyx/scan_fastloader_way.cpp
)

# CLI
if(BUILD_CLI)
	add_executable(nyxus ${SOURCE} src/nyx/main.cpp)
endif()

# Python bindings.
pybind11_add_module(backend 
	${SOURCE}
	src/nyx/python/new_bindings_py.cpp
)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
target_compile_definitions(backend PRIVATE VERSION_INFO=${VERSION_INFO})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_link_libraries(backend PRIVATE stdc++fs)
	
	if(BUILD_CLI)
		target_link_libraries(nyxus PRIVATE stdc++fs)
	endif()
endif()

target_link_libraries(backend PRIVATE -static-libgcc -static-libstdc++)
target_link_libraries(backend PRIVATE ${Nyxus_LIBRARIES})
target_link_libraries(backend PRIVATE ${Python_LIBRARIES})

if(BUILD_CLI)
	target_link_libraries(nyxus PRIVATE -static-libgcc -static-libstdc++)
	target_link_libraries(nyxus PRIVATE ${Nyxus_LIBRARIES})
	target_link_libraries(nyxus PRIVATE ${Python_LIBRARIES})
endif()

