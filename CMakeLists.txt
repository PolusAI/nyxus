project(SensemakerPythonLibrary)
cmake_minimum_required(VERSION 3.4...3.18)

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
add_subdirectory(lib/pybind11)

#==== Include directories. Note: libtiff (required by FastLoader) is 
#	the only dependency needing manual preparation - unzip, cmake-install 
#	it, and put its headers in lib/libtiff/include, and 
#	its library - in lib/libtiff/lib
include_directories(
	lib/pybind11/include/
	lib/fastloader/
	lib/hedgehog/
)

# Search for libtiff. If it's found then great. If not then we expect 
# manual preparation. unzip, cmake-install it, and put its headers in 
# lib/libtiff/include, and its library - in lib/libtiff/lib
find_package(TIFF)

if(TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIR})
else()
	include_directories(
		lib/libtiff
		lib/libtiff/include
	)
endif()


#==== Source files
pybind11_add_module(nyx_backend 
	src/nyx/bindings_py.cpp 
	src/nyx/globals.cpp 
	src/nyx/featureset.cpp 
	src/nyx/common_stats.cpp 
	src/nyx/dirs_and_files.cpp 
	src/nyx/features_calc_workflow.cpp 
	src/nyx/roi_label.cpp
	src/nyx/f_cellprofiler_texture.cpp
	src/nyx/f_circle.cpp 
	src/nyx/f_neighbors.cpp
	src/nyx/f_contour.cpp 
	src/nyx/zernike.cpp
	src/nyx/f_convex_hull.cpp 
	src/nyx/f_euler_number.cpp 
	src/nyx/f_geo_len_thickness.cpp 
	src/nyx/f_hexagonality_polygonality.cpp 
	src/nyx/f_particle_metrics.cpp 
	src/nyx/main.cpp 
	src/nyx/output_2_buffer.cpp 
	src/nyx/output_2_csv.cpp 
	src/nyx/parallel.cpp 
	src/nyx/scan_fastloader_way.cpp 
	src/nyx/test_cxx_containers.cpp 
	src/nyx/test_histogram.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a define (VERSION_INFO) here.
target_compile_definitions(nyx_backend PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	target_link_libraries(nyx_backend PRIVATE stdc++fs)
endif()

#==== Separate link instruction for libtiff. See libtiff's deployment instrunction above
if(TIFF_FOUND)
	target_link_libraries(nyx_backend PRIVATE ${TIFF_LIBRARIES})
else()
	find_library(LIBTIFF_LIBRARY libTiff_x64 lib/libtiff/lib)
	target_link_libraries(nyx_backend PRIVATE ${LIBTIFF_LIBRARY})
endif()

