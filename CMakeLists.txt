cmake_minimum_required(VERSION 3.4...3.18)

#==== Compiler Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- ")

project(SensemakerPythonLibrary)

add_subdirectory(lib/pybind11)

include_directories(
	lib/pybind11/include/
	lib/fastloader/
	lib/hedgehog/
	lib/libtiff
	lib/libtiff/include
	"C:/ProgramData/Anaconda3/include"
	)

pybind11_add_module(nyx_backend 
	src/nyx/bindings_py.cpp 
	src/nyx/globals.cpp 
	src/nyx/featureset.cpp 
	src/nyx/common_stats.cpp 
	src/nyx/dirs_and_files.cpp 
	src/nyx/features_calc_workflow.cpp 
	src/nyx/f_circle.cpp 
	src/nyx/f_contour.cpp 
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

find_library(LIBTIFF_LIBRARY libTiff_x64 lib/libtiff/lib)
target_link_libraries(nyx_backend PRIVATE ${LIBTIFF_LIBRARY})

