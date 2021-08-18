cmake_minimum_required(VERSION 3.10)

# set the project name
project(Nyx VERSION 1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the executable
add_executable(nyx_cmdline.exe src/nyx/bindings_py.cpp src/nyx/common_stats.cpp src/nyx/dirs_and_files.cpp src/nyx/features_calc_workflow.cpp src/nyx/f_circle.cpp src/nyx/f_contour.cpp src/nyx/f_convex_hull.cpp src/nyx/f_euler_number.cpp src/nyx/f_geo_len_thickness.cpp src/nyx/f_hexagonality_polygonality.cpp src/nyx/f_particle_metrics.cpp src/nyx/main.cpp src/nyx/output.cpp src/nyx/parallel.cpp src/nyx/scan_fastloader_way.cpp src/nyx/test_cxx_containers.cpp src/nyx/test_histogram.cpp)

# Configuring CMAKE for verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
find_package(Hedgehog 1.1.0 REQUIRED)
find_package(FastLoader 2.1.4 REQUIRED)
#find_package(CUDAToolkit QUIET)

