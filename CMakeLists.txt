cmake_minimum_required(VERSION 3.10)

#==== Configuring CMAKE for verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release")

#==== Dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
#find_package(CUDA REQUIRED)
# find_package(Hedgehog REQUIRED)	# Hedgehog 1.1.0 REQUIRED
# find_package(FastLoader REQUIRED)	# FastLoader 2.1.4 REQUIRED

#==== Set the project name 
project(sensemaker VERSION 1.0 LANGUAGES CXX)

# -- and C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#==== outdir
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

#==== Compiler Options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive- ")
include_directories(
	$(Sensemaker_SOURCE_DIR)/lib/pybind11/include/
	$(Sensemaker_SOURCE_DIR)/lib/fastloader/
	$(Sensemaker_SOURCE_DIR)/lib/hedgehog/
	"C:/ProgramData/Anaconda3/include"
	"C:/WORK/AXLE/tiff-4.3.0/libtiff"
	"C:/WORK/AXLE/tiff-4.3.0/BUILD2/libtiff"
	)
	
set(SOURCES
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
	src/nyx/test_histogram.cpp
)	


#==== Library
#add_library(nyx_backend SHARED src/nyx/globals.cpp src/nyx/featureset.cpp src/nyx/bindings_py.cpp src/nyx/common_stats.cpp src/nyx/dirs_and_files.cpp src/nyx/features_calc_workflow.cpp src/nyx/f_circle.cpp src/nyx/f_contour.cpp src/nyx/f_convex_hull.cpp src/nyx/f_euler_number.cpp src/nyx/f_geo_len_thickness.cpp src/nyx/f_hexagonality_polygonality.cpp src/nyx/f_particle_metrics.cpp src/nyx/main.cpp src/nyx/output_2_buffer.cpp src/nyx/output_2_csv.cpp src/nyx/parallel.cpp src/nyx/scan_fastloader_way.cpp src/nyx/test_cxx_containers.cpp src/nyx/test_histogram.cpp)
add_library(nyx_backend SHARED ${SOURCES})

#--	add_subdirectory(lib/pybind11)
#pybind11_add_module(nyx_backend MODULE ${SOURCES} "src/nyx/bindings_py.cpp")

#add_executable(nyx_cmdline src/nyx/bindings_py.cpp src/nyx/common_stats.cpp src/nyx/dirs_and_files.cpp src/nyx/features_calc_workflow.cpp src/nyx/f_circle.cpp src/nyx/f_contour.cpp src/nyx/f_convex_hull.cpp src/nyx/f_euler_number.cpp src/nyx/f_geo_len_thickness.cpp src/nyx/f_hexagonality_polygonality.cpp src/nyx/f_particle_metrics.cpp src/nyx/main.cpp src/nyx/output.cpp src/nyx/parallel.cpp src/nyx/scan_fastloader_way.cpp src/nyx/test_cxx_containers.cpp src/nyx/test_histogram.cpp)

#--no effect-- link_directories("C:/ProgramData/Anaconda3/libs" "C:/Program Files (x86)/GnuWin32/lib")
target_link_libraries(nyx_backend "C:/Program Files (x86)/GnuWin32/lib/libTiff_x64.lib" "C:/ProgramData/Anaconda3/libs/python38.lib")
