cmake_minimum_required(VERSION 3.10)

# set the project name
project(SlideSensemaker VERSION 1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the executable
add_executable(ssm src/main.cpp src/scan_fastloader_way.cpp src/dirs_and_files.cpp)

# Configuring CMAKE for verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
find_package(Hedgehog 1.1.0 REQUIRED)
find_package(FastLoader 2.1.4 REQUIRED)
find_package(CUDAToolkit QUIET)

