#!/bin/bash 

#=== Request echoing of each command executed 
set -x

# set OPT = "-c -Ofast" 
# echo "$OPT"

CXX=/usr/local/gcc94/bin/gcc
export CXX

GXX=/usr/local/gcc94/bin/g++
export GXX

INCLU='-I /home/ec2-user/work/prep-fastloader/FastLoader-main -I /home/ec2-user/work/prep-hedgehog/hedgehog-master -I /home/ec2-user/work/sensemaker4-nyx/lib/pybind11/include -I /home/ec2-user/gcc_install/gcc-9.4.0/isl-0.18/interface '
export INCLU

BUILDDIR=./build-4-linux
export BUILDDIR

OPTS='-w -c -std=c++17 -O2 '
export OPTS

#=== Prepare the build output directory
mkdir -p $BUILDDIR
rm $BUILDDIR/*
cd $BUILDDIR

#=== We're in the 'build' directory so all the source files are in '../'
$CXX $OPTS $INCLU ../src/nyx/f_haralick_texture.cpp           
$CXX $OPTS $INCLU ../src/nyx/f_convex_hull.cpp           
$CXX $OPTS $INCLU ../src/nyx/f_euler_number.cpp               
$CXX $OPTS $INCLU ../src/nyx/f_neighbors.cpp         
$CXX $OPTS $INCLU ../src/nyx/main.cpp             
$CXX $OPTS $INCLU ../src/nyx/parallel.cpp             
$CXX $OPTS $INCLU ../src/nyx/specfunc.cpp             
$CXX $OPTS $INCLU ../src/nyx/zernike.cpp
$CXX $OPTS $INCLU ../src/nyx/common_stats.cpp    
$CXX $OPTS $INCLU ../src/nyx/f_circle.cpp                
$CXX $OPTS $INCLU ../src/nyx/features_calc_workflow.cpp  
$CXX $OPTS $INCLU ../src/nyx/f_geo_len_thickness.cpp          
$CXX $OPTS $INCLU ../src/nyx/f_particle_metrics.cpp  
$CXX $OPTS $INCLU ../src/nyx/output_2_buffer.cpp  
$CXX $OPTS $INCLU ../src/nyx/roi_label.cpp            
$CXX $OPTS $INCLU ../src/nyx/test_cxx_containers.cpp
$CXX $OPTS $INCLU ../src/nyx/dirs_and_files.cpp  
$CXX $OPTS $INCLU ../src/nyx/f_contour.cpp               
$CXX $OPTS $INCLU ../src/nyx/featureset.cpp              
$CXX $OPTS $INCLU ../src/nyx/f_hexagonality_polygonality.cpp  
$CXX $OPTS $INCLU ../src/nyx/globals.cpp             
$CXX $OPTS $INCLU ../src/nyx/output_2_csv.cpp     
$CXX $OPTS $INCLU ../src/nyx/scan_fastloader_way.cpp  
$CXX $OPTS $INCLU ../src/nyx/test_histogram.cpp
$CXX $OPTS $INCLU ../src/nyx/histogram.cpp
$CXX $OPTS $INCLU ../src/nyx/environment.cpp
$CXX $OPTS $INCLU ../src/nyx/rotation.cpp
$CXX $OPTS $INCLU ../src/nyx/glrlm.cpp
$CXX $OPTS $INCLU ../src/nyx/glszm.cpp
$CXX $OPTS $INCLU ../src/nyx/gldm.cpp
$CXX $OPTS $INCLU ../src/nyx/ngtdm.cpp
$CXX $OPTS $INCLU ../src/nyx/hu.cpp
$CXX $OPTS $INCLU ../src/nyx/gabor.cpp
$CXX $OPTS $INCLU ../src/nyx/f_erosion_pixels.cpp
$CXX $OPTS $INCLU ../src/nyx/image_matrix.cpp

$GXX \
f_haralick_texture.o \
f_convex_hull.o \
f_euler_number.o \
f_neighbors.o \
main.o \
parallel.o \
specfunc.o \
zernike.o \
common_stats.o \
f_circle.o \
features_calc_workflow.o \
f_geo_len_thickness.o \
f_particle_metrics.o \
output_2_buffer.o \
roi_label.o \
test_cxx_containers.o \
dirs_and_files.o \
f_contour.o \
featureset.o \
f_hexagonality_polygonality.o \
globals.o \
output_2_csv.o \
scan_fastloader_way.o \
test_histogram.o \
histogram.o \
environment.o \
rotation.o \
glrlm.o \
glszm.o \
gldm.o \
ngtdm.o \
hu.o \
gabor.o \
f_erosion_pixels.o \
image_matrix.o \
-lm -ltiff -lfftw3 \
-lpthread \
-static-libstdc++ \
-o nyxus.exe

cd .. # Leave BUILDDIR

ls -la $BUILDDIR | grep nyxus.exe 

