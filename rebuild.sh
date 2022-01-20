#!/bin/bash 

#=== Request echoing of each command executed 
set -x

# set OPT = "-c -Ofast" 
# echo "$OPT"

CXX=/usr/local/gcc94/bin/gcc
export CXX

GXX=/usr/local/gcc94/bin/g++
export GXX

INCLU='-I /home/ec2-user/work/prep-fastloader/FastLoader-main -I /home/ec2-user/work/prep-hedgehog/hedgehog-master -I /usr/include/python3.7m -I /home/ec2-user/work/sensemaker4-nyx/lib/pybind11/include -I /home/ec2-user/gcc_install/gcc-9.4.0/isl-0.18/interface '
export INCLU

BUILDDIR=./build-4-linux
export BUILDDIR

OPTS='-c -std=c++17 -D CHECKTIMING -g '
export OPTS

#=== Prepare the build output directory
mkdir -p $BUILDDIR
rm $BUILDDIR/*
cd $BUILDDIR

#=== Touch main.cpp to advance the build timestamp
touch ../src/nyx/main.cpp

#=== We're in the 'build' directory so all the source files are in '../'
# ---Features---
$GXX $OPTS $INCLU ../src/nyx/features/caliper.cpp
$GXX $OPTS $INCLU ../src/nyx/features/caliper_nassenstein.cpp
$GXX $OPTS $INCLU ../src/nyx/features/chords.cpp
$GXX $OPTS $INCLU ../src/nyx/features/chords_nontriv.cpp
$GXX $OPTS $INCLU ../src/nyx/features/circle.cpp
$GXX $OPTS $INCLU ../src/nyx/features/contour.cpp
$GXX $OPTS $INCLU ../src/nyx/features/convex_hull_nontriv.cpp
$GXX $OPTS $INCLU ../src/nyx/features/ellipse_fitting.cpp
$GXX $OPTS $INCLU ../src/nyx/features/erosion_pixels.cpp
$GXX $OPTS $INCLU ../src/nyx/features/euler_number.cpp
$GXX $OPTS $INCLU ../src/nyx/features/extrema.cpp
$GXX $OPTS $INCLU ../src/nyx/features/fractal_dim.cpp
$GXX $OPTS $INCLU ../src/nyx/features/gabor.cpp
$GXX $OPTS $INCLU ../src/nyx/features/gabor_nontriv.cpp
$GXX $OPTS $INCLU ../src/nyx/features/geo_len_thickness.cpp
$GXX $OPTS $INCLU ../src/nyx/features/glcm.cpp
$GXX $OPTS $INCLU ../src/nyx/features/gldm.cpp
$GXX $OPTS $INCLU ../src/nyx/features/glrlm.cpp
$GXX $OPTS $INCLU ../src/nyx/features/glszm.cpp
$GXX $OPTS $INCLU ../src/nyx/features/hexagonality_polygonality.cpp
$GXX $OPTS $INCLU ../src/nyx/features/histogram.cpp
$GXX $OPTS $INCLU ../src/nyx/features/image_matrix.cpp
$GXX $OPTS $INCLU ../src/nyx/features/image_matrix_nontriv.cpp
$GXX $OPTS $INCLU ../src/nyx/features/image_moments.cpp
$GXX $OPTS $INCLU ../src/nyx/features/intensity.cpp
$GXX $OPTS $INCLU ../src/nyx/features/neighbors.cpp
$GXX $OPTS $INCLU ../src/nyx/features/ngtdm.cpp
$GXX $OPTS $INCLU ../src/nyx/features/pixel_intensity.cpp
$GXX $OPTS $INCLU ../src/nyx/features/radial_distribution.cpp
$GXX $OPTS $INCLU ../src/nyx/features/roi_label.cpp
$GXX $OPTS $INCLU ../src/nyx/features/roi_radius.cpp
$GXX $OPTS $INCLU ../src/nyx/features/rotation.cpp
$GXX $OPTS $INCLU ../src/nyx/features/specfunc.cpp
$GXX $OPTS $INCLU ../src/nyx/features/zernike.cpp
# ---Helpers---
$GXX $OPTS $INCLU ../src/nyx/helpers/system_resource.cpp
$GXX $OPTS $INCLU ../src/nyx/helpers/timing.cpp
# ---Python bindings---
$GXX $OPTS $INCLU ../src/nyx/python_interface/bindings_py.cpp
# ---Root source files---
$GXX $OPTS $INCLU ../src/nyx/common_stats.cpp
$GXX $OPTS $INCLU ../src/nyx/dirs_and_files.cpp
$GXX $OPTS $INCLU ../src/nyx/environment.cpp
$GXX $OPTS $INCLU ../src/nyx/feature_method.cpp
$GXX $OPTS $INCLU ../src/nyx/feature_mgr.cpp
$GXX $OPTS $INCLU ../src/nyx/feature_mgr_init.cpp
$GXX $OPTS $INCLU ../src/nyx/features_calc_workflow.cpp
$GXX $OPTS $INCLU ../src/nyx/featureset.cpp
$GXX $OPTS $INCLU ../src/nyx/globals.cpp
$GXX $OPTS $INCLU ../src/nyx/image_loader.cpp
$GXX $OPTS $INCLU ../src/nyx/main.cpp
$GXX $OPTS $INCLU ../src/nyx/output_2_buffer.cpp
$GXX $OPTS $INCLU ../src/nyx/output_2_csv.cpp
$GXX $OPTS $INCLU ../src/nyx/parallel.cpp
$GXX $OPTS $INCLU ../src/nyx/phase1.cpp
$GXX $OPTS $INCLU ../src/nyx/phase2.cpp
$GXX $OPTS $INCLU ../src/nyx/phase3.cpp
$GXX $OPTS $INCLU ../src/nyx/reduce_by_feature.cpp
$GXX $OPTS $INCLU ../src/nyx/reduce_trivial_rois.cpp
$GXX $OPTS $INCLU ../src/nyx/roi_cache.cpp
$GXX $OPTS $INCLU ../src/nyx/scan_fastloader_way.cpp

$GXX \
caliper.o \
caliper_nassenstein.o \
chords.o \
chords_nontriv.o \
circle.o \
contour.o \
convex_hull_nontriv.o \
ellipse_fitting.o \
erosion_pixels.o \
euler_number.o \
extrema.o \
fractal_dim.o \
gabor.o \
gabor_nontriv.o \
geo_len_thickness.o \
glcm.o \
gldm.o \
glrlm.o \
glszm.o \
hexagonality_polygonality.o \
histogram.o \
image_matrix.o \
image_matrix_nontriv.o \
image_moments.o \
intensity.o \
neighbors.o \
ngtdm.o \
pixel_intensity.o \
radial_distribution.o \
roi_label.o \
roi_radius.o \
rotation.o \
specfunc.o \
zernike.o \
system_resource.o \
timing.o \
bindings_py.o \
common_stats.o \
dirs_and_files.o \
environment.o \
feature_method.o \
feature_mgr.o \
feature_mgr_init.o \
features_calc_workflow.o \
featureset.o \
globals.o \
image_loader.o \
main.o \
output_2_buffer.o \
output_2_csv.o \
parallel.o \
phase1.o \
phase2.o \
phase3.o \
reduce_by_feature.o \
reduce_trivial_rois.o \
roi_cache.o \
scan_fastloader_way.o \
-lpython3.7m \
-lm -ltiff -lfftw3 \
-lpthread \
-static-libstdc++ \
-o nyxus.exe

cd .. # Leave BUILDDIR

ls -la $BUILDDIR | grep nyxus.exe 

