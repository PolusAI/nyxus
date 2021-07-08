#!/bin/bash 

#=== Request echoing of each command executed 
set -x

# set OPT = "-c -Ofast" 
# echo "$OPT"

CXX=/usr/local/gcc94/bin/gcc
export CXX

GXX=/usr/local/gcc94/bin/g++
export GXX

INCLU='-I /home/ec2-user/work/prep-fastloader/FastLoader-main -I /home/ec2-user/work/prep-hedgehog/hedgehog-master'
export INCLU

BUILDDIR=./build
export BUILDDIR

OPTS='-w -c -std=c++17'
export OPTS

#=== Prepare the build output directory
mkdir -p $BUILDDIR
rm $BUILDDIR/*
cd $BUILDDIR

#=== We're in the 'build' directory so all the source files are in '../'
$CXX $OPTS $INCLU ../dirs_and_files.cpp 
$CXX $OPTS $INCLU ../features.cpp  
$CXX $OPTS $INCLU ../main.cpp  
$CXX $OPTS $INCLU ../output.cpp  
$CXX $OPTS $INCLU ../scan_fastloader_way.cpp  

$GXX \
dirs_and_files.o \
features.o  \
main.o  \
output.o  \
scan_fastloader_way.o  \
-lm -ltiff -lfftw3 \
-lpthread \
-static-libstdc++ \
-o sensemaker.exe 

cd .. # Leave BUILDDIR

ls -la $BUILDDIR
