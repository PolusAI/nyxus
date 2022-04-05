
@echo off
Rem Download and Install zlib
@echo on
git clone https://github.com/madler/zlib.git
pushd zlib
mkdir build_local
pushd build_local
cmake ..
cmake --build . --config Release
cmake --install . 
popd
popd

@echo off
Rem Download and Install libdeflate and libtiff 
@echo on

curl https://download.osgeo.org/libtiff/tiff-4.3.0.zip -o libtiff.zip
tar -xf libtiff.zip
pushd tiff-4.3.0
mkdir build_local
pushd build_local
git clone https://github.com/ebiggers/libdeflate.git
pushd libdeflate
nmake /f Makefile.msc
popd
cmake -DDeflate_INCLUDE_DIR=./libdeflate -DDeflate_LIBRARY_RELEASE=./libdeflate/libdeflate.lib ..
cmake --build . --config Release
cmake --install . 
popd
popd