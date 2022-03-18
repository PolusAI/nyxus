@echo off
Rem Download and Install libtiff
@echo on
curl https://download.osgeo.org/libtiff/tiff-4.3.0.zip -o libtiff.zip
tar -xf libtiff.zip
pushd tiff-4.3.0
mkdir build_local
pushd build_local
cmake ..
cmake --build . --config Release
cmake --install . 
popd
popd