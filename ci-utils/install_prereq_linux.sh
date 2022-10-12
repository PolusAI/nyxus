#!/bin/bash
# Usage: $bash install_prereq_linux.sh $INSTALL_DIR
# Default $INSTALL_DIR = ./local_install
#
if [ -z "$1" ]
then
      echo "No path to the Nyxus source location provided"
      echo "Creating local_install directory"
      Z5_INSTALL_DIR="local_install"
else
     Z5_INSTALL_DIR=$1
fi
echo $Z5_INSTALL_DIR
mkdir -p $Z5_INSTALL_DIR
mkdir -p $Z5_INSTALL_DIR/include

for i in {1..5}
do
    curl -L https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.bz2 -o boost_1_79_0.tar.bz2 
    if [ -f "boost_1_79_0.tar.bz2" ] ; then
        break
    fi
done
tar --bzip2 -xf boost_1_79_0.tar.bz2 
cd boost_1_79_0 
./bootstrap.sh 
./b2 headers
cp -r boost ../$Z5_INSTALL_DIR/include
cd ../

git clone https://github.com/madler/zlib.git
cd zlib
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone https://github.com/Blosc/c-blosc.git 
cd c-blosc 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone https://github.com/xtensor-stack/xtl.git 
cd xtl 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone https://github.com/xtensor-stack/xtensor.git 
cd xtensor 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone https://github.com/xtensor-stack/xsimd.git 
cd xsimd 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone  https://github.com/nlohmann/json.git
cd json
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/ ..  
make install/fast
cd ../../

git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/  -DPYBIND11_TEST=OFF ..
make install
cd ../../

git clone https://github.com/constantinpape/z5.git
cd z5
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/ -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF  ..
make install
cd ../../

git clone  https://github.com/ebiggers/libdeflate.git
cd libdeflate
PREFIX= LIBDIR=/lib64  DESTDIR=../$Z5_INSTALL_DIR/ make  install
cd ../

for i in {1..5}
do
    curl https://download.osgeo.org/libtiff/tiff-4.4.0.zip -o libtiff.zip 
    if [ -f "libtiff.zip" ] ; then
        break
    fi
done

unzip libtiff.zip
cd libtiff
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/   ..
make install
cd ../../

if [ "$ON_GITHUB" == "TRUE" ]; then
    mkdir -p /tmp/nyxus/
    cp -r local_install/lib64 /tmp/nyxus/lib/
    cp -r local_install/lib /tmp/nyxus/lib/
fi