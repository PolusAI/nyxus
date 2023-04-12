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

for i in {1..5}
do
    curl -L https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.14.zip -o libdeflate.zip
    if [ -f "libdeflate.zip" ] ; then
        break
    fi
done

unzip libdeflate.zip
cd libdeflate-1.14
PREFIX= LIBDIR=/lib64  DESTDIR=../$Z5_INSTALL_DIR/ make  install
cd ../
echo AFTER LIBDEFLATE
ls $Z5_INSTALL_DIR/lib64
ls $Z5_INSTALL_DIR/lib

for i in {1..5}
do
    curl https://download.osgeo.org/libtiff/tiff-4.5.0.zip -o libtiff.zip 
    if [ -f "libtiff.zip" ] ; then
        break
    fi
done

unzip libtiff.zip
cd tiff-4.5.0
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/   ..
make install
cd ../../

curl -L  https://github.com/glennrp/libpng/archive/refs/tags/v1.6.39.zip -o v1.6.39.zip
unzip v1.6.39.zip
cd libpng-1.6.39/
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/   ..
make install
cd ../../

curl -L https://github.com/uclouvain/openjpeg/archive/refs/tags/v2.5.0.zip -o v2.5.0.zip
unzip v2.5.0.zip
cd openjpeg-2.5.0/
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/   ..
make install
cd ../../

curl -L https://github.com/DCMTK/dcmtk/archive/refs/tags/DCMTK-3.6.7.zip -o DCMTK-3.6.7.zip
unzip DCMTK-3.6.7.zip
cd dcmtk-DCMTK-3.6.7/
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/  -DDCMTK_WITH_ICONV=OFF -DBUILD_SHARED_LIBS=ON  ..
make install
cd ../../


if [ "$ON_GITHUB" == "TRUE" ]; then
    echo HERE
    echo $PWD
    mkdir -p /tmp/nyxus/
    ls local_install/lib/
    ls local_install/lib64/
    cp -r local_install/lib64/ /tmp/nyxus/lib64/
    cp -r local_install/lib/ /tmp/nyxus/lib/
    ls /tmp/nyxus/lib/
    ls /tmp/nyxus/lib64/
fi
