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
make install -j4
cd ../../

git clone https://github.com/constantinpape/z5.git
cd z5
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/ -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF  ..
make install -j4
cd ../../

JPEG_INSTALL_PATH=$PWD
curl -L http://www.ijg.org/files/jpegsrc.v9e.tar.gz -o jpegsrc.v9e.tar.gz
tar -xzf jpegsrc.v9e.tar.gz
cd jpeg-9e
./configure --prefix=
make DESTDIR=$JPEG_INSTALL_PATH/$Z5_INSTALL_DIR install
./libtool --finish $JPEG_INSTALL_PATH/$Z5_INSTALL_DIR/lib
cd ..

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
make install -j4
cd ../../

curl -L  https://github.com/glennrp/libpng/archive/refs/tags/v1.6.39.zip -o v1.6.39.zip
unzip v1.6.39.zip
cd libpng-1.6.39/
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/   ..
make install -j4
cd ../../

curl -L https://github.com/uclouvain/openjpeg/archive/refs/tags/v2.5.0.zip -o v2.5.0.zip
unzip v2.5.0.zip
cd openjpeg-2.5.0/
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/ -DBUILD_CODEC=OFF   ..
make install -j4
cd ../../

curl -L https://github.com/DCMTK/dcmtk/archive/refs/tags/DCMTK-3.6.7.zip -o DCMTK-3.6.7.zip
unzip DCMTK-3.6.7.zip
cd dcmtk-DCMTK-3.6.7/
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/  -DDCMTK_WITH_ICONV=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_APPS=OFF  ..
make install -j4
cd ../../

ROOTDIR=$(pwd)
curl -L https://github.com/sameeul/fmjpeg2koj/archive/refs/heads/fix_cmake.zip -o fmjpeg2koj.zip
unzip fmjpeg2koj.zip
cd fmjpeg2koj-fix_cmake/
mkdir build_man
cd build_man/
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../$Z5_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$Z5_INSTALL_DIR/  -DFMJPEG2K=$ROOTDIR/$Z5_INSTALL_DIR/  ..
make install -j4
cd ../../

curl -L https://github.com/apache/arrow/archive/refs/tags/apache-arrow-12.0.0.zip -o  arrow-apache-arrow-12.0.0.zip
unzip arrow-apache-arrow-12.0.0.zip
cd arrow-apache-arrow-12.0.0
mkdir build
cd build/
cmake ..
make install -j4
cd ../../ 