#!/bin/bash
# Usage: $bash install_prereq_linux.sh --min_build yes --install_dir <LOCATION>
# Defaults:
#   $install_dir = ./local_install
#   $min_build == no
#
# $min_build = yes will only install pybind11, libtiff and libdeflate
#

BUILD_Z5_DEP=1
BULD_DCMTK_DEP=1
BUILD_ARROW_DEP=1

MIN_BUILD=0
FULL_BUILD=1

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

if [[ "${min_build,,}" == "yes" ]]; then
    BUILD_Z5_DEP=0
    BULD_DCMTK_DEP=0
    BUILD_ARROW_DEP=0
fi

if [[ -z $install_dir ]]
then
      echo "No path to the Nyxus source location provided"
      echo "Creating local_install directory"
      LOCAL_INSTALL_DIR="local_install"
else
    LOCAL_INSTALL_DIR=$install_dir
fi

mkdir -p $LOCAL_INSTALL_DIR
mkdir -p $LOCAL_INSTALL_DIR/include

if [[ $BUILD_Z5_DEP -eq 1 ]] || [[ $BULD_DCMTK_DEP -eq 1 ]]; then
    git clone https://github.com/madler/zlib.git
    cd zlib
    mkdir build_man
    cd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../
fi
if [[ $BUILD_Z5_DEP -eq 1 ]]; then
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
    cp -r boost ../$LOCAL_INSTALL_DIR/include
    cd ../

    git clone https://github.com/Blosc/c-blosc.git 
    cd c-blosc 
    mkdir build_man
    cd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    git clone https://github.com/xtensor-stack/xtl.git 
    cd xtl 
    mkdir build_man
    cd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    git clone https://github.com/xtensor-stack/xtensor.git 
    cd xtensor 
    mkdir build_man
    cd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    git clone https://github.com/xtensor-stack/xsimd.git 
    cd xsimd 
    mkdir build_man
    cd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    git clone  https://github.com/nlohmann/json.git
    cd json
    mkdir build_man
    cd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/ ..  
    make install/fast
    cd ../../

    git clone https://github.com/constantinpape/z5.git
    cd z5
    mkdir build_man
    cd build_man/
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR/ -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF  ..
    make install -j4
    cd ../../
fi


git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/  -DPYBIND11_TEST=OFF ..
make install -j4
cd ../../

if [[ $BULD_DCMTK_DEP -eq 1 ]]; then
    JPEG_INSTALL_PATH=$PWD
    curl -L http://www.ijg.org/files/jpegsrc.v9e.tar.gz -o jpegsrc.v9e.tar.gz
    tar -xzf jpegsrc.v9e.tar.gz
    cd jpeg-9e
    ./configure --prefix=
    make DESTDIR=$JPEG_INSTALL_PATH/$LOCAL_INSTALL_DIR install
    ./libtool --finish $JPEG_INSTALL_PATH/$LOCAL_INSTALL_DIR/lib
    cd ..
fi

for i in {1..5}
do
    curl -L https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.14.zip -o libdeflate.zip
    if [ -f "libdeflate.zip" ] ; then
        break
    fi
done

unzip libdeflate.zip
cd libdeflate-1.14
PREFIX= LIBDIR=/lib64  DESTDIR=../$LOCAL_INSTALL_DIR/ make  install
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
cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR/   ..
make install -j4
cd ../../

if [[ $BULD_DCMTK_DEP -eq 1 ]]; then
    curl -L  https://github.com/glennrp/libpng/archive/refs/tags/v1.6.39.zip -o v1.6.39.zip
    unzip v1.6.39.zip
    cd libpng-1.6.39/
    mkdir build_man
    cd build_man/
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR/   ..
    make install -j4
    cd ../../

    curl -L https://github.com/uclouvain/openjpeg/archive/refs/tags/v2.5.0.zip -o v2.5.0.zip
    unzip v2.5.0.zip
    cd openjpeg-2.5.0/
    mkdir build_man
    cd build_man/
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR/ -DBUILD_CODEC=OFF   ..
    make install -j4
    cd ../../

    curl -L https://github.com/DCMTK/dcmtk/archive/refs/tags/DCMTK-3.6.7.zip -o DCMTK-3.6.7.zip
    unzip DCMTK-3.6.7.zip
    cd dcmtk-DCMTK-3.6.7/
    mkdir build_man
    cd build_man/
    cmake -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR/  -DDCMTK_WITH_ICONV=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_APPS=OFF  ..
    make install -j4
    cd ../../

    ROOTDIR=$(pwd)
    curl -L https://github.com/sameeul/fmjpeg2koj/archive/refs/heads/fix_cmake.zip -o fmjpeg2koj.zip
    unzip fmjpeg2koj.zip
    cd fmjpeg2koj-fix_cmake/
    mkdir build_man
    cd build_man/
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../$LOCAL_INSTALL_DIR/   -DCMAKE_PREFIX_PATH=../../$LOCAL_INSTALL_DIR/  -DFMJPEG2K=$ROOTDIR/$LOCAL_INSTALL_DIR/  ..
    make install -j4
    cd ../../
fi

