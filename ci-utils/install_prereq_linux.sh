#!/bin/bash
# Usage: $bash install_prereq_linux.sh --min_build yes --build_arrow yes --install_dir <LOCATION>
# Defaults:
#   $install_dir = ./local_install
#   $min_build == no
#   $build_arrow == no
#
# $min_build = yes will only install pybind11, libtiff and libdeflate
#

BUILD_Z5_DEP=1
BULD_DCMTK_DEP=1
BUILD_ARROW_DEP=0
BUILD_BOOST_DEP=1

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
    BUILD_BOOST_DEP=0
fi

if [[ "${build_arrow}" == "yes" ]]; then
    BUILD_ARROW_DEP=1
    BUILD_BOOST_DEP=1
fi


if [[ -z $install_dir ]]
then
      echo "No path to the Nyxus source location provided"
      echo "Creating local_install directory"
      LOCAL_INSTALL_DIR="local_install"
else
    LOCAL_INSTALL_DIR=$install_dir
fi

mkdir -p "$LOCAL_INSTALL_DIR"
mkdir -p "$LOCAL_INSTALL_DIR"/include

curl -L https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip -o v2.12.0.zip
unzip v2.12.0.zip
cd pybind11-2.12.0
mkdir build_man
cd build_man
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/  -DPYBIND11_TEST=OFF ..
make install -j4
cd ../../

curl -L https://github.com/madler/zlib/releases/download/v1.3/zlib13.zip -o zlib13.zip
unzip zlib13.zip
cd zlib-1.3
mkdir build_man
cd build_man
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

if [[ $BUILD_BOOST_DEP -eq 1 ]]; then
    for i in {1..5}
    do
        curl -L https://archives.boost.io/release/1.79.0/source/boost_1_79_0.tar.bz2 -o boost_1_79_0.tar.bz2 
        if [ -f "boost_1_79_0.tar.bz2" ] ; then
            break
        fi
    done
    tar --bzip2 -xf boost_1_79_0.tar.bz2 
    cd boost_1_79_0 
    ./bootstrap.sh 
    ./b2 headers
    cp -r boost ../"$LOCAL_INSTALL_DIR"/include
    cd ../
fi

if [[ $BUILD_Z5_DEP -eq 1 ]]; then
    curl -L https://github.com/Blosc/c-blosc/archive/refs/tags/v1.21.5.zip -o v1.21.5.zip
    unzip v1.21.5.zip
    cd c-blosc-1.21.5
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    curl -L https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.5.zip -o 0.7.5.zip
    unzip 0.7.5.zip
    cd xtl-0.7.5 
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    curl -L https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.7.zip -o 0.24.7.zip
    unzip 0.24.7.zip
    cd xtensor-0.24.7
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    curl -L https://github.com/xtensor-stack/xsimd/archive/refs/tags/11.1.0.zip -o 11.1.0.zip
    unzip 11.1.0.zip 
    cd xsimd-11.1.0 
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/ ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    curl -L https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.zip -o v3.11.2.zip
    unzip v3.11.2.zip 
    cd json-3.11.2
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/ ..  
    make install/fast
    cd ../../

    curl -L https://github.com/constantinpape/z5/archive/refs/heads/master.zip -o z5.zip
    unzip z5.zip 
    cd z5-master
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/ -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF  ..
    make install -j4
    cd ../../
fi


curl -L https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.0.zip -o 3.1.0.zip
unzip 3.1.0.zip
cd libjpeg-turbo-3.1.0
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/   ..
make install -j4
cd ../../

if [[ $BULD_DCMTK_DEP -eq 1 ]]; then
    curl -L  https://github.com/glennrp/libpng/archive/refs/tags/v1.6.39.zip -o v1.6.39.zip
    unzip v1.6.39.zip
    cd libpng-1.6.39
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/   ..
    make install -j4
    cd ../../

    curl -L https://github.com/uclouvain/openjpeg/archive/refs/tags/v2.5.0.zip -o v2.5.0.zip
    unzip v2.5.0.zip
    cd openjpeg-2.5.0
    mkdir build_man
    cd build_man
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/ -DBUILD_CODEC=OFF   ..
    make install -j4
    cd ../../
fi

curl -L https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.19.zip -o v1.19.zip
unzip v1.19.zip
cd libdeflate-1.19
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/ ..
make install -j4
cd ../../


for i in {1..5}
do
    curl -L https://download.osgeo.org/libtiff/tiff-4.7.0.zip -o tiff-4.7.0.zip 
    if [ -f "tiff-4.7.0.zip" ] ; then
        break
    fi
done

unzip tiff-4.7.0.zip
cd tiff-4.7.0
mkdir build_man
cd build_man
if [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/ -DJPEG_LIBRARY_RELEASE=../../"$LOCAL_INSTALL_DIR"/lib/libjpeg.dylib -DJPEG_INCLUDE_DIRS=../../"$LOCAL_INSTALL_DIR"/include ..
else
    cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/ ..
fi
make install -j4
cd ../../

if [[ $BULD_DCMTK_DEP -eq 1 ]]; then
    curl -L https://github.com/DCMTK/dcmtk/archive/refs/tags/DCMTK-3.6.7.zip -o DCMTK-3.6.7.zip
    unzip DCMTK-3.6.7.zip
    cd dcmtk-DCMTK-3.6.7/CMake
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/TIFF_LIBRARY/TIFF_LIBRARIES/' 3rdparty.cmake
        sed -i '' 's/JPEG_LIBRARY/JPEG_LIBRARIES/' 3rdparty.cmake
    else
        sed -i 's/TIFF_LIBRARY/TIFF_LIBRARIES/' 3rdparty.cmake
        sed -i 's/JPEG_LIBRARY/JPEG_LIBRARIES/' 3rdparty.cmake
    fi
    cd ..
    mkdir build_man
    cd build_man
    if [[ "$OSTYPE" == "darwin"* ]]; then
        cmake -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/  -DDCMTK_WITH_ICONV=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_APPS=OFF -DJPEG_LIBRARY_RELEASE=../../"$LOCAL_INSTALL_DIR"/lib/libjpeg.dylib -DJPEG_INCLUDE_DIRS=../../"$LOCAL_INSTALL_DIR"/include ..
    else
        cmake -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/  -DDCMTK_WITH_ICONV=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_APPS=OFF  ..
    fi
    make install -j4
    cd ../../

    ROOTDIR=$(pwd)
    curl -L https://github.com/sameeul/fmjpeg2koj/archive/refs/heads/fix_cmake.zip -o fmjpeg2koj.zip
    unzip fmjpeg2koj.zip
    cd fmjpeg2koj-fix_cmake
    mkdir build_man
    cd build_man
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../"$LOCAL_INSTALL_DIR"/   -DCMAKE_PREFIX_PATH=../../"$LOCAL_INSTALL_DIR"/  -DFMJPEG2K="$ROOTDIR"/"$LOCAL_INSTALL_DIR"/  ..
    make install -j4
    cd ../../
fi

if [[ $BUILD_ARROW_DEP -eq 1 ]]; then

    curl -L https://github.com/apache/arrow/archive/refs/tags/apache-arrow-19.0.0.zip -o  arrow-apache-arrow-19.0.0.zip
    unzip arrow-apache-arrow-19.0.0.zip
    cd arrow-apache-arrow-19.0.0
    cd cpp
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=../../../"$LOCAL_INSTALL_DIR"/ \
            -DCMAKE_PREFIX_PATH=../../../"$LOCAL_INSTALL_DIR"/ \
            -DARROW_OPTIONAL_INSTALL=ON \
            -DARROW_COMPUTE=ON \
            -DARROW_CSV=ON \
            -DARROW_DATASET=ON \
            -DARROW_ACERO=ON \
            -DARROW_PARQUET=ON \
            -DARROW_WITH_SNAPPY=ON \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            .. 
    make -j4
    make install

    cd ../../../
fi
