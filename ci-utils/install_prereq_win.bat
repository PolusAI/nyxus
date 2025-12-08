@echo OFF
setlocal
set BUILD_Z5_DEP=1
set BUILD_DCMTK_DEP=1
set BUILD_ARROW=1
SET ROOTDIR="%cd%"

setlocal enabledelayedexpansion
:GETOPTS
    if /I "%~1" == "--min_build" set MIN_BUILD=%2& shift
    shift

if not (%1)==() goto GETOPTS
if /I "%min_build%" == "yes" (
    set BUILD_Z5_DEP=0
    set BUILD_DCMTK_DEP=0
)
SETLOCAL DisableDelayedExpansion

mkdir local_install
mkdir local_install\include

curl -L https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip -o v2.12.0.zip
tar -xvf v2.12.0.zip
pushd pybind11-2.12.0
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/  -DPYBIND11_TEST=OFF ..
cmake --build . --config Release --target install  
popd
popd

curl -L https://github.com/madler/zlib/releases/download/v1.3.1/zlib131.zip -o zlib131.zip
tar -xvf zlib131.zip
pushd zlib-1.3.1
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . --config Release --target install --parallel 4  
popd
popd

if "%BUILD_Z5_DEP%" == "1" (
    for /l %%x in (1, 1, 5) do (
        curl -L https://archives.boost.io/release/1.79.0/source/boost_1_88_0.zip -o boost_1_88_0.zip
        if  exist boost_1_88_0.zip (
            goto :continue_boost
        )
    )
    :continue_boost
    tar  -xf boost_1_88_0.zip
    pushd boost_1_88_0 
    call bootstrap.bat 
    .\b2 headers --prefix=../local_install
    .\b2 -j %NUMBER_OF_PROCESSORS% variant=release link=shared runtime-link=shared threading=multi install --prefix=../local_install
    xcopy /E /I /y boost ..\local_install\include\boost
    popd

    curl -L https://github.com/Blosc/c-blosc/archive/refs/tags/v1.21.6.zip -o v1.21.6.zip
    tar -xf v1.21.6.zip
    pushd c-blosc-1.21.6
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..   
    cmake --build . --config Release --target install  --parallel 4
    popd
    popd

    curl -L https://github.com/xtensor-stack/xtl/archive/refs/tags/0.8.0.zip -o 0.8.0.zip
    tar -xf 0.8.0.zip
    pushd xtl-0.8.0 
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
    cmake --build . --config Release --target install 
    popd
    popd

    curl -L https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.26.0.zip -o 0.26.0.zip
    tar -xf 0.26.0.zip
    pushd xtensor-0.26.0
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
    cmake --build . --config Release --target install 
    popd
    popd

    curl -L https://github.com/xtensor-stack/xsimd/archive/refs/tags/13.2.0.zip -o 13.2.0.zip
    tar -xf 13.2.0.zip 
    pushd xsimd-13.2.0 
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
    cmake --build . --config Release --target install  
    popd
    popd

    curl -L https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.zip -o v3.11.2.zip
    tar -xf v3.11.2.zip 
    pushd json-3.11.2
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ -DJSON_BuildTests=OFF ..  
    cmake --build . --config Release --target install  --parallel 4
    popd
    popd

    curl -L https://github.com/constantinpape/z5/archive/refs/tags/2.0.20.zip -o 2.0.20.zip
    tar -xf 2.0.20.zip 
    pushd z5-2.0.20
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/ -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF ..
    cmake --build . --config Release --target install  --parallel 4
    popd
    popd
)


set _ROOTDIR=%ROOTDIR:\=/%
if "%BUILD_ARROW%" == "1" (

    curl -L https://github.com/apache/arrow/archive/refs/tags/apache-arrow-20.0.0.zip -o  arrow-apache-arrow-20.0.0.zip
    unzip arrow-apache-arrow-20.0.0.zip
    pushd arrow-apache-arrow-20.0.0
    pushd cpp
    mkdir build
    pushd build
    cmake .. -A x64 -DCMAKE_INSTALL_PREFIX=../../../local_install/ -DCMAKE_PREFIX_PATH=../../../local_install/ -DARROW_PARQUET=ON -DARROW_WITH_SNAPPY=ON -DARROW_BUILD_TESTS=OFF -DBOOST_ROOT=%_ROOTDIR%/boost_1_88_0
    cmake --build . --config Release --target install --parallel 4
    popd 
    popd
    popd
)



if "%BUILD_DCMTK_DEP%" == "1" (
    curl -L https://github.com/glennrp/libpng/archive/refs/tags/v1.6.39.zip -o v1.6.39.zip
    tar -xvf v1.6.39.zip
    pushd libpng-1.6.39
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/   ..
    cmake --build . --config Release --target install --parallel 4
    popd
    popd

    curl -L https://github.com/uclouvain/openjpeg/archive/refs/tags/v2.5.0.zip -o v2.5.0.zip
    tar -xvf v2.5.0.zip
    pushd openjpeg-2.5.0
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/  -DBUILD_CODEC=OFF  ..
    cmake --build . --config Release --target install --parallel 4
    popd
    popd
)

curl -L https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.19.zip -o v1.19.zip
tar -xf v1.19.zip
pushd libdeflate-1.19
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/  ..
cmake --build . --config Release --target install --parallel 4
popd
popd

curl -L https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.0.zip -o 3.1.0.zip
tar -xf 3.1.0.zip
pushd libjpeg-turbo-3.1.0
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/  ..
cmake --build . --config Release --target install --parallel 4
popd
popd

for /l %%x in (1, 1, 5) do (
    curl -L https://download.osgeo.org/libtiff/tiff-4.7.0.zip -o tiff-4.7.0.zip
    if  exist tiff-4.7.0.zip (
        goto :continue_tiff
    )
)
:continue_tiff
tar -xf tiff-4.7.0.zip
pushd tiff-4.7.0
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/  ..
cmake --build . --config Release --target install
popd
popd

if "%BUILD_DCMTK_DEP%" == "1" (
    copy local_install\lib\zlib.lib local_install\lib\zlib_o.lib
    copy local_install\lib\libpng16.lib local_install\lib\libpng_o.lib
    copy local_install\lib\openjp2.lib local_install\lib\openjp2_o.lib
    copy local_install\lib\tiff.lib local_install\lib\libtiff_o.lib

    curl -L https://github.com/DCMTK/dcmtk/archive/refs/tags/DCMTK-3.6.9.zip -o DCMTK-3.6.9.zip
    tar -xvf DCMTK-3.6.9.zip
    pushd dcmtk-DCMTK-3.6.9
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ -DCMAKE_PREFIX_PATH=../../local_install/ -DBUILD_SHARED_LIBS=ON -DDCMTK_WITH_ICONV=OFF -DDCMTK_WITH_TIFF=ON -DWITH_LIBTIFFINC=../../local_install -DDCMTK_WITH_PNG=ON -DWITH_LIBPNGINC=../../local_install -DDCMTK_WITH_ZLIB=ON -DWITH_ZLIBINC=../../local_install -DDCMTK_WITH_OPENJPEG=ON -DWITH_OPENJPEGINC=../../local_install  -DBUILD_APPS=OFF ..
    cmake --build . --config Release --target install --parallel 4
    popd
    popd

    curl -L https://github.com/sameeul/fmjpeg2koj/archive/refs/heads/fix_cmake.zip -o fmjpeg2koj.zip
    tar -xvf fmjpeg2koj.zip
    pushd fmjpeg2koj-fix_cmake
    mkdir build_man
    pushd build_man
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS_RELEASE="/MT /O2 /D NDEBUG /Zc:__cplusplus" -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/  -DFMJPEG2K=%ROOTDIR%\local_install\  ..
    cmake --build . --config Release --target install --parallel 4
    popd
    popd
)

if errorlevel 1 exit 1
