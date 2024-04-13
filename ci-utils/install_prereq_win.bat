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

curl -L https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip -o v2.11.1.zip
tar -xvf v2.11.1.zip
pushd pybind11-2.11.1
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/  -DPYBIND11_TEST=OFF ..
cmake --build . --config Release --target install  
popd
popd

curl -L https://github.com/madler/zlib/releases/download/v1.3/zlib13.zip -o zlib13.zip
tar -xvf zlib13.zip
pushd zlib-1.3
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . --config Release --target install --parallel 4  
popd
popd

for /l %%x in (1, 1, 5) do (
    curl -L https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.zip -o boost_1_79_0.zip
    if  exist boost_1_79_0.zip (
        goto :continue_boost
    )
)
:continue_boost
tar  -xf boost_1_79_0.zip
pushd boost_1_79_0 
call bootstrap.bat 
.\b2 headers --prefix=../local_install
.\b2 install --prefix=../local_install
xcopy /E /I /y boost ..\local_install\include\boost
popd





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

for /l %%x in (1, 1, 5) do (
    curl -L https://download.osgeo.org/libtiff/tiff-4.6.0.zip -o tiff-4.6.0.zip
    if  exist tiff-4.6.0.zip (
        goto :continue_tiff
    )
)
:continue_tiff
tar -xf tiff-4.6.0.zip
pushd tiff-4.6.0
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

    curl -L https://github.com/DCMTK/dcmtk/archive/refs/tags/DCMTK-3.6.7.zip -o DCMTK-3.6.7.zip
    tar -xvf DCMTK-3.6.7.zip
    pushd dcmtk-DCMTK-3.6.7
    mkdir build_man
    pushd build_man
    cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ -DCMAKE_PREFIX_PATH=../../local_install/ -DBUILD_SHARED_LIBS=ON -DDCMTK_WITH_ICONV=OFF -DDCMTK_WITH_TIFF=ON -DWITH_LIBTIFFINC=../../local_install -DDCMTK_WITH_PNG=ON -DWITH_LIBPNGINC=../../local_install -DDCMTK_WITH_ZLIB=ON -DWITH_ZLIBINC=../../local_install -DDCMTK_WITH_OPENJPEG=ON -DWITH_OPENJPEGINC=../../local_install  -DBUILD_APPS=OFF ..
    cmake --build . --config Release --target install --parallel 4
    popd
    popd

    set ROOTDIR=%cd%
    echo %ROOTDIR%
    curl -L https://github.com/sameeul/fmjpeg2koj/archive/refs/heads/test_ci.zip -o fmjpeg2koj.zip
    tar -xvf fmjpeg2koj.zip
    pushd fmjpeg2koj-test_ci
    mkdir build_man
    pushd build_man
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS_RELEASE="/MT /O2 /D NDEBUG" -DCMAKE_INSTALL_PREFIX=%ROOTDIR%/local_install/   -DCMAKE_PREFIX_PATH=%ROOTDIR%/local_install/  -DFMJPEG2K=%ROOTDIR%/local_install/  ..
    cmake --build . --config Release --target install --parallel 4
    popd
    popd
)

if errorlevel 1 exit 1

if "%ON_GITHUB%"=="TRUE" xcopy /E /I /y local_install\bin %TEMP%\nyxus\bin
