mkdir local_install
mkdir local_install\include

git clone https://github.com/madler/zlib.git
pushd zlib
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
.\b2 headers
xcopy /E /I /y boost ..\local_install\include\boost
popd

git clone https://github.com/Blosc/c-blosc.git 
pushd c-blosc 
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..   
cmake --build . --config Release --target install  --parallel 4
popd
popd

git clone https://github.com/xtensor-stack/xtl.git 
pushd xtl 
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . --config Release --target install 
popd
popd

git clone https://github.com/xtensor-stack/xtensor.git 
pushd xtensor 
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . --config Release --target install 
popd
popd

git clone https://github.com/xtensor-stack/xsimd.git 
pushd xsimd 
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . --config Release --target install  
popd
popd

git clone  https://github.com/nlohmann/json.git
pushd json
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ -DJSON_BuildTests=OFF ..  
cmake --build . --config Release --target install  --parallel 4
popd
popd

git clone https://github.com/pybind/pybind11.git
pushd pybind11
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/  -DPYBIND11_TEST=OFF ..
cmake --build . --config Release --target install  
popd
popd

git clone https://github.com/constantinpape/z5.git
pushd z5
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/ -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF ..
cmake --build . --config Release --target install  --parallel 4
popd
popd


for /l %%x in (1, 1, 5) do (
    curl https://download.osgeo.org/libtiff/tiff-4.5.0.zip -o libtiff.zip
    if  exist libtiff.zip (
        goto :continue_tiff
    )
)
:continue_tiff
tar -xf libtiff.zip
pushd tiff-4.5.0
mkdir build_man
pushd build_man
for /l %%x in (1, 1, 5) do (
    curl -L https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.14.zip -o libdeflate.zip
    if  exist libdeflate.zip (
        goto :continue_libdeflate
    )
)
:continue_libdeflate
tar -xf libdeflate.zip
pushd libdeflate-1.14
nmake /f Makefile.msc
popd
cmake -DDeflate_INCLUDE_DIR=./libdeflate-1.14 -DDeflate_LIBRARY_RELEASE=./libdeflate-1.14/libdeflate.lib -DCMAKE_INSTALL_PREFIX=../../local_install/ ..
cmake --build . --config Release --target install
copy libdeflate-1.14\libdeflate.dll ..\..\local_install\bin\
copy libdeflate-1.14\*.lib ..\..\local_install\lib\
popd
popd

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

copy local_install\lib\tiff.lib local_install\lib\libtiff_o.lib
copy local_install\lib\zlib.lib local_install\lib\zlib_o.lib
copy local_install\lib\libpng16.lib local_install\lib\libpng_o.lib
copy local_install\lib\openjp2.lib local_install\lib\openjp2_o.lib

curl -L https://github.com/DCMTK/dcmtk/archive/refs/tags/DCMTK-3.6.7.zip -o DCMTK-3.6.7.zip
tar -xvf DCMTK-3.6.7.zip
pushd dcmtk-DCMTK-3.6.7
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ -DCMAKE_PREFIX_PATH=../../local_install/ -DBUILD_SHARED_LIBS=ON -DDCMTK_WITH_ICONV=OFF -DDCMTK_WITH_TIFF=ON -DWITH_LIBTIFFINC=../../local_install -DDCMTK_WITH_PNG=ON -DWITH_LIBPNGINC=../../local_install -DDCMTK_WITH_ZLIB=ON -DWITH_ZLIBINC=../../local_install -DDCMTK_WITH_OPENJPEG=ON -DWITH_OPENJPEGINC=../../local_install  -DBUILD_APPS=OFF ..
cmake --build . --config Release --target install --parallel 4
popd
popd

curl -L https://github.com/apache/arrow/archive/refs/tags/apache-arrow-12.0.0.zip -o  arrow-apache-arrow-12.0.0.zip
tar -xvf arrow-apache-arrow-12.0.0.zip
pushd arrow-apache-arrow-12.0.0
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ -DCMAKE_PREFIX_PATH=../../local_install/
cmake --build . --config Release --target install --parallel 4
popd
popd

SET ROOTTDIR="%cd%"
curl -L https://github.com/DraconPern/fmjpeg2koj/archive/refs/tags/v1.0.3.zip -o fmjpeg2koj.zip
tar -xvf fmjpeg2koj.zip
pushd fmjpeg2koj-1.0.3
mkdir build_man
pushd build_man
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS_RELEASE="/MT /O2 /D NDEBUG" -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/  -DFMJPEG2K=%ROOTTDIR%\local_install\  ..
cmake --build . --config Release --target install --parallel 4
popd
popd


if errorlevel 1 exit 1

if "%ON_GITHUB%"=="TRUE" xcopy /E /I /y local_install\bin %TEMP%\nyxus\bin