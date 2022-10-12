mkdir local_install
mkdir local_install\include

git clone https://github.com/madler/zlib.git
pushd zlib
mkdir build_man
pushd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . --config Release --target install  
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
cmake --build . --config Release --target install 
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
cmake --build . --config Release --target install 
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
cmake --build . --config Release --target install  
popd
popd


for /l %%x in (1, 1, 5) do (
    curl https://download.osgeo.org/libtiff/tiff-4.3.0.zip -o libtiff.zip
    if  exist libtiff.zip (
        goto :continue_tiff
    )
)
:continue_tiff
tar -xf libtiff.zip
pushd tiff-4.3.0
mkdir build_man
pushd build_man
curl -L https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.14.zip -o libdeflate.zip
tar -xf libdeflate.zip
pushd libdeflate-1.14
nmake /f Makefile.msc
popd
cmake -DDeflate_INCLUDE_DIR=./libdeflate -DDeflate_LIBRARY_RELEASE=./libdeflate/libdeflate.lib -DCMAKE_INSTALL_PREFIX=../../local_install/ ..
cmake --build . --config Release --target install
copy libdeflate\libdeflate.dll ..\..\local_install\bin\
copy libdeflate\*.lib ..\..\local_install\lib\
popd
popd

if "%ON_GITHUB%"=="TRUE" xcopy /E /I /y local_install\bin %TEMP%\nyxus\bin