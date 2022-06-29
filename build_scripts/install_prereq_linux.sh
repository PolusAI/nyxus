#!/bin/bash
mkdir local_install
mkdir local_install/include
curl -L https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.bz2 -o boost_1_79_0.tar.bz2
tar --bzip2 -xf boost_1_79_0.tar.bz2 
cd boost_1_79_0 
./bootstrap.sh 
./b2 headers
cp -r boost ../local_install/include
cd ../

git clone https://github.com/Blosc/c-blosc.git 
cd c-blosc 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone https://github.com/xtensor-stack/xtl.git 
cd xtl 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone https://github.com/xtensor-stack/xtensor.git 
cd xtensor 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone https://github.com/xtensor-stack/xsimd.git 
cd xsimd 
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
cmake --build . 
cmake --build . --target install 
cd ../../

git clone  https://github.com/nlohmann/json.git
cd json
mkdir build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/ ..  
make install/fast
cd ../../


git clone https://github.com/constantinpape/z5.git
cd z5
mkdir build_man
cd build_man/
cmake -DCMAKE_INSTALL_PREFIX=../../local_install/   -DCMAKE_PREFIX_PATH=../../local_install/ -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF ..
make install
cd ../../

if [ "$ON_GITHUB"=="TRUE" ]; then
    mkdir /tmp/nyxus/
    cp -r local_install/lib64 /tmp/nyxus/lib/
    cp -r local_install/lib /tmp/nyxus/lib/
fi
