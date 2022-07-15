#!/bin/bash
mkdir -p $Z5_INSTALL_DIR
mkdir -p $Z5_INSTALL_DIR/include
export CMAKE_INSTALL_PREFIX=$Z5_INSTALL_DIR

if [ "$Z5_INSTALLED" == "FALSE" ]; then
    if [ "$Boost_INSTALLED" == "FALSE" ]; then
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
    fi

    if [ "$BLOSC_INSTALLED" == "FALSE" ]; then
        git clone https://github.com/Blosc/c-blosc.git 
        cd c-blosc 
        mkdir build_man
        cd build_man
        cmake ..  
        cmake --build . 
        cmake --build . --target install 
        cd ../../
    fi

    git clone https://github.com/xtensor-stack/xtl.git 
    cd xtl 
    mkdir build_man
    cd build_man
    cmake ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    git clone https://github.com/xtensor-stack/xtensor.git 
    cd xtensor 
    mkdir build_man
    cd build_man
    cmake ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    git clone https://github.com/xtensor-stack/xsimd.git 
    cd xsimd 
    mkdir build_man
    cd build_man
    cmake ..  
    cmake --build . 
    cmake --build . --target install 
    cd ../../

    git clone  https://github.com/nlohmann/json.git
    cd json
    mkdir build_man
    cd build_man
    cmake ..  
    make install/fast
    cd ../../


    git clone https://github.com/constantinpape/z5.git
    cd z5
    mkdir build_man
    cd build_man/
    cmake -DWITH_BLOSC=ON -DBUILD_Z5PY=OFF ..
    make install
    cd ../../

fi

if [ "$ON_GITHUB" == "TRUE" ]; then
    mkdir -p /tmp/nyxus/
    cp -r local_install/lib64 /tmp/nyxus/lib/
    cp -r local_install/lib /tmp/nyxus/lib/
fi
