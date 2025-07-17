#!/usr/bin/env bash
set -e

#----------------------------------------------------------------------
#
# Usage: $bash build_conda.sh $NYXUS_SRC_ROOT
#
if [ -z "$1" ]
then
      echo "No path to the Nyxus source location provided"
      echo "Usage: \$bash build_conda.sh \$NYXUS_SRC_ROOT"
      echo "       where \$NYXUS_SRC_ROOT points to location of Nyxus source"
      exit
fi

MINICONDA=$PWD/miniconda-for-nyxus # Modify this to your preferred location for persistence
CPP_BUILD_DIR=$PWD
SRC_ROOT=$1 #source dir location
NYXUS_ROOT=$SRC_ROOT
PYTHON=3.11

git config --global --add safe.directory $NYXUS_ROOT

#----------------------------------------------------------------------
# Run these only once

function setup_miniconda() {
  MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  curl -L $MINICONDA_URL -o miniconda.sh
  echo "installing miniconda"
  bash miniconda.sh -b -p $MINICONDA
  rm -f miniconda.sh
  LOCAL_PATH=$PATH
  export PATH="$MINICONDA/bin:$PATH"

  echo "updating miniconda"
  conda update -y -q conda
  conda config --set auto_update_conda false
  conda info -a

  conda config --set show_channel_urls True
  conda config --add channels https://repo.continuum.io/pkgs/free
  conda config --add channels conda-forge

  echo "install dependencies"
  conda create -y -n nyxus-$PYTHON -c conda-forge  \
        python=$PYTHON \
        --file ${SRC_ROOT}/ci-utils/envs/conda_cpp.txt \
        --file ${SRC_ROOT}/ci-utils/envs/conda_linux_compiler.txt \
        --file ${SRC_ROOT}/ci-utils/envs/conda_py.txt \
        --file ${SRC_ROOT}/ci-utils/envs/conda_linux_gpu.txt 

  export PATH=$LOCAL_PATH
}

setup_miniconda

#----------------------------------------------------------------------
# Activate conda in bash and activate conda environment

. $MINICONDA/etc/profile.d/conda.sh
conda activate nyxus-$PYTHON
export NYXUS_HOME=$CONDA_PREFIX
#Build CLI
mkdir -p $CPP_BUILD_DIR
pushd $CPP_BUILD_DIR
cmake -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      -DBUILD_CLI=ON \
      -DUSEGPU=ON \
      -DALLEXTRAS=ON \
      $NYXUS_ROOT

cmake --build . --parallel 4
