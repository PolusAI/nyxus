#!/bin/bash

export MACOSX_DEPLOYMENT_TARGET=11.0

PYTHON_VERSIONS=("p3.6" "p3.7" "p3.8" "p3.9")

for PYTHON_VERSION in ${PYTHON_VERSIONS[@]}; do
    source /Users/hsidky/miniconda/bin/activate ${PYTHON_VERSION}
    env CMAKE_ARGS="-DPython_ROOT_DIR=/Users/hsidky/miniconda/envs/${PYTHON_VERSION}/" python setup.py bdist_wheel -d dist
done