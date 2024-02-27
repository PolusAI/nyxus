#!/bin/bash
# Usage: $bash install_cuda_yum.sh <VERSION>

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo &&
yum clean all

version=$1

if [ $version -eq 11 ]; then
    echo "Installing cuda toolkit 11"
    yum -y install cuda-toolkit-11-3-11.3.1-1
elif [ $version -eq 12 ]; then
    echo "Installing cuda toolkit 12"
    yum -y install cuda-toolkit-12-3-12.3.1-1
else
    echo "Invalid version. Please provide either 11 or 12."
    exit 1
fi

ls -al /usr/local
export PATH=$PATH:/usr/local/cuda/bin
nvcc --version