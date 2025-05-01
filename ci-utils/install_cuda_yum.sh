#!/bin/bash
# Usage: $bash install_cuda_yum.sh <VERSION>

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-rhel8.repo &&
dnf clean all &&
dnf -y module install nvidia-driver:latest-dkms

version=$1

if [ $version -eq 11 ]; then
    echo "Installing cuda toolkit 11"
    dnf -y install cuda-11-8
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
elif [ $version -eq 12 ]; then
    echo "Installing cuda toolkit 12"
    dnf -y install cuda-12-3
    echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
else
    echo "Invalid version. Please provide either 11 or 12."
    exit 1
fi

ls -al /usr/local
export PATH=$PATH:/usr/local/cuda/bin
nvcc --version