#!/bin/bash
# Usage: $bash install_cuda_yum.sh <VERSION>

set -euo pipefail

version=${1:?Usage: bash ci-utils/install_cuda_yum.sh <11|12>}

retry() {
    local attempts="$1"
    shift
    local try

    for try in $(seq 1 "$attempts"); do
        if "$@"; then
            return 0
        fi
        if [ "$try" -lt "$attempts" ]; then
            echo "Command failed (attempt $try/$attempts), retrying: $*"
            sleep 5
        fi
    done

    echo "Command failed after $attempts attempts: $*"
    return 1
}

if [ "$version" -eq 11 ]; then
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
    yum clean all
    echo "Installing CUDA toolkit 11.8"
    retry 3 yum -y install cuda-toolkit-11-8
    export PATH="/usr/local/cuda-11.8/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
elif [ "$version" -eq 12 ]; then
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    dnf clean all
    echo "Installing CUDA toolkit 12.8"
    retry 3 dnf -y install cuda-toolkit-12-8
    export PATH="/usr/local/cuda-12.8/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "Invalid version. Please provide either 11 or 12."
    exit 1
fi

ls -al /usr/local
command -v nvcc
nvcc --version
