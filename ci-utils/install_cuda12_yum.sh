yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo &&
yum clean all
yum -y install cuda-toolkit-12-3
ls -al /usr/local
export PATH=$PATH:/usr/local/cuda/bin
nvcc --version