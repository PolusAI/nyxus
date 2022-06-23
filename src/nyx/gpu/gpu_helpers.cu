#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

void try_init_gpu()
{
	int device = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	std::cout << "GPU device info:\n";
	std::cout << "  name:                        " << deviceProp.name << "\n";
	std::cout << "  capability:                  " << deviceProp.major << "." << deviceProp.minor << "\n";
	std::cout << "  multiProcessorCount:         " << deviceProp.multiProcessorCount << "\n";
	std::cout << "  maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << "\n";
	std::cout << "  warpSize:                    " << deviceProp.warpSize << "\n";
	std::cout << "  regsPerBlock:                " << deviceProp.regsPerBlock << "\n";
	std::cout << "  concurrentKernels:           " << deviceProp.concurrentKernels << "\n";
	std::cout << "  clockRate:                   " << deviceProp.clockRate << "\n";
	std::cout << "  canMapHostMemory:            " << deviceProp.canMapHostMemory << "\n";
	std::cout << "  computeMode:                 " << deviceProp.computeMode << "\n";
}

bool gpu_initialize(int dev_id)
{
	// Are there any GPU devices?
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	if (nDevices < 1)
		return false;

	// Establish the context
	if (cudaSetDevice(dev_id) != cudaSuccess)
		return false;

	return true;
}
