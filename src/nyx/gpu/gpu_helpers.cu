#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

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
