#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace NyxusGpu
{

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

	std::vector<std::map<std::string, std::string>> get_gpu_properties() 
	{
		int n_devices;
		std::vector<std::map<std::string, std::string>> props;

		cudaGetDeviceCount(&n_devices);

		for (int i = 0; i < n_devices; ++i) 
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);

			std::map<std::string, std::string> temp;

			temp["Device number"] = std::to_string(i);
			temp["Device name"] = prop.name;
			temp["Memory"] = std::to_string(prop.totalGlobalMem / pow(2, 30)) + " GB";
			temp["Capability"] = std::to_string(prop.major) + std::to_string(prop.minor);

			props.push_back(temp);
		}

		return props;
	}

}