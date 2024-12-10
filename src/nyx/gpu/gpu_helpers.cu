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

	bool get_best_device (
		// in
		const std::vector<int> & devIds, 
		// out
		int & best_id,
		std::string & lastCuErmsg)
	{
		// given a set of suggested devices, choose the least memory-busy one
		size_t max_freemem_amt = 0;
		for (int k : devIds)
		{
			if (cudaSetDevice(k) != cudaSuccess)
			{
				lastCuErmsg = "invalid device ID " + std::to_string(k);
				continue;
			}

			size_t f, t; // free and total bytes
			auto e = cudaMemGetInfo(&f, &t);
			if (e != cudaSuccess)
			{
				lastCuErmsg = "cuda error " + std::to_string(e) + " : " + cudaGetErrorString(e);
				continue;
			}

			// this device worked
			if (f > max_freemem_amt)
			{
				max_freemem_amt = f;
				best_id = k;
			}
		}

		return best_id >= 0;
	}

}