#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../cache.h"

/*static*/	bool GpusideCache::gpu_delete(void* devptr)
{
	CHECKERR(cudaFree(devptr));
	return true;
}

/*static*/	bool GpusideCache::allocate_on_device (void** ptr, size_t szb)
{
	CHECKERR(cudaMalloc(ptr, szb));
	return true;
}

/*static*/	bool GpusideCache::upload_on_device (void* devbuffer, void* hobuffer, size_t szb)
{
	CHECKERR(cudaMemcpy(devbuffer, hobuffer, szb, cudaMemcpyHostToDevice));
	return true;
}

/*static*/	bool GpusideCache::download_on_host (void* hobuffer, void* devbuffer, size_t szb)
{
	CHECKERR(cudaMemcpy(hobuffer, devbuffer, szb, cudaMemcpyDeviceToHost));
	return true;
}

/*static*/	bool GpusideCache::devicereduce_evaluate_buffer_szb (size_t& devicereduce_buf_szb, size_t maxLen)
{
	auto ercode = cub::DeviceReduce::Sum((void*)nullptr, devicereduce_buf_szb, (double*)nullptr /*in: d_prereduce*/, (double*)nullptr /*out: result*/, maxLen /*in: cloudlen*/);
	CHECKERR(ercode);
	return true;
}

/*static*/	bool GpusideCache::gpu_get_free_mem (size_t& amt)
{
	size_t mf, ma;
	CHECKERR(cudaMemGetInfo(&mf, &ma));
	amt = mf;
	return true;
}
