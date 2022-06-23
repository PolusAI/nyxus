#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#define whole_chunks2(n,t) ((unsigned int)n+(t)-1)/(t)

#define CHECKERR(ok)	\
if (ok != cudaSuccess)	\
{	\
std::cerr << __FILE__ << ":" << __LINE__ << " cuda error code " << ok << ": " << cudaGetErrorString(ok) << "\n";	\
return false;	\
};	\


bool gpu_initialize(int dev_id);


