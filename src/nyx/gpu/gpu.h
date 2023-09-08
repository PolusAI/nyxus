#pragma once
#include <cufft.h>
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#define whole_chunks2(n,t) ((unsigned int)n+(t)-1)/(t)

#ifndef WITH_PYTHON_H

    #define CHECKCUFFTERR(call) \
    if ((call) != CUFFT_SUCCESS)    \
    {	\
    std::cerr << __FILE__ << ":" << __LINE__ << "cuda cuFFT error code " << call << ": " << _cudaGetErrorEnum(call) << "\n";	\
    return false;	\
    };

    #define CHECKERR(ok)	\
    if (ok != cudaSuccess)	\
    {	\
    std::cerr << __FILE__ << ":" << __LINE__ << " cuda error code " << ok << ": " << cudaGetErrorString(ok) << "\n";	\
    return false;	\
    };

#else

    #define CHECKCUFFTERR(call) \
    if (((call) != CUFFT_SUCCESS)    \
    {	\
    throw (std::runtime_error(std::string(FILE)+ ":"+std::to_string(LINE)+ ":"+"cuda cuFFT error code "+std::to_string(call)));	\
    return false;	\
    };

    #define CHECKERR(ok)	\
    if (ok != cudaSuccess)	\
    {	\
    throw (std::runtime_error(std::string(FILE)+ ":"+std::to_string(LINE)+ ":"+"cuda error code "+std::to_string(ok)));	\
    return false;	\
    };

#endif


bool gpu_initialize(int dev_id);
