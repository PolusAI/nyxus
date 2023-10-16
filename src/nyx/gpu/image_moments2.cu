#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>

#include "gpu.h"





//???? ----------
__global__ void kerTest1(
    double* outdata,
    size_t cloudlen)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Specialize BlockReduce for type float. 
    typedef cub::BlockReduce<float, blockSize> BlockReduceT;

    // --- Allocate temporary storage in shared memory 
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    float result;
    if (tid < cloudlen) 
        result = BlockReduceT(temp_storage).Sum(1); // .Sum(indata[tid]);

    // --- Update block reduction value
    if (threadIdx.x == 0) 
        outdata[blockIdx.x] = result;

    return;
}

bool drvTest1(
    double& output,
    size_t cloudlen)  
{
    cudaError_t ok;

    // 'd_lastBlockCounter' is assumed to be zeroed externally:
    //      auto ok = cudaMemset(d_lastBlockCounter, 0, sizeof(int));
    //      CHECKERR(ok);

    int nblo = whole_chunks2(cloudlen, blockSize);

    double* devBlocksData = nullptr;
    size_t szb = nblo * sizeof(double);
    CHECKERR(cudaMalloc(&devBlocksData, szb));
    double* hoBlocksData = new double[nblo];

    kerTest1 <<< nblo/*NUM_BLOCKS*/, blockSize >> > (devBlocksData, cloudlen);

    // Wait for the kernel to finish, and returns any errors encountered during the launch
    ok = cudaDeviceSynchronize();
    if (ok != cudaSuccess)
    {
        std::cerr << __FILE__ << ":" << __LINE__ << " cudaDeviceSynchronize returned error code " << ok << ": " << cudaGetErrorString(ok) << "\n";
        std::cerr << "cudaDeviceSynchronize returned error code " << ok << " \n";
        return false;
    }

    // Check for any errors launching the kernel
    ok = cudaGetLastError();
    if (ok != cudaSuccess)
    {
        std::cerr << __FILE__ << ":" << __LINE__ << " Kernel launch failed with error code " << ok << ": " << cudaGetErrorString(ok) << "\n";
        return false;
    }

    CHECKERR(cudaMemcpy(hoBlocksData, devBlocksData, szb, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < nblo; i++)
        sum += hoBlocksData[i];
    delete hoBlocksData;
    output = sum;

    // Leaving the result on device-side. It's supposed to be pulled externally:
    //      auto ok = cudaMemcpy(&hoM, d_M, result_vector_length*sizeof(hoM[0]), cudaMemcpyDeviceToHost);
    //      CHECKERR(ok);

    return true;
}
//
//
//


