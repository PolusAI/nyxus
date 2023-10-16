#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../features/pixel.h"

namespace Nyxus
{
    extern double* devPrereduce;    // reduction helper [roi_cloud_len]
    extern double* d_out;           // 1 double
    extern void* d_temp_storage;    // allocated [] elements by cub::DeviceReduce::Sum()
    extern size_t temp_storage_bytes;
};

__device__ double pow_pos_int_raw (double a, int b)
{
    double retval = 1.0;
    for (int i = 0; i < b; i++)
        retval *= a;
    return retval;
}

__global__ void kerRawMoment (
    double* d_prereduce,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x, 
    StatsInt base_y,
    int p,
    int q)
{
    int tid = threadIdx.x + blockIdx.x * blockSize;

    if (tid >= cloudlen)
        return;

    float inten_ = d_roicloud[tid].inten,
        x_ = d_roicloud[tid].x,
        y_ = d_roicloud[tid].y, 
        x0 = base_x, 
        y0 = base_y,
        localX = x_ - x0,
        localY = y_ - y0;
    
    d_prereduce[tid] = inten_
        * pow_pos_int_raw(localX, p)
        * pow_pos_int_raw(localY, q);
}

__global__ void kerRawMomentWeighted (
    double* d_prereduce,
    const RealPixIntens* d_realintens, 
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y,
    int p,
    int q)
{
    int tid = threadIdx.x + blockIdx.x * blockSize;

    if (tid >= cloudlen)
        return;

    d_prereduce[tid] = d_realintens[tid]
        * pow_pos_int_raw(double(d_roicloud[tid].x - base_x), p) 
        * pow_pos_int_raw(double(d_roicloud[tid].y - base_y), q);
}

bool drvRawMoment (
    double &retval, 
    int p, int q,
    const Pixel2 *d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y)
{
    int nblo = whole_chunks2(cloudlen, blockSize);
    kerRawMoment <<< nblo, blockSize >>> (Nyxus::devPrereduce, d_roicloud, cloudlen, base_x, base_y, p, q);

    CHECKERR (cudaPeekAtLastError());
    CHECKERR (cudaGetLastError());   
    CHECKERR (cudaDeviceSynchronize());

    //=== device-reduce:
    // Determine temporary device storage requirements and allocate it, if not done yet
    if (!Nyxus::d_temp_storage)
    {
        CHECKERR(cub::DeviceReduce::Sum(Nyxus::d_temp_storage, Nyxus::temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, Nyxus::d_out, cloudlen/*num_items*/));
        // Allocate temporary storage
        CHECKERR(cudaMalloc(&Nyxus::d_temp_storage, Nyxus::temp_storage_bytes));
    }
    // Run sum-reduction
    CHECKERR(cub::DeviceReduce::Sum (Nyxus::d_temp_storage, Nyxus::temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, Nyxus::d_out, cloudlen/*num_items*/));
    double h_out;
    CHECKERR(cudaMemcpy(&h_out, Nyxus::d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    
    retval = h_out;

    return true;
}

bool drvRawMomentWeighted(
    double& retval,
    int p, int q,
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y)
{
    int nblo = whole_chunks2(cloudlen, blockSize);
    kerRawMomentWeighted <<< nblo, blockSize >>> (Nyxus::devPrereduce, d_realintens, d_roicloud, cloudlen, base_x, base_y, p, q);

    CHECKERR(cudaPeekAtLastError());
    CHECKERR(cudaDeviceSynchronize());
    CHECKERR(cudaGetLastError());

    //=== device-reduce:
    // Determine temporary device storage requirements
    if (!Nyxus::d_temp_storage)
    {
        CHECKERR(cub::DeviceReduce::Sum(Nyxus::d_temp_storage, Nyxus::temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, Nyxus::d_out, cloudlen/*num_items*/));
        // Allocate temporary storage
        CHECKERR(cudaMalloc(&Nyxus::d_temp_storage, Nyxus::temp_storage_bytes));
    }
    // Run sum-reduction
    CHECKERR(cub::DeviceReduce::Sum (Nyxus::d_temp_storage, Nyxus::temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, Nyxus::d_out, cloudlen/*num_items*/));
    double h_out;
    CHECKERR(cudaMemcpy(&h_out, Nyxus::d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    retval = h_out;

    return true;
}

bool ImageMomentsFeature_calcRawMoments (
    // output
    double & _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _30,
    // input
    const Pixel2* d_roicloud, size_t cloud_len, StatsInt base_x, StatsInt base_y) // image data
{
    // Mark as unassigned a value
    _00 = _01 = _02 = _03 = _10 = _11 = _12 = _20 = _21 = _30 = -1; 

    // Calculate
    if (drvRawMoment(_00, 0, 0, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_01, 0, 1, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_02, 0, 2, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_03, 0, 3, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_10, 1, 0, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_11, 1, 1, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_12, 1, 2, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_20, 2, 0, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_21, 2, 1, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMoment(_30, 3, 0, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    return true;
}

bool ImageMomentsFeature_calcRawMomentsWeighted (
    // output
    double& _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _30,
    // input
    const RealPixIntens* d_realintens, const Pixel2* d_roicloud, size_t cloud_len, StatsInt base_x, StatsInt base_y) 
{
    // Mark as unassigned a value
    _00 = _01 = _02 = _03 = _10 = _11 = _12 = _20 = _21 = _30 = -1;

    // Calculate
    if (drvRawMomentWeighted(_00, 0, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_01, 0, 1, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_02, 0, 2, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_03, 0, 3, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_10, 1, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_11, 1, 1, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_12, 1, 2, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_20, 2, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_21, 2, 1, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    if (drvRawMomentWeighted(_30, 3, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y) == false)
        return false;

    return true;
}

