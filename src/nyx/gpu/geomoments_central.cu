#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../features/pixel.h"

namespace Nyxus
{
    extern double* devPrereduce;      // reduction helper [roi_cloud_len]
    extern double* devBlockSubsums;   // [whole chunks]
    extern double* hoBlockSubsums;    // [whole chunks]
};

__device__ double pow_pos_int_central (double a, int b)
{
    if (b == 0)
        return 1.0;
    double retval = 1.0;
    for (int i = 0; i < b; i++)
        retval *= a;
    return retval;
}

__global__ void kerCentralMoment (
    double* d_prereduce,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y,
    double origin_x, 
    double origin_y,
    int p,
    int q)
{
    int tid = threadIdx.x + blockIdx.x * blockSize;
    
    if (tid >= cloudlen)
        return;

    d_prereduce[tid] = double(d_roicloud[tid].inten) 
        * pow_pos_int_central (double(d_roicloud[tid].x - base_x) - origin_x, p) 
        * pow_pos_int_central (double(d_roicloud[tid].y - base_y) - origin_y, q);
}

__global__ void kerCentralMomentWeighted (
    double* d_prereduce,
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y,
    double origin_x,
    double origin_y,
    int p,
    int q)
{
    int tid = threadIdx.x + blockIdx.x * blockSize;

    if (tid >= cloudlen)
        return;

    d_prereduce[tid] = d_realintens[tid]
        * pow_pos_int_central(double(d_roicloud[tid].x - base_x) - origin_x, p)
        * pow_pos_int_central(double(d_roicloud[tid].y - base_y) - origin_y, q);
}

bool drvCentralMoment(
    double& retval,
    int p, int q,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y,
    double origin_x,
    double origin_y)
{
    int nblo = whole_chunks2(cloudlen, blockSize);
    kerCentralMoment <<< nblo, blockSize >>> (Nyxus::devPrereduce, d_roicloud, cloudlen, base_x, base_y, origin_x, origin_y, p, q);

    CHECKERR(cudaPeekAtLastError());
    CHECKERR(cudaDeviceSynchronize());
    CHECKERR(cudaGetLastError());

    //=== device-reduce:
    // Determine temporary device storage requirements
    double* d_out = nullptr;
    CHECKERR(cudaMalloc(&d_out, sizeof(double)));
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, d_out, cloudlen/*num_items*/);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, d_out, cloudlen/*num_items*/);
    double h_out;
    CHECKERR(cudaMemcpy(&h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    retval = h_out;

    return true;
}

bool drvCentralMomentWeighted (
    double& retval,
    int p, int q,
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y,
    double origin_x,
    double origin_y)
{
    int nblo = whole_chunks2(cloudlen, blockSize);
    kerCentralMomentWeighted <<< nblo, blockSize >>> (Nyxus::devPrereduce, d_realintens, d_roicloud, cloudlen, base_x, base_y, origin_x, origin_y, p, q);

    CHECKERR(cudaPeekAtLastError());
    CHECKERR(cudaDeviceSynchronize());
    CHECKERR(cudaGetLastError());

    //=== device-reduce:
    // Determine temporary device storage requirements
    double* d_out = nullptr;
    CHECKERR(cudaMalloc(&d_out, sizeof(double)));
    void* d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, d_out, cloudlen/*num_items*/);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, Nyxus::devPrereduce/*d_in*/, d_out, cloudlen/*num_items*/);
    // d_out <-- [38]
    double h_out;
    CHECKERR(cudaMemcpy(&h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    retval = h_out;

    return true;
}

bool ImageMomentsFeature_calcCentralMoments (
    // output
    double& _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _22, double& _30,
    // input
    const Pixel2* d_roicloud, size_t cloud_len, StatsInt base_x, StatsInt base_y, double origin_x, double origin_y)
{
    // Mark as unassigned a value
    _00 = _01 = _02 = _03 = _10 = _11 = _12 = _20 = _21 = _22 = _30 = -1;

    // Calculate
    if (drvCentralMoment(_00, 0, 0, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_01, 0, 1, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_02, 0, 2, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_03, 0, 3, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_10, 1, 0, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_11, 1, 1, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_12, 1, 2, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_20, 2, 0, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_21, 2, 1, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_22, 2, 2, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMoment(_30, 3, 0, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    return true;
}

bool ImageMomentsFeature_calcCentralMomentsWeighted (
    // output
    double& _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _22, double& _30,
    // input
    const RealPixIntens* d_realintens, const Pixel2* d_roicloud, size_t cloud_len, StatsInt base_x, StatsInt base_y, double origin_x, double origin_y)
{
    // Mark as unassigned a value
    _00 = _01 = _02 = _03 = _10 = _11 = _12 = _20 = _21 = _22 = _30 = -1;

    // Calculate
    if (drvCentralMomentWeighted (_00, 0, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_01, 0, 1, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_02, 0, 2, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_03, 0, 3, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_10, 1, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_11, 1, 1, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_12, 1, 2, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_20, 2, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_21, 2, 1, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_22, 2, 2, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    if (drvCentralMomentWeighted (_30, 3, 0, d_realintens, d_roicloud, cloud_len, base_x, base_y, origin_x, origin_y) == false)
        return false;

    return true;
}

