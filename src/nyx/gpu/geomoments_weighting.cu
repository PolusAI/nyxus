#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"

static __device__ const double WEIGHTING_EPSILON = 0.01;

__device__ double pixel_sqdist_2_contour(StatsInt x, StatsInt y, Pixel2* d_roicontour, size_t contour_len)
{
    double dx = double(x) - double(d_roicontour[0].x),
        dy = double(y) - double(d_roicontour[0].y);
    double dist = dx * dx + dy * dy;
    double mindist = dist;
    for (size_t i = 1; i < contour_len; i++)
    {
        dx = double(x) - double(d_roicontour[i].x);
        dy = double(y) - double(d_roicontour[i].y);
        double dist = dx * dx + dy * dy;
        mindist = __min(mindist, dist);
    }
    return mindist;
}

__global__ void kerCalcWeightedImage3(
    // output
    RealPixIntens* d_realintens,
    // input
    Pixel2* d_roicloud,
    size_t cloud_len,
    Pixel2* d_roicontour,
    size_t contour_len)
{
    // pixel index
    int pxIdx = threadIdx.x + blockIdx.x * blockSize;
    if (pxIdx >= cloud_len)
        return;

    PixIntens pin = d_roicloud[pxIdx].inten;
    StatsInt x = d_roicloud[pxIdx].x,
        y = d_roicloud[pxIdx].y;

    // pixel distance
    double mind2 = pixel_sqdist_2_contour(x, y, d_roicontour, contour_len);
    double dist = std::sqrt(mind2);

    // weighted intensity
    d_realintens[pxIdx] = RealPixIntens(double(pin) / (dist + WEIGHTING_EPSILON));
}

bool ImageMomentsFeature_calc_weighted_intens(
    // output
    RealPixIntens* d_realintens_buf,
    // input
    Pixel2* d_roicloud,
    size_t cloud_len,
    Pixel2* d_roicontour,
    size_t contour_len)
{
    int nb = whole_chunks2(cloud_len, blockSize);
    kerCalcWeightedImage3 << < nb, blockSize >> > (d_realintens_buf, d_roicloud, cloud_len, d_roicontour, contour_len);

    cudaError_t ok = cudaDeviceSynchronize();
    CHECKERR(ok);

    ok = cudaGetLastError();
    CHECKERR(ok);

    return true;
}

