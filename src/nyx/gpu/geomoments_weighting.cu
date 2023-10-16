#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"

static __device__ const double WEIGHTING_EPSILON = 0.01;

__device__ double sqdist (StatsInt x, StatsInt y, const Pixel2 & pxl)
{
    double dx = double(x) - double(pxl.x),
        dy = double(y) - double(pxl.y);
    double dist = dx * dx + dy * dy;
    return dist;
}

__device__ double pixel_sqdist_2_contour (StatsInt x, StatsInt y, const Pixel2* d_roicontour, size_t contour_len)
{
    size_t n = contour_len;
    size_t a = 0, 
        b = n;
    auto extrem_d = sqdist(x, y, d_roicontour[a]);
    auto extrem_i = a;
    int step = (b - a) / logf (b - a);
    do
    {
        for (size_t i = a + step; i < b; i += step)
        {
            auto dist = sqdist(x, y, d_roicontour[i]);
            if (extrem_d > dist)
            {
                extrem_d = dist;
                extrem_i = i;
            }
        }

        // left or right ?
        auto stepL = extrem_i >= step ? step : extrem_i,
            stepR = extrem_i + step < n ? step : n - extrem_i;

        a = extrem_i - stepL;
        b = extrem_i + stepR;
        step = b - a <= 10 ? 1 : (b - a) / logf(b - a);
    } while (b - a > 2);

    return extrem_d;
}


__global__ void kerCalcWeightedImage3(
    // output
    RealPixIntens* d_realintens,
    // input
    const Pixel2* d_roicloud,
    size_t cloud_len,
    const Pixel2* d_roicontour,
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
    const Pixel2* d_roicloud,
    size_t cloud_len,
    const Pixel2* d_roicontour,
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

