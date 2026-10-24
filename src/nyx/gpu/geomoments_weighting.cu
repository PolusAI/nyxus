#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"
#include "geomoments.cuh"

namespace NyxusGpu
{

    static __device__ const double WEIGHTING_EPSILON = 0.001;

    __device__ double sqdist(StatsInt x, StatsInt y, const Pixel2& pxl)
    {
        double dx = double(x) - double(pxl.x),
            dy = double(y) - double(pxl.y);
        double dist = dx * dx + dy * dy;
        return dist;
    }

    // Pixel2::min_sqdist over one border of the contour, the ordered walk W[0..n), n >= 1: up to 10
    // points are scanned exhaustively, a longer walk is searched coarse to fine
    __device__ double pixel_sqdist_2_border (StatsInt x, StatsInt y, const Pixel2* W, size_t n)
    {
        auto extrem_d = sqdist(x, y, W[0]);
        if (n <= 10)
        {
            for (size_t i = 1; i < n; i++)
            {
                auto dist = sqdist(x, y, W[i]);
                if (extrem_d > dist)
                    extrem_d = dist;
            }
            return extrem_d;
        }

        size_t a = 0,
            b = n;
        auto extrem_i = a;
        int step = (b - a) / logf(b - a);
        do
        {
            for (size_t i = a + step; i < b; i += step)
            {
                auto dist = sqdist(x, y, W[i]);
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

    // Pixel2::min_sqdist over a contour made of borders: border k is d_roicontour[ends[k-1] .. ends[k])
    // (from 0 for k == 0), and each border is searched on its own
    __device__ double pixel_sqdist_2_contour(StatsInt x, StatsInt y, const Pixel2* d_roicontour, const size_t* d_border_ends, size_t n_borders)
    {
        double mind = 0.0;
        size_t begin = 0;
        for (size_t k = 0; k < n_borders; k++)
        {
            size_t end = d_border_ends[k];
            double dist = pixel_sqdist_2_border(x, y, d_roicontour + begin, end - begin);
            if (k == 0 || dist < mind)
                mind = dist;
            begin = end;
        }
        return mind;
    }

    __global__ void kerCalcWeightedImage3(
        // output
        RealPixIntens* d_realintens,
        // input
        int ipow,
        const Pixel2* d_roicloud,
        size_t cloud_len,
        const Pixel2* d_roicontour,
        const size_t* d_border_ends,
        size_t n_borders)
    {
        // pixel index
        size_t pxIdx = threadIdx.x + blockIdx.x * blockSize;
        if (pxIdx >= cloud_len)
            return;

        StatsInt x = d_roicloud[pxIdx].x,
            y = d_roicloud[pxIdx].y;

        // distance
        double mind2 = pixel_sqdist_2_contour(x, y, d_roicontour, d_border_ends, n_borders);
        double dist = std::sqrt(mind2);

        // weighted intensity
        d_realintens[pxIdx] = int_pow(d_roicloud[pxIdx].inten, ipow) * std::log(dist + WEIGHTING_EPSILON);
    }

    __device__ bool aligned (const Pixel2& p0, const Pixel2& p)
    {
        return p0.x == p.x || p0.y == p.y;
    }

    __device__ double dist_to_segment (const Pixel2 & p0, const Pixel2& p1, const Pixel2& p2)
    {
        double dx = p2.x - p1.x,
            dy = p2.y - p1.y;

        double h = dx * dx + dy * dy;
        if (h <= 0)
            return (double)INT_MAX;

        double retval = std::fabs(dy * p0.x - dx * p0.y + p2.x * p1.y - p2.y * p1.x) / sqrt(h);
        return retval;
    }

    __global__ void kerCalcWeightedImageWholeslide (
        // output
        RealPixIntens* d_realintens,
        // input
        int ipow,
        const Pixel2* d_roicloud,
        size_t cloud_len,
        const Pixel2* d_roicontour,
        size_t contour_len)
    {
        // pixel index
        size_t pxIdx = threadIdx.x + blockIdx.x * blockSize;
        if (pxIdx >= cloud_len)
            return;

        [[maybe_unused]] StatsInt x = d_roicloud[pxIdx].x,
            y = d_roicloud[pxIdx].y;

        // the contour holds the slide's corners, each once: fewer than 4 only on a slide one pixel
        // wide or high, every pixel of which lies on the slide's edge
        if (contour_len < 4)
        {
            d_realintens[pxIdx] = WEIGHTING_EPSILON;
            return;
        }

        const Pixel2 & p = d_roicloud[pxIdx], 
            & c0 = d_roicontour[0],
            & c1 = d_roicontour[1],
            & c2 = d_roicontour[2],
            & c3 = d_roicontour[3];

        // skip contour pixels
        if (aligned(p, c0) || aligned(p, c1) || aligned(p, c2) || aligned(p, c3))
        {
            d_realintens[pxIdx] = WEIGHTING_EPSILON;
            return;
        }

        // min distance. We assume the 4-vertex ROI shape (whole slide)
        double d1 = dist_to_segment (p, c0, c1),
            d2 = dist_to_segment (p, c1, c2),
            d3 = dist_to_segment (p, c2, c3),
            d4 = dist_to_segment (p, c3, c0);

        double dist = fmin(fmin(d1, d2), fmin(d3, d4));

        // weighted intensity
        d_realintens[pxIdx] = int_pow(d_roicloud[pxIdx].inten, ipow) * std::log(dist + WEIGHTING_EPSILON);
    }

    bool ImageMomentsFeature_calc_weighted_intens(
        // output
        RealPixIntens* d_realintens_buf,
        // input
        bool wholeslide,
        bool need_shape_moments,
        const Pixel2* d_roicloud,
        size_t cloud_len,
        const Pixel2* d_roicontour,
        size_t contour_len,
        const size_t* d_border_ends,
        size_t n_borders)
    {
        // prepare the shape/intensity selector
        int ipow = need_shape_moments ? 0 : 1;

        int nb = whole_chunks2(cloud_len, blockSize);

        if (wholeslide)
            kerCalcWeightedImageWholeslide <<< nb, blockSize >>> (d_realintens_buf, ipow, d_roicloud, cloud_len, d_roicontour, contour_len);
        else
            kerCalcWeightedImage3 <<< nb, blockSize >>> (d_realintens_buf, ipow, d_roicloud, cloud_len, d_roicontour, d_border_ends, n_borders);

        cudaError_t ok = cudaDeviceSynchronize();
        CHECKERR(ok);

        ok = cudaGetLastError();
        CHECKERR(ok);

        return true;
    }

}
