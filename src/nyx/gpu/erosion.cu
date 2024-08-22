#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "gpu.h"
#include "../gpucache.h"

#include "../helpers/timing.h"


namespace NyxusGpu
{
    __global__ void kerImatFromShapeCloud (PixIntens* d_imat, const Pixel2* d_roicloud, size_t cloudlen, size_t imat_w, int extra_inten)
    {
        int tid = threadIdx.x + blockIdx.x * blockSize;

        if (tid >= cloudlen)
            return;

        const Pixel2* p = &d_roicloud[tid];
        size_t x = p->x,
            y = p->y;
        size_t offs = y * imat_w + x;
        d_imat[offs] = 1 /*not using intensity! p->inten*/ + extra_inten;
    }

    __global__ void kerCopyImat (PixIntens* imatD, PixIntens* imatS, size_t roi_w, size_t roi_h)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if (x >= roi_w)
            return;

        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (y >= roi_h)
            return;

        size_t idx = y * roi_w + x;
        imatD[idx] = imatS[idx];
    }

    __global__ void kerErosion (PixIntens* d_imat1, PixIntens* d_imat2, size_t roi_w, size_t roi_h, double* d_prereduce, size_t cloud_len)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        if (x >= roi_w)
            return;

        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (y >= roi_h)
            return;

        // SE
        static const int SE_NR = 3, SE_NC = 3; 	// rows, columns
        int strucElem[SE_NR][SE_NC] = { {0,1,0}, {1,1,1}, {0,1,0} };

        int hsenc = SE_NC / 2,
            hsenr = SE_NR / 2;
        if (y <= hsenr || y >= roi_h - hsenr)
            return;
        if (x <= hsenc || x >= roi_w - hsenc)
            return;

        // Check pixel's neighborhood within the SE
        PixIntens N[SE_NR][SE_NC];

        int row1 = y - hsenr;
        int row2 = y + hsenr;
        int col1 = x - hsenc;
        int col2 = x + hsenc;

        bool all0 = true;
 
        for (int r = row1; r <= row2; r++)
            for (int c = col1; c <= col2; c++)
            {
                auto pi = d_imat1 [r * roi_w + c];
                N[r - row1][c - col1] = pi;

                if (pi)
                    all0 = false;
            }
        if (all0)
        {
            d_imat2[y*roi_w+x] = 0; // cpuside: I2.set_at(row * width + col, 0)
            return;
        }

        // Apply the structuring element
        PixIntens minI = 999999; // cpuside: UINT32_MAX
        //Nv.clear();
        for (int r = 0; r < SE_NR; r++)
            for (int c = 0; c < SE_NC; c++)
            {
                int s = strucElem[r][c];
                if (s)
                    minI = minI <= N[r][c] ? minI : N[r][c]; // cpuside: Nv.push_back(N[r][c])
            }
        d_imat2[y * roi_w + x] = minI; // cpuside: I2.set_at(row * width + col, minPixel)

        if (minI != 0)
            // catch the 1st, cpuside: if (d_prereduce[0] == 0)
            d_prereduce[0] = y + double(x) / pow(10, ceil(log10(double(x)))); // flag "nnz > 0"
    }

    __global__ void kerErosion2 (PixIntens* d_imat1, PixIntens* d_imat2, size_t roi_w, size_t roi_h, double* d_prereduce, Pixel2* d_cloud, size_t cloud_len)
    {
        int tid = threadIdx.x + blockIdx.x * blockSize;
        if (tid >= cloud_len)
            return;

        Pixel2 * px = & (d_cloud[tid]);
        size_t x = px->x,
            y = px->y;

        // SE
        static const int SE_NR = 3, SE_NC = 3; 	// rows, columns
        int strucElem[SE_NR][SE_NC] = { {0,1,0}, {1,1,1}, {0,1,0} };

        int hsenc = SE_NC / 2,
            hsenr = SE_NR / 2;
        if (y <= hsenr || y >= roi_h - hsenr)
            return;
        if (x <= hsenc || x >= roi_w - hsenc)
            return;

        // Check pixel's neighborhood within the SE
        PixIntens N[SE_NR][SE_NC];

        int row1 = y - hsenr;
        int row2 = y + hsenr;
        int col1 = x - hsenc;
        int col2 = x + hsenc;

        bool all0 = true;
        for (int r = row1; r <= row2; r++)
            for (int c = col1; c <= col2; c++)
            {
                auto pi = d_imat1[r * roi_w + c];
                N[r - row1][c - col1] = pi;

                if (pi)
                    all0 = false;
            }
        if (all0)
        {
            d_imat2[y * roi_w + x] = 0;
            return;
        }

        // Apply the structuring element
        PixIntens minI = 999999; // cpuside: UINT32_MAX
        for (int r = 0; r < SE_NR; r++)
            for (int c = 0; c < SE_NC; c++)
            {
                int s = strucElem[r][c];
                if (s)
                    minI = minI <= N[r][c] ? minI : N[r][c]; // cpuside: Nv.push_back(N[r][c]);
            }
        d_imat2[y * roi_w + x] = minI; // cpuside: I2.set_at(row * width + col, minPixel);

        if (minI != 0)
            // catch the 1st
            d_prereduce[0] = y + double(x) / pow(10, ceil(log10(double(x)))); // flag "nnz > 0"
    }

    bool ErosionFeature_calculate_via_gpu(size_t roi_index, size_t roi_w, size_t roi_h, int max_n_erosions, int & fval)
    {
        // context of ROI #roi_index:
        //
        // proper batch has been sent to gpu-side by this point
        //
        gpureal* state = &NyxusGpu::gpu_featurestatebuf.devbuffer[roi_index * GpusideState::__COUNT__];
        size_t cloud_len = NyxusGpu::gpu_roiclouds_2d.ho_lengths[roi_index];
        size_t cloud_offset = NyxusGpu::gpu_roiclouds_2d.ho_offsets[roi_index];
        Pixel2* d_cloud = &NyxusGpu::gpu_roiclouds_2d.devbuffer[cloud_offset];
        PixIntens* d_imat1 = &NyxusGpu::dev_imat1[roi_index],
            * d_imat2 = &NyxusGpu::dev_imat2[roi_index];
        double* d_prereduce = NyxusGpu::dev_prereduce;

        //***** zero imat2
        size_t szb = roi_w * roi_h * sizeof(d_imat1[0]);
        CHECKERR(cudaMemset (d_imat2, 0, szb));

        //***** imat2 <-- ROI cloud 
        int nblo = whole_chunks2 (cloud_len, blockSize);
        kerImatFromShapeCloud << < nblo, blockSize >> > (d_imat2, d_cloud, cloud_len, roi_w, +1);
        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        int i;  // erosion iteration
        for (i = 0; i < max_n_erosions; i++)
        {
            //***** imat1 <-- imat2
            cudaMemcpy (d_imat1, d_imat2, szb, cudaMemcpyDeviceToDevice);

            //***** flag
            double pixelsRemainining = 0;
            CHECKERR(cudaMemcpy(&d_prereduce[0], &pixelsRemainining, sizeof(pixelsRemainining), cudaMemcpyHostToDevice));

            //***** erode once

            // v2
            int nblo = whole_chunks2(cloud_len, blockSize);
            kerErosion2 <<<nblo, blockSize>>> (d_imat1, d_imat2, roi_w, roi_h, d_prereduce, d_cloud, cloud_len);
            CHECKERR(cudaDeviceSynchronize());
            CHECKERR(cudaGetLastError());

            //***** flag (check if any pixel is NZ)
            CHECKERR(cudaMemcpy(&pixelsRemainining, &d_prereduce[0], sizeof(pixelsRemainining), cudaMemcpyDeviceToHost));
            if (pixelsRemainining == 0)
                break;
        }

        fval = i;

        return true;
    }

}