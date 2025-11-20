#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../cache.h"   //xxxxxxx       #include "../gpucache.h"
#include "../features/pixel.h"
#include "geomoments.cuh"

namespace NyxusGpu
{

    __global__ void kerCentralMomentAll_snu(
        // out
        double* d_prereduce00,
        double* d_prereduce01,
        double* d_prereduce02,
        double* d_prereduce03,
        double* d_prereduce10,
        double* d_prereduce11,
        double* d_prereduce12,
        double* d_prereduce13,
        double* d_prereduce20,
        double* d_prereduce21,
        double* d_prereduce22,
        double* d_prereduce23,
        double* d_prereduce30,
        double* d_prereduce31,
        double* d_prereduce32,
        double* d_prereduce33,
        // in
        int ipow,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        gpureal* origin_x,
        gpureal* origin_y)
    {
        size_t tid = threadIdx.x + blockIdx.x * blockSize;

        if (tid >= cloudlen)
            return;

        double I = int_pow (d_roicloud[tid].inten, ipow),
            OX = (int) *origin_x,
            OY = (int) *origin_y,
            X = double(d_roicloud[tid].x) - OX,
            Y = double(d_roicloud[tid].y) - OY;

        double P0 = 1.0, 
            P1 = X, 
            P2 = X * X, 
            P3 = X * X * X, 
            Q0 = 1.0, 
            Q1 = Y, 
            Q2 = Y * Y, 
            Q3 = Y * Y * Y; 

        d_prereduce00[tid] = I * P0 * Q0;
        d_prereduce01[tid] = I * P0 * Q1;
        d_prereduce02[tid] = I * P0 * Q2;
        d_prereduce03[tid] = I * P0 * Q3;

        d_prereduce10[tid] = I * P1 * Q0;
        d_prereduce11[tid] = I * P1 * Q1;
        d_prereduce12[tid] = I * P1 * Q2;
        d_prereduce13[tid] = I * P1 * Q3;

        d_prereduce20[tid] = I * P2 * Q0;
        d_prereduce21[tid] = I * P2 * Q1;
        d_prereduce22[tid] = I * P2 * Q2;
        d_prereduce23[tid] = I * P2 * Q3;

        d_prereduce30[tid] = I * P3 * Q0;
        d_prereduce31[tid] = I * P3 * Q1;
        d_prereduce32[tid] = I * P3 * Q2;
        d_prereduce33[tid] = I * P3 * Q3;

    }

    __global__ void kerCentralMomentWeightedAll_snu(
        // out
        double* d_prereduce00,
        double* d_prereduce02,
        double* d_prereduce03,
        double* d_prereduce11,
        double* d_prereduce12,
        double* d_prereduce20,
        double* d_prereduce21,
        double* d_prereduce30,
        // in
        const RealPixIntens* d_realintens,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        gpureal* origin_x,
        gpureal* origin_y)
    {
        size_t tid = threadIdx.x + blockIdx.x * blockSize;

        if (tid >= cloudlen)
            return;

        float I = d_realintens[tid],
            OX = (int) *origin_x,
            OY = (int) *origin_y,
            X = d_roicloud[tid].x - OX,
            Y = d_roicloud[tid].y - OY;

        float P0 = 1,
            P1 = X,
            P2 = X * X,
            P3 = X * X * X,
            Q0 = 1,
            Q1 = Y,
            Q2 = Y * Y,
            Q3 = Y * Y * Y;

        d_prereduce00[tid] = I * P0 * Q0;
        d_prereduce02[tid] = I * P0 * Q2;
        d_prereduce03[tid] = I * P0 * Q3;
        d_prereduce11[tid] = I * P1 * Q1;
        d_prereduce12[tid] = I * P1 * Q2;
        d_prereduce20[tid] = I * P2 * Q0;
        d_prereduce21[tid] = I * P2 * Q1;
        d_prereduce30[tid] = I * P3 * Q0;
    }

    bool drvCentralMomentAll__snu(
        // out
        gpureal* d_result,
        // in
        bool need_shape_moments,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        gpureal* origin_x,
        gpureal* origin_y,
        double* d_prereduce,
        void* d_devreduce_tempstorage,
        size_t& devreduce_tempstorage_szb)
    {
        // prepare the shape/intensity selector
        int ipow = need_shape_moments ? 0 : 1;

        // prepare lanes of partial totals
        double* d_pr00 = d_prereduce,
            * d_pr01 = &d_prereduce[cloudlen],
            * d_pr02 = &d_prereduce[cloudlen * 2],
            * d_pr03 = &d_prereduce[cloudlen * 3],
            * d_pr10 = &d_prereduce[cloudlen * 4],
            * d_pr11 = &d_prereduce[cloudlen * 5],
            * d_pr12 = &d_prereduce[cloudlen * 6],
            * d_pr13 = &d_prereduce[cloudlen * 7],
            * d_pr20 = &d_prereduce[cloudlen * 8],
            * d_pr21 = &d_prereduce[cloudlen * 9],
            * d_pr22 = &d_prereduce[cloudlen * 10],
            * d_pr23 = &d_prereduce[cloudlen * 11],
            * d_pr30 = &d_prereduce[cloudlen * 12],
            * d_pr31 = &d_prereduce[cloudlen * 13],
            * d_pr32 = &d_prereduce[cloudlen * 14],
            * d_pr33 = &d_prereduce[cloudlen * 15];

        int nblo = whole_chunks2(cloudlen, blockSize);
        kerCentralMomentAll_snu <<< nblo, blockSize >>> (
            // out
            d_pr00, d_pr01, d_pr02, d_pr03,
            d_pr10, d_pr11, d_pr12, d_pr13,
            d_pr20, d_pr21, d_pr22, d_pr23,
            d_pr30, d_pr31, d_pr32, d_pr33,
            // in
            ipow, 
            d_roicloud, 
            cloudlen, 
            origin_x, 
            origin_y);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        //=== device-reduce:

        bool k; // oK
        k = sumreduce (&d_result[GpusideState::CM00], cloudlen, d_pr00, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM01], cloudlen, d_pr01, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce (&d_result[GpusideState::CM10], cloudlen, d_pr10, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM13], cloudlen, d_pr13, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce (&d_result[GpusideState::CM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM22], cloudlen, d_pr22, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM23], cloudlen, d_pr23, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce (&d_result[GpusideState::CM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM31], cloudlen, d_pr31, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM32], cloudlen, d_pr32, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce (&d_result[GpusideState::CM33], cloudlen, d_pr33, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        return true;
    }

    bool drvCentralMomentWeightedAll__snu(
        // out
        gpureal* d_result,
        // in
        const RealPixIntens* d_realintens,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        gpureal* origin_x,
        gpureal* origin_y,
        double* d_prereduce,
        void* d_devreduce_tempstorage,
        size_t& devreduce_tempstorage_szb)
    {
        // prepare lanes of partial totals
        double
            * d_pr00 = &d_prereduce[0],
            //not using *d_pr01 = &d_prereduce[cloudlen * 1],
            * d_pr02 = &d_prereduce[cloudlen * 2],
            * d_pr03 = &d_prereduce[cloudlen * 3],
            //not using * d_pr10 = &d_prereduce[cloudlen * 4],
            * d_pr11 = &d_prereduce[cloudlen * 5],
            * d_pr12 = &d_prereduce[cloudlen * 6],
            //not using *d_pr13 = &d_prereduce[cloudlen * 7],
            * d_pr20 = &d_prereduce[cloudlen * 8],
            * d_pr21 = &d_prereduce[cloudlen * 9],
            //not using *d_pr22 = &d_prereduce[cloudlen * 10],
            //not using *d_pr23 = &d_prereduce[cloudlen * 11],
            * d_pr30 = &d_prereduce[cloudlen * 12];

        int nblo = whole_chunks2(cloudlen, blockSize);
        kerCentralMomentWeightedAll_snu << < nblo, blockSize >> > (
            // out
            d_pr00, d_pr02, d_pr03,
            d_pr11, d_pr12,
            d_pr20, d_pr21,
            d_pr30,
            // in
            d_realintens, d_roicloud, cloudlen, origin_x, origin_y);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        //=== device-reduce:

        bool k; // oK
        k = sumreduce(&d_result[GpusideState::WCM00], cloudlen, d_pr00, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WCM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WCM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WCM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WCM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WCM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WCM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WCM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        return true;
    }

    bool ImageMomentsFeature_calcCentralMoments__snu(
        // output
        gpureal* d_intermed,
        // input
        bool need_shape_moments,
        const Pixel2* d_roicloud, size_t cloud_len,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb)
    {
        if (drvCentralMomentAll__snu(d_intermed, need_shape_moments, d_roicloud, cloud_len, &d_intermed[ORGX], &d_intermed[ORGY], d_prereduce, d_temp_storage, temp_storage_szb) == false)
            return false;

        return true;
    }

    bool ImageMomentsFeature_calcCentralMomentsWeighted__snu(
        // output
        gpureal* d_intermed,
        // input
        const RealPixIntens* d_realintens, const Pixel2* d_roicloud, size_t cloud_len, gpureal* dev_origin_x, gpureal* dev_origin_y,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb)
    {
        if (drvCentralMomentWeightedAll__snu(d_intermed, d_realintens, d_roicloud, cloud_len, &d_intermed[WORGX], &d_intermed[WORGY], d_prereduce, d_temp_storage, temp_storage_szb) == false)
            return false;

        return true;
    }

}