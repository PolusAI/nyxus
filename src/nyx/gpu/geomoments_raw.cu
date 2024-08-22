#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../gpucache.h"
#include "geomoments.cuh"
#include "../features/pixel.h"

namespace NyxusGpu
{

    // geomoments_central.cu
    bool sumreduce(
        gpureal* d_result,
        size_t cloudlen,
        double* d_prereduce,
        void* d_devreduce_tempstorage,
        size_t& devreduce_tempstorage_szb);

    __device__ double pow_pos_int_raw(double a, int b)
    {
        double retval = 1.0;
        for (int i = 0; i < b; i++)
            retval *= a;
        return retval;
    }

    __global__ void kerRawMoment(
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

        d_prereduce[tid] = inten_ * pow_pos_int_raw(localX, p) * pow_pos_int_raw(localY, q);
    }

    bool drvRawMoment__snu(
        // out
        gpureal* d_result,
        // in
        int p, int q,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        StatsInt base_x,
        StatsInt base_y,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb)
    {
        int nblo = whole_chunks2(cloudlen, blockSize);
        kerRawMoment << < nblo, blockSize >> > (d_prereduce, d_roicloud, cloudlen, base_x, base_y, p, q);

        CHECKERR(cudaGetLastError());
        CHECKERR(cudaDeviceSynchronize());

        //=== device-reduce:

        // Determine temporary device storage requirements and allocate it, if not done yet
        size_t szb;
        CHECKERR(cub::DeviceReduce::Sum(nullptr, szb, d_prereduce/*d_in*/, d_result, cloudlen/*num_items*/));
        if (temp_storage_szb != szb)
        {
            // new size, new storage
            temp_storage_szb = szb;
            CHECKERR(cudaFree(d_temp_storage));
            CHECKERR(cudaMalloc(&d_temp_storage, temp_storage_szb));
        }

        // Run sum-reduction
        auto ercode = cub::DeviceReduce::Sum(d_temp_storage, temp_storage_szb, d_prereduce/*d_in*/, d_result, cloudlen/*num_items*/);
        if (ercode)
        {
            CHECKERR(ercode);
        }

        return true;
    }

    __global__ void kerRawMomentAll_snu(
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
        const Pixel2* d_roicloud,
        size_t cloudlen)
    {
        int tid = threadIdx.x + blockIdx.x * blockSize;

        if (tid >= cloudlen)
            return;

        float I = 1, //<--shape moments, not intensity--    = d_roicloud[tid].inten,
            X = d_roicloud[tid].x,  // = d_roicloud[tid].x - base_x,
            Y = d_roicloud[tid].y;  // = d_roicloud[tid].y - base_y;

        float P0 = I,
            P1 = I * X,
            P2 = I * X * X,
            P3 = I * X * X * X,
            Q0 = I,
            Q1 = I * Y,
            Q2 = I * Y * Y,
            Q3 = I * Y * Y * Y;

        d_prereduce00[tid] = P0 * Q0;
        d_prereduce01[tid] = P0 * Q1;
        d_prereduce02[tid] = P0 * Q2;
        d_prereduce03[tid] = P0 * Q3;

        d_prereduce10[tid] = P1 * Q0;
        d_prereduce11[tid] = P1 * Q1;
        d_prereduce12[tid] = P1 * Q2;
        d_prereduce13[tid] = P1 * Q3;

        d_prereduce20[tid] = P2 * Q0;
        d_prereduce21[tid] = P2 * Q1;
        d_prereduce22[tid] = P2 * Q2;
        d_prereduce23[tid] = P2 * Q3;

        d_prereduce30[tid] = P3 * Q0;
        d_prereduce31[tid] = P3 * Q1;
        d_prereduce32[tid] = P3 * Q2;
        d_prereduce33[tid] = P3 * Q3;
    }

    bool drvRawMomentAll__snu(
        // out
        gpureal* d_result,
        // in
        const Pixel2* d_roicloud,
        size_t cloudlen,
        double* d_prereduce,
        void* d_devreduce_tempstorage,
        size_t& devreduce_tempstorage_szb)
    {
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
        kerRawMomentAll_snu << < nblo, blockSize >> > (
            // out
            d_pr00, d_pr01, d_pr02, d_pr03,
            d_pr10, d_pr11, d_pr12, d_pr13,
            d_pr20, d_pr21, d_pr22, d_pr23,
            d_pr30, d_pr31, d_pr32, d_pr33,
            // in
            d_roicloud, cloudlen);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        //=== device-reduce:

        bool k; // oK
        k = sumreduce(&d_result[GpusideState::RM00], cloudlen, d_pr00, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM01], cloudlen, d_pr01, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce(&d_result[GpusideState::RM10], cloudlen, d_pr10, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM13], cloudlen, d_pr13, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce(&d_result[GpusideState::RM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM22], cloudlen, d_pr22, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM23], cloudlen, d_pr23, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce(&d_result[GpusideState::RM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM31], cloudlen, d_pr31, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM32], cloudlen, d_pr32, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::RM33], cloudlen, d_pr33, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        return true;
    }

    __global__ void kerRawMomentWeighted(
        // out
        double* d_prereduce00,
        double* d_prereduce01,
        double* d_prereduce02,
        double* d_prereduce03,
        double* d_prereduce10,
        double* d_prereduce11,
        double* d_prereduce12,
        double* d_prereduce20,
        double* d_prereduce21,
        double* d_prereduce30,
        // in
        const RealPixIntens* d_realintens,
        const Pixel2* d_roicloud,
        size_t cloudlen)
    {
        int tid = threadIdx.x + blockIdx.x * blockSize;

        if (tid >= cloudlen)
            return;

        float I = d_realintens[tid], // <--shape moments--   d_realintens[tid],
            X = d_roicloud[tid].x,
            Y = d_roicloud[tid].y;

        float P0 = 1,
            P1 = X,
            P2 = X * X,
            P3 = X * X * X,
            Q0 = 1,
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

        d_prereduce20[tid] = I * P2 * Q0;
        d_prereduce21[tid] = I * P2 * Q1;

        d_prereduce30[tid] = I * P3 * Q0;
    }

    bool drvRawMomentWeightedAll__snu(
        // in
        gpureal* d_result,
        // out
        const RealPixIntens* d_realintens,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        double* d_prereduce,
        void*& d_devreduce_tempstorage,
        size_t& devreduce_tempstorage_szb)
    {
        // prepare lanes of partial totals
        double* d_pr00 = d_prereduce,
            * d_pr01 = &d_prereduce[cloudlen],
            * d_pr02 = &d_prereduce[cloudlen * 2],
            * d_pr03 = &d_prereduce[cloudlen * 3],
            * d_pr10 = &d_prereduce[cloudlen * 4],
            * d_pr11 = &d_prereduce[cloudlen * 5],
            * d_pr12 = &d_prereduce[cloudlen * 6],
            * d_pr20 = &d_prereduce[cloudlen * 8],
            * d_pr21 = &d_prereduce[cloudlen * 9],
            * d_pr30 = &d_prereduce[cloudlen * 12];

        int nblo = whole_chunks2(cloudlen, blockSize);
        kerRawMomentWeighted << < nblo, blockSize >> > (
            // out
            d_pr00, d_pr01, d_pr02, d_pr03,
            d_pr10, d_pr11, d_pr12,
            d_pr20, d_pr21,
            d_pr30,
            // in
            d_realintens, d_roicloud, cloudlen);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        //=== device-reduce:

        bool k; // oK
        k = sumreduce(&d_result[GpusideState::WRM00], cloudlen, d_pr00, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WRM01], cloudlen, d_pr01, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WRM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WRM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce(&d_result[GpusideState::WRM10], cloudlen, d_pr10, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WRM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WRM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce(&d_result[GpusideState::WRM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);
        k = sumreduce(&d_result[GpusideState::WRM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        k = sumreduce(&d_result[GpusideState::WRM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb); OK(k);

        return true;
    }

    bool ImageMomentsFeature_calcRawMoments__snu(
        // output
        gpureal* d_intermed,
        // input
        const Pixel2* d_roicloud, size_t cloud_len,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb)
    {
        if (drvRawMomentAll__snu(d_intermed, d_roicloud, cloud_len, d_prereduce, d_temp_storage, temp_storage_szb) == false)
            return false;

        return true;
    }

    bool ImageMomentsFeature_calcRawMomentsWeighted__snu(
        // output
        gpureal* d_intermed,
        // input
        const RealPixIntens* d_realintens, const Pixel2* d_roicloud, size_t cloud_len,
        double* d_prereduce,
        void*& d_temp_storage,
        size_t& temp_storage_szb)
    {
        bool k = drvRawMomentWeightedAll__snu(
            d_intermed,
            d_realintens,
            d_roicloud,
            cloud_len,
            d_prereduce,
            d_temp_storage,
            temp_storage_szb);
        OK(k);

        return true;
    }

}