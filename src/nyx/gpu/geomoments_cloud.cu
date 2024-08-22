#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../gpucache.h"
#include "geomoments.cuh"
#include "../features/pixel.h"

// geomoments_central.cu
__device__ bool sumreduce_cloud(
    gpureal* d_result,
    size_t cloudlen,
    double* d_prereduce,
    void* d_devreduce_tempstorage,
    size_t& devreduce_tempstorage_szb)
{
    size_t szb;
    if (cudaSuccess != cub::DeviceReduce::Sum(nullptr, szb, d_prereduce/*d_in*/, d_result, cloudlen/*num_items*/))
        return false;

    return true;

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_devreduce_tempstorage, devreduce_tempstorage_szb, d_prereduce/*d_in*/, d_result, cloudlen/*num_items*/);

    return true;
}

__global__ void kerRawMomentAll_snu_cloud(
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

    float I = d_roicloud[tid].inten,
        X = d_roicloud[tid].x,
        Y = d_roicloud[tid].y;

    float P0 = I,
        P1 = I * X,
        P2 = I * X * X,
        P3 = I * X * X * X,
        Q0 = I,
        Q1 = I * Y,
        Q2 = I * Y * Y,
        Q3 = I * Y * Y;

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
}

__global__ void kerCentralMomentAll_snu_cloud (
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
    size_t cloudlen,
    gpureal* origin_x,
    gpureal* origin_y)
{
    int tid = threadIdx.x + blockIdx.x * blockSize;

    if (tid >= cloudlen)
        return;

    float I = d_roicloud[tid].inten,
        OX = *origin_x,
        OY = *origin_y,
        X = d_roicloud[tid].x - OX,
        Y = d_roicloud[tid].y - OY;

    float P0 = I,
        P1 = I * X,
        P2 = I * X * X,
        P3 = I * X * X * X,
        Q0 = I,
        Q1 = I * Y,
        Q2 = I * Y * Y,
        Q3 = I * Y * Y;

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

__global__  void kerNormCentralMoms_cloud (gpureal* s, bool weighted)  // s - state
{
    // safety
    int tid = threadIdx.x + blockIdx.x * blockSize;
    if (tid)
        return;

    // Formula:
    //  double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    //  double retval = CentralMom(D, p, q) / pow(Moment(D, 0, 0), temp);

    double k, rm00 = s[GpusideState::RM00];

    if (weighted)
    {
        // 02
        k = (0. + 2.) / 2. + 1.;
        s[GpusideState::WNU02] = s[GpusideState::WCM02] / pow(rm00, k);
        // 03
        k = (0. + 3.) / 2. + 1.;
        s[GpusideState::WNU03] = s[GpusideState::WCM03] / pow(rm00, k);
        // 11
        k = (1. + 1.) / 2. + 1.;
        s[GpusideState::WNU11] = s[GpusideState::WCM11] / pow(rm00, k);
        // 12
        k = (1. + 2.) / 2. + 1.;
        s[GpusideState::WNU12] = s[GpusideState::WCM12] / pow(rm00, k);
        // 20
        k = (2. + 0.) / 2. + 1.;
        s[GpusideState::WNU20] = s[GpusideState::WCM20] / pow(rm00, k);
        // 21
        k = (2. + 1.) / 2. + 1.;
        s[GpusideState::WNU21] = s[GpusideState::WCM21] / pow(rm00, k);
        // 30
        k = (3. + 0.) / 2. + 1.;
        s[GpusideState::WNU30] = s[GpusideState::WCM30] / pow(rm00, k);
    }
    else
    {
        // 02
        k = (0. + 2.) / 2. + 1.;
        s[GpusideState::NU02] = s[GpusideState::CM02] / pow(rm00, k);
        // 03
        k = (0. + 3.) / 2. + 1.;
        s[GpusideState::NU03] = s[GpusideState::CM03] / pow(rm00, k);
        // 11
        k = (1. + 1.) / 2. + 1.;
        s[GpusideState::NU11] = s[GpusideState::CM11] / pow(rm00, k);
        // 12
        k = (1. + 2.) / 2. + 1.;
        s[GpusideState::NU12] = s[GpusideState::CM12] / pow(rm00, k);
        // 20
        k = (2. + 0.) / 2. + 1.;
        s[GpusideState::NU20] = s[GpusideState::CM20] / pow(rm00, k);
        // 21
        k = (2. + 1.) / 2. + 1.;
        s[GpusideState::NU21] = s[GpusideState::CM21] / pow(rm00, k);
        // 30
        k = (3. + 0.) / 2. + 1.;
        s[GpusideState::NU30] = s[GpusideState::CM30] / pow(rm00, k);
    }
}

__device__ double sqdist_cloud (StatsInt x, StatsInt y, const Pixel2& pxl)
{
    double dx = double(x) - double(pxl.x),
        dy = double(y) - double(pxl.y);
    double dist = dx * dx + dy * dy;
    return dist;
}

__device__ double pixel_sqdist_2_contour_cloud (StatsInt x, StatsInt y, const Pixel2* d_roicontour, size_t contour_len)
{
    size_t n = contour_len;
    size_t a = 0,
        b = n;
    auto extrem_d = sqdist_cloud (x, y, d_roicontour[a]);
    auto extrem_i = a;
    int step = (b - a) / logf(b - a);
    do
    {
        for (size_t i = a + step; i < b; i += step)
        {
            auto dist = sqdist_cloud (x, y, d_roicontour[i]);
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

static __device__ const double WEIGHTING_EPSILON = 0.01;

__global__ void kerCalcWeightedImage3_cloud (
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
    double mind2 = pixel_sqdist_2_contour_cloud (x, y, d_roicontour, contour_len);
    double dist = std::sqrt(mind2);

    // weighted intensity
    d_realintens[pxIdx] = RealPixIntens(double(pin) / (dist + WEIGHTING_EPSILON));
}

__global__ void kerRawMomentWeighted_cloud (
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

    float I = d_realintens[tid],
        X = d_roicloud[tid].x,
        Y = d_roicloud[tid].y;

    float P0 = I,
        P1 = I * X,
        P2 = I * X * X,
        P3 = I * X * X * X,
        Q0 = I,
        Q1 = I * Y,
        Q2 = I * Y * Y,
        Q3 = I * Y * Y;

    d_prereduce00[tid] = P0 * Q0;
    d_prereduce01[tid] = P0 * Q1;
    d_prereduce02[tid] = P0 * Q2;
    d_prereduce03[tid] = P0 * Q3;

    d_prereduce10[tid] = P1 * Q0;
    d_prereduce11[tid] = P1 * Q1;
    d_prereduce12[tid] = P1 * Q2;

    d_prereduce20[tid] = P2 * Q0;
    d_prereduce21[tid] = P2 * Q1;

    d_prereduce30[tid] = P3 * Q0;
}

__device__ void hu(
    // out
    gpureal & h1,
    gpureal & h2,
    gpureal & h3,
    gpureal & h4,
    gpureal & h5,
    gpureal & h6,
    gpureal & h7,
    // in
    gpureal nu02,
    gpureal nu03,
    gpureal nu11,
    gpureal nu12,
    gpureal nu20,
    gpureal nu21,
    gpureal nu30)
{
    // Formula: double h1 = NormCentralMom(D, 2, 0) + NormCentralMom(D, 0, 2);
    h1 = nu20 + nu02;

    // Formula: double h2 = pow((NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)), 2) + 4 * (pow(NormCentralMom(D, 1, 1), 2));
    h2 = pow((nu20 - nu02), 2.0) + 4. * pow(nu11, 2.0);

    // Formula: double h3 = pow((NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)), 2) +
    //    pow((3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)), 2);
    h3 = pow((nu30 - 3. * nu12), 2.0) + pow((3. * nu21 - nu03), 2.0);

    // Formula: double h4 = pow((NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) +
    //    pow((NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)), 2);
    h4 = pow((nu30 + nu12), 2.0) + pow((nu21 + nu03), 2.0);

    // Formula: double h5 = (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) *
    //    (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
    //    (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - 3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) +
    //    (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
    //    (pow(3 * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
    h5 = (nu30 - 3. * nu12) *
        (nu30 + nu12) *
        (pow(nu30 + nu12, 2.0) - 3. * pow(nu21 + nu03, 2.0)) +
        (3. * nu21 - nu03) * (nu21 + nu03) *
        (pow(3. * (nu30 + nu12), 2.0) - pow(nu21 + nu03, 2.0));

    // Formula: double h6 = (NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
    //    pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) + (4 * NormCentralMom(D, 1, 1) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
    //        NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3));
    h6 = (nu20 - nu02) * (pow(nu30 + nu12, 2.0) -
        pow(nu21 + nu03, 2.0)) + (4. * nu11 * (nu30 + nu12) *
            nu21 + nu03);

    // Formula: double h7 = (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
    //    3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) - (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
    //    (3 * pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
    h7 = (3. * nu21 - nu03) * (nu30 + nu12) * (pow(nu30 + nu12, 2.0) -
        3 * pow(nu21 + nu03, 2.0)) - (nu30 - 3 * nu12) * (nu21 + nu03) *
        (3 * pow(nu30 + nu12, 2.0) - pow(nu21 + nu03, 2.0));
}

__global__ void kerCentralMomentWeightedAll_snu_cloud (
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
    double* d_prereduce22,
    double* d_prereduce30,
    // in
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    gpureal* origin_x,
    gpureal* origin_y)
{
    int tid = threadIdx.x + blockIdx.x * blockSize;

    if (tid >= cloudlen)
        return;

    float I = d_realintens[tid],
        OX = *origin_x,
        OY = *origin_y,
        X = d_roicloud[tid].x - OX,
        Y = d_roicloud[tid].y - OY;

    float P0 = I,
        P1 = I * X,
        P2 = I * X * X,
        P3 = I * X * X * X,
        Q0 = I,
        Q1 = I * Y,
        Q2 = I * Y * Y,
        Q3 = I * Y * Y;

    d_prereduce00[tid] = P0 * Q0;
    d_prereduce01[tid] = P0 * Q1;
    d_prereduce02[tid] = P0 * Q2;
    d_prereduce03[tid] = P0 * Q3;

    d_prereduce10[tid] = P1 * Q0;
    d_prereduce11[tid] = P1 * Q1;
    d_prereduce12[tid] = P1 * Q2;

    d_prereduce20[tid] = P2 * Q0;
    d_prereduce21[tid] = P2 * Q1;
    d_prereduce22[tid] = P2 * Q2;

    d_prereduce30[tid] = P3 * Q0;
}

__global__ void drvCloud(
    // out
    gpureal* d_result,
    RealPixIntens* d_realintens_buf,
    // in
    const Pixel2* d_roicloud,
    size_t cloudlen,
    const Pixel2* d_roicontour,
    size_t contour_len,
    double* d_prereduce,
    void* d_devreduce_tempstorage,
    size_t& devreduce_tempstorage_szb)
{
    // safety
    int tid = threadIdx.x + blockIdx.x * blockSize;
    if (tid)
        return;

    int status = 1;
    d_result[FEATURE_GEOMOM_STATUS] = status++;

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
    kerRawMomentAll_snu_cloud << < nblo, blockSize >> > (
        // out
        d_pr00,
        d_pr01,
        d_pr02,
        d_pr03,
        d_pr10,
        d_pr11,
        d_pr12,
        d_pr13,
        d_pr20,
        d_pr21,
        d_pr22,
        d_pr23,
        d_pr30,
        d_pr31,
        d_pr32,
        d_pr33,
        // in
        d_roicloud, cloudlen);

 //???    CHECKERR(cudaPeekAtLastError());
//    if (cudaSuccess != cudaDeviceSynchronize())
//        return;
    __syncthreads();
    if (cudaSuccess != cudaGetLastError())
        return;

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //=== device-reduce:

    bool ok;
    ok = sumreduce_cloud (&d_result[GpusideState::RM00], cloudlen, d_pr00, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM01], cloudlen, d_pr01, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb);

    ok = sumreduce_cloud (&d_result[GpusideState::RM10], cloudlen, d_pr10, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM13], cloudlen, d_pr13, d_devreduce_tempstorage, devreduce_tempstorage_szb);

    ok = sumreduce_cloud (&d_result[GpusideState::RM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM22], cloudlen, d_pr22, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM23], cloudlen, d_pr23, d_devreduce_tempstorage, devreduce_tempstorage_szb);

    ok = sumreduce_cloud (&d_result[GpusideState::RM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM31], cloudlen, d_pr31, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM32], cloudlen, d_pr32, d_devreduce_tempstorage, devreduce_tempstorage_szb);
    ok = sumreduce_cloud (&d_result[GpusideState::RM33], cloudlen, d_pr33, d_devreduce_tempstorage, devreduce_tempstorage_szb);

    d_result[FEATURE_GEOMOM_STATUS] = status++;


    //*** origins
    //      drvCalcOrigin(&d_intermed[ORGX], &d_intermed[ORGY], &d_intermed[RM00], &d_intermed[RM01], &d_intermed[RM10])
    d_result[ORGX] = d_result[RM10] / d_result[RM00];
    d_result[ORGY] = d_result[RM01] / d_result[RM00];

    d_result[FEATURE_GEOMOM_STATUS] = status++;


    //*** central

    // prepare lanes of partial totals
/*
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
        * d_pr30 = &d_prereduce[cloudlen * 12];
    double* d_pr31 = &d_prereduce[cloudlen * 13],
        * d_pr32 = &d_prereduce[cloudlen * 14],
        * d_pr33 = &d_prereduce[cloudlen * 15];
        */

    kerCentralMomentAll_snu_cloud <<< nblo, blockSize >>> (
        // out
        d_pr00,
        d_pr01,
        d_pr02,
        d_pr03,
        d_pr10,
        d_pr11,
        d_pr12,
        d_pr13,
        d_pr20,
        d_pr21,
        d_pr22,
        d_pr23,
        d_pr30,
        d_pr31,
        d_pr32,
        d_pr33,
        // in
        d_roicloud, cloudlen, &d_result[ORGX], &d_result[ORGY]);

    //???    CHECKERR(cudaPeekAtLastError());
    //  CHECKERR(cudaDeviceSynchronize());
    __syncthreads();
    // CHECKERR(cudaGetLastError());
    if (cudaSuccess != cudaGetLastError())
        return;

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //=== device-reduce:

    ok = sumreduce_cloud (&d_result[GpusideState::CM00], cloudlen, d_pr00, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM01], cloudlen, d_pr01, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::CM10], cloudlen, d_pr10, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM13], cloudlen, d_pr13, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::CM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM22], cloudlen, d_pr22, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM23], cloudlen, d_pr23, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::CM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM31], cloudlen, d_pr31, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM32], cloudlen, d_pr32, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::CM33], cloudlen, d_pr33, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** normed central
    double k, rm00 = d_result[GpusideState::RM00];
        // 02
    k = (0. + 2.) / 2. + 1.;
    d_result[GpusideState::NU02] = d_result[GpusideState::CM02] / pow(rm00, k);
    // 03
    k = (0. + 3.) / 2. + 1.;
    d_result[GpusideState::NU03] = d_result[GpusideState::CM03] / pow(rm00, k);
    // 11
    k = (1. + 1.) / 2. + 1.;
    d_result[GpusideState::NU11] = d_result[GpusideState::CM11] / pow(rm00, k);
    // 12
    k = (1. + 2.) / 2. + 1.;
    d_result[GpusideState::NU12] = d_result[GpusideState::CM12] / pow(rm00, k);
    // 20
    k = (2. + 0.) / 2. + 1.;
    d_result[GpusideState::NU20] = d_result[GpusideState::CM20] / pow(rm00, k);
    // 21
    k = (2. + 1.) / 2. + 1.;
    d_result[GpusideState::NU21] = d_result[GpusideState::CM21] / pow(rm00, k);
    // 30
    k = (3. + 0.) / 2. + 1.;
    d_result[GpusideState::NU30] = d_result[GpusideState::CM30] / pow(rm00, k);

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** normed raw
    double cm22 = d_result[GpusideState::CM22];
    double nc0 = pow(cm22, 0),
        nc1 = pow(cm22, 1),
        nc2 = pow(cm22, 2),
        nc3 = pow(cm22, 3);

    d_result[GpusideState::W00] = d_result[GpusideState::CM00] / nc0;
    d_result[GpusideState::W01] = d_result[GpusideState::CM01] / nc1;
    d_result[GpusideState::W02] = d_result[GpusideState::CM02] / nc2;
    d_result[GpusideState::W03] = d_result[GpusideState::CM03] / nc3;

    d_result[GpusideState::W10] = d_result[GpusideState::CM10] / nc0;  // 10
    d_result[GpusideState::W11] = d_result[GpusideState::CM11] / nc1;  // 11
    d_result[GpusideState::W12] = d_result[GpusideState::CM12] / nc2;  // 12
    d_result[GpusideState::W13] = d_result[GpusideState::CM13] / nc3;  // 13

    d_result[GpusideState::W20] = d_result[GpusideState::CM20] / nc0;  // 20
    d_result[GpusideState::W21] = d_result[GpusideState::CM21] / nc1;  // 21
    d_result[GpusideState::W22] = d_result[GpusideState::CM22] / nc2;  // 22
    d_result[GpusideState::W23] = d_result[GpusideState::CM23] / nc3;  // 23

    d_result[GpusideState::W30] = d_result[GpusideState::CM30] / nc0;  // 30
    d_result[GpusideState::W31] = d_result[GpusideState::CM31] / nc1;  // 31
    d_result[GpusideState::W32] = d_result[GpusideState::CM32] / nc2;  // 32
    d_result[GpusideState::W33] = d_result[GpusideState::CM33] / nc3;  // 33

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** Hu

    gpureal nu02 = d_result[GpusideState::NU02],
        nu03 = d_result[GpusideState::NU03],
        nu11 = d_result[GpusideState::NU11],
        nu12 = d_result[GpusideState::NU12],
        nu20 = d_result[GpusideState::NU20],
        nu21 = d_result[GpusideState::NU21],
        nu30 = d_result[GpusideState::NU30];
    gpureal h1, h2, h3, h4, h5, h6, h7;
    hu (h1, h2, h3, h4, h5, h6, h7, nu02, nu03, nu11, nu12, nu20, nu21, nu30);
    d_result[H1] = h1;
    d_result[H2] = h2;
    d_result[H3] = h3;
    d_result[H4] = h4;
    d_result[H5] = h5;
    d_result[H6] = h6;
    d_result[H7] = h7;

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** weighted roi cloud
    kerCalcWeightedImage3_cloud << <nblo, blockSize >>> (d_realintens_buf, d_roicloud, cloudlen, d_roicontour, contour_len);

    //cudaError_t ok = cudaDeviceSynchronize();
    __syncthreads();
    //CHECKERR(ok);
    if (cudaSuccess != cudaGetLastError())
        return;

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** weighted raw
    /*
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
    */

    kerRawMomentWeighted_cloud << < nblo, blockSize >> > (
        // out
        d_pr00,
        d_pr01,
        d_pr02,
        d_pr03,
        d_pr10,
        d_pr11,
        d_pr12,
        d_pr20,
        d_pr21,
        d_pr30,
        // in
        d_realintens_buf, d_roicloud, cloudlen);

    //???    CHECKERR(cudaPeekAtLastError());
     // CHECKERR(cudaDeviceSynchronize());
    __syncthreads();
    //CHECKERR(cudaGetLastError());
    if (cudaSuccess != cudaGetLastError())
        return;

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //=== device-reduce:

    ok = sumreduce_cloud (&d_result[GpusideState::WRM00], cloudlen, d_pr00, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WRM01], cloudlen, d_pr01, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WRM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WRM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::WRM10], cloudlen, d_pr10, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WRM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WRM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::WRM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WRM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::WRM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** weighted origin
    //      if (drvCalcOrigin(&d_intermed[WORGX], &d_intermed[WORGY], &d_intermed[WRM00], &d_intermed[WRM01], &d_intermed[WRM10]) == false)
    d_result[WORGX] = d_result[WRM10] / d_result[WRM00];
    d_result[WORGY] = d_result[WRM01] / d_result[WRM00];

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** weighted central moments
    /*
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
        * d_pr22 = &d_prereduce[cloudlen * 10],
        * d_pr30 = &d_prereduce[cloudlen * 12];
            */
    kerCentralMomentWeightedAll_snu_cloud << < nblo, blockSize >> > (
        // out
        d_pr00,
        d_pr01,
        d_pr02,
        d_pr03,
        d_pr10,
        d_pr11,
        d_pr12,
        d_pr20,
        d_pr21,
        d_pr22,
        d_pr30,
        // in
        d_realintens_buf, d_roicloud, cloudlen, &d_result[WORGX], &d_result[WORGY]);

    //???    CHECKERR(cudaPeekAtLastError());
     // CHECKERR(cudaDeviceSynchronize());
    __syncthreads();
    //CHECKERR(cudaGetLastError());
    if (cudaSuccess != cudaGetLastError())
        return;

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //=== device-reduce:

    ok = sumreduce_cloud (&d_result[GpusideState::WCM02], cloudlen, d_pr02, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WCM03], cloudlen, d_pr03, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::WCM11], cloudlen, d_pr11, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WCM12], cloudlen, d_pr12, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::WCM20], cloudlen, d_pr20, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }
    ok = sumreduce_cloud (&d_result[GpusideState::WCM21], cloudlen, d_pr21, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    ok = sumreduce_cloud (&d_result[GpusideState::CM30], cloudlen, d_pr30, d_devreduce_tempstorage, devreduce_tempstorage_szb); if (!ok) { return; }

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** weighted normed central
    double wrm00 = d_result[GpusideState::WRM00];
    // 02
    k = (0. + 2.) / 2. + 1.;
    d_result[GpusideState::WNU02] = d_result[GpusideState::WCM02] / pow(wrm00, k);
    // 03
    k = (0. + 3.) / 2. + 1.;
    d_result[GpusideState::WNU03] = d_result[GpusideState::WCM03] / pow(wrm00, k);
    // 11
    k = (1. + 1.) / 2. + 1.;
    d_result[GpusideState::WNU11] = d_result[GpusideState::WCM11] / pow(wrm00, k);
    // 12
    k = (1. + 2.) / 2. + 1.;
    d_result[GpusideState::WNU12] = d_result[GpusideState::WCM12] / pow(wrm00, k);
    // 20
    k = (2. + 0.) / 2. + 1.;
    d_result[GpusideState::WNU20] = d_result[GpusideState::WCM20] / pow(wrm00, k);
    // 21
    k = (2. + 1.) / 2. + 1.;
    d_result[GpusideState::WNU21] = d_result[GpusideState::WCM21] / pow(wrm00, k);
    // 30
    k = (3. + 0.) / 2. + 1.;
    d_result[GpusideState::WNU30] = d_result[GpusideState::WCM30] / pow(wrm00, k);

    d_result[FEATURE_GEOMOM_STATUS] = status++;

    //*** weighted Hu
    gpureal wnu02 = d_result[GpusideState::WNU02],
        wnu03 = d_result[GpusideState::WNU03],
        wnu11 = d_result[GpusideState::WNU11],
        wnu12 = d_result[GpusideState::WNU12],
        wnu20 = d_result[GpusideState::WNU20],
        wnu21 = d_result[GpusideState::WNU21],
        wnu30 = d_result[GpusideState::WNU30];
    gpureal wh1, wh2, wh3, wh4, wh5, wh6, wh7;
    hu (wh1, wh2, wh3, wh4, wh5, wh6, wh7, wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30);
    d_result[WH1] = wh1;
    d_result[WH2] = wh2;
    d_result[WH3] = wh3;
    d_result[WH4] = wh4;
    d_result[WH5] = wh5;
    d_result[WH6] = wh6;
    d_result[WH7] = wh7;

    //                  d_result[FEATURE_GEOMOM_STATUS] = 0; // we successfully reached the end

}
