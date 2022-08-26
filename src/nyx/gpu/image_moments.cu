#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"

static const int NUM_BLOCKS = 16; // used by reducing kernels
static const int blockSize = 512;   // used by non-reducing kernels

namespace Nyxus
{
    // Image matrices (allocated and initialized with data each time a pending ROI batch is formed)
    extern PixIntens* ImageMatrixBuffer;
    PixIntens* devImageMatrixBuffer = nullptr;

    // ROI contour data
    size_t* devRoiContourIndices = nullptr;
    StatsInt* devRoiContourData = nullptr;

    // Moment calculation result buffer
    const int M_length = (3+10+11)*2;   // origin temps (3), raw moments (10), central moments (11), 2 times
    double* devM = nullptr;
    double* hoM = nullptr;
    // Parallel reduction helper
    int* dev_lastBlockCounter;
}

__device__ bool lastBlock(int* counter) 
{
    __threadfence(); //ensure that partial result is visible by all blocks
    int last = 0;
    if (threadIdx.x == 0)
        last = atomicAdd(counter, 1);
    return __syncthreads_or(last == gridDim.x - 1);
}

__device__ double pow_pos_int (double a, int b)
{
    if (b == 0)
        return 1.0;
    double retval = 1.0;
    for (int i = 0; i < b; i++)
        retval *= a;
    return retval;
}

__global__ void kerSpatialMom(
    double* gOut,
    const PixIntens* gArr, size_t arraySize,
    size_t nx,
    int p,
    int q,
    int* lastBlockCounter   // helper
)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * blockSize;
    const int gridSize = blockSize * gridDim.x;
    double sum = 0;
    for (size_t i = gthIdx; i < arraySize; i += gridSize)
    {
        // Formula: sum += D.yx(y,x) * pow(x, p) * pow(y, q)
        double y = i / nx;
        double x = i % nx;
        double k = pow_pos_int(x, p) * pow_pos_int(y, q); 
        sum += k * (double)gArr[i];
    }

    __shared__ double shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) 
    { 
        // uniform
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx + size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
    if (lastBlock(lastBlockCounter)) 
    {
        shArr[thIdx] = thIdx < gridSize ? gOut[thIdx] : 0;
        __syncthreads();
        for (int size = blockSize / 2; size > 0; size /= 2) 
        { 
            // uniform
            if (thIdx < size)
                shArr[thIdx] += shArr[thIdx + size];
            __syncthreads();
        }
        if (thIdx == 0)
            gOut[0] = shArr[0];
    }
}

__global__ void kerCentralMom(
    double* gOut,
    const PixIntens* gArr, size_t arraySize,
    size_t nx,
    int p,
    int q,
    double origin_x,
    double origin_y,
    int* lastBlockCounter   // helper
)
{
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * blockSize;
    const int gridSize = blockSize * gridDim.x;
    double sum = 0.;
    for (size_t i = gthIdx; i < arraySize; i += gridSize)
    {
        // Formula: sum += D.yx(y,x) * pow((double(x) - originOfX), p) * pow((double(y) - originOfY), q);
        size_t y = i / nx;
        size_t x = i % nx;
        double xc = double(x) - origin_x,
            yc = double(y) - origin_y;
        double k = pow_pos_int(xc, p) * pow_pos_int(yc, q); 
        sum += k * (double)gArr[i];
    }

    __shared__ double shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) 
    { 
        //uniform
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx + size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
    if (lastBlock(lastBlockCounter)) {
        shArr[thIdx] = thIdx < gridSize ? gOut[thIdx] : 0;
        __syncthreads();
        for (int size = blockSize / 2; size > 0; size /= 2) 
        { 
            //uniform
            if (thIdx < size)
                shArr[thIdx] += shArr[thIdx + size];
            __syncthreads();
        }
        if (thIdx == 0)
            gOut[0] = shArr[0];
    }
}

bool ImageMomentsFeature_spatial_moment4 (
    int p, int q,
    PixIntens* devIMbuffer, size_t imOffset, size_t nx, size_t ny,
    double* dev_M,   // result
    int* dev_lastBlockCounter)  // helper
{
    size_t len = nx * ny;
    PixIntens* devI = devIMbuffer + imOffset;

    cudaError_t ok;

    // 'dev_lastBlockCounter' is assumed to be zeroed externally:
    //      auto ok = cudaMemset(dev_lastBlockCounter, 0, sizeof(int));
    //      CHECKERR(ok);

    kerSpatialMom <<< NUM_BLOCKS, blockSize >>> (dev_M, devI, len, nx, p, q, dev_lastBlockCounter);

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

    // Leaving the result on device-side. It's supposed to be pulled externally:
    //      auto ok = cudaMemcpy(&hoM, devM, result_vector_length*sizeof(hoM[0]), cudaMemcpyDeviceToHost);
    //      CHECKERR(ok);

    return true;
}

bool ImageMomentsFeature_central_moment4 (
    //---   double& m, // output
    int p, int q,
    PixIntens* devIMbuffer, size_t imOffset, size_t nx, size_t ny,
    double originX,
    double originY,
    double* devM,   // helper #1
    int* dev_lastBlockCounter)  // helper #2
{
    size_t len = nx * ny;
    PixIntens* devI = devIMbuffer + imOffset;

    // 'dev_lastBlockCounter' is assumed to be zeroed externally:
    //      auto ok = cudaMemset(dev_lastBlockCounter, 0, sizeof(int));
    //      CHECKERR(ok);

    cudaError_t ok;
    kerCentralMom <<< NUM_BLOCKS, blockSize >>> (devM, devI, len, nx, p, q, originX, originY, dev_lastBlockCounter);

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

    // Leaving the result on device-side. It's supposed to be pulled externally:
    //      auto ok = cudaMemcpy(&hoM, devM, result_vector_length*sizeof(hoM[0]), cudaMemcpyDeviceToHost);
    //      CHECKERR(ok);

    return true;
}

bool ImageMomentsFeature_calcOrigins4(
    double& originOfX, double& originOfY,   // output
    PixIntens* devImageMatrixBuffer, size_t imOffset, size_t nx, size_t ny,
    double* devM, double* hoM,      // device- and host- side result vectors at least 3 items long
    int* dev_lastBlockCounter)      // reduction helper at least 3 items long
{
    //==== Calculate moments 00, 01, and 10 leaving the result on GPU-side

    if (ImageMomentsFeature_spatial_moment4(
        0, 0,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny, // image matrix data
        devM+0, // result
        dev_lastBlockCounter+0  // helper
        ) == false)
        return false;
    if (ImageMomentsFeature_spatial_moment4(
        0, 1,   // p,q
        devImageMatrixBuffer, imOffset, nx, ny, // image matrix data
        devM+1, // result
        dev_lastBlockCounter+1  // helper
        ) == false)
        return false;
    if (ImageMomentsFeature_spatial_moment4(
        1, 0,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny, // image matrix data
        devM+2, // result
        dev_lastBlockCounter+2  // helper
        ) == false)
        return false;

    //==== Retrieve the moment values on the host-side

    auto ok = cudaMemcpy (hoM, devM, 3 * sizeof(hoM[0]), cudaMemcpyDeviceToHost);
    CHECKERR(ok)

    double m00 = hoM[0], 
        m01 = hoM[1], 
        m10 = hoM[2];

    //==== Calculate the origin

    originOfX = m10 / m00;
    originOfY = m01 / m00;

    return true;
}

bool ImageMomentsFeature_calcSpatialMoments4(
    PixIntens* devImageMatrixBuffer, size_t imOffset, size_t nx, size_t ny,  // data
    double* devM, int* dev_lastBlockCounter // reduction helpers
)
{
    if (ImageMomentsFeature_spatial_moment4(
        0,0,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+0, dev_lastBlockCounter+0) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        0,1,  // p, q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+1, dev_lastBlockCounter+1) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        0, 2,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+2, dev_lastBlockCounter+2) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        0, 3,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+3, dev_lastBlockCounter+3) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        1, 0,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+4, dev_lastBlockCounter+4) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        1, 1,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+5, dev_lastBlockCounter+5) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        1, 2,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+6, dev_lastBlockCounter+6) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        2, 0,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+7, dev_lastBlockCounter+7) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        2, 1,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+8, dev_lastBlockCounter+8) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment4(
        3, 0,  // p,q
        devImageMatrixBuffer, imOffset, nx, ny,
        devM+9, dev_lastBlockCounter+9) == false)
        return false;

    return true;
}

bool ImageMomentsFeature_calcCentralMoments4(
    PixIntens* devImageMatrixBuffer, size_t imOffset, size_t nx, size_t ny, double originX, double originY, // image data
    double* devM,   // results
    int* dev_lastBlockCounter) // reduction helper
{
    if (!ImageMomentsFeature_central_moment4(
        0, 0,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+0, dev_lastBlockCounter+0))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        0, 1,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+1, dev_lastBlockCounter+1))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        0, 2,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+2, dev_lastBlockCounter+2))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        0, 3,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+3, dev_lastBlockCounter+3))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        1, 0,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+4, dev_lastBlockCounter+4))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        1, 1,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+5, dev_lastBlockCounter+5))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        1, 2,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+6, dev_lastBlockCounter+6))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        2, 0,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+7, dev_lastBlockCounter+7))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        2, 1,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+8, dev_lastBlockCounter+8))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        2, 2,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+9, dev_lastBlockCounter+9))
        return false;

    if (!ImageMomentsFeature_central_moment4(
        3, 0,
        devImageMatrixBuffer, imOffset, nx, ny, originX, originY, 
        devM+10, dev_lastBlockCounter+10))
        return false;

    return true;
}

bool ImageMomentsFeature_calcNormCentralMoments3(
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,   // output
    double cm02, double cm03, double cm11, double cm12, double cm20, double cm21, double cm30,
    double m00) 
{
    // Formula:
    //  double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    //  double retval = CentralMom(D, p, q) / pow(Moment(D, 0, 0), temp);

    double k;
    // 02
    k = (0. + 2.) / 2. + 1.;
    nu02 = cm02 / pow(m00, k);
    // 03
    k = (0. + 3.) / 2. + 1.;
    nu03 = cm03 / pow(m00, k);
    // 11
    k = (1. + 1.) / 2. + 1.;
    nu11 = cm11 / pow(m00, k);
    // 12
    k = (1. + 2.) / 2. + 1.;
    nu12 = cm12 / pow(m00, k);
    // 20
    k = (2. + 0.) / 2. + 1.;
    nu20 = cm20 / pow(m00, k);
    // 21
    k = (2. + 1.) / 2. + 1.;
    nu21 = cm21 / pow(m00, k);
    // 30
    k = (3. + 0.) / 2. + 1.;
    nu30 = cm30 / pow(m00, k);

    return true;
}

bool ImageMomentsFeature_calcNormSpatialMoments3(
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w20, double& w30,   // output
    double cm00, double cm01, double cm02, double cm03, double cm10, double cm20, double cm30,
    double cm22) 
{
    // Formula: 
    //  double stddev = CentralMom(D, 2, 2);
    //  int w = std::max(q, p);
    //  double normCoef = pow(stddev, w);
    //  double retval = CentralMom(D, p, q) / normCoef;

    int w;
    double normCoef;
    // 00
    w = 0;
    normCoef = pow(cm22, w);
    w00 = cm00 / normCoef;
    // 01
    w = 1;
    normCoef = pow(cm22, w);
    w01 = cm01 / normCoef;
    // 02
    w = 2;
    normCoef = pow(cm22, w);
    w02 = cm02 / normCoef;
    // 03
    w = 3;
    normCoef = pow(cm22, w);
    w03 = cm03 / normCoef;
    // 10
    w = 1;
    normCoef = pow(cm22, w);
    w10 = cm10 / normCoef;
    // 20
    w = 2;
    normCoef = pow(cm22, w);
    w20 = cm20 / normCoef;
    // 30
    w = 3;
    normCoef = pow(cm22, w);
    w30 = cm30 / normCoef;

    return true;
}

bool ImageMomentsFeature_calcHuInvariants3(
    double& h1, double& h2, double& h3, double& h4, double& h5, double& h6, double& h7,   // output
    double nu02, double nu03, double nu11, double nu12, double nu20, double nu21, double nu30) // reduction helpers
{
    // Formula: double h1 = NormCentralMom(D, 2, 0) + NormCentralMom(D, 0, 2);
    h1 = nu20 + nu02;

    // Formula: double h2 = pow((NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)), 2) + 4 * (pow(NormCentralMom(D, 1, 1), 2));
    h2 = pow((nu20 - nu02), 2) + 4. * pow(nu11, 2);

    // Formula: double h3 = pow((NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)), 2) +
    //    pow((3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)), 2);
    h3 = pow((nu30 - 3. * nu12), 2) + pow((3. * nu21 - nu03), 2);

    // Formula: double h4 = pow((NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) +
    //    pow((NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)), 2);
    h4 = pow((nu30 + nu12), 2) + pow((nu21 + nu03), 2);

    // Formula: double h5 = (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) *
    //    (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
    //    (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - 3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) +
    //    (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
    //    (pow(3 * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
    h5 = (nu30 - 3. * nu12) *
        (nu30 + nu12) *
        (pow(nu30 + nu12, 2) - 3. * pow(nu21 + nu03, 2)) +
        (3. * nu21 - nu03) * (nu21 + nu03) *
        (pow(3. * (nu30 + nu12), 2) - pow(nu21 + nu03, 2));

    // Formula: double h6 = (NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
    //    pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) + (4 * NormCentralMom(D, 1, 1) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
    //        NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3));
    h6 = (nu20 - nu02) * (pow(nu30 + nu12, 2) -
        pow(nu21 + nu03, 2)) + (4. * nu11 * (nu30 + nu12) *
            nu21 + nu03);

    // Formula: double h7 = (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
    //    3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) - (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
    //    (3 * pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
    h7 = (3. * nu21 - nu03) * (nu30 + nu12) * (pow(nu30 + nu12, 2) -
        3 * pow(nu21 + nu03, 2)) - (nu30 - 3 * nu12) * (nu21 + nu03) *
        (3 * pow(nu30 + nu12, 2) - pow(nu21 + nu03, 2));

    return true;
}

__global__ void kerCalcWeightedImage3(
    PixIntens* devImageMatrixBuffer, size_t imOffset,
    size_t len,
    size_t width,
    size_t aabb_min_x,
    size_t aabb_min_y,
    size_t roi_index,
    size_t* dev_Indices,
    StatsInt* dev_Contour_Data)
{
    // Pixel index
    int pxIdx = threadIdx.x + blockIdx.x * blockSize;
    if (pxIdx >= len)
        return;

    PixIntens* dev_Img = devImageMatrixBuffer + imOffset;

    PixIntens pi = dev_Img[pxIdx];
    if (pi == 0)
        return;

    StatsInt pxX = pxIdx % width + aabb_min_x,
        pxY = pxIdx / width + aabb_min_y;

    size_t minDistPow2 = 999999999;

    size_t contourBase = dev_Indices[roi_index];
    StatsInt contourLen = dev_Contour_Data[contourBase];
    for (size_t i = 0; i < contourLen; i++)
    {
        size_t offs = 1 + i * 2;
        StatsInt x = dev_Contour_Data[offs],
            y = dev_Contour_Data[offs + 1];
        size_t dist = (x - pxX) * (x - pxX) + (y - pxY) * (y - pxY);
        if (minDistPow2 > dist)
            minDistPow2 = dist;
    }

    double dist = sqrt((double)minDistPow2);
    const double epsilon = 0.1;
    PixIntens wpi = pi / (dist + epsilon) + 0.5/*rounding*/;
    dev_Img[pxIdx] = wpi;
}

bool ImageMomentsFeature_calcWeightedMatrix3(
    PixIntens* devImageMatrixBuffer, size_t imOffset,    // input & output
    size_t nx,
    size_t ny,
    size_t roi_index,
    StatsInt aabb_min_x,
    StatsInt aabb_min_y)
{
    size_t len = nx * ny;
    int nb = whole_chunks2(len, blockSize);
    kerCalcWeightedImage3 <<< nb, blockSize >>> (devImageMatrixBuffer, imOffset, len, nx, aabb_min_x, aabb_min_y, roi_index, Nyxus::devRoiContourIndices, Nyxus::devRoiContourData);

    cudaError_t ok = cudaDeviceSynchronize();
    CHECKERR(ok);

    ok = cudaGetLastError();
    CHECKERR(ok);

    return true;
}

bool ImageMomentsFeature_calculate3(
    double& m00, double& m01, double& m02, double& m03, double& m10, double& m11, double& m12, double& m20, double& m21, double& m30,   // spatial moments
    double& cm02, double& cm03, double& cm11, double& cm12, double& cm20, double& cm21, double& cm30,   // central moments
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,    // normalized central moments
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w20, double& w30,   // normalized spatial moments
    double& hm1, double& hm2, double& hm3, double& hm4, double& hm5, double& hm6, double& hm7,  // Hu moments
    double& wm00, double& wm01, double& wm02, double& wm03, double& wm10, double& wm11, double& wm12, double& wm20, double& wm21, double& wm30,   // weighted spatial moments
    double& wcm02, double& wcm03, double& wcm11, double& wcm12, double& wcm20, double& wcm21, double& wcm30,   // weighted central moments
    double& whm1, double& whm2, double& whm3, double& whm4, double& whm5, double& whm6, double& whm7,   // weighted Hum moments
    size_t imOffset,
    size_t roi_index,
    StatsInt aabb_min_x,
    StatsInt aabb_min_y, 
    StatsInt width, 
    StatsInt height)
{
    //==== Initialize reduction helpers (a helper per calculated value)
    int n_values = (3 + 10 + 11) * 2; // 3 for origin calculation, 10 spatial moments, 11 central moments, 2 times
    auto ok = cudaMemset(Nyxus::dev_lastBlockCounter, 0, n_values * sizeof(Nyxus::dev_lastBlockCounter[0]));
    CHECKERR(ok);

    //==== Origin
    size_t nx = width,
        ny = height;

    double originX, originY;
    
    bool good = ImageMomentsFeature_calcOrigins4(
        originX, originY,   // output
        Nyxus::devImageMatrixBuffer, imOffset, nx, ny,
        Nyxus::devM, Nyxus::hoM,    // devM [0-2] 
        Nyxus::dev_lastBlockCounter    // helper
    );

    if (!good)
        return false;

    //==== Spatial moments
    
    good = ImageMomentsFeature_calcSpatialMoments4(
        Nyxus::devImageMatrixBuffer, imOffset, nx, ny,   // image data
        Nyxus::devM+3,   // output devM[3-12] -> m00, m01, m02, m03, m10, m11, m12, m20, m21, m30
        Nyxus::dev_lastBlockCounter+3);    // helper
    if (!good)
        return false;

    //==== Central moments

    good = ImageMomentsFeature_calcCentralMoments4(
        Nyxus::devImageMatrixBuffer, imOffset, nx, ny, originX, originY,   // image data
        Nyxus::devM+3+10,   // devM[13-23] -> cm00, cm01, cm02, cm03, cm10, cm11, cm12, cm20, cm21, cm22, cm30,   // output
        Nyxus::dev_lastBlockCounter+3+10);    // helpers

    if (!good)
        return false;

    //==== Retrieve results (raw and central moments)
    ok = cudaMemcpy(Nyxus::hoM, Nyxus::devM, sizeof(Nyxus::hoM[0])*Nyxus::M_length, cudaMemcpyDeviceToHost);
    CHECKERR(ok);
    m00 = Nyxus::hoM[3]; m01 = Nyxus::hoM[4]; m02 = Nyxus::hoM[5]; m03 = Nyxus::hoM[6]; 
        m10 = Nyxus::hoM[7]; m11 = Nyxus::hoM[8]; m12 = Nyxus::hoM[9]; m20 = Nyxus::hoM[10]; m21 = Nyxus::hoM[11]; 
        m30 = Nyxus::hoM[12];
    double cm00 = Nyxus::hoM[13], 
        cm01 = Nyxus::hoM[14]; 
    cm02 = Nyxus::hoM[15]; 
    cm03 = Nyxus::hoM[16]; 
    double cm10 = Nyxus::hoM[17];
    cm11 = Nyxus::hoM[18]; 
    cm12 = Nyxus::hoM[19]; 
    cm20 = Nyxus::hoM[20]; 
    cm21 = Nyxus::hoM[21]; 
    double cm22 = Nyxus::hoM[22]; 
    cm30 = Nyxus::hoM[23];

    //==== Norm central moments
    good = ImageMomentsFeature_calcNormCentralMoments3(
        nu02, nu03, nu11, nu12, nu20, nu21, nu30,   // output
        cm02, cm03, cm11, cm12, cm20, cm21, cm30,   // central moments
        m00    // spatial moment
    );
    if (!good)
        return false;

    //==== Norm spatial moments
    good = ImageMomentsFeature_calcNormSpatialMoments3(
        w00, w01, w02, w03, w10, w20, w30,  // output
        cm00, cm01, cm02, cm03, cm10, cm20, cm30, cm22);
    if (!good)
        return false;

    //==== Hu insvariants
    good = ImageMomentsFeature_calcHuInvariants3(
        hm1, hm2, hm3, hm4, hm5, hm6, hm7,  // output
        nu02, nu03, nu11, nu12, nu20, nu21, nu30);
    if (!good)
        return false;

    //==== Weighted moments

    // Use the image matrix and cached ROIs' contours on the GPU side to calculate weights
    good = ImageMomentsFeature_calcWeightedMatrix3(
        Nyxus::devImageMatrixBuffer, imOffset,   // input & output
        nx, ny,
        roi_index,
        aabb_min_x,
        aabb_min_y);
    if (!good)
        return false;

    // Calculate the origin
    good = ImageMomentsFeature_calcOrigins4(
        originX, originY,   // output
        Nyxus::devImageMatrixBuffer, imOffset, nx, ny,
        Nyxus::devM+3+10+11, Nyxus::hoM+3+10+11,    // devM [24-26] (3 items)
        Nyxus::dev_lastBlockCounter+3+10+11    // helper
    );

    if (!good)
        return false; 

    good = ImageMomentsFeature_calcSpatialMoments4(
        Nyxus::devImageMatrixBuffer, imOffset, nx, ny,   // image data
        Nyxus::devM +3+10+11+3,   // -> wm00, wm01, wm02, wm03, wm10, wm11, wm12, wm20, wm21, wm30 (10 items)
        Nyxus::dev_lastBlockCounter +3+10+11+3    // helpers
    );

    if (!good)
        return false;

    good = ImageMomentsFeature_calcCentralMoments4(
        Nyxus::devImageMatrixBuffer, imOffset, nx, ny, originX, originY,   // image data
        Nyxus::devM +3+10+11+3+10,   // devM[13-23] -> wcm00, wcm01, wcm02, wcm03, wcm10, wcm11, wcm12, wcm20, wcm21, wcm22, wcm30 (11 items)
        Nyxus::dev_lastBlockCounter +3+10+11+3+10);    // helpers

    if (!good)
        return false;

    //==== Retrieve results (weighted raw and central moments)
    size_t base = 3/*origin temps*/ + 10/*spatial*/ + 11/*central*/ + 3/*origin temps #2*/;
    ok = cudaMemcpy(Nyxus::hoM, Nyxus::devM, sizeof(Nyxus::hoM[0]) * Nyxus::M_length, cudaMemcpyDeviceToHost);
    CHECKERR(ok);
    wm00 = Nyxus::hoM[base];
        wm01 = Nyxus::hoM[base+1]; 
        wm02 = Nyxus::hoM[base+2]; 
        wm03 = Nyxus::hoM[base+3];
        wm10 = Nyxus::hoM[base+4]; 
        wm11 = Nyxus::hoM[base+5]; 
        wm12 = Nyxus::hoM[base+6]; 
        wm20 = Nyxus::hoM[base+7]; 
        wm21 = Nyxus::hoM[base+8];
        wm30 = Nyxus::hoM[base+9];
        cm00 = Nyxus::hoM[base+10]; 
        cm01 = Nyxus::hoM[base+11]; 
        cm02 = Nyxus::hoM[base+12]; 
        cm03 = Nyxus::hoM[base+13];
        cm10 = Nyxus::hoM[base+14]; 
        cm11 = Nyxus::hoM[base+15]; 
        cm12 = Nyxus::hoM[base+16]; 
        cm20 = Nyxus::hoM[base+17]; 
        cm21 = Nyxus::hoM[base+18]; 
        cm22 = Nyxus::hoM[base+19];
        cm30 = Nyxus::hoM[base+20];

    //calcWeightedHuInvariants(W);
    // --1
    double wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30;
    good = ImageMomentsFeature_calcNormCentralMoments3(
        wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30,   // output
        wcm02, wcm03, wcm11, wcm12, wcm20, wcm21, wcm30,   // weighted central moments
        m00    // spatial moment
    );
    if (!good)
        return false;
    // --2
    good = ImageMomentsFeature_calcHuInvariants3(
        whm1, whm2, whm3, whm4, whm5, whm6, whm7,  // output
        wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30);
    if (!good)
        return false;

    return true;
}

bool send_contours_to_gpu(const std::vector<size_t>& Indices, const std::vector< StatsInt>& ContourData)
{
    size_t szb = Indices.size() * sizeof(Nyxus::devRoiContourIndices[0]);
    auto ok = cudaMalloc(reinterpret_cast<void**> (&Nyxus::devRoiContourIndices), szb);
    CHECKERR(ok);

    const size_t* hoIndices = Indices.data();
    ok = cudaMemcpy(Nyxus::devRoiContourIndices, hoIndices, szb, cudaMemcpyHostToDevice);
    CHECKERR(ok);

    szb = ContourData.size() * sizeof(Nyxus::devRoiContourData[0]);
    ok = cudaMalloc(reinterpret_cast<void**> (&Nyxus::devRoiContourData), szb);
    CHECKERR(ok);

    const StatsInt* hoContourData = ContourData.data();
    ok = cudaMemcpy(Nyxus::devRoiContourData, hoContourData, szb, cudaMemcpyHostToDevice);
    CHECKERR(ok);

    return true;
}

bool free_contour_data_on_gpu()
{
    CHECKERR(cudaFree(Nyxus::devRoiContourIndices));
    CHECKERR(cudaFree(Nyxus::devRoiContourData));
    return true;
}

bool send_imgmatrices_to_gpu (PixIntens* hoImageMatrixBuffer, size_t buf_len)
{
    // Reserve the image matrix buffer
    size_t szb = buf_len * sizeof(Nyxus::devImageMatrixBuffer[0]);
    auto ok = cudaMalloc(reinterpret_cast<void**> (&Nyxus::devImageMatrixBuffer), szb);
    CHECKERR(ok);

    // Transfer the image matrix data
    ok = cudaMemcpy(Nyxus::devImageMatrixBuffer, hoImageMatrixBuffer, szb, cudaMemcpyHostToDevice);
    CHECKERR(ok);
    
    // Reserve the GPU-side moment calculation result buffer
    ok = cudaMalloc(reinterpret_cast<void**> (&Nyxus::devM), Nyxus::M_length*sizeof(double));
    CHECKERR(ok);

    // Reserve the host-side moment calculation result buffer
    Nyxus::hoM = new double[Nyxus::M_length];

    // Reserve the reduction helpers buffer
    ok = cudaMalloc((void**)&Nyxus::dev_lastBlockCounter, Nyxus::M_length*sizeof(int));
    CHECKERR(ok);

    return true;
}

bool free_imgmatrices_on_gpu()
{
    CHECKERR(cudaFree(Nyxus::devImageMatrixBuffer));

    auto ok = cudaFree(Nyxus::dev_lastBlockCounter);
    CHECKERR(ok);

    ok = cudaFree(Nyxus::devM);
    if (ok != cudaSuccess)
        return false;

    delete Nyxus::hoM;

    return true;
}


