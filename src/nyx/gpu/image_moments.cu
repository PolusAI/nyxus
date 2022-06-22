#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"

static const int blockSize = 512;

size_t* devIndices = nullptr;
StatsInt* devContourData = nullptr;

__device__ bool lastBlock(int* counter) {
    __threadfence(); //ensure that partial result is visible by all blocks
    int last = 0;
    if (threadIdx.x == 0)
        last = atomicAdd(counter, 1);
    return __syncthreads_or(last == gridDim.x - 1);
}

__global__ void kerSpatialMom (
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
    double q__ = (double)q, p__ = (double)p;
    for (size_t i = gthIdx; i < arraySize; i += gridSize)
    {
        // Formula: sum += D.yx(y,x) * pow(x, p) * pow(y, q)
        double y = i / nx;
        double x = i % nx;
        double k = pow(x, p__) * pow(y, q__);
        sum += k * (double)gArr[i];
    }

    __shared__ double shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx + size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
    if (lastBlock(lastBlockCounter)) {
        shArr[thIdx] = thIdx < gridSize ? gOut[thIdx] : 0;
        __syncthreads();
        for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
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
    double q_ = (double)q, p_ = (double)p;
    for (size_t i = gthIdx; i < arraySize; i += gridSize)
    {
        // Formula: sum += D.yx(y,x) * pow((double(x) - originOfX), p) * pow((double(y) - originOfY), q);
        size_t y = i / nx;
        size_t x = i % nx;
        double xc = double(x) - origin_x, 
            yc = double(y) - origin_y;
        double k = pow(xc, p_) * pow(yc, q_);
        sum += k * (double)gArr[i];
    }

    __shared__ double shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx + size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
    if (lastBlock(lastBlockCounter)) {
        shArr[thIdx] = thIdx < gridSize ? gOut[thIdx] : 0;
        __syncthreads();
        for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
            if (thIdx < size)
                shArr[thIdx] += shArr[thIdx + size];
            __syncthreads();
        }
        if (thIdx == 0)
            gOut[0] = shArr[0];
    }
}

bool ImageMomentsFeature_spatial_moment (
    double & m, // output
    int p, int q, 
    PixIntens* devI, size_t nx, size_t ny,
    double* devM,   // helper #1
    int* dev_lastBlockCounter)  // helper #2
{
    size_t len = nx * ny;

    cudaError_t ok = cudaMemset(dev_lastBlockCounter, 0, sizeof(int));
    CHECKERR(ok);

    int nb = 8; // a modest number of blocks 
    kerSpatialMom <<< nb, blockSize >>> (devM, devI, len, nx, p, q, dev_lastBlockCounter);

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

    // Pull the moment
    ok = cudaMemcpy(&m, devM, sizeof(m), cudaMemcpyDeviceToHost);
    CHECKERR(ok);

    return true;
}

bool ImageMomentsFeature_central_moment(
    double& m, // output
    int p, int q,
    PixIntens* devI, size_t nx, size_t ny,
    double originX, 
    double originY,
    double* devM,   // helper #1
    int* dev_lastBlockCounter)  // helper #2
{
    size_t len = nx * ny;

    // -- kernel v2
    cudaError_t ok = cudaMemset(dev_lastBlockCounter, 0, sizeof(int));
    CHECKERR(ok);

    int nb = 8; 
    kerCentralMom <<< nb, blockSize >>> (devM, devI, len, nx, p, q, originX, originY, dev_lastBlockCounter);

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

    // Pull the moment
    ok = cudaMemcpy(&m, devM, sizeof(m), cudaMemcpyDeviceToHost);
    if (ok != cudaSuccess)
    {
        std::cerr << __FILE__ << ":" << __LINE__ << " cuda error code " << ok << ": " << cudaGetErrorString(ok) << "\n";
        return false;
    }

    return true;
}

bool ImageMomentsFeature_calcOrigins (
    double& originOfX, double& originOfY,   // output
    PixIntens* devI, size_t nx, size_t ny,
    double* devM,   // helper #1
    int* dev_lastBlockCounter)
{
    // calc orgins
    //double m00 = Moment(D, 0, 0), m10 = Moment(D, 1, 0), m01 = Moment(D, 0, 1);

    double m00;
    if (ImageMomentsFeature_spatial_moment(
        m00, // output
        0,0,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;
    double m01;
    if (ImageMomentsFeature_spatial_moment(
        m01, // output
        0,1,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;
    double m10;
    if (ImageMomentsFeature_spatial_moment(
        m10, // output
        1,0,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    originOfX = m10 / m00;
    originOfY = m01 / m00;
    return true;
}

bool ImageMomentsFeature_calcSpatialMoments (
    double& m00, double& m01, double& m02, double& m03, double& m10, double& m11, double& m12, double& m20, double& m21, double& m30,   // output
    PixIntens* devI, size_t nx, size_t ny,  // data
    double* devM, int* dev_lastBlockCounter // reduction helpers
)
{
    if (ImageMomentsFeature_spatial_moment(
        m00, // output
        0,0,  // p,q
        devI, nx, ny, 
        devM, dev_lastBlockCounter) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m01, // output
        0,1,  // p, q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m02, // output
        0,2,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m03, // output
        0,3,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m10, // output
        1,0,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m11, // output
        1,1,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m12, // output
        1,2,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m20, // output
        2,0,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m21, // output
        2,1,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    if (ImageMomentsFeature_spatial_moment(
        m30, // output
        3,0,  // p,q
        devI, nx, ny,
        devM,   // helper #1
        dev_lastBlockCounter // helper #2
    ) == false)
        return false;

    return true;
}

bool ImageMomentsFeature_calcCentralMoments(
    double& mu00, double& mu01, double& mu02, double& mu03, double& mu10, double& mu11, double& mu12, double& mu20, double& mu21, double& mu22, double& mu30,   // output
    PixIntens* devI, size_t nx, size_t ny,  double originX, double originY, // data
    double* devM, int* dev_lastBlockCounter) // reduction helpers
{
    if (!ImageMomentsFeature_central_moment(
        mu00, 0, 0,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (!ImageMomentsFeature_central_moment(
        mu01, 0, 1,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (! ImageMomentsFeature_central_moment(
        mu02, 0,2, 
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (! ImageMomentsFeature_central_moment(
        mu03, 0,3,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (!ImageMomentsFeature_central_moment(
        mu10, 1,0,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (! ImageMomentsFeature_central_moment(
        mu11, 1,1,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (! ImageMomentsFeature_central_moment(
        mu12, 1,2,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (! ImageMomentsFeature_central_moment(
        mu20, 2,0,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (! ImageMomentsFeature_central_moment(
        mu21, 2,1,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (!ImageMomentsFeature_central_moment(
        mu22, 2, 2,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    if (! ImageMomentsFeature_central_moment(
        mu30, 3,0,
        devI, nx, ny, originX, originY, devM, dev_lastBlockCounter))
        return false;

    return true;
}

bool ImageMomentsFeature_calcNormCentralMoments(
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,   // output
    PixIntens* devI, size_t nx, size_t ny,  // data
    double cm02, double cm03, double cm11, double cm12, double cm20, double cm21, double cm30,
    double m00, 
    double* devM, int* dev_lastBlockCounter) // reduction helpers
{
    // Formula:
    //  double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    //  double retval = CentralMom(D, p, q) / pow(Moment(D, 0, 0), temp);

    double k;
    // 02
    k = (0. + 2.) / 2. + 1. ;
    nu02 = cm02 / pow(m00, k);
    // 03
    k = (0. + 3.) / 2. + 1. ;
    nu03 = cm03 / pow(m00, k);
    // 11
    k = (1. + 1.) / 2. + 1. ;
    nu11 = cm11 / pow(m00, k);
    // 12
    k = (1. + 2.) / 2. + 1. ;
    nu12 = cm12 / pow(m00, k);
    // 20
    k = (2. + 0.) / 2. + 1. ;
    nu20 = cm20 / pow(m00, k);
    // 21
    k = (2. + 1.) / 2. + 1. ;
    nu21 = cm21 / pow(m00, k);
    // 30
    k = (3. + 0.) / 2. + 1. ;
    nu30 = cm30 / pow(m00, k);

    return true;
}

bool ImageMomentsFeature_calcNormSpatialMoments(
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w20, double& w30,   // output
    PixIntens* devI, size_t nx, size_t ny,  // data
    double cm00, double cm01, double cm02, double cm03, double cm10, double cm20, double cm30,
    double cm22, 
    double* devM, int* dev_lastBlockCounter) // reduction helpers
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

bool ImageMomentsFeature_calcHuInvariants(
    double& h1, double& h2, double& h3, double& h4, double& h5, double& h6, double& h7,   // output
    PixIntens* devI, size_t nx, size_t ny,  // data
    double nu02, double nu03, double nu11, double nu12, double nu20, double nu21, double nu30,
    double* devM, int* dev_lastBlockCounter) // reduction helpers
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

bool ImageMomentsFeature_calculate (
    double& m00, double& m01, double& m02, double& m03, double& m10, double& m11, double& m12, double& m20, double& m21, double& m30,   // spatial moments
    double& cm02, double& cm03, double& cm11, double& cm12, double& cm20, double& cm21, double& cm30,   // central moments
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,    // normalized central moments
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w20, double& w30,   // normalized spatial moments
    double& hm1, double& hm2, double& hm3, double& hm4, double& hm5, double& hm6, double& hm7,  // Hu moments
    const ImageMatrix& Im, 
    const ImageMatrix& weighted_Im,
    double& wm00, double& wm01, double& wm02, double& wm03, double& wm10, double& wm11, double& wm12, double& wm20, double& wm21, double& wm30,   // weighted spatial moments
    double& wcm02, double& wcm03, double& wcm11, double& wcm12, double& wcm20, double& wcm21, double& wcm30,   // weighted central moments
    double& whm1, double& whm2, double& whm3, double& whm4, double& whm5, double& whm6, double& whm7)    // weighted Hum moments
{
	const pixData& I = Im.ReadablePixels();
	const PixIntens* hoI = I.data();

    // Send the image matrix to device
    size_t szb = I.size() * sizeof(hoI[0]);
    PixIntens* devI = nullptr;
	auto ok = cudaMalloc (reinterpret_cast<void**> (&devI), szb); 
	if (ok != cudaSuccess) 
		return false;

	ok = cudaMemcpy (devI, hoI, szb, cudaMemcpyHostToDevice);
	if (ok != cudaSuccess)
		return false;

    // Prepare the moment calculation result
    double* devM = nullptr;
    ok = cudaMalloc(reinterpret_cast<void**> (&devM), sizeof(devM[0]));
    if (ok != cudaSuccess)
        return false;

    // Prepare the reduction helper
    int* dev_lastBlockCounter;
    ok = cudaMalloc((void**)&dev_lastBlockCounter, sizeof(int));
    CHECKERR(ok);

    // Calculate the origin
    size_t nx = I.width(), 
        ny = I.height();

    double originX, originY;
    bool good = ImageMomentsFeature_calcOrigins(
        originX, originY,   // output
        devI, nx, ny,
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //==== Spatial moments
    good = ImageMomentsFeature_calcSpatialMoments(
        m00, m01, m02, m03, m10, m11, m12, m20, m21, m30,   // output
        devI, nx, ny,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //==== Central moments
    double cm00, cm01, cm10, cm22;
    good = ImageMomentsFeature_calcCentralMoments(
        cm00, cm01, cm02, cm03, cm10, cm11, cm12, cm20, cm21, cm22, cm30,   // output
        devI, nx, ny, originX, originY,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //==== Norm central moments
    good = ImageMomentsFeature_calcNormCentralMoments(
        nu02, nu03, nu11, nu12, nu20, nu21, nu30,   // output
        devI, nx, ny,   // image data
        cm02, cm03, cm11, cm12, cm20, cm21, cm30,   // central moments
        m00,    // spatial moment
        devM, dev_lastBlockCounter    // helpers
        );
    if (!good)
        return false;

    //==== Norm spatial moments
    //double w00, w01, w02, w03, w10, w20, w30;
    good = ImageMomentsFeature_calcNormSpatialMoments(
        w00, w01, w02, w03, w10, w20, w30,  // output
        devI, nx, ny,   // image data
        cm00, cm01, cm02, cm03, cm10, cm20, cm30,
        cm22,
        devM, dev_lastBlockCounter    // helpers        
    );
    if (!good)
        return false;

    //==== Hu insvariants
    good = ImageMomentsFeature_calcHuInvariants(
        hm1, hm2, hm3, hm4, hm5, hm6, hm7,  // output
        devI, nx, ny,   // image data
        nu02, nu03, nu11, nu12, nu20, nu21, nu30,
        devM, dev_lastBlockCounter    // helpers        
    );
    if (!good)
        return false;

    //==== Weighted moments
    // Send the weighted image matrix to device
    const pixData& W = weighted_Im.ReadablePixels();
    const PixIntens* hoW = W.data();

    ok = cudaMemcpy(devI, hoW, szb, cudaMemcpyHostToDevice);    // Now devI stores the weighted image matrix
    if (ok != cudaSuccess)
        return false;

    // Calculate the origin
    good = ImageMomentsFeature_calcOrigins(
        originX, originY,   // output
        devI, nx, ny,
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //calcWeightedSpatialMoments(W);
    good = ImageMomentsFeature_calcSpatialMoments(
        wm00, wm01, wm02, wm03, wm10, wm11, wm12, wm20, wm21, wm30,   // output
        devI, nx, ny,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //calcWeightedCentralMoments(W);
    double wcm00, wcm01, wcm10, wcm22;
    good = ImageMomentsFeature_calcCentralMoments(
        wcm00, wcm01, wcm02, wcm03, wcm10, wcm11, wcm12, wcm20, wcm21, wcm22, wcm30,   // output
        devI, nx, ny, originX, originY,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //calcWeightedHuInvariants(W);
    // --1
    double wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30;   
    good = ImageMomentsFeature_calcNormCentralMoments(
        wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30,   // output
        devI, nx, ny,   // image data
        wcm02, wcm03, wcm11, wcm12, wcm20, wcm21, wcm30,   // weighted central moments
        m00,    // spatial moment
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;
    // --2
    good = ImageMomentsFeature_calcHuInvariants(
        whm1, whm2, whm3, whm4, whm5, whm6, whm7,  // output
        devI, nx, ny,   // image data
        wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30,
        devM, dev_lastBlockCounter    // helpers        
    );
    if (!good)
        return false;

    //==== Free buffers

    // Free the reduction helper
    ok = cudaFree(dev_lastBlockCounter);
    CHECKERR(ok);

    // Free device-side image matrix 
	ok = cudaFree(devI);
    CHECKERR(ok);

	ok = cudaFree(devM);
	if (ok != cudaSuccess)
		return false;

	return true;
}

__global__ void kerCalcWeightedImage(
    PixIntens* dev_Img,
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
        size_t dist = (x-pxX) * (x-pxX) + (y-pxY) * (y-pxY);
        if (minDistPow2 > dist)
            minDistPow2 = dist;
    }

    double dist = sqrt((double)minDistPow2);
    const double epsilon = 0.1;
    PixIntens wpi = pi / (dist + epsilon) + 0.5/*rounding*/;
    dev_Img[pxIdx] = wpi;
}

bool ImageMomentsFeature_calcWeightedMatrix(
    PixIntens* devI,   // input & output
    size_t nx,
    size_t ny,
    size_t roi_index, 
    StatsInt aabb_min_x, 
    StatsInt aabb_min_y)
{
    size_t len = nx * ny;
    int nb = whole_chunks2(len, blockSize);
    kerCalcWeightedImage <<< nb, blockSize >>> (devI, len, nx, aabb_min_x, aabb_min_y, roi_index, devIndices, devContourData);

    cudaError_t ok = cudaDeviceSynchronize();
    CHECKERR(ok);

    ok = cudaGetLastError();
    CHECKERR(ok);

    return true;
}

bool ImageMomentsFeature_calculate2(
    double& m00, double& m01, double& m02, double& m03, double& m10, double& m11, double& m12, double& m20, double& m21, double& m30,   // spatial moments
    double& cm02, double& cm03, double& cm11, double& cm12, double& cm20, double& cm21, double& cm30,   // central moments
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,    // normalized central moments
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w20, double& w30,   // normalized spatial moments
    double& hm1, double& hm2, double& hm3, double& hm4, double& hm5, double& hm6, double& hm7,  // Hu moments
    const ImageMatrix& Im,
    double& wm00, double& wm01, double& wm02, double& wm03, double& wm10, double& wm11, double& wm12, double& wm20, double& wm21, double& wm30,   // weighted spatial moments
    double& wcm02, double& wcm03, double& wcm11, double& wcm12, double& wcm20, double& wcm21, double& wcm30,   // weighted central moments
    double& whm1, double& whm2, double& whm3, double& whm4, double& whm5, double& whm6, double& whm7,   // weighted Hum moments
    size_t roi_index, 
    StatsInt aabb_min_x,
    StatsInt aabb_min_y)
{
    const pixData& I = Im.ReadablePixels();
    const PixIntens* hoI = I.data();

    // Send the image matrix to device
    size_t szb = I.size() * sizeof(hoI[0]);
    PixIntens* devI = nullptr;
    auto ok = cudaMalloc(reinterpret_cast<void**> (&devI), szb);
    if (ok != cudaSuccess)
        return false;

    ok = cudaMemcpy(devI, hoI, szb, cudaMemcpyHostToDevice);
    if (ok != cudaSuccess)
        return false;

    // Prepare the moment calculation result
    double* devM = nullptr;
    ok = cudaMalloc(reinterpret_cast<void**> (&devM), sizeof(devM[0]));
    if (ok != cudaSuccess)
        return false;

    // Prepare the reduction helper
    int* dev_lastBlockCounter;
    ok = cudaMalloc((void**)&dev_lastBlockCounter, sizeof(int));
    CHECKERR(ok);

    // Calculate the origin
    size_t nx = I.width(),
        ny = I.height();

    double originX, originY;
    bool good = ImageMomentsFeature_calcOrigins(
        originX, originY,   // output
        devI, nx, ny,
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //==== Spatial moments
    good = ImageMomentsFeature_calcSpatialMoments(
        m00, m01, m02, m03, m10, m11, m12, m20, m21, m30,   // output
        devI, nx, ny,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //==== Central moments
    double cm00, cm01, cm10, cm22;
    good = ImageMomentsFeature_calcCentralMoments(
        cm00, cm01, cm02, cm03, cm10, cm11, cm12, cm20, cm21, cm22, cm30,   // output
        devI, nx, ny, originX, originY,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //==== Norm central moments
    good = ImageMomentsFeature_calcNormCentralMoments(
        nu02, nu03, nu11, nu12, nu20, nu21, nu30,   // output
        devI, nx, ny,   // image data
        cm02, cm03, cm11, cm12, cm20, cm21, cm30,   // central moments
        m00,    // spatial moment
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //==== Norm spatial moments
    //double w00, w01, w02, w03, w10, w20, w30;
    good = ImageMomentsFeature_calcNormSpatialMoments(
        w00, w01, w02, w03, w10, w20, w30,  // output
        devI, nx, ny,   // image data
        cm00, cm01, cm02, cm03, cm10, cm20, cm30,
        cm22,
        devM, dev_lastBlockCounter    // helpers        
    );
    if (!good)
        return false;

    //==== Hu insvariants
    good = ImageMomentsFeature_calcHuInvariants(
        hm1, hm2, hm3, hm4, hm5, hm6, hm7,  // output
        devI, nx, ny,   // image data
        nu02, nu03, nu11, nu12, nu20, nu21, nu30,
        devM, dev_lastBlockCounter    // helpers        
    );
    if (!good)
        return false;

    //==== Weighted moments

    // Use the image matrix and cached ROIs' contours on the GPU side to calculate weights
    good = ImageMomentsFeature_calcWeightedMatrix(
        devI,   // input & output
        nx, ny, 
        roi_index, 
        aabb_min_x,
        aabb_min_y);
    if (!good)
        return false;

    // Calculate the origin
    good = ImageMomentsFeature_calcOrigins(
        originX, originY,   // output
        devI, nx, ny,
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //calcWeightedSpatialMoments(W);
    good = ImageMomentsFeature_calcSpatialMoments(
        wm00, wm01, wm02, wm03, wm10, wm11, wm12, wm20, wm21, wm30,   // output
        devI, nx, ny,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //calcWeightedCentralMoments(W);
    double wcm00, wcm01, wcm10, wcm22;
    good = ImageMomentsFeature_calcCentralMoments(
        wcm00, wcm01, wcm02, wcm03, wcm10, wcm11, wcm12, wcm20, wcm21, wcm22, wcm30,   // output
        devI, nx, ny, originX, originY,   // image data
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;

    //calcWeightedHuInvariants(W);
    // --1
    double wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30;
    good = ImageMomentsFeature_calcNormCentralMoments(
        wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30,   // output
        devI, nx, ny,   // image data
        wcm02, wcm03, wcm11, wcm12, wcm20, wcm21, wcm30,   // weighted central moments
        m00,    // spatial moment
        devM, dev_lastBlockCounter    // helpers
    );
    if (!good)
        return false;
    // --2
    good = ImageMomentsFeature_calcHuInvariants(
        whm1, whm2, whm3, whm4, whm5, whm6, whm7,  // output
        devI, nx, ny,   // image data
        wnu02, wnu03, wnu11, wnu12, wnu20, wnu21, wnu30,
        devM, dev_lastBlockCounter    // helpers        
    );
    if (!good)
        return false;

    //==== Free buffers

    // Free the reduction helper
    ok = cudaFree(dev_lastBlockCounter);
    CHECKERR(ok);

    // Free device-side image matrix 
    ok = cudaFree(devI);
    CHECKERR(ok);

    ok = cudaFree(devM);
    if (ok != cudaSuccess)
        return false;

    return true;
}

bool send_contours_to_gpu(const std::vector<size_t>& Indices, const std::vector< StatsInt>& ContourData)
{
    size_t szb = Indices.size() * sizeof(devIndices[0]);
    auto ok = cudaMalloc (reinterpret_cast<void**> (&devIndices), szb);
    CHECKERR(ok);

    const size_t* hoIndices = Indices.data();
    ok = cudaMemcpy (devIndices, hoIndices, szb, cudaMemcpyHostToDevice);
    CHECKERR(ok);

    szb = ContourData.size() * sizeof(devContourData[0]);
    ok = cudaMalloc (reinterpret_cast<void**> (&devContourData), szb);
    CHECKERR(ok);

    const StatsInt* hoContourData = ContourData.data();
    ok = cudaMemcpy (devContourData, hoContourData, szb, cudaMemcpyHostToDevice);
    CHECKERR(ok);

    return true;
}

bool free_contour_data_on_gpu()
{
    CHECKERR(cudaFree(devIndices));
    CHECKERR(cudaFree(devContourData));
    return true;
}

