#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"
#include "../gpucache.h"
#include "geomoments.cuh"

namespace NyxusGpu
{

    __global__  void kerNormCentralMoms(gpureal* s, bool weighted)  // s - state
    {
        // safety
        int tid = threadIdx.x + blockIdx.x * blockSize;
        if (tid)
            return;

        //
        // Based on:
        // Zhihu Huang and Jinsong Leng, "Analysis of Hu's moment invariants on image scaling and rotation," 
        // 2010 2nd International Conference on Computer Engineering and Technology, Chengdu, China, 2010, 
        // pp. V7-476-V7-480, doi: 10.1109/ICCET.2010.5485542
        //
        // Formula:
        //      gamma = (p + q + 2.0) / 2.0
        //      NCM_pq = CentralMom(D, p, q) / (CentralMom(D, 0, 0)^gamma)

        double cm00, k;

        if (weighted)
        {
            cm00 = s[GpusideState::WCM00];
            // 02
            k = (0. + 2.) / 2. + 1.;
            s[GpusideState::WNU02] = s[GpusideState::WCM02] / pow(cm00, k);
            // 03
            k = (0. + 3.) / 2. + 1.;
            s[GpusideState::WNU03] = s[GpusideState::WCM03] / pow(cm00, k);
            // 11
            k = (1. + 1.) / 2. + 1.;
            s[GpusideState::WNU11] = s[GpusideState::WCM11] / pow(cm00, k);
            // 12
            k = (1. + 2.) / 2. + 1.;
            s[GpusideState::WNU12] = s[GpusideState::WCM12] / pow(cm00, k);
            // 20
            k = (2. + 0.) / 2. + 1.;
            s[GpusideState::WNU20] = s[GpusideState::WCM20] / pow(cm00, k);
            // 21
            k = (2. + 1.) / 2. + 1.;
            s[GpusideState::WNU21] = s[GpusideState::WCM21] / pow(cm00, k);
            // 30
            k = (3. + 0.) / 2. + 1.;
            s[GpusideState::WNU30] = s[GpusideState::WCM30] / pow(cm00, k);
        }
        else
        {
            cm00 = s[GpusideState::CM00];
            // 02
            k = (0. + 2.) / 2. + 1.;
            s[GpusideState::NU02] = s[GpusideState::CM02] / pow(cm00, k);
            // 03
            k = (0. + 3.) / 2. + 1.;
            s[GpusideState::NU03] = s[GpusideState::CM03] / pow(cm00, k);
            // 11
            k = (1. + 1.) / 2. + 1.;
            s[GpusideState::NU11] = s[GpusideState::CM11] / pow(cm00, k);
            // 12
            k = (1. + 2.) / 2. + 1.;
            s[GpusideState::NU12] = s[GpusideState::CM12] / pow(cm00, k);
            // 20
            k = (2. + 0.) / 2. + 1.;
            s[GpusideState::NU20] = s[GpusideState::CM20] / pow(cm00, k);
            // 21
            k = (2. + 1.) / 2. + 1.;
            s[GpusideState::NU21] = s[GpusideState::CM21] / pow(cm00, k);
            // 30
            k = (3. + 0.) / 2. + 1.;
            s[GpusideState::NU30] = s[GpusideState::CM30] / pow(cm00, k);
        }
    }

    bool drvNormCentralMoms(gpureal* d_state, bool weighted)
    {
        kerNormCentralMoms << <1, 1 >> > (d_state, weighted);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        return true;
    }

    bool ImageMomentsFeature_calcNormCentralMoments3(gpureal* d_state, bool weighted)
    {
        return drvNormCentralMoms(d_state, weighted);
    }

    __global__ void kerNormSpatialMoms(gpureal* s)
    {
        // safety
        int tid = threadIdx.x + blockIdx.x * blockSize;
        if (tid)
            return;

        //
        // Based on:
        // Intel's Integrated Performance Primitives, Part 2, p. 697 
        // https://cdrdv2-public.intel.com/671189/ippi.pdf
        //
        // Formula:
        //      gamma = (p + q + 2.0) / 2.0
        //      NSM_pq = SpatialMom(D, p, q) / (SpatialMom(D, 0, 0)^gamma)

        // Normalizing coefficients 
        double m00 = s[GpusideState::RM00];
        double k;

        // 00
        k = (0. + 0.) / 2. + 1.;
        s[GpusideState::W00] = s[GpusideState::RM00] / pow(m00, k);
        // 01
        k = (0. + 1.) / 2. + 1.;
        s[GpusideState::W01] = s[GpusideState::RM01] / pow(m00, k);
        // 02
        k = (0. + 2.) / 2. + 1.;
        s[GpusideState::W02] = s[GpusideState::RM02] / pow(m00, k);
        // 03
        k = (0. + 3.) / 2. + 1.;
        s[GpusideState::W03] = s[GpusideState::RM03] / pow(m00, k);
        // 10
        k = (1. + 0.) / 2. + 1.;
        s[GpusideState::W10] = s[GpusideState::RM10] / pow(m00, k);
        // 11
        k = (1. + 1.) / 2. + 1.;
        s[GpusideState::W11] = s[GpusideState::RM11] / pow(m00, k);
        // 12
        k = (1. + 2.) / 2. + 1.;
        s[GpusideState::W12] = s[GpusideState::RM12] / pow(m00, k);
        // 13
        k = (1. + 3.) / 2. + 1.;
        s[GpusideState::W13] = s[GpusideState::RM13] / pow(m00, k);
        // 20
        k = (2. + 0.) / 2. + 1.;
        s[GpusideState::W20] = s[GpusideState::RM20] / pow(m00, k);
        // 21
        k = (2. + 1.) / 2. + 1.;
        s[GpusideState::W21] = s[GpusideState::RM21] / pow(m00, k);
        // 22
        k = (2. + 2.) / 2. + 1.;
        s[GpusideState::W22] = s[GpusideState::RM22] / pow(m00, k);
        // 23
        k = (2. + 3.) / 2. + 1.;
        s[GpusideState::W23] = s[GpusideState::RM23] / pow(m00, k);
        // 30
        k = (3. + 0.) / 2. + 1.;
        s[GpusideState::W30] = s[GpusideState::RM30] / pow(m00, k);
        // 31
        k = (3. + 1.) / 2. + 1.;
        s[GpusideState::W31] = s[GpusideState::RM31] / pow(m00, k);
        // 32
        k = (3. + 2.) / 2. + 1.;
        s[GpusideState::W32] = s[GpusideState::RM32] / pow(m00, k);
        // 33
        k = (3. + 3.) / 2. + 1.;
        s[GpusideState::W33] = s[GpusideState::RM33] / pow(m00, k);
    }

    bool drvNormSpatialMoms(gpureal* d_state)
    {
        kerNormSpatialMoms << <1, 1 >> > (d_state);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        return true;
    }

    bool ImageMomentsFeature_calcNormSpatialMoments3(gpureal* d_state)
    {
        return drvNormSpatialMoms(d_state);
    }

}