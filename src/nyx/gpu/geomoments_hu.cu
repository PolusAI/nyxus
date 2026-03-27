#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"
#include "../cache.h"
#include "geomoments.cuh"

namespace NyxusGpu
{

    __global__ void kerHu(gpureal* s, bool weighted)    // s - state
    {
        // safety
        int tid = threadIdx.x + blockIdx.x * blockSize;
        if (tid)
            return;

        gpureal* h1 = &s[GpusideState::H1],
            * h2 = &s[GpusideState::H2],
            * h3 = &s[GpusideState::H3],
            * h4 = &s[GpusideState::H4],
            * h5 = &s[GpusideState::H5],
            * h6 = &s[GpusideState::H6],
            * h7 = &s[GpusideState::H7],
            * nu02 = &s[GpusideState::NU02],
            * nu20 = &s[GpusideState::NU20],
            * nu11 = &s[GpusideState::NU11],
            * nu12 = &s[GpusideState::NU12],
            * nu21 = &s[GpusideState::NU21],
            * nu03 = &s[GpusideState::NU03],
            * nu30 = &s[GpusideState::NU30];

        if (weighted)
        {
            h1 = &s[GpusideState::WH1];
            h2 = &s[GpusideState::WH2];
            h3 = &s[GpusideState::WH3];
            h4 = &s[GpusideState::WH4];
            h5 = &s[GpusideState::WH5];
            h6 = &s[GpusideState::WH6];
            h7 = &s[GpusideState::WH7];
            nu02 = &s[GpusideState::WNU02];
            nu20 = &s[GpusideState::WNU20];
            nu11 = &s[GpusideState::WNU11];
            nu12 = &s[GpusideState::WNU12];
            nu21 = &s[GpusideState::WNU21];
            nu03 = &s[GpusideState::WNU03];
            nu30 = &s[GpusideState::WNU30];
        }

        // Formula: double h1 = NormCentralMom(D, 2, 0) + NormCentralMom(D, 0, 2);
        *h1 = *nu20 + *nu02;

        // Formula: double h2 = pow((NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)), 2) + 4 * (pow(NormCentralMom(D, 1, 1), 2));
        *h2 = pow((*nu20 - *nu02), 2.0) + 4. * pow(*nu11, 2.0);

        // Formula: double h3 = pow((NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)), 2) +
        //    pow((3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)), 2);
        *h3 = pow((*nu30 - 3. * *nu12), 2.0) + pow((3. * *nu21 - *nu03), 2.0);

        // Formula: double h4 = pow((NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) +
        //    pow((NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)), 2);
        *h4 = pow((*nu30 + *nu12), 2.0) + pow((*nu21 + *nu03), 2.0);

        // Formula: double h5 = (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) *
        //    (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
        //    (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - 3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) +
        //    (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
        //    (pow(3 * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
        *h5 = (*nu30 - 3. * *nu12) *
            (*nu30 + *nu12) *
            (pow(*nu30 + *nu12, 2.0) - 3. * pow(*nu21 + *nu03, 2.0)) +
            (3. * *nu21 - *nu03) * (*nu21 + *nu03) *
            (pow(3. * (*nu30 + *nu12), 2.0) - pow(*nu21 + *nu03, 2.0));

        // Formula: double h6 = (NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
        //    pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) + (4 * NormCentralMom(D, 1, 1) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
        //        NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3));
        *h6 = (*nu20 - *nu02) * (pow(*nu30 + *nu12, 2.0) -
            pow(*nu21 + *nu03, 2.0)) + (4. * *nu11 * (*nu30 + *nu12) *
                *nu21 + *nu03);

        // Formula: double h7 = (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
        //    3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) - (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
        //    (3 * pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
        *h7 = (3. * *nu21 - *nu03) * (*nu30 + *nu12) * (pow(*nu30 + *nu12, 2.0) -
            3 * pow(*nu21 + *nu03, 2.0)) - (*nu30 - 3 * *nu12) * (*nu21 + *nu03) *
            (3 * pow(*nu30 + *nu12, 2.0) - pow(*nu21 + *nu03, 2.0));
    }

    bool drvHu(gpureal* d_state, bool weighted)
    {
        kerHu << <1, 1 >> > (d_state, weighted);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        return true;
    }

    bool ImageMomentsFeature_calcHuInvariants3(gpureal* d_state, bool weighted)
    {
        return drvHu(d_state, weighted);
    }

}