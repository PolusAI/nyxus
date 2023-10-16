#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"

bool ImageMomentsFeature_calcHuInvariants3(
    double& h1, double& h2, double& h3, double& h4, double& h5, double& h6, double& h7,   // output
    double nu02, double nu03, double nu11, double nu12, double nu20, double nu21, double nu30) // reduction helpers
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

    return true;
}

