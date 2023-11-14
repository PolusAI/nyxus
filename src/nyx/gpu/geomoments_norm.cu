#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"

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
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w11, double& w12, double& w13, double& w20, double& w21, double& w22, double& w23, double& w30, double& w31, double& w32, double& w33,   // output
    double cm00, double cm01, double cm02, double cm03, double cm10, double cm11, double cm12, double cm13, double cm20, double cm21, double cm22, double cm23, double cm30, double cm31, double cm32, double cm33)
{
    // Formula: 
    //  double stddev = CentralMom(D, 2, 2);
    //  int w = std::max(q, p);
    //  double normCoef = pow(stddev, w);
    //  double retval = CentralMom(D, p, q) / normCoef;

    // Normalizing coefficients 
    double nc0 = pow(cm22, 0),
        nc1 = pow(cm22, 1),
        nc2 = pow(cm22, 2),
        nc3 = pow(cm22, 3);

    w00 = cm00 / nc0;  // 00
    w01 = cm01 / nc1;  // 01
    w02 = cm02 / nc2;  // 02
    w03 = cm03 / nc3;  // 03

    w10 = cm10 / nc0;  // 10
    w11 = cm11 / nc1;  // 11
    w12 = cm12 / nc2;  // 12
    w13 = cm13 / nc3;  // 13

    w20 = cm20 / nc0;  // 20
    w21 = cm21 / nc1;  // 21
    w22 = cm22 / nc2;  // 22
    w23 = cm23 / nc3;  // 23

    w30 = cm30 / nc0;  // 30
    w31 = cm31 / nc1;  // 31
    w32 = cm32 / nc2;  // 32
    w33 = cm33 / nc3;  // 33

    return true;
}

