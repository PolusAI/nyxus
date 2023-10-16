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
    normCoef = pow(cm22, w * 1.0);
    w00 = cm00 / normCoef;
    // 01
    w = 1;
    normCoef = pow(cm22, w * 1.0);
    w01 = cm01 / normCoef;
    // 02
    w = 2;
    normCoef = pow(cm22, w * 1.0);
    w02 = cm02 / normCoef;
    // 03
    w = 3;
    normCoef = pow(cm22, w * 1.0);
    w03 = cm03 / normCoef;
    // 10
    w = 1;
    normCoef = pow(cm22, w * 1.0);
    w10 = cm10 / normCoef;
    // 20
    w = 2;
    normCoef = pow(cm22, w * 1.0);
    w20 = cm20 / normCoef;
    // 30
    w = 3;
    normCoef = pow(cm22, w * 1.0);
    w30 = cm30 / normCoef;

    return true;
}

