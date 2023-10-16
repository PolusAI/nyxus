#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../features/pixel.h"

bool drvRawMoment(
    double& retval,
    int p, int q,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y);

bool drvRawMomentWeighted(
    double& retval,
    int p, int q,
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y);

bool ImageMomentsFeature_calcOrigins (
    // output
    double& originOfX, double& originOfY,
    // input
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y)
{
    double _00, _10, _01;
    if (drvRawMoment(_00, 0, 0, d_roicloud, cloudlen, base_x, base_y) == false)
        return false;
    if (drvRawMoment(_10, 1, 0, d_roicloud, cloudlen, base_x, base_y) == false)
        return false;
    if (drvRawMoment(_01, 0, 1, d_roicloud, cloudlen, base_x, base_y) == false)
        return false;

    // Calculate the origin
    originOfX = _10 / _00;
    originOfY = _01 / _00;

    return true;
}

bool ImageMomentsFeature_calcOriginsWeighted(
    // output
    double& originOfX, double& originOfY,
    // input
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y)
{
    double _00, _10, _01;
    if (drvRawMomentWeighted(_00, 0, 0, d_realintens, d_roicloud, cloudlen, base_x, base_y) == false)
        return false;
    if (drvRawMomentWeighted(_10, 1, 0, d_realintens, d_roicloud, cloudlen, base_x, base_y) == false)
        return false;
    if (drvRawMomentWeighted(_01, 0, 1, d_realintens, d_roicloud, cloudlen, base_x, base_y) == false)
        return false;

    // Calculate the origin
    originOfX = _10 / _00;
    originOfY = _01 / _00;

    return true;
}
