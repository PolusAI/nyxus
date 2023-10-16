#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/image_matrix.h"
#include "gpu.h"

bool ImageMomentsFeature_calcOrigins(
    // output
    double& originOfX, double& originOfY,
    // input
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y);

bool ImageMomentsFeature_calcOriginsWeighted(
    // output
    double& originOfX, double& originOfY,
    // input
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloudlen,
    StatsInt base_x,
    StatsInt base_y);

bool ImageMomentsFeature_calcRawMoments(
    // output
    double& _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _30,
    // input
    const Pixel2* d_roicloud,
    size_t cloud_len, 
    StatsInt base_x, 
    StatsInt base_y);

bool ImageMomentsFeature_calcRawMomentsWeighted(
    // output
    double& _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _30,
    // input
    const RealPixIntens* d_realintens,
    const Pixel2* d_roicloud,
    size_t cloud_len,
    StatsInt base_x,
    StatsInt base_y);

bool ImageMomentsFeature_calcCentralMoments(
    // output
    double& _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _22, double& _30,
    // input
    const Pixel2* d_roicloud, size_t cloud_len, StatsInt base_x, StatsInt base_y, double origin_x, double origin_y);

bool ImageMomentsFeature_calcCentralMomentsWeighted(
    // output
    double& _00, double& _01, double& _02, double& _03, double& _10, double& _11, double& _12, double& _20, double& _21, double& _22, double& _30,
    // input
    const RealPixIntens* d_realintens, const Pixel2* d_roicloud, size_t cloud_len, StatsInt base_x, StatsInt base_y, double origin_x, double origin_y);

bool ImageMomentsFeature_calc_weighted_intens(
    // output
    RealPixIntens* d_realintens_buf,
    // input
    const Pixel2* d_roicloud,
    size_t cloud_len,
    const Pixel2* d_roicontour,
    size_t contour_len);

bool ImageMomentsFeature_calcHuInvariants3(
    double& h1, double& h2, double& h3, double& h4, double& h5, double& h6, double& h7,   // output
    double nu02, double nu03, double nu11, double nu12, double nu20, double nu21, double nu30); // reduction helpers

bool ImageMomentsFeature_calcNormCentralMoments3(
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,   // output
    double cm02, double cm03, double cm11, double cm12, double cm20, double cm21, double cm30,
    double m00);

bool ImageMomentsFeature_calcNormSpatialMoments3(
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w20, double& w30,   // output
    double cm00, double cm01, double cm02, double cm03, double cm10, double cm20, double cm30,
    double cm22);

namespace Nyxus
{
    // Objects implementing GPU-based calculation of geometric moments
    // -- device-side copy of a ROI cloud
    Pixel2* devRoiCloudBuffer = nullptr;
    size_t roi_cloud_len = 0;
    RealPixIntens* devRealintensBuffer = nullptr;
    // -- device-side copy of ROI's contour data
    Pixel2* devContourCloudBuffer = nullptr;
    size_t contour_cloud_len = 0;
    // -- result of partial geometric moment expression (before sum-reduce)
    double* devPrereduce = nullptr;
    // -- reduce helpers
    double* d_out = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
}

bool ImageMomentsFeature_calculate (
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
    //==== Origin
    double originX, originY;

    bool good = ImageMomentsFeature_calcOrigins(
        // output
        originX,
        originY,
        // input
        Nyxus::devRoiCloudBuffer, 
        Nyxus::roi_cloud_len,
        aabb_min_x,
        aabb_min_y);

    if (!good)
        return false;

    //==== Spatial moments
    good = ImageMomentsFeature_calcRawMoments(
        m00, m01, m02, m03, m10, m11, m12, m20, m21, m30,
        Nyxus::devRoiCloudBuffer, Nyxus::roi_cloud_len, aabb_min_x, aabb_min_y); 

    if (!good)
        return false;

    //==== Central moments
    double cm00_ = -1, cm01_ = -1, cm10_ = -1, cm22_ = -1;  // needd by norm raw moments, mark with -1 as unassigned
    good = ImageMomentsFeature_calcCentralMoments(
        cm00_, cm01_, cm02, cm03, cm10_, cm11, cm12, cm20, cm21, cm22_, cm30,
        Nyxus::devRoiCloudBuffer, Nyxus::roi_cloud_len, aabb_min_x, aabb_min_y, originX, originY);

    if (!good)
        return false;

    //==== Norm central moments
    good = ImageMomentsFeature_calcNormCentralMoments3(
        nu02, nu03, nu11, nu12, nu20, nu21, nu30,   // output
        cm02, cm03, cm11, cm12, cm20, cm21, cm30,   // central moments
        m00);    // spatial moment

    if (!good)
        return false;
    
    //==== Norm-spatial moments
    good = ImageMomentsFeature_calcNormSpatialMoments3(
        w00, w01, w02, w03, w10, w20, w30,  // output
        cm00_, cm01_, cm02, cm03, cm10_, cm20, cm30, cm22_);
    if (!good)
        return false;

    //==== Hu insvariants
    good = ImageMomentsFeature_calcHuInvariants3(
        hm1, hm2, hm3, hm4, hm5, hm6, hm7,  // output
        nu02, nu03, nu11, nu12, nu20, nu21, nu30);
    if (!good)
        return false;

    //==== Weighted intensities
    good = ImageMomentsFeature_calc_weighted_intens(
        Nyxus::devRealintensBuffer, // output
        Nyxus::devRoiCloudBuffer,
        Nyxus::roi_cloud_len,
        Nyxus::devContourCloudBuffer,
        Nyxus::contour_cloud_len);

    if (!good)
        return false;

    //==== Weighted origin
    good = ImageMomentsFeature_calcOriginsWeighted(
        // output
        originX,
        originY,
        // input
        Nyxus::devRealintensBuffer,
        Nyxus::devRoiCloudBuffer,
        Nyxus::roi_cloud_len,
        aabb_min_x,
        aabb_min_y);

    if (!good)
        return false;

    //==== Weighted raw moments
    good = ImageMomentsFeature_calcRawMomentsWeighted (
        wm00, wm01, wm02, wm03, wm10, wm11, wm12, wm20, wm21, wm30,
        Nyxus::devRealintensBuffer, Nyxus::devRoiCloudBuffer, Nyxus::roi_cloud_len, aabb_min_x, aabb_min_y);

    if (!good)
        return false;

    //==== Weighted central moments
    double wcm00_ = -1, wcm01_ = -1, wcm10_ = -1, wcm22_ = -1;  // needd by norm raw moments, mark with -1 as unassigned
    good = ImageMomentsFeature_calcCentralMomentsWeighted (
        wcm00_, wcm01_, wcm02, wcm03, wcm10_, wcm11, wcm12, wcm20, wcm21, wcm22_, wcm30,
        Nyxus::devRealintensBuffer, Nyxus::devRoiCloudBuffer, Nyxus::roi_cloud_len, aabb_min_x, aabb_min_y, originX, originY);

    if (!good)
        return false;

    //==== Weighted Hu invariants
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

// 'pixcloud_size' is the max pixel cloud of ROIs in a batch allocated in phase 2

bool allocate_2dmoments_buffers_on_gpu (size_t cloudsize)
{
    // Save the cloud size
    Nyxus::roi_cloud_len = cloudsize;

    // Reserve the ROI cloud and contour buffers
    size_t szb1 = cloudsize * sizeof(Nyxus::devRoiCloudBuffer[0]);
    auto ok = cudaMalloc (reinterpret_cast<void**> (&Nyxus::devRoiCloudBuffer), szb1);
    CHECKERR(ok);
    ok = cudaMalloc(reinterpret_cast<void**> (&Nyxus::devContourCloudBuffer), szb1);
    CHECKERR(ok);

    // Reserve a buffer for real valued intensities to support weighted moments
    size_t szb2 = cloudsize * sizeof(Nyxus::devRealintensBuffer[0]);
    ok = cudaMalloc (reinterpret_cast<void**> (&Nyxus::devRealintensBuffer), szb2);
    CHECKERR(ok);

    // Allocate the reduction helper buffer
    size_t szb3 = cloudsize * sizeof(Nyxus::devPrereduce[0]);
    ok = cudaMalloc (reinterpret_cast<void**> (&Nyxus::devPrereduce), szb3);
    CHECKERR(ok);

    CHECKERR(cudaMalloc(&Nyxus::d_out, sizeof(double)));

    return true;
}

bool free_2dmoments_buffers_on_gpu()
{
    CHECKERR(cudaFree(Nyxus::devRoiCloudBuffer));
    CHECKERR(cudaFree(Nyxus::devRealintensBuffer));
    CHECKERR(cudaFree(Nyxus::devContourCloudBuffer));
    CHECKERR(cudaFree(Nyxus::devPrereduce));

    CHECKERR(cudaFree(Nyxus::d_out));
    if (Nyxus::d_temp_storage)
        CHECKERR(cudaFree(Nyxus::d_temp_storage));

    return true;
}

bool send_roi_data_2_gpu (Pixel2* data, size_t n)
{
    // Save the cloud size
    Nyxus::roi_cloud_len = n;

    // Transfer the pixel cloud
    cudaError_t ok = cudaMemcpy(Nyxus::devRoiCloudBuffer, data, Nyxus::roi_cloud_len * sizeof(data[0]), cudaMemcpyHostToDevice);
    CHECKERR(ok);

    return true;
}

bool send_contour_data_2_gpu (Pixel2* data, size_t n)
{
    // Save the cloud size
    Nyxus::contour_cloud_len = n;

    // Transfer the pixel cloud
    size_t szb = n * sizeof(data[0]);
    cudaError_t ok = cudaMemcpy(Nyxus::devContourCloudBuffer, data, szb, cudaMemcpyHostToDevice);
    CHECKERR(ok);
    return true;
}

bool free_roi_data_on_gpu()
{
    if (Nyxus::d_temp_storage)
    {
        CHECKERR(cudaFree(Nyxus::d_temp_storage));
        Nyxus::d_temp_storage = nullptr;
        Nyxus::temp_storage_bytes = 0;
    }

    return true;
}


