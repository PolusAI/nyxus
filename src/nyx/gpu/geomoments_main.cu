#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "gpu.h"
#include "geomoments.cuh"
#include "../gpucache.h"

#include "../helpers/timing.h"

namespace NyxusGpu
{
    bool ImageMomentsFeature_calc_weighted_intens(
        // output
        RealPixIntens* d_realintens_buf,
        // input
        const Pixel2* d_roicloud,
        size_t cloud_len,
        const Pixel2* d_roicontour,
        size_t contour_len);

    bool ImageMomentsFeature_calcHuInvariants3(gpureal* d_state, bool weighted);

    bool ImageMomentsFeature_calcNormCentralMoments3(gpureal* d_state, bool weighted);

    bool ImageMomentsFeature_calcNormSpatialMoments3(gpureal* d_state);

    bool ImageMomentsFeature_calculate (size_t roi_index)
    {
        // context of ROI #roi_index:
        //
        // proper batch has been sent to gpu-side by this point
        //
        gpureal* state = &NyxusGpu::gpu_featurestatebuf.devbuffer[roi_index * GpusideState::__COUNT__];
        size_t cloud_len = NyxusGpu::gpu_roiclouds_2d.ho_lengths[roi_index];
        size_t cloud_offset = NyxusGpu::gpu_roiclouds_2d.ho_offsets[roi_index];
        Pixel2* d_cloud = &NyxusGpu::gpu_roiclouds_2d.devbuffer[cloud_offset];
        size_t contour_len = NyxusGpu::gpu_roicontours_2d.ho_lengths[roi_index];
        size_t contour_offset = NyxusGpu::gpu_roicontours_2d.ho_offsets[roi_index];
        Pixel2* d_contour = &NyxusGpu::gpu_roicontours_2d.devbuffer[contour_offset];
        double* d_prereduce = NyxusGpu::dev_prereduce;
        auto& d_devicereduce_tempstorage = NyxusGpu::dev_devicereduce_temp_storage;
        size_t devicereduce_tempstorage_szb = NyxusGpu::devicereduce_temp_storage_szb;
        RealPixIntens* d_realintens = NyxusGpu::dev_realintens;

        //==== Raw moments

        // result in Nyxus::d_intermediate [RMpq]
        if (!ImageMomentsFeature_calcRawMoments__snu(
            // out
            state,
            // in
            d_cloud,
            cloud_len,
            d_prereduce,
            d_devicereduce_tempstorage,
            devicereduce_tempstorage_szb))
            return false;

        // result in Nyxus::d_intermediate [ORGX] and [ORGY]
        if (!ImageMomentsFeature_calcOrigins (
            // output
            state,
            // input
            d_cloud,
            cloud_len,
            d_prereduce,
            d_devicereduce_tempstorage,
            devicereduce_tempstorage_szb))
            return false;

        // result in Nyxus::d_intermediate [CM00, CM01, ... CM33]
        if (!ImageMomentsFeature_calcCentralMoments__snu(
            // out
            state,
            // in
            d_cloud,
            cloud_len,
            d_prereduce,
            d_devicereduce_tempstorage,
            devicereduce_tempstorage_szb))
            return false;

        //==== Norm raw moments
        // result in state [W...]
        if (!ImageMomentsFeature_calcNormSpatialMoments3(state))
            return false;

        //==== Norm central moments
        // result in state [NU...]
        if (!ImageMomentsFeature_calcNormCentralMoments3(state, false))
            return false;

        //==== Hu insvariants
        // result in state[H1...7]
        if (!ImageMomentsFeature_calcHuInvariants3(state, false))
            return false;

        //==== Weighted intensities
        if (!ImageMomentsFeature_calc_weighted_intens(
            d_realintens, // output
            d_cloud,
            cloud_len,
            d_contour,
            contour_len))
            return false;

        //==== Weighted raw moments (result in state[WRM...])
        if (!ImageMomentsFeature_calcRawMomentsWeighted__snu(
            state,
            d_realintens, d_cloud, cloud_len,
            d_prereduce,
            d_devicereduce_tempstorage,
            devicereduce_tempstorage_szb))
            return false;

        //==== Weighted origin (state[WRM00, 01, 10] ---> state[WORGX] and [WORGY])
        if (!ImageMomentsFeature_calcOriginsWeighted__snu(
            // output
            state,
            // input
            NyxusGpu::dev_realintens,
            d_cloud,
            cloud_len,
            d_prereduce,
            d_devicereduce_tempstorage,
            devicereduce_tempstorage_szb))
            return false;

        //==== Weighted central moments (result in state[WCM...])
        if (!ImageMomentsFeature_calcCentralMomentsWeighted__snu(
            state,
            d_realintens, d_cloud, cloud_len, &state[WORGX], &state[WORGY],
            d_prereduce,
            d_devicereduce_tempstorage,
            devicereduce_tempstorage_szb))
            return false;

        //==== Weighted Hu invariants 
        // --1: weighted norm central moments (state[WNU...])
        if (!ImageMomentsFeature_calcNormCentralMoments3(state, true))
            return false;

        // --2
        if (!ImageMomentsFeature_calcHuInvariants3(state, true))
            return false;

        // Check if there's any pending error 
        CHECKERR(cudaPeekAtLastError());

        return true;
    }

}