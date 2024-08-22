#include "../features/pixel.h"  // types StatsInt, etc

namespace NyxusGpu
{

    bool ImageMomentsFeature_calcOrigins (
        // output
        gpureal* d_intermediate,
        // input
        const Pixel2* d_roicloud,
        size_t cloudlen,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb);

    bool ImageMomentsFeature_calcRawMoments__snu(
        // output
        gpureal* d_intermediate,
        // input
        const Pixel2* d_roicloud,
        size_t cloud_len,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb);

    bool ImageMomentsFeature_calcCentralMoments__snu(
        // output
        gpureal* d_intermediate,
        // input
        const Pixel2* d_roicloud, size_t cloud_len,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb);

    bool ImageMomentsFeature_calcOriginsWeighted__snu(
        // output
        gpureal* d_intermed,
        // input
        const RealPixIntens* d_realintens,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        double* d_prereduce,
        void*& d_temp_storage,
        size_t& temp_storage_szb);

    bool ImageMomentsFeature_calcRawMomentsWeighted__snu(
        // output
        gpureal* dev_state,
        // input
        const RealPixIntens* d_realintens, const Pixel2* d_roicloud, size_t cloud_len,
        double* d_prereduce,
        void*& d_temp_storage,
        size_t& temp_storage_szb);

    bool ImageMomentsFeature_calcCentralMomentsWeighted__snu(
        // output
        gpureal* dev_state,
        // input
        const RealPixIntens* d_realintens, const Pixel2* d_roicloud, size_t cloud_len, gpureal* dev_origin_x, gpureal* dev_origin_y,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb);

}