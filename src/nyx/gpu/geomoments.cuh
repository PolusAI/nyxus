#include "../features/pixel.h"  // types StatsInt, etc

namespace NyxusGpu
{

    inline __device__ double int_pow(double a, int b)
    {
        double retval = 1.0;
        for (int i = 0; i < b; i++)
            retval *= a;
        return retval;
    }

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
        bool need_shape_moments,
        const Pixel2* d_roicloud,
        size_t cloud_len,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb);

    bool ImageMomentsFeature_calcCentralMoments__snu(
        // output
        gpureal* d_intermediate,
        // input
        bool need_shape_moments,
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

    bool sumreduce(
        gpureal* d_result,
        size_t cloudlen,
        double* d_prereduce,
        void* d_devreduce_tempstorage,
        size_t& devreduce_tempstorage_szb);

    bool sumreduceNV2(
        // [in]
        double* g_odata,
        // [out]
        const unsigned int n,
        double* g_idata,
        void* unused1,  // compatibility with CUB-based sumreduce()
        size_t unused2);

    void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int& blocks, int& threads);

}