#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../cache.h"
#include "../features/pixel.h"
#include "geomoments.cuh"

namespace NyxusGpu
{

    __global__ void kerCalcOrigins(
        gpureal* org_x,
        gpureal* org_y,
        gpureal* m00,
        gpureal* m01,
        gpureal* m10)
    {
        // safety
        int tid = threadIdx.x + blockIdx.x * blockSize;
        if (tid)
            return;

        *org_x = *m10 / *m00;
        *org_y = *m01 / *m00;
    }

    bool drvCalcOrigin(
        gpureal* org_x,
        gpureal* org_y,
        gpureal* m00,
        gpureal* m01,
        gpureal* m10)
    {
        kerCalcOrigins << <1, 1 >> > (org_x, org_y, m00, m01, m10);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        return true;
    }

    bool ImageMomentsFeature_calcOrigins (
        // output
        gpureal* d_intermed,
        // input
        const Pixel2* d_roicloud,
        size_t cloudlen,
        double* d_prereduce,    // reduction helper [roi_cloud_len]
        void* d_temp_storage,
        size_t& temp_storage_szb)
    {
        // result will be in d_intermed [ORGX] and [ORGY]
        if (drvCalcOrigin(&d_intermed[ORGX], &d_intermed[ORGY], &d_intermed[RM00], &d_intermed[RM01], &d_intermed[RM10]) == false)
            return false;

        return true;
    }

    // [WRM00, 01, 10] ---> [WORGX, WORGY]
    bool ImageMomentsFeature_calcOriginsWeighted__snu(
        // output
        gpureal* d_intermed,
        // input
        const RealPixIntens* d_realintens,
        const Pixel2* d_roicloud,
        size_t cloudlen,
        double* d_prereduce,
        void*& d_temp_storage,
        size_t& temp_storage_szb)
    {
        // result will be in d_intermed [WORGX] and [WORGY]
        if (drvCalcOrigin(&d_intermed[WORGX], &d_intermed[WORGY], &d_intermed[WRM00], &d_intermed[WRM01], &d_intermed[WRM10]) == false)
            return false;

        return true;
    }

}