#include <vector>
#include "gabor.cuh"
#include "../gpucache.h"

using namespace std;

namespace CuGabor {

    __global__ void multiply(cufftDoubleComplex* A, int row_size, int col_size, cufftDoubleComplex* B, cufftDoubleComplex* result, int batch_size) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        int index = i * col_size * batch_size + j;
        double a, b, c, d;
        if (i < row_size && j < batch_size * col_size) {
            a = A[index].x;
            b = A[index].y;
            c = B[index].x;
            d = B[index].y;

            result[index].x = a * c - b * d;
            result[index].y = a * d + b * c;
        }
    }

    bool cmat_mult(cufftDoubleComplex* A, int row_size, int col_size, cufftDoubleComplex* B, cufftDoubleComplex* result, int batch_size) {
        int block = 16;
        dim3 threadsPerBlock(block, block);
        dim3 blocksPerGrid(ceil(row_size / block) + 1, ceil(col_size * batch_size / block) + 1);

        multiply << <blocksPerGrid, threadsPerBlock >> > (A, row_size, col_size, B, result, batch_size);

        // Wait for device to finish all operation
        cudaError_t ok = cudaDeviceSynchronize();
        CHECKERR(ok);

        // Check if kernel execution generated and error
        ok = cudaGetLastError();
        CHECKERR(ok);

        return true;
    }

    __global__ void multiply(CuComplex* A, int row_size, int col_size, CuComplex* B, CuComplex* result, int batch_size) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        int index = i * col_size * batch_size + j;
        double a, b, c, d;
        if (i < row_size && j < batch_size * col_size) {
            a = A[index].x;
            b = A[index].y;
            c = B[index].x;
            d = B[index].y;

            result[index].x = a * c - b * d;
            result[index].y = a * d + b * c;
        }
    }

    bool cmat_mult(CuComplex* A, int row_size, int col_size, CuComplex* B, CuComplex* result, int batch_size) {
        int block = 16;
        dim3 threadsPerBlock(block, block);
        dim3 blocksPerGrid(ceil(row_size / block) + 1, ceil(col_size * batch_size / block) + 1);

        multiply << <blocksPerGrid, threadsPerBlock >> > (A, row_size, col_size, B, result, batch_size);

        // Wait for device to finish all operation
        cudaError_t ok = cudaDeviceSynchronize();
        CHECKERR(ok);

        // Check if kernel execution generated and error
        ok = cudaGetLastError();
        CHECKERR(ok);

        return true;
    }

    bool conv_dud_gpu_fft(double* out,
        const unsigned int* image,
        double* kernel,
        int image_n, int image_m, int kernel_n, int kernel_m)
    {
        int batch_size = 1;

        // calculate new size of image based on padding size
        int row_size = image_m + kernel_m - 1;
        int col_size = image_n + kernel_n - 1;
        int size = row_size * col_size;

        if ((2 * size * batch_size) >= CUFFT_MAX_SIZE) {
            throw invalid_argument("Batch of images is too large. The maximumum number of values in cuFFT is 2^27.");
        }

        std::vector<Complex> linear_image(size * batch_size);
        std::vector<Complex> result(size * batch_size);
        std::vector<Complex> linear_kernel(size * batch_size);

        int index, index2;

        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {

                index = (i * col_size + j);
                linear_image[index].y = 0.f;

                if (i < image_m && j < image_n) {

                    index2 = (i * image_n + j);
                    linear_image[index].x = image[index2];

                }
                else {
                    linear_image[index].x = 0.f; // add padding
                }
            }
        }

        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {

                index = (i * col_size + j);
                index2 = (i * kernel_n + j);

                if (i < kernel_m && j < kernel_n) {
                    linear_kernel[index].x = kernel[2 * index2];
                }
                else {
                    linear_kernel[index].x = 0.f;
                    linear_kernel[index].y = 0.f;
                }
            }
        }

        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = (i * col_size + j);
            }
        }

        CuComplex* d_image;
        CuComplex* d_result;
        CuComplex* d_kernel;

        int n[2] = { row_size, col_size };

        auto ok = cudaMalloc((void**)&d_image, sizeof(CuComplex) * size * batch_size);
        CHECKERR(ok);

        ok = cudaMalloc((void**)&d_result, sizeof(CuComplex) * size * batch_size);
        CHECKERR(ok);

        ok = cudaMalloc((void**)&d_kernel, sizeof(CuComplex) * size * batch_size);
        CHECKERR(ok);

        // copy data to GPU
        ok = cudaMemcpy(d_image, linear_image.data(), batch_size * size * sizeof(CuComplex), cudaMemcpyHostToDevice);
        CHECKERR(ok);

        ok = cudaMemcpy(d_kernel, linear_kernel.data(), batch_size * size * sizeof(CuComplex), cudaMemcpyHostToDevice);
        CHECKERR(ok);

        cufftHandle plan;
        cufftHandle plan_k;
        int idist = size;
        int odist = size;

        int inembed[] = { row_size, col_size };
        int onembed[] = { row_size, col_size };

        int istride = 1;
        int ostride = 1;

        auto call = cufftPlanMany(&plan, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size);
        CHECKCUFFTERR(call);

        call = cufftPlanMany(&plan_k, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_size);
        CHECKCUFFTERR(call);


        call = cufftExecC2C(plan, d_image, d_image, CUFFT_FORWARD);
        CHECKCUFFTERR(call);

        call = cufftExecC2C(plan_k, d_kernel, d_kernel, CUFFT_FORWARD);
        CHECKERR(ok);

        ok = cudaDeviceSynchronize();
        CHECKERR(ok);

        // element-wise multiplication of the image and kernel
        bool success = cmat_mult(d_image, row_size, col_size, d_kernel, d_result, batch_size);
        if (!success) {
            return false;
        }

        // transform out of fourier space
        call = cufftExecC2C(plan, d_result, d_result, CUFFT_INVERSE);
        CHECKCUFFTERR(call);

        // copy results from device to host
        ok = cudaMemcpy(result.data(), d_result, batch_size * size * sizeof(CuComplex), cudaMemcpyDeviceToHost);
        CHECKERR(ok);

        // transfer to output array 
        for (int i = 0; i < size; ++i) {
            out[2 * i] = (result[i].x / (size));
            out[2 * i + 1] = (result[i].y / (size));
        }

        // free device memory
        cufftDestroy(plan);
        cufftDestroy(plan_k);
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);

        return true;
    }


    __global__ void re_2_complex_img(
        // out
        cufftDoubleComplex* linear_image,
        // in
        size_t row_size,
        size_t col_size,
        size_t roi_w,
        size_t roi_h,
        PixIntens* I,
        size_t batch_idx)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= row_size || j >= col_size)
            return;

        size_t index = batch_idx + (i * col_size + j);
        linear_image[index].y = 0.f;

        if (i < roi_w && j < roi_h)
        {

            size_t index2 = (i * roi_h + j);
            linear_image[index].x = I[index2];

        }
        else
        {
            linear_image[index].x = 0; // add padding
        }
    }

    __global__ void kerImatFromCloud(PixIntens* I, const Pixel2* C, size_t clen, size_t w)
    {
        int tid = threadIdx.x + blockIdx.x * blockSize;

        if (tid >= clen)
            return;

        const Pixel2* p = &C[tid];
        size_t x = p->x,
            y = p->y;
        size_t offs = y * w + x;
        I[offs] = p->inten;
    }

    // result in NyxusGpu::dev_imat1
    bool drvImatFromCloud(size_t roiidx, size_t w, size_t h)
    {
        size_t cloud_len = NyxusGpu::gpu_roiclouds_2d.ho_lengths[roiidx];
        size_t cloud_offset = NyxusGpu::gpu_roiclouds_2d.ho_offsets[roiidx];
        Pixel2* d_cloud = &NyxusGpu::gpu_roiclouds_2d.devbuffer[cloud_offset];

        // zero the matrix
        size_t szb = sizeof(NyxusGpu::dev_imat1[0]) * w * h;
        cudaMemset(NyxusGpu::dev_imat1, 0, szb);

        // apply the cloud to it
        int nblo = whole_chunks2(cloud_len, blockSize);
        kerImatFromCloud << < nblo, blockSize >> > (NyxusGpu::dev_imat1, d_cloud, cloud_len, w);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        return true;
    }

    // layouts are different! Complex column-major, energy row-major
    __global__ void kerCalcEnergy(
        // out
        PixIntens* nonpadded_e_montage,  // montage of smaller (nonpadded) images of size ROI_w*ROI_h
        // in
        cufftDoubleComplex* padded_montage, // montage of complex
        size_t padded_frame_offset,
        size_t padded_w,
        size_t padded_h,
        size_t roi_offset,
        size_t roi_w,
        size_t roi_h,
        size_t kerside)
    {
        int c = threadIdx.x + blockIdx.x * blockDim.x;
        int r = threadIdx.y + blockIdx.y * blockDim.y;
        if (c >= roi_w || r >= roi_h)
            return;

        size_t idx_p = r * padded_w + c + kerside / 2;

        double plen = (double)(padded_w * padded_h);
        cufftDoubleComplex w = padded_montage[padded_frame_offset + idx_p];

        double wx = w.x / plen,
            wy = w.y / plen,
            e = sqrt(wx * wx + wy * wy);

        size_t idx_roi = r * roi_w + c;
        nonpadded_e_montage[roi_offset + idx_roi] = e;
    }

    bool conv_dud_gpu_fft_multi_filter(
        double* out,
        const unsigned int* image,
        double* kernel,
        int image_n, int image_m,
        int kernel_n, int kernel_m,
        int n_filters,
        double* dev_filterbank)
    {
        // calculate new size of image based on padding size
        int row_size = image_m + kernel_m - 1;
        int col_size = image_n + kernel_n - 1;
        int size = row_size * col_size; // image+kernel size [pixels]

        size_t bufsize = 2 * size * n_filters;
        if ((bufsize) >= CUFFT_MAX_SIZE)
        {
            std::string ermsg = "ERROR: size of (image + kernel) * #_filters buffer " + std::to_string(bufsize) + " exceeds cuFFT's maximum " + std::to_string(CUFFT_MAX_SIZE);
#ifdef WITH_PYTHON_H
            throw invalid_argument(ermsg);
#endif

            std::cerr << ermsg << "\n";
            return false;
        }

        cufftDoubleComplex* result = NyxusGpu::gabor_result.hobuffer;
        cufftDoubleComplex* linear_kernel = NyxusGpu::gabor_linear_kernel.hobuffer;

        cufftDoubleComplex* d_image = NyxusGpu::gabor_linear_image.devbuffer;
        cufftDoubleComplex* d_result = NyxusGpu::gabor_result.devbuffer;
        cufftDoubleComplex* d_kernel = NyxusGpu::gabor_linear_kernel.devbuffer;
        cudaError_t ok;

        for (int batch = 0; batch < n_filters; ++batch)
        {
            size_t batch_idx = batch * size;

            int block = 16;
            dim3 tpb(block, block);
            dim3 bpg(ceil(row_size / block) + 1, ceil(col_size / block) + 1);

            re_2_complex_img << <bpg, tpb >> > (d_image, row_size, col_size, image_m, image_n, NyxusGpu::dev_imat1, batch_idx);

            CHECKERR(cudaDeviceSynchronize());
            CHECKERR(cudaGetLastError());
        }

        int n[2] = { row_size, col_size };

        bool ok2 = dense_2_padded_filterbank(d_kernel, dev_filterbank, image_m, image_n, kernel_m, kernel_n, n_filters);
        if (!ok2)
        {
            std::cerr << "ERROR: CuGabor::dense_2_padded_filterbank failed \n";
            return false;
        }

        cufftHandle plan;
        cufftHandle plan_k;

        int idist = size;
        int odist = size;

        int inembed[] = { row_size, col_size };
        int onembed[] = { row_size, col_size };

        int istride = 1;
        int ostride = 1;

        auto call = cufftPlanMany(&plan, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, n_filters);
        CHECKCUFFTERR(call);

        call = cufftPlanMany(&plan_k, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, n_filters);
        CHECKCUFFTERR(call);

        call = cufftExecZ2Z(plan, d_image, d_image, CUFFT_FORWARD);
        CHECKCUFFTERR(call);

        call = cufftExecZ2Z(plan_k, (cufftDoubleComplex*)d_kernel, (cufftDoubleComplex*)d_kernel, CUFFT_FORWARD);
        CHECKCUFFTERR(call);

        ok = cudaDeviceSynchronize();
        CHECKERR(ok);

        // element-wise multiplication of the image and kernel
        bool success = cmat_mult(d_image, row_size, col_size, d_kernel, d_result, n_filters);

        if (!success)
        {
            std::cerr << "ERROR: CuGabor::cmat_mult failed \n";
            return false;
        }

        // transform out of fourier space
        call = cufftExecZ2Z(plan, d_result, d_result, CUFFT_INVERSE);
        CHECKCUFFTERR(call);

        //
        // no need to memcpy(result<-d_result) above ^^^ !
        //

        for (int k = 0; k < n_filters; k++)
        {
            size_t padded_off = k * size,
                roi_off = k * image_m * image_n;

            int block = 16;
            dim3 tpb(block, block);
            dim3 bpg(ceil(image_m / block) + 1, ceil(image_n / block) + 1);

            /*
                    // out
                    PixIntens* nonpadded_e_montage,  // montage of smaller (nonpadded) images of size ROI_w*ROI_h
                    // in
                    cufftDoubleComplex* padded_montage, // montage of complex
                    size_t padded_frame_offset,
                    size_t padded_w,
                    size_t padded_h,
                    size_t roi_offset,
                    size_t roi_w,
                    size_t roi_h,
                    size_t kerside
             */
            kerCalcEnergy << < bpg, tpb >> > (
                // out
                NyxusGpu::gabor_energy_image.devbuffer,
                // in
                d_result,
                padded_off,
                row_size,   // padded w
                col_size,   // padded h
                roi_off,
                image_m,    // roi w
                image_n,    // roi h
                kernel_n);

            CHECKERR(cudaDeviceSynchronize());
            CHECKERR(cudaGetLastError());
        }

        // free device memory
        cufftDestroy(plan);
        cufftDestroy(plan_k);

        // download energy on host
        size_t szb = n_filters * image_m * image_n * sizeof(NyxusGpu::gabor_energy_image.hobuffer[0]);
        CHECKERR(cudaMemcpy(NyxusGpu::gabor_energy_image.hobuffer, NyxusGpu::gabor_energy_image.devbuffer, szb, cudaMemcpyDeviceToHost));

        return true;
    }

    bool send_filterbank_2_gpuside(double** dev_filterbank, const double* ho_filterbank, size_t filterbank_len_all_batches)
    {
        auto szb = sizeof(dev_filterbank[0]) * filterbank_len_all_batches;

        auto ok = cudaMalloc((void**)dev_filterbank, szb);
        CHECKERR(ok);

        ok = cudaMemcpy(*dev_filterbank, ho_filterbank, szb, cudaMemcpyHostToDevice);
        CHECKERR(ok);

        return true;
    }

    // X --> Y
    __global__ void ker_dense_2_padded(cufftDoubleComplex* Y, double* X, int y_rowsize, int y_colsize, int x_rowsize, int x_colsize, int batchsize)
    {
        int c = threadIdx.x + blockIdx.x * blockDim.x;
        int r = threadIdx.y + blockIdx.y * blockDim.y;
        int batch = threadIdx.z + blockIdx.z * blockDim.z;

        // dimensions of the padded image
        int row_size = y_rowsize + x_rowsize - 1;
        int col_size = y_colsize + x_colsize - 1;

        if (c >= row_size || r >= col_size || batch >= batchsize)
            return;

        int size = row_size * col_size;
        int batch_idx = batch * size;
        int batch_idx2 = batch * x_rowsize * x_colsize;

        int index = batch_idx + (c * col_size + r);
        int index2 = batch_idx2 + (c * x_colsize + r);

        if (c < x_rowsize && r < x_colsize)
        {
            Y[index].x = X[2 * index2];
            Y[index].y = X[2 * index2 + 1];
        }
        else
        {
            Y[index].x = 0.f;
            Y[index].y = 0.f;
        }
    }

    // X --> Y
    bool dense_2_padded_filterbank(cufftDoubleComplex* Y, double* X, int y_rowsize, int y_colsize, int x_rowsize, int x_colsize, int batchsize)
    {
        int rowsize = y_rowsize + x_rowsize - 1;
        int colsize = y_colsize + x_colsize - 1;

        int blo = 4;
        dim3 tpb(blo, blo, blo);
        dim3 bpg(ceil(rowsize / blo) + 1, ceil(colsize / blo) + 1, ceil(batchsize / blo) + 1);

        ker_dense_2_padded << < bpg, tpb >> > (Y, X, y_rowsize, y_colsize, x_rowsize, x_colsize, batchsize);

        cudaError_t ok = cudaDeviceSynchronize();
        CHECKERR(ok);

        // Check if kernel execution generated and error
        ok = cudaGetLastError();
        CHECKERR(ok);

        return true;
    }

}

