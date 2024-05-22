#include "gabor.cuh"
#include <vector>

using namespace std;

namespace CuGabor {
    
    __global__ void multiply(cufftDoubleComplex* A, int row_size, int col_size, cufftDoubleComplex* B, cufftDoubleComplex* result, int batch_size) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        int index = i*col_size*batch_size+j;
        double a,b,c,d;
        if (i < row_size && j < batch_size*col_size) {
            a = A[index].x;
            b = A[index].y;
            c = B[index].x;
            d = B[index].y;

            result[index].x = a*c - b*d;
            result[index].y = a*d + b*c;
        }
    }

    bool cmat_mult(cufftDoubleComplex* A, int row_size, int col_size, cufftDoubleComplex* B, cufftDoubleComplex* result, int batch_size){
        int block = 16;
        dim3 threadsPerBlock(block, block);
        dim3 blocksPerGrid(ceil(row_size/block)+1, ceil(col_size*batch_size/block)+1);

        multiply<<<blocksPerGrid, threadsPerBlock>>>(A, row_size, col_size, B, result, batch_size);

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

        int index = i*col_size*batch_size+j;
        double a,b,c,d;
        if (i < row_size && j < batch_size*col_size) {
            a = A[index].x;
            b = A[index].y;
            c = B[index].x;
            d = B[index].y;

            result[index].x = a*c - b*d;
            result[index].y = a*d + b*c;
        }
    }

    bool cmat_mult(CuComplex* A, int row_size, int col_size, CuComplex* B, CuComplex* result, int batch_size){
        int block = 16;
        dim3 threadsPerBlock(block, block);
        dim3 blocksPerGrid(ceil(row_size/block)+1, ceil(col_size*batch_size/block)+1);

        multiply<<<blocksPerGrid, threadsPerBlock>>>(A, row_size, col_size, B, result, batch_size);

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

                index = (i*col_size + j);
                linear_image[index].y = 0.f;

                if (i < image_m && j < image_n) { 

                    index2 = (i*image_n + j) ;
                    linear_image[index].x = image[index2];

                } else {
                    linear_image[index].x = 0.f; // add padding
                }
            }
        }

        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {

                index = (i*col_size + j);
                index2 = (i*kernel_n + j);

                if (i < kernel_m && j < kernel_n) {
                    linear_kernel[index].x = kernel[2*index2];
                } else {
                    linear_kernel[index].x = 0.f;
                    linear_kernel[index].y = 0.f;
                }
            }
        }

        for (int i = 0; i < row_size; ++i) {
            for (int j = 0; j < col_size; ++j) {
                index = (i*col_size + j);
            }
        }

        CuComplex* d_image;
        CuComplex* d_result;
        CuComplex* d_kernel;

        int n[2] = {row_size, col_size};

        auto ok = cudaMalloc((void**)&d_image, sizeof(CuComplex)*size*batch_size);
        CHECKERR(ok);

        ok = cudaMalloc((void**)&d_result, sizeof(CuComplex)*size*batch_size);
        CHECKERR(ok);
        
        ok = cudaMalloc((void**)&d_kernel, sizeof(CuComplex)*size*batch_size);  
        CHECKERR(ok);
        
        // copy data to GPU
        ok = cudaMemcpy(d_image, linear_image.data(), batch_size*size*sizeof(CuComplex), cudaMemcpyHostToDevice);
        CHECKERR(ok);

        ok = cudaMemcpy(d_kernel, linear_kernel.data(), batch_size*size*sizeof(CuComplex), cudaMemcpyHostToDevice);
        CHECKERR(ok);

        cufftHandle plan;
        cufftHandle plan_k;
        int idist = size;
        int odist = size;
        
        int inembed[] = {row_size, col_size};
        int onembed[] = {row_size, col_size};

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
        if(!success) {
            return false;
        }

        // transform out of fourier space
        call = cufftExecC2C(plan, d_result, d_result, CUFFT_INVERSE);
        CHECKCUFFTERR(call);

        // copy results from device to host
        ok = cudaMemcpy(result.data(), d_result, batch_size*size*sizeof(CuComplex), cudaMemcpyDeviceToHost); 
        CHECKERR(ok);

        // transfer to output array 
        for(int i = 0; i < size; ++i) {
            out[2*i] = (result[i].x/(size));
            out[2*i + 1] = (result[i].y/(size));
        }

        // free device memory
        cufftDestroy(plan);
        cufftDestroy(plan_k);
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);

        return true;                    
    }

    bool conv_dud_gpu_fft_multi_filter(double* out, 
                            const unsigned int* image, 
                            double* kernel, 
                            int image_n, int image_m, int kernel_n, int kernel_m, int batch_size,
                            double* dev_filterbank){
        
        typedef double2 Complex; // comment out to use float
        typedef cufftDoubleComplex CuComplex; // comment out to use float

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
        
        int batch_idx, batch_idx2;
        for (int batch = 0; batch < batch_size; ++batch) {

            batch_idx = batch * size;

            for (int i = 0; i < row_size; ++i) {
                for (int j = 0; j < col_size; ++j) {

                    index = batch_idx + (i*col_size + j);
                    linear_image[index].y = 0.f;

                    if (i < image_m && j < image_n) { 

                        index2 = (i*image_n + j);
                        linear_image[index].x = image[index2];

                    } else {
                        linear_image[index].x = 0; // add padding
                    }
                }
            }
        }

        CuComplex* d_image;
        CuComplex* d_result;
        CuComplex* d_kernel;

        int n[2] = {row_size, col_size};

        auto ok = cudaMalloc((void**)&d_image, sizeof(CuComplex)*size*batch_size);
        CHECKERR(ok);

        ok = cudaMalloc((void**)&d_result, sizeof(CuComplex)*size*batch_size);
        CHECKERR(ok);
        
        ok = cudaMalloc((void**)&d_kernel, sizeof(CuComplex)*size*batch_size);
        CHECKERR(ok);
        
        // copy data to GPU
        ok = cudaMemcpy(d_image, linear_image.data(), batch_size*size*sizeof(CuComplex), cudaMemcpyHostToDevice);
        CHECKERR(ok);
        
        bool ok2 = dense_2_padded_filterbank (d_kernel, dev_filterbank, image_m, image_n, kernel_m, kernel_n, batch_size);
        if (!ok2)
            return false;

        cufftHandle plan;
        cufftHandle plan_k;

        int idist = size;
        int odist = size;
        
        int inembed[] = {row_size, col_size};
        int onembed[] = {row_size, col_size};

        int istride = 1;
        int ostride = 1;

        auto call = cufftPlanMany(&plan, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch_size);
        CHECKCUFFTERR(call);

        call = cufftPlanMany(&plan_k, 2, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch_size);
        CHECKCUFFTERR(call);

        call = cufftExecZ2Z(plan, d_image, d_image, CUFFT_FORWARD);
        CHECKCUFFTERR(call);

        call = cufftExecZ2Z(plan_k, d_kernel, d_kernel, CUFFT_FORWARD);
        CHECKCUFFTERR(call);

        ok = cudaDeviceSynchronize();
        CHECKERR(ok);

        // element-wise multiplication of the image and kernel
        bool success = cmat_mult(d_image, row_size, col_size, d_kernel, d_result, batch_size);

        if(!success) {
            return false;
        }

        // transform out of fourier space
        call = cufftExecZ2Z(plan, d_result, d_result, CUFFT_INVERSE);
        CHECKCUFFTERR(call);

        // copy results from device to host
        ok = cudaMemcpy(result.data(), d_result, batch_size*size*sizeof(CuComplex), cudaMemcpyDeviceToHost);
        CHECKERR(ok); 

        // transfer to output array 
        for(int batch = 0; batch < batch_size; ++batch){
            batch_idx = batch*size;
            for(int i = 0; i < size; ++i) {
                out[2*batch_idx + 2*i] = (result[batch_idx + i].x/((double)size));
                out[2*batch_idx + 2*i + 1] = (result[batch_idx + i].y/((double)size));
            }
        }
        
        // free device memory
        cufftDestroy(plan);
        cufftDestroy(plan_k);
        cudaFree(d_image);
        cudaFree(d_result);
        cudaFree(d_kernel);

        return true;                    
    }

     bool send_filterbank_2_gpuside (double** dev_filterbank, const double* ho_filterbank, size_t filterbank_len_all_batches)
     {
         auto szb = sizeof(dev_filterbank[0]) * filterbank_len_all_batches;

         auto ok = cudaMalloc ((void**)dev_filterbank, szb);
         CHECKERR(ok);

         ok = cudaMemcpy (*dev_filterbank, ho_filterbank, szb, cudaMemcpyHostToDevice);
         CHECKERR(ok);

         return true;
     }

     // X --> Y
     __global__ void ker_dense_2_padded (cufftDoubleComplex* Y, double* X, int y_rowsize, int y_colsize, int x_rowsize, int x_colsize, int batchsize)
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
     bool dense_2_padded_filterbank (cufftDoubleComplex* Y, double* X, int y_rowsize, int y_colsize, int x_rowsize, int x_colsize, int batchsize)
     {
         int rowsize = y_rowsize + x_rowsize - 1;
         int colsize = y_colsize + x_colsize - 1;

         int blo = 4;
         dim3 tpb (blo, blo, blo);
         dim3 bpg (ceil(rowsize / blo) + 1, ceil(colsize / blo) + 1, ceil(batchsize / blo) + 1);

         ker_dense_2_padded <<< bpg, tpb >>> (Y, X, y_rowsize, y_colsize, x_rowsize, x_colsize, batchsize);

         cudaError_t ok = cudaDeviceSynchronize();
         CHECKERR(ok);

         // Check if kernel execution generated and error
         ok = cudaGetLastError();
         CHECKERR(ok);

         return true;
     }

}

