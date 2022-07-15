#include <cufftXt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "gpu.h"

typedef float2 Complex;
typedef cufftComplex CuComplex;

#define CUFFT_MAX_SIZE pow(2,27)

namespace CuGabor{

    __global__ void multiply(
        CuComplex* A, 
        int row_size, 
        int col_size, 
        CuComplex* B, 
        CuComplex* result, 
        int batch_size);

    /**
     * @brief Multiply two complex matrices
     * 
     */
    bool cmat_mult(
        CuComplex* A, 
        int row_size, 
        int col_size, 
        CuComplex* B, 
        CuComplex* result, 
        int batch_size);

    /**
     * @brief Multiply two complex matrices
     * 
     */
    bool cmat_mult(
        cufftDoubleComplex* A, 
        int row_size, 
        int col_size, 
        cufftDoubleComplex* B, 
        cufftDoubleComplex* result, 
        int batch_size);

    /**
     * @brief Compute the convolution of an image using cuFFT on gpu
     * 
     * @param out Results array
     * @param image Image to convolve
     * @param kernel Kernel
     * @param image_n Widht of image
     * @param image_m Heigh of image
     * @param kernel_n Width of kernel
     * @param kernel_m Heigh of image
     */
    bool conv_dud_gpu_fft(double* out, // must be zeroed
                    const unsigned int* image, 
                    double* kernel, 
                    int image_n, int image_m, int kernel_n, int kernel_m);

    /**
     * @brief Compute the convolution of multiple images using cuFFT on gpu.
     * 
     * Images must be the same height and width.
     * 
     * @param out Results array
     * @param image Images to convolve
     * @param kernel Kernels
     * @param image_n Width of images
     * @param image_m Height of images
     * @param kernel_n Width of kernels
     * @param kernel_m Height of kernels
     * @param batch_size Number of images
     */
    bool conv_dud_gpu_fft_multi_filter(double* out, 
                    const unsigned int* image, 
                    double* kernel, 
                    int image_n, int image_m, int kernel_n, int kernel_m, int batch_size);
}