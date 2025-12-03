#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include "gpu.h"
#include "../cache.h"   //xxxxxxxxxx    #include "../gpucache.h"
#include "../features/pixel.h"
#include "geomoments.cuh"

namespace NyxusGpu
{
    bool sumreduce(
        gpureal* d_result,
        size_t cloudlen,
        double* d_prereduce,
        void* d_devreduce_tempstorage,
        size_t& devreduce_tempstorage_szb)
    {
        // CUB can't handle arrays of over 2 billion elements :-(
        OK(cloudlen < std::numeric_limits<int>::max());

        // Request the scratch space size and make sure that we have preallocated enough of it
        size_t szb;
        CHECKERR(cub::DeviceReduce::Sum(nullptr, szb, d_prereduce/*d_in*/, d_result, cloudlen/*num_items*/));
        if (devreduce_tempstorage_szb != szb)
        {
            // new size, new storage
            devreduce_tempstorage_szb = szb;
            CHECKERR(cudaFree(d_devreduce_tempstorage));
            CHECKERR(cudaMalloc(&d_devreduce_tempstorage, devreduce_tempstorage_szb));
        }

        // Run the actual sum-reduction
        CHECKERR(cub::DeviceReduce::Sum(d_devreduce_tempstorage, devreduce_tempstorage_szb, d_prereduce/*d_in*/, d_result, cloudlen/*num_items*/));

        return true;
    }

//*** NV reduce

    // Utility class used to avoid linker errors with extern
    // unsized shared memory arrays with templated type
    template <class T>
    struct SharedMemory {
        __device__ inline operator T* () {
            extern __shared__ int __smem[];
            return (T*)__smem;
        }

        __device__ inline operator const T* () const {
            extern __shared__ int __smem[];
            return (T*)__smem;
        }
    };

    // specialize for double to avoid unaligned memory
    // access compile errors
    template <>
    struct SharedMemory<double> {
        __device__ inline operator double* () {
            extern __shared__ double __smem_d[];
            return (double*)__smem_d;
        }

        __device__ inline operator const double* () const {
            extern __shared__ double __smem_d[];
            return (double*)__smem_d;
        }
    };

    /*
        This version uses sequential addressing -- no divergence or bank conflicts.
    */
    template <class T>
    __global__ void ker_sumreduceNV2 (
        // [in]
        T* g_odata, 
        // [out]
        const unsigned int n, 
        const T* g_idata) 
    {
        // Handle to thread block group
        T* sdata = SharedMemory<T>();   // cg::thread_block cta = cg::this_thread_block();

        // load shared mem
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[tid] = (i < n) ? g_idata[i] : 0;

        __syncthreads(); // cg::sync(cta);

        // do reduction in shared mem
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
        {
            if (tid < s) 
            {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads(); // cg::sync(cta);
        }

        // write result for this block to global mem
        if (tid == 0) 
            g_odata[blockIdx.x] = sdata[0];
    }

    // Instantiate the reduction function 

    template __global__ void ker_sumreduceNV2 <double>(
        // [in]
        double* g_odata,
        // [out]
        const unsigned int n,
        const double* g_idata);

    unsigned int nextPow2(unsigned int x) {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Compute the number of threads and blocks to use for the given reduction
    // kernel For the kernels >= 3, we set threads / block to the minimum of
    // maxThreads and n/2. For kernels < 3, we set to the minimum of maxThreads and
    // n.  For kernel 6, we observe the maximum specified number of blocks, because
    // each thread in that kernel can process a variable number of elements.
    ////////////////////////////////////////////////////////////////////////////////
    void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int& blocks, int& threads)
    {
        // get device capability, to avoid block/grid size exceed the upper bound
        cudaDeviceProp prop;
        int device;
        checkCudaErrors(cudaGetDevice(&device));
        checkCudaErrors(cudaGetDeviceProperties(&prop, device));

        if (whichKernel < 3) {
            threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
            blocks = (n + threads - 1) / threads;
        }
        else {
            threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
            blocks = (n + (threads * 2 - 1)) / (threads * 2);
        }

        if ((float)threads * blocks >
            (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
            printf("n is too large, please choose a smaller number!\n");
        }

        if (blocks > prop.maxGridSize[0]) {
            printf(
                "Grid size <%d> exceeds the device capability <%d>, set block size as "
                "%d (original %d)\n",
                blocks, prop.maxGridSize[0], threads * 2, threads);

            blocks /= 2;
            threads *= 2;
        }

        if (whichKernel >= 6) {
            blocks = std::min(maxBlocks, blocks);
        }
    }

    bool sumreduceNV2 (
    // [in]
    double* g_odata,
        // [out]
        const unsigned int n,
        double* g_idata,
        // compatibility with CUB-based sumreduce()
        void* unused1, 
        size_t unused2)
    {
        int numBlocks = 0;
        int numThreads = 0;
        getNumBlocksAndThreads(
            // [in]
            2/*whichKernel*/, n/*size*/, 1/*maxBlocks*/, 256/*maxThreads*/,
            // [out]
            numBlocks, numThreads);

        double* d_idata = g_idata,
            * d_odata = nullptr;
        cudaMalloc((void**)&d_odata, numBlocks * sizeof(d_odata[0]));
        
        //std::vector<double> odata(numBlocks, 0.0);
        //double* h_odata = odata.data();
        double* h_odata = (double*)malloc(numBlocks * sizeof(double));

        dim3 dimBlock(numThreads, 1, 1);
        dim3 dimGrid(numBlocks, 1, 1);
        // when there is only one warp per block, we need to allocate two warps
        // worth of shared memory so that we don't index shared memory out of bounds
        int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(double/*T*/) : numThreads * sizeof(double/*T*/);
        ker_sumreduceNV2<double/*T*/> << <dimGrid, dimBlock, smemSize >> > (d_odata, n/*size*/, d_idata);

        CHECKERR(cudaDeviceSynchronize());
        CHECKERR(cudaGetLastError());

        CHECKERR(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(d_odata[0]), cudaMemcpyDeviceToHost));

        // sum up partial sums on CPU:
        double totNV = 0.0;
        for (size_t i = 0; i < numBlocks; i++)
            totNV += h_odata[i];

        free(h_odata);
        h_odata = nullptr;

        CHECKERR(cudaMemcpy(g_odata, &totNV, sizeof(totNV), cudaMemcpyHostToDevice));
        CHECKERR(cudaFree(d_odata));

        return true;
    }

}