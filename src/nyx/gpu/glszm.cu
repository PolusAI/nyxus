#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <iostream>

#include "../features/pixel.h"

#include "gpu.h"
#include "../cache.h"		//xxxxxxxxxxxx	#include "../gpucache.h"

#include "../helpers/timing.h"

namespace NyxusGpu
{
	// per-ROI device-side buffer
	enum GLSZM_state
	{
		// helper totals
		f_LAHGLE = 0,
		f_LALGLE,
		f_SAHGLE,
		f_SALGLE,
		f_ZE,
		mu_ZV,
		mu_GLV,
		// features depending on Ng/SI
		f_SAE,
		f_LAE,
		f_SZN,
		f_SZNN,
		f_LGLZE,
		f_HGLZE,
		// features depending on Ns/SJ
		f_GLN,
		f_GLNN,
		// features depending on Ns and Ng
		f_GLV,
		f_ZV,
		// features derived from totals
		f_ZP,
		__COUNT__
	};

	// returns 0-based row-major index of 1-based c[olumn] and r[ow] indices
	// 1st argument - column
	// 2nd argument - row
	__device__ size_t yx_matlab (size_t r, size_t c, size_t w)
	{
		return (r-1) * w + (c-1);
	}

	// expects the wide thread layout
	// <<<rows:Ns, cols:Ng>>>
	__global__ void kerCalcSiSj(
		// out
		float* state,
		float* SJ,
		float* SI,
		// in
		const PixIntens* I,
		const int* P,
		int Ns,
		int Ng,
		double sum_p,
		double EPS)
	{
		// Ns dimension for (int j = 1; j <= Ns; ++j)
		int j = threadIdx.x + blockIdx.x * blockDim.x; 
		if (j == 0 || j > Ns)
			return;

		// Ng dimension for (int i = 1; i <= Ng; ++i)
		int i = threadIdx.y + blockIdx.y * blockDim.y;
		if (i == 0 || i > Ng)
			return;

		double inten = (double)I[i - 1];
		float p = P[yx_matlab(i, j, Ns)];

		atomicAdd (&SI[i], p);
		atomicAdd (&SJ[j], p);

		// Once we're iterating matrix P, let's compute specialized sums
		double i2 = inten * inten,
			j2 = double(j) * double(j);

		float f_LAHGLE_ = p* i2* j2;
		atomicAdd(&state[f_LAHGLE], f_LAHGLE_);

		float f_LALGLE_ = p * j2 / i2;
		atomicAdd(&state[f_LALGLE], f_LALGLE_);

		float f_SAHGLE_ = p * i2 / j2;
		atomicAdd(&state[f_SAHGLE], f_SAHGLE_);

		float f_SALGLE_ = p / (i2 * j2);
		atomicAdd(&state[f_SALGLE], f_SALGLE_);

		float entrTerm = std::log2(p / sum_p + EPS);
		float f_ZE_ = p / sum_p * entrTerm;
		atomicAdd(&state[f_ZE], f_ZE_);

		float mu_ZV_ = p / sum_p * double(j);
		atomicAdd(&state[mu_ZV], mu_ZV_);

		float mu_GLV_ = p / sum_p * double(inten);
		atomicAdd(&state[mu_GLV], mu_GLV_);
	}

	__global__ void kerFeaturesOfSJ (float* state, float* SJ, int Ns)
	{
		// Ns dimension for (int j = 1; j <= Ns; ++j)
		int j = threadIdx.x + blockIdx.x * blockDim.x;
		if (j == 0 || j > Ns)
			return;

		float s = SJ[j];
		atomicAdd(&state[f_SAE], s / (j * j));
		atomicAdd(&state[f_LAE], s * (j * j));
		atomicAdd(&state[f_SZN], s * s);
		atomicAdd(&state[f_SZNN], s * s);
	}

	__global__ void kerFeaturesOfSI (float* state, const float* SI, const PixIntens* I, int Ng, double sum_p)
	{
		// Ns dimension for (int j = 1; j <= Ng; ++j)
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i == 0 || i > Ng)
			return;

		float x = SI[i - 1];
		atomicAdd(&state[f_GLN], x * x);
		atomicAdd(&state[f_GLNN], x * x);

		double inten = (double) I[i - 1];
		x = SI[i];
		atomicAdd (&state[f_LGLZE], x / (inten * inten));
		atomicAdd (&state[f_HGLZE], x * inten * inten);
	}

	__global__ void kerFeaturesByNsNg(
		float* state,
		float* SJ,
		float* SI,
		const PixIntens* I,
		const int* P,
		int Ns,
		int Ng,
		double sum_p,
		double EPS)
	{
		// Ns dimension for (int j = 1; j <= Ns; ++j)
		int j = threadIdx.x + blockIdx.x * blockDim.x;
		if (j == 0 || j > Ns)
			return;

		// Ng dimension for (int i = 1; i <= Ng; ++i)
		int i = threadIdx.y + blockIdx.y * blockDim.y;
		if (i == 0 || i > Ng)
			return;

		double inten = I[i - 1];
		double p = P[yx_matlab(i, j, Ns)];

		// GLV
		double d2 = (inten - mu_GLV) * (inten - mu_GLV);
		atomicAdd (&state[f_GLV], float(p / sum_p * d2));
		// ZV
		double mu2 = (double(j) - mu_ZV) * (double(j) - mu_ZV);
		atomicAdd (&state[f_ZV], p / sum_p * mu2);
	}

	void finalizeGLSZM (float* state, int Ns, int Ng, double sum_p, size_t nnz, double EPS)
	{
		state[f_SAE] = double(state[f_SAE]) / sum_p;
		state[f_LAE] = double(state[f_LAE]) / sum_p;
		state[f_GLN] = double(state[f_GLN]) /sum_p;
		state[f_GLNN] = double(state[f_GLNN]) / (sum_p * sum_p);
		state[f_SZN] = double(state[f_SZN]) / sum_p;
		state[f_SZNN] = double(state[f_SZNN]) / (sum_p * sum_p);
		state[f_ZP] = sum_p / double(nnz);
		state[f_ZE] *= -1;
		state[f_LGLZE] = double(state[f_LGLZE]) / sum_p;
		state[f_HGLZE] = double(state[f_HGLZE]) / sum_p;
		state[f_SALGLE] = double(state[f_SALGLE]) / sum_p;
		state[f_SAHGLE] = double(state[f_SAHGLE]) / sum_p;
		state[f_LALGLE] = double(state[f_LALGLE]) / sum_p;
		state[f_LAHGLE] = double(state[f_LAHGLE]) / sum_p;
	}

bool GLSZMfeature_calc (
	// out
	double& fv_SAE,
	double& fv_LAE,
	double& fv_GLN,
	double& fv_GLNN,
	double& fv_SZN,
	double& fv_SZNN,
	double& fv_ZP,
	double& fv_GLV,
	double& fv_ZV,
	double& fv_ZE,
	double& fv_LGLZE,
	double& fv_HGLZE,
	double& fv_SALGLE,
	double& fv_SAHGLE,
	double& fv_LALGLE,
	double& fv_LAHGLE,
	// in
	int Ng, 
	int Ns, 
	PixIntens* I,	// [Ng]
	int* P,	// [Ng * Ns]
	double sum_p, 
	size_t nnz,
	double EPS)	
	{
		GpuCache<float> gcSI;
		GpuCache<float> gcSJ;
		PixIntens* dev_I;
		int* dev_P;
		GpuCache<float> gcState;

		// allocate GPU-side 
		//		1
		size_t n = Ns * Ng,
			szbp = n * sizeof(P[0]);
		CHECKERR(cudaMalloc(&dev_P, szbp));
		//		2
		OK(gcSI.alloc(Ng+1));
		//		3
		OK(gcSJ.alloc(Ns+1));
		//		4
		size_t szbI = Ng * sizeof(I[0]);
		CHECKERR(cudaMalloc(&dev_I, szbI));
		//		5
		OK (gcState.alloc(GLSZM_state::__COUNT__))

		// upload P and I
		CHECKERR(cudaMemcpy(dev_P, P, szbp, cudaMemcpyHostToDevice));
		CHECKERR(cudaMemcpy(dev_I, I, szbI, cudaMemcpyHostToDevice));

		// zero SJ, SI, state
		size_t szbSJ = gcSJ.total_len * sizeof(gcSJ.devbuffer[0]);
		CHECKERR(cudaMemset(gcSJ.devbuffer, 0, szbSJ));

		size_t szbSI = gcSI.total_len * sizeof(gcSI.devbuffer[0]);
		CHECKERR(cudaMemset(gcSI.devbuffer, 0, szbSI));

		size_t szbState = GLSZM_state::__COUNT__ * sizeof(gcState.devbuffer[0]);
		CHECKERR(cudaMemset(gcState.devbuffer, 0, szbState));

		// calculate helper totals
		int block = 16;
		dim3 ttt (block, block);
		dim3 bbb (ceil( (Ns+1) / block) + 1, ceil( (Ng+1) / block) + 1);	// wide thread layout
		kerCalcSiSj <<<bbb, ttt>>> (gcState.devbuffer, gcSJ.devbuffer, gcSI.devbuffer, dev_I, dev_P, Ns, Ng, sum_p, EPS);

		CHECKERR(cudaDeviceSynchronize());
		CHECKERR(cudaGetLastError());

		// calculate features
		//		-by SI (GLN, GLNN, LGLZE, HGLZE)
		int nblo = whole_chunks2(Ng+1, blockSize);
		kerFeaturesOfSI << < nblo, blockSize >> > (gcState.devbuffer, gcSI.devbuffer, dev_I, Ng, sum_p);
		CHECKERR(cudaDeviceSynchronize());
		CHECKERR(cudaGetLastError());
		//		-by SJ (SAE, LAE, SZN, SZNN)
		nblo = whole_chunks2(Ns + 1, blockSize);
		kerFeaturesOfSJ << < nblo, blockSize >> > (gcState.devbuffer, gcSJ.devbuffer, Ns);
		CHECKERR(cudaDeviceSynchronize());
		CHECKERR(cudaGetLastError());

		//		-by Ns and Ng (GLV, ZV)
		kerFeaturesByNsNg <<<bbb, ttt>>> (gcState.devbuffer, gcSJ.devbuffer, gcSI.devbuffer, dev_I, dev_P, Ns, Ng, sum_p, EPS);
		CHECKERR(cudaDeviceSynchronize());
		CHECKERR(cudaGetLastError());

		// download partial results
		OK(gcState.download());

		// finalize calculating features
		finalizeGLSZM (gcState.hobuffer, Ns, Ng, sum_p, nnz, EPS);

		fv_SAE = gcState.hobuffer[f_SAE];
		fv_LAE = gcState.hobuffer[f_LAE];
		fv_GLN = gcState.hobuffer[f_GLN];
		fv_GLNN = gcState.hobuffer[f_GLNN];
		fv_SZN = gcState.hobuffer[f_SZN];
		fv_SZNN = gcState.hobuffer[f_SZNN];
		fv_ZP = gcState.hobuffer[f_ZP];
		fv_GLV = gcState.hobuffer[f_GLV];
		fv_ZV = gcState.hobuffer[f_ZV];
		fv_ZE = gcState.hobuffer[f_ZE];
		fv_LGLZE = gcState.hobuffer[f_LGLZE];
		fv_HGLZE = gcState.hobuffer[f_HGLZE];
		fv_SALGLE = gcState.hobuffer[f_SALGLE];
		fv_SAHGLE = gcState.hobuffer[f_SAHGLE];
		fv_LALGLE = gcState.hobuffer[f_LALGLE];
		fv_LAHGLE = gcState.hobuffer[f_LAHGLE];

		// deallocate
		CHECKERR(cudaFree(dev_P));
		CHECKERR(cudaFree(dev_I));
		OK(gcSI.clear());
		OK(gcSJ.clear());
		OK(gcState.clear());

		return true;
	}
}