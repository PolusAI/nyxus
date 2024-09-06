#pragma once

#include <cufftXt.h>
#include "features/pixel.h"
#include <unordered_map>

// 1 instance of this per each ROI cached in device memory
enum GpusideState
{
	// * * * * *	erosion

	N_EROSIONS = 0,
	N_EROSIONS_COMPLEMENT,
	
	// * * * * *	2D geometric moments

	FEATURE_GEOMOM_STATUS,
	ORGX,
	ORGY,
	TMPRM00,
	TMPRM10,
	TMPRM01,
	// raw moments
	RM00, RM01, RM02, RM03,
	RM10, RM11, RM12, RM13,
	RM20, RM21, RM22, RM23,
	RM30,
	RM31,	//
	RM32,	// used by NRM_31-33
	RM33,	//
	// central moments
	CM00, CM01, CM02, CM03,
	CM10, CM11, CM12, CM13,
	CM20, CM21, CM22, CM23,
	CM30, CM31, CM32, CM33,
	// normed raw moments
	W00, W01, W02, W03,
	W10, W11, W12, W13,
	W20, W21, W22, W23,
	W30, W31, W32, W33,
	// normed central moments
	NU02, NU03, NU11, NU12, NU20, NU21, NU30,
	// Hu
	H1, H2, H3, H4, H5, H6, H7,
	// weighted origins
	WORGX,
	WORGY,
	// weighted raw moments
	WRM00, WRM01, WRM02, WRM03,
	WRM10, WRM11, WRM12,
	WRM20, WRM21, 
	WRM30,
	// weighted central moments
	WCM00, // necessary for WNCM_pq
	WCM02, WCM03,
	WCM11, WCM12,
	WCM20, WCM21,
	WCM30,
	// weighted normed central moments
	WNU02, WNU03, 
	WNU11, WNU12, 
	WNU20, WNU21, 
	WNU30,
	// weighted Hu
	WH1, WH2, WH3, WH4, WH5, WH6, WH7,
	__COUNT__
};


class LR;

template <class T> 
class GpuCache
{
public:
	T* hobuffer = nullptr;
	T* devbuffer = nullptr;
	size_t* ho_lengths = nullptr;
	size_t* ho_offsets = nullptr;
	size_t* dev_offsets = nullptr;
	size_t total_len, num_rois;
	bool alloc(size_t total_len) { OK(alloc(total_len, 1)); return true; };
	bool alloc (size_t total_len, size_t num_rois);
	static size_t szb (size_t total_len, size_t num_rois);
	bool upload();
	bool download();
	bool clear();
	~GpuCache() { clear(); }
};

namespace NyxusGpu
{

#ifdef USE_GPU

	// state buffer shared by all GPU-enabled features
	extern GpuCache<gpureal> gpu_featurestatebuf;	// n_rois * GeomomentsState::__COUNT__

	extern GpuCache<Pixel2> gpu_roiclouds_2d;	// continuous ROI clouds
	extern GpuCache<Pixel2> gpu_roicontours_2d;	// continuous ROI contours

	// helper buffers of 2D moments features
	extern RealPixIntens* dev_realintens;	// max cloud size over batch
	extern double* dev_prereduce;	// --"--
	extern void* dev_devicereduce_temp_storage;	// allocated [] elements by cub::DeviceReduce::Sum()
	extern size_t devicereduce_temp_storage_szb;
	extern size_t gpu_batch_len;

	// helper buffers of erosion features (single instance shared by all batch ROIs)
	extern PixIntens* dev_imat1;
	extern PixIntens* dev_imat2;

	// helper buffers of Gabor
	extern GpuCache <cufftDoubleComplex> gabor_linear_image; // (img_plus_ker_size* n_filters); // ROI + kernel image
	extern GpuCache <cufftDoubleComplex> gabor_result; // (img_plus_ker_size* n_filters);
	extern GpuCache <cufftDoubleComplex> gabor_linear_kernel; // (img_plus_ker_size* n_filters);
	extern GpuCache <PixIntens> gabor_energy_image; // (img_plus_ker_size* n_filters);

	// these need to be called after "prescan"
	bool gpu_get_free_mem(size_t& amt);
	bool allocate_gpu_cache(
		// out
		GpuCache<Pixel2>& clouds,
		GpuCache<Pixel2>& konturs,
		RealPixIntens** realintens,
		double** prereduce,
		GpuCache<gpureal>& state,
		size_t& devicereduce_buf_szb,
		void** devicereduce_buf,
		size_t& gpu_batch_len,
		PixIntens** imat1,
		PixIntens** imat2,
		GpuCache <cufftDoubleComplex> & gabor_linear_image,
		GpuCache <cufftDoubleComplex> & gabor_result,
		GpuCache <cufftDoubleComplex> & gabor_linear_kernel,
		GpuCache <PixIntens> & gabor_energy,
			// in
		size_t roi_cloud_len,
		size_t roi_kontur_cloud_len,
		size_t n_rois,
		size_t roi_area,
		size_t roi_w,
		size_t roi_h,
		int gabor_n_filters,
		int gabor_ker_side);
	bool free_gpu_cache(
		GpuCache<Pixel2>& clouds,
		GpuCache<Pixel2>& konturs,
		RealPixIntens*& realintens,
		double*& prereduce,
		GpuCache<gpureal>& intermediate,
		void*& tempstorage,
		PixIntens* imat1,
		PixIntens* imat2,
		GpuCache <cufftDoubleComplex>& gabor_linear_image,
		GpuCache <cufftDoubleComplex>& gabor_result,
		GpuCache <cufftDoubleComplex>& gabor_linear_kernel,
		GpuCache <PixIntens> & gabor_energy_image
	);

	// these need to be called in "reduce_trivial"
	void send_roi_data_gpuside(const std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData, size_t off_this_batch, size_t actual_batch_len);
	void free_roi_data_gpuside();
	void send_roi_batch_data_2_gpu(
		// out
		GpuCache<Pixel2>& cloud,
		GpuCache<Pixel2>& contours,
		RealPixIntens** real_intens,
		double** prereduce,
		GpuCache<gpureal>& intermediate,
		size_t& devicereduce_buf_szb,
		void** devicereduce_buf,
		// in
		const std::vector<int>& labels,
		std::unordered_map <int, LR>& roi_data,
		size_t batch_offset,
		size_t batch_len);

#endif

} // NyxusGpu
