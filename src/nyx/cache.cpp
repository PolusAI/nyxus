#ifdef USE_GPU

#include <stdexcept>
#include "cache.h"
#include "roi_cache.h"

#define OKDETL(x,detl) if (x == false) \
{ \
	return std::string("error due to ") + detl + " at " + __FILE__ + ":" + std::to_string(__LINE__); \
} \

namespace
{
	// Border ends the contour buffers hold: one per border, at most the bound the caller gives and,
	// since every border holds a contour pixel, at most the contour buffer's capacity
	size_t kontur_borders_capacity (size_t roi_kontur_borders, size_t kontur_capacity)
	{
		return (std::max) ((std::min) (roi_kontur_borders, kontur_capacity), size_t(1));
	}
}


	size_t GpusideCache::ram_comsumption_szb (
		bool needContour,
		bool needErosion,
		bool needGabor,
		bool needMoments,
		size_t roi_cloud_len,
		size_t roi_kontur_cloud_len,
		size_t roi_kontur_borders,
		size_t n_rois,
		size_t roi_w,
		size_t roi_h,
		int n_gabor_filters,
		int gabor_ker_side)
	{
		size_t szb_clo = GpuCache<Pixel2>::/*clouds.*/szb(roi_cloud_len, n_rois),
			szb_state = GpuCache<gpureal>::/*state.*/szb(GpusideState::__COUNT__, n_rois);

		size_t szb_kon = 0;
		if (needContour)
		{
			// the contour pixels, and where each border's run of them ends, one end per border
			szb_kon = GpuCache<Pixel2>::/*konturs.*/szb(roi_kontur_cloud_len, n_rois) +
				GpuCache<size_t>::/*kontur_borders.*/szb(kontur_borders_capacity(roi_kontur_borders, roi_cloud_len), n_rois);
		}

		size_t szb_mom_ri = 0,
			szb_moms_pr = 0,
			szb_moms_dr = 0;

		if (needMoments)
		{
			szb_mom_ri = sizeof(RealPixIntens) * roi_cloud_len;
			szb_moms_pr = sizeof(double) * roi_cloud_len * 16;
			szb_moms_dr = sizeof(void*/*devicereduce_buf[0]*/) * roi_cloud_len;	// always enough
		}

		size_t szb_imat = 0;
		if (needErosion)
		{
			szb_imat = 2 * sizeof(StatsInt) * roi_w * roi_h;		// erosion requires 2 matrices
		}

		size_t szb_gabor1 = 0,
			szb_gabor2 = 0;
		if (needGabor)
		{
			szb_gabor1 = 3 * sizeof(cufftDoubleComplex) * n_gabor_filters * (roi_w + gabor_ker_side - 1) * (roi_h + gabor_ker_side - 1);	// 3 arrays
			szb_gabor2 = sizeof(PixIntens) * n_gabor_filters * (roi_w + gabor_ker_side - 1) * (roi_h + gabor_ker_side - 1);
		}

		size_t szb = szb_clo +
			szb_kon +
			szb_mom_ri +
			szb_moms_pr +
			szb_state +
			szb_moms_dr +
			szb_imat +
			szb_gabor1 +
			szb_gabor2;

		return szb;
	}

	std::optional<std::string> GpusideCache::allocate_gpu_cache(
		// out
		GpuCache<Pixel2>& clouds,	// geo moments
		GpuCache<Pixel2>& konturs,
		GpuCache<size_t>& kontur_borders,
		RealPixIntens** realintens,
		double** prereduce,
		GpuCache<gpureal>& intermediate,
		size_t& devicereduce_buf_szb,
		void** devicereduce_buf,
		size_t& batch_len,
		PixIntens** imat1,				// erosion
		PixIntens** imat2,				// (imat1 is shared by erosion and Gabor)
		GpuCache <cufftDoubleComplex>& gabor_linear_image,	// gabor
		GpuCache <cufftDoubleComplex>& gabor_result,
		GpuCache <cufftDoubleComplex>& gabor_linear_kernel,
		GpuCache <RealPixIntens>& gabor_energy_image,
		// in
		bool needContour,
		bool needErosion,
		bool needGabor,
		bool needMoments,
		size_t roi_cloud_len,
		size_t roi_kontur_cloud_len,
		size_t roi_kontur_borders,
		size_t n_rois,
		size_t roi_area,
		size_t roi_w,
		size_t roi_h,
		int n_gabor_filters,
		int gabor_ker_side)
	{
		// the cache as an earlier run left it, every buffer that is set: a run that failed may not have
		// freed it, and a buffer kept would have the size that run gave it
		OKDETL(free_gpu_cache(clouds, konturs, kontur_borders, *realintens, *prereduce, intermediate, *devicereduce_buf, *imat1, *imat2,
			gabor_linear_image, gabor_result, gabor_linear_kernel, gabor_energy_image), "freeing the previous GPU cache");

		//****** plan GPU memory
		size_t amt = 0;
		OKDETL(gpu_get_free_mem(amt), "gpu_get_free_mem()");

		int n_gabFilters = n_gabor_filters + 1;		// '+1': an extra filter for the baseline signal

		// Calculate the amt of required memory
		size_t ccl0 = roi_area * n_rois;	// combined cloud length, initial
		size_t szb = ram_comsumption_szb(
			needContour,
			needErosion,
			needGabor,
			needMoments,
			ccl0,
			roi_kontur_cloud_len,
			roi_kontur_borders,
			n_rois, roi_w, roi_h, n_gabFilters, gabor_ker_side);

		batch_len = n_rois;
		size_t critAmt = float(amt) * 0.75; // 75% GPU RAM as critical RAM

		if (critAmt < szb)
		{
			size_t try_nrois = 0;
			for (try_nrois = n_rois; try_nrois >= 0; try_nrois--)
			{
				// failed to find a batch ?
				if (try_nrois == 0)
					return "cannot rightsize a ROI batch: even 1 ROI requires over critical memory amount " + std::to_string(critAmt);

				size_t ccl = roi_area * try_nrois;	// combined cloud length
				size_t try_szb = ram_comsumption_szb(
					needContour,
					needErosion,
					needGabor,
					needMoments,
					ccl, ccl, roi_kontur_borders, try_nrois, roi_w, roi_h, n_gabFilters, gabor_ker_side);

				if (critAmt > try_szb)
				{
					batch_len = try_nrois;
					break;
				}
			}

			// have we found a compromise ?
			if (batch_len < n_rois)
			{
				return "cannot rightsize a ROI batch: batch_len " + std::to_string(batch_len) + " < n_rois " + std::to_string(n_rois);
			}
		}

		size_t batch_roi_cloud_len = roi_area * batch_len;

		//****** allocate: on any failure what was allocated is freed, so the cache is either ready or empty

		auto allocate_buffers = [&] () -> std::optional<std::string>
		{
			// ROI clouds (always on)

			OKDETL(clouds.alloc(batch_roi_cloud_len, batch_len), "clouds.alloc()");

			// contours

			if (needContour)
			{
				OKDETL(konturs.alloc(batch_roi_cloud_len, batch_len), "konturs.alloc()");

				// one end per border, as many as the plan counts
				OKDETL(kontur_borders.alloc(kontur_borders_capacity(roi_kontur_borders, batch_roi_cloud_len), batch_len), "kontur_borders.alloc()");
			}

			// moments

			if (needMoments)
			{
				// moments / real intensities
				OKDETL(allocate_on_device((void**)realintens, sizeof(RealPixIntens) * batch_roi_cloud_len), "allocate_on_device(moments1)");

				// moments / pre-reduce
				OKDETL(allocate_on_device((void**)prereduce, sizeof(double) * batch_roi_cloud_len * 16), "allocate_on_device(moments2)");	// 16 is the max number of simultaneous totals calculated by a kernel, e.g. RM00-33

				// moments / intermediate
				OKDETL(intermediate.alloc(GpusideState::__COUNT__, batch_len), "allocate_on_device(moments3)");

				// moments / CUB DeviceReduce's temp buffer
				OKDETL(devicereduce_evaluate_buffer_szb(devicereduce_buf_szb, batch_roi_cloud_len), "allocate_on_device(moments4)");
				OKDETL(allocate_on_device((void**)devicereduce_buf, devicereduce_buf_szb), "allocate_on_device(moments5)");
			}

			// erosion / image matrices 1 and 2

			if (needErosion)
			{
				// imat1 is shared by erosion and Gabor: allocated by whichever of the two comes first
				if (!*imat1)
					OKDETL(allocate_on_device((void**)imat1, sizeof(imat1[0]) * roi_w * roi_h), "allocate_on_device(eros1)");

				OKDETL(allocate_on_device((void**)imat2, sizeof(imat2[0]) * roi_w * roi_h), "allocate_on_device(eros2)");
			}

#ifdef NYXUS_GPU_TEST_HOOKS
			if (test_fail_before_gabor)
				return std::string("allocation stopped before the Gabor buffers, as a test asked");
#endif

			// Gabor

			if (needGabor)
			{
				// imat1 is shared by erosion and Gabor
				if (!*imat1)
					OKDETL(allocate_on_device((void**)imat1, sizeof(imat1[0]) * roi_w * roi_h), "allocate_on_device(gabor)");

				size_t gabTotlen = (roi_w + gabor_ker_side - 1) * (roi_h + gabor_ker_side - 1) * n_gabFilters;
				OKDETL(gabor_linear_image.alloc(gabTotlen, 1), "gabor_linear_image.alloc()");
				OKDETL(gabor_result.alloc(gabTotlen, 1), "gabor_result.alloc()");
				OKDETL(gabor_linear_kernel.alloc(gabTotlen, 1), "gabor_linear_kernel.alloc()");
				OKDETL(gabor_energy_image.alloc(gabTotlen, 1), "gabor_energy_image.alloc()");
			}

			return std::nullopt;
		};

		if (auto err = allocate_buffers())
		{
			free_gpu_cache(clouds, konturs, kontur_borders, *realintens, *prereduce, intermediate, *devicereduce_buf, *imat1, *imat2,
				gabor_linear_image, gabor_result, gabor_linear_kernel, gabor_energy_image);
			return err;
		}

		using_contour = needContour;
		return std::nullopt;
	}

	void GpusideCache::send_roi_batch_data_2_gpu(
		// out
		GpuCache<Pixel2>& clouds,
		GpuCache<Pixel2>& konturs,
		GpuCache<size_t>& kontur_borders,
		RealPixIntens** realintens,
		double** prereduce,
		GpuCache<gpureal>& intermediate,
		size_t& devicereduce_buf_szb,
		void** devicereduce_buf,
		// in
		const std::vector<int>& labels,
		std::unordered_map <int, LR>& roi_data,
		size_t batch_offset,
		size_t batch_len)
	{
		// The batch is ROIs batch_offset .. batch_offset + batch_len - 1. A failure throws: the GPU-side
		// features would otherwise read buffers this batch left unfilled.

		//****** clouds as a solid buffer
		size_t off = 0;
		size_t roiIdx = 0;
		for (size_t i = batch_offset; i < batch_offset + batch_len; i++)
		{
			int lbl = labels[i];
			LR& roi = roi_data[lbl];

			if (! clouds.hobuffer || off + roi.raw_pixels.size() > clouds.total_len)
				throw std::runtime_error ("ROI " + std::to_string(lbl) + ": the batch's pixels exceed the GPU cloud buffer");

			clouds.ho_lengths[roiIdx] = roi.raw_pixels.size();
			clouds.ho_offsets[roiIdx] = off;
			roiIdx++;

			// save to buffer fixingb pixel coordinates
			auto xmin = roi.aabb.get_xmin();
			auto ymin = roi.aabb.get_ymin();
			for (Pixel2 p : roi.raw_pixels)	// copy, not reference
			{
				p.x -= xmin;
				p.y -= ymin;
				clouds.hobuffer[off++] = p;
			}
		}

		//****** contours as a solid buffer

		if (using_contour)
		{
			off = 0;
			roiIdx = 0;
			size_t offEnds = 0;
			for (size_t i = batch_offset; i < batch_offset + batch_len; i++)
			{
				int lbl = labels[i];
				LR& roi = roi_data[lbl];

				const std::vector<Pixel2>& K = roi.contour_pixels_;
				const std::vector<size_t>& ends = roi.contour_border_ends_;

				// the kernels read the borders as the CPU searches do (BorderedContour), and a buffer
				// sized by the memory plan holds them only if every border holds a contour pixel
				if (!border_ends_cover (K.size(), ends))
					throw std::runtime_error ("ROI " + std::to_string(lbl) + ": contour border ends do not cover its " + std::to_string(K.size()) + " contour pixels");
				if (! konturs.hobuffer || ! kontur_borders.hobuffer || off + K.size() > konturs.total_len || offEnds + ends.size() > kontur_borders.total_len)
					throw std::runtime_error ("ROI " + std::to_string(lbl) + ": the batch's contours exceed the GPU contour buffers");

				konturs.ho_lengths[roiIdx] = K.size();
				konturs.ho_offsets[roiIdx] = off;

				// where each border's run ends, relative to the ROI's first contour pixel
				kontur_borders.ho_offsets[roiIdx] = offEnds;
				kontur_borders.ho_lengths[roiIdx] = ends.size();
				for (size_t end : ends)
					kontur_borders.hobuffer[offEnds++] = end;
				roiIdx++;

				// save to buffer fixing pixel coordinates
				auto xmin = roi.aabb.get_xmin();
				auto ymin = roi.aabb.get_ymin();
				for (Pixel2 p : K)	// copy, not reference
				{
					p.x -= xmin;
					p.y -= ymin;
					konturs.hobuffer[off++] = p;
				}
			}

			// only the batch's ends, not the whole capacity
			if (!kontur_borders.upload (offEnds))
				throw std::runtime_error ("uploading the contour border ends to the GPU failed");
		}

		//****** upload

		if (!clouds.upload() || (using_contour && !konturs.upload()))
			throw std::runtime_error ("uploading the batch's ROI clouds or contours to the GPU failed");
	}

	bool GpusideCache::free_gpu_cache(
		GpuCache<Pixel2>& clouds,
		GpuCache<Pixel2>& konturs,
		GpuCache<size_t>& kontur_borders,
		RealPixIntens*& realintens,
		double*& prereduce,
		GpuCache<gpureal>& intermediate,
		void*& tempstorage,
		PixIntens*& imat1,
		PixIntens*& imat2,
		GpuCache <cufftDoubleComplex>& gabor_linear_image,
		GpuCache <cufftDoubleComplex>& gabor_result,
		GpuCache <cufftDoubleComplex>& gabor_linear_kernel,
		GpuCache <RealPixIntens>& gabor_energy_image)
	{
		// Every buffer that is set, whichever families the allocation that set it needed, is freed and
		// nulled; the cache is then empty and not ready
		auto free_device = [] (auto& devptr)
		{
			if (! devptr)
				return true;
			bool ok = gpu_delete(devptr);
			devptr = nullptr;
			return ok;
		};

		// every free runs, whether or not an earlier one failed
		bool ok = clouds.clear();
		ok = konturs.clear() && ok;
		ok = kontur_borders.clear() && ok;
		ok = free_device(realintens) && ok;
		ok = free_device(prereduce) && ok;
		ok = intermediate.clear() && ok;
		ok = free_device(tempstorage) && ok;	// devicereduce's temp storage
		ok = free_device(imat1) && ok;
		ok = free_device(imat2) && ok;
		ok = gabor_linear_image.clear() && ok;
		ok = gabor_result.clear() && ok;
		ok = gabor_linear_kernel.clear() && ok;
		ok = gabor_energy_image.clear() && ok;

		using_contour = false;
		if (!ok)
			std::cerr << "gpu cache related error in " << __FILE__ << ":" << __LINE__ << ": freeing the GPU cache\n";
		return ok;
	}

#ifdef USE_GPU

	void GpusideCache::send_roi_data_gpuside(const std::vector<int>& roilabels, std::unordered_map <int, LR>& roidata, size_t batch_offset, size_t batch_len)
	{
		send_roi_batch_data_2_gpu(
			// out
			gpu_roiclouds_2d,
			gpu_roicontours_2d,
			gpu_roicontourborders_2d,
			&dev_realintens,
			&dev_prereduce,
			gpu_featurestatebuf,
			devicereduce_temp_storage_szb,
			&dev_devicereduce_temp_storage,
			// in
			roilabels,
			roidata,
			batch_offset,
			batch_len);
	}


#endif

#endif