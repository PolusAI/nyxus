#ifdef USE_GPU

#include "cache.h"
#include "roi_cache.h"

#define OK(x) if (x == false) \
{ \
	std::cerr << "gpu cache related error in " << __FILE__ << ":" << __LINE__; \
	return false; \
} \

#define OKV(x) if (x == false) \
{ \
	std::cerr << "gpu cache related error in " << __FILE__ << ":" << __LINE__; \
	return; \
} \


	size_t GpusideCache::ram_comsumption_szb (
		bool needContour,
		bool needErosion,
		bool needGabor,
		bool needMoments,
		size_t roi_cloud_len,
		size_t roi_kontur_cloud_len,
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
			szb_kon = GpuCache<Pixel2>::/*konturs.*/szb(roi_kontur_cloud_len, n_rois);
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

	bool GpusideCache::allocate_gpu_cache(
		// out
		GpuCache<Pixel2>& clouds,	// geo moments
		GpuCache<Pixel2>& konturs,
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
		GpuCache <PixIntens>& gabor_energy_image,
		// in
		bool needContour,
		bool needErosion,
		bool needGabor,
		bool needMoments,
		size_t roi_cloud_len,
		size_t roi_kontur_cloud_len,
		size_t n_rois,
		size_t roi_area,
		size_t roi_w,
		size_t roi_h,
		int n_gabor_filters,
		int gabor_ker_side)
	{
		using_contour =
		using_erosion =
		using_gabor =
		using_moments = false;

		//****** plan GPU memory

		size_t amt = 0;
		OK(gpu_get_free_mem(amt));

		//xxxxxx	VERBOSLVL1(std::cout << "GPU RAM amt = " << Nyxus::virguler_ulong(amt) << "\n");

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
			n_rois, roi_w, roi_h, n_gabFilters, gabor_ker_side);

		//xxxxxx	VERBOSLVL1(std::cout << "szb for " << Nyxus::virguler_ulong(n_rois) << " ROIs (ideal ROI px:" << Nyxus::virguler_ulong(roi_cloud_len) << ", ideal contour px:" << Nyxus::virguler_ulong(roi_kontur_cloud_len) << ") = " << Nyxus::virguler_ulong(szb) << "\n");

		batch_len = n_rois;
		size_t critAmt = amt * 0.75; // 75% GPU RAM as critical RAM

		//xxxxxx	VERBOSLVL1(std::cout << "critical GPU RAM amt = " << Nyxus::virguler_ulong(critAmt) << "\n");

		if (critAmt < szb)
		{
			//xxxxxx	VERBOSLVL1(std::cout << "Need to split " << Nyxus::virguler_ulong(n_rois)  << " ROIs into batches \n");

			size_t try_nrois = 0;
			for (try_nrois = n_rois; try_nrois >= 0; try_nrois--)
			{
				// failed to find a batch ?
				if (try_nrois == 0)
				{
					//xxxxxx	VERBOSLVL1(std::cerr << "error: cannot make a ROI batch \n");
					std::cerr << "error: cannot make a ROI batch \n";
					return false;
				}

				size_t ccl = roi_area * try_nrois;	// combined cloud length
				size_t try_szb = ram_comsumption_szb(
					needContour,
					needErosion,
					needGabor,
					needMoments,
					ccl, ccl, try_nrois, roi_w, roi_h, n_gabFilters, gabor_ker_side);

				//xxxxxx	if (try_nrois % 10 == 0) VERBOSLVL1(std::cout << "try_szb (" << ccl << ", " << ccl << ", " << try_nrois << ") = " << try_szb << "\n");

				if (critAmt > try_szb)
				{
					//xxxxxx	VERBOSLVL1(std::cout << "batching is successful, batch_len=" << try_nrois << "\n");
					batch_len = try_nrois;
					break;
				}
			}

			// have we found a compromise ?
			if (batch_len < n_rois)
			{
				//xxxxxx	VERBOSLVL1(std::cerr << "error: cannot make a ROI batch \n");
				return false;
			}
		}

		size_t batch_roi_cloud_len = roi_area * batch_len;

		//xxxxxx	VERBOSLVL1(std::cout << "batch_len = " << batch_len << " of ideal " << n_rois << "\n");
		//xxxxxx	VERBOSLVL1(std::cout << "batch_roi_cloud_len = " << batch_roi_cloud_len  << "\n");

		//****** allocate

		// ROI clouds (always on)

		OK(clouds.clear());
		OK(clouds.alloc(batch_roi_cloud_len, batch_len));

		// contours

		if (needContour)
		{
			using_contour = true;

			OK(konturs.clear());
			OK(konturs.alloc(batch_roi_cloud_len, batch_len));
		}

		// moments

		if (needMoments)
		{
			using_moments = true;

			// moments / real intensities
			OK(allocate_on_device((void**)realintens, sizeof(RealPixIntens) * batch_roi_cloud_len));

			// moments / pre-reduce
			OK(allocate_on_device((void**)prereduce, sizeof(double) * batch_roi_cloud_len * 16));	// 16 is the max number of simultaneous totals calculated by a kernel, e.g. RM00-33

			// moments / intermediate
			OK(intermediate.alloc(GpusideState::__COUNT__, batch_len));

			// moments / CUB DeviceReduce's temp buffer
			OK(devicereduce_evaluate_buffer_szb(devicereduce_buf_szb, batch_roi_cloud_len));
			OK(allocate_on_device((void**)devicereduce_buf, devicereduce_buf_szb));
		}

		// erosion / image matrices 1 and 2

		if (needErosion)
		{
			using_erosion = true;

			// imat1 is shared by erosion and Gabor
			if (!*imat1)
				OK(allocate_on_device((void**)imat1, sizeof(imat1[0]) * roi_w * roi_h));

			OK(allocate_on_device((void**)imat2, sizeof(imat2[0]) * roi_w * roi_h));
		}

		// Gabor

		if (needGabor)
		{
			using_gabor = true;

			// imat1 is shared by erosion and Gabor
			if (!*imat1)
				OK(allocate_on_device((void**)imat1, sizeof(imat1[0]) * roi_w * roi_h));

			size_t gabTotlen = (roi_w + gabor_ker_side - 1) * (roi_h + gabor_ker_side - 1) * n_gabFilters;
			OK(gabor_linear_image.alloc(gabTotlen, 1));
			OK(gabor_result.alloc(gabTotlen, 1));
			OK(gabor_linear_kernel.alloc(gabTotlen, 1));
			OK(gabor_energy_image.alloc(gabTotlen, 1));
		}

		return true;
	}

	void GpusideCache::send_roi_batch_data_2_gpu(
		// out
		GpuCache<Pixel2>& clouds,
		GpuCache<Pixel2>& konturs,
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
		//***** ROI clouds

		// stats of ROI cloud sizes
		size_t totCloLen = 0, maxLen = 0;
		for (size_t i = batch_offset; i < batch_len; i++)
		{
			int lbl = labels[i];
			LR& roi = roi_data[lbl];
			auto n = roi.raw_pixels.size();
			totCloLen += n;
			maxLen = (std::max)(maxLen, n);
		}

		//****** contours

		if (using_contour)
		{
			// stats of ROI cloud sizes
			size_t totKontLen = 0;
			for (size_t i = batch_offset; i < batch_len; i++)
			{
				int lbl = labels[i];
				LR& roi = roi_data[lbl];
				totKontLen += roi.contour.size();
			}
		}

		//****** clouds as a solid buffer
		size_t off = 0;
		size_t roiIdx = 0;
		for (size_t i = batch_offset; i < batch_len; i++)
		{
			int lbl = labels[i];
			LR& roi = roi_data[lbl];

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
			for (size_t i = batch_offset; i < batch_len; i++)
			{
				int lbl = labels[i];
				LR& roi = roi_data[lbl];

				konturs.ho_lengths[roiIdx] = roi.contour.size();
				konturs.ho_offsets[roiIdx] = off;
				roiIdx++;

				// save to buffer fixing pixel coordinates
				auto xmin = roi.aabb.get_xmin();
				auto ymin = roi.aabb.get_ymin();
				for (Pixel2 p : roi.contour)	// copy, not reference
				{
					p.x -= xmin;
					p.y -= ymin;
					konturs.hobuffer[off++] = p;
				}
			}
		}

		//****** upload

		OKV(clouds.upload());
		OKV(konturs.upload());
	}

	bool GpusideCache::free_gpu_cache(
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
		GpuCache <PixIntens>& gabor_energy_image)
	{
		// clouds 

		OK(clouds.clear());


		// contour

		if (using_contour)
		{
			OK(konturs.clear());
		}

		// moments

		if (using_moments)
		{
			OK(gpu_delete(realintens));
			OK(gpu_delete(prereduce));
			OK(intermediate.clear());
			OK(gpu_delete(tempstorage));
			realintens = nullptr;
			prereduce = nullptr;
			OK(intermediate.clear());
			tempstorage = nullptr;	// devicereduce's temp storage
		}

		// erosion or Gabor
		if (using_erosion || using_gabor)
		{
			OK(gpu_delete(imat1));
			imat1 = nullptr;
		}

		// erosion

		if (using_erosion)
		{
			OK(gpu_delete(imat2));
			imat2 = nullptr;
		}

		// Gabor

		if (using_gabor)
		{
			OK(gabor_linear_image.clear());
			OK(gabor_result.clear());
			OK(gabor_linear_kernel.clear());
			OK(gabor_energy_image.clear());
		}

		return true;
	}

#ifdef USE_GPU

	void GpusideCache::send_roi_data_gpuside(const std::vector<int>& roilabels, std::unordered_map <int, LR>& roidata, size_t batch_offset, size_t batch_len)
	{
		send_roi_batch_data_2_gpu(
			// out
			gpu_roiclouds_2d,
			gpu_roicontours_2d,
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