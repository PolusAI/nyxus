#pragma once

#include "test_2d_moments_common.h"

// A whole slide's contour is the 4 corners of its bounding box (ContourFeature::buildWholeSlideContour),
// kept on the ROI each once. The slides here have their pixels' own extent as the box, as a caller
// building the ROI directly gives it (the file pipelines widen a whole slide's box by one pixel each
// way). One pixel wide or high, such a slide has fewer than 4 distinct corners - 2 on a 1xN or Nx1
// slide, 1 on a 1x1 slide - and every one of its pixels lies on the slide's edge, so every pixel takes
// the edge weight, epsilon (BasicGeomoms2D::apply_dist2contour_weighting_wholeslide). The zeroth
// weighted moments are then the pixel count times epsilon, as the weights are stored (float), and
// EDGE_INTEGRATED_INTENSITY counts each distinct corner once at the slide's maximum intensity.
// The weighting reads no corner past the contour's end, which a build with bounds-checked
// std::vector access (_MSVC_STL_HARDENING=1 on MSVC, ASan elsewhere) verifies.
void test_2d_moments_wholeslide_thin_slide_mechanics()
{
	const double eps = double(RealPixIntens(0.001));
	for (const auto& [w, h] : { std::pair<int, int>{ 1, 1 }, { 1, 7 }, { 7, 1 } })
	{
		SCOPED_TRACE(std::to_string(w) + " x " + std::to_string(h) + " slide");
		LR roidata(1403);
		PixIntens maxI = 0;
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				const auto intensity = static_cast<PixIntens>(geomoment_fixture_intensity(x, y));
				maxI = std::max(maxI, intensity);
				if (roidata.aux_area == 0)
					Nyxus::init_label_record_3(roidata, x, y, intensity);
				else
					Nyxus::update_label_record_3(roidata, x, y, intensity);
				roidata.raw_pixels.push_back(Pixel2(x, y, intensity));
			}
		roidata.make_nonanisotropic_aabb();
		roidata.aux_image_matrix.allocate(roidata.aabb.get_width(), roidata.aabb.get_height());
		roidata.aux_image_matrix.calculate_from_pixelcloud(roidata.raw_pixels, roidata.aabb);
		roidata.initialize_fvals();

		Fsettings s = make_2d_geomoment_settings();
		s[static_cast<int>(NyxSetting::SINGLEROI)].bval = true;

		ContourFeature contour;
		contour.calculate(roidata, s);
		contour.save_value(roidata.fvals);
		const size_t n_corners = (w == 1 && h == 1) ? 1 : 2;
		ASSERT_EQ(roidata.contour_pixels_.size(), n_corners);
		const double edgeSum = roidata.fvals[(int)Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY][0];
		ASSERT_EQ(edgeSum, double(n_corners) * double(maxI));

		Smoms2D_feature shape_moments;
		shape_moments.calculate(roidata, s);
		shape_moments.save_value(roidata.fvals);
		Imoms2D_feature intensity_moments;
		intensity_moments.calculate(roidata, s);
		intensity_moments.save_value(roidata.fvals);

		const double expected = double(w * h) * eps;
		const double shapeSum = roidata.fvals[(int)Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00][0],
			intenSum = roidata.fvals[(int)Nyxus::Feature2D::IMOM_WRM_00][0];
		ASSERT_NEAR(shapeSum, expected, 1e-12 * expected);
		ASSERT_NEAR(intenSum, expected, 1e-12 * expected);
	}
}

// GpuCache::clear() frees each buffer that is set on its own. The buffers are allocated device side
// first, so after a failed host allocation the device buffers are set while the host ones are not,
// and clear() still frees them.
void test_2d_moments_gpu_cache_clear_mechanics()
{
#ifdef USE_GPU
	GpuCache<size_t> c;
	ASSERT_TRUE(GpusideCache::allocate_on_device((void**)&c.devbuffer, 16 * sizeof(size_t)));
	ASSERT_TRUE(GpusideCache::allocate_on_device((void**)&c.dev_offsets, 2 * sizeof(size_t)));
	ASSERT_EQ(c.hobuffer, nullptr);
	ASSERT_EQ(c.ho_offsets, nullptr);
	ASSERT_TRUE(c.clear());
	ASSERT_EQ(c.devbuffer, nullptr);
	ASSERT_EQ(c.dev_offsets, nullptr);

	// a cache with nothing allocated clears too
	GpuCache<size_t> empty;
	ASSERT_TRUE(empty.clear());
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// The border-end buffer belongs to the GPU memory plan: allocate_gpu_cache() allocates as many ends
// as the plan counts, the bound on the batch's borders it is given (the segmented workflow passes
// twice the slides' row runs), so the plan and the allocation agree. A batch fills it for ROIs
// batch_offset .. batch_offset + batch_len - 1, clouds and contours alike: the second batch of one
// ROI each carries the second ROI's pixels, contour and border ends. A batch with more borders than
// the bound is refused.
void test_2d_moments_gpu_border_upload_mechanics()
{
#ifdef USE_GPU
	std::unordered_map<int, LR> rois;
	// ROI 1 a 3 x 3 square, 3 row runs; ROI 2 two 3 x 3 squares under one label, two borders, 6 row runs
	gpu_test_roi (rois, 1, { "111", "111", "111" }, 2, 2);
	LR& two = gpu_test_roi (rois, 2, { "111....111", "111....111", "111....111" }, 2, 10);
	ASSERT_EQ(two.contour_border_ends_, (std::vector<size_t>{ 8, 16 }));
	const std::vector<int> labels { 1, 2 };
	const size_t borders = 2 * (3 + 6);

	GpuCacheTestRig cache;
	auto err = cache.allocate (2, 30, 10, 3, borders);
	ASSERT_FALSE(err.has_value()) << err.value_or("");
	GpusideCache& g = cache.g;
	ASSERT_NE(g.gpu_roicontourborders_2d.hobuffer, nullptr);
	ASSERT_EQ(g.gpu_roicontourborders_2d.total_len, borders);
	ASSERT_EQ(g.gpu_roicontourborders_2d.num_rois, g.gpu_roicontours_2d.num_rois);
	const size_t planWith = g.ram_comsumption_szb(true, false, false, true, 60, 60, borders, 2, 10, 3, 1, 0),
		planWithout = g.ram_comsumption_szb(false, false, false, true, 60, 60, borders, 2, 10, 3, 1, 0);
	ASSERT_EQ(planWith - planWithout, GpuCache<Pixel2>::szb(60, 2) + GpuCache<size_t>::szb(borders, 2));

	// a bound above the contour buffer's capacity, as twice the row runs of a lattice of one-pixel
	// ROIs can be, is capped at it in the plan and the allocation: every border holds a contour pixel
	ASSERT_EQ(g.ram_comsumption_szb(true, false, false, true, 60, 60, 1000000, 2, 10, 3, 1, 0),
		g.ram_comsumption_szb(true, false, false, true, 60, 60, 60, 2, 10, 3, 1, 0));
	GpuCacheTestRig capped;
	ASSERT_FALSE(capped.allocate (2, 30, 10, 3, 1000000).has_value());
	ASSERT_EQ(capped.g.gpu_roicontourborders_2d.total_len, (size_t)60);
	const size_t unset = 12345;
	g.gpu_roiclouds_2d.ho_lengths[0] = unset;
	g.gpu_roiclouds_2d.ho_offsets[0] = unset;
	g.gpu_roicontours_2d.ho_lengths[0] = unset;
	g.send_roi_batch_data_2_gpu(g.gpu_roiclouds_2d, g.gpu_roicontours_2d, g.gpu_roicontourborders_2d, &g.dev_realintens,
		&g.dev_prereduce, g.gpu_featurestatebuf, g.devicereduce_temp_storage_szb, &g.dev_devicereduce_temp_storage,
		labels, rois, 1, 1);
	ASSERT_EQ(g.gpu_roiclouds_2d.ho_lengths[0], two.raw_pixels.size());
	ASSERT_EQ(g.gpu_roiclouds_2d.ho_offsets[0], (size_t)0);
	ASSERT_EQ(g.gpu_roicontours_2d.ho_lengths[0], two.contour_pixels_.size());
	ASSERT_EQ(g.gpu_roicontourborders_2d.ho_lengths[0], (size_t)2);
	ASSERT_EQ(g.gpu_roicontourborders_2d.ho_offsets[0], (size_t)0);
	ASSERT_EQ(g.gpu_roicontourborders_2d.hobuffer[0], (size_t)8);
	ASSERT_EQ(g.gpu_roicontourborders_2d.hobuffer[1], (size_t)16);

	// more borders than the plan's bound
	GpuCacheTestRig small;
	err = small.allocate (2, 30, 10, 3, 2);
	ASSERT_FALSE(err.has_value()) << err.value_or("");
	ASSERT_THROW(small.g.send_roi_batch_data_2_gpu(small.g.gpu_roiclouds_2d, small.g.gpu_roicontours_2d, small.g.gpu_roicontourborders_2d,
		&small.g.dev_realintens, &small.g.dev_prereduce, small.g.gpu_featurestatebuf, small.g.devicereduce_temp_storage_szb,
		&small.g.dev_devicereduce_temp_storage, labels, rois, 0, 2), std::runtime_error);

	// a ROI whose border ends do not cover its contour is refused before anything is uploaded
	two.contour_border_ends_ = { 8 };
	ASSERT_THROW(g.send_roi_batch_data_2_gpu(g.gpu_roiclouds_2d, g.gpu_roicontours_2d, g.gpu_roicontourborders_2d, &g.dev_realintens,
		&g.dev_prereduce, g.gpu_featurestatebuf, g.devicereduce_temp_storage_szb, &g.dev_devicereduce_temp_storage,
		labels, rois, 1, 1), std::runtime_error);
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// The GPU's distance weighting reads the same borders as the CPU's: a ROI of two separate 3 x 3
// squares, whose borders of 8 pixels both searches scan in full, and a 1 x 7 whole slide, whose two
// distinct corners take the edge weight on both, give the CPU's zeroth weighted moments.
void test_2d_moments_gpu_cpu_parity_mechanics()
{
#ifdef USE_GPU
	for (bool wholeslide : { false, true })
	{
		SCOPED_TRACE(wholeslide ? "1 x 7 whole slide" : "two squares");
		Fsettings s = make_2d_geomoment_settings();
		s[static_cast<int>(NyxSetting::SINGLEROI)].bval = wholeslide;

		// the same ROI twice: one computed on the CPU, one on the GPU
		std::unordered_map<int, LR> cpu, gpu;
		for (auto* rois : { &cpu, &gpu })
		{
			if (!wholeslide)
			{
				gpu_test_roi (*rois, 1, { "111....111", "111....111", "111....111" }, 0, 0);
				continue;
			}
			LR& r = (*rois)[1];
			r.label = 1;
			for (int x = 0; x < 7; x++)
			{
				const auto intensity = static_cast<PixIntens>(geomoment_fixture_intensity(x, 0));
				if (r.aux_area == 0)
					Nyxus::init_label_record_3(r, x, 0, intensity);
				else
					Nyxus::update_label_record_3(r, x, 0, intensity);
				r.raw_pixels.push_back(Pixel2(x, 0, intensity));
			}
			r.make_nonanisotropic_aabb();
			r.aux_image_matrix.allocate(r.aabb.get_width(), r.aabb.get_height());
			r.aux_image_matrix.calculate_from_pixelcloud(r.raw_pixels, r.aabb);
			r.initialize_fvals();
			ContourFeature contour;
			contour.calculate(r, s);
			ASSERT_EQ(r.contour_pixels_.size(), (size_t)2);
		}

		LR& c = cpu[1];
		Smoms2D_feature shape;
		shape.calculate(c, s);
		shape.save_value(c.fvals);
		Imoms2D_feature inten;
		inten.calculate(c, s);
		inten.save_value(c.fvals);

		LR& g = gpu[1];
		GpuCacheTestRig cache;
		auto err = cache.allocate (1, g.raw_pixels.size(), g.aabb.get_width(), g.aabb.get_height(), 2 * g.aabb.get_height());
		ASSERT_FALSE(err.has_value()) << err.value_or("");
		const std::vector<int> labels { 1 };
		cache.g.send_roi_batch_data_2_gpu(cache.g.gpu_roiclouds_2d, cache.g.gpu_roicontours_2d, cache.g.gpu_roicontourborders_2d,
			&cache.g.dev_realintens, &cache.g.dev_prereduce, cache.g.gpu_featurestatebuf, cache.g.devicereduce_temp_storage_szb,
			&cache.g.dev_devicereduce_temp_storage, labels, gpu, 0, 1);
		Smoms2D_feature::gpu_process_all_rois(labels, gpu, 0, 1, wholeslide, cache.g);
		Imoms2D_feature::gpu_process_all_rois(labels, gpu, 0, 1, wholeslide, cache.g);

		for (auto f : { Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00, Nyxus::Feature2D::IMOM_WRM_00 })
		{
			const double cv = c.fvals[(int)f][0], gv = g.fvals[(int)f][0];
			ASSERT_NEAR(gv, cv, 1e-6 * std::abs(cv)) << "feature " << (int)f;
		}
	}
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// gpu_cache_sizes() sizes the GPU cache for the ROIs as the trivial pipeline holds them. Custom
// anisotropy resamples each pixel to at most ceil(ax) x ceil(ay) pixels, so the largest ROI and a
// batch's pixels grow by that, and the borders, bounded by twice the row runs, by ceil(ay). A factor
// below 1 adds no pixel. The physical pixel size does not resample a 2D ROI, so it changes nothing.
// A whole slide's contour is its 4 corners, one border.
void test_2d_moments_gpu_cache_sizes_mechanics()
{
	auto sizes = [] (bool whole, double ax, double ay, bool physical)
	{
		Environment e;
		e.singleROI = whole;
		e.anisoOptions.set_aniso_x (ax);
		e.anisoOptions.set_aniso_y (ay);
		e.use_physical_spacing_ = physical;
		SlideProps& p = e.dataset.dataset_props.emplace_back ("i.tif", "m.tif");
		p.max_roi_area = 100;
		p.n_rois = 3;
		p.n_row_runs = 20;
		p.phys_x = 1.0;
		p.phys_y = 3.0;
		p.phys_z = 1.0;
		e.dataset.update_dataset_props_extrema();
		const Nyxus::GpuCacheSizes s = Nyxus::gpu_cache_sizes (e);
		return std::vector<size_t>{ s.roi_area, s.roicloud_len, s.kontur_len, s.kontur_borders };
	};
	ASSERT_EQ(sizes (false, 1.0, 1.0, false), (std::vector<size_t>{ 100, 300, 300, 40 }));
	ASSERT_EQ(sizes (false, 2.0, 1.5, false), (std::vector<size_t>{ 400, 1200, 1200, 80 }));
	ASSERT_EQ(sizes (false, 0.5, 0.5, false), (std::vector<size_t>{ 100, 300, 300, 40 }));
	ASSERT_EQ(sizes (false, 1.0, 1.0, true), (std::vector<size_t>{ 100, 300, 300, 40 }));
	ASSERT_EQ(sizes (true, 2.0, 1.5, false), (std::vector<size_t>{ 400, 1200, 4, 1 }));
}

// A ROI resampled by the factors fits a cache sized for it, and overflows one sized for the ROI as
// acquired: a 3 x 3 ROI at --anisox=2 --anisoy=2 is a 6 x 6 one, 36 pixels, not 9.
void test_2d_moments_gpu_anisotropic_batch_mechanics()
{
#ifdef USE_GPU
	std::unordered_map<int, LR> rois;
	gpu_test_roi (rois, 1, { "111111", "111111", "111111", "111111", "111111", "111111" }, 0, 0);
	const std::vector<int> labels { 1 };

	GpuCacheTestRig resampled;
	ASSERT_FALSE(resampled.allocate (1, 9 * 2 * 2, 6, 6, 2 * 6 * 2).has_value());
	ASSERT_NO_THROW(resampled.send (labels, rois, 0, 1));

	GpuCacheTestRig acquired;
	ASSERT_FALSE(acquired.allocate (1, 9, 6, 6, 2 * 6 * 2).has_value());
	ASSERT_THROW(acquired.send (labels, rois, 0, 1), std::runtime_error);
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// allocate_gpu_cache() over a cache that a failed run left allocated frees what it allocates again,
// so the device memory does not grow: a batch whose upload throws, then a second allocation, leave
// the free memory where the first allocation left it. free_gpu_cache() forgets every buffer it
// frees, the erosion ones included.
void test_2d_moments_gpu_cache_reallocate_mechanics()
{
#ifdef USE_GPU
	std::unordered_map<int, LR> rois;
	LR& r = gpu_test_roi (rois, 1, { "111", "111", "111" }, 2, 2);
	const std::vector<int> labels { 1 };
	const size_t area = size_t(1) << 20;	// a moments buffer of this many pixels shows in the free memory

	GpuCacheTestRig cache;
	ASSERT_FALSE(cache.allocate (1, area, 1024, 1024, 6, true).has_value());
	size_t free1 = 0, free2 = 0;
	ASSERT_TRUE(GpusideCache::gpu_get_free_mem (free1));

	r.contour_border_ends_ = { 1 };	// does not cover the contour: the upload throws
	ASSERT_THROW(cache.send (labels, rois, 0, 1), std::runtime_error);
	ASSERT_FALSE(cache.allocate (1, area, 1024, 1024, 6, true).has_value());
	ASSERT_TRUE(GpusideCache::gpu_get_free_mem (free2));
	ASSERT_GE(free2 + (size_t(16) << 20), free1) << "the second allocation leaked " << (free1 - free2) << " bytes";

	ASSERT_TRUE(cache.free());
	GpusideCache& g = cache.g;
	ASSERT_EQ(g.dev_realintens, nullptr);
	ASSERT_EQ(g.dev_prereduce, nullptr);
	ASSERT_EQ(g.dev_devicereduce_temp_storage, nullptr);
	ASSERT_EQ(g.dev_imat1, nullptr);
	ASSERT_EQ(g.dev_imat2, nullptr);
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// A freed cache holds nothing and has no capacity, so an upload into it is refused rather than
// copied through a null buffer.
void test_2d_moments_gpu_upload_after_free_mechanics()
{
#ifdef USE_GPU
	std::unordered_map<int, LR> rois;
	gpu_test_roi (rois, 1, { "111", "111", "111" }, 2, 2);
	const std::vector<int> labels { 1 };

	GpuCacheTestRig cache;
	ASSERT_FALSE(cache.allocate (1, 9, 3, 3, 6).has_value());
	ASSERT_NO_THROW(cache.send (labels, rois, 0, 1));
	ASSERT_TRUE(cache.free());
	ASSERT_EQ(cache.g.gpu_roiclouds_2d.total_len, (size_t)0);
	ASSERT_EQ(cache.g.gpu_roicontourborders_2d.total_len, (size_t)0);
	ASSERT_THROW(cache.send (labels, rois, 0, 1), std::runtime_error);
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// allocate_gpu_cache() allocates every buffer the families it is asked for need, and says so.
void test_2d_moments_gpu_cache_families_mechanics()
{
#ifdef USE_GPU
	GpuCacheTestRig cache;
	ASSERT_FALSE(cache.allocate (1, 121, 11, 11, 22, true, true).has_value());
	GpusideCache& g = cache.g;
	ASSERT_TRUE(g.is_ready());
	for (const void* p : { (const void*) g.gpu_roiclouds_2d.devbuffer, (const void*) g.gpu_roicontours_2d.devbuffer,
		(const void*) g.gpu_roicontourborders_2d.devbuffer, (const void*) g.dev_realintens, (const void*) g.dev_prereduce,
		(const void*) g.gpu_featurestatebuf.devbuffer, (const void*) g.dev_devicereduce_temp_storage,
		(const void*) g.dev_imat1, (const void*) g.dev_imat2, (const void*) g.gabor_linear_image.devbuffer,
		(const void*) g.gabor_result.devbuffer, (const void*) g.gabor_linear_kernel.devbuffer, (const void*) g.gabor_energy_image.devbuffer })
		ASSERT_NE(p, nullptr);
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// A second allocation for larger ROIs gives the Gabor buffers the new size, rather than keeping those
// of the first (the erosion matrices' size shows in test_2d_morphology_gpu_erosion_mechanics).
void test_2d_moments_gpu_cache_regrow_mechanics()
{
#ifdef USE_GPU
	GpuCacheTestRig cache;
	ASSERT_FALSE(cache.allocate (1, 9, 3, 3, 6, true, true).has_value());
	ASSERT_FALSE(cache.allocate (1, 41 * 41, 41, 41, 82, true, true).has_value());
	ASSERT_EQ(cache.g.gabor_linear_image.total_len, (size_t)(41 + 16 - 1) * (41 + 16 - 1) * (4 + 1));
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// An allocation frees every buffer an earlier one left, whichever families the new one needs: after
// an allocation with every family, one with none of them leaves only the clouds, which every
// allocation has, and a free then leaves all 13 buffers null.
void test_2d_moments_gpu_cache_drops_families_mechanics()
{
#ifdef USE_GPU
	GpuCacheTestRig cache;
	GpusideCache& g = cache.g;
	auto family_buffers = [&g] ()
	{
		return std::vector<const void*> { g.gpu_roicontours_2d.devbuffer,
			g.gpu_roicontourborders_2d.devbuffer, g.dev_realintens, g.dev_prereduce,
			g.gpu_featurestatebuf.devbuffer, g.dev_devicereduce_temp_storage,
			g.dev_imat1, g.dev_imat2, g.gabor_linear_image.devbuffer,
			g.gabor_result.devbuffer, g.gabor_linear_kernel.devbuffer, g.gabor_energy_image.devbuffer };
	};
	ASSERT_FALSE(cache.allocate (1, 121, 11, 11, 22, true, true).has_value());
	for (const void* p : family_buffers())
		ASSERT_NE(p, nullptr);
	ASSERT_FALSE(cache.allocate (1, 121, 11, 11, 22, false, false, false).has_value());
	const std::vector<const void*> after_no_families = family_buffers();
	ASSERT_TRUE(g.is_ready());
	ASSERT_NE(g.gpu_roiclouds_2d.devbuffer, nullptr);
	for (size_t i = 0; i < after_no_families.size(); i++)
		ASSERT_EQ(after_no_families[i], nullptr) << "family buffer " << i << " kept after an allocation with no families";
	ASSERT_TRUE(cache.free());
	ASSERT_FALSE(g.is_ready());
	ASSERT_EQ(g.gpu_roiclouds_2d.devbuffer, nullptr);
	for (const void* p : family_buffers())
		ASSERT_EQ(p, nullptr);
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// A GpuCache allocation that fails frees what it had allocated, the device buffers included, and
// clear() frees each buffer that is set on its own, nulls it, and goes on after a device free that
// fails, reporting it: GpusideCache::is_ready() reads the clouds' host buffer, and an allocation of
// the cache relies on a failed or cleared GpuCache holding nothing.
void test_2d_moments_gpu_cache_alloc_clear_mechanics()
{
#ifdef USE_GPU
	struct Reset { ~Reset() { GpusideCache::test_fail_next_host_alloc = false; } } reset;

	// a plain cache and one with per-ROI offsets, whose first host allocation fails
	{
		GpuCache<float> c;
		GpusideCache::test_fail_next_host_alloc = true;
		ASSERT_FALSE(c.alloc (16, 2));
		ASSERT_FALSE(GpusideCache::test_fail_next_host_alloc);
		ASSERT_EQ(c.hobuffer, nullptr);
		ASSERT_EQ(c.devbuffer, nullptr);
		ASSERT_EQ(c.total_len, 0u);
		ASSERT_TRUE(c.alloc (16, 2));	// the next allocation succeeds
		ASSERT_NE(c.hobuffer, nullptr);
		ASSERT_NE(c.devbuffer, nullptr);
		ASSERT_EQ(c.total_len, 32u);
	}
	{
		GpuCache<Pixel2> c;
		GpusideCache::test_fail_next_host_alloc = true;
		ASSERT_FALSE(c.alloc (16, 2));
		for (const void* p : { (const void*) c.hobuffer, (const void*) c.devbuffer, (const void*) c.ho_lengths,
			(const void*) c.ho_offsets, (const void*) c.dev_offsets })
			ASSERT_EQ(p, nullptr);
	}

	// only the device buffer set, as a failed allocation could leave it
	{
		GpuCache<float> c;
		ASSERT_TRUE(GpusideCache::allocate_on_device ((void**)&c.devbuffer, 64 * sizeof(float)));
		ASSERT_TRUE(c.clear());
		ASSERT_EQ(c.devbuffer, nullptr);
	}

	// a device free that fails: reported, and the host buffer freed and both pointers nulled all the same
	{
		GpuCache<float> c;
		static float not_on_device;
		c.hobuffer = (float*) std::malloc (sizeof(float));
		c.devbuffer = &not_on_device;
		ASSERT_FALSE(c.clear());
		cudaGetLastError();	// the refused free is CUDA's last error, which a later CUB call would report
		ASSERT_EQ(c.hobuffer, nullptr);
		ASSERT_EQ(c.devbuffer, nullptr);
	}
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}

// With the GPU on and its cache not ready -- never allocated, as on the paths other than the
// segmented 2D workflow, or after an allocation or a free that failed part way -- the moments are
// computed on the CPU, with the CPU's values.
void test_2d_moments_gpu_unready_cache_mechanics()
{
#ifdef USE_GPU
	for (GpuUnready why : { GpuUnready::never_allocated, GpuUnready::allocation_failed_part_way, GpuUnready::free_failed_part_way })
	{
		SCOPED_TRACE(gpu_unready_name (why));
		std::vector<std::vector<double>> on, off;
		ASSERT_NO_FATAL_FAILURE(gpu_unready_vs_cpu ({ Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00, Nyxus::Feature2D::IMOM_WRM_00 }, why, on, off));
		for (auto f : { Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00, Nyxus::Feature2D::IMOM_WRM_00 })
		{
			ASSERT_EQ(on[(int)f][0], off[(int)f][0]) << "feature " << (int)f;
			ASSERT_NE(off[(int)f][0], 0.0) << "feature " << (int)f;
		}
	}
#else
	GTEST_SKIP() << "built without USE_GPU";
#endif
}
