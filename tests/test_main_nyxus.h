#pragma once

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include <tiffio.h>
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/image_loader.h"
#include "test_dsb2018_data.h"
#include "test_data.h"

namespace Nyxus
{
    /// @brief The (channel, timeframe) volume ImageLoader::stream_volume_planes delivers, assembled
    /// into X*Y*Z intensity and mask buffers ('seg' stays empty for a pair without a mask), so a
    /// test checks the voxels every volumetric consumer reads.
    static void assemble_streamed_volume (ImageLoader& il, size_t channel, size_t timeframe,
        std::vector<uint32_t>& inten, std::vector<uint32_t>& seg)
    {
        const size_t slice = il.get_full_width() * il.get_full_height(),
            n = slice * il.get_full_depth();
        inten.assign(n, 0u);
        seg.clear();
        il.stream_volume_planes(channel, timeframe,
            [&](size_t z, const std::vector<uint32_t>& I, const std::vector<uint32_t>& L)
            {
                std::copy(I.begin(), I.end(), inten.begin() + z * slice);
                if (!L.empty())
                {
                    if (seg.empty())
                        seg.assign(n, 0u);
                    std::copy(L.begin(), L.end(), seg.begin() + z * slice);
                }
            });
    }

    /// @brief Tests the agreement with ground truth up to the tolerance specified as a fraction of the ground truth
    static bool agrees_gt(double fval, double ground_truth, double frac_tolerance = 1000.)
    {
        auto diff = fval - ground_truth;
        auto tolerance = ground_truth / frac_tolerance;
        bool good = std::abs(diff) <= std::abs(tolerance);

        // Show the failure info in failed
        if (!good)
            std::cout << "\tabs of (actual=" << fval << " - groundtruth=" << ground_truth << ")=" << std::abs(diff) << " > tolerance=" << tolerance << "\n";

        return good;
    }

    static void load_test_roi_data (LR& roidata, const NyxusPixel* testData, size_t count)
    {
        // -- mocking gatherRoisMetrics():
        for (auto i=0; i<count; i++)
        {
            const NyxusPixel& px = testData[i];
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_3 (roidata, px.x, px.y, px.intensity);
            else
                update_label_record_3 (roidata, px.x, px.y, px.intensity);
        }

        // -- mocking scanTrivialRois():
        for (auto i = 0; i < count; i++)
        {
            const NyxusPixel& px = testData[i];
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(px.x, px.y, px.intensity));
        }
    }

    static void load_masked_test_roi_data (LR& roidata, const NyxusPixel* intensityData, const NyxusPixel* maskData, size_t count)
    {
        // -- mocking phase 1, gatherRoisMetrics():
        for (auto i = 0; i < count; i++)
        {
            // Check if pixel [i] belongs to the ROI
            const NyxusPixel& maskPixel = maskData[i];
            if (maskPixel.intensity == 0)
                continue;   // Skip this pixel
            // Pixel [i] is within the ROI, feed it to ROI shape and intensity range examiner
            const NyxusPixel& px = intensityData[i];
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_3(roidata, px.x, px.y, px.intensity);
            else
                update_label_record_3(roidata, px.x, px.y, px.intensity);
        }
        // -- mocking phase 2, scanTrivialRois():
        for (auto i = 0; i < count; i++)
        {
            // Check if pixel [i] belongs to the ROI
            const NyxusPixel& maskPixel = maskData[i];
            if (maskPixel.intensity == 0)
                continue;   // Skip this pixel
            // Pixel [i] is within the ROI, feed it to ROI pixel accumulator
            const NyxusPixel& px = intensityData[i];
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(px.x, px.y, px.intensity));
        }

        // Anisotropy (none)
        roidata.make_nonanisotropic_aabb();

        // -- allocating the image matrix (roidata.aux_image_matrix)
        //      (Phase 1 creates roidata.aabb giving us ROI's dimensions)
        roidata.aux_image_matrix.allocate(
            roidata.aabb.get_width(),
            roidata.aabb.get_height());
        // -- filling the image matrix
        roidata.aux_image_matrix.calculate_from_pixelcloud (roidata.raw_pixels, roidata.aabb);
    }

    static void load_test_roi_data(LR& roidata, int data_idx, bool allocate_IM = true)
    {
        int dummyLabel = 100;

        // -- mocking gatherRoisMetrics():
        int i = 0;
        size_t w = dsb_data[data_idx].x;
        size_t h = dsb_data[data_idx].y;
        for (auto& px : dsb_data[data_idx].pixels)
        {
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_hierarchical (roidata, "theSegFname", "theIntFname", i%w, i/w, dummyLabel, px);
            else
                update_label_record_2 (roidata, i%w, i/w, dummyLabel, px);

            ++i;
        }

        // -- mocking scanTrivialRois():
        i = 0;
        for (auto& px : dsb_data[data_idx].pixels) {
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(i%w, i/w, px));
            ++i;
        }

        // Anisotropy (none)
        roidata.make_nonanisotropic_aabb();

        // image matrix
        if (allocate_IM)
            roidata.aux_image_matrix = ImageMatrix(roidata.raw_pixels);
    }

}

// The config every 2D shape assertion is measured under -- 2 um pixels, 128 grey levels, IBSI off.
// It sits here rather than in one family's header because morphology, radial and zernike all pin
// goldens to it: one copy is what keeps the three families measuring the same thing (SPEC 6.3.1).

static Fsettings make_shape2d_settings()
{
	Fsettings s;
	s.resize(static_cast<int>(NyxSetting::__COUNT__));
	s[static_cast<int>(NyxSetting::SOFTNAN)].rval = 0.0;
	s[static_cast<int>(NyxSetting::TINY)].rval = 0.0;
	s[static_cast<int>(NyxSetting::SINGLEROI)].bval = false;
	s[static_cast<int>(NyxSetting::GREYDEPTH)].ival = 128;
	s[static_cast<int>(NyxSetting::PIXELSIZEUM)].rval = 2.0;
	s[static_cast<int>(NyxSetting::XYRES)].rval = 1.0;
	s[static_cast<int>(NyxSetting::PIXELDISTANCE)].ival = 1;
	s[static_cast<int>(NyxSetting::USEGPU)].bval = false;
	s[static_cast<int>(NyxSetting::VERBOSLVL)].ival = 0;
	s[static_cast<int>(NyxSetting::IBSI)].bval = false;
	return s;
}

// A file under tests/data/ometiff. Lives here rather than in a family header because the ometiff
// tests and the out-of-core guard tests both read that directory (SPEC 6.3.1).
inline fs::path ometiff_data_path (const char* name)
{
    fs::path p (__FILE__);
    fs::path rel (std::string("/data/ometiff/") + name);
    return fs::path (p.parent_path().string() + rel.make_preferred().string());
}

// One tiled 16-bit plane at the given tile size, so a test can build a grid of any shape --
// including one that is not square, which is where a row/column mix-up shows. Lives here rather
// than in a family header because tests of both dimensions build such a plane (SPEC 6.3.1).
inline void write_tiled_plane_u16 (const fs::path& f, uint32_t W, uint32_t H, uint32_t tile,
    const std::function<uint16_t(uint32_t, uint32_t)>& val)
{
    TIFF* t = TIFFOpen (f.string().c_str(), "w");
    ASSERT_NE(t, nullptr) << f.string();
    TIFFSetField (t, TIFFTAG_IMAGEWIDTH, W);
    TIFFSetField (t, TIFFTAG_IMAGELENGTH, H);
    TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, 16);
    TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField (t, TIFFTAG_TILEWIDTH, tile);
    TIFFSetField (t, TIFFTAG_TILELENGTH, tile);
    std::vector<uint16_t> buf (tile * tile);
    for (uint32_t y0 = 0; y0 < H; y0 += tile)
        for (uint32_t x0 = 0; x0 < W; x0 += tile)
        {
            for (uint32_t r = 0; r < tile; r++)
                for (uint32_t c = 0; c < tile; c++)
                    buf[r * tile + c] = val (x0 + c, y0 + r);
            ASSERT_GE(TIFFWriteTile (t, buf.data(), x0, y0, 0, 0), 0) << f.string();
        }
    ASSERT_EQ(TIFFWriteDirectory (t), 1) << f.string();
    TIFFClose (t);
}

#ifdef USE_GPU
#include "../src/nyx/environment.h"
#include "../src/nyx/features/contour.h"

// GPU-cache fixtures. They live here rather than in a family header because the moments,
// morphology (erosion) and Gabor families each check their own GPU path through them (SPEC 6.3.1).

// A GPU cache of its own, allocated the way the segmented workflow allocates GpusideCache: contour
// and moments buffers unless told otherwise, erosion and Gabor (4 filters of 16 x 16 taps) if asked.
// Freed when it goes out of scope.
struct GpuCacheTestRig
{
	GpusideCache g;

	// 'borders': the most contour borders the batch can have, the plan's bound on border ends
	std::optional<std::string> allocate (size_t n_rois, size_t roi_area, size_t roi_w, size_t roi_h, size_t borders,
		bool erosion = false, bool gabor = false, bool contour_and_moments = true)
	{
		return g.allocate_gpu_cache(
			g.gpu_roiclouds_2d, g.gpu_roicontours_2d, g.gpu_roicontourborders_2d, &g.dev_realintens, &g.dev_prereduce,
			g.gpu_featurestatebuf, g.devicereduce_temp_storage_szb, &g.dev_devicereduce_temp_storage, g.gpu_batch_len,
			&g.dev_imat1, &g.dev_imat2, g.gabor_linear_image, g.gabor_result, g.gabor_linear_kernel, g.gabor_energy_image,
			contour_and_moments, erosion, gabor, contour_and_moments,
			roi_area * n_rois, roi_area * n_rois, borders, n_rois, roi_area, roi_w, roi_h, 4, 16);
	}

	bool free()
	{
		return g.free_gpu_cache(g.gpu_roiclouds_2d, g.gpu_roicontours_2d, g.gpu_roicontourborders_2d, g.dev_realintens, g.dev_prereduce,
			g.gpu_featurestatebuf, g.dev_devicereduce_temp_storage, g.dev_imat1, g.dev_imat2,
			g.gabor_linear_image, g.gabor_result, g.gabor_linear_kernel, g.gabor_energy_image);
	}

	// sends ROIs batch_offset .. batch_offset + batch_len - 1 of 'labels'
	void send (const std::vector<int>& labels, std::unordered_map<int, LR>& rois, size_t batch_offset, size_t batch_len)
	{
		g.send_roi_batch_data_2_gpu(g.gpu_roiclouds_2d, g.gpu_roicontours_2d, g.gpu_roicontourborders_2d, &g.dev_realintens,
			&g.dev_prereduce, g.gpu_featurestatebuf, g.devicereduce_temp_storage_szb, &g.dev_devicereduce_temp_storage,
			labels, rois, batch_offset, batch_len);
	}

	~GpuCacheTestRig() { free(); }
};

// ROI 'label' over the pixels of 'rows' ('1' = ROI) placed at (x0, y0), with a non-constant intensity,
// its contour traced: what the GPU upload reads
static LR& gpu_test_roi (std::unordered_map<int, LR>& rois, int label, const std::vector<std::string>& rows, int x0, int y0)
{
	LR& r = rois[label];
	r.label = label;
	for (int y = 0; y < (int)rows.size(); y++)
		for (int x = 0; x < (int)rows[y].size(); x++)
		{
			if (rows[y][x] != '1')
				continue;
			const int ax = x0 + x, ay = y0 + y;
			const auto intensity = static_cast<PixIntens>(10 + 3 * ax + 5 * ay + (ax * ay) % 7);
			if (r.aux_area == 0)
				Nyxus::init_label_record_3(r, ax, ay, intensity);
			else
				Nyxus::update_label_record_3(r, ax, ay, intensity);
			r.raw_pixels.push_back(Pixel2(ax, ay, intensity));
		}
	r.make_nonanisotropic_aabb();
	r.aux_image_matrix.allocate(r.aabb.get_width(), r.aabb.get_height());
	r.aux_image_matrix.calculate_from_pixelcloud(r.raw_pixels, r.aabb);
	r.initialize_fvals();
	ContourFeature contour;
	contour.calculate(r, make_shape2d_settings());
	return r;
}

// a filled disk of radius 'r' as mask rows
static std::vector<std::string> gpu_test_disk_rows (int r)
{
	std::vector<std::string> rows (2 * r + 1, std::string (2 * r + 1, '.'));
	for (int y = -r; y <= r; y++)
		for (int x = -r; x <= r; x++)
			if (x * x + y * y <= r * r)
				rows[y + r][x + r] = '1';
	return rows;
}

// Why a GPU cache is not ready: nothing allocated it, as on the 2.5D, 3D and array paths; an
// allocation failed part way, after every family but Gabor, and freed what it had allocated; or a free
// failed part way, one of its device frees refused.
enum class GpuUnready { never_allocated, allocation_failed_part_way, free_failed_part_way };

static const char* gpu_unready_name (GpuUnready why)
{
	switch (why)
	{
	case GpuUnready::never_allocated: return "never allocated";
	case GpuUnready::allocation_failed_part_way: return "an allocation that failed part way and freed what it had allocated";
	default: return "a free that failed part way";
	}
}

// The values reduce_trivial_rois_manual gives a disk of radius 5 with 'features' enabled, the GPU on
// and its cache not ready for the reason 'why', against those it gives with the GPU off. Callers wrap
// it in ASSERT_NO_FATAL_FAILURE.
static void gpu_unready_vs_cpu (const std::initializer_list<Nyxus::Feature2D>& features, GpuUnready why,
	std::vector<std::vector<double>>& gpu_on, std::vector<std::vector<double>>& gpu_off)
{
	for (bool on : { true, false })
	{
		Environment e;
		e.set_dim (2);
		e.theFeatureSet.enableAll (false);
		e.theFeatureSet.enableFeatures (features, true);
		ASSERT_TRUE(e.theFeatureMgr.compile());
		e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
		ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
		e.compile_feature_settings();
		e.set_using_gpu (on);
		GpusideCache& g = e.devCache;
		auto allocate = [&g] ()
		{
			return g.allocate_gpu_cache(
				g.gpu_roiclouds_2d, g.gpu_roicontours_2d, g.gpu_roicontourborders_2d, &g.dev_realintens, &g.dev_prereduce,
				g.gpu_featurestatebuf, g.devicereduce_temp_storage_szb, &g.dev_devicereduce_temp_storage, g.gpu_batch_len,
				&g.dev_imat1, &g.dev_imat2, g.gabor_linear_image, g.gabor_result, g.gabor_linear_kernel, g.gabor_energy_image,
				true, true, true, true, 121, 121, 22, 1, 121, 11, 11, 4, 16);
		};
		if (on && why == GpuUnready::allocation_failed_part_way)
		{
			struct Reset { ~Reset() { GpusideCache::test_fail_before_gabor = false; } } reset;
			GpusideCache::test_fail_before_gabor = true;
			ASSERT_TRUE(allocate().has_value());
			ASSERT_FALSE(g.is_ready());
			ASSERT_EQ(g.gpu_roiclouds_2d.hobuffer, nullptr);
			ASSERT_EQ(g.dev_realintens, nullptr);
			ASSERT_EQ(g.dev_imat2, nullptr);
		}
		if (on && why == GpuUnready::free_failed_part_way)
		{
			ASSERT_FALSE(allocate().has_value());
			ASSERT_TRUE(g.is_ready());
			// a device pointer cudaFree refuses, in place of one of the Gabor buffers
			ASSERT_TRUE(GpusideCache::gpu_delete (g.gabor_result.devbuffer));
			static cufftDoubleComplex not_on_device;
			g.gabor_result.devbuffer = &not_on_device;
			ASSERT_FALSE(g.free_gpu_cache(g.gpu_roiclouds_2d, g.gpu_roicontours_2d, g.gpu_roicontourborders_2d, g.dev_realintens,
				g.dev_prereduce, g.gpu_featurestatebuf, g.dev_devicereduce_temp_storage, g.dev_imat1, g.dev_imat2,
				g.gabor_linear_image, g.gabor_result, g.gabor_linear_kernel, g.gabor_energy_image));
			cudaGetLastError();	// the refused free is CUDA's last error, which a later CUB call would report
			ASSERT_FALSE(g.is_ready());
			ASSERT_EQ(g.gabor_result.devbuffer, nullptr);
			ASSERT_EQ(g.gabor_energy_image.devbuffer, nullptr);
		}
		gpu_test_roi (e.roiData, 1, gpu_test_disk_rows (5), 2, 2);
		std::vector<int> labels { 1 };
		ASSERT_NO_THROW(Nyxus::reduce_trivial_rois_manual (labels, e));
		(on ? gpu_on : gpu_off) = e.roiData[1].fvals;
	}
}
#endif
