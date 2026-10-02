#pragma once

// Anisotropy on the 2D paths: which pixels each family measures.
//
// An anisotropic 2D run scans a batch twice. The families defined on the image grid (first-order,
// the intensity histogram, the texture families, Gabor and the image-quality families) reduce the
// pixels as acquired, so their values are those of a run without anisotropy. The geometric families
// reduce the cloud resampled by the spacing, so they measure the ROI in physical space.
//
// The fixture is a 24x20 slide with two labels of different shapes, an ellipse and a notched
// rectangle, under the factors (1.3, 0.7): unequal and inexact, so the resampling duplicates some
// columns and drops some rows, and a grid family fed the resampled cloud moves.

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include "test_main_nyxus.h"		// gtest, libtiff, fs, globals.h, roi_cache.h
#include "../src/nyx/environment.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/focus_score.h"
#include "../src/nyx/features/gabor.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/gldm.h"
#include "../src/nyx/features/gldzm.h"
#include "../src/nyx/features/glrlm.h"
#include "../src/nyx/features/glszm.h"
#include "../src/nyx/features/intensity.h"
#include "../src/nyx/features/intensity_histogram.h"
#include "../src/nyx/features/ngldm.h"
#include "../src/nyx/features/ngtdm.h"
#include "../src/nyx/features/power_spectrum.h"
#include "../src/nyx/features/saturation.h"
#include "../src/nyx/features/sharpness.h"

namespace Nyxus
{
	bool featurize_wholeslide (Environment& env, size_t sidx, ImageLoader& imlo, LR& vroi);
}

namespace
{
	const uint32_t aniso2_W = 24, aniso2_H = 20, aniso2_tile = 16;
	const double aniso2_ax = 1.3, aniso2_ay = 0.7;

	// label 1: an ellipse; label 2: a rectangle with one corner notched out; 0 elsewhere
	uint16_t aniso2_label (uint32_t x, uint32_t y)
	{
		const double ex = (x - 6.5) / 4.6, ey = (y - 9.5) / 7.2;
		if (ex * ex + ey * ey <= 1.0)
			return 1;
		if (x >= 13 && x <= 21 && y >= 3 && y <= 16 && ! (x >= 18 && y >= 12))
			return 2;
		return 0;
	}

	uint16_t aniso2_inten (uint32_t x, uint32_t y)
	{
		switch (aniso2_label (x, y))
		{
		case 1:
			return (uint16_t) (100 + (x * 37 + y * 17) % 61);
		case 2:
			return (uint16_t) (400 + (x * 13 + y * 29) % 47);
		default:
			return (uint16_t) (3 + (x + 2 * y) % 5);
		}
	}

	// One tiled 16-bit page of the slide, 'val' giving each pixel
	void write_aniso2_slide (const fs::path& f, uint16_t (*val) (uint32_t, uint32_t), uint32_t tile = aniso2_tile)
	{
		TIFF* t = TIFFOpen (f.string().c_str(), "w");
		ASSERT_NE(t, nullptr) << f.string();
		TIFFSetField (t, TIFFTAG_IMAGEWIDTH, aniso2_W);
		TIFFSetField (t, TIFFTAG_IMAGELENGTH, aniso2_H);
		TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, 16);
		TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
		TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
		TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField (t, TIFFTAG_TILEWIDTH, tile);
		TIFFSetField (t, TIFFTAG_TILELENGTH, tile);
		std::vector<uint16_t> buf (tile * tile);
		for (uint32_t y0 = 0; y0 < aniso2_H; y0 += tile)
			for (uint32_t x0 = 0; x0 < aniso2_W; x0 += tile)
			{
				for (uint32_t r = 0; r < tile; r++)
					for (uint32_t c = 0; c < tile; c++)
						buf[r * tile + c] = (x0 + c < aniso2_W && y0 + r < aniso2_H) ? val (x0 + c, y0 + r) : 0;
				ASSERT_GE(TIFFWriteTile (t, buf.data(), x0, y0, 0, 0), 0) << f.string();
			}
		ASSERT_EQ(TIFFWriteDirectory (t), 1) << f.string();
		TIFFClose (t);
	}

	struct Aniso2Pair
	{
		fs::path dir, inten, mask;

		Aniso2Pair (const char* name, uint16_t (*lab) (uint32_t, uint32_t) = aniso2_label, uint32_t tile = aniso2_tile)
		{
			dir = fs::temp_directory_path() / name;
			std::error_code ec;
			fs::remove_all (dir, ec);
			fs::create_directories (dir);
			inten = dir / "i.tif";
			mask = dir / "m.tif";
			write_aniso2_slide (inten, aniso2_inten, tile);
			write_aniso2_slide (mask, lab, tile);
		}
		~Aniso2Pair()
		{
			std::error_code ec;
			fs::remove_all (dir, ec);
		}
	};

	// The pixels of label 'lab' the resampling at (ax, ay) emits: every virtual pixel below
	// (size_t)(extent * factor) carries the physical pixel its coordinate truncates back to
	std::vector<std::pair<size_t, size_t>> aniso2_virtual_pixels (uint16_t (*lab_of) (uint32_t, uint32_t), int lab, double ax, double ay)
	{
		std::vector<std::pair<size_t, size_t>> v;
		const size_t vw = (size_t) (double(aniso2_W) * ax), vh = (size_t) (double(aniso2_H) * ay);
		for (size_t vr = 0; vr < vh; vr++)
			for (size_t vc = 0; vc < vw; vc++)
				if (lab_of ((uint32_t) (double(vc) / ax), (uint32_t) (double(vr) / ay)) == lab)
					v.push_back ({ vc, vr });
		return v;
	}

	// An environment with every 2D and image-quality feature requested (the focus-score family only
	// when 'with_focus'), prescanned over the pair ('mask' empty for a whole slide), with the factors
	// given as --aniso* when 'anisotropic' and the pair open in the environment's loader
	void prepare_aniso2_env (Environment& e, bool anisotropic, const std::string& ipath, const std::string& mpath,
		double ax = aniso2_ax, double ay = aniso2_ay, bool with_focus = true)
	{
		e.set_dim (2);
		e.singleROI = mpath.empty();
		e.theFeatureSet.enableAll (true);
		if (! with_focus)
			e.theFeatureSet.enableFeatures (FocusScoreFeature::featureset, false);
		ASSERT_TRUE(e.theFeatureMgr.compile());
		e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
		ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
		e.compile_feature_settings();
		e.refresh_feature_settings_singleroi();
		ASSERT_TRUE(e.set_ram_limit (64));
		if (anisotropic)
		{
			e.anisoOptions.set_aniso_x (ax);
			e.anisoOptions.set_aniso_y (ay);
		}

		SlideProps& sp = e.dataset.dataset_props.emplace_back (ipath, mpath);
		ASSERT_TRUE(Nyxus::scan_slide_props (sp, 2, e.anisoOptions, e.use_physical_spacing(),
			e.fpimageOptions, e.resultOptions.need_annotation()));
		e.dataset.update_dataset_props_extrema();
		ASSERT_TRUE(e.theImLoader.open (sp, e.fpimageOptions));
	}

	using Aniso2Values = std::vector<std::vector<double>>;

	// Segmented pass over the pair: label -> values
	std::map<int, Aniso2Values> run_aniso2_segmented (const Aniso2Pair& p, bool anisotropic)
	{
		std::map<int, Aniso2Values> out;
		Environment e;
		prepare_aniso2_env (e, anisotropic, p.inten.string(), p.mask.string());
		EXPECT_TRUE(Nyxus::gatherRoisMetrics (0, p.inten.string(), p.mask.string(), e, e.theImLoader));
		std::vector<int> labels (e.uniqueLabels.begin(), e.uniqueLabels.end());
		std::sort (labels.begin(), labels.end());
		EXPECT_EQ(labels, std::vector<int>({ 1, 2 }));
		for (auto lab : labels)
			e.roiData[lab].initialize_fvals();
		EXPECT_TRUE(Nyxus::processTrivialRois (e, labels, p.inten.string(), p.mask.string(), e.get_ram_limit()));
		e.theImLoader.close();
		for (auto lab : labels)
			out[lab] = e.roiData[lab].fvals;
		return out;
	}

	// Whole-slide pass over the intensity slide
	Aniso2Values run_aniso2_wholeslide (const Aniso2Pair& p, bool anisotropic)
	{
		Environment e;
		prepare_aniso2_env (e, anisotropic, p.inten.string(), "");
		LR vroi (1);
		EXPECT_TRUE(Nyxus::featurize_wholeslide (e, 0, e.theImLoader, vroi));
		e.theImLoader.close();
		return vroi.fvals;
	}

	bool aniso2_same (double a, double b)
	{
		return (std::isnan (a) && std::isnan (b)) || a == b;
	}

	// The features of the families defined on the image grid, listed here independently of the
	// split the pipeline makes
	std::vector<int> aniso2_grid_features()
	{
		std::vector<int> v;
		for (auto F : { PixelIntensityFeatures::featureset, IntensityHistogramFeatures::featureset,
			GLCMFeature::featureset, GLRLMFeature::featureset, GLDZMFeature::featureset, GLSZMFeature::featureset,
			GLDMFeature::featureset, NGLDMfeature::featureset, NGTDMFeature::featureset, GaborFeature::featureset })
			for (auto f : F)
				v.push_back ((int) f);
		for (auto F : { FocusScoreFeature::featureset, PowerSpectrumFeature::featureset, SaturationFeature::featureset,
			SharpnessFeature::featureset })
			for (auto f : F)
				v.push_back ((int) f);
		return v;
	}

	// Every feature of a grid family carries the value the run without anisotropy reports
	void expect_aniso2_grid_invariant (const Aniso2Values& iso, const Aniso2Values& aniso, const std::string& what)
	{
		for (int f : aniso2_grid_features())
		{
			const auto& want = iso[f];
			const auto& got = aniso[f];
			ASSERT_EQ(got.size(), want.size()) << what << " feature " << f;
			for (size_t i = 0; i < want.size(); i++)
				EXPECT_TRUE(aniso2_same (got[i], want[i]))
					<< what << " feature " << f << "[" << i << "]: " << got[i] << " with anisotropy, " << want[i] << " without";
		}
	}
}

// Segmented 2D pass. What this discriminates: a pass that feeds the grid families the resampled
// cloud reports them on duplicated and dropped pixels, so first-order, the histogram and every
// texture value move; and a pass that feeds the geometric families the pixels as acquired reports
// the physical pixel count rather than the resampled one.
void test_2d_anisotropy_segmented_splits_the_families_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_segmented");
	auto iso = run_aniso2_segmented (p, false),
		aniso = run_aniso2_segmented (p, true);

	for (int lab : { 1, 2 })
	{
		const std::string what = "ROI " + std::to_string (lab);
		expect_aniso2_grid_invariant (iso[lab], aniso[lab], what);

		// the geometric families measured the resampled cloud, box and all
		auto vp = aniso2_virtual_pixels (aniso2_label, lab, aniso2_ax, aniso2_ay);
		ASSERT_FALSE(vp.empty());
		EXPECT_EQ(aniso[lab][(int) Nyxus::Feature2D::AREA_PIXELS_COUNT][0], (double) vp.size()) << what;
		size_t xmin = vp[0].first, xmax = xmin, ymin = vp[0].second, ymax = ymin;
		for (auto [x, y] : vp)
		{
			xmin = (std::min) (xmin, x); xmax = (std::max) (xmax, x);
			ymin = (std::min) (ymin, y); ymax = (std::max) (ymax, y);
		}
		EXPECT_EQ(aniso[lab][(int) Nyxus::Feature2D::BBOX_WIDTH][0], (double) (xmax - xmin + 1)) << what;
		EXPECT_EQ(aniso[lab][(int) Nyxus::Feature2D::BBOX_HEIGHT][0], (double) (ymax - ymin + 1)) << what;
	}
}

// Whole-slide 2D pass, which takes its box from the prescan rather than from phase 1
void test_2d_anisotropy_wholeslide_splits_the_families_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_wholeslide");
	auto iso = run_aniso2_wholeslide (p, false),
		aniso = run_aniso2_wholeslide (p, true);
	expect_aniso2_grid_invariant (iso, aniso, "whole slide");
	const double want = double((size_t) (aniso2_W * aniso2_ax) * (size_t) (aniso2_H * aniso2_ay));
	EXPECT_EQ(aniso[(int) Nyxus::Feature2D::AREA_PIXELS_COUNT][0], want) << "the geometric families did not measure the resampled slide";
}

// A ROI one pixel wide in a column the resampling at 0.4 never reads: no virtual pixel carries it,
// so it has no geometry to measure. The pass refuses it by name rather than reporting its shape
// as zeros. What this discriminates: a pass that reduces the empty cloud anyway returns true.
void test_2d_anisotropy_vanished_roi_is_refused_mechanics()
{
	struct Thin
	{
		static uint16_t label (uint32_t x, uint32_t y) { return (x == 1 && y >= 4 && y <= 12) ? 7 : 0; }
	};
	ASSERT_TRUE(aniso2_virtual_pixels (Thin::label, 7, 0.4, 1.0).empty()) << "the fixture's column is read after all";

	Aniso2Pair p ("nyxus_2d_aniso_vanished", Thin::label);
	Environment e;
	// the focus-score family is left out: its local-score tile walk does not end on a ROI narrower
	// than its tile scale, which this one-pixel ROI is
	prepare_aniso2_env (e, true, p.inten.string(), p.mask.string(), 0.4, 1.0, /*with_focus=*/ false);
	ASSERT_TRUE(Nyxus::gatherRoisMetrics (0, p.inten.string(), p.mask.string(), e, e.theImLoader));
	ASSERT_EQ(e.uniqueLabels.size(), 1u);
	e.roiData[7].initialize_fvals();

	testing::internal::CaptureStderr();
	bool ok = Nyxus::processTrivialRois (e, { 7 }, p.inten.string(), p.mask.string(), e.get_ram_limit());
	const std::string err = testing::internal::GetCapturedStderr();
	e.theImLoader.close();

	EXPECT_FALSE(ok) << "a ROI with no resampled pixel was reduced";
	EXPECT_NE(err.find ("ROI 7 maps to no pixel"), std::string::npos) << err;
}

// The whole-slide trivial/oversized decision on an anisotropic run covers the pass over the slide
// as acquired. The prescan scales the slide's box by the factors, so at (0.5, 0.5) its box is a
// quarter of the slide the first pass holds. The RAM limit goes between the two estimates. What
// this discriminates: a decision built on the prescan's box reads the slide as trivial and
// featurizes it in a pass that does not fit the limit; sized from the slide as acquired, it is
// refused.
void test_2d_anisotropy_wholeslide_sized_as_acquired_mechanics()
{
	const uint32_t W = 512, H = 512, TILE = 256;
	const fs::path dir = fs::temp_directory_path() / "nyxus_2d_aniso_wholeslide_size";
	std::error_code ec;
	fs::remove_all (dir, ec);
	fs::create_directories (dir);
	const fs::path ip = dir / "i.tif";
	{
		TIFF* t = TIFFOpen (ip.string().c_str(), "w");
		ASSERT_NE(t, nullptr) << ip.string();
		TIFFSetField (t, TIFFTAG_IMAGEWIDTH, W);
		TIFFSetField (t, TIFFTAG_IMAGELENGTH, H);
		TIFFSetField (t, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField (t, TIFFTAG_BITSPERSAMPLE, 16);
		TIFFSetField (t, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
		TIFFSetField (t, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
		TIFFSetField (t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField (t, TIFFTAG_TILEWIDTH, TILE);
		TIFFSetField (t, TIFFTAG_TILELENGTH, TILE);
		std::vector<uint16_t> buf (TILE * TILE);
		for (uint32_t y0 = 0; y0 < H; y0 += TILE)
			for (uint32_t x0 = 0; x0 < W; x0 += TILE)
			{
				for (uint32_t r = 0; r < TILE; r++)
					for (uint32_t c = 0; c < TILE; c++)
						buf[r * TILE + c] = (uint16_t) (1 + ((x0 + c) * 7 + (y0 + r) * 3) % 251);
				ASSERT_GE(TIFFWriteTile (t, buf.data(), x0, y0, 0, 0), 0);
			}
		ASSERT_EQ(TIFFWriteDirectory (t), 1);
		TIFFClose (t);
	}

	Environment e;
	prepare_aniso2_env (e, true, ip.string(), "", 0.5, 0.5);
	const SlideProps& p = e.dataset.dataset_props[0];
	ASSERT_LT(p.max_roi_w, (size_t) W) << "the prescan box is the scaled one";

	// the estimate the prescan's box gives, and the one the slide as acquired gives
	LR scaled (1), acquired (1);
	scaled.aux_area = acquired.aux_area = p.max_roi_area;
	scaled.aabb.init_from_wh (p.max_roi_w, p.max_roi_h);
	acquired.aabb.init_from_wh (W, H);
	const size_t f_scaled = scaled.get_ram_footprint_estimate (1),
		f_acquired = acquired.get_ram_footprint_estimate (1);
	const size_t limit_mb = ((f_scaled + f_acquired) / 2) / (1024 * 1024);
	ASSERT_GT(limit_mb * 1024 * 1024, f_scaled);
	ASSERT_LT(limit_mb * 1024 * 1024, f_acquired) << "the two estimates must straddle a megabyte boundary";
	ASSERT_TRUE(e.set_ram_limit (limit_mb));

	LR vroi (1);
	testing::internal::CaptureStderr();
	const bool ok = Nyxus::featurize_wholeslide (e, 0, e.theImLoader, vroi);
	const std::string err = testing::internal::GetCapturedStderr();
	e.theImLoader.close();
	fs::remove_all (dir, ec);

	EXPECT_FALSE(ok) << "the slide as acquired does not fit the limit, and its first pass holds all of it";
	EXPECT_NE(err.find ("slide is non-trivial"), std::string::npos) << err;
}

namespace
{
	// An environment with first-order and basic morphology requested -- the out-of-core pass of both
	// is the in-RAM one -- at factors (ax, ay), through phase 1 over the pair, with the pair open
	void prepare_aniso2_ooc_env (Environment& e, const Aniso2Pair& p, double ax, double ay)
	{
		e.set_dim (2);
		e.singleROI = false;
		e.theFeatureSet.enableAll (false);
		e.theFeatureSet.enableFeatures (PixelIntensityFeatures::featureset);
		e.theFeatureSet.enableFeatures (BasicMorphologyFeatures::featureset);
		ASSERT_TRUE(e.theFeatureMgr.compile());
		e.theFeatureMgr.apply_user_selection (e.theFeatureSet);
		ASSERT_TRUE(e.theFeatureMgr.init_feature_classes());
		e.compile_feature_settings();
		e.refresh_feature_settings_singleroi();
		ASSERT_TRUE(e.set_ram_limit (64));
		e.anisoOptions.set_aniso_x (ax);
		e.anisoOptions.set_aniso_y (ay);
		SlideProps& sp = e.dataset.dataset_props.emplace_back (p.inten.string(), p.mask.string());
		ASSERT_TRUE(Nyxus::scan_slide_props (sp, 2, e.anisoOptions, e.use_physical_spacing(),
			e.fpimageOptions, e.resultOptions.need_annotation()));
		e.dataset.update_dataset_props_extrema();
		ASSERT_TRUE(e.theImLoader.open (sp, e.fpimageOptions));
		ASSERT_TRUE(Nyxus::gatherRoisMetrics (0, p.inten.string(), p.mask.string(), e, e.theImLoader));
		for (auto lab : e.uniqueLabels)
			e.roiData[lab].initialize_fvals();
	}
}

// An oversized anisotropic 2D ROI is featurized out-of-core as an in-RAM one is: the grid families
// over the pixels as acquired, the geometric families over the ROI resampled by the factors. The
// slide fits one 32-px tile. What this discriminates: an out-of-core pass that streams only the
// pixels as acquired reports the ROI's shape in pixel units while the same ROI under the RAM limit
// reports it resampled, so one ROI's geometry would depend on whether it fits in RAM.
void test_2d_anisotropy_out_of_core_matches_in_ram_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_ooc", aniso2_label, 32);

	Environment in_ram, ooc;
	prepare_aniso2_ooc_env (in_ram, p, aniso2_ax, aniso2_ay);
	prepare_aniso2_ooc_env (ooc, p, aniso2_ax, aniso2_ay);
	const std::vector<int> labels = { 1, 2 };
	ASSERT_TRUE(Nyxus::processTrivialRois (in_ram, labels, p.inten.string(), p.mask.string(), in_ram.get_ram_limit()));
	ASSERT_TRUE(Nyxus::processNontrivialRois (ooc, labels, p.inten.string(), p.mask.string()));
	in_ram.theImLoader.close();
	ooc.theImLoader.close();

	for (int lab : labels)
	{
		for (auto F : { PixelIntensityFeatures::featureset, BasicMorphologyFeatures::featureset })
			for (auto f : F)
			{
				const double want = in_ram.roiData[lab].fvals[(int) f][0],
					got = ooc.roiData[lab].fvals[(int) f][0];
				EXPECT_NEAR(got, want, 1e-9 * (std::max) (1.0, std::abs (want)))
					<< "ROI " << lab << " feature " << (int) f << " differs out-of-core";
			}

		// and the geometry is the resampled ROI's
		EXPECT_EQ(ooc.roiData[lab].fvals[(int) Nyxus::Feature2D::AREA_PIXELS_COUNT][0],
			(double) aniso2_virtual_pixels (aniso2_label, lab, aniso2_ax, aniso2_ay).size()) << "ROI " << lab;
	}
}

// The out-of-core twin of the vanished-ROI refusal: a ROI the resampling leaves with no pixel is
// refused by name rather than reported with zero geometry.
void test_2d_anisotropy_out_of_core_vanished_roi_is_refused_mechanics()
{
	struct Thin
	{
		static uint16_t label (uint32_t x, uint32_t y) { return (x == 1 && y >= 4 && y <= 12) ? 7 : 0; }
	};
	ASSERT_TRUE(aniso2_virtual_pixels (Thin::label, 7, 0.4, 1.0).empty());

	Aniso2Pair p ("nyxus_2d_aniso_ooc_vanished", Thin::label, 32);
	Environment e;
	prepare_aniso2_ooc_env (e, p, 0.4, 1.0);
	ASSERT_EQ(e.uniqueLabels.size(), 1u);

	testing::internal::CaptureStderr();
	const bool ok = Nyxus::processNontrivialRois (e, { 7 }, p.inten.string(), p.mask.string());
	const std::string err = testing::internal::GetCapturedStderr();
	e.theImLoader.close();

	EXPECT_FALSE(ok) << "a ROI with no resampled pixel was reduced out-of-core";
	EXPECT_NE(err.find ("ROI 7 maps to no pixel"), std::string::npos) << err;
}
