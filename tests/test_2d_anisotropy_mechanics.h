#pragma once

// Anisotropy on the 2D paths: which pixels each family measures.
//
// An anisotropic 2D run scans a batch twice. The families defined on the image grid (first-order,
// the intensity histogram, the texture families, Gabor and the image-quality families) reduce the
// pixels as acquired, so their values are those of a run without anisotropy. The intensity-weighted
// features of the geometric families (EDGE_*, WEIGHTED_CENTROID_X/Y, MASS_DISPLACEMENT, IMOM_*, the
// radial distribution and ZERNIKE2D) are measured on the pixels as acquired too, each pixel as a
// pixel of the spacing's size. The other geometric features reduce the cloud resampled by the
// spacing, so they measure the ROI in physical space.
//
// The fixture is a 24x20 slide with two labels of different shapes, an ellipse and a notched
// rectangle, under the factors (1.3, 0.7): unequal and inexact, so the resampling duplicates some
// columns and drops some rows, and a grid family fed the resampled cloud moves.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include "test_main_nyxus.h"		// gtest, libtiff, fs, globals.h, roi_cache.h
#include "../src/nyx/environment.h"
#include "../src/nyx/features/2d_geomoments.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/contour.h"
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
#include "../src/nyx/features/radial_distribution.h"
#include "../src/nyx/features/saturation.h"
#include "../src/nyx/features/sharpness.h"
#include "../src/nyx/features/zernike.h"

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

	// label 7: one pixel wide at column 1, which a 0.4 resampling never reads; label 3: a block beside it
	uint16_t aniso2_thin_label (uint32_t x, uint32_t y)
	{
		if (x == 1 && y >= 4 && y <= 12)
			return 7;
		if (x >= 8 && x <= 18 && y >= 3 && y <= 14)
			return 3;
		return 0;
	}

	using Aniso2Values = std::vector<std::vector<double>>;

	// Segmented pass over the pair: label -> values
	std::map<int, Aniso2Values> run_aniso2_segmented (const Aniso2Pair& p, bool anisotropic, double ax = aniso2_ax, double ay = aniso2_ay)
	{
		std::map<int, Aniso2Values> out;
		Environment e;
		prepare_aniso2_env (e, anisotropic, p.inten.string(), p.mask.string(), ax, ay);
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

	uint16_t aniso2_whole (uint32_t, uint32_t)
	{
		return 1;
	}

	// The intensity-weighted features of the geometric families that need no contour, computed here
	// from the fixture's pixels of label 'lab' as pixels of size (ax, ay): each pixel weighs its
	// intensity times its area; the centroids place it at its centre, (x + 1/2) ax - 1/2, the moments
	// at its offset from the ROI's first column and row, (x - xmin) ax. Hu's invariants are the
	// standard ones (skimage.measure.moments_hu). Feature code -> value and the tolerance it is held to:
	// 1e-10 relative, except IMOM_CM_01 and IMOM_CM_10, which are 0 up to rounding and are held to the
	// first raw moments' scale.
	struct Aniso2Ref { double want, tol; };
	std::map<int, Aniso2Ref> aniso2_as_acquired_reference (uint16_t (*lab_of) (uint32_t, uint32_t), int lab, double ax, double ay)
	{
		struct P { double x, y, w; };
		std::vector<P> px;
		uint32_t xmin = aniso2_W, ymin = aniso2_H;
		for (uint32_t y = 0; y < aniso2_H; y++)
			for (uint32_t x = 0; x < aniso2_W; x++)
				if (lab_of (x, y) == lab)
				{
					px.push_back ({ (double) x, (double) y, (double) aniso2_inten (x, y) });
					xmin = (std::min) (xmin, x);
					ymin = (std::min) (ymin, y);
				}

		std::map<int, double> ref;
		double n = 0, sx = 0, sy = 0, wx = 0, wy = 0, w = 0;
		for (auto& p : px)
		{
			const double u = (p.x + 0.5) * ax - 0.5, v = (p.y + 0.5) * ay - 0.5;
			n += 1; sx += u; sy += v;
			wx += u * p.w; wy += v * p.w; w += p.w;
		}
		const double wcx = wx / w, wcy = wy / w;
		ref[(int) Nyxus::Feature2D::WEIGHTED_CENTROID_X] = wcx;
		ref[(int) Nyxus::Feature2D::WEIGHTED_CENTROID_Y] = wcy;
		ref[(int) Nyxus::Feature2D::MASS_DISPLACEMENT] = std::hypot (wcx - sx / n, wcy - sy / n);

		auto raw = [&] (int p, int q)
		{
			double s = 0;
			for (auto& k : px)
				s += k.w * ax * ay * std::pow ((k.x - xmin) * ax, p) * std::pow ((k.y - ymin) * ay, q);
			return s;
		};
		const double m00 = raw (0, 0), cx = raw (1, 0) / m00, cy = raw (0, 1) / m00;
		auto central = [&] (int p, int q)
		{
			double s = 0;
			for (auto& k : px)
				s += k.w * ax * ay * std::pow ((k.x - xmin) * ax - cx, p) * std::pow ((k.y - ymin) * ay - cy, q);
			return s;
		};
		auto eta = [&] (int p, int q) { return central (p, q) / std::pow (m00, (p + q) / 2.0 + 1.0); };

		// every IMOM_RM_pq, IMOM_CM_pq, IMOM_NRM_pq and IMOM_NCM_pq the registry names
		for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
		{
			int p = -1, q = -1;
			if (std::sscanf (name.c_str(), "IMOM_RM_%1d%1d", &p, &q) == 2)
				ref[(int) code] = raw (p, q);
			else if (std::sscanf (name.c_str(), "IMOM_CM_%1d%1d", &p, &q) == 2)
				ref[(int) code] = central (p, q);
			else if (std::sscanf (name.c_str(), "IMOM_NRM_%1d%1d", &p, &q) == 2)
				ref[(int) code] = raw (p, q) / std::pow (m00, (p + q) / 2.0 + 1.0);
			else if (std::sscanf (name.c_str(), "IMOM_NCM_%1d%1d", &p, &q) == 2)
				ref[(int) code] = eta (p, q);
		}

		const double n20 = eta (2, 0), n02 = eta (0, 2), n11 = eta (1, 1), n30 = eta (3, 0), n03 = eta (0, 3),
			n21 = eta (2, 1), n12 = eta (1, 2);
		const double a = n30 + n12, b = n21 + n03;
		ref[(int) Nyxus::Feature2D::IMOM_HU1] = n20 + n02;
		ref[(int) Nyxus::Feature2D::IMOM_HU2] = (n20 - n02) * (n20 - n02) + 4 * n11 * n11;
		ref[(int) Nyxus::Feature2D::IMOM_HU3] = (n30 - 3 * n12) * (n30 - 3 * n12) + (3 * n21 - n03) * (3 * n21 - n03);
		ref[(int) Nyxus::Feature2D::IMOM_HU4] = a * a + b * b;
		ref[(int) Nyxus::Feature2D::IMOM_HU5] = (n30 - 3 * n12) * a * (a * a - 3 * b * b) + (3 * n21 - n03) * b * (3 * a * a - b * b);
		ref[(int) Nyxus::Feature2D::IMOM_HU6] = (n20 - n02) * (a * a - b * b) + 4 * n11 * a * b;
		ref[(int) Nyxus::Feature2D::IMOM_HU7] = (3 * n21 - n03) * a * (a * a - 3 * b * b) - (n30 - 3 * n12) * b * (3 * a * a - b * b);

		std::map<int, Aniso2Ref> held;
		for (const auto& [f, want] : ref)
			held[f] = { want, 1e-10 * std::abs (want) + 1e-30 };
		const double first = 1e-12 * (std::abs (raw (1, 0)) + std::abs (raw (0, 1)));
		held[(int) Nyxus::Feature2D::IMOM_CM_01].tol = held[(int) Nyxus::Feature2D::IMOM_CM_10].tol = first;
		return held;
	}

	// Every feature 'ref' holds carries its reference value
	void expect_aniso2_as_acquired (const Aniso2Values& got, const std::map<int, Aniso2Ref>& ref, const std::string& what)
	{
		ASSERT_EQ(ref.size(), 3u + 59u) << "the reference does not cover the centroids and the 59 moments";
		for (const auto& [f, r] : ref)
			EXPECT_NEAR(got[f][0], r.want, r.tol)
				<< what << " feature " << f << ": " << got[f][0] << ", measured on the pixels as acquired " << r.want;
	}

	// ROI 'gone' (the emptied one): geometry not available, grid and as-acquired features measured;
	// ROI 'kept': its geometry is that of its resampled pixels
	void expect_aniso2_vanished_roi (const LR& gone, const LR& kept, double ax, double ay)
	{
		for (auto f : { Nyxus::Feature2D::AREA_PIXELS_COUNT, Nyxus::Feature2D::CENTROID_X, Nyxus::Feature2D::BBOX_WIDTH })
			EXPECT_TRUE(std::isnan (gone.fvals[(int) f][0])) << "feature " << (int) f << " of the emptied ROI is " << gone.fvals[(int) f][0];
		EXPECT_FALSE(std::isnan (gone.fvals[(int) Nyxus::Feature2D::MEAN][0])) << "the emptied ROI's grid features are measured";
		EXPECT_GT(gone.fvals[(int) Nyxus::Feature2D::MEAN][0], 0.0);
		EXPECT_EQ(kept.fvals[(int) Nyxus::Feature2D::AREA_PIXELS_COUNT][0],
			(double) aniso2_virtual_pixels (aniso2_thin_label, 3, ax, ay).size()) << "the ROI beside it was not measured";

		// its pixels as acquired are all there, so its as-acquired features are measured
		expect_aniso2_as_acquired (gone.fvals, aniso2_as_acquired_reference (aniso2_thin_label, gone.label, ax, ay), "the emptied ROI");
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
// so it has no geometry to measure. Its geometric features are reported as not available (NaN, which
// the writers emit as the soft-NaN value) with a warning naming it, its grid features are measured,
// and the run carries on with the ROI beside it. What this discriminates: a pass that stops at the
// emptied ROI returns false and leaves the other ROI's geometry unmeasured; one that reduces the
// empty cloud reports a shape of zeros.
void test_2d_anisotropy_vanished_roi_reports_no_geometry_mechanics()
{
	ASSERT_TRUE(aniso2_virtual_pixels (aniso2_thin_label, 7, 0.4, 1.0).empty()) << "the fixture's column is read after all";

	Aniso2Pair p ("nyxus_2d_aniso_vanished", aniso2_thin_label);
	Environment e;
	// the focus-score family is left out: its local-score tile walk does not end on a ROI narrower
	// than its tile scale, which the one-pixel ROI is
	prepare_aniso2_env (e, true, p.inten.string(), p.mask.string(), 0.4, 1.0, /*with_focus=*/ false);
	ASSERT_TRUE(Nyxus::gatherRoisMetrics (0, p.inten.string(), p.mask.string(), e, e.theImLoader));
	ASSERT_EQ(e.uniqueLabels.size(), 2u);
	for (auto lab : e.uniqueLabels)
		e.roiData[lab].initialize_fvals();

	testing::internal::CaptureStderr();
	const bool ok = Nyxus::processTrivialRois (e, { 3, 7 }, p.inten.string(), p.mask.string(), e.get_ram_limit());
	const std::string err = testing::internal::GetCapturedStderr();
	e.theImLoader.close();

	EXPECT_TRUE(ok) << "one emptied ROI stopped the run";
	EXPECT_NE(err.find ("ROI 7 maps to no pixel"), std::string::npos) << err;
	expect_aniso2_vanished_roi (e.roiData[7], e.roiData[3], 0.4, 1.0);
	EXPECT_FALSE(std::isnan (e.roiData[7].fvals[(int) Nyxus::Feature2D::ZERNIKE2D][0])) << "ZERNIKE2D is measured as acquired";
	EXPECT_EQ(e.roiData[7].fvals[(int) Nyxus::Feature2D::ZERNIKE2D].size(), (size_t) ZernikeFeature::NUM_FEATURE_VALS)
		<< "the writers read every Zernike sub-value";
	EXPECT_TRUE(std::isnan (e.roiData[7].fvals[(int) Nyxus::Feature2D::PERIMETER][0])) << "the contour method's geometry is not available";
	EXPECT_TRUE(std::isnan (e.roiData[7].fvals[(int) Nyxus::Feature2D::NUM_NEIGHBORS][0])) << "the neighbor pass measured it";
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
	// first-order with its HISTOGRAM, and basic morphology: the out-of-core pass of both is the
	// in-RAM one
	void select_aniso2_ooc_default (FeatureSet& fs)
	{
		fs.enableFeatures (PixelIntensityFeatures::featureset);
		fs.enableFeatures ({ Nyxus::Feature2D::HISTOGRAM });
		fs.enableFeatures (BasicMorphologyFeatures::featureset);
	}

	// the default, and the intensity moments: with basic morphology's weighted centroids, as-acquired
	// features
	void select_aniso2_ooc_with_moments (FeatureSet& fs)
	{
		select_aniso2_ooc_default (fs);
		fs.enableFeatures (Imoms2D_feature::featureset);
	}

	// An environment with 'select' requested at factors (ax, ay), through phase 1 over the pair, with
	// the pair open
	void prepare_aniso2_ooc_env (Environment& e, const Aniso2Pair& p, double ax, double ay,
		void (*select) (FeatureSet&) = select_aniso2_ooc_default)
	{
		e.set_dim (2);
		e.singleROI = false;
		e.theFeatureSet.enableAll (false);
		select (e.theFeatureSet);
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
// slide fits one 32-px tile. HISTOGRAM is requested too: it belongs to the first-order method but not
// to its featureset. What this discriminates: an out-of-core pass that streams only the pixels as
// acquired reports the ROI's shape in pixel units while the same ROI under the RAM limit reports it
// resampled; one that files HISTOGRAM as geometric reruns the whole first-order method on the
// resampled pixels and overwrites MEAN and the rest.
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
		EXPECT_EQ(ooc.roiData[lab].fvals[(int) Nyxus::Feature2D::HISTOGRAM], in_ram.roiData[lab].fvals[(int) Nyxus::Feature2D::HISTOGRAM])
			<< "ROI " << lab << " HISTOGRAM differs out-of-core";

		// and the geometry is the resampled ROI's
		EXPECT_EQ(ooc.roiData[lab].fvals[(int) Nyxus::Feature2D::AREA_PIXELS_COUNT][0],
			(double) aniso2_virtual_pixels (aniso2_label, lab, aniso2_ax, aniso2_ay).size()) << "ROI " << lab;
	}
}

// The out-of-core twin: the emptied ROI's geometric features are not available, its grid and
// as-acquired features are measured, and the ROI beside it is featurized.
void test_2d_anisotropy_out_of_core_vanished_roi_reports_no_geometry_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_ooc_vanished", aniso2_thin_label, 32);
	Environment e;
	prepare_aniso2_ooc_env (e, p, 0.4, 1.0, select_aniso2_ooc_with_moments);
	ASSERT_EQ(e.uniqueLabels.size(), 2u);

	testing::internal::CaptureStderr();
	const bool ok = Nyxus::processNontrivialRois (e, { 3, 7 }, p.inten.string(), p.mask.string());
	const std::string err = testing::internal::GetCapturedStderr();
	e.theImLoader.close();

	EXPECT_TRUE(ok) << "one emptied ROI stopped the out-of-core run";
	EXPECT_NE(err.find ("ROI 7 maps to no pixel"), std::string::npos) << err;
	expect_aniso2_vanished_roi (e.roiData[7], e.roiData[3], 0.4, 1.0);
}

// The out-of-core resampled pass runs a geometric method requested only as another's dependency.
// CIRCULARITY comes from the convex-hull method, which needs the contour method's PERIMETER. The
// out-of-core contour is compared with itself rather than with the in-RAM one, which it does not
// match even without anisotropy. What this discriminates: a pass that runs only the methods
// providing a selected feature skips the contour method when CIRCULARITY alone is selected, and
// CIRCULARITY then comes out 0.
void test_2d_anisotropy_out_of_core_runs_dependencies_mechanics()
{
	struct Select
	{
		static void circularity (FeatureSet& fs) { fs.enableFeatures ({ Nyxus::Feature2D::CIRCULARITY }); }
		static void with_perimeter (FeatureSet& fs) { fs.enableFeatures ({ Nyxus::Feature2D::CIRCULARITY, Nyxus::Feature2D::PERIMETER }); }
	};
	Aniso2Pair p ("nyxus_2d_aniso_ooc_deps", aniso2_label, 32);
	Environment alone, both;
	prepare_aniso2_ooc_env (alone, p, aniso2_ax, aniso2_ay, Select::circularity);
	prepare_aniso2_ooc_env (both, p, aniso2_ax, aniso2_ay, Select::with_perimeter);
	ASSERT_TRUE(Nyxus::processNontrivialRois (alone, { 1, 2 }, p.inten.string(), p.mask.string()));
	ASSERT_TRUE(Nyxus::processNontrivialRois (both, { 1, 2 }, p.inten.string(), p.mask.string()));
	alone.theImLoader.close();
	both.theImLoader.close();

	const int f = (int) Nyxus::Feature2D::CIRCULARITY;
	for (int lab : { 1, 2 })
	{
		EXPECT_GT(alone.roiData[lab].fvals[f][0], 0.0) << "ROI " << lab << ": the contour method did not run";
		EXPECT_EQ(alone.roiData[lab].fvals[f][0], both.roiData[lab].fvals[f][0]) << "ROI " << lab;
	}
}

namespace
{
	// every as-acquired feature (see split_2d_selection), listed here independently of it
	void select_aniso2_as_acquired (FeatureSet& fs)
	{
		fs.enableFeatures ({ Nyxus::Feature2D::WEIGHTED_CENTROID_X, Nyxus::Feature2D::WEIGHTED_CENTROID_Y,
			Nyxus::Feature2D::MASS_DISPLACEMENT });
		fs.enableFeatures ({ Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, Nyxus::Feature2D::EDGE_MAX_INTENSITY,
			Nyxus::Feature2D::EDGE_MIN_INTENSITY, Nyxus::Feature2D::EDGE_MEAN_INTENSITY, Nyxus::Feature2D::EDGE_STDDEV_INTENSITY });
		fs.enableFeatures (Imoms2D_feature::featureset);
		fs.enableFeatures (RadialDistributionFeature::featureset);
		fs.enableFeatures (ZernikeFeature::featureset);
	}

	// the methods that provide the as-acquired features, whole: the resampled pass runs the contour
	// and morphology methods for their geometric features, and overwrites the as-acquired ones
	void select_aniso2_methods_whole (FeatureSet& fs)
	{
		fs.enableFeatures (BasicMorphologyFeatures::featureset);
		fs.enableFeatures (ContourFeature::featureset);
		fs.enableFeatures (Imoms2D_feature::featureset);
		fs.enableFeatures (RadialDistributionFeature::featureset);
		fs.enableFeatures (ZernikeFeature::featureset);
	}

	// The as-acquired features of the contour, moment, radial and Zernike methods, whose values this
	// test holds to the methods themselves rather than to a reference of its own
	std::vector<int> aniso2_contour_weighted_features()
	{
		std::vector<int> v = { (int) Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, (int) Nyxus::Feature2D::EDGE_MAX_INTENSITY,
			(int) Nyxus::Feature2D::EDGE_MIN_INTENSITY, (int) Nyxus::Feature2D::EDGE_MEAN_INTENSITY,
			(int) Nyxus::Feature2D::EDGE_STDDEV_INTENSITY };
		for (auto F : { Imoms2D_feature::featureset, RadialDistributionFeature::featureset, ZernikeFeature::featureset })
			for (auto f : F)
				v.push_back ((int) f);
		return v;
	}

	// Label 'lab' of the fixture as the pass over the pixels as acquired caches it -- tile by tile,
	// row-major in each, on a slide of 'tile'-px tiles -- with its image matrix, measured by
	// measure_as_acquired_2d at (ax, ay) with every feature requested
	Aniso2Values aniso2_measured_directly (const Environment& e, int lab, uint32_t tile, double ax, double ay)
	{
		LR r (lab);
		for (uint32_t ty = 0; ty < aniso2_H; ty += tile)
			for (uint32_t tx = 0; tx < aniso2_W; tx += tile)
				for (uint32_t y = ty; y < (std::min) (ty + tile, aniso2_H); y++)
					for (uint32_t x = tx; x < (std::min) (tx + tile, aniso2_W); x++)
						if (aniso2_label (x, y) == lab)
							r.raw_pixels.push_back (Pixel2 ((int) x, (int) y, (PixIntens) aniso2_inten (x, y)));
		EXPECT_FALSE(r.raw_pixels.empty());
		r.aabb.init_x (r.raw_pixels[0].x);
		r.aabb.init_y (r.raw_pixels[0].y);
		r.aux_min = r.aux_max = r.raw_pixels[0].inten;
		for (const Pixel2& px : r.raw_pixels)
		{
			r.aabb.update_x (px.x);
			r.aabb.update_y (px.y);
			r.aux_min = (std::min) (r.aux_min, (PixIntens) px.inten);
			r.aux_max = (std::max) (r.aux_max, (PixIntens) px.inten);
		}
		r.aux_area = (unsigned int) r.raw_pixels.size();
		r.initialize_fvals();
		r.aux_image_matrix.allocate (r.aabb.get_width(), r.aabb.get_height());
		r.aux_image_matrix.calculate_from_pixelcloud (r.raw_pixels, r.aabb);

		FeatureSet all, on_grid, as_acquired, geometric;
		all.enableAll (true);
		Nyxus::split_2d_selection (all, on_grid, as_acquired, geometric);
		Nyxus::measure_as_acquired_2d (e, r, as_acquired, ax, ay);
		return r.fvals;
	}

	// Every value of 'features' in 'got' equals the one in 'want', NaN included
	void expect_aniso2_same_values (const Aniso2Values& got, const Aniso2Values& want, const std::vector<int>& features,
		double rel, const std::string& what)
	{
		for (int f : features)
		{
			ASSERT_EQ(got[f].size(), want[f].size()) << what << " feature " << f;
			for (size_t i = 0; i < want[f].size(); i++)
				if (! (std::isnan (got[f][i]) && std::isnan (want[f][i])))
					EXPECT_NEAR(got[f][i], want[f][i], rel * std::abs (want[f][i]) + 1e-30)
						<< what << " feature " << f << "[" << i << "]";
		}
	}
}

// The intensity-weighted features of the geometric families are measured on the pixels as acquired,
// as pixels of the spacing's size, in a segmented batch with every feature requested. At (1.3, 0.7)
// the resampling copies some columns and drops some rows, so the cloud it emits weighs the ROI's
// intensities unevenly. The centroids and the moments that need no contour are held to the
// reference this file computes; the rest to measure_as_acquired_2d over the ROI as acquired, and
// the EDGE_* statistics, taken along the contour of the pixels as acquired, to the run without
// anisotropy. What this discriminates: a pass that measures these features on the resampled cloud,
// or that lets the resampled pass's contour, morphology and moment methods overwrite them, misses
// the reference by 2% to a factor of several hundred; one that leaves out the pixel's area scales
// every raw and central moment by 1 / 0.91.
void test_2d_anisotropy_weighted_geometry_as_acquired_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_weighted");
	auto iso = run_aniso2_segmented (p, false),
		aniso = run_aniso2_segmented (p, true);
	Environment e;
	prepare_aniso2_env (e, true, p.inten.string(), p.mask.string());
	e.theImLoader.close();
	for (int lab : { 1, 2 })
	{
		const std::string what = "ROI " + std::to_string (lab);
		expect_aniso2_as_acquired (aniso[lab], aniso2_as_acquired_reference (aniso2_label, lab, aniso2_ax, aniso2_ay), what);
		expect_aniso2_same_values (aniso[lab], aniso2_measured_directly (e, lab, aniso2_tile, aniso2_ax, aniso2_ay),
			aniso2_contour_weighted_features(), 1e-12, what + " against the pixels as acquired");
		for (auto f : { Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, Nyxus::Feature2D::EDGE_MAX_INTENSITY,
			Nyxus::Feature2D::EDGE_MIN_INTENSITY, Nyxus::Feature2D::EDGE_MEAN_INTENSITY, Nyxus::Feature2D::EDGE_STDDEV_INTENSITY })
			EXPECT_EQ(aniso[lab][(int) f][0], iso[lab][(int) f][0]) << what << " feature " << (int) f << " moved with the anisotropy";
	}
}

// The same features with nothing else requested, in RAM and out-of-core: the batch is scanned once,
// as acquired. And out-of-core with the methods that provide them whole, whose resampled pass
// overwrites them unless they are held across it. The out-of-core pass materializes the ROI for
// them, so it measures them as the in-RAM pass does.
void test_2d_anisotropy_weighted_geometry_as_acquired_alone_and_out_of_core_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_weighted_ooc", aniso2_label, 32);
	const std::vector<int> labels = { 1, 2 };

	Environment alone, ooc_alone, ooc_whole;
	prepare_aniso2_ooc_env (alone, p, aniso2_ax, aniso2_ay, select_aniso2_as_acquired);
	prepare_aniso2_ooc_env (ooc_alone, p, aniso2_ax, aniso2_ay, select_aniso2_as_acquired);
	prepare_aniso2_ooc_env (ooc_whole, p, aniso2_ax, aniso2_ay, select_aniso2_methods_whole);
	ASSERT_TRUE(Nyxus::processTrivialRois (alone, labels, p.inten.string(), p.mask.string(), alone.get_ram_limit()));
	ASSERT_TRUE(Nyxus::processNontrivialRois (ooc_alone, labels, p.inten.string(), p.mask.string()));
	ASSERT_TRUE(Nyxus::processNontrivialRois (ooc_whole, labels, p.inten.string(), p.mask.string()));
	alone.theImLoader.close();
	ooc_alone.theImLoader.close();
	ooc_whole.theImLoader.close();

	for (int lab : labels)
	{
		const auto ref = aniso2_as_acquired_reference (aniso2_label, lab, aniso2_ax, aniso2_ay);
		const auto direct = aniso2_measured_directly (alone, lab, 32, aniso2_ax, aniso2_ay);
		const std::string what = "ROI " + std::to_string (lab);
		for (auto [env, how] : { std::pair<Environment*, const char*> { &alone, " in RAM, alone" },
			{ &ooc_alone, " out-of-core, alone" }, { &ooc_whole, " out-of-core, with the whole methods" } })
		{
			expect_aniso2_as_acquired (env->roiData[lab].fvals, ref, what + how);
			expect_aniso2_same_values (env->roiData[lab].fvals, direct, aniso2_contour_weighted_features(), 1e-12, what + how);
		}
		EXPECT_EQ(ooc_whole.roiData[lab].fvals[(int) Nyxus::Feature2D::AREA_PIXELS_COUNT][0],
			(double) aniso2_virtual_pixels (aniso2_label, lab, aniso2_ax, aniso2_ay).size()) << what << ": the geometry is the resampled ROI's";
	}
}

// The whole-slide twin: the slide's every pixel, as acquired. The whole slide's contour is its box,
// whose corner pixels the EDGE_* statistics take whatever the spacing.
void test_2d_anisotropy_wholeslide_weighted_geometry_as_acquired_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_weighted_wholeslide");
	auto iso = run_aniso2_wholeslide (p, false),
		aniso = run_aniso2_wholeslide (p, true);
	expect_aniso2_as_acquired (aniso, aniso2_as_acquired_reference (aniso2_whole, 1, aniso2_ax, aniso2_ay), "whole slide");
	for (auto f : { Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, Nyxus::Feature2D::EDGE_MEAN_INTENSITY })
		EXPECT_EQ(aniso[(int) f][0], iso[(int) f][0]) << "whole slide feature " << (int) f << " moved with the anisotropy";
}

// At a uniform spacing every pixel is a scaled copy of the unit one, so the normalized central
// moments and Hu's invariants are those of the run without anisotropy. At (0.5, 0.5) the resampling
// drops three pixels in four. What this discriminates: a pass that measures them on the resampled
// cloud moves HU3 by a factor of 2 or more; one that leaves out the pixel's area scales every
// normalized central moment of order p + q by (1/4)^((p + q) / 2).
void test_2d_anisotropy_weighted_geometry_scale_invariant_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_weighted_uniform");
	auto iso = run_aniso2_segmented (p, false),
		half = run_aniso2_segmented (p, true, 0.5, 0.5);
	for (int lab : { 1, 2 })
	{
		for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
			if (name.rfind ("IMOM_NCM_", 0) == 0 || name.rfind ("IMOM_HU", 0) == 0)
			{
				const double want = iso[lab][(int) code][0];
				EXPECT_NEAR(half[lab][(int) code][0], want, 1e-10 * std::abs (want) + 1e-30)
					<< "ROI " << lab << " " << name << " at spacing (0.5, 0.5)";
			}
	}
}

// At the uniform spacing (2, 2) every distance doubles exactly, so the radial distribution, which
// bins pixels by a ratio of distances and wedges them by angle, and the Zernike moments, whose unit
// disk scales with the image, are those of the run without anisotropy, and so are the EDGE_*
// statistics. What this discriminates: a pass that measures them on the cloud resampled by 2,
// which holds every pixel four times and traces a contour around the copies, moves them; so does one
// that scales a pixel's offset from the centroid but not the Zernike radius.
void test_2d_anisotropy_radial_zernike_edge_scale_invariant_mechanics()
{
	Aniso2Pair p ("nyxus_2d_aniso_radial_uniform");
	auto iso = run_aniso2_segmented (p, false),
		twice = run_aniso2_segmented (p, true, 2.0, 2.0);
	std::vector<int> features = aniso2_contour_weighted_features();
	features.erase (std::remove_if (features.begin(), features.end(), [] (int f)
		{
			for (auto g : Imoms2D_feature::featureset)
				if (f == (int) g)
					return true;
			return false;
		}), features.end());
	for (int lab : { 1, 2 })
		expect_aniso2_same_values (twice[lab], iso[lab], features, 1e-12, "ROI " + std::to_string (lab) + " at spacing (2, 2)");
}

// The scaled distances are the unit-grid ones taken on coordinates scaled by the pixel size: at
// (2, 3) they equal, bit for bit, the distances between the same points with x doubled and y
// tripled, the hill-descent minimum and maximum over a contour included, since the descent visits
// the same contour indices in both. What this discriminates: a scale applied to one axis only, to
// the squared distance, or left out of the descent's comparisons.
void test_2d_pixel_scaled_distances_mechanics()
{
	std::vector<Pixel2> K, K23;
	for (int i = 0; i < 40; i++)
	{
		const double t = 2 * std::acos (-1.0) * i / 40.0;
		const int x = (int) std::lround (20 + 11 * std::cos (t)), y = (int) std::lround (15 + 6 * std::sin (t));
		K.push_back (Pixel2 (x, y, 1));
		K23.push_back (Pixel2 (2 * x, 3 * y, 1));
	}
	for (auto [x, y] : { std::pair<int, int> { 20, 15 }, { 14, 12 }, { 27, 19 }, { 3, 2 } })
	{
		const Pixel2 a (x, y, 1), a23 (2 * x, 3 * y, 1);
		EXPECT_EQ(a.sqdist (K[7], 2.0, 3.0), a23.sqdist (K23[7])) << x << "," << y;
		EXPECT_EQ(a.min_sqdist (K, 2.0, 3.0), a23.min_sqdist (K23)) << x << "," << y;
		EXPECT_EQ(a.max_sqdist (K, 2.0, 3.0), a23.max_sqdist (K23)) << x << "," << y;
		EXPECT_EQ(a.dist_to_segment (K[3], K[21], 2.0, 3.0), a23.dist_to_segment (K23[3], K23[21])) << x << "," << y;
	}
	std::vector<Pixel2> cloud, cloud23;
	for (int y = 10; y <= 20; y++)
		for (int x = 12; x <= 28; x++)
		{
			cloud.push_back (Pixel2 (x, y, 1));
			cloud23.push_back (Pixel2 (2 * x, 3 * y, 1));
		}
	EXPECT_EQ(Pixel2::find_center (cloud, K, 2.0, 3.0), Pixel2::find_center (cloud23, K23));
}