#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/caliper.h"
#include "../src/nyx/features/chords.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/convex_hull.h"
#include "../src/nyx/features/erosion.h"
#include "../src/nyx/features/hexagonality_polygonality.h"
#include "../src/nyx/features/neighbors.h"
#include "../src/nyx/features/radial_distribution.h"
#include "../src/nyx/features/zernike.h"
#include "test_data.h"
#include "test_main_nyxus.h"

static std::unordered_map<std::string, double> oracle_3p_remaining2d_feature_golden_values{
	{"EROSIONS_2_VANISH_COMPLEMENT", 0.0},
	{"MIN_FERET_ANGLE", 40.0},
	{"MAX_FERET_ANGLE", 0.0},
	{"STAT_FERET_DIAM_MIN", 4.0},
	{"STAT_FERET_DIAM_MAX", 5.0},
	{"STAT_FERET_DIAM_MEAN", 4.9473684210526319},
	{"STAT_FERET_DIAM_MEDIAN", 5.0},
	{"STAT_FERET_DIAM_STDDEV", 0.22329687826943606},
	{"STAT_FERET_DIAM_MODE", 5.0},
	// FIXED (caliper reimpl): Martin is now the area-bisecting chord and Nassenstein the bottom-tangent
	// vertical chord (one diameter per angle), not the old min+max of a Y-grid of horizontal chords.
	// The old goldens pinned the bug (Martin min 0.8, Nassenstein min/mode 0.0 — impossible for a solid
	// shape). These are the corrected values on the 8x8 fixture; the diameters are vetted vs imea on a
	// clean ellipse in TEST_SHAPE2D_CALIPER_MARTIN_NASSENSTEIN_IMEA_ELLIPSE_ORACLE.
	{"STAT_MARTIN_DIAM_MIN", 4.0},
	{"STAT_MARTIN_DIAM_MAX", 5.0},
	{"STAT_MARTIN_DIAM_MEAN", 4.64819},
	{"STAT_MARTIN_DIAM_MEDIAN", 4.65875},
	{"STAT_MARTIN_DIAM_STDDEV", 0.315754},
	{"STAT_MARTIN_DIAM_MODE", 4.0},
	{"STAT_NASSENSTEIN_DIAM_MIN", 1.0},
	{"STAT_NASSENSTEIN_DIAM_MAX", 6.0},
	{"STAT_NASSENSTEIN_DIAM_MEAN", 4.60324},
	{"STAT_NASSENSTEIN_DIAM_MEDIAN", 5.0},
	{"STAT_NASSENSTEIN_DIAM_STDDEV", 1.41327},
	{"STAT_NASSENSTEIN_DIAM_MODE", 4.0},
	{"MAXCHORDS_MAX", 6.0},
	{"MAXCHORDS_MIN", 3.0},
	{"MAXCHORDS_MEDIAN", 4.0},
	{"MAXCHORDS_MEAN", 4.5500000000000007},
	{"MAXCHORDS_MODE", 4.0},
	{"MAXCHORDS_STDDEV", 0.94451324138833304},
	{"ALLCHORDS_MAX", 6.0},
	{"ALLCHORDS_MIN", 1.0},
	// FIXED (chords.cpp histo built from MC): all-chords median/mode now computed over ALL chords, not max-chords
	{"ALLCHORDS_MEDIAN", 3.0},
	{"ALLCHORDS_MEAN", 2.9134615384615379},
	{"ALLCHORDS_MODE", 3.0},
	{"ALLCHORDS_STDDEV", 1.3446086298393252},
};

static std::unordered_map<std::string, double> unvetted_nyxus_regression_remaining2d_feature_golden_values{
	// POLYGONALITY_AVE depends only on neighbors/area/perimeter, so the Pick's-theorem
	// convex-hull-area fix (convex_hull_nontriv.cpp) leaves it unchanged. HEXAGONALITY_AVE and
	// HEXAGONALITY_STDDEV read CONVEX_HULL_AREA (via area_hull in hexagonality_polygonality.cpp),
	// so the fix (bare shoelace 4 -> Pick's pixel-count 9 for the 3x3 label-1 ROI) shifted them:
	// HEXAGONALITY_AVE 6.4263 -> 6.8823, HEXAGONALITY_STDDEV 0.3144 -> 0.1850. This shift is
	// correct: the Polus reference computes area_hull = area/solidity = skimage convex_area (a
	// pixel count), which is exactly what Pick's theorem produces. These are Polus-specific scores
	// with no external oracle, so the goldens are self-referential regression snapshots; the
	// assertions below now value-compare against them (agrees_gt) so any future drift is caught.
	{"POLYGONALITY_AVE", 2.0833333333333357},
	{"HEXAGONALITY_AVE", 6.8823312738837217},
	{"HEXAGONALITY_STDDEV", 0.18495557498763179},
	// FIXED (chords.cpp idxmax used iteMin): max-angle now indexes the longest chord (angle 0), not the min
	{"MAXCHORDS_MAX_ANG", 0.0},
	{"MAXCHORDS_MIN_ANG", 0.94247779607693793},
	{"ALLCHORDS_MAX_ANG", 0.0},
	{"ALLCHORDS_MIN_ANG", 0.15707963267948966},
};

static std::unordered_map<std::string, std::vector<double>> unvetted_nyxus_regression_remaining2d_vector_feature_golden_values{
	{"FRAC_AT_D", {
		0.038461538460059175, 0.0, 0.11538461538017751, 0.1538461538402367,
		0.3076923076804734, 0.0, 0.11538461538017751, 0.26923076922041422,
	}},
	{"MEAN_FRAC", {
		50.999999948999999, 0.0, 53.333333315555556, 50.749999987312499,
		47.374999994078124, 0.0, 33.666666655444445, 21.999999996857142,
	}},
	{"RADIAL_CV", {
		2.6457513106495707, 0.0, 1.298797520721114, 1.024429214739045,
		0.64750329537582818, 0.0, 1.3575192606324717, 1.3284260624865412,
	}},
};

static std::unordered_map<std::string, std::vector<double>> oracle_3p_remaining2d_vector_feature_golden_values{
	{"ZERNIKE2D", {
		0.02049738595695693, 0.035831084484416686, 0.073953766599300461,
		0.035435050265597692, 0.092323797445497555, 0.011030627605166297,
		0.13199834370886107, 0.13453286019693309, 0.00788523106321295,
		0.082424064819857396, 0.049062071772591059, 0.0040585552756590825,
		0.14488178557089382, 0.23625456011991602, 0.038032570269059741,
		0.0011694758904577424, 0.016507094944884948, 0.10703041567067684,
		0.021302528534918392, 0.00061791897183974015, 0.10313303720229962,
		0.23275354391334316, 0.08692094259111556, 0.0063362223871874139,
		0.00016460740533666494, 0.085700825034398798, 0.15183975656312645,
		0.052012830525298454, 0.0045112452293896111, 0.00015124210515210458,
	}},
};

// ---------------------------------------------------------------------------------------------------
// Martin / Nassenstein caliper vetting vs imea (external oracle).
//
// The 8x8 shape2d fixture above is too small/aliased to serve as a tight caliper oracle, so the
// corrected Martin (area-bisecting chord) and Nassenstein (bottom-tangent vertical chord) diameters
// are vetted on a clean, larger convex fixture: a filled ellipse a=20, b=10 (same rasterization as
// morph_oracle/caliper_proto.py). imea (imea.measure_2d.statistical_length, dalpha=10) is the
// reference. Nyxus rotates the convex hull and measures analytically while imea rotates the filled
// raster, so the two agree only up to a ~1-2px hull-vs-raster convention gap (same gap already
// accepted for Feret) — hence a 15% relative tolerance on the robust stats. The point that this pins
// is that the diameters are now the *correct* quantities (min > 0), not the old min+max-chord bug
// that produced physically-impossible 0-length Nassenstein diameters.
static std::unordered_map<std::string, double> imea_ellipse_caliper_oracle{
	{"STAT_MARTIN_DIAM_MIN", 19.0},
	{"STAT_MARTIN_DIAM_MAX", 41.0},
	{"STAT_MARTIN_DIAM_MEAN", 27.61},
	{"STAT_MARTIN_DIAM_MEDIAN", 25.5},
	{"STAT_NASSENSTEIN_DIAM_MIN", 16.0},
	{"STAT_NASSENSTEIN_DIAM_MAX", 41.0},
	{"STAT_NASSENSTEIN_DIAM_MEAN", 25.17},
	{"STAT_NASSENSTEIN_DIAM_MEDIAN", 21.5},
};

static Fsettings make_remaining2d_settings()
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

static void calculate_remaining2d_shape_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_remaining2d_settings();

	LR roidata(1201);
	load_masked_test_roi_data(
		roidata,
		shape2d_morphology_intensity,
		shape2d_morphology_mask,
		sizeof(shape2d_morphology_mask) / sizeof(NyxusPixel));
	roidata.initialize_fvals();

	BasicMorphologyFeatures basic;
	basic.calculate(roidata, s);
	basic.save_value(roidata.fvals);

	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);

	ConvexHullFeature hull;
	hull.calculate(roidata, s);
	hull.save_value(roidata.fvals);

	CaliperFeretFeature feret;
	feret.calculate(roidata, s);
	feret.save_value(roidata.fvals);

	CaliperMartinFeature martin;
	martin.calculate(roidata, s);
	martin.save_value(roidata.fvals);

	CaliperNassensteinFeature nassenstein;
	nassenstein.calculate(roidata, s);
	nassenstein.save_value(roidata.fvals);

	ChordsFeature chords;
	chords.calculate(roidata, s);
	chords.save_value(roidata.fvals);

	ErosionPixelsFeature erosion;
	erosion.calculate(roidata, s);
	erosion.save_value(roidata.fvals);

	RadialDistributionFeature radial;
	radial.calculate(roidata, s);
	radial.save_value(roidata.fvals);

	ZernikeFeature zernike;
	zernike.calculate(roidata, s);
	zernike.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

static void calculate_remaining2d_polygonality_feature_values(std::unordered_map<int, LR>& roiData)
{
	Fsettings s = make_remaining2d_settings();
	s[static_cast<int>(NyxSetting::PIXELSIZEUM)].rval = 1.0;
	std::unordered_set<int> uniqueLabels;

	for (const auto& px : neighborhood2d_scene_labels)
	{
		int label = static_cast<int>(px.intensity);
		uniqueLabels.insert(label);

		auto [it, inserted] = roiData.try_emplace(label, label);
		LR& roi = it->second;

		if (inserted)
			init_label_record_3(roi, static_cast<int>(px.x), static_cast<int>(px.y), 1);
		else
			update_label_record_3(roi, static_cast<int>(px.x), static_cast<int>(px.y), 1);

		roi.raw_pixels.push_back(Pixel2(static_cast<size_t>(px.x), static_cast<size_t>(px.y), static_cast<PixIntens>(1)));
	}

	BasicMorphologyFeatures basic;
	ContourFeature contour;
	ConvexHullFeature hull;
	CaliperFeretFeature feret;
	for (auto& item : roiData)
	{
		LR& roi = item.second;
		roi.make_nonanisotropic_aabb();
		roi.aux_image_matrix = ImageMatrix(roi.raw_pixels);
		roi.initialize_fvals();

		basic.calculate(roi, s);
		basic.save_value(roi.fvals);

		contour.calculate(roi, s);
		contour.save_value(roi.fvals);

		hull.calculate(roi, s);
		hull.save_value(roi.fvals);

		feret.calculate(roi, s);
		feret.save_value(roi.fvals);
	}

	NeighborsFeature::manual_reduce(roiData, s, uniqueLabels);

	HexagonalityPolygonalityFeature hexpoly;
	for (auto& item : roiData)
	{
		hexpoly.calculate(item.second, s);
		hexpoly.save_value(item.second.fvals);
	}
}

// Build a filled ellipse (a=20, b=10) ROI and compute its caliper features. Mirrors the
// rasterization in morph_oracle/caliper_proto.py so the imea reference values above line up.
static void calculate_ellipse_caliper_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_remaining2d_settings();

	LR roi(1);
	const double a = 20.0, b = 10.0, cx = 26.0, cy = 16.0;	// pad=6, matches the prototype fixture
	bool first = true;
	for (int y = 0; y <= 32; y++)
		for (int x = 0; x <= 52; x++)
		{
			double dx = (x - cx) / a, dy = (y - cy) / b;
			if (dx * dx + dy * dy <= 1.0)
			{
				if (first)
				{
					init_label_record_3(roi, x, y, 1);
					first = false;
				}
				else
					update_label_record_3(roi, x, y, 1);
				roi.raw_pixels.push_back(Pixel2(static_cast<size_t>(x), static_cast<size_t>(y), static_cast<PixIntens>(1)));
			}
		}
	roi.make_nonanisotropic_aabb();
	roi.aux_image_matrix = ImageMatrix(roi.raw_pixels);
	roi.initialize_fvals();

	ContourFeature contour;
	contour.calculate(roi, s);
	contour.save_value(roi.fvals);

	ConvexHullFeature hull;
	hull.calculate(roi, s);
	hull.save_value(roi.fvals);

	CaliperMartinFeature martin;
	martin.calculate(roi, s);
	martin.save_value(roi.fvals);

	CaliperNassensteinFeature nassenstein;
	nassenstein.calculate(roi, s);
	nassenstein.save_value(roi.fvals);

	fvals = roi.fvals;
}

// Assert a caliper stat agrees with imea within a relative tolerance (hull-vs-raster convention gap).
static void assert_caliper_close_to_imea(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double reltol = 0.15)
{
	SCOPED_TRACE(std::string("CALIPER_VS_IMEA__") + feature_name);
	ASSERT_TRUE(imea_ellipse_caliper_oracle.count(feature_name) > 0);
	const double imea_ref = imea_ellipse_caliper_oracle[feature_name];
	const double actual = fvals[static_cast<int>(feature)][0];
	const double denom = std::max(std::abs(imea_ref), 1e-9);
	ASSERT_LE(std::abs(actual - imea_ref) / denom, reltol)
		<< feature_name << " nyxus=" << actual << " imea=" << imea_ref;
}

static void assert_unvetted_no_direct_oracle_remaining2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	ASSERT_TRUE(unvetted_nyxus_regression_remaining2d_feature_golden_values.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], unvetted_nyxus_regression_remaining2d_feature_golden_values[feature_name], frac_tolerance));
}

static void assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__") + feature_name);
	ASSERT_TRUE(oracle_3p_remaining2d_feature_golden_values.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], oracle_3p_remaining2d_feature_golden_values[feature_name], frac_tolerance));
}

static void assert_unvetted_no_direct_oracle_remaining2d_polygonality_feature(
	const std::unordered_map<int, LR>& roiData,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	// Value-compare against the regression golden so any drift (e.g. a change in the shared
	// CONVEX_HULL_AREA that feeds area_hull) is caught, instead of the old bounds-only check that
	// left the golden values on this map never actually compared.
	ASSERT_TRUE(unvetted_nyxus_regression_remaining2d_feature_golden_values.count(feature_name) > 0);
	const double actual = roiData.at(1).fvals[static_cast<int>(feature)][0];
	ASSERT_GT(actual, 0.0);
	ASSERT_TRUE(agrees_gt(actual, unvetted_nyxus_regression_remaining2d_feature_golden_values[feature_name], frac_tolerance));
}

static void assert_unvetted_no_direct_oracle_remaining2d_polygonality_score(
	const std::unordered_map<int, LR>& roiData,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	assert_unvetted_no_direct_oracle_remaining2d_polygonality_feature(roiData, feature, feature_name, frac_tolerance);
	// The polygonality/hexagonality scores are bounded above by 10 by construction - keep this as
	// a cheap semantic invariant on top of the value comparison.
	ASSERT_LE(roiData.at(1).fvals[static_cast<int>(feature)][0], 10.0);
}

static void assert_remaining2d_polygonality_no_value_for_sparse_neighbors(
	const std::unordered_map<int, LR>& roiData,
	Nyxus::Feature2D feature)
{
	for (int label : {2, 3, 4, 5})
		ASSERT_EQ(roiData.at(label).fvals[static_cast<int>(feature)][0], -1.0);
}

static void assert_unvetted_no_direct_oracle_remaining2d_vector_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double abs_tolerance = 1e-9)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	ASSERT_TRUE(unvetted_nyxus_regression_remaining2d_vector_feature_golden_values.count(feature_name) > 0);
	const auto& actual = fvals[static_cast<int>(feature)];
	const auto& golden_values = unvetted_nyxus_regression_remaining2d_vector_feature_golden_values[feature_name];
	ASSERT_EQ(actual.size(), golden_values.size());
	for (size_t i = 0; i < golden_values.size(); ++i)
		ASSERT_NEAR(actual[i], golden_values[i], abs_tolerance) << feature_name << "[" << i << "]";
}

static void assert_verifiable_with_3p_builtin_oracle_remaining2d_vector_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double abs_tolerance = 1e-9)
{
	SCOPED_TRACE(std::string("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__") + feature_name);
	ASSERT_TRUE(oracle_3p_remaining2d_vector_feature_golden_values.count(feature_name) > 0);
	const auto& actual = fvals[static_cast<int>(feature)];
	const auto& golden_values = oracle_3p_remaining2d_vector_feature_golden_values[feature_name];
	ASSERT_EQ(actual.size(), golden_values.size());
	for (size_t i = 0; i < golden_values.size(); ++i)
		ASSERT_NEAR(actual[i], golden_values[i], abs_tolerance) << feature_name << "[" << i << "]";
}

// The seven test functions that used to live here have been distributed to their taxonomy homes
// (registry target_test): erosion/caliper/chords/chord-angle/polygonality -> test_morphology_regression.h,
// radial distribution -> test_intensity_histogram_regression.h, zernike2d -> test_zernike_regression.h.
// This header now carries only the shared fixture/oracle-data those files include.
