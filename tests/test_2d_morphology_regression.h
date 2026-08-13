#pragma once

#include "test_2d_morphology_common.h"
#include "test_2d_remaining_common.h"

#include "test_ref_vals.h"

static const ref_vals_map<double> morphology_2d_regression_caliper_chords_ref_vals{
	{"EROSIONS_2_VANISH_COMPLEMENT", 0.0},
	{"MIN_FERET_ANGLE", 40.0},
	// FIX (caliper float-precision): re-pinned to the float-precision hull-rotation values (see rotation.cpp
	// rotate_around_center_fp). The old integer-Pixel2 rotation truncated every rotated vertex inward,
	// so these 8x8-fixture goldens shifted when the truncation was removed. MAX_FERET_ANGLE moved 0->110
	// because the per-angle Feret ties differently once the diameters are no longer integer-quantized
	// (the Feret angle is a regression-only Nyxus-frame convention, not oracle-vetted). MODE values are
	// unchanged. The diameters themselves are vetted vs imea (<=10%) on the ellipse oracle below.
	{"MAX_FERET_ANGLE", 110.0},
	{"MAXCHORDS_MAX", 6.0},
	{"MAXCHORDS_MIN", 3.0},
	{"MAXCHORDS_MEDIAN", 4.0},
	{"MAXCHORDS_MEAN", 4.5500000000000007},
	{"MAXCHORDS_MODE", 4.0},
	{"MAXCHORDS_STDDEV", 0.94451324138833304},
	{"ALLCHORDS_MAX", 6.0},
	// FIXED (chords.cpp histo built from MC): all-chords median/mode now computed over ALL chords, not max-chords
	{"ALLCHORDS_MEDIAN", 3.0},
	{"ALLCHORDS_MEAN", 2.9134615384615379},
	{"ALLCHORDS_MODE", 3.0},
	{"ALLCHORDS_STDDEV", 1.3446086298393252},
};

static const ref_vals_map<double> morphology_2d_regression_polygonality_chords_ref_vals{
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
	// FIX (caliper float-precision): HEXAGONALITY_STDDEV re-pinned (depends on STAT_FERET_DIAM_MIN/MAX, which
	// shifted with the float-precision hull rotation); the AVE scores stayed within tolerance.
	{"HEXAGONALITY_STDDEV", 0.188079},
	// FIXED (chords.cpp idxmax used iteMin): max-angle now indexes the longest chord (angle 0), not the min
	{"MAXCHORDS_MAX_ANG", 0.0},
	{"MAXCHORDS_MIN_ANG", 0.94247779607693793},
	{"ALLCHORDS_MAX_ANG", 0.0},
	{"ALLCHORDS_MIN_ANG", 0.15707963267948966},
};

static void assert_unvetted_no_direct_oracle_remaining2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_regression_polygonality_chords_ref_vals.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_regression_polygonality_chords_ref_vals.at(feature_name), frac_tolerance));
}

static void assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_regression_caliper_chords_ref_vals.count(feature_name) > 0) << feature_name;
	const double ref_val = morphology_2d_regression_caliper_chords_ref_vals.at(feature_name);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], ref_val, frac_tolerance));
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
	ASSERT_TRUE(morphology_2d_regression_polygonality_chords_ref_vals.count(feature_name) > 0);
	const double actual = roiData.at(1).fvals[static_cast<int>(feature)][0];
	ASSERT_GT(actual, 0.0);
	ASSERT_TRUE(agrees_gt(actual, morphology_2d_regression_polygonality_chords_ref_vals.at(feature_name), frac_tolerance));
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


// Pinned Nyxus output for the shape-2D features with no external reference. Establishes no vetting
// (SPEC 1), which is why nothing outside this file compares against it any more.
static const ref_vals_map<double> morphology_2d_regression_ref_vals
{
	{"AREA_PIXELS_COUNT", 26.0},
	{"AREA_UM2", 104.0},
	{"CENTROID_X", 2.61538461538462},
	{"CENTROID_Y", 2.84615384615385},
	{"WEIGHTED_CENTROID_X", 2.84160305343511},
	{"WEIGHTED_CENTROID_Y", 3.43893129770992},
	{"COMPACTNESS", 0.027517678630878},
	{"BBOX_XMIN", 0.0},
	{"BBOX_YMIN", 0.0},
	{"BBOX_WIDTH", 6.0},
	{"BBOX_HEIGHT", 7.0},
	{"DIAMETER_EQUAL_AREA", 5.75362739175159},
	{"EXTENT", 0.619047619047619},
	{"ASPECT_RATIO", 0.857142857142857},
	{"MAJOR_AXIS_LENGTH", 6.96881616898619},
	{"MINOR_AXIS_LENGTH", 5.48870991295738},
	{"ELONGATION", 0.787610087547462},
	{"ECCENTRICITY", 0.616173960820708},
	{"ORIENTATION", 70.4173944984207},
	{"ROUNDNESS", 0.681656295209303},
	{"PERIMETER", 26.9349412836191},
	{"CIRCULARITY", 0.671081973229055},
	{"EULER_NUMBER", 0.0},
	{"DIAMETER_MIN_ENCLOSING_CIRCLE", 6.32475519180298},
	{"ROI_RADIUS_MEAN", 1.07692307692308},
	{"ROI_RADIUS_MAX", 4.0},
	{"ROI_RADIUS_MEDIAN", 1.0}
};

static const ref_vals_map<double> morphology_2d_regression_diameters_ref_vals{
	{"DIAMETER_CIRCUMSCRIBING_CIRCLE", 12.3317073399088},
	{"DIAMETER_INSCRIBING_CIRCLE", 0.828486893405308},
};

static void assert_morphology_feature_regression(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(morphology_2d_regression_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_regression_ref_vals.at(feature_name), frac_tolerance));
}

static void assert_morphology_diameter_regression(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(morphology_2d_regression_diameters_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_regression_diameters_ref_vals.at(feature_name), frac_tolerance));
}

void test_2d_morphology_basic_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::AREA_PIXELS_COUNT, "AREA_PIXELS_COUNT");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::AREA_UM2, "AREA_UM2");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::CENTROID_X, "CENTROID_X");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::CENTROID_Y, "CENTROID_Y");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_X, "WEIGHTED_CENTROID_X");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_Y, "WEIGHTED_CENTROID_Y");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::COMPACTNESS, "COMPACTNESS");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::BBOX_XMIN, "BBOX_XMIN");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::BBOX_YMIN, "BBOX_YMIN");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::BBOX_WIDTH, "BBOX_WIDTH");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::BBOX_HEIGHT, "BBOX_HEIGHT");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::DIAMETER_EQUAL_AREA, "DIAMETER_EQUAL_AREA");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::EXTENT, "EXTENT");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::ASPECT_RATIO, "ASPECT_RATIO");
}

void test_2d_morphology_ellipse_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::MAJOR_AXIS_LENGTH, "MAJOR_AXIS_LENGTH");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::MINOR_AXIS_LENGTH, "MINOR_AXIS_LENGTH");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::ELONGATION, "ELONGATION");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::ECCENTRICITY, "ECCENTRICITY");
	// ORIENTATION is vetted vs scikit-image in test_2d_morphology_skimage.h; ROUNDNESS is vetted by
	// documented-formula conformance in test_2d_morphology_analytic.h. The snapshot below is drift-only.
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::ROUNDNESS, "ROUNDNESS");
}

void test_2d_morphology_contour_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::PERIMETER, "PERIMETER");
}

void test_2d_morphology_misc_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::EULER_NUMBER, "EULER_NUMBER");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE, "DIAMETER_MIN_ENCLOSING_CIRCLE");
}

// Documented-formula conformance for CIRCULARITY/ROUNDNESS is a correctness claim, so it lives in the
// oracle file test_2d_morphology_analytic.h, not here -- this file is a snapshot drift guard and claims
// nothing (SPEC 2).

void test_2d_morphology_radius_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::ROI_RADIUS_MEAN, "ROI_RADIUS_MEAN");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::ROI_RADIUS_MAX, "ROI_RADIUS_MAX");
	assert_morphology_feature_regression(fvals, Nyxus::Feature2D::ROI_RADIUS_MEDIAN, "ROI_RADIUS_MEDIAN");
}

// DIAMETER_EQUAL_PERIMETER is now vetted against imea's perimeter_equal_diameter (the same ISO
// perimeter/pi transform) in test_2d_morphology_imea.h.

void test_2d_morphology_fractal_circle_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// Fractal dimensions are validated in test_2d_morphology_fractal_dimension_blob512_fraclac.
	// Here we keep the inscribing/circumscribing circle diameters.
	assert_morphology_diameter_regression(fvals, Nyxus::Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE, "DIAMETER_CIRCUMSCRIBING_CIRCLE");
	assert_morphology_diameter_regression(fvals, Nyxus::Feature2D::DIAMETER_INSCRIBING_CIRCLE, "DIAMETER_INSCRIBING_CIRCLE");
}

// GEODETIC_LENGTH + THICKNESS are now vetted against imea's geodeticlength_and_thickness (the same
// DIN ISO 9276-6 rectangle model) in test_2d_morphology_imea.h, after the geo_len_thickness.cpp
// perimeter-truncation fix. EROSIONS_2_VANISH is vetted vs scikit-image in test_2d_morphology_skimage.h.

// ---------------------------------------------------------------------------------------------------
// Migrated from test_2d_remaining_features.h (Wave 6): erosion-complement, caliper (feret/martin/
// nassenstein), chord stats and chord angles, and polygonality/hexagonality. All map to
// test_2d_morphology_regression.h per the registry target_test. Shared fixture/oracle-data lives in
// test_2d_remaining_common.h.
// ---------------------------------------------------------------------------------------------------

void test_2d_morphology_erosion_complement_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::EROSIONS_2_VANISH_COMPLEMENT, "EROSIONS_2_VANISH_COMPLEMENT");
}

// The 18 STAT_* caliper statistics and ALLCHORDS_MIN moved to test_2d_morphology_imea.h: their
// registry rows are status=vetted, oracle=imea and target that file, and they were only reachable
// from here through a lookup that spanned an imea table and this one.
void test_2d_morphology_caliper_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MIN_FERET_ANGLE, "MIN_FERET_ANGLE");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAX_FERET_ANGLE, "MAX_FERET_ANGLE");
}

void test_2d_morphology_chord_stat_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MAX, "MAXCHORDS_MAX");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MIN, "MAXCHORDS_MIN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MEDIAN, "MAXCHORDS_MEDIAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MEAN, "MAXCHORDS_MEAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MODE, "MAXCHORDS_MODE");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_STDDEV, "MAXCHORDS_STDDEV");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MAX, "ALLCHORDS_MAX");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MEDIAN, "ALLCHORDS_MEDIAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MEAN, "ALLCHORDS_MEAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MODE, "ALLCHORDS_MODE");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_STDDEV, "ALLCHORDS_STDDEV");
}

void test_2d_morphology_chord_angle_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MAX_ANG, "MAXCHORDS_MAX_ANG");
	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MIN_ANG, "MAXCHORDS_MIN_ANG");
	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MAX_ANG, "ALLCHORDS_MAX_ANG");
	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MIN_ANG, "ALLCHORDS_MIN_ANG");
}

void test_2d_morphology_polygonality_hexagonality_regression()
{
	std::unordered_map<int, LR> roiData;
	calculate_remaining2d_polygonality_feature_values(roiData);

	ASSERT_EQ(roiData.at(1).fvals[static_cast<int>(Nyxus::Feature2D::NUM_NEIGHBORS)][0], 4.0);
	assert_unvetted_no_direct_oracle_remaining2d_polygonality_score(roiData, Nyxus::Feature2D::POLYGONALITY_AVE, "POLYGONALITY_AVE");
	assert_unvetted_no_direct_oracle_remaining2d_polygonality_score(roiData, Nyxus::Feature2D::HEXAGONALITY_AVE, "HEXAGONALITY_AVE");
	assert_unvetted_no_direct_oracle_remaining2d_polygonality_feature(roiData, Nyxus::Feature2D::HEXAGONALITY_STDDEV, "HEXAGONALITY_STDDEV");

	assert_remaining2d_polygonality_no_value_for_sparse_neighbors(roiData, Nyxus::Feature2D::POLYGONALITY_AVE);
	assert_remaining2d_polygonality_no_value_for_sparse_neighbors(roiData, Nyxus::Feature2D::HEXAGONALITY_AVE);
	assert_remaining2d_polygonality_no_value_for_sparse_neighbors(roiData, Nyxus::Feature2D::HEXAGONALITY_STDDEV);
}
