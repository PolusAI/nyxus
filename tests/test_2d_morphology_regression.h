#pragma once

#include "test_2d_morphology_common.h"
#include "test_2d_remaining_common.h"

#include "test_ref_vals.h"

// Pinned Nyxus output for the shape-2D features with no external reference. Establishes no vetting
// (SPEC 1), which is why nothing outside this file compares against it any more.
static ref_vals_map<double> morphology_2d_regression_ref_vals
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

static ref_vals_map<double> morphology_2d_regression_diameters_ref_vals{
	{"DIAMETER_CIRCUMSCRIBING_CIRCLE", 12.3317073399088},
	{"DIAMETER_INSCRIBING_CIRCLE", 0.828486893405308},
};

static void assert_morphology_feature_regression(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(morphology_2d_regression_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_regression_ref_vals[feature_name], frac_tolerance));
}

static void assert_morphology_diameter_regression(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(morphology_2d_regression_diameters_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_regression_diameters_ref_vals[feature_name], frac_tolerance));
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

void test_2d_morphology_caliper_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MIN_FERET_ANGLE, "MIN_FERET_ANGLE");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAX_FERET_ANGLE, "MAX_FERET_ANGLE");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MIN, "STAT_FERET_DIAM_MIN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MAX, "STAT_FERET_DIAM_MAX");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEAN, "STAT_FERET_DIAM_MEAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEDIAN, "STAT_FERET_DIAM_MEDIAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_STDDEV, "STAT_FERET_DIAM_STDDEV");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MODE, "STAT_FERET_DIAM_MODE");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MIN, "STAT_MARTIN_DIAM_MIN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MAX, "STAT_MARTIN_DIAM_MAX");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEAN, "STAT_MARTIN_DIAM_MEAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEDIAN, "STAT_MARTIN_DIAM_MEDIAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_STDDEV, "STAT_MARTIN_DIAM_STDDEV");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MODE, "STAT_MARTIN_DIAM_MODE");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MIN, "STAT_NASSENSTEIN_DIAM_MIN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MAX, "STAT_NASSENSTEIN_DIAM_MAX");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEAN, "STAT_NASSENSTEIN_DIAM_MEAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEDIAN, "STAT_NASSENSTEIN_DIAM_MEDIAN");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_STDDEV, "STAT_NASSENSTEIN_DIAM_STDDEV");
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MODE, "STAT_NASSENSTEIN_DIAM_MODE");
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
	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MIN, "ALLCHORDS_MIN");
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
