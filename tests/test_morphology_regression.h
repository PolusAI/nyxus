#pragma once

#include "test_morphology_common.h"
#include "test_remaining2d_common.h"

void test_shape2d_basic_morphology_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::AREA_PIXELS_COUNT, "AREA_PIXELS_COUNT");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::AREA_UM2, "AREA_UM2");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::CENTROID_X, "CENTROID_X");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::CENTROID_Y, "CENTROID_Y");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_X, "WEIGHTED_CENTROID_X");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_Y, "WEIGHTED_CENTROID_Y");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::MASS_DISPLACEMENT, "MASS_DISPLACEMENT");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::COMPACTNESS, "COMPACTNESS");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_XMIN, "BBOX_XMIN");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_YMIN, "BBOX_YMIN");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_WIDTH, "BBOX_WIDTH");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_HEIGHT, "BBOX_HEIGHT");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_EQUAL_AREA, "DIAMETER_EQUAL_AREA");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTENT, "EXTENT");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ASPECT_RATIO, "ASPECT_RATIO");
}

void test_shape2d_ellipse_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::MAJOR_AXIS_LENGTH, "MAJOR_AXIS_LENGTH");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::MINOR_AXIS_LENGTH, "MINOR_AXIS_LENGTH");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ELONGATION, "ELONGATION");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ECCENTRICITY, "ECCENTRICITY");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ORIENTATION, "ORIENTATION");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ROUNDNESS, "ROUNDNESS");
}

void test_shape2d_contour_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::PERIMETER, "PERIMETER");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_MEAN_INTENSITY, "EDGE_MEAN_INTENSITY");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_STDDEV_INTENSITY, "EDGE_STDDEV_INTENSITY");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_MAX_INTENSITY, "EDGE_MAX_INTENSITY");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_MIN_INTENSITY, "EDGE_MIN_INTENSITY");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, "EDGE_INTEGRATED_INTENSITY");
}

void test_shape2d_misc_shape_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EULER_NUMBER, "EULER_NUMBER");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE, "DIAMETER_MIN_ENCLOSING_CIRCLE");
}

void test_shape2d_unvetted_no_direct_oracle_radius_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ROI_RADIUS_MEAN, "ROI_RADIUS_MEAN");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ROI_RADIUS_MAX, "ROI_RADIUS_MAX");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ROI_RADIUS_MEDIAN, "ROI_RADIUS_MEDIAN");
}

void test_shape2d_verifiable_with_3p_builtin_oracle_contour_diameter_equal_perimeter()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_EQUAL_PERIMETER, "DIAMETER_EQUAL_PERIMETER");
}

void test_shape2d_verifiable_with_3p_builtin_oracle_fractal_circle_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// Fractal dimensions are validated in test_shape2d_fractal_dimension_blob512_oracle.
	// Here we keep the inscribing/circumscribing circle diameters.
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE, "DIAMETER_CIRCUMSCRIBING_CIRCLE");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_INSCRIBING_CIRCLE, "DIAMETER_INSCRIBING_CIRCLE");
}

void test_shape2d_verifiable_with_3p_builtin_oracle_geodetic_thickness_erosion_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::GEODETIC_LENGTH, "GEODETIC_LENGTH");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::THICKNESS, "THICKNESS");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EROSIONS_2_VANISH, "EROSIONS_2_VANISH");
}

// ---------------------------------------------------------------------------------------------------
// Migrated from test_2d_remaining_features.h (Wave 6): erosion-complement, caliper (feret/martin/
// nassenstein), chord stats and chord angles, and polygonality/hexagonality. All map to
// test_morphology_regression.h per the registry target_test. Shared fixture/oracle-data lives in
// test_remaining2d_common.h.
// ---------------------------------------------------------------------------------------------------

void test_remaining2d_verifiable_with_3p_builtin_oracle_erosion_complement_feature()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::EROSIONS_2_VANISH_COMPLEMENT, "EROSIONS_2_VANISH_COMPLEMENT");
}

void test_remaining2d_verifiable_with_3p_builtin_oracle_caliper_features()
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

// Vets the reimplemented Martin (area-bisecting chord) and Nassenstein (bottom-tangent vertical
// chord) diameters against imea on a clean filled ellipse (a=20, b=10). See the oracle block in
// test_remaining2d_common.h. Robust stats (min/max/mean/median) agree with imea within the
// hull-vs-raster convention tolerance; the >0 lower bound pins that the old min+max-chord bug
// (0-length Nassenstein diameters) is gone.
void test_shape2d_caliper_martin_nassenstein_imea_ellipse_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_ellipse_caliper_values(fvals);

	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MIN, "STAT_MARTIN_DIAM_MIN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MAX, "STAT_MARTIN_DIAM_MAX");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEAN, "STAT_MARTIN_DIAM_MEAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEDIAN, "STAT_MARTIN_DIAM_MEDIAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MIN, "STAT_NASSENSTEIN_DIAM_MIN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MAX, "STAT_NASSENSTEIN_DIAM_MAX");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEAN, "STAT_NASSENSTEIN_DIAM_MEAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEDIAN, "STAT_NASSENSTEIN_DIAM_MEDIAN");

	// Bug-gone invariant: a solid shape cannot have a 0-length diameter (the old code produced 0).
	ASSERT_GT(fvals[static_cast<int>(Nyxus::Feature2D::STAT_MARTIN_DIAM_MIN)][0], 2.0);
	ASSERT_GT(fvals[static_cast<int>(Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MIN)][0], 2.0);
}

// Vets the Feret diameter distribution against imea on the same filled ellipse. Feret is a correct
// rotating-calipers implementation; robust stats (min/max/mean/median) agree with imea within the
// hull-vs-raster convention tolerance. (MIN/MAX_FERET_ANGLE stay regression — they are a Nyxus-frame
// angle convention with no directly comparable imea output.)
void test_shape2d_caliper_feret_imea_ellipse_oracle()
{
	std::vector<std::vector<double>> fvals;
	calculate_ellipse_caliper_values(fvals);

	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MIN, "STAT_FERET_DIAM_MIN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MAX, "STAT_FERET_DIAM_MAX");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEAN, "STAT_FERET_DIAM_MEAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEDIAN, "STAT_FERET_DIAM_MEDIAN");
}

void test_remaining2d_verifiable_with_3p_builtin_oracle_chord_stat_features()
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

void test_remaining2d_unvetted_no_direct_oracle_chord_angle_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MAX_ANG, "MAXCHORDS_MAX_ANG");
	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::MAXCHORDS_MIN_ANG, "MAXCHORDS_MIN_ANG");
	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MAX_ANG, "ALLCHORDS_MAX_ANG");
	assert_unvetted_no_direct_oracle_remaining2d_feature(fvals, Nyxus::Feature2D::ALLCHORDS_MIN_ANG, "ALLCHORDS_MIN_ANG");
}

void test_remaining2d_unvetted_no_direct_oracle_polygonality_hexagonality_features()
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
