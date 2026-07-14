#pragma once

#include "test_morphology_common.h"

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
