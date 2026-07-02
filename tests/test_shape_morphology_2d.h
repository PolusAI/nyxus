#pragma once

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <unordered_map>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/convex_hull.h"
#include "../src/nyx/features/ellipse_fitting.h"
#include "../src/nyx/features/extrema.h"
#include "../src/nyx/features/roi_radius.h"
#include "../src/nyx/features/euler_number.h"
#include "../src/nyx/features/fractal_dim.h"
#include "../src/nyx/features/circle.h"
#include "../src/nyx/features/geodetic_len_thickness.h"
#include "../src/nyx/features/erosion.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// "Unvetted" means this table is not claiming an accepted independent
// built-in/package oracle for these rows. Some entries, such as area,
// centroids, and bounding-box limits, are easy to inspect on this small
// fixture; that kind of hand recomputation is useful as a sanity check but is
// not V&V by our tracker definition, because it restates the fixture geometry
// and Nyxus coordinate/spacing conventions rather than using a third-party
// oracle. Rows with accepted external built-in/API comparators live in the
// oracle_3p table below.
static std::unordered_map<std::string, double> unvetted_nyxus_regression_shape2d_feature_golden_values{
	{"AREA_PIXELS_COUNT", 26.0},
	{"AREA_UM2", 104.0},
	{"CENTROID_X", 2.61538461538462},
	{"CENTROID_Y", 2.84615384615385},
	{"WEIGHTED_CENTROID_X", 2.84160305343511},
	{"WEIGHTED_CENTROID_Y", 3.43893129770992},
	{"MASS_DISPLACEMENT", 0.634476074243407},
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
	{"EDGE_MEAN_INTENSITY", 41.8333333333333},
	{"EDGE_STDDEV_INTENSITY", 16.7691944455582},
	{"EDGE_MAX_INTENSITY", 68.0},
	{"EDGE_MIN_INTENSITY", 12.0},
	{"EDGE_INTEGRATED_INTENSITY", 753.0},
	{"CIRCULARITY", 0.671081973229055},
	{"CONVEX_HULL_AREA", 20.0},
	{"SOLIDITY", 1.3},
	{"EULER_NUMBER", 0.0},
	{"DIAMETER_MIN_ENCLOSING_CIRCLE", 6.32475519180298},
	{"ROI_RADIUS_MEAN", 1.07692307692308},
	{"ROI_RADIUS_MAX", 4.0},
	{"ROI_RADIUS_MEDIAN", 1.0},
};

static std::unordered_map<std::string, double> oracle_3p_shape2d_feature_golden_values{
	{"DIAMETER_EQUAL_PERIMETER", 8.57365809435587},
	{"FRACT_DIM_BOXCOUNT", 1.5849625007211565},   // FIX (fractal_dim.cpp): mean log-log slope = log2(3); old -0.83 was slope-of-slopes (~0)
	{"FRACT_DIM_PERIMETER", 0.3187149603076458},  // FIX: Richardson D = 1 - slope; old -1.97 was the raw slope
	{"DIAMETER_CIRCUMSCRIBING_CIRCLE", 12.3317073399088},
	{"DIAMETER_INSCRIBING_CIRCLE", 0.828486893405308},
	{"GEODETIC_LENGTH", 10.0},
	{"THICKNESS", 3.0},
	{"EROSIONS_2_VANISH", 1.0},
	{"EXTREMA_P1_X", 2.0},
	{"EXTREMA_P1_Y", 0.0},
	{"EXTREMA_P2_X", 3.0},
	{"EXTREMA_P2_Y", 0.0},
	{"EXTREMA_P3_X", 5.0},
	{"EXTREMA_P3_Y", 2.0},
	{"EXTREMA_P4_X", 5.0},
	{"EXTREMA_P4_Y", 3.0},
	{"EXTREMA_P5_X", 4.0},
	{"EXTREMA_P5_Y", 6.0},
	{"EXTREMA_P6_X", 3.0},
	{"EXTREMA_P6_Y", 6.0},
	{"EXTREMA_P7_X", 0.0},
	{"EXTREMA_P7_Y", 3.0},
	{"EXTREMA_P8_X", 0.0},
	{"EXTREMA_P8_Y", 2.0},
};

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

static void calculate_shape2d_feature_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_shape2d_settings();

	LR roidata(101);
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

	EllipseFittingFeature ellipse;
	ellipse.calculate(roidata, s);
	ellipse.save_value(roidata.fvals);

	ExtremaFeature extrema;
	extrema.calculate(roidata, s);
	extrema.save_value(roidata.fvals);

	RoiRadiusFeature radius;
	radius.calculate(roidata, s);
	radius.save_value(roidata.fvals);

	EulerNumberFeature euler;
	euler.calculate(roidata, s);
	euler.save_value(roidata.fvals);

	FractalDimensionFeature fractal;
	fractal.calculate(roidata, s);
	fractal.save_value(roidata.fvals);

	EnclosingInscribingCircumscribingCircleFeature circle;
	circle.calculate(roidata, s);
	circle.save_value(roidata.fvals);

	GeodeticLengthThicknessFeature geodetic;
	geodetic.calculate(roidata, s);
	geodetic.save_value(roidata.fvals);

	ErosionPixelsFeature erosion;
	erosion.calculate(roidata, s);
	erosion.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

static void assert_unvetted_no_direct_oracle_shape2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	ASSERT_TRUE(unvetted_nyxus_regression_shape2d_feature_golden_values.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], unvetted_nyxus_regression_shape2d_feature_golden_values[feature_name], frac_tolerance));
}

static void assert_verifiable_with_3p_builtin_oracle_shape2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__") + feature_name);
	ASSERT_TRUE(oracle_3p_shape2d_feature_golden_values.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], oracle_3p_shape2d_feature_golden_values[feature_name], frac_tolerance));
}

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

void test_shape2d_verifiable_with_3p_builtin_oracle_contour_diameter_equal_perimeter()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_EQUAL_PERIMETER, "DIAMETER_EQUAL_PERIMETER");
}

void test_shape2d_convex_hull_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::CIRCULARITY, "CIRCULARITY");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::CONVEX_HULL_AREA, "CONVEX_HULL_AREA");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::SOLIDITY, "SOLIDITY");
}

void test_shape2d_verifiable_with_3p_builtin_oracle_extrema_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P1_X, "EXTREMA_P1_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P1_Y, "EXTREMA_P1_Y");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P2_X, "EXTREMA_P2_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P2_Y, "EXTREMA_P2_Y");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P3_X, "EXTREMA_P3_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P3_Y, "EXTREMA_P3_Y");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P4_X, "EXTREMA_P4_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P4_Y, "EXTREMA_P4_Y");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P5_X, "EXTREMA_P5_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P5_Y, "EXTREMA_P5_Y");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P6_X, "EXTREMA_P6_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P6_Y, "EXTREMA_P6_Y");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P7_X, "EXTREMA_P7_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P7_Y, "EXTREMA_P7_Y");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P8_X, "EXTREMA_P8_X");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EXTREMA_P8_Y, "EXTREMA_P8_Y");
}

void test_shape2d_misc_shape_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EULER_NUMBER, "EULER_NUMBER");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE, "DIAMETER_MIN_ENCLOSING_CIRCLE");
}

void test_shape2d_verifiable_with_3p_builtin_oracle_fractal_circle_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::FRACT_DIM_BOXCOUNT, "FRACT_DIM_BOXCOUNT");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::FRACT_DIM_PERIMETER, "FRACT_DIM_PERIMETER");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE, "DIAMETER_CIRCUMSCRIBING_CIRCLE");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_INSCRIBING_CIRCLE, "DIAMETER_INSCRIBING_CIRCLE");
}

void test_shape2d_unvetted_no_direct_oracle_radius_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ROI_RADIUS_MEAN, "ROI_RADIUS_MEAN");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ROI_RADIUS_MAX, "ROI_RADIUS_MAX");
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::ROI_RADIUS_MEDIAN, "ROI_RADIUS_MEDIAN");
}

void test_shape2d_verifiable_with_3p_builtin_oracle_geodetic_thickness_erosion_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::GEODETIC_LENGTH, "GEODETIC_LENGTH");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::THICKNESS, "THICKNESS");
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::EROSIONS_2_VANISH, "EROSIONS_2_VANISH");
}
