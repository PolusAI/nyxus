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

static std::unordered_map<std::string, double> shape2d_truth{
	{"AREA_PIXELS_COUNT", 26.0},
	{"AREA_UM2", 104.0},
	{"CENTROID_X", 2.61538461538462},
	{"CENTROID_Y", 2.84615384615385},
	{"WEIGHTED_CENTROID_X", 2.84160305343511},   // fix #10: 0-based weighted centroid (dropped MATLAB +1)
	{"WEIGHTED_CENTROID_Y", 3.43893129770992},   // fix #10
	{"MASS_DISPLACEMENT", 0.634476074243407},    // fix #10: |weighted - geometric centroid|, verified vs numpy
	{"COMPACTNESS", 0.0275177},   // fix #18: std of MEAN-centroid (not coord-sum) distances, dx/dy in double
	{"BBOX_XMIN", 0.0},
	{"BBOX_YMIN", 0.0},
	{"BBOX_WIDTH", 6.0},
	{"BBOX_HEIGHT", 7.0},
	{"DIAMETER_EQUAL_AREA", 5.75362739175159},   // fix #5: sqrt(4*area/pi), area=26 -> 5.7536 (verified vs numpy)
	{"EXTENT", 0.619047619047619},
	{"ASPECT_RATIO", 0.857142857142857},
	{"MAJOR_AXIS_LENGTH", 6.96881616898619},
	{"MINOR_AXIS_LENGTH", 5.48870991295738},
	{"ELONGATION", 0.787610087547462},
	{"ECCENTRICITY", 0.616173960820708},
	{"ORIENTATION", 70.4173944984207},
	{"ROUNDNESS", 0.681656295209303},
	{"PERIMETER", 26.9349412836191},
	{"DIAMETER_EQUAL_PERIMETER", 8.57365809435587},
	{"EDGE_MEAN_INTENSITY", 41.8333333333333},
	{"EDGE_STDDEV_INTENSITY", 16.7691944455582},
	{"EDGE_MAX_INTENSITY", 68.0},
	{"EDGE_MIN_INTENSITY", 12.0},
	{"EDGE_INTEGRATED_INTENSITY", 753.0},
	{"CIRCULARITY", 0.671081973229055},
	{"CONVEX_HULL_AREA", 27.0},                   // fix #6: Pick's-theorem pixel-count hull (skimage 28, Δ4%)
	{"SOLIDITY", 0.9629629629629629},             // fix #6: now <=1 (was 1.3, impossible); skimage 0.929
	{"EULER_NUMBER", 0.0},                        // fix #7: 1 object - 1 hole at (3,3) = 0 (verified: scipy label/fill_holes)
	{"FRACT_DIM_BOXCOUNT", 1.5849625007211565},   // fix #8: mean log-log slope = log2(3); old -0.83 was broken
	{"FRACT_DIM_PERIMETER", 0.3187149603076458},  // fix #8: Richardson D=1-slope (no external oracle); old -1.97 was broken
	{"DIAMETER_MIN_ENCLOSING_CIRCLE", 6.32475519180298},     // translation-invariant: unaffected by the contour +1 offset
	{"DIAMETER_CIRCUMSCRIBING_CIRCLE", 6.8888},              // fix #17: undo contour (+1,+1) frame offset in centroid-relative radii
	{"DIAMETER_INSCRIBING_CIRCLE", 1.26865},                // fix #17: undo contour (+1,+1) frame offset in centroid-relative radii
	{"ROI_RADIUS_MEAN", 0.75603285574970702},    // fix #11 (sqrt) + #19 (contour frame): distances to the true-global contour
	{"ROI_RADIUS_MAX", 2.0},                      // fix #11: was 4 == 2^2 (squared); now 2 (<= image diagonal)
	{"ROI_RADIUS_MEDIAN", 1.0},
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

static void assert_shape2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	ASSERT_TRUE(shape2d_truth.count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], shape2d_truth[feature_name], frac_tolerance));
}

static void assert_unvetted_no_direct_oracle_shape2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	assert_shape2d_feature(fvals, feature, feature_name, frac_tolerance);
}

static void assert_verifiable_with_3p_builtin_oracle_shape2d_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__") + feature_name);
	assert_shape2d_feature(fvals, feature, feature_name, frac_tolerance);
}

void test_shape2d_basic_morphology_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_shape2d_feature(fvals, Nyxus::Feature2D::AREA_PIXELS_COUNT, "AREA_PIXELS_COUNT");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::AREA_UM2, "AREA_UM2");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::CENTROID_X, "CENTROID_X");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::CENTROID_Y, "CENTROID_Y");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_X, "WEIGHTED_CENTROID_X");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_Y, "WEIGHTED_CENTROID_Y");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::MASS_DISPLACEMENT, "MASS_DISPLACEMENT");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::COMPACTNESS, "COMPACTNESS");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_XMIN, "BBOX_XMIN");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_YMIN, "BBOX_YMIN");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_WIDTH, "BBOX_WIDTH");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::BBOX_HEIGHT, "BBOX_HEIGHT");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_EQUAL_AREA, "DIAMETER_EQUAL_AREA");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::EXTENT, "EXTENT");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::ASPECT_RATIO, "ASPECT_RATIO");
}

void test_shape2d_ellipse_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_shape2d_feature(fvals, Nyxus::Feature2D::MAJOR_AXIS_LENGTH, "MAJOR_AXIS_LENGTH");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::MINOR_AXIS_LENGTH, "MINOR_AXIS_LENGTH");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::ELONGATION, "ELONGATION");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::ECCENTRICITY, "ECCENTRICITY");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::ORIENTATION, "ORIENTATION");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::ROUNDNESS, "ROUNDNESS");
}

void test_shape2d_contour_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_shape2d_feature(fvals, Nyxus::Feature2D::PERIMETER, "PERIMETER");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_MEAN_INTENSITY, "EDGE_MEAN_INTENSITY");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_STDDEV_INTENSITY, "EDGE_STDDEV_INTENSITY");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_MAX_INTENSITY, "EDGE_MAX_INTENSITY");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_MIN_INTENSITY, "EDGE_MIN_INTENSITY");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, "EDGE_INTEGRATED_INTENSITY");
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

	assert_shape2d_feature(fvals, Nyxus::Feature2D::CIRCULARITY, "CIRCULARITY");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::CONVEX_HULL_AREA, "CONVEX_HULL_AREA");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::SOLIDITY, "SOLIDITY");
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

	assert_shape2d_feature(fvals, Nyxus::Feature2D::EULER_NUMBER, "EULER_NUMBER");
	assert_shape2d_feature(fvals, Nyxus::Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE, "DIAMETER_MIN_ENCLOSING_CIRCLE");
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
