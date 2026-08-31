#pragma once

#include <string>
#include <vector>

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

// Provenance (SPEC 6.4):
//   tool      = MATLAB R2026a, Image Processing Toolbox 26.1
//   functions = regionprops, bweuler
//   fixture   = shape2d_morphology_{mask,intensity} in tests/test_data.h
//   config    = make_shape2d_settings(): PIXELSIZEUM=2.0, IBSI=false, single ROI
//   recipe    = morphology.shape2d_native
//   generator = tests/vetting/oracles/gen_morphology_matlab.m
//   report    = tests/vetting/audit/morphology_2d_matlab_vetting_report.md
//
// Coordinate conventions applied to make regionprops directly comparable, per feature:
//   Centroid / WeightedCentroid  MATLAB is 1-based pixel centres, Nyxus 0-based -> minus 1
//   BoundingBox                  MATLAB is [x_ul y_ul w h] with the corner at min-0.5 in 1-based
//                                coords -> the 0-based min index is BoundingBox(1) - 0.5
// AREA_UM2 is Area scaled by PIXELSIZEUM^2; ASPECT_RATIO and ELONGATION are ratios of two
// regionprops outputs (bbox w/h and minor/major), not properties of their own.
//
// The ellipse triple is the reason this family is vetted against MATLAB rather than scikit-image:
// MATLAB applies the same +1/12 pixel finite-size correction to the second central moments that
// Nyxus does, so MAJOR_AXIS_LENGTH / MINOR_AXIS_LENGTH / ECCENTRICITY agree to ~1e-15 here, while
// skimage (which omits it) differs ~1.4% and is left unvetted for them in
// test_2d_morphology_skimage.h.
static const ref_vals_map<double> morphology_2d_matlab_regionprops_ref_vals{
	{"AREA_PIXELS_COUNT", 26.0},
	{"AREA_UM2", 104.0},
	{"CENTROID_X", 2.6153846153846154},
	{"CENTROID_Y", 2.8461538461538463},
	{"WEIGHTED_CENTROID_X", 2.8416030534351147},
	{"WEIGHTED_CENTROID_Y", 3.4389312977099236},
	{"BBOX_XMIN", 0.0},
	{"BBOX_YMIN", 0.0},
	{"BBOX_WIDTH", 6.0},
	{"BBOX_HEIGHT", 7.0},
	{"ASPECT_RATIO", 0.8571428571428571},
	{"EXTENT", 0.61904761904761907},
	{"MAJOR_AXIS_LENGTH", 6.9688161689861872},
	{"MINOR_AXIS_LENGTH", 5.48870991295738},
	{"ELONGATION", 0.78761008754746209},
	{"ECCENTRICITY", 0.61617396082070774},
	// bweuler(mask, 8): one object with one hole -> 1 - 1
	{"EULER_NUMBER", 0.0},
};

// SPEC 7 "same-definition oracle" tier, rel=1e-3. The measured agreement is ~1e-15 (see the audit
// report), so the tier is conservative; it is still tight enough to catch the conventions that
// matter -- dropping the +1/12 moment correction moves the axis lengths 1.4%, i.e. 14x this bound.
static void assert_morphology_regionprops_matlab(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("MATLAB_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_matlab_regionprops_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_matlab_regionprops_ref_vals.at(feature_name), frac_tolerance));
}

void test_2d_morphology_basic_matlab()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::AREA_PIXELS_COUNT, "AREA_PIXELS_COUNT");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::AREA_UM2, "AREA_UM2");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::CENTROID_X, "CENTROID_X");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::CENTROID_Y, "CENTROID_Y");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_X, "WEIGHTED_CENTROID_X");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::WEIGHTED_CENTROID_Y, "WEIGHTED_CENTROID_Y");
}

void test_2d_morphology_bbox_matlab()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::BBOX_XMIN, "BBOX_XMIN");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::BBOX_YMIN, "BBOX_YMIN");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::BBOX_WIDTH, "BBOX_WIDTH");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::BBOX_HEIGHT, "BBOX_HEIGHT");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::ASPECT_RATIO, "ASPECT_RATIO");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::EXTENT, "EXTENT");
}

void test_2d_morphology_ellipse_matlab()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::MAJOR_AXIS_LENGTH, "MAJOR_AXIS_LENGTH");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::MINOR_AXIS_LENGTH, "MINOR_AXIS_LENGTH");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::ELONGATION, "ELONGATION");
	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::ECCENTRICITY, "ECCENTRICITY");
}

void test_2d_morphology_euler_matlab()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_regionprops_matlab(fvals, Nyxus::Feature2D::EULER_NUMBER, "EULER_NUMBER");
}

static const ref_vals_map<double> morphology_2d_matlab_extrema_ref_vals{
	// EXTREMA P1..P8 (X,Y) match MATLAB regionprops('Extrema') exactly under the documented
	// coordinate convention: MATLAB returns 1-based sub-pixel *corner* coords, Nyxus returns 0-based
	// pixel *centers*. The corner is direction-specific -> the offset is per-point: left/top coords
	// map as (matlab - 0.5), right/bottom coords as (matlab - 1.5). Verified on this 8x8 fixture by
	// gen_morphology_matlab.m: raw Extrema P1(2.5,0.5) P2(4.5,0.5) P3(6.5,2.5)
	// P4(6.5,4.5) P5(5.5,7.5) P6(3.5,7.5) P7(0.5,4.5) P8(0.5,2.5) -> after the per-point offset ->
	// P1(2,0) P2(3,0) P3(5,2) P4(5,3) P5(4,6) P6(3,6) P7(0,3) P8(0,2), i.e. these goldens exactly.
	// (The earlier "~1px off" on the right/bottom coords was a harness bug: it used a uniform -0.5.)
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

static void assert_morphology_extrema_matlab(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("MATLAB_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_matlab_extrema_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_matlab_extrema_ref_vals.at(feature_name), frac_tolerance));
}

void test_2d_morphology_extrema_matlab()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P1_X, "EXTREMA_P1_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P1_Y, "EXTREMA_P1_Y");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P2_X, "EXTREMA_P2_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P2_Y, "EXTREMA_P2_Y");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P3_X, "EXTREMA_P3_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P3_Y, "EXTREMA_P3_Y");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P4_X, "EXTREMA_P4_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P4_Y, "EXTREMA_P4_Y");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P5_X, "EXTREMA_P5_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P5_Y, "EXTREMA_P5_Y");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P6_X, "EXTREMA_P6_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P6_Y, "EXTREMA_P6_Y");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P7_X, "EXTREMA_P7_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P7_Y, "EXTREMA_P7_Y");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P8_X, "EXTREMA_P8_X");
	assert_morphology_extrema_matlab(fvals, Nyxus::Feature2D::EXTREMA_P8_Y, "EXTREMA_P8_Y");
}

// PERIMETER was asserted here against a MATLAB recipe that does not produce the pinned number:
// nnz(bwperim(...)) counts perimeter PIXELS (846 on this fixture), and regionprops('Perimeter')
// returns 952.848, while the golden is 999.26. That golden is scikit-image's, so the assertion now
// lives in test_2d_morphology_perimeter_skimage() -- the oracle the registry names for the feature.
