#pragma once

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

static const ref_vals_map<double> morphology_2d_matlab_extrema_ref_vals{
	// EXTREMA P1..P8 (X,Y) match MATLAB/Octave regionprops('Extrema') EXACTLY under the documented
	// coordinate convention: MATLAB returns 1-based sub-pixel *corner* coords, Nyxus returns 0-based
	// pixel *centers*. The corner is direction-specific -> the offset is per-point: left/top coords
	// map as (matlab - 0.5), right/bottom coords as (matlab - 1.5). Verified on this 8x8 fixture by
	// octave/matlab_oracle_tests/extrema_8x8.m: raw Extrema P1(2.5,0.5) P2(4.5,0.5) P3(6.5,2.5)
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


//
// The value of PERIMETER feature should match the pixel count of contour calculated 
// in Matlab as:
//      BW_noholes = imfill(imread('circles.png'), "holes");
//      nnz(bwperim(BW_noholes))    % returns 846
//
void test_2d_morphology_perimeter_matlab()
{
    // featue settings for this particular test
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 128;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data (roidata, roiDataForPerimeterTest, sizeof(roiDataForPerimeterTest)/sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    ContourFeature f;
    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::PERIMETER][0], 999.26));
}
