#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_morphology_common.h"

//
// The value of PERIMETER feature should match the pixel count of contour calculated
// in Matlab as:
//      BW_noholes = imfill(imread('circles.png'), "holes");
//      nnz(bwperim(BW_noholes))    % returns 846
//
void test_morphology_perimeter_matlab()
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

void test_morphology_extrema_matlab()
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
