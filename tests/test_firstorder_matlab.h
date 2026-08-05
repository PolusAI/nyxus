#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/dataset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_firstorder_regression.h"  // calculate_pixel_intensity_feature_values helper

using namespace Nyxus;

// MATLAB-oracle first-order goldens (SPEC §6: test_firstorder_matlab.h).
// Moved out of test_firstorder_regression.h so regression files do not claim oracle status.

static constexpr double oracle_3p_matlab_uniformity_feature_golden_value = 0.0647664;

void test_pixel_intensity_uniformity()
{
    // Feed data to the ROI
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("",""));

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // settings important for this feature
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = 20;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::IBSI].bval = false;

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::UNIFORMITY][0], oracle_3p_matlab_uniformity_feature_golden_value, 100)); // Using 1% tolerance vs MATLAB
}

