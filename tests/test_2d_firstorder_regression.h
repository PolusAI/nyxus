#pragma once

// 2D first-order drift guards: pinned Nyxus output, no external reference. Only the features whose
// oracle_coverage.csv target_test is this file live here; everything else moved to the oracle file
// column J names (test_2d_firstorder_matlab.h / _pyradiomics.h / _ibsi.h).

#include "test_2d_firstorder_common.h"

// ROI pixel accumulation routines implemented in Nyxus

void test_2d_firstorder_entropy_regression()
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::ENTROPY][0], 4.12733));
}

