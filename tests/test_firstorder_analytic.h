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

// Analytic / closed-form first-order goldens (SPEC §6: test_firstorder_analytic.h).
// Moved out of test_firstorder_regression.h so regression files do not claim oracle status.

static constexpr double oracle_3p_builtin_hyperskewness_feature_golden_value = 1.978293086605381;
static constexpr double oracle_3p_builtin_hyperflatness_feature_golden_value = 5.126659243028459;
static constexpr double oracle_3p_builtin_uniformity_piu_feature_golden_value = 29.477577192725725;
static constexpr double oracle_3p_builtin_covered_image_intensity_range_feature_golden_value = 8.088960097657740e-01;
static constexpr double oracle_3p_builtin_robust_mean_feature_golden_value = 3.142136800000000e+04;

void test_pixel_intensity_verifiable_with_3p_builtin_oracle_hyperskewness()
{
    SCOPED_TRACE("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__HYPERSKEWNESS");

    // Feed data to the ROI
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("",""));

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    Fsettings s;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::HYPERSKEWNESS][0], oracle_3p_builtin_hyperskewness_feature_golden_value));
}

void test_pixel_intensity_verifiable_with_3p_builtin_oracle_hyperflatness()
{
    SCOPED_TRACE("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__HYPERFLATNESS");

    // Feed data to the ROI
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("",""));

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    Fsettings s;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::HYPERFLATNESS][0], oracle_3p_builtin_hyperflatness_feature_golden_value));
}

void test_pixel_intensity_verifiable_with_3p_builtin_oracle_uniformity_piu()
{
    SCOPED_TRACE("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__UNIFORMITY_PIU");

    // Feed data to the ROI
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("",""));

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::UNIFORMITY_PIU][0], oracle_3p_builtin_uniformity_piu_feature_golden_value));
}

void test_pixel_intensity_verifiable_with_3p_builtin_oracle_covered_image_intensity_range()
{
    SCOPED_TRACE("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__COVERED_IMAGE_INTENSITY_RANGE");

    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals, Fsettings(), 0, 0.0, 65535.0);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::COVERED_IMAGE_INTENSITY_RANGE][0], oracle_3p_builtin_covered_image_intensity_range_feature_golden_value));
}

void test_pixel_intensity_verifiable_with_3p_builtin_oracle_robust_mean()
{
    SCOPED_TRACE("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__ROBUST_MEAN");

    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::ROBUST_MEAN][0], oracle_3p_builtin_robust_mean_feature_golden_value));
}

