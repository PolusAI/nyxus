#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/dataset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

using namespace Nyxus;

// ROI pixel accumulation routines implemented in Nyxus
static constexpr double oracle_3p_builtin_hyperskewness_feature_golden_value = 1.978293086605381;
static constexpr double oracle_3p_builtin_hyperflatness_feature_golden_value = 5.126659243028459;
static constexpr double oracle_3p_matlab_uniformity_feature_golden_value = 0.0647664;
static constexpr double oracle_3p_builtin_uniformity_piu_feature_golden_value = 29.477577192725725;
static constexpr double oracle_3p_builtin_covered_image_intensity_range_feature_golden_value = 8.088960097657740e-01;
static constexpr double oracle_3p_builtin_robust_mean_feature_golden_value = 3.142136800000000e+04;

static void calculate_pixel_intensity_feature_values(
    std::vector<std::vector<double>>& fvals,
    Fsettings s = Fsettings(),
    int slide_idx = -1,
    double slide_min = -1.0,
    double slide_max = -1.0)
{
    Dataset ds;
    ds.dataset_props.push_back(SlideProps("", ""));
    if (slide_idx >= 0)
    {
        ds.dataset_props[slide_idx].min_preroi_inten = slide_min;
        ds.dataset_props[slide_idx].max_preroi_inten = slide_max;
    }

    LR roidata(100);   // dummy label 100
    roidata.slide_idx = slide_idx;
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    roidata.make_nonanisotropic_aabb();

    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    roidata.initialize_fvals();
    f.save_value(roidata.fvals);
    fvals = roidata.fvals;
}

void test_firstorder_integrated_intensity_regression()
{
    // Feed data to the ROI
    Dataset ds;
    ds.dataset_props.push_back (SlideProps("",""));

    LR roidata (100);   // dummy label 100
    roidata.slide_idx = 0; // just 1 slide that we've just added to 'ds.dataset_props'
    load_test_roi_data (roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData)/sizeof(NyxusPixel));

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
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::INTEGRATED_INTENSITY][0], 5015224));
}

void test_firstorder_min_max_range_regression()
{
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
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MIN][0], 11079));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MAX][0], 64090));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::RANGE][0], 53011));
}

void test_firstorder_mean_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MEAN][0], 3.256638961038961e+04));
}

void test_firstorder_median_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MEDIAN][0], 2.980350000000000e+04));
}

void test_firstorder_mode_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MODE][0], 19552));
}

void test_firstorder_standard_deviation_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::STANDARD_DEVIATION][0], 1.473096831710767e+04));
}

void test_firstorder_skewness_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::SKEWNESS][0], 0.450256759704494));
}

void test_firstorder_kurtosis_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::EXCESS_KURTOSIS][0], 1.927888720710090 - 3));
}

void test_firstorder_pearson_kurtosis_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::KURTOSIS][0], 1.927888720710090));
}

void test_firstorder_hyperskewness_regression()
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

void test_firstorder_hyperflatness_regression()
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

void test_firstorder_mean_absolute_deviation_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION][0], 1.283308449991567e+04));
}

void test_firstorder_robust_mean_absolute_deviation_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 1.044061849600000e+04));
}

void test_firstorder_standard_error_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::STANDARD_ERROR][0], 1.187055255225567e+03));
}

void test_firstorder_root_mean_squared_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::ROOT_MEAN_SQUARED][0], 3.572341052638121e+04));
}

void test_firstorder_entropy_regression()
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

void test_firstorder_energy_regression()
{
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
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::ENERGY][0], 1.965289571840000e+11));
}

void test_firstorder_uniformity_regression()
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

void test_firstorder_uniformity_piu_regression()
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

void test_firstorder_percentiles_iqr_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::P01][0], 1.189536940000000e+04));
    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::P10][0], 1.610747200000000e+04));
    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::P25][0], 1.907482583333333e+04));
    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::P75][0], 4.580120500000000e+04));
    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::P90][0], 5.338177800000000e+04));
    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::P99][0], 6.341676030000000e+04));
    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::INTERQUARTILE_RANGE][0], 2.672637916666667e+04));
}

void test_firstorder_cov_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::COV][0], 4.523365498399634e-01));
}

void test_firstorder_covered_image_range_regression()
{
    SCOPED_TRACE("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__COVERED_IMAGE_INTENSITY_RANGE");

    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals, Fsettings(), 0, 0.0, 65535.0);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::COVERED_IMAGE_INTENSITY_RANGE][0], oracle_3p_builtin_covered_image_intensity_range_feature_golden_value));
}

void test_firstorder_median_absolute_deviation_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::MEDIAN_ABSOLUTE_DEVIATION][0], 1.269384415584416e+04));
}

void test_firstorder_qcod_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::QCOD][0], 4.119607630640470e-01));
}

void test_firstorder_robust_mean_regression()
{
    SCOPED_TRACE("VERIFIABLE_WITH_3P_BUILTIN_ORACLE__ROBUST_MEAN");

    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::ROBUST_MEAN][0], oracle_3p_builtin_robust_mean_feature_golden_value));
}

void test_firstorder_standard_deviation_biased_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::STANDARD_DEVIATION_BIASED][0], 1.468306260221863e+04));
}

void test_firstorder_variance_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::VARIANCE][0], 2.170014275596299e+08));
}

void test_firstorder_variance_biased_regression()
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals);

    ASSERT_TRUE(agrees_gt(fvals[(int)Nyxus::Feature2D::VARIANCE_BIASED][0], 2.155923273806713e+08));
}

