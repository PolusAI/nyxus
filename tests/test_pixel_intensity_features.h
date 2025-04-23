#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/intensity.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

using namespace Nyxus;

// ROI pixel accumulation routines implemented in Nyxus

void test_pixel_intensity_integrated_intensity()
{
    // Feed data to the ROI
    LR roidata (100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data (roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData)/sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::INTEGRATED_INTENSITY][0], 5015224));
}

void test_pixel_intensity_min_max_range()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MIN][0], 11079));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MAX][0], 64090));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::RANGE][0], 53011));
}

void test_pixel_intensity_mean()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MEAN][0], 3.256638961038961e+04));
}

void test_pixel_intensity_median()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MEDIAN][0], 2.980350000000000e+04));
}

void test_pixel_intensity_mode()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MODE][0], 19552));
}

void test_pixel_intensity_standard_deviation()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::STANDARD_DEVIATION][0], 1.473096831710767e+04));
}

void test_pixel_intensity_skewness()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::SKEWNESS][0], 0.450256759704494));
}

void test_pixel_intensity_kurtosis()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::EXCESS_KURTOSIS][0], 1.927888720710090-3));
}

void test_pixel_intensity_hyperskewness()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::HYPERSKEWNESS][0], 1.978293086605381));
}

void test_pixel_intensity_hyperflatness()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::HYPERFLATNESS][0], 5.126659243028459));
}

void test_pixel_intensity_mean_absolute_deviation()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION][0], 1.283308449991567e+04));
}

void test_pixel_intensity_robust_mean_absolute_deviation()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 1.115934515469031e+04));
}

void test_pixel_intensity_standard_error()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::STANDARD_ERROR][0], 1.187055255225567e+03));
}

void test_pixel_intensity_root_mean_squared()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::ROOT_MEAN_SQUARED][0], 3.572341052638121e+04));
}

void test_pixel_intensity_entropy()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::ENTROPY][0], 2.898016861688313));
}

void test_pixel_intensity_energy()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::ENERGY][0], 1.965289571840000e+11));
}

void test_pixel_intensity_uniformity()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Anisotropy (none)
    roidata.make_nonanisotropic_aabb();

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::UNIFORMITY][0], 322, 100)); // Using 1% tolerance vs MATLAB
}

void test_pixel_intensity_uniformity_piu()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::UNIFORMITY_PIU][0], 29.477577192725725));
}

void test_pixel_intensity_percentiles_iqr()
{
    // Feed data to the ROI
    LR roidata(100);   // dummy label 100
    roidata.slide_idx = -1; // we don't have a real slide for this test ROI
    load_test_roi_data(roidata, pixelIntensityFeaturesTestData, sizeof(pixelIntensityFeaturesTestData) / sizeof(NyxusPixel));

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::P01][0], 1.208140000000000e+04));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::P10][0], 16329));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::P25][0], 19552));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::P75][0], 45723));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::P90][0], 5.336070000000000e+04));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::P99][0], 6.338096000000000e+04));
    ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::INTERQUARTILE_RANGE][0], 26171));
}

