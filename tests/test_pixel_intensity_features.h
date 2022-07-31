#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include <src/nyx/parallel.h>
#include "../src/nyx/features/intensity.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"

// ROI pixel accumulation routines implemented in Nyxus
namespace Nyxus
{
    void init_label_record_2(LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity, unsigned int tile_index);
    void update_label_record_2(LR& lr, int x, int y, int label, PixIntens intensity, unsigned int tile_index);

    /// @brief Tests the agreement with ground truth up to the tolerance specified as a fraction of the ground truth
    bool agrees_gt(double fval, double ground_truth, double frac_tolerance = 1000.)
    {
        auto diff = fval - ground_truth;
        auto tolerance = ground_truth / frac_tolerance;
        bool good = std::abs(diff) <= std::abs(tolerance);
        return good;
    }

    void load_test_roi_data(LR& roidata)
    {
        int dummyLabel = 100, dummyTile = 200;

        // -- mocking gatherRoisMetrics():
        for (auto& px : testData)
        {
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_2(roidata, "theSegFname", "theIntFname", px.x, px.y, dummyLabel, px.intensity, dummyTile);
            else
                update_label_record_2(roidata, px.x, px.y, dummyLabel, px.intensity, dummyTile);
        }

        // -- mocking scanTrivialRois():
        for (auto& px : testData)
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(px.x, px.y, px.intensity));
    }
}

void test_pixel_intensity_integrated_intensity()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[INTEGRATED_INTENSITY][0], 5015224));
}

void test_pixel_intensity_min_max_range()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[MIN][0], 11079));
    ASSERT_TRUE(agrees_gt(roidata.fvals[MAX][0], 64090));
    ASSERT_TRUE(agrees_gt(roidata.fvals[RANGE][0], 53011));
}

void test_pixel_intensity_mean()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[MEAN][0], 3.256638961038961e+04));
}

void test_pixel_intensity_median()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[MEDIAN][0], 2.980350000000000e+04));
}

void test_pixel_intensity_mode()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[MODE][0], 19552));
}

void test_pixel_intensity_standard_deviation()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[STANDARD_DEVIATION][0], 1.473096831710767e+04));
}

void test_pixel_intensity_skewness()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[SKEWNESS][0], 0.450256759704494));
}

void test_pixel_intensity_kurtosis()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[KURTOSIS][0], 1.927888720710090));
}

void test_pixel_intensity_hyperskewness()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[HYPERSKEWNESS][0], 1.978293086605381));
}

void test_pixel_intensity_hyperflatness()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[HYPERFLATNESS][0], 5.126659243028459));
}

void test_pixel_intensity_mean_absolute_deviation()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[MEAN_ABSOLUTE_DEVIATION][0], 1.283308449991567e+04));
}

void test_pixel_intensity_robust_mean_absolute_deviation()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 1.115934515469031e+04));
}

void test_pixel_intensity_standard_error()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[STANDARD_ERROR][0], 1.187055255225567e+03));
}

void test_pixel_intensity_root_mean_squared()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[ROOT_MEAN_SQUARED][0], 3.572341052638121e+04));
}

void test_pixel_intensity_entropy()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[ENTROPY][0], 2.898016861688313));
}

void test_pixel_intensity_energy()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[ENERGY][0], 1.965289571840000e+11));
}

void test_pixel_intensity_uniformity()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[UNIFORMITY][0], 3392));
}

void test_pixel_intensity_uniformity_piu()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[UNIFORMITY_PIU][0], 29.477577192725725));
}

void test_pixel_intensity_p01()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[P01][0], 1.208140000000000e+04));
}

void test_pixel_intensity_p10()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[P10][0], 16329));
}

void test_pixel_intensity_p25()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[P25][0], 19552));
}

void test_pixel_intensity_p75()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[P75][0], 45723));
}

void test_pixel_intensity_p90()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[P90][0], 5.336070000000000e+04));
}

void test_pixel_intensity_p99()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[P99][0], 6.338096000000000e+04));
}

void test_pixel_intensity_interquartile_range()
{
    // Feed data to the ROI
    LR roidata;
    load_test_roi_data(roidata);

    // Calculate features
    PixelIntensityFeatures f;
    ASSERT_NO_THROW(f.calculate(roidata));

    // Retrieve the feature values
    roidata.initialize_fvals();
    f.save_value(roidata.fvals);

    // Check the feature values vs ground truth
    ASSERT_TRUE(agrees_gt(roidata.fvals[INTERQUARTILE_RANGE][0], 26171));
}