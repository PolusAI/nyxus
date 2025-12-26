#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/intensity.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
static std::unordered_map<std::string, double> IBSI_intensity_values {
    {"MEAN", 2.15},
    {"VARIANCE", 3.05},
    {"SKEWNESS", 1.08},
    {"EXCESS_KURTOSIS", -0.355},
    {"MEDIAN", 1},
    {"MINIMUM", 1},
    {"P10", 1},
    {"P90", 4},
    {"MAXIMUM", 6},
    {"INTERQUARTILE", 3},
    {"RANGE", 5},
    {"MEAN_ABSOLUTE_DEVIATION", 1.55},
    {"ROBUST_MEAN_ABSOLUTE_DEVIATION", 1.11},
    {"ENERGY", 567},
    {"ROOT_MEAN_SQUARED", 2.77}
};

void test_intensity_feature(const Feature2D& feature, const std::string& feature_name, bool round = false) {
    std::vector<NyxusPixel> combined_image;
    std::vector<NyxusPixel> combined_mask;

    for(auto& p: ibsi_phantom_z1_intensity)
        combined_image.push_back(p);

    for(auto& p: ibsi_phantom_z2_intensity)
        combined_image.push_back(p);

    for(auto& p: ibsi_phantom_z3_intensity)
        combined_image.push_back(p);

    for(auto& p: ibsi_phantom_z4_intensity)
        combined_image.push_back(p);
    // -------------------------
    for(auto& p: ibsi_phantom_z1_mask)
        combined_mask.push_back(p);

    for(auto& p: ibsi_phantom_z2_mask)
        combined_mask.push_back(p);

    for(auto& p: ibsi_phantom_z3_mask)
        combined_mask.push_back(p);

    for(auto& p: ibsi_phantom_z4_mask)
        combined_mask.push_back(p);

    double total = 0;
    
    Dataset ds;
    ds.dataset_props.push_back (SlideProps("",""));

    LR roidata;
    Fsettings s;
    PixelIntensityFeatures f;

    // image 1

    load_masked_test_roi_data (roidata, combined_image.data(), combined_mask.data(),  combined_image.size());
    ASSERT_NO_THROW(f.calculate(roidata, s, ds));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[(int)feature][0];

    if (round) total = std::round(total);

    ASSERT_TRUE(agrees_gt(total, IBSI_intensity_values[feature_name], 100.));
}

void test_ibsi_mean_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::MEAN, "MEAN");
}

void test_ibsi_skewness_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::SKEWNESS, "SKEWNESS");
}

void test_ibsi_kurtosis_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::EXCESS_KURTOSIS, "EXCESS_KURTOSIS");
}

void test_ibsi_median_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::MEDIAN, "MEDIAN");
}

void test_ibsi_minimum_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::MIN, "MINIMUM");
}

void test_ibsi_p10_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::P10, "P10");
}

// As noted in ibsi documentation, P90 can vary based on implementation from 4-4.2
// therefore, we round the result
void test_ibsi_p90_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::P90, "P90", true);
}

void test_ibsi_interquartile_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::INTERQUARTILE_RANGE, "INTERQUARTILE");
}

void test_ibsi_range_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::RANGE, "RANGE");
}

void test_ibsi_mean_absolute_deviation_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION, "MEAN_ABSOLUTE_DEVIATION");
}

/* This feature needs to be updated to pass test
void test_ibsi_robust_mean_absolute_deviation_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION, "ROBUST_MEAN_ABSOLUTE_DEVIATION");
}
*/

void test_ibsi_energy_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::ENERGY, "ENERGY");
}

void test_ibsi_root_mean_squared_intensity()
{
    test_intensity_feature(Nyxus::Feature2D::ROOT_MEAN_SQUARED, "ROOT_MEAN_SQUARED");
}