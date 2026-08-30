#pragma once

#include "test_2d_firstorder_common.h"
#include "test_ref_vals.h"

// dig. phantom values for intensity based features
static const ref_vals_map<double> firstorder_2d_ibsi_ref_vals {
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

void assert_firstorder_feature_ibsi(const Feature2D& feature, const std::string& feature_name, bool round = false) {
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

    ASSERT_TRUE(agrees_gt(total, firstorder_2d_ibsi_ref_vals.at(feature_name), 100.));
}

void test_2d_firstorder_mean_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::MEAN, "MEAN");
}

void test_2d_firstorder_skewness_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::SKEWNESS, "SKEWNESS");
}

void test_2d_firstorder_kurtosis_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::EXCESS_KURTOSIS, "EXCESS_KURTOSIS");
}

void test_2d_firstorder_median_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::MEDIAN, "MEDIAN");
}

void test_2d_firstorder_minimum_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::MIN, "MINIMUM");
}

void test_2d_firstorder_p10_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::P10, "P10");
}

// As noted in ibsi documentation, P90 can vary based on implementation from 4-4.2
// therefore, we round the result
void test_2d_firstorder_p90_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::P90, "P90", true);
}

void test_2d_firstorder_interquartile_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::INTERQUARTILE_RANGE, "INTERQUARTILE");
}

void test_2d_firstorder_range_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::RANGE, "RANGE");
}

void test_2d_firstorder_mean_absolute_deviation_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION, "MEAN_ABSOLUTE_DEVIATION");
}

/* This feature needs to be updated to pass test
void test_2d_firstorder_robust_mean_absolute_deviation_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION, "ROBUST_MEAN_ABSOLUTE_DEVIATION");
}
*/

void test_2d_firstorder_energy_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::ENERGY, "ENERGY");
}

void test_2d_firstorder_root_mean_squared_ibsi()
{
    assert_firstorder_feature_ibsi(Nyxus::Feature2D::ROOT_MEAN_SQUARED, "ROOT_MEAN_SQUARED");
}