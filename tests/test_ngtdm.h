#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/ngtdm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
static std::unordered_map<std::string, float> ngtdm_values {
    {"NGTDM_COARSENESS", 0.0269458},
    {"NGTDM_CONTRAST", 1.23285},
    {"NGTDM_BUSYNESS", 115.112},
    {"NGTDM_COMPLEXITY", 71.0732},
    {"NGTDM_STRENGTH", 0.015942}
};

void test_ngtdm_feature(const Feature2D& feature, const std::string& feature_name) {

    double total = 0;
    
    LR roidata;
    // Calculate features
    NGTDMFeature f;
    Environment::ibsi_compliance = false;

    // image 1
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    
    // image 2
    // Calculate features
    LR roidata1;
    // Calculate features
    NGTDMFeature f1;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];

    // image 3
    // Calculate features

    LR roidata2;
    // Calculate features
    NGTDMFeature f2;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];

    // image 4
    // Calculate features
    
    LR roidata3;
    // Calculate features
    NGTDMFeature f3;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += roidata3.fvals[feature][0];

    ASSERT_TRUE(agrees_gt(total/4, ngtdm_values[feature_name], 100.));
}

void test_ngtdm_coarseness()
{
    test_ngtdm_feature(NGTDM_COARSENESS, "NGTDM_COARSENESS");
}

void test_ngtdm_contrast()
{
    test_ngtdm_feature(NGTDM_CONTRAST, "NGTDM_CONTRAST");
}

void test_ngtdm_busyness()
{
    test_ngtdm_feature(NGTDM_BUSYNESS, "NGTDM_BUSYNESS");
}

void test_ngtdm_complexity()
{   
    test_ngtdm_feature(NGTDM_COMPLEXITY, "NGTDM_COMPLEXITY");
}

void test_ngtdm_strength()
{
    test_ngtdm_feature(NGTDM_STRENGTH, "NGTDM_STRENGTH");
}