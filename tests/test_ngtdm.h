#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/ngtdm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
// Calculated at grey scalefactlr 100
static std::unordered_map<std::string, float> ngtdm_values {
    {"NGTDM_COARSENESS", 0.008374068},
    {"NGTDM_CONTRAST", 3169.92908},
    {"NGTDM_BUSYNESS", 1.444571},
    {"NGTDM_COMPLEXITY", 3608.3891},
    {"NGTDM_STRENGTH", 52.076642}
};

void test_ngtdm_feature(const Feature2D& feature_, const std::string& feature_name) 
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

    // Set feature's state
    Environment::ibsi_compliance = false;
    NGTDMFeature::n_levels = 100;

    int feature = int(feature_);

    double total = 0;
    
    LR roidata;
    NGTDMFeature f;

    // image 1

    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    
    // image 2

    LR roidata1;
    NGTDMFeature f1;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];

    // image 3

    LR roidata2;
    NGTDMFeature f2;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];

    // image 4
    
    LR roidata3;
    NGTDMFeature f3;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += roidata3.fvals[feature][0];

    // Verdict
    ASSERT_TRUE(agrees_gt(total/4, ngtdm_values[feature_name], 100.));
}

void test_ngtdm_coarseness()
{
    test_ngtdm_feature(Nyxus::Feature2D::NGTDM_COARSENESS, "NGTDM_COARSENESS");
}

void test_ngtdm_contrast()
{
    test_ngtdm_feature(Nyxus::Feature2D::NGTDM_CONTRAST, "NGTDM_CONTRAST");
}

void test_ngtdm_busyness()
{
    test_ngtdm_feature(Nyxus::Feature2D::NGTDM_BUSYNESS, "NGTDM_BUSYNESS");
}

void test_ngtdm_complexity()
{   
    test_ngtdm_feature(Nyxus::Feature2D::NGTDM_COMPLEXITY, "NGTDM_COMPLEXITY");
}

void test_ngtdm_strength()
{
    test_ngtdm_feature(Nyxus::Feature2D::NGTDM_STRENGTH, "NGTDM_STRENGTH");
}