#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glrlm.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
static std::unordered_map<std::string, float> IBSI_glrlm_values {
    {"GLRLM_SRE", 0.641},
    {"GLRLM_LRE", 3.78},
    {"GLRLM_LGLRE", 0.604},
    {"GLRLM_HGLRE", 9.82},
    {"GLRLM_SRLGLE", 0.294},
    {"GLRLM_SRHGLE", 8.57},
    {"GLRLM_LRLGLE", 3.14},
    {"GLRLM_LRHGLE", 17.4},
    {"GLRLM_GLN", 5.2},
    {"GLRLM_GLNN", 0.46},
    {"GLRLM_RLN", 6.12},
    {"GLRLM_RLNN", 0.492}, 
    {"GLRLM_RP", 0.627}, 
    {"GLRLM_GLV", 3.35},
    {"GLRLM_RV", 0.761},
    {"GLRLM_RE", 2.17}
};

void test_ibsi_glrlm_feature(const Feature2D& feature_, const std::string& feature_name) 
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
    s[(int)NyxSetting::IBSI].bval = true;
    //

    int feature = int(feature_);

    double total = 0;
    
    // image 1
    LR roidata;
    GLRLMFeature f;

    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    total += roidata.fvals[feature][1];
    total += roidata.fvals[feature][2];
    total += roidata.fvals[feature][3];
    
    // image 2
    LR roidata1;
    GLRLMFeature f1;

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];
    total += roidata1.fvals[feature][1];
    total += roidata1.fvals[feature][2];
    total += roidata1.fvals[feature][3];
    
    // image 3
    LR roidata2;
    GLRLMFeature f2;

    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];
    total += roidata2.fvals[feature][1];
    total += roidata2.fvals[feature][2];
    total += roidata2.fvals[feature][3];
    
    // image 4
    LR roidata3;
    GLRLMFeature f3;

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += roidata3.fvals[feature][0];
    total += roidata3.fvals[feature][1];
    total += roidata3.fvals[feature][2];
    total += roidata3.fvals[feature][3];

    // Verdict
    ASSERT_TRUE(agrees_gt(total/16, IBSI_glrlm_values[feature_name], 100.));
}

void test_ibsi_glrlm_sre()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_SRE, "GLRLM_SRE");
}

void test_ibsi_glrlm_lre()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_LRE, "GLRLM_LRE");
}

void test_ibsi_glrlm_lglre()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_LGLRE, "GLRLM_LGLRE");
}

void test_ibsi_glrlm_hglre()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_HGLRE, "GLRLM_HGLRE");
}   

void test_ibsi_glrlm_srlgle()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_SRLGLE, "GLRLM_SRLGLE");
}

void test_ibsi_glrlm_srhgle()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_SRHGLE, "GLRLM_SRHGLE");
}

void test_ibsi_glrlm_lrlgle()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_LRLGLE, "GLRLM_LRLGLE");
}

void test_ibsi_glrlm_lrhgle()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_LRHGLE, "GLRLM_LRHGLE");
}

void test_ibsi_glrlm_gln()
{   
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_GLN, "GLRLM_GLN");
}

void test_ibsi_glrlm_glnn()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_GLNN, "GLRLM_GLNN");
}

void test_ibsi_glrlm_rln()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_RLN, "GLRLM_RLN");
}

void test_ibsi_glrlm_rlnn()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_RLNN, "GLRLM_RLNN");
}

void test_ibsi_glrlm_rp()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_RP, "GLRLM_RP");
}

void test_ibsi_glrlm_glv()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_GLV, "GLRLM_GLV");
}

void test_ibsi_glrlm_rv()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_RV, "GLRLM_RV");
}

void test_ibsi_glrlm_re()
{
    test_ibsi_glrlm_feature(Nyxus::Feature2D::GLRLM_RE, "GLRLM_RE");
}