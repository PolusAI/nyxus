#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glrlm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
// Calculated at 100 grey levels
static std::unordered_map<std::string, float> glrlm_values {
    {"GLRLM_SRE", 0.677679}, 
    {"GLRLM_LRE", 3.41805}, 
    {"GLRLM_LGLRE", 0.11546}, 
    {"GLRLM_HGLRE", 2486.087}, 
    {"GLRLM_SRLGLE", 0.104}, 
    {"GLRLM_SRHGLE", 2157.737}, 
    {"GLRLM_LRLGLE", 0.165085}, 
    {"GLRLM_LRHGLE", 4464.084}, 
    {"GLRLM_GLN", 4.866}, 
    {"GLRLM_GLNN", 0.37445}, 
    {"GLRLM_RLN", 7.068975}, 
    {"GLRLM_RLNN", 0.518777}, 
    {"GLRLM_RP", 0.705}, 
    {"GLRLM_GLV", 951.70428}, 
    {"GLRLM_RV", 0.709646}, 
    {"GLRLM_RE", 2.3747}
};

void test_glrlm_feature(const Feature2D& feature_, const std::string& feature_name) 
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
    GLRLMFeature::n_levels = 100;

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
    ASSERT_TRUE(agrees_gt(total/16, glrlm_values[feature_name], 100.));
}

void test_glrlm_sre()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_SRE, "GLRLM_SRE");
}

void test_glrlm_lre()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_LRE, "GLRLM_LRE");
}

void test_glrlm_lglre()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_LGLRE, "GLRLM_LGLRE");
}

void test_glrlm_hglre()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_HGLRE, "GLRLM_HGLRE");
}   

void test_glrlm_srlgle()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_SRLGLE, "GLRLM_SRLGLE");
}

void test_glrlm_srhgle()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_SRHGLE, "GLRLM_SRHGLE");
}

void test_glrlm_lrlgle()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_LRLGLE, "GLRLM_LRLGLE");
}

void test_glrlm_lrhgle()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_LRHGLE, "GLRLM_LRHGLE");
}

void test_glrlm_gln()
{   
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_GLN, "GLRLM_GLN");
}

void test_glrlm_glnn()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_GLNN, "GLRLM_GLNN");
}

void test_glrlm_rln()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_RLN, "GLRLM_RLN");
}

void test_glrlm_rlnn()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_RLNN, "GLRLM_RLNN");
}

void test_glrlm_rp()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_RP, "GLRLM_RP");
}

void test_glrlm_glv()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_GLV, "GLRLM_GLV");
}

void test_glrlm_rv()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_RV, "GLRLM_RV");
}

void test_glrlm_re()
{
    test_glrlm_feature(Nyxus::Feature2D::GLRLM_RE, "GLRLM_RE");
}