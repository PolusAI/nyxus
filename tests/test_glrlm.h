#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/glrlm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
static std::unordered_map<std::string, float> glrlm_values {
    {"GLRLM_SRE", 0.875714},
    {"GLRLM_LRE", 1.57431},
    {"GLRLM_LGLRE", 0.493811},
    {"GLRLM_HGLRE", 4.68802},
    {"GLRLM_SRLGLE", 0.412494},
    {"GLRLM_SRHGLE", 4.27033},
    {"GLRLM_LRLGLE", 0.86152},
    {"GLRLM_LRHGLE", 6.57483},
    {"GLRLM_GLN", 2.80937},
    {"GLRLM_GLNN", 0.417335},
    {"GLRLM_RLN", 5.47778},
    {"GLRLM_RLNN", 0.829799}, 
    {"GLRLM_RP", 0.348317}, 
    {"GLRLM_GLV", 0.630247},
    {"GLRLM_RV", 0.114035},
    {"GLRLM_RE", 1.58115}
};

void test_glrlm_feature(const Feature2D& feature_, const std::string& feature_name) {
    int feature = int(feature_);

    double total = 0;
    
    LR roidata;
    // Calculate features
    GLRLMFeature f;
    Environment::ibsi_compliance = false; 
    // image 1

    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    total += roidata.fvals[feature][1];
    total += roidata.fvals[feature][2];
    total += roidata.fvals[feature][3];
    
    // image 2
    // Calculate features
    LR roidata1;
    // Calculate features
    GLRLMFeature f1;

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];
    total += roidata1.fvals[feature][1];
    total += roidata1.fvals[feature][2];
    total += roidata1.fvals[feature][3];
    
    
    // image 3
    // Calculate features

    LR roidata2;
    // Calculate features
    GLRLMFeature f2;

    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];
    total += roidata2.fvals[feature][1];
    total += roidata2.fvals[feature][2];
    total += roidata2.fvals[feature][3];
    
    // image 4
    // Calculate features
    
    LR roidata3;
    // Calculate features
    GLRLMFeature f3;

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += roidata3.fvals[feature][0];
    total += roidata3.fvals[feature][1];
    total += roidata3.fvals[feature][2];
    total += roidata3.fvals[feature][3];

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