#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/glszm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
// Calculated at 100 grey levels
static std::unordered_map<std::string, float> glszm_values {
    {"GLSZM_SAE", 0.38873},
    {"GLSZM_LAE", 32.5},
    {"GLSZM_LGLZE", 0.1962550},
    {"GLSZM_HGLZE", 1497.574999},
    {"GLSZM_SALGLE", 0.109386},
    {"GLSZM_SAHGLE", 881.470652},
    {"GLSZM_LALGLE", 0.769358},
    {"GLSZM_LAHGLE", 11048.97500},
    {"GLSZM_GLN", 1.487499999},
    {"GLSZM_GLNN", 0.2771875000},
    {"GLSZM_SZN", 1.96249999},
    {"GLSZM_SZNN", 0.35968749999},
    {"GLSZM_ZP", 0.2750000},
    {"GLSZM_GLV", 551.70375000},
    {"GLSZM_ZV", 16.6875000000},
    {"GLSZM_ZE", 2.28429019}
};

void test_glszm_feature(const Feature2D& feature_, const std::string& feature_name) 
{
    // Set feature's state
    Environment::ibsi_compliance = false;
    GLSZMFeature::n_levels = 100;

    int feature = int(feature_);

    double total = 0;

    // image 1

    LR roidata;
    GLSZMFeature f;    
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    
    // image 2

    LR roidata1;
    GLSZMFeature f1;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];
  
    // image 3

    LR roidata2;
    GLSZMFeature f2;
    Environment::ibsi_compliance = false;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];
    
    // image 4
    
    LR roidata3;
    GLSZMFeature f3;
    Environment::ibsi_compliance = false;

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    total += roidata3.fvals[feature][0];

    // Verdict
    ASSERT_TRUE(agrees_gt(total/4, glszm_values[feature_name], 100.));
}

void test_glszm_sae()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_SAE, "GLSZM_SAE");
}

void test_glszm_lae()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_LAE, "GLSZM_LAE");
}

void test_glszm_lglze()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_LGLZE, "GLSZM_LGLZE");
}


void test_glszm_hglze()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_HGLZE, "GLSZM_HGLZE");
}

void test_glszm_salgle()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_SALGLE, "GLSZM_SALGLE");
}

void test_glszm_sahgle()
{  
    test_glszm_feature(Nyxus::Feature2D::GLSZM_SAHGLE, "GLSZM_SAHGLE");
}

void test_glszm_lalgle()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_LALGLE, "GLSZM_LALGLE");
}

void test_glszm_lahgle()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_LAHGLE, "GLSZM_LAHGLE");
}

void test_glszm_gln()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_GLN, "GLSZM_GLN");
}

void test_glszm_glnn()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_GLNN, "GLSZM_GLNN");
}

void test_glszm_szn()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_SZN, "GLSZM_SZN");
}

void test_glszm_sznn()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_SZNN, "GLSZM_SZNN");
}

void test_glszm_zp()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_ZP, "GLSZM_ZP");
}

void test_glszm_glv()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_GLV, "GLSZM_GLV");
}

void test_glszm_zv()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_ZV, "GLSZM_ZV");
}

void test_glszm_ze()
{
    test_glszm_feature(Nyxus::Feature2D::GLSZM_ZE, "GLSZM_ZE");
}