#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/glszm.h"
#include "../src/nyx/features/pixel.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
static std::unordered_map<std::string, float> IBSI_glszm_values {
    {"GLSZM_SAE", 0.363},
    {"GLSZM_LAE", 43.9},
    {"GLSZM_LGLZE", 0.371},
    {"GLSZM_HGLZE", 16.4},
    {"GLSZM_SALGLE", 0.0259},
    {"GLSZM_SAHGLE", 10.3},
    {"GLSZM_LALGLE", 40.4},
    {"GLSZM_LAHGLE", 113},
    {"GLSZM_GLN", 1.41},
    {"GLSZM_GLNN", 0.323},
    {"GLSZM_SZN", 1.49},
    {"GLSZM_SZNN", 0.333},
    {"GLSZM_ZP", 0.24},
    {"GLSZM_GLV", 3.97},
    {"GLSZM_ZV", 21},
    {"GLSZM_ZE", 1.93}
};

void test_ibsi_glszm_feature(const AvailableFeatures& feature, const std::string& feature_name) 
{

    double total = 0;
    
    LR roidata;
    // Calculate features
    GLSZMFeature f;
    Environment::ibsi_compliance = true;

    // image 1
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];;
    
    // image 2
    // Calculate features
    LR roidata1;
    // Calculate features
    GLSZMFeature f1;
    Environment::ibsi_compliance = true;

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
    GLSZMFeature f2;
    Environment::ibsi_compliance = true;

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
    GLSZMFeature f3;
    Environment::ibsi_compliance = true;

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    total += roidata3.fvals[feature][0];

    ASSERT_TRUE(agrees_gt(total/4, IBSI_glszm_values[feature_name], 100.));

}

void test_ibsi_glszm_sae()
{
    test_ibsi_glszm_feature(GLSZM_SAE, "GLSZM_SAE");
}

void test_ibsi_glszm_lae()
{
    test_ibsi_glszm_feature(GLSZM_LAE, "GLSZM_LAE");
}

void test_ibsi_glszm_lglze()
{
    test_ibsi_glszm_feature(GLSZM_LGLZE, "GLSZM_LGLZE");
}


void test_ibsi_glszm_hglze()
{
    test_ibsi_glszm_feature(GLSZM_HGLZE, "GLSZM_HGLZE");
}

void test_ibsi_glszm_salgle()
{
    test_ibsi_glszm_feature(GLSZM_SALGLE, "GLSZM_SALGLE");
}

void test_ibsi_glszm_sahgle()
{  
    test_ibsi_glszm_feature(GLSZM_SAHGLE, "GLSZM_SAHGLE");
}

void test_ibsi_glszm_lalgle()
{
    test_ibsi_glszm_feature(GLSZM_LALGLE, "GLSZM_LALGLE");
}

void test_ibsi_glszm_lahgle()
{
    test_ibsi_glszm_feature(GLSZM_LAHGLE, "GLSZM_LAHGLE");
}

void test_ibsi_glszm_gln()
{
    test_ibsi_glszm_feature(GLSZM_GLN, "GLSZM_GLN");
}

void test_ibsi_glszm_glnn()
{
    test_ibsi_glszm_feature(GLSZM_GLNN, "GLSZM_GLNN");
}

void test_ibsi_glszm_szn()
{
    test_ibsi_glszm_feature(GLSZM_SZN, "GLSZM_SZN");
}

void test_ibsi_glszm_sznn()
{
    test_ibsi_glszm_feature(GLSZM_SZNN, "GLSZM_SZNN");
}

void test_ibsi_glszm_zp()
{
    test_ibsi_glszm_feature(GLSZM_ZP, "GLSZM_ZP");
}

void test_ibsi_glszm_glv()
{
    test_ibsi_glszm_feature(GLSZM_GLV, "GLSZM_GLV");
}

void test_ibsi_glszm_zv()
{
    test_ibsi_glszm_feature(GLSZM_ZV, "GLSZM_ZV");
}

void test_ibsi_glszm_ze()
{
    test_ibsi_glszm_feature(GLSZM_ZE, "GLSZM_ZE");
}