#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/gldm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// dig. phantom values for intensity based features
static std::unordered_map<std::string, float> IBSI_gldm_values {
    {"GLDM_SDE", 0.158},
    {"GLDM_LDE", 19.2},
    {"GLDM_LGLE", 0.702},
    {"GLDM_HGLE", 7.49},
    {"GLDM_SDLGLE", 0.0473},
    {"GLDM_SDHGLE", 3.06},
    {"GLDM_LDLGLE", 17.6},
    {"GLDM_LDHGLE", 49.5},
    {"GLDM_GLN", 10.2},
    {"GLDM_DN", 3.96},
    {"GLDM_DNN", 0.212},
    {"GLDM_GLV", 2.7},
    {"GLDM_DV", 2.73},
    {"GLDM_DE", 2.71}

};


void test_ibsi_gldm_feature(const AvailableFeatures& feature, const std::string& feature_name) {
    double total = 0;
    
    LR roidata;
    // Calculate features
    GLDMFeature f;
    Environment::ibsi_compliance = true; 

    // image 1
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];

    
    // image 2
    LR roidata1;
    // Calculate features
    GLDMFeature f1;
    Environment::ibsi_compliance = true; 

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];

    // image 3

    LR roidata2;
    // Calculate features
    GLDMFeature f2;
    Environment::ibsi_compliance = true; 

    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];
    
    // image 4

    LR roidata3;
    // Calculate features
    GLDMFeature f3;
    Environment::ibsi_compliance = true; 

    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += roidata3.fvals[feature][0];

    std::cerr << "value: " << total/4 << std::endl;

    ASSERT_TRUE(agrees_gt(total/4, IBSI_gldm_values[feature_name], 100.));
}

void test_ibsi_gldm_sde()
{
    test_ibsi_gldm_feature(GLDM_SDE, "GLDM_SDE");
}

void test_ibsi_gldm_lde()
{
   test_ibsi_gldm_feature(GLDM_LDE, "GLDM_LDE");
}

void test_ibsi_gldm_lgle()
{
   test_ibsi_gldm_feature(GLDM_SDE, "GLDM_SDE");
}

void test_ibsi_gldm_hgle()
{
    test_ibsi_gldm_feature(GLDM_HGLE, "GLDM_HGLE");
}

void test_ibsi_gldm_sdlgle()
{
    test_ibsi_gldm_feature(GLDM_SDLGLE, "GLDM_SDLGLE");    
}

void test_ibsi_gldm_sdhgle()
{
    test_ibsi_gldm_feature(GLDM_SDHGLE, "GLDM_SDHGLE");
}

void test_ibsi_gldm_ldlgle()
{
    test_ibsi_gldm_feature(GLDM_LDLGLE, "GLDM_LDLGLE");
}

void test_ibsi_gldm_ldhgle()
{
    test_ibsi_gldm_feature(GLDM_LDHGLE, "GLDM_LDHGLE");
}

void test_ibsi_gldm_gln()
{
   test_ibsi_gldm_feature(GLDM_GLN, "GLDM_GLN");
}

void test_ibsi_gldm_dn()
{
    test_ibsi_gldm_feature(GLDM_DN, "GLDM_DN");
}

void test_ibsi_gldm_dnn()
{
    test_ibsi_gldm_feature(GLDM_DNN, "GLDM_DNN");
}

void test_ibsi_gldm_glv()
{
    test_ibsi_gldm_feature(GLDM_GLV, "GLDM_GLV");
}

void test_ibsi_gldm_dv()
{
    test_ibsi_gldm_feature(GLDM_DV, "GLDM_DV");
}

void test_ibsi_gldm_de()
{
    test_ibsi_gldm_feature(GLDM_DE, "GLDM_DE");
}