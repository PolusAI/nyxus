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

// values for each feature produced by Nyxus on 01/18/23 after IBSI updates
static std::unordered_map<std::string, float> gldm_values {
    {"GLDM_SDE", 0.419444},
    {"GLDM_LDE", 4.33864},
    {"GLDM_LGLE", 0.419444},
    {"GLDM_HGLE", 4.6233},
    {"GLDM_SDLGLE", 0.15421},
    {"GLDM_SDHGLE", 2.49861},
    {"GLDM_LDLGLE", 2.83137},
    {"GLDM_LDHGLE", 14.4415},
    {"GLDM_GLN", 3.37992},
    {"GLDM_DN", 3.70606},
    {"GLDM_DNN", 0.526864},
    {"GLDM_GLV", 0.634569},
    {"GLDM_DV", 0.286155},
    {"GLDM_DE", 1.84336}

};

/// @brief Smoke test of GLDM
void test_gldm_feature(const Feature2D& feature, const std::string& feature_name) 
{
    LR roidata;

    // Calculate features
    GLDMFeature f;
    Environment::ibsi_compliance = false;

    // image pair
    load_masked_test_roi_data(roidata, cat2500_int, cat2500_seg, sizeof(cat2500_seg) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);
}

void test_gldm_sde()
{
    test_gldm_feature(GLDM_SDE, "GLDM_SDE");
}

void test_gldm_lde()
{
   test_gldm_feature(GLDM_LDE, "GLDM_LDE");
}

void test_gldm_lgle()
{
   test_gldm_feature(GLDM_SDE, "GLDM_SDE");
}

void test_gldm_hgle()
{
    test_gldm_feature(GLDM_HGLE, "GLDM_HGLE");
}

void test_gldm_sdlgle()
{
    test_gldm_feature(GLDM_SDLGLE, "GLDM_SDLGLE");    
}

void test_gldm_sdhgle()
{
    test_gldm_feature(GLDM_SDHGLE, "GLDM_SDHGLE");
}

void test_gldm_ldlgle()
{
    test_gldm_feature(GLDM_LDLGLE, "GLDM_LDLGLE");
}

void test_gldm_ldhgle()
{
    test_gldm_feature(GLDM_LDHGLE, "GLDM_LDHGLE");
}

void test_gldm_gln()
{
   test_gldm_feature(GLDM_GLN, "GLDM_GLN");
}

void test_gldm_dn()
{
    test_gldm_feature(GLDM_DN, "GLDM_DN");
}

void test_gldm_dnn()
{
    test_gldm_feature(GLDM_DNN, "GLDM_DNN");
}

void test_gldm_glv()
{
    test_gldm_feature(GLDM_GLV, "GLDM_GLV");
}

void test_gldm_dv()
{
    test_gldm_feature(GLDM_DV, "GLDM_DV");
}

void test_gldm_de()
{
    test_gldm_feature(GLDM_DE, "GLDM_DE");
}