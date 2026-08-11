#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glszm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include "test_ref_vals.h"

// dig. phantom values for intensity based features
// Calculated at 100 grey levels
static ref_vals_map<double> glszm_2d_regression_ref_vals {
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

void assert_glszm_feature_regression(const Feature2D& feature_, const std::string& feature_name) 
{
    // featue settings for this particular test
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 64;   // important
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    int feature = int(feature_);

    double total = 0;

    // image 1

    LR roidata;
    GLSZMFeature f;    
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    
    // image 2

    LR roidata1;
    GLSZMFeature f1;

    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];
  
    // image 3

    LR roidata2;
    GLSZMFeature f2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];
    
    // image 4
    
    LR roidata3;
    GLSZMFeature f3;
    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    total += roidata3.fvals[feature][0];

    // Verdict
    ASSERT_TRUE(agrees_gt(total/4, glszm_2d_regression_ref_vals[feature_name], 100.));
}

void test_2d_glszm_sae_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_SAE, "GLSZM_SAE");
}

void test_2d_glszm_lae_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_LAE, "GLSZM_LAE");
}

void test_2d_glszm_lglze_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_LGLZE, "GLSZM_LGLZE");
}


void test_2d_glszm_hglze_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_HGLZE, "GLSZM_HGLZE");
}

void test_2d_glszm_salgle_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_SALGLE, "GLSZM_SALGLE");
}

void test_2d_glszm_sahgle_regression()
{  
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_SAHGLE, "GLSZM_SAHGLE");
}

void test_2d_glszm_lalgle_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_LALGLE, "GLSZM_LALGLE");
}

void test_2d_glszm_lahgle_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_LAHGLE, "GLSZM_LAHGLE");
}

void test_2d_glszm_gln_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_GLN, "GLSZM_GLN");
}

void test_2d_glszm_glnn_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_GLNN, "GLSZM_GLNN");
}

void test_2d_glszm_szn_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_SZN, "GLSZM_SZN");
}

void test_2d_glszm_sznn_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_SZNN, "GLSZM_SZNN");
}

void test_2d_glszm_zp_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_ZP, "GLSZM_ZP");
}

void test_2d_glszm_glv_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_GLV, "GLSZM_GLV");
}

void test_2d_glszm_zv_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_ZV, "GLSZM_ZV");
}

void test_2d_glszm_ze_regression()
{
    assert_glszm_feature_regression(Nyxus::Feature2D::GLSZM_ZE, "GLSZM_ZE");
}