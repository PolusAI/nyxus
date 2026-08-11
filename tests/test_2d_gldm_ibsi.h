#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/gldm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include "test_ref_vals.h"

// dig. phantom values for intensity based features
static ref_vals_map<double> gldm_2d_ibsi_ref_vals {
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


void assert_gldm_feature_ibsi(const Feature2D& feature_, const std::string& feature_name) 
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
    s[(int)NyxSetting::IBSI].bval = true;   // activate the IBSI compliance mode
    //
    
    int feature = int(feature_);

    double total = 0;
    
    // image 1

    LR roidata;
    GLDMFeature f;
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += roidata.fvals[feature][0];
    
    // image 2

    LR roidata1;
    GLDMFeature f1;
    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += roidata1.fvals[feature][0];

    // image 3

    LR roidata2;
    GLDMFeature f2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];
    
    // image 4

    LR roidata3;
    GLDMFeature f3;
    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += roidata3.fvals[feature][0];

    // Verdict
    ASSERT_TRUE(agrees_gt(total/4, gldm_2d_ibsi_ref_vals[feature_name], 100.));
}

void test_2d_gldm_sde_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_SDE, "GLDM_SDE");
}

void test_2d_gldm_lde_ibsi()
{
   assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_LDE, "GLDM_LDE");
}

void test_2d_gldm_lgle_ibsi()
{
   assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_LGLE, "GLDM_LGLE");
}

void test_2d_gldm_hgle_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_HGLE, "GLDM_HGLE");
}

void test_2d_gldm_sdlgle_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_SDLGLE, "GLDM_SDLGLE");
}

void test_2d_gldm_sdhgle_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_SDHGLE, "GLDM_SDHGLE");
}

void test_2d_gldm_ldlgle_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_LDLGLE, "GLDM_LDLGLE");
}

void test_2d_gldm_ldhgle_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_LDHGLE, "GLDM_LDHGLE");
}

void test_2d_gldm_gln_ibsi()
{
   assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_GLN, "GLDM_GLN");
}

void test_2d_gldm_dn_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_DN, "GLDM_DN");
}

void test_2d_gldm_dnn_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_DNN, "GLDM_DNN");
}

void test_2d_gldm_glv_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_GLV, "GLDM_GLV");
}

void test_2d_gldm_dv_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_DV, "GLDM_DV");
}

void test_2d_gldm_de_ibsi()
{
    assert_gldm_feature_ibsi(Nyxus::Feature2D::GLDM_DE, "GLDM_DE");
}
