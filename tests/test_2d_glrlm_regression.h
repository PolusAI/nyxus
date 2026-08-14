#pragma once

#include <gtest/gtest.h>

#include <string>

#include "../src/nyx/feature_settings.h"   // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"         // Feature2D
#include "../src/nyx/features/glrlm.h"     // GLRLMFeature
#include "../src/nyx/roi_cache.h"          // LR
#include "test_data.h"                     // the IBSI phantom slices
#include "test_main_nyxus.h"               // agrees_gt, load_masked_test_roi_data
#include "test_ref_vals.h"                 // ref_vals_map

// dig. phantom values for intensity based features
// Calculated at 100 grey levels
static ref_vals_map<double> glrlm_2d_regression_ref_vals {
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

static std::string unvetted_nyxus_regression_glrlm_feature_golden_key(const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    if (feature_name.size() > ave_suffix.size() &&
        feature_name.compare(feature_name.size() - ave_suffix.size(), ave_suffix.size(), ave_suffix) == 0)
        return feature_name.substr(0, feature_name.size() - ave_suffix.size());

    return feature_name;
}

void assert_glrlm_feature_regression(const Nyxus::Feature2D& feature_, const std::string& feature_name) 
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
    GLRLMFeature::n_levels = 100;

    int feature = int(feature_);
    const std::string truth_key = unvetted_nyxus_regression_glrlm_feature_golden_key(feature_name);
    ASSERT_TRUE(glrlm_2d_regression_ref_vals.count(truth_key) > 0);
    const bool is_ave_feature = truth_key != feature_name;

    double total = 0;

    // image 1
    LR roidata;
    GLRLMFeature f;
    Nyxus::load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    if (is_ave_feature)
        total += roidata.fvals[feature][0];
    else
    {
        total += roidata.fvals[feature][0];
        total += roidata.fvals[feature][1];
        total += roidata.fvals[feature][2];
        total += roidata.fvals[feature][3];
    }
    
    // image 2
    LR roidata1;
    GLRLMFeature f1;
    Nyxus::load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    if (is_ave_feature)
        total += roidata1.fvals[feature][0];
    else
    {
        total += roidata1.fvals[feature][0];
        total += roidata1.fvals[feature][1];
        total += roidata1.fvals[feature][2];
        total += roidata1.fvals[feature][3];
    }
    
    // image 3

    LR roidata2;
    GLRLMFeature f2;
    Nyxus::load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    if (is_ave_feature)
        total += roidata2.fvals[feature][0];
    else
    {
        total += roidata2.fvals[feature][0];
        total += roidata2.fvals[feature][1];
        total += roidata2.fvals[feature][2];
        total += roidata2.fvals[feature][3];
    }
    
    // image 4
    
    LR roidata3;
    GLRLMFeature f3;
    Nyxus::load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    if (is_ave_feature)
        total += roidata3.fvals[feature][0];
    else
    {
        total += roidata3.fvals[feature][0];
        total += roidata3.fvals[feature][1];
        total += roidata3.fvals[feature][2];
        total += roidata3.fvals[feature][3];
    }

    // Verdict
    const double divisor = is_ave_feature ? 4.0 : 16.0;
    ASSERT_TRUE(Nyxus::agrees_gt(total / divisor, glrlm_2d_regression_ref_vals[truth_key], 100.));
}

void test_2d_glrlm_sre_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_SRE, "GLRLM_SRE");
}

void test_2d_glrlm_lre_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LRE, "GLRLM_LRE");
}

void test_2d_glrlm_lglre_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LGLRE, "GLRLM_LGLRE");
}

void test_2d_glrlm_hglre_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_HGLRE, "GLRLM_HGLRE");
}   

void test_2d_glrlm_srlgle_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_SRLGLE, "GLRLM_SRLGLE");
}

void test_2d_glrlm_srhgle_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_SRHGLE, "GLRLM_SRHGLE");
}

void test_2d_glrlm_lrlgle_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LRLGLE, "GLRLM_LRLGLE");
}

void test_2d_glrlm_lrhgle_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LRHGLE, "GLRLM_LRHGLE");
}

void test_2d_glrlm_gln_regression()
{   
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_GLN, "GLRLM_GLN");
}

void test_2d_glrlm_glnn_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_GLNN, "GLRLM_GLNN");
}

void test_2d_glrlm_rln_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RLN, "GLRLM_RLN");
}

void test_2d_glrlm_rlnn_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RLNN, "GLRLM_RLNN");
}

void test_2d_glrlm_rp_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RP, "GLRLM_RP");
}

void test_2d_glrlm_glv_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_GLV, "GLRLM_GLV");
}

void test_2d_glrlm_rv_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RV, "GLRLM_RV");
}

void test_2d_glrlm_re_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RE, "GLRLM_RE");
}

void test_2d_glrlm_sre_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_SRE_AVE, "GLRLM_SRE_AVE");
}

void test_2d_glrlm_lre_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LRE_AVE, "GLRLM_LRE_AVE");
}

void test_2d_glrlm_gln_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_GLN_AVE, "GLRLM_GLN_AVE");
}

void test_2d_glrlm_glnn_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_GLNN_AVE, "GLRLM_GLNN_AVE");
}

void test_2d_glrlm_rln_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RLN_AVE, "GLRLM_RLN_AVE");
}

void test_2d_glrlm_rlnn_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RLNN_AVE, "GLRLM_RLNN_AVE");
}

void test_2d_glrlm_rp_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RP_AVE, "GLRLM_RP_AVE");
}

void test_2d_glrlm_glv_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_GLV_AVE, "GLRLM_GLV_AVE");
}

void test_2d_glrlm_rv_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RV_AVE, "GLRLM_RV_AVE");
}

void test_2d_glrlm_re_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_RE_AVE, "GLRLM_RE_AVE");
}

void test_2d_glrlm_lglre_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LGLRE_AVE, "GLRLM_LGLRE_AVE");
}

void test_2d_glrlm_hglre_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_HGLRE_AVE, "GLRLM_HGLRE_AVE");
}

void test_2d_glrlm_srlgle_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_SRLGLE_AVE, "GLRLM_SRLGLE_AVE");
}

void test_2d_glrlm_srhgle_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_SRHGLE_AVE, "GLRLM_SRHGLE_AVE");
}

void test_2d_glrlm_lrlgle_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LRLGLE_AVE, "GLRLM_LRLGLE_AVE");
}

void test_2d_glrlm_lrhgle_ave_regression()
{
    assert_glrlm_feature_regression(Nyxus::Feature2D::GLRLM_LRHGLE_AVE, "GLRLM_LRHGLE_AVE");
}
