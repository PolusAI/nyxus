#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// Digital phantom values for intensity based features
// Calculated at 100 grey levels, offset 1, and asymmetric cooc matrix
static std::unordered_map<std::string, double> glcm_values 
{
    {"GLCM_ACOR", 1.3401234375000004e+03},
    {"GLCM_ASM", 2.8767361111111111e-01},
    {"GLCM_CLUPROM", 6.4010300458485820e+06},
    {"GLCM_CLUSHADE", 2.0646084008680562e+04},
    {"GLCM_CLUTEND", 1.5639042057291665e+03},
    {"GLCM_CONTRAST", 1.4448130208333334e+03},
    {"GLCM_CORRELATION", 1.0095524430002521e-02},
    {"GLCM_DIFAVE", 2.4330208333333330e+01},
    {"GLCM_DIFENTRO", 1.7527497019323600e+00},
    {"GLCM_DIFVAR", 7.7157956597222220e+02},
    {"GLCM_DIS", 2.4330208333333340e+01},
    {"GLCM_ENERGY", 2.8767361111111111e-01},
    {"GLCM_ENTROPY", -2.0943564580288626e+01},
    {"GLCM_HOM1", 5.1027480278491990e-01},
    {"GLCM_HOM2", 6.9449788922648480e+00},
    {"GLCM_ID", 5.1027480278491990e-01},
    {"GLCM_IDN", 8.4432100308124380e-01},
    {"GLCM_IDM", 4.9717513725531143e-01},
    {"GLCM_IDMN", 9.0029152005531590e-01},
    {"GLCM_INFOMEAS1", -2.3913067639121394e-01},
    {"GLCM_INFOMEAS2", 5.9972197335335700e-01},
    {"GLCM_IV", 5.6216708893582570e-04},
    {"GLCM_JAVE", 3.5266406250000000e+01},
    {"GLCM_JE", 2.2639111980622557e+00},
    {"GLCM_JMAX", 4.4713541666666670e-01},
    {"GLCM_JVAR", 7.9056543511284720e+02},
    {"GLCM_SUMAVERAGE", 6.8281250000000000e+01},
    {"GLCM_SUMENTROPY", 1.9554838705137936e+00},
    {"GLCM_SUMVARIANCE", 1.5639042057291665e+03},
    {"GLCM_VARIANCE", 7.9056543511284720e+02}
};

static std::string glcm_truth_key(const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    if (feature_name.size() > ave_suffix.size() &&
        feature_name.compare(feature_name.size() - ave_suffix.size(), ave_suffix.size(), ave_suffix) == 0)
        return feature_name.substr(0, feature_name.size() - ave_suffix.size());

    return feature_name;
}

void test_glcm_feature(const Feature2D& feature_, const std::string& feature_name) 
{
    // featue settings for this particular test
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;   // important
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    //

    // Set feature's state
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 100;   // important
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;   // important
    GLCMFeature::symmetric_glcm = false;
    GLCMFeature::angles = { 0, 45, 90, 135 };

    int feature = int(feature_);
    const std::string truth_key = glcm_truth_key(feature_name);
    ASSERT_TRUE(glcm_values.count(truth_key) > 0);
    const bool is_ave_feature = truth_key != feature_name;

    double total = 0;

    // image 1

     LR roidata;
    GLCMFeature f;   
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
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
    GLCMFeature f1;
    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

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
    GLCMFeature f2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

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
    GLCMFeature f3;
    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

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
    ASSERT_TRUE(agrees_gt(total / divisor, glcm_values[truth_key], 100.));
}

void test_glcm_ACOR()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ACOR, "GLCM_ACOR");
}

void test_glcm_angular_2d_moment()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ASM, "GLCM_ASM");
}

void test_glcm_CLUPROM()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUPROM, "GLCM_CLUPROM");
}

void test_glcm_CLUSHADE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUSHADE, "GLCM_CLUSHADE");
}

void test_glcm_CLUTEND()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUTEND, "GLCM_CLUTEND");
}

void test_glcm_contrast()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_CONTRAST, "GLCM_CONTRAST");
}

void test_glcm_correlation()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CORRELATION, "GLCM_CORRELATION");
}

void test_glcm_difference_average()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFAVE, "GLCM_DIFAVE");
}

void test_glcm_difference_entropy()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFENTRO, "GLCM_DIFENTRO");
}

void test_glcm_difference_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFVAR, "GLCM_DIFVAR");
}

void test_glcm_DIS()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIS, "GLCM_DIS");
}

void test_glcm_energy()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENERGY, "GLCM_ENERGY");
}

void test_glcm_entropy()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENTROPY, "GLCM_ENTROPY");
}

void test_glcm_hom1()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_HOM1, "GLCM_HOM1");
}

void test_glcm_hom2()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_HOM2, "GLCM_HOM2");
}

void test_glcm_ID()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ID, "GLCM_ID");
}

void test_glcm_IDN()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDN, "GLCM_IDN");
}

void test_glcm_IDM()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDM, "GLCM_IDM");
}

void test_glcm_IDMN()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDMN, "GLCM_IDMN");
}

void test_glcm_infomeas1()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS1, "GLCM_INFOMEAS1");
}

void test_glcm_infomeas2()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS2, "GLCM_INFOMEAS2");
}

void test_glcm_IV()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IV, "GLCM_IV");
}

void test_glcm_JAVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JAVE, "GLCM_JAVE");
}

void test_glcm_JE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JE, "GLCM_JE");
}

void test_glcm_JMAX()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JMAX, "GLCM_JMAX");
}

void test_glcm_JVAR()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JVAR, "GLCM_JVAR");
}

void test_glcm_sum_average()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMAVERAGE, "GLCM_SUMAVERAGE");
}

void test_glcm_sum_entropy()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_SUMENTROPY, "GLCM_SUMENTROPY");
}

void test_glcm_sum_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMVARIANCE, "GLCM_SUMVARIANCE");
}

void test_glcm_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_VARIANCE, "GLCM_VARIANCE");
}

void test_glcm_ASM_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ASM_AVE, "GLCM_ASM_AVE");
}

void test_glcm_ACOR_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ACOR_AVE, "GLCM_ACOR_AVE");
}

void test_glcm_CLUPROM_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUPROM_AVE, "GLCM_CLUPROM_AVE");
}

void test_glcm_CLUSHADE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUSHADE_AVE, "GLCM_CLUSHADE_AVE");
}

void test_glcm_CLUTEND_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUTEND_AVE, "GLCM_CLUTEND_AVE");
}

void test_glcm_CONTRAST_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CONTRAST_AVE, "GLCM_CONTRAST_AVE");
}

void test_glcm_CORRELATION_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CORRELATION_AVE, "GLCM_CORRELATION_AVE");
}

void test_glcm_DIFAVE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFAVE_AVE, "GLCM_DIFAVE_AVE");
}

void test_glcm_DIFENTRO_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFENTRO_AVE, "GLCM_DIFENTRO_AVE");
}

void test_glcm_DIFVAR_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFVAR_AVE, "GLCM_DIFVAR_AVE");
}

void test_glcm_DIS_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIS_AVE, "GLCM_DIS_AVE");
}

void test_glcm_ENERGY_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENERGY_AVE, "GLCM_ENERGY_AVE");
}

void test_glcm_ENTROPY_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENTROPY_AVE, "GLCM_ENTROPY_AVE");
}

void test_glcm_HOM1_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_HOM1_AVE, "GLCM_HOM1_AVE");
}

void test_glcm_ID_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ID_AVE, "GLCM_ID_AVE");
}

void test_glcm_IDN_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDN_AVE, "GLCM_IDN_AVE");
}

void test_glcm_IDM_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDM_AVE, "GLCM_IDM_AVE");
}

void test_glcm_IDMN_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDMN_AVE, "GLCM_IDMN_AVE");
}

void test_glcm_IV_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IV_AVE, "GLCM_IV_AVE");
}

void test_glcm_JAVE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JAVE_AVE, "GLCM_JAVE_AVE");
}

void test_glcm_JE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JE_AVE, "GLCM_JE_AVE");
}

void test_glcm_INFOMEAS1_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS1_AVE, "GLCM_INFOMEAS1_AVE");
}

void test_glcm_INFOMEAS2_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS2_AVE, "GLCM_INFOMEAS2_AVE");
}

void test_glcm_VARIANCE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_VARIANCE_AVE, "GLCM_VARIANCE_AVE");
}

void test_glcm_JMAX_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JMAX_AVE, "GLCM_JMAX_AVE");
}

void test_glcm_JVAR_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JVAR_AVE, "GLCM_JVAR_AVE");
}

void test_glcm_SUMAVERAGE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMAVERAGE_AVE, "GLCM_SUMAVERAGE_AVE");
}

void test_glcm_SUMENTROPY_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMENTROPY_AVE, "GLCM_SUMENTROPY_AVE");
}

void test_glcm_SUMVARIANCE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMVARIANCE_AVE, "GLCM_SUMVARIANCE_AVE");
}
