#pragma once

#include <gtest/gtest.h>

#include <string>

#include "../src/nyx/feature_settings.h"   // Fsettings, NyxSetting
#include "../src/nyx/featureset.h"         // Feature2D
#include "../src/nyx/features/glcm.h"      // GLCMFeature
#include "../src/nyx/roi_cache.h"          // LR
#include "test_data.h"                     // the IBSI phantom slices
#include "test_main_nyxus.h"               // agrees_gt, load_masked_test_roi_data
#include "test_ref_vals.h"                 // ref_vals_map

// Digital phantom values for intensity based features
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021. Dataset: dig phantom. Aggr. method: 2D, averaged)
static const ref_vals_map<double> glcm_2d_ibsi_ref_vals {
    {"GLCM_ACOR", 5.09},    // p. 76, consensus: very strong
    {"GLCM_ASM", 0.368},    // p. 68, consensus: very strong
    {"GLCM_CLUPROM", 79.1}, // p. 79, consensus: very strong
    {"GLCM_CLUSHADE", 7},   // p. 78, consensus: very strong
    {"GLCM_CLUTEND", 5.47}, // p. 78, consensus: very strong
    {"GLCM_CONTRAST", 5.28},    // p. 69, consensus: very strong
    {"GLCM_CORRELATION", -0.0121},  // p. 76, consensus: very strong
    {"GLCM_DIFAVE", 1.42},  // p. 64, consensus: very strong
    {"GLCM_DIFENTRO", 1.4}, // p. 65, consensus: very strong
    {"GLCM_DIFVAR", 2.9},   // p. 65, consensus: very strong
    {"GLCM_DIS", 1.42},     // p. 70, consensus: very strong
    {"GLCM_ID", 0.678},     // p. 71, consensus: very strong
    {"GLCM_IDN", 0.851},    // p. 72, consensus: very strong
    {"GLCM_IDM", 0.619},    // p. 73, consensus: very strong
    {"GLCM_IDMN", 0.899},   // p. 74, consensus: very strong
    {"GLCM_INFOMEAS1", -0.155}, // p. 80, consensus: very strong
    {"GLCM_INFOMEAS2", 0.487},  // p. 81, consensus: very strong
    {"GLCM_HOM2", 0.619},       // = IBSI IDM (WF0Z, p.73); PyRadiomics 'Idm' twin of GLCM_IDM
    {"GLCM_ENTROPY", 2.05},     // = IBSI JE (TU9B, p.63); joint entropy twin of GLCM_JE
    {"GLCM_IV", 0.0567},    // p. 75, consensus: very strong
    {"GLCM_JAVE", 2.14},    // p. 62, consensus: very strong
    {"GLCM_JE", 2.05},      // p. 63, consensus: very strong
    {"GLCM_JMAX", 0.519},   // p. 61, consensus: very strong
    {"GLCM_JVAR", 2.69},    // p. 63, consensus: very strong
    {"GLCM_SUMAVERAGE", 4.28},  // p. 66, consensus: very strong
    {"GLCM_SUMENTROPY", 1.6},   // p. 67, consensus: very strong
    {"GLCM_SUMVARIANCE", 5.47},  // p. 67, consensus: very strong
    {"GLCM_VARIANCE", 2.69}     // = IBSI JVAR (UR99, p.63): IBSI has no separate "variance", and Nyxus
                                // computes GLCM_VARIANCE as the joint variance about the grey-level
                                // marginal mean - a different routine from GLCM_JVAR reaching the same
                                // quantity (measured: both 2.687695905 here)
};

// An _AVE feature is the mean of its base feature over the 4 angles, and the IBSI consensus values
// above are themselves reported for the "2D, averaged" aggregation, so an _AVE feature and its base
// are checked against the same golden - the base by averaging the 4 angled values here, the _AVE by
// reading the single value the feature itself aggregated.
static double glcm_ibsi_roi_total(const LR& roidata, int feature, bool is_ave_feature)
{
    // The _AVE feature has already averaged the angles, so it occupies a single slot
    if (is_ave_feature)
        return roidata.fvals[feature][0];

    return roidata.fvals[feature][0] + roidata.fvals[feature][1] +
        roidata.fvals[feature][2] + roidata.fvals[feature][3];
}

static std::string glcm_ibsi_golden_key(const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    if (feature_name.size() > ave_suffix.size() &&
        feature_name.compare(feature_name.size() - ave_suffix.size(), ave_suffix.size(), ave_suffix) == 0)
        return feature_name.substr(0, feature_name.size() - ave_suffix.size());

    return feature_name;
}

void assert_glcm_feature_ibsi(const Nyxus::Feature2D& feature_, const std::string& feature_name)
{
    // featue settings for this particular test
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 0; // needs to be ==0 in the IBSI mode
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = true;   // activate the IBSI compliance mode

    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 0;   // needs to be ==0 in the IBSI mode
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;   // important

    //
    
    int feature = int(feature_);

    // A key absent from the table would otherwise be default-inserted as 0, and agrees_gt turns a
    // ground truth of 0 into a tolerance of 0 - an assertion that passes only on an exact 0.
    const std::string golden_key = glcm_ibsi_golden_key(feature_name);
    auto golden_it = glcm_2d_ibsi_ref_vals.find(golden_key);
    ASSERT_TRUE(golden_it != glcm_2d_ibsi_ref_vals.end());
    const double golden = golden_it->second;
    const bool is_ave_feature = golden_key != feature_name;

    double total = 0;

    // image 1
    LR roidata;
    GLCMFeature f;
    GLCMFeature::angles = { 0, 45, 90, 135 };
    Nyxus::load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    total += glcm_ibsi_roi_total(roidata, feature, is_ave_feature);

    // image 2

    LR roidata1;
    GLCMFeature f1;
    GLCMFeature::angles = {0, 45, 90, 135};

    Nyxus::load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    total += glcm_ibsi_roi_total(roidata1, feature, is_ave_feature);

    // image 3

    LR roidata2;
    GLCMFeature f2;
    GLCMFeature::angles = {0, 45, 90, 135};
    Nyxus::load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += glcm_ibsi_roi_total(roidata2, feature, is_ave_feature);

    // image 4
    
    LR roidata3;
    GLCMFeature f3;
    GLCMFeature::angles = {0, 45, 90, 135};
    Nyxus::load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    total += glcm_ibsi_roi_total(roidata3, feature, is_ave_feature);

    // Verdict: 4 slices x 4 angles for a base feature, 4 slices for an already-averaged _AVE one
    const double divisor = is_ave_feature ? 4. : 16.;
    ASSERT_TRUE(Nyxus::agrees_gt (total / divisor, golden, 100.));
}

void test_2d_glcm_acor_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ACOR, "GLCM_ACOR");
}

void test_2d_glcm_cluprom_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CLUPROM, "GLCM_CLUPROM");
}

void test_2d_glcm_clushade_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CLUSHADE, "GLCM_CLUSHADE");
}

void test_2d_glcm_clutend_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CLUTEND, "GLCM_CLUTEND");
}

void test_2d_glcm_difference_average_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIFAVE, "GLCM_DIFAVE");
}

void test_2d_glcm_difference_entropy_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIFENTRO, "GLCM_DIFENTRO");
}

void test_2d_glcm_difference_variance_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIFVAR, "GLCM_DIFVAR");
}

void test_2d_glcm_dis_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIS, "GLCM_DIS");
}

void test_2d_glcm_id_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ID, "GLCM_ID");
}

void test_2d_glcm_idn_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IDN, "GLCM_IDN");
}

void test_2d_glcm_idm_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IDM, "GLCM_IDM");
}

void test_2d_glcm_idmn_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IDMN, "GLCM_IDMN");
}

void test_2d_glcm_asm_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ASM, "GLCM_ASM");
}

void test_2d_glcm_contrast_ibsi()
{
   assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CONTRAST, "GLCM_CONTRAST");
}

void test_2d_glcm_correlation_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CORRELATION, "GLCM_CORRELATION");
}

void test_2d_glcm_infomeas1_ibsi()
{
   assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_INFOMEAS1, "GLCM_INFOMEAS1");
}

void test_2d_glcm_infomeas2_ibsi()
{
   assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_INFOMEAS2, "GLCM_INFOMEAS2");
}

void test_2d_glcm_iv_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IV, "GLCM_IV");
}

void test_2d_glcm_jave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JAVE, "GLCM_JAVE");
}

void test_2d_glcm_hom2_ibsi()   // regression-fix: HOM2 == IBSI IDM once /sum_p normalization is applied
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_HOM2, "GLCM_HOM2");
}

void test_2d_glcm_entropy_ibsi()   // regression-fix: ENTROPY == IBSI JE once /sum_p normalization is applied
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ENTROPY, "GLCM_ENTROPY");
}

void test_2d_glcm_je_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JE, "GLCM_JE");
}

void test_2d_glcm_jmax_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JMAX, "GLCM_JMAX");
}

void test_2d_glcm_jvar_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JVAR, "GLCM_JVAR");
}

void test_2d_glcm_sum_average_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_SUMAVERAGE, "GLCM_SUMAVERAGE");
}

void test_2d_glcm_sum_entropy_ibsi()
{
   assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_SUMENTROPY, "GLCM_SUMENTROPY");
}

// The _AVE features below are what Nyxus reports for the "2D, averaged" aggregation the IBSI
// consensus values are quoted at, so each is checked against its base feature's golden. Without
// these the angle-averaging step itself is asserted nowhere: the checks above read the 4 angled
// values and average them in the test, never touching the value the feature aggregated.

void test_2d_glcm_acor_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ACOR_AVE, "GLCM_ACOR_AVE");
}

void test_2d_glcm_asm_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ASM_AVE, "GLCM_ASM_AVE");
}

void test_2d_glcm_contrast_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CONTRAST_AVE, "GLCM_CONTRAST_AVE");
}

void test_2d_glcm_correlation_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CORRELATION_AVE, "GLCM_CORRELATION_AVE");
}

void test_2d_glcm_idmn_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IDMN_AVE, "GLCM_IDMN_AVE");
}

void test_2d_glcm_idn_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IDN_AVE, "GLCM_IDN_AVE");
}

void test_2d_glcm_sum_average_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_SUMAVERAGE_AVE, "GLCM_SUMAVERAGE_AVE");
}

void test_2d_glcm_cluprom_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CLUPROM_AVE, "GLCM_CLUPROM_AVE");
}

void test_2d_glcm_clushade_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CLUSHADE_AVE, "GLCM_CLUSHADE_AVE");
}

void test_2d_glcm_clutend_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_CLUTEND_AVE, "GLCM_CLUTEND_AVE");
}

void test_2d_glcm_difference_average_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIFAVE_AVE, "GLCM_DIFAVE_AVE");
}

void test_2d_glcm_difference_entropy_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIFENTRO_AVE, "GLCM_DIFENTRO_AVE");
}

void test_2d_glcm_difference_variance_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIFVAR_AVE, "GLCM_DIFVAR_AVE");
}

void test_2d_glcm_dis_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_DIS_AVE, "GLCM_DIS_AVE");
}

void test_2d_glcm_entropy_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ENTROPY_AVE, "GLCM_ENTROPY_AVE");
}

void test_2d_glcm_id_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_ID_AVE, "GLCM_ID_AVE");
}

void test_2d_glcm_idm_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IDM_AVE, "GLCM_IDM_AVE");
}

void test_2d_glcm_infomeas1_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_INFOMEAS1_AVE, "GLCM_INFOMEAS1_AVE");
}

void test_2d_glcm_infomeas2_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_INFOMEAS2_AVE, "GLCM_INFOMEAS2_AVE");
}

void test_2d_glcm_iv_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_IV_AVE, "GLCM_IV_AVE");
}

void test_2d_glcm_jave_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JAVE_AVE, "GLCM_JAVE_AVE");
}

void test_2d_glcm_je_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JE_AVE, "GLCM_JE_AVE");
}

void test_2d_glcm_jmax_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JMAX_AVE, "GLCM_JMAX_AVE");
}

void test_2d_glcm_jvar_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_JVAR_AVE, "GLCM_JVAR_AVE");
}

void test_2d_glcm_sum_entropy_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_SUMENTROPY_AVE, "GLCM_SUMENTROPY_AVE");
}

void test_2d_glcm_sum_variance_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_SUMVARIANCE_AVE, "GLCM_SUMVARIANCE_AVE");
}

void test_2d_glcm_variance_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_VARIANCE, "GLCM_VARIANCE");
}

void test_2d_glcm_variance_ave_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_VARIANCE_AVE, "GLCM_VARIANCE_AVE");
}

void test_2d_glcm_sum_variance_ibsi()
{
    assert_glcm_feature_ibsi(Nyxus::Feature2D::GLCM_SUMVARIANCE, "GLCM_SUMVARIANCE");
}
