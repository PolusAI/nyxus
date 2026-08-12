#pragma once

#include <gtest/gtest.h>
#include <unordered_map>
#include "../src/nyx/environment.h"
#include "../src/nyx/features/ngldm.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_ref_vals.h"

// Digital phantom values for intensity based features
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021. https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
static ref_vals_map<double> ngldm_2d_ibsi_ref_vals
{
	{"NGLDM_LDE",		0.158},	// Low dependence emphasis, p.120, consensus - strong
	{"NGLDM_HDE",		19.2},	// High dependence emphasis, p.121
	{"NGLDM_LGLCE",		0.702},	// Low grey level count emphasis, p. 121
	{"NGLDM_HGLCE",		7.49},	// High grey level count emphasis, p. 122
	{"NGLDM_LDLGLE",	0.0473},	// Low dependence low grey level emphasis, p. 122
	{"NGLDM_LDHGLE",	3.06},	// Low dependence high grey level emphasis, p. 123
	{"NGLDM_HDLGLE",	17.6},	// High dependence low grey level emphasis, p. 123
	{"NGLDM_HDHGLE",	49.5},	// High dependence high grey level emphasis, 124
	{"NGLDM_GLNU",		10.2},	// Grey level non-uniformity, p. 124
	{"NGLDM_GLNUN",		0.562},	// Normalised grey level non-uniformity, p. 125
	{"NGLDM_DCNU",		3.96},	// Dependence count non-uniformity, p. 125
	{"NGLDM_DCNUN",		0.212},	// Normalised dependence count non-uniformity
	//--not in IBSI-- {"NGLDM_GLM",		-1},    // Grey level mean
	{"NGLDM_GLV",		2.7},	// Grey level variance, p. 127
	//--not in IBSI-- {"NGLDM_DCM",		-1},    // Dependency count mean
	{"NGLDM_DCP", 1.0},	    // Dependence count percentage, p. 126
	{"NGLDM_DCV",		2.73},	// Dependence count variance, p. 127
	{"NGLDM_DCENT",		2.71},	// Dependence count entropy, p. 128
	{"NGLDM_DCENE",		0.17}	// Dependence count energy, p. 128
};

//
// Tests calculating the NGLD-matrix with the IBSI mode enabled, using IBSI-provided ground truth
//

void assert_ngldm_matrix_ibsi_mode ()
{
    // Load a test image
    LR roidata;
    load_masked_test_roi_data (roidata, ibsi_fig3_19_ngldm_sample_image_int, ibsi_fig3_19_ngldm_sample_image_mask, sizeof(ibsi_fig3_19_ngldm_sample_image_mask) / sizeof(NyxusPixel));

    // In this test, we only calculate and examine the NGLD-matrix without calculating features
    NGLDMfeature f;

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

    // Have the feature object to create the NGLDM matrix kit (matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and NGLDM's dimensions)
    std::vector<PixIntens> greyLevelsLUT;
    SimpleMatrix<unsigned int> NGLDM;
    int Ng = -1,	// number of grey levels
        Nr = -1;	// maximum number of non-zero dependencies
    ASSERT_NO_THROW (f.prepare_NGLDM_matrix_kit (NGLDM, greyLevelsLUT, Ng, Nr, roidata, STNGS_NGREYS(s), STNGS_IBSI(s)));

    // Count discrepancies
    int n_mismatches = 0;
    for (int g=0; g<Ng; g++)
        for (int r = 0; r < Nr; r++)
        {
            auto ibsi_reference_matrix_value = ibsi_fig3_19_ngldm_reference_matrix[g * Nr + r];
            auto actual = NGLDM.yx(g, r);
            if (ibsi_reference_matrix_value != actual)
            {
                n_mismatches++;
                std::cout << "NGLD-matrix #1 mismatch! Expecting [g=" << g << ", r=" << r << "] = " << ibsi_reference_matrix_value << " not " << actual << "\n";
            }
        }

    ASSERT_TRUE (n_mismatches == 0);
}

//
// Tests calculating the NGLD-matrix with the IBSI mode disabled, using community-provided ground truth
//

void assert_ngldm_matrix_nonibsi_mode()
{
    // Load a test image
    LR roidata;
    load_masked_test_roi_data (roidata, nonibsi_rayryeng_ngldm_sample_image_int, nonibsi_rayryeng_ngldm_sample_image_mask, sizeof(nonibsi_rayryeng_ngldm_sample_image_mask) / sizeof(NyxusPixel));

    // In this test, we only calculate and examine the NGLD-matrix without calculating features
    NGLDMfeature f;

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

    // Have the feature object to create the NGLDM matrix kit (matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and NGLDM's dimensions)
    std::vector<PixIntens> greyLevelsLUT;
    SimpleMatrix<unsigned int> NGLDM;
    int Ng = -1,	// number of grey levels
        Nr = -1;	// maximum number of non-zero dependencies
    ASSERT_NO_THROW(f.prepare_NGLDM_matrix_kit(NGLDM, greyLevelsLUT, Ng, Nr, roidata, STNGS_NGREYS(s), STNGS_IBSI(s)));

    // Count discrepancies
    int n_mismatches = 0;
    for (int g = 0; g < Ng; g++)
        for (int r = 0; r < Nr; r++)
        {
            auto rayryeng_reference_matrix_value = nonibsi_rayryeng_ngldm_reference_matrix[g * Nr + r];
            auto actual = NGLDM.yx(g, r);
            if (rayryeng_reference_matrix_value != actual)
            {
                n_mismatches++;
                std::cout << "NGLD-matrix #2 mismatch! Expecting [g=" << g << ", r=" << r << "] = " << rayryeng_reference_matrix_value << " not " << actual << "\n";
            }
        }

    ASSERT_TRUE(n_mismatches == 0);
}

void assert_ngldm_feature_against_golden_values(
    const Feature2D& feature_,
    const std::string& feature_name,
    const std::unordered_map<std::string, double>& feature_reference_values,
    const std::string& review_prefix)
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

    int feature = int(feature_);

    SCOPED_TRACE(review_prefix + feature_name);
    ASSERT_TRUE(feature_reference_values.count(feature_name) > 0);

    double total = 0;

    //==== image 1

    // Load data (slice #1)
    LR roidata1;
    load_masked_test_roi_data (roidata1, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask, sizeof(ibsi_phantom_z1_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f1;
    ASSERT_NO_THROW (f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value (roidata1.fvals);

    total += roidata1.fvals[feature][0];

    //==== image 2

    // Load data (slice #2)
    LR roidata2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask, sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f2;
    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];

    //==== image 3

    // Load data (slice #3)
    LR roidata3;
    load_masked_test_roi_data (roidata3, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask, sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f3;
    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value (roidata3.fvals);

    total += roidata3.fvals[feature][0];

    //==== image 4

    // Load data (slice #4)
    LR roidata4;
    load_masked_test_roi_data (roidata4, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask, sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f4;
    ASSERT_NO_THROW(f4.calculate(roidata4, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata4.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f4.save_value (roidata4.fvals);

    total += roidata4.fvals[feature][0];

    // Verdict
    double aveTotal = total / 4.0;
    ASSERT_TRUE(agrees_gt(aveTotal, feature_reference_values.at(feature_name), 2.));
}

void assert_ngldm_feature_ibsi(const Feature2D& feature_, const std::string& feature_name)
{
	assert_ngldm_feature_against_golden_values(
		feature_,
		feature_name,
		ngldm_2d_ibsi_ref_vals,
		"VERIFIABLE_WITH_3P_BUILTIN_ORACLE__");
}

void test_2d_ngldm_matrix_correctness_ibsi()
{
    assert_ngldm_matrix_ibsi_mode();
}

void test_2d_ngldm_matrix_correctness_nonibsi_mode_ibsi()
{
    assert_ngldm_matrix_nonibsi_mode();
}

void test_2d_ngldm_lde_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_LDE, "NGLDM_LDE");
}

void test_2d_ngldm_hde_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_HDE, "NGLDM_HDE");
}

void test_2d_ngldm_lglce_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_LGLCE, "NGLDM_LGLCE");
}

void test_2d_ngldm_hglce_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_HGLCE, "NGLDM_HGLCE");
}

void test_2d_ngldm_ldlgle_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_LDLGLE, "NGLDM_LDLGLE");
}

void test_2d_ngldm_ldhgle_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_LDHGLE, "NGLDM_LDHGLE");
}

void test_2d_ngldm_hdlgle_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_HDLGLE, "NGLDM_HDLGLE");
}

void test_2d_ngldm_hdhgle_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_HDHGLE, "NGLDM_HDHGLE");
}

void test_2d_ngldm_glnu_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_GLNU, "NGLDM_GLNU");
}

void test_2d_ngldm_glnun_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_GLNUN, "NGLDM_GLNUN");
}

void test_2d_ngldm_dcnu_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_DCNU, "NGLDM_DCNU");
}

void test_2d_ngldm_dcnun_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_DCNUN, "NGLDM_DCNUN");
}

void test_2d_ngldm_dcp_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_DCP, "NGLDM_DCP");
}

void test_2d_ngldm_glv_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_GLV, "NGLDM_GLV");
}

void test_2d_ngldm_dcv_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_DCV, "NGLDM_DCV");
}

void test_2d_ngldm_dcent_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_DCENT, "NGLDM_DCENT");
}

void test_2d_ngldm_dcene_ibsi()
{
	assert_ngldm_feature_ibsi(Nyxus::Feature2D::NGLDM_DCENE, "NGLDM_DCENE");
}



