#pragma once

#include "test_2d_ngldm_common.h"   // gtest, <string>, test_data.h, NGLDMfeature, make_ngldm2d_settings, assert_ngldm_feature_against_golden_values
#include "test_ref_vals.h"          // ref_vals_map, and the <vector> the matrix check below declares

// Digital phantom values for intensity based features
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021. https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
//
// 17 of the family's 19 features. NGLDM_GLM (grey level mean) and NGLDM_DCM (dependence count mean)
// have no entry in the IBSI NGLDM table and no mirp column either, so they cannot be oracle-vetted;
// they are drift-pinned in test_2d_ngldm_regression.h.
//
// These are published to three significant figures, which is what sets this file's rel=1e-2
// tolerance. The full-precision digits are pinned against mirp in test_2d_ngldm_mirp.h.
static const ref_vals_map<double> ngldm_2d_ibsi_ref_vals
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
	{"NGLDM_GLV",		2.7},	// Grey level variance, p. 127
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
    Fsettings s = make_ngldm2d_settings(true);

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

// frac_tolerance = 100, i.e. rel=1e-2. The goldens above are the published IBSI consensus values,
// quoted to three significant figures, so the residual is dominated by that rounding: the measured
// worst case is 0.45% (NGLDM_GLNU, 10.2 published vs 10.2464 computed) and every other feature is
// under 0.2%. The exact digits are pinned separately against mirp in test_2d_ngldm_mirp.h.
void assert_ngldm_feature_ibsi(const Feature2D& feature_, const std::string& feature_name)
{
	assert_ngldm_feature_against_golden_values(
		feature_,
		feature_name,
		ngldm_2d_ibsi_ref_vals,
		"IBSI_ORACLE__",
		100.);
}

void test_2d_ngldm_matrix_correctness_ibsi()
{
    assert_ngldm_matrix_ibsi_mode();
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



