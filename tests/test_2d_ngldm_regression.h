#pragma once

// NGLDM_GLM and NGLDM_DCM are Nyxus mean-style rows with no counterpart in the IBSI NGLDM table, so
// they cannot be IBSI-vetted and their goldens are pinned Nyxus output. They lived in
// test_2d_ngldm_ibsi.h next to the genuine IBSI consensus values; SPEC 2 keeps one kind per file, so
// they moved here with the snapshot table they read. The fixture they run on is shared through
// test_2d_ngldm_common.h, so this file no longer reaches it by including an oracle file.

#include "test_2d_ngldm_common.h"   // gtest, <string>, assert_ngldm_feature_against_golden_values
#include "test_ref_vals.h"          // ref_vals_map

// GLM and DCM are Nyxus mean-style rows that are not defined in the IBSI
// NGLDM table, so only those two remain local regression references here.
static const ref_vals_map<double> ngldm_2d_regression_ref_vals
{
	{"NGLDM_GLM",		2.1178319573443410e+00},
	{"NGLDM_DCM",		3.9832043343653250e+00}
};

// frac_tolerance = 1e9, i.e. rel=1e-9. These are Nyxus' own output pinned to full precision, so a
// drift guard should catch any change at all; the shared helper's old 50% band could not.
void assert_ngldm_feature_regression(const Feature2D& feature_, const std::string& feature_name)
{
	assert_ngldm_feature_against_golden_values(
		feature_,
		feature_name,
		ngldm_2d_regression_ref_vals,
		"REGRESSION__",
		1e9);
}

void test_2d_ngldm_glm_regression()
{
	assert_ngldm_feature_regression(Nyxus::Feature2D::NGLDM_GLM, "NGLDM_GLM");
}

void test_2d_ngldm_dcm_regression()
{
	assert_ngldm_feature_regression(Nyxus::Feature2D::NGLDM_DCM, "NGLDM_DCM");
}

// The NGLD-matrix with IBSI mode OFF, against a third-party matrix rather than Nyxus' own output.
//
// This lived in test_2d_ngldm_ibsi.h as test_2d_ngldm_matrix_correctness_nonibsi_mode_ibsi(), where
// nothing about it was IBSI: its ground truth is the NGLDM worked out in a StackOverflow answer
// (test_data.h names the URL beside the image), and it runs the non-IBSI mode, which is the mode the
// IBSI definition does not describe. The `_ibsi` suffix therefore claimed an oracle the assertion
// does not use.
//
// It is regression rather than an oracle test because the reference cannot be regenerated here: a
// forum post is not a tool this tree can run, there is no version to record and no generator to
// write (SPEC 6.4). What it does do is hold the non-IBSI matrix -- unique grey tones rather than the
// 0..max IBSI range -- to the shape a second implementation arrived at, so a change in the
// dependence counting is caught. Making it an oracle claim means running MATLAB or Octave and
// pinning what THAT produces, with a generator beside it.
void assert_ngldm_matrix_nonibsi_mode()
{
    // Load a test image
    LR roidata;
    load_masked_test_roi_data (roidata, nonibsi_rayryeng_ngldm_sample_image_int, nonibsi_rayryeng_ngldm_sample_image_mask, sizeof(nonibsi_rayryeng_ngldm_sample_image_mask) / sizeof(NyxusPixel));

    // In this test, we only calculate and examine the NGLD-matrix without calculating features
    NGLDMfeature f;

    // featue settings for this particular test
    Fsettings s = make_ngldm2d_settings(false);

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

void test_2d_ngldm_matrix_correctness_nonibsi_mode_regression()
{
    assert_ngldm_matrix_nonibsi_mode();
}
