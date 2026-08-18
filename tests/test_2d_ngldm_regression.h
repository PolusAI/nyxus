#pragma once

// NGLDM_GLM and NGLDM_DCM are Nyxus mean-style rows with no counterpart in the IBSI NGLDM table, so
// they cannot be IBSI-vetted and their goldens are pinned Nyxus output. They lived in
// test_2d_ngldm_ibsi.h next to the genuine IBSI consensus values; SPEC 2 keeps one kind per file, so
// they moved here with the snapshot table they read. The fixture they run on is shared through
// test_2d_ngldm_common.h, so this file no longer reaches it by including an oracle file.

#include <string>

#include "test_2d_ngldm_common.h"   // fixture: assert_ngldm_feature_against_golden_values
#include "test_ref_vals.h"

// GLM and DCM are Nyxus mean-style rows that are not defined in the IBSI
// NGLDM table, so only those two remain local regression references here.
static ref_vals_map<double> ngldm_2d_regression_ref_vals
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
		"UNVETTED_NO_DIRECT_ORACLE__",
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
