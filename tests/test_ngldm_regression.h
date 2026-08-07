#pragma once

// NGLDM_GLM and NGLDM_DCM are Nyxus mean-style rows with no counterpart in the IBSI NGLDM table, so
// they cannot be IBSI-vetted and their goldens are pinned Nyxus output. They lived in
// test_ngldm_ibsi.h next to the genuine IBSI consensus values; SPEC 2 keeps one kind per file, so
// they move here with the snapshot table and helper they use.

#include "test_ngldm_ibsi.h"   // shared fixture: assert_ngldm_feature_ibsi and the phantom loader

// GLM and DCM are Nyxus mean-style rows that are not defined in the IBSI
// NGLDM table, so only those two remain local regression references here.
static std::unordered_map<std::string, double> unvetted_nyxus_regression_ngldm_feature_reference_values
{
	{"NGLDM_GLM",		2.1178319573443410e+00},
	{"NGLDM_DCM",		3.9832043343653250e+00}
};

void assert_ngldm_feature_regression(const Feature2D& feature_, const std::string& feature_name)
{
	assert_ngldm_feature_against_golden_values(
		feature_,
		feature_name,
		unvetted_nyxus_regression_ngldm_feature_reference_values,
		"UNVETTED_NO_DIRECT_ORACLE__");
}

void test_ngldm_glm_regression()
{
	assert_ngldm_feature_regression(Nyxus::Feature2D::NGLDM_GLM, "NGLDM_GLM");
}

void test_ngldm_dcm_regression()
{
	assert_ngldm_feature_regression(Nyxus::Feature2D::NGLDM_DCM, "NGLDM_DCM");
}
