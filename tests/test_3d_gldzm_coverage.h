#pragma once

#include "test_3d_coverage_common.h"

// Per-family slice of the 3D coverage sweep (Wave 9). The shared harness, the two parameterized
// fixtures, their TEST_P bodies, and the global count-guard live in test_3d_coverage_common.h; this
// file only re-instantiates the two suites for the "gldzm" family. Every public 3D feature is
// classified into exactly one family (first-match on the calculator featuresets), so the per-family
// instantiations together reproduce the original 94-embedded + 119-unvetted split with no drift.

INSTANTIATE_TEST_SUITE_P(
	GLDZM_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("gldzm", true)),
	sanitize_3d_feature_test_name);

INSTANTIATE_TEST_SUITE_P(
	GLDZM_UNVETTED_LOCAL_REGRESSION,
	Test3DFeature_UNVETTED_LOCAL_REGRESSION,
	testing::ValuesIn(feature_3d_cases_for_family("gldzm", false)),
	sanitize_3d_feature_test_name);

// Regression baselines for this family's slice of the sweep: pinned Nyxus output for the public 3D
// gldzm features that no third-party oracle backs yet. They establish no vetting (SPEC 1), and they
// live here rather than in the shared harness so the table sits with the assertions that read it.
static const ref_vals_map<std::vector<double>> gldzm_3d_regression_coverage_ref_vals
{
	{ "3GLDZM_GLM", { 47.230300235279401 } },
	{ "3GLDZM_GLNU", { 3435.1800942680934 } },
	{ "3GLDZM_GLNUN", { 0.026851399515903585 } },
	{ "3GLDZM_GLV", { 111.77220626552923 } },
	{ "3GLDZM_HGLZE", { 2342.4734665801629 } },
	{ "3GLDZM_LDE", { 314.01248309662088 } },
	{ "3GLDZM_LDHGLE", { 734618.35720259824 } },
	{ "3GLDZM_LDLGLE", { 0.16729167507144088 } },
	{ "3GLDZM_LGLZE", { 0.0005581993242951194 } },
	{ "3GLDZM_SDE", { 0.022387420258025731 } },
	{ "3GLDZM_SDHGLE", { 61.230746106573264 } },
	{ "3GLDZM_SDLGLE", { 1.8362515436029654e-05 } },
	{ "3GLDZM_ZDE", { 10.230312642315168 } },
	{ "3GLDZM_ZDM", { 15.306504185784746 } },
	{ "3GLDZM_ZDNU", { 4330.2817177741472 } },
	{ "3GLDZM_ZDNUN", { 0.033848043255251946 } },
	{ "3GLDZM_ZDV", { 79.723412707174901 } },
	{ "3GLDZM_ZP", { 0.46617376982276121 } },
};

static const bool gldzm_3d_coverage_baseline_registered =
	register_coverage_baseline("gldzm", &gldzm_3d_regression_coverage_ref_vals);
