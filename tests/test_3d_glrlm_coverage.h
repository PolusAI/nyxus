#pragma once

#include "test_3d_coverage_common.h"

// Per-family slice of the 3D coverage sweep (Wave 9). The shared harness, the two parameterized
// fixtures, their TEST_P bodies, and the global count-guard live in test_3d_coverage_common.h; this
// file only re-instantiates the two suites for the "glrlm" family. Every public 3D feature is
// classified into exactly one family (first-match on the calculator featuresets), so the per-family
// instantiations together reproduce the original 94-embedded + 119-unvetted split with no drift.

INSTANTIATE_TEST_SUITE_P(
	GLRLM_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("glrlm", true)),
	sanitize_3d_feature_test_name);

INSTANTIATE_TEST_SUITE_P(
	GLRLM_UNVETTED_LOCAL_REGRESSION,
	Test3DFeature_UNVETTED_LOCAL_REGRESSION,
	testing::ValuesIn(feature_3d_cases_for_family("glrlm", false)),
	sanitize_3d_feature_test_name);

// Regression baselines for this family's slice of the sweep: pinned Nyxus output for the public 3D
// glrlm features that no third-party oracle backs yet. They establish no vetting (SPEC 1), and they
// live here rather than in the shared harness so the table sits with the assertions that read it.
static const ref_vals_map<std::vector<double>> glrlm_3d_regression_coverage_ref_vals
{
	{ "3GLRLM_GLNN_AVE", { 0.030276986604527156 } },
	{ "3GLRLM_GLN_AVE", { 7866.1651094407043 } },
	{ "3GLRLM_GLV_AVE", { 295.15266194934725 } },
	{ "3GLRLM_HGLRE_AVE", { 1843.919606614327 } },
	{ "3GLRLM_LGLRE_AVE", { 0.10312612492203123 } },
	{ "3GLRLM_LRE_AVE", { 26.969695007835824 } },
	{ "3GLRLM_LRHGLE_AVE", { 3615.1714395919889 } },
	{ "3GLRLM_LRLGLE_AVE", { 24.851889404429883 } },
	{ "3GLRLM_RE_AVE", { 6.2830967278091743 } },
	{ "3GLRLM_RLNN_AVE", { 0.6876088115538187 } },
	{ "3GLRLM_RLN_AVE", { 177698.02943612196 } },
	{ "3GLRLM_RP_AVE", { 0.94009893441446613 } },
	{ "3GLRLM_RV_AVE", { 22.305382605370092 } },
	{ "3GLRLM_SRE_AVE", { 0.84624689639030182 } },
	{ "3GLRLM_SRHGLE_AVE", { 1740.5351176831964 } },
	{ "3GLRLM_SRLGLE_AVE", { 0.019094835618041379 } },
};

static const bool glrlm_3d_coverage_baseline_registered =
	register_coverage_baseline("glrlm", &glrlm_3d_regression_coverage_ref_vals);
