#pragma once

#include "test_3d_coverage_common.h"

// Per-family slice of the 3D coverage sweep (Wave 9). The shared harness, the two parameterized
// fixtures, their TEST_P bodies, and the global count-guard live in test_3d_coverage_common.h; this
// file only re-instantiates the two suites for the "ngldm" family. Every public 3D feature is
// classified into exactly one family (first-match on the calculator featuresets), so the per-family
// instantiations together reproduce the original 94-embedded + 119-unvetted split with no drift.

INSTANTIATE_TEST_SUITE_P(
	NGLDM_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("ngldm", true)),
	sanitize_3d_feature_test_name);

INSTANTIATE_TEST_SUITE_P(
	NGLDM_UNVETTED_LOCAL_REGRESSION,
	Test3DFeature_UNVETTED_LOCAL_REGRESSION,
	testing::ValuesIn(feature_3d_cases_for_family("ngldm", false)),
	sanitize_3d_feature_test_name);

// Regression baselines for this family's slice of the sweep: pinned Nyxus output for the public 3D
// ngldm features that no third-party oracle backs yet. They establish no vetting (SPEC 1), and they
// live here rather than in the shared harness so the table sits with the assertions that read it.
static ref_vals_map<std::vector<double>> ngldm_3d_regression_coverage_ref_vals
{
	{ "3NGLDM_DCENE", { 0.14348407632898436 } },
	{ "3NGLDM_DCENT", { 5.2277449211654039 } },
	{ "3NGLDM_DCM", { 13.485998122653307 } },
	{ "3NGLDM_DCNU", { 85056.840050062572 } },   // dependence-count (column) marginal; distinct from GLNU's grey-level marginal
	{ "3NGLDM_DCNUN", { 0.16633455892143026 } },
	{ "3NGLDM_DCP", { 1 } },
	{ "3NGLDM_DCV", { 86.17064428912758 } },
	{ "3NGLDM_GLM", { 16.955115769712151 } },
	{ "3NGLDM_GLNU", { 115443.18172715895 } },
	{ "3NGLDM_GLNUN", { 0.22575716076180957 } },
	{ "3NGLDM_GLV", { 190.08150972702501 } },
	{ "3NGLDM_HDE", { 261.01822590738425 } },
	{ "3NGLDM_HDHGLE", { 20099.770197121401 } },
	{ "3NGLDM_HDLGLE", { 0.025201544837470152 } },
	{ "3NGLDM_HGLCE", { 740.43602941176471 } },
	{ "3NGLDM_LDE", { 0.10159976999534079 } },
	{ "3NGLDM_LDHGLE", { 73.919882197712482 } },
	{ "3NGLDM_LDLGLE", { 5.8337460459982142e-05 } },
	{ "3NGLDM_LGLCE", { 0.00035968375469422158 } },
};

static const bool ngldm_3d_coverage_baseline_registered =
	register_coverage_baseline("ngldm", &ngldm_3d_regression_coverage_ref_vals);
