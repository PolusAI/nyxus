#pragma once

#include "test_3d_coverage_common.h"

// Per-family slice of the 3D coverage sweep (Wave 9). The shared harness, the two parameterized
// fixtures, their TEST_P bodies, and the global count-guard live in test_3d_coverage_common.h; this
// file only re-instantiates the two suites for the "firstorder" family. Every public 3D feature is
// classified into exactly one family (first-match on the calculator featuresets), so the per-family
// instantiations together reproduce the original 94-embedded + 119-unvetted split with no drift.

INSTANTIATE_TEST_SUITE_P(
	FIRSTORDER_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("firstorder", true)),
	sanitize_3d_feature_test_name);

INSTANTIATE_TEST_SUITE_P(
	FIRSTORDER_UNVETTED_LOCAL_REGRESSION,
	Test3DFeature_UNVETTED_LOCAL_REGRESSION,
	testing::ValuesIn(feature_3d_cases_for_family("firstorder", false)),
	sanitize_3d_feature_test_name);

// Regression baselines for this family's slice of the sweep: pinned Nyxus output for the public 3D
// firstorder features that no third-party oracle backs yet. They establish no vetting (SPEC 1), and they
// live here rather than in the shared harness so the table sits with the assertions that read it.
static ref_vals_map<std::vector<double>> firstorder_3d_regression_coverage_ref_vals
{
	{ "3COV", { 0.29486207043456802 } },
	{ "3COVERED_IMAGE_INTENSITY_RANGE", { 1.0002043207290587 } },
	{ "3EXCESS_KURTOSIS", { -1.2127631603215119 } },
	{ "3HYPERFLATNESS", { 3.8027657005973312 } },
	{ "3HYPERSKEWNESS", { 0.32001332615517414 } },
	{ "3MEDIAN_ABSOLUTE_DEVIATION", { 507.12380480410445 } },
	{ "3QCOD", { 0.25724851827174233 } },
	{ "3ROBUST_MEAN", { 1977.5189642596645 } },  // FIX: baseline was pinning the bug value 0; 3ROBUST_MEAN is now computed (mean of voxels in [P10,P90]) ~ 3MEAN 1983.32, trimmed
	{ "3STANDARD_ERROR", { 1.116333919044723 } },
	{ "3UNIFORMITY_PIU", { 50.59288537549407 } },
};

static const bool firstorder_3d_coverage_baseline_registered =
	register_coverage_baseline("firstorder", &firstorder_3d_regression_coverage_ref_vals);
