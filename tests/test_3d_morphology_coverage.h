#pragma once

#include "test_3d_coverage_common.h"

// Per-family slice of the 3D coverage sweep (Wave 9). The shared harness, the two parameterized
// fixtures, their TEST_P bodies, and the global count-guard live in test_3d_coverage_common.h; this
// file only re-instantiates the two suites for the "morphology" family. Every public 3D feature is
// classified into exactly one family (first-match on the calculator featuresets), so the per-family
// instantiations together reproduce the original 94-embedded + 119-unvetted split with no drift.

INSTANTIATE_TEST_SUITE_P(
	MORPHOLOGY_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("morphology", true)),
	sanitize_3d_feature_test_name);

INSTANTIATE_TEST_SUITE_P(
	MORPHOLOGY_UNVETTED_LOCAL_REGRESSION,
	Test3DFeature_UNVETTED_LOCAL_REGRESSION,
	testing::ValuesIn(feature_3d_cases_for_family("morphology", false)),
	sanitize_3d_feature_test_name);

// Regression baselines for this family's slice of the sweep: pinned Nyxus output for the public 3D
// morphology features that no third-party oracle backs yet. They establish no vetting (SPEC 1), and they
// live here rather than in the shared harness so the table sits with the assertions that read it.
static const ref_vals_map<std::vector<double>> morphology_3d_regression_coverage_ref_vals
{
	{ "3AREA", { 59992 } },
	{ "3AREA_2_VOLUME", { 0.21860475559470999 } },
	{ "3COMPACTNESS1", { 0.010537043861899255 } },
	{ "3COMPACTNESS2", { 0.039449347281835329 } },
	{ "3ELONGATION", { 0.84332105599389762 } },  // FIX(axis mislabel): was 0.8099; now = MIRP morph_pca_elongation
	{ "3FLATNESS", { 0.68299758045903847 } },  // FIX(axis mislabel): was 1.186 (>1, impossible); now = MIRP morph_pca_flatness
	{ "3LEAST_AXIS_LEN", { 71.514499741981993 } },  // FIX(axis mislabel): was 104.7 (>MAJOR, impossible); now = MIRP morph_pca_least_axis
	{ "3MAJOR_AXIS_LEN", { 104.70681271508683 } },  // FIX(axis mislabel): was 88.3; now = MIRP morph_pca_maj_axis (largest)
	{ "3MINOR_AXIS_LEN", { 88.301459868642283 } },  // FIX(axis mislabel): was 71.5; now = MIRP morph_pca_min_axis
	{ "3SPHERICAL_DISPROPORTION", { 2.9375598657539634 } },
	{ "3SPHERICITY", { 0.34041859424142729 } },
};

static const bool morphology_3d_coverage_baseline_registered =
	register_coverage_baseline("morphology", &morphology_3d_regression_coverage_ref_vals);
