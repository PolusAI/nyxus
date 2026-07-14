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
