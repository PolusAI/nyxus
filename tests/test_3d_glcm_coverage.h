#pragma once

#include "test_3d_coverage_common.h"

// Per-family slice of the 3D coverage sweep (Wave 9). The shared harness, the two parameterized
// fixtures, their TEST_P bodies, and the global count-guard live in test_3d_coverage_common.h; this
// file only re-instantiates the suites for the "glcm" family. Every public 3D feature is classified
// into exactly one family (first-match on the calculator featuresets).
//
// The unvetted half (GLCM_UNVETTED_LOCAL_REGRESSION) and its local ref-vals table have been retired:
// as of this branch, every one of the 36 features that table covered is individually drift-guarded
// in test_3d_glcm_regression.h instead (the "_grey64_regression" tests), and "glcm" is listed in
// test_3d_coverage_common.h's families_with_individually_ported_regression escape valve so the
// TEST_3D_FEATURE_COVERAGE_COUNTS guard doesn't expect a local table here anymore. The oracle-backed
// half stays on the generic sweep -- it reads glcm_3d_pyradiomics_ref_vals (test_3d_glcm_pyradiomics.h)
// directly and was never sourced from a table local to this file.

INSTANTIATE_TEST_SUITE_P(
	GLCM_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("glcm", true)),
	sanitize_3d_feature_test_name);
