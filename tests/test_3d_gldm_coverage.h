#pragma once

#include "test_3d_coverage_common.h"

// Per-family slice of the legacy oracle-backed coverage sweep. The global count guard separately
// verifies that every unvetted public feature has a named regression assertion.

INSTANTIATE_TEST_SUITE_P(
	GLDM_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("gldm", true)),
	sanitize_3d_feature_test_name);
