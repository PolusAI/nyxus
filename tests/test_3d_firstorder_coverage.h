#pragma once

#include "test_3d_coverage_common.h"

// Retain the family's oracle-backed coverage sweep. Regression snapshots and assertions live in
// test_3d_firstorder_regression.h and are deliberately not instantiated through this legacy file.
INSTANTIATE_TEST_SUITE_P(
	FIRSTORDER_WITH_3P_EMBEDDED_GT,
	Test3DFeature_WITH_3P_EMBEDDED_GT,
	testing::ValuesIn(feature_3d_cases_for_family("firstorder", true)),
	sanitize_3d_feature_test_name);
