#pragma once

#include "test_remaining2d_common.h"

// ---------------------------------------------------------------------------------------------------
// Migrated from test_2d_remaining_features.h (Wave 6): ZERNIKE2D. Per registry decision (§6.1) mahotas
// is not an accepted oracle, so ZERNIKE2D stays analytic/regression -> test_zernike_regression.h.
// Shared fixture/oracle-data lives in test_remaining2d_common.h.
// ---------------------------------------------------------------------------------------------------

void test_remaining2d_verifiable_with_3p_builtin_oracle_zernike2d_feature()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_verifiable_with_3p_builtin_oracle_remaining2d_vector_feature(fvals, Nyxus::Feature2D::ZERNIKE2D, "ZERNIKE2D");
}
