#pragma once

#include "test_morphology_common.h"

void test_shape2d_convex_hull_features()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, Nyxus::Feature2D::CIRCULARITY, "CIRCULARITY");
	// CONVEX_HULL_AREA / SOLIDITY are verifiable against scikit-image convex_hull_image(offset_coordinates=False)
	// (see the oracle_3p table); Nyxus reproduces that convention exactly, so a tight 1% tolerance suffices.
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::CONVEX_HULL_AREA, "CONVEX_HULL_AREA", 100.0);
	assert_verifiable_with_3p_builtin_oracle_shape2d_feature(fvals, Nyxus::Feature2D::SOLIDITY, "SOLIDITY", 100.0);
}
