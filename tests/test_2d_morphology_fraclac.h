#pragma once

#include "test_2d_morphology_common.h"

void test_2d_morphology_fractal_dimension_blob512_fraclac()
{
	std::vector<std::vector<double>> fvals;
	calculate_fractal_blob512_feature_values(fvals);

	// FRACT_DIM_BOXCOUNT: same method as the oracle (single origin-aligned box count) -> tight.
	// FRACT_DIM_PERIMETER: cross-method (Nyxus divider vs box-count-of-edge) -> 3% tolerance.
	SCOPED_TRACE("FRACT_DIM_BLOB512_ORACLE");
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(Nyxus::Feature2D::FRACT_DIM_BOXCOUNT)][0],
		morphology_2d_fraclac_ref_vals["FRACT_DIM_BOXCOUNT"], 100.0));    // 1% (same method)
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(Nyxus::Feature2D::FRACT_DIM_PERIMETER)][0],
		morphology_2d_fraclac_ref_vals["FRACT_DIM_PERIMETER"], 33.0));    // ~3% (cross method)
}
