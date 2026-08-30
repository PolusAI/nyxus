#pragma once

#include <vector>

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

// Provenance (SPEC 6.4): tool=ImageJ FracLac (box counting); recipe=morphology.fractal_blob512,
// i.e. the 512x512 mask tests/data/fractal_blob512_seg.ome.tif as a single ROI;
// generators=tests/vetting/oracles/fraclac/{shiftgrid_boxcount.ijm,ref_boxcount.py}.
static const ref_vals_map<double> morphology_2d_fraclac_ref_vals{
	{"FRACT_DIM_BOXCOUNT", 1.8706},   // vetted by ImageJ/FracLac: an independent implementation of the same box-count method
	{"FRACT_DIM_PERIMETER", 1.0493},  // vetted by ImageJ/FracLac edge box-count: cross-method vs Nyxus' divider
};

void test_2d_morphology_fractal_dimension_blob512_fraclac()
{
	std::vector<std::vector<double>> fvals;
	calculate_fractal_blob512_feature_values(fvals);

	// FRACT_DIM_BOXCOUNT: same method as the oracle (single origin-aligned box count) -> tight.
	// FRACT_DIM_PERIMETER: cross-method (Nyxus divider vs box-count-of-edge) -> 3% tolerance.
	SCOPED_TRACE("FRACT_DIM_BLOB512_ORACLE");
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(Nyxus::Feature2D::FRACT_DIM_BOXCOUNT)][0],
		morphology_2d_fraclac_ref_vals.at("FRACT_DIM_BOXCOUNT"), 100.0));    // 1% (same method)
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(Nyxus::Feature2D::FRACT_DIM_PERIMETER)][0],
		morphology_2d_fraclac_ref_vals.at("FRACT_DIM_PERIMETER"), 33.0));    // ~3% (cross method)
}
