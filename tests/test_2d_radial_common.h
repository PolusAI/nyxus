#pragma once

// The fixture the 2D radial intensity-distribution tests share. gtest is deliberately absent: this
// file builds a ROI and a settings bundle and asserts nothing - the files that include it bring
// gtest in themselves, and SPEC 6.3.1 keeps every golden table with the assertions that read it.

// Every include below is named for a symbol this file uses directly rather than left to arrive
// transitively: <vector> for calculate_radial_2d_values' parameter type.
#include <vector>

#include "../src/nyx/feature_settings.h"                // Fsettings
#include "../src/nyx/features/contour.h"                // ContourFeature
#include "../src/nyx/features/radial_distribution.h"    // RadialDistributionFeature
#include "../src/nyx/roi_cache.h"                       // LR
#include "test_data.h"                                  // NyxusPixel, shape2d_morphology_{intensity,mask}
#include "test_main_nyxus.h"                            // load_masked_test_roi_data, make_shape2d_settings

// The 8x8 shape2d fixture at make_shape2d_settings() - a single 26-pixel concave ROI with one
// interior hole, carrying an asymmetric intensity gradient. RadialDistributionFeature reads the ROI
// contour, so ContourFeature runs first; nothing else in the shape set feeds it, which is why the
// radial family builds its own ROI instead of borrowing the morphology fixture (SPEC 6.3.1).
// Recipe radial.shape2d_native in tests/vetting/config_recipes.md.
static void build_radial_2d_roi (LR& roidata)
{
	Fsettings s = make_shape2d_settings();

	load_masked_test_roi_data(
		roidata,
		shape2d_morphology_intensity,
		shape2d_morphology_mask,
		sizeof(shape2d_morphology_mask) / sizeof(NyxusPixel));
	roidata.initialize_fvals();

	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);

	RadialDistributionFeature radial;
	radial.calculate(roidata, s);
	radial.save_value(roidata.fvals);
}

// The same ROI, returning only the feature values the three tables read.
static void calculate_radial_2d_values (std::vector<std::vector<double>>& fvals)
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	fvals = roidata.fvals;
}
