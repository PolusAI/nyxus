#pragma once

// The fixture the 2D Zernike tests share. gtest is deliberately absent: this file builds a ROI and a
// settings bundle and asserts nothing - the files that include it bring gtest in themselves, and
// SPEC 6.3.1 keeps every golden table with the assertions that read it.

// Every include below is named for a symbol this file uses directly rather than left to arrive
// transitively: <vector> for calculate_zernike_2d_values' parameter type.
#include <vector>

#include "../src/nyx/feature_settings.h"        // Fsettings
#include "../src/nyx/features/zernike.h"        // ZernikeFeature
#include "../src/nyx/roi_cache.h"               // LR
#include "test_data.h"                          // NyxusPixel, shape2d_morphology_{intensity,mask}
#include "test_main_nyxus.h"                    // load_masked_test_roi_data, make_shape2d_settings

// The 8x8 shape2d fixture at make_shape2d_settings() - a single 26-pixel concave ROI with one
// interior hole, carrying an asymmetric intensity gradient. ZernikeFeature reads the ROI's image
// matrix and nothing any other feature produces, so the loader is the whole prerequisite; the
// family builds its own ROI rather than borrowing the morphology fixture (SPEC 6.3.1).
// Recipe zernike.shape2d_native in tests/vetting/config_recipes.md.
static void build_zernike_2d_roi (LR& roidata)
{
	Fsettings s = make_shape2d_settings();

	load_masked_test_roi_data(
		roidata,
		shape2d_morphology_intensity,
		shape2d_morphology_mask,
		sizeof(shape2d_morphology_mask) / sizeof(NyxusPixel));
	roidata.initialize_fvals();

	ZernikeFeature zernike;
	zernike.calculate(roidata, s);
	zernike.save_value(roidata.fvals);
}

// The same ROI, returning only the feature values the tables read.
static void calculate_zernike_2d_values (std::vector<std::vector<double>>& fvals)
{
	LR roidata(101);
	build_zernike_2d_roi(roidata);
	fvals = roidata.fvals;
}
