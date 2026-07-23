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

// ORIENTATION and EROSIONS_2_VANISH vetted vs scikit-image (tests/vetting/oracles/gen_morphology_skimage.py).
// ORIENTATION: skimage regionprops orientation is measured from the row axis; Nyxus measures the same
// ellipse's major axis from the x axis, so NYXUS == 90 - degrees(skimage.orientation) = 70.4173944984
// (matches to 10 decimals -- the angle is invariant to the pixel-size second-moment correction that makes
// the AXIS LENGTHS differ ~1.4%, which is why MAJOR/MINOR/ECCENTRICITY stay regression).
// EROSIONS_2_VANISH: Nyxus' 3x3 (8-connected) structuring element == skimage square(3); the count (1)
// matches, and disk(1)/4-connected gives 2, so the test also pins the connectivity convention.
void test_shape2d_skimage_orientation_and_erosions()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// skimage-derived goldens (90 - degrees(regionprops.orientation); square(3) erosion count)
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::ORIENTATION)][0], 70.4173944984207, 1e-3)
		<< "ORIENTATION does not match 90 - skimage.orientation(deg)";
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::EROSIONS_2_VANISH)][0], 1.0, 1e-9)
		<< "EROSIONS_2_VANISH does not match skimage square(3) erosion count";
}
