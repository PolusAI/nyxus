#pragma once

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

static ref_vals_map<double> morphology_2d_skimage_hull_ref_vals{
	// CONVEX_HULL_AREA / SOLIDITY are cross-checked against scikit-image on this exact ROI. Nyxus
	// computes a Pick's-theorem pixel-count hull area (convex_hull_nontriv.cpp) = 27, solidity
	// 26/27 = 0.9629630. Because Nyxus hulls through pixel CENTRES, this reproduces skimage's
	// convex_hull_image(offset_coordinates=False) == 27 EXACTLY, so we vet against THAT convention
	// (27 / 0.9629630) with a tight 1% tolerance (frac_tolerance=100). The hull area is a
	// provably-exact integer lattice count, so 1% is float/platform slack -- not a convention fudge --
	// and it still catches a >=1 px regression. (skimage's regionprops DEFAULT uses
	// offset_coordinates=True, which first expands every pixel to its +/-0.5 corners and rasterises
	// the hull to 28 / 0.9285714; that +1 px is a corner-expansion convention, not an error, and is
	// why we pin the offset_coordinates=False value rather than the default.) SOLIDITY is thus a real
	// skimage-vetted <= 1 check (unlike the old impossible 1.3), matched exactly rather than within a
	// loose band.
	{"CONVEX_HULL_AREA", 27.0},
	{"SOLIDITY", 0.9629629629629629},
	{"EROSIONS_2_VANISH", 1.0},
};

static void assert_morphology_hull_skimage(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("SKIMAGE_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_skimage_hull_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_skimage_hull_ref_vals[feature_name], frac_tolerance));
}

void test_2d_morphology_convex_hull_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// CIRCULARITY is not a skimage claim: the registry vets it as oracle=analytic, and
	// test_2d_morphology_analytic.h already asserts it against sqrt(4*pi*A)/P. What stood here
	// compared it to a Nyxus snapshot, so it added no coverage under a skimage name.
	// CONVEX_HULL_AREA / SOLIDITY are verifiable against scikit-image convex_hull_image(offset_coordinates=False)
	// (see the oracle_3p table); Nyxus reproduces that convention exactly, so a tight 1% tolerance suffices.
	assert_morphology_hull_skimage(fvals, Nyxus::Feature2D::CONVEX_HULL_AREA, "CONVEX_HULL_AREA", 100.0);
	assert_morphology_hull_skimage(fvals, Nyxus::Feature2D::SOLIDITY, "SOLIDITY", 100.0);
}

// ORIENTATION and EROSIONS_2_VANISH vetted vs scikit-image (tests/vetting/oracles/gen_morphology_skimage.py).
// ORIENTATION: skimage regionprops orientation is measured from the row axis; Nyxus measures the same
// ellipse's major axis from the x axis, so NYXUS == 90 - degrees(skimage.orientation) = 70.4173944984
// (matches to 10 decimals -- the angle is invariant to the pixel-size second-moment correction that makes
// the AXIS LENGTHS differ ~1.4%, which is why MAJOR/MINOR/ECCENTRICITY stay regression).
// EROSIONS_2_VANISH: Nyxus' 3x3 (8-connected) structuring element == skimage square(3); the count (1)
// matches, and disk(1)/4-connected gives 2, so the test also pins the connectivity convention.
void test_2d_morphology_orientation_and_erosions_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// skimage-derived goldens (90 - degrees(regionprops.orientation); square(3) erosion count)
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::ORIENTATION)][0], 70.4173944984207, 1e-3)
		<< "ORIENTATION does not match 90 - skimage.orientation(deg)";
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::EROSIONS_2_VANISH)][0], 1.0, 1e-9)
		<< "EROSIONS_2_VANISH does not match skimage square(3) erosion count";
}
