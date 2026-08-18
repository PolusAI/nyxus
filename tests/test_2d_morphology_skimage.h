#pragma once

#include <string>
#include <vector>

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

static ref_vals_map<double> morphology_2d_skimage_shape2d_ref_vals{
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
	// regionprops.orientation is the major-axis angle from the ROW axis, CCW; Nyxus measures the
	// same ellipse from the X axis, so NYXUS == 90 - degrees(skimage.orientation). The angle is
	// invariant to the +1/12 pixel finite-size second-moment correction (it shifts mu20 and mu02
	// equally, leaving mu20-mu02 and mu11 unchanged), which is why ORIENTATION vets here while the
	// AXIS LENGTHS do not -- those differ ~1.4% and are vetted against the matlab oracle instead
	// (run as Octave regionprops, SPEC 4).
	{"ORIENTATION", 70.417394498420663},
	// regionprops.equivalent_diameter_area = sqrt(4*Area/pi); no convention gap
	{"DIAMETER_EQUAL_AREA", 5.7536273917515919},
};

// PERIMETER on the circles fixture (roiDataForPerimeterTest), a separate benchmark from the shape2d
// mask above, so it needs its own table per SPEC 6.3.1. skimage's measure.perimeter (the
// 4-neighbourhood boundary walk, which is what regionprops.perimeter uses) and Nyxus' chain-code
// contour walk are the same algorithm and agree to 3.8e-15 on this 14309-pixel object.
//
// The two do NOT agree on the small shape2d mask (skimage 12.657 vs Nyxus 26.935): there the object
// is 26 pixels with a 1-pixel hole, and the two contour conventions have nothing to converge to.
// PERIMETER is therefore vetted on this benchmark only, and stays a regression row on shape2d.
static ref_vals_map<double> morphology_2d_skimage_circles_ref_vals{
	{"PERIMETER", 999.25901807804496},
};

static void assert_morphology_shape2d_skimage(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("SKIMAGE_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_skimage_shape2d_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_skimage_shape2d_ref_vals[feature_name], frac_tolerance));
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
	assert_morphology_shape2d_skimage(fvals, Nyxus::Feature2D::CONVEX_HULL_AREA, "CONVEX_HULL_AREA", 100.0);
	assert_morphology_shape2d_skimage(fvals, Nyxus::Feature2D::SOLIDITY, "SOLIDITY", 100.0);
}

// ORIENTATION and EROSIONS_2_VANISH vetted vs scikit-image (tests/vetting/oracles/gen_morphology_skimage.py).
// EROSIONS_2_VANISH: Nyxus' 3x3 (8-connected) structuring element == skimage square(3); the count (1)
// matches, and disk(1)/4-connected gives 2, so the test also pins the connectivity convention.
// Both read the table above rather than repeating the number inline -- a golden pinned in two places
// drifts in one of them, and only the copy an assertion reads is under test.
void test_2d_morphology_orientation_and_erosions_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_shape2d_skimage(fvals, Nyxus::Feature2D::ORIENTATION, "ORIENTATION");
	assert_morphology_shape2d_skimage(fvals, Nyxus::Feature2D::EROSIONS_2_VANISH, "EROSIONS_2_VANISH");
}

// DIAMETER_EQUAL_AREA = sqrt(4*Area/pi), skimage's regionprops.equivalent_diameter_area. Same closed
// form on both sides and Area is an exact pixel count, so this agrees to double precision.
void test_2d_morphology_diameter_equal_area_skimage()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_shape2d_skimage(fvals, Nyxus::Feature2D::DIAMETER_EQUAL_AREA, "DIAMETER_EQUAL_AREA");
}

// PERIMETER vs skimage.measure.perimeter on the circles benchmark. Beware the MATLAB vocabulary
// here: nnz(bwperim(...)) counts perimeter PIXELS (846) and regionprops('Perimeter') returns 952.848
// -- neither is this quantity, so neither can vet it (audit/morphology_2d_skimage_vetting_report.md).
void test_2d_morphology_perimeter_skimage()
{
	Fsettings s;
	s.resize((int)NyxSetting::__COUNT__);
	s[(int)NyxSetting::SOFTNAN].rval = 0.0;
	s[(int)NyxSetting::TINY].rval = 0.0;
	s[(int)NyxSetting::SINGLEROI].bval = false;
	s[(int)NyxSetting::GREYDEPTH].ival = 128;
	s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
	s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
	s[(int)NyxSetting::USEGPU].bval = false;
	s[(int)NyxSetting::VERBOSLVL].ival = 0;
	s[(int)NyxSetting::IBSI].bval = false;

	LR roidata(100);   // dummy label 100
	roidata.slide_idx = -1; // we don't have a real slide for this test ROI
	load_test_roi_data (roidata, roiDataForPerimeterTest, sizeof(roiDataForPerimeterTest)/sizeof(NyxusPixel));

	// Anisotropy (none)
	roidata.make_nonanisotropic_aabb();

	ContourFeature f;
	ASSERT_NO_THROW(f.calculate(roidata, s));

	roidata.initialize_fvals();
	f.save_value (roidata.fvals);

	SCOPED_TRACE("SKIMAGE_ORACLE__PERIMETER");
	ASSERT_TRUE(morphology_2d_skimage_circles_ref_vals.count("PERIMETER") > 0);
	ASSERT_TRUE(agrees_gt(roidata.fvals[(int)Nyxus::Feature2D::PERIMETER][0],
		morphology_2d_skimage_circles_ref_vals["PERIMETER"]));
}
