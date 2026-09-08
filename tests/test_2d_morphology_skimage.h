#pragma once

#include <string>
#include <vector>

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

static const ref_vals_map<double> morphology_2d_skimage_shape2d_ref_vals{
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
static const ref_vals_map<double> morphology_2d_skimage_circles_ref_vals{
	{"PERIMETER", 999.25901807804496},
};

// ROI_RADIUS_MAX / ROI_RADIUS_MEDIAN on filled digital disks, a third benchmark and so a third table
// (SPEC 6.3.1). RoiRadiusFeature measures every ROI pixel against the ROI's own contour and reports
// the mean/max/median of those distances; the reference is
// skimage.segmentation.find_boundaries(connectivity=1, mode='inner') -- the foreground pixels
// 4-adjacent to background, which is the only convention in the quantity -- plus a plain minimum
// over those pixels, cross-checked in the generator against scipy's distance transform.
// Generator: tests/vetting/oracles/gen_morphology_radius_skimage.py.
//
// Disks, not the shape2d raster, for the same reason PERIMETER is vetted on the circles benchmark:
// on 26 pixels with a hole the two boundary conventions have nothing to converge to. A disk also has
// an EXACT closed form for MAX -- sqrt((R-1)^2+1), the centre's distance to the boundary pixel at
// offset (1, R-1) -- so MAX grows LINEARLY in R, which the squared-distance defect this table was
// written for could never do: it reported 82, 362 and 1522 here. That closed form is a separate
// oracle and is asserted as one, in test_2d_morphology_analytic.h.
//
// ROI_RADIUS_MEAN is deliberately absent. It is the one of the three that does not survive the
// separate contour defect underneath: `buildRegularContour` reports every contour pixel one pixel
// right and one pixel down of where it is, so Nyxus measures against a shifted boundary. Shifting
// the reference boundary the same way reproduces all three Nyxus values exactly, and on a disk MAX
// and MEDIAN come back unchanged -- the maximizing pixel just moves with the shift -- while MEAN
// does not. MEAN stays a regression row until the contour offset is fixed.
static const ref_vals_map<double> morphology_2d_skimage_radius_disks_ref_vals{
	// R=10: 317 pixels, 56 boundary pixels; MAX/(R-1) = 1.006154
	{"ROI_RADIUS_MAX_R10", 9.055385138137417},
	{"ROI_RADIUS_MEDIAN_R10", 2.23606797749979},
	// R=20: 1257 pixels, 112 boundary pixels; MAX/(R-1) = 1.001384
	{"ROI_RADIUS_MAX_R20", 19.026297590440446},
	{"ROI_RADIUS_MEDIAN_R20", 5.0},
	// R=40: 5025 pixels, 224 boundary pixels; MAX/(R-1) = 1.000329
	{"ROI_RADIUS_MAX_R40", 39.01281840626232},
	{"ROI_RADIUS_MEDIAN_R40", 11.045361017187261},
};

static void assert_morphology_shape2d_skimage(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("SKIMAGE_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_skimage_shape2d_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_skimage_shape2d_ref_vals.at(feature_name), frac_tolerance));
}

static void assert_morphology_radius_disks_skimage(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("SKIMAGE_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_skimage_radius_disks_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_skimage_radius_disks_ref_vals.at(feature_name), frac_tolerance));
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

// ROI_RADIUS_MAX / ROI_RADIUS_MEDIAN vs the skimage inner boundary on three filled disks
// (tests/vetting/oracles/gen_morphology_radius_skimage.py). Three radii rather than one because the
// defect these replaced -- the features reported SQUARED distances -- is a units error, and one
// radius cannot see units: 82 and 9.055 are both "a plausible number" on a single disk. Across
// R = 10, 20, 40 the reference grows by 2.1x and 2.05x, and a squared distance grows by 4.4x and
// 4.2x, so the ratios separate the two even before the values do.
void test_2d_morphology_roi_radius_disks_skimage()
{
	for (double R : {10.0, 20.0, 40.0})
	{
		const std::string tag = "_R" + std::to_string((int)R);

		std::vector<std::vector<double>> fvals;
		calculate_disk_radius_values(R, fvals);

		assert_morphology_radius_disks_skimage(fvals, Nyxus::Feature2D::ROI_RADIUS_MAX, "ROI_RADIUS_MAX" + tag);
		assert_morphology_radius_disks_skimage(fvals, Nyxus::Feature2D::ROI_RADIUS_MEDIAN, "ROI_RADIUS_MEDIAN" + tag);
	}
	// The closed form these values also satisfy is a second, independent claim with a different
	// oracle, so it is asserted in test_2d_morphology_analytic.h rather than here (SPEC 2): an
	// `analytic` assertion inside a `_skimage` function would credit scikit-image with it.
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
		morphology_2d_skimage_circles_ref_vals.at("PERIMETER")));
}
