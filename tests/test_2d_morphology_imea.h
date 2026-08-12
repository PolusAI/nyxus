#pragma once

#include "test_2d_morphology_common.h"
#include "test_2d_remaining_common.h"   // morphology_2d_imea_ref_vals + assert_caliper_close_to_imea

#include "test_ref_vals.h"

// imea goldens, one table per benchmark. They cannot be one table: the same features are measured on
// two fixtures and every shared key differs by roughly 5x (STAT_FERET_DIAM_MIN is 4.473 on the
// shape2d mask and 21.0 on the a=20/b=10 ellipse), so merging would collide 12 keys and drop one
// fixture's numbers. SPEC 6.3.1 covers this with the _<subject> qualifier.
//
// The shape2d table also absorbs the three geodetic keys, which share its fixture and overlap nothing.
static ref_vals_map<double> morphology_2d_imea_shape2d_ref_vals{
	{"STAT_FERET_DIAM_MIN", 4.47301},
	{"STAT_FERET_DIAM_MAX", 6.3222},
	{"STAT_FERET_DIAM_MEAN", 5.40848},
	{"STAT_FERET_DIAM_MEDIAN", 5.19615},
	{"STAT_FERET_DIAM_STDDEV", 0.550668},
	{"STAT_FERET_DIAM_MODE", 5.0},
	// FIXED (caliper reimpl): Martin is now the area-bisecting chord and Nassenstein the bottom-tangent
	// vertical chord (one diameter per angle), not the old min+max of a Y-grid of horizontal chords.
	// The old goldens pinned the bug (Martin min 0.8, Nassenstein min/mode 0.0 — impossible for a solid
	// shape). These are the corrected values on the 8x8 fixture; the diameters are vetted vs imea on a
	// clean ellipse in TEST_2D_MORPHOLOGY_CALIPER_MARTIN_NASSENSTEIN_IMEA.
	// FIX (caliper float-precision): re-pinned again after the float-precision hull rotation removed the inward
	// integer-truncation bias (MODE unchanged).
	{"STAT_MARTIN_DIAM_MIN", 4.25885},
	{"STAT_MARTIN_DIAM_MAX", 6.12801},
	{"STAT_MARTIN_DIAM_MEAN", 5.01762},
	{"STAT_MARTIN_DIAM_MEDIAN", 4.97511},
	{"STAT_MARTIN_DIAM_STDDEV", 0.553162},
	{"STAT_MARTIN_DIAM_MODE", 4.0},
	{"STAT_NASSENSTEIN_DIAM_MIN", 1.67316},
	{"STAT_NASSENSTEIN_DIAM_MAX", 6.24165},
	{"STAT_NASSENSTEIN_DIAM_MEAN", 4.77746},
	{"STAT_NASSENSTEIN_DIAM_MEDIAN", 5.03857},
	{"STAT_NASSENSTEIN_DIAM_STDDEV", 1.09628},
	{"STAT_NASSENSTEIN_DIAM_MODE", 4.0},
	{"ALLCHORDS_MIN", 1.0},
	{"DIAMETER_EQUAL_PERIMETER", 8.57365809435587},
	{"GEODETIC_LENGTH", 11.13182483477333},
	{"THICKNESS", 2.3356458070362205}
};

static ref_vals_map<double> morphology_2d_imea_ellipse_ref_vals{
	{"STAT_MARTIN_DIAM_MIN", 19.0},
	{"STAT_MARTIN_DIAM_MAX", 41.0},
	{"STAT_MARTIN_DIAM_MEAN", 27.61},
	{"STAT_MARTIN_DIAM_MEDIAN", 25.5},
	{"STAT_NASSENSTEIN_DIAM_MIN", 16.0},
	{"STAT_NASSENSTEIN_DIAM_MAX", 41.0},
	{"STAT_NASSENSTEIN_DIAM_MEAN", 25.17},
	{"STAT_NASSENSTEIN_DIAM_MEDIAN", 21.5},
	// Feret is a correct rotating-calipers implementation (unlike the Martin/Nassenstein bug); it
	// agrees with imea within the same ~1-2px hull-vs-raster convention gap. Reference from imea
	// (imea.measure_2d.statistical_length feret_diameters, dalpha=10) on the same ellipse.
	{"STAT_FERET_DIAM_MIN", 21.0},
	{"STAT_FERET_DIAM_MAX", 41.0},
	{"STAT_FERET_DIAM_MEAN", 31.72},
	{"STAT_FERET_DIAM_MEDIAN", 32.5},
	// Minimum enclosing circle (Welzl / cv2.minEnclosingCircle) is centroid-independent and matches
	// imea/OpenCV exactly: for the ellipse a=20 its diameter = the major axis = 2a = 40. (The circle
	// fixture's value 30 is asserted inline.) NOTE: DIAMETER_CIRCUMSCRIBING_CIRCLE and
	// DIAMETER_INSCRIBING_CIRCLE are NOT here — they are imea's crude max/min centroid-to-contour
	// distance approximation (not a true geometric circle), sensitive to Nyxus's contour convention +
	// the centroid-1 offset (a symmetric circle yields 35.6/23.3, not ~30/~30), so they stay regression.
	{"DIAMETER_MIN_ENCLOSING_CIRCLE", 40.0},
};

static void assert_caliper_imea(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("IMEA_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_imea_shape2d_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_imea_shape2d_ref_vals[feature_name], frac_tolerance));
}

// The 19 caliper/chord statistics whose registry rows read status=vetted, oracle=imea,
// target_test=test_2d_morphology_imea.h. They were asserted from test_2d_morphology_regression.h
// through remaining2d_caliper_ref_val(), a lookup spanning an imea table and a regression table -- so
// a _regression function was resolving imea-vetted values, and the rows' target_test was never met.
void test_2d_morphology_caliper_stats_imea()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MIN, "STAT_FERET_DIAM_MIN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MAX, "STAT_FERET_DIAM_MAX");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEAN, "STAT_FERET_DIAM_MEAN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEDIAN, "STAT_FERET_DIAM_MEDIAN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_STDDEV, "STAT_FERET_DIAM_STDDEV");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MODE, "STAT_FERET_DIAM_MODE");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MIN, "STAT_MARTIN_DIAM_MIN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MAX, "STAT_MARTIN_DIAM_MAX");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEAN, "STAT_MARTIN_DIAM_MEAN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEDIAN, "STAT_MARTIN_DIAM_MEDIAN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_STDDEV, "STAT_MARTIN_DIAM_STDDEV");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MODE, "STAT_MARTIN_DIAM_MODE");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MIN, "STAT_NASSENSTEIN_DIAM_MIN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MAX, "STAT_NASSENSTEIN_DIAM_MAX");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEAN, "STAT_NASSENSTEIN_DIAM_MEAN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEDIAN, "STAT_NASSENSTEIN_DIAM_MEDIAN");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_STDDEV, "STAT_NASSENSTEIN_DIAM_STDDEV");
	assert_caliper_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MODE, "STAT_NASSENSTEIN_DIAM_MODE");
	assert_caliper_imea(fvals, Nyxus::Feature2D::ALLCHORDS_MIN, "ALLCHORDS_MIN");
}

static void assert_caliper_close_to_imea(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	// FIX (caliper float-precision): tightened 0.15 -> 0.10 after the float-precision hull rotation removed the
	// integer-truncation inward bias. Measured residuals on the a=20,b=10 ellipse: Martin 1.8-4.6%,
	// Feret 1.4-4.8%, Nassenstein 2.4-3.7% except its bottom-tangent MIN (8.9%) and MEDIAN (6.0%). The
	// floor is the Nassenstein MIN: a near-apex vertical tangent chord measured on the convex hull vs
	// imea's raster - a definitional hull-vs-raster gap, not a precision loss, so 0.10 is the honest bound.
	double reltol = 0.10)
{
	SCOPED_TRACE(std::string("CALIPER_VS_IMEA__") + feature_name);
	ASSERT_TRUE(morphology_2d_imea_ellipse_ref_vals.count(feature_name) > 0);
	const double imea_ref = morphology_2d_imea_ellipse_ref_vals[feature_name];
	const double actual = fvals[static_cast<int>(feature)][0];
	const double denom = std::max(std::abs(imea_ref), 1e-9);
	ASSERT_LE(std::abs(actual - imea_ref) / denom, reltol)
		<< feature_name << " nyxus=" << actual << " imea=" << imea_ref;
}



// DIAMETER_EQUAL_PERIMETER vetted vs imea (tests/vetting/oracles/gen_morphology_imea.py).
//
// imea.measure_2d.macro.perimeter_equal_diameter(perimeter) is a third-party implementation of the
// DIN ISO 9276-6 perimeter-equal diameter (perimeter/pi) -- the same documented transform Nyxus
// applies in contour.cpp. Fed the Nyxus PERIMETER (26.9349412836191) it returns 8.57365809435588,
// matching the Nyxus DIAMETER_EQUAL_PERIMETER to double precision (measured |diff| = 1.07e-14), so
// this is asserted at the SPEC 7 "exact" tier.
//
// SCOPE: this vets the TRANSFORM, not the perimeter it consumes. Nyxus walks its chain-code contour
// (26.9349) while imea takes cv2.arcLength over the OpenCV contour (12.6569), so imea's END-TO-END
// diameter_equal_perimeter on this fixture is 4.0288 -- it does NOT agree with Nyxus, and the whole
// gap is inherited from PERIMETER, which stays a regression row of its own. What is pinned here is
// that Nyxus' derived diameter is the ISO quantity a third-party package computes from the same input.
void test_2d_morphology_diameter_equal_perimeter_imea()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// imea 0.3.5: macro.perimeter_equal_diameter(26.9349412836191)
	const double imea_dep = 8.573658094355881;
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_EQUAL_PERIMETER)][0], imea_dep, 1e-9)
		<< "DIAMETER_EQUAL_PERIMETER does not match imea perimeter_equal_diameter(PERIMETER)";
}

// GEODETIC_LENGTH + THICKNESS vetted vs imea (tests/vetting/oracles/gen_morphology_imea.py).
//
// imea.measure_2d.macro.geodeticlength_and_thickness(area, perimeter) is a third-party implementation
// of the DIN ISO 9276-6 rectangle model -- the rectangle with the same area and perimeter as the
// object, whose side lengths are the roots of x^2 - (P/2)x + A = 0, i.e. P/4 +- sqrt(P^2/16 - A).
// That is the same documented model Nyxus applies in geo_len_thickness.cpp. Fed the Nyxus
// AREA_PIXELS_COUNT (26) and PERIMETER (26.9349412836191) it returns 11.13182483477333 /
// 2.3356458070362205, which are the Nyxus values bit for bit (measured |diff| = 0 for both), so this
// is asserted at the SPEC 7 "exact" tier.
//
// SCOPE: this vets the TRANSFORM, not the area and perimeter it consumes. imea derives its own
// perimeter from cv2.arcLength (12.6569 on this fixture) instead of a chain-code contour walk, and
// with that perimeter (P/4)^2 - A goes negative and imea clamps the root, so its END-TO-END
// geodeticlength and thickness both collapse to 3.1642 -- neither agrees with Nyxus (differences 7.97
// and 0.83). The whole gap is inherited from PERIMETER, which stays a regression row of its own. What
// is pinned here is that Nyxus' two side lengths are the ISO quantities a third-party package computes
// from the same two inputs -- the same scope as the DIAMETER_EQUAL_PERIMETER claim above.
void test_2d_morphology_geodetic_length_thickness_imea()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// imea 0.3.5: macro.geodeticlength_and_thickness(26.0, 26.9349412836191)
	const double imea_geodetic_length = 11.13182483477333;
	const double imea_thickness = 2.3356458070362205;
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::GEODETIC_LENGTH)][0], imea_geodetic_length, 1e-9)
		<< "GEODETIC_LENGTH does not match imea geodeticlength_and_thickness(AREA, PERIMETER)";
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::THICKNESS)][0], imea_thickness, 1e-9)
		<< "THICKNESS does not match imea geodeticlength_and_thickness(AREA, PERIMETER)";
}

// Vets the reimplemented Martin (area-bisecting chord) and Nassenstein (bottom-tangent vertical
// chord) diameters against imea on a clean filled ellipse (a=20, b=10). See the oracle block in
// test_2d_remaining_common.h. Robust stats (min/max/mean/median) agree with imea within the
// hull-vs-raster convention tolerance; the >0 lower bound pins that the old min+max-chord bug
// (0-length Nassenstein diameters) is gone.
void test_2d_morphology_caliper_martin_nassenstein_imea()
{
	std::vector<std::vector<double>> fvals;
	calculate_ellipse_caliper_values(fvals);

	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MIN, "STAT_MARTIN_DIAM_MIN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MAX, "STAT_MARTIN_DIAM_MAX");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEAN, "STAT_MARTIN_DIAM_MEAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_MEDIAN, "STAT_MARTIN_DIAM_MEDIAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MIN, "STAT_NASSENSTEIN_DIAM_MIN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MAX, "STAT_NASSENSTEIN_DIAM_MAX");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEAN, "STAT_NASSENSTEIN_DIAM_MEAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MEDIAN, "STAT_NASSENSTEIN_DIAM_MEDIAN");

	// Bug-gone invariant: a solid shape cannot have a 0-length diameter (the old code produced 0).
	ASSERT_GT(fvals[static_cast<int>(Nyxus::Feature2D::STAT_MARTIN_DIAM_MIN)][0], 2.0);
	ASSERT_GT(fvals[static_cast<int>(Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_MIN)][0], 2.0);
}
// Vets the Feret diameter distribution against imea on the same filled ellipse. Feret is a correct
// rotating-calipers implementation; robust stats (min/max/mean/median) agree with imea within the
// hull-vs-raster convention tolerance. (MIN/MAX_FERET_ANGLE stay regression — they are a Nyxus-frame
// angle convention with no directly comparable imea output.)
void test_2d_morphology_caliper_feret_imea()
{
	std::vector<std::vector<double>> fvals;
	calculate_ellipse_caliper_values(fvals);

	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MIN, "STAT_FERET_DIAM_MIN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MAX, "STAT_FERET_DIAM_MAX");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEAN, "STAT_FERET_DIAM_MEAN");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_MEDIAN, "STAT_FERET_DIAM_MEDIAN");
}
// Vets the minimum-enclosing-circle diameter (Welzl / cv2.minEnclosingCircle) against its exact
// geometric/imea value on two clean fixtures: ellipse a=20 -> 2a=40, circle r=15 -> 30. This is
// centroid-independent, so it matches to <0.1%. (DIAMETER_CIRCUMSCRIBING/INSCRIBING_CIRCLE are left
// regression: imea's centroid-to-contour-distance approximation, convention-sensitive.)
void test_2d_morphology_min_enclosing_circle_imea()
{
	std::vector<std::vector<double>> ell;
	calculate_ellipse_caliper_values(ell);
	assert_caliper_close_to_imea(ell, Nyxus::Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE, "DIAMETER_MIN_ENCLOSING_CIRCLE", 0.05);

	std::vector<std::vector<double>> cir;
	calculate_circle_shape_values(cir);
	const double d = cir[static_cast<int>(Nyxus::Feature2D::DIAMETER_MIN_ENCLOSING_CIRCLE)][0];
	ASSERT_NEAR(d, 30.0, 30.0 * 0.05) << "circle min-enclosing nyxus=" << d << " expected~30";
}
