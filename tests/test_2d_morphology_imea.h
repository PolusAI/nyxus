#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "test_2d_morphology_common.h"   // calculate_{shape2d_feature,ellipse_caliper,circle_shape}_values

#include "test_ref_vals.h"

// imea goldens, one table per benchmark. They cannot be one table: the same features are measured on
// two fixtures and every shared key differs by roughly 5x (STAT_FERET_DIAM_MIN is 4.473 on the
// shape2d mask and 21.0 on the a=20/b=10 ellipse), so merging would collide keys and drop one
// fixture's numbers. SPEC 6.3.1 covers this with the _<subject> qualifier.
//
// This table holds only what imea actually produces on the shape2d fixture: the two ISO transforms.
// The caliper/chord statistics are NOT comparable to imea on this 8x8 raster -- its own values there
// differ from Nyxus by 3.9-79.3%, too coarse for the hull-vs-raster conventions to converge -- so
// they are regression snapshots in test_2d_morphology_regression.h and the diameters are vetted
// against imea on the clean ellipse below. Measured in audit/morphology_2d_imea_vetting_report.md.
static const ref_vals_map<double> morphology_2d_imea_shape2d_ref_vals{
	{"DIAMETER_EQUAL_PERIMETER", 8.573658094355881},
	{"GEODETIC_LENGTH", 11.13182483477333},
	{"THICKNESS", 2.3356458070362205}
};

// The caliper benchmark: a filled ellipse a=20, b=10 (calculate_ellipse_caliper_values), clean
// enough that the hull-vs-raster convention gap is bounded instead of dominating.
//
// Provenance (SPEC 6.4): tool=imea 0.3.5; env=nyxus_mirp (conda);
// config=imea.shape_measurements_2d(mask, spatial_resolution_xy=1.0, dalpha=10);
// generator=tests/vetting/oracles/gen_morphology_imea.py.
//
// dalpha=10 is not arbitrary -- it is the step Nyxus' own calipers sweep
// (rot_angle_increment = 10 degrees, caliper.h), so the two sample the same angles. Every value here
// comes from one run at that step; mixing steps inflates the tolerance
// (audit/morphology_2d_imea_vetting_report.md). Worst residual 4.99%.
static const ref_vals_map<double> morphology_2d_imea_ellipse_ref_vals{
	{"STAT_MARTIN_DIAM_MIN", 19.0},
	{"STAT_MARTIN_DIAM_MAX", 41.0},
	{"STAT_MARTIN_DIAM_MEAN", 27.5},
	{"STAT_MARTIN_DIAM_MEDIAN", 25.5},
	{"STAT_MARTIN_DIAM_STDDEV", 7.197607627229728},
	{"STAT_NASSENSTEIN_DIAM_MIN", 18.0},
	{"STAT_NASSENSTEIN_DIAM_MAX", 41.0},
	{"STAT_NASSENSTEIN_DIAM_MEAN", 24.833333333333332},
	{"STAT_NASSENSTEIN_DIAM_MEDIAN", 21.0},
	{"STAT_NASSENSTEIN_DIAM_STDDEV", 7.365459931328117},
	{"STAT_FERET_DIAM_MIN", 21.0},
	{"STAT_FERET_DIAM_MAX", 41.0},
	{"STAT_FERET_DIAM_MEAN", 31.555555555555557},
	{"STAT_FERET_DIAM_MEDIAN", 32.0},
	{"STAT_FERET_DIAM_STDDEV", 6.7595382996689475},
	// the shortest chord across the ellipse is one pixel on both sides
	{"ALLCHORDS_MIN", 1.0},
	// Minimum enclosing circle (Welzl / cv2.minEnclosingCircle) is centroid-independent and matches
	// imea/OpenCV to 8 digits: for the ellipse a=20 its diameter is the major axis, 2a = 40. (The
	// circle fixture's value 30 is asserted inline.) NOTE: DIAMETER_CIRCUMSCRIBING_CIRCLE and
	// DIAMETER_INSCRIBING_CIRCLE are NOT here -- they are imea's crude max/min centroid-to-contour
	// distance approximation (not a true geometric circle), sensitive to Nyxus's contour convention +
	// the centroid-1 offset (a symmetric circle yields 35.6/23.3, not ~30/~30), so they stay regression.
	{"DIAMETER_MIN_ENCLOSING_CIRCLE", 40.00019836425781},
};

static void assert_iso_transform_imea(const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature, const std::string& feature_name, double frac_tolerance = 1000.0)
{
	SCOPED_TRACE(std::string("IMEA_ORACLE__") + feature_name);
	ASSERT_TRUE(morphology_2d_imea_shape2d_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_imea_shape2d_ref_vals.at(feature_name), frac_tolerance));
}

static void assert_caliper_close_to_imea(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	// Measured residuals against the dalpha=10 goldens: Martin 1.4-4.6%, Nassenstein 0.3-3.8%,
	// Feret 1.3-5.0%. The floor is STAT_FERET_DIAM_STDDEV at 4.99% -- Nyxus' Feret sweep runs
	// theta = 0..180 inclusive (caliper_feret.cpp) so it counts the 0/180 direction twice, which
	// tilts the spread slightly; Martin and Nassenstein stop at theta < 180. That is a definitional
	// sampling difference, not a precision loss, so 0.06 is the honest bound.
	double reltol = 0.06)
{
	SCOPED_TRACE(std::string("CALIPER_VS_IMEA__") + feature_name);
	ASSERT_TRUE(morphology_2d_imea_ellipse_ref_vals.count(feature_name) > 0);
	const double imea_ref = morphology_2d_imea_ellipse_ref_vals.at(feature_name);
	const double actual = fvals[static_cast<int>(feature)][0];
	const double denom = std::max(std::abs(imea_ref), 1e-9);
	ASSERT_LE(std::abs(actual - imea_ref) / denom, reltol)
		<< feature_name << " nyxus=" << actual << " imea=" << imea_ref;
}

// The caliper spreads and the shortest chord, on the clean ellipse where imea and Nyxus are
// comparable. The MODE statistics are deliberately absent: imea's own mode is an artifact of its
// angular step (19..24 across dalpha 5..30), so it cannot be vetted at any honest tolerance and
// stays a regression row.
void test_2d_morphology_caliper_spread_imea()
{
	std::vector<std::vector<double>> fvals;
	calculate_ellipse_caliper_values(fvals);

	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_FERET_DIAM_STDDEV, "STAT_FERET_DIAM_STDDEV");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_MARTIN_DIAM_STDDEV, "STAT_MARTIN_DIAM_STDDEV");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::STAT_NASSENSTEIN_DIAM_STDDEV, "STAT_NASSENSTEIN_DIAM_STDDEV");
	assert_caliper_close_to_imea(fvals, Nyxus::Feature2D::ALLCHORDS_MIN, "ALLCHORDS_MIN", 1e-9);
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

	// imea 0.3.5: macro.perimeter_equal_diameter(26.9349412836191). Read from the table rather than
	// repeated inline -- the two copies had already drifted (8.57365809435587 vs ...881), and only
	// the copy an assertion reads is under test.
	assert_iso_transform_imea(fvals, Nyxus::Feature2D::DIAMETER_EQUAL_PERIMETER, "DIAMETER_EQUAL_PERIMETER");
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
	assert_iso_transform_imea(fvals, Nyxus::Feature2D::GEODETIC_LENGTH, "GEODETIC_LENGTH");
	assert_iso_transform_imea(fvals, Nyxus::Feature2D::THICKNESS, "THICKNESS");
}

// Vets the reimplemented Martin (area-bisecting chord) and Nassenstein (bottom-tangent vertical
// chord) diameters against imea on a clean filled ellipse (a=20, b=10). See the oracle block in
// test_2d_morphology_common.h. Robust stats (min/max/mean/median) agree with imea within the
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
