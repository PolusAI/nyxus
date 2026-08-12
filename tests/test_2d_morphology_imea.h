#pragma once

#include "test_2d_morphology_common.h"
#include "test_2d_remaining_common.h"   // morphology_2d_imea_ref_vals + assert_caliper_close_to_imea

#include "test_ref_vals.h"

static ref_vals_map<double> morphology_2d_imea_geodetic_ref_vals{
	{"DIAMETER_EQUAL_PERIMETER", 8.57365809435587},
	// Real-valued rectangle-model roots after the geo_len_thickness.cpp perimeter-truncation fix
	// (were the integer-truncated 10.0 / 3.0). Now vetted against imea's geodeticlength_and_thickness
	// in test_2d_morphology_imea.h.
	{"GEODETIC_LENGTH", 11.13182483477333},
	{"THICKNESS", 2.3356458070362205},
};

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
