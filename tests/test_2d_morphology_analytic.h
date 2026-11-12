#pragma once

#include <cmath>
#include <vector>

#include "test_2d_morphology_common.h"

// Documented-formula conformance (oracle=analytic, SPEC 4). These features have a recognized closed
// form but their VALUE uses Nyxus' own conventions (pixel-count area, contour perimeter, moment-fit
// major axis), so no third-party tool reproduces the number. What we CAN pin is that the code applies
// the published formula to its own constituents without an implementation bug -- recompute the formula
// from AREA_PIXELS_COUNT / PERIMETER / MAJOR_AXIS_LENGTH and require an exact match. This is weaker
// than external-oracle vetting, and it is a correctness claim, so per SPEC 2 it lives in this
// `_analytic` oracle file rather than in test_2d_morphology_regression.h (which claims nothing).
void test_2d_morphology_documented_formula_conformance_analytic()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	const double PI = 3.14159265358979323846;
	const double A = fvals[static_cast<int>(Nyxus::Feature2D::AREA_PIXELS_COUNT)][0];
	const double P = fvals[static_cast<int>(Nyxus::Feature2D::PERIMETER)][0];
	const double major = fvals[static_cast<int>(Nyxus::Feature2D::MAJOR_AXIS_LENGTH)][0];

	// CIRCULARITY = sqrt(4*pi*A) / P   (convex_hull_nontriv.cpp)
	const double circ_formula = std::sqrt(4.0 * PI * A) / P;
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::CIRCULARITY)][0], circ_formula, 1e-9)
		<< "CIRCULARITY does not match sqrt(4*pi*A)/P";

	// ROUNDNESS = 4*A / (pi*major^2)   (ellipse_fitting.cpp)
	const double round_formula = 4.0 * A / (PI * major * major);
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::ROUNDNESS)][0], round_formula, 1e-9)
		<< "ROUNDNESS does not match 4A/(pi*major^2)";

	// DIAMETER_EQUAL_PERIMETER (= P/pi) is vetted against the third-party imea implementation of the
	// same ISO transform in test_2d_morphology_imea.h, so it is not re-derived here. GEODETIC_LENGTH and
	// THICKNESS (the rectangle-model roots P/4 +- sqrt(P^2/16 - A)) are vetted the same way, against
	// imea's geodeticlength_and_thickness, so they are not re-derived here either.
}

// ROI_RADIUS_MAX against the closed form for a filled digital disk (oracle=analytic, SPEC 4), on the
// same `bench_radius_disks` fixture test_2d_morphology_skimage.h pins against scikit-image. Two
// independent oracles for one feature is the redundancy SPEC 3.1 tracks, and here they check
// genuinely different things: the skimage row says Nyxus agrees with a reference implementation of
// the same definition, this one says the value obeys the geometry regardless of any tool.
//
// THE CLOSED FORM IS EXACT, not a band. ROI_RADIUS_MAX is the largest distance from any ROI pixel to
// the ROI's boundary, and on a disk the centre attains it, so the value is the centre's distance to
// the NEAREST boundary pixel. That pixel is not the axial (R, 0) at distance R -- it is the pixel at
// offset (1, R-1):
//
//   it is inside      1 + (R-1)^2 <= R^2  <=>  2 <= 2R,  true for every R >= 1
//   it is boundary    its neighbour (1, R) has 1 + R^2 > R^2, so that one is background
//   and it is nearer  (R-1)^2 + 1 = R^2 - 2R + 2 < R^2  for every R > 1
//
// so  ROI_RADIUS_MAX == sqrt((R-1)^2 + 1)  exactly. Verified at R = 5, 10, 20, 40, 80 and 160: the
// nearest boundary pixel is at (1, R-1) at every one of them.
//
// That form is also why MAX/(R-1) = sqrt(1 + 1/(R-1)^2) converges to 1 from above as 1/(2(R-1)^2) --
// 1.006154, 1.001384, 1.000329 at R = 10, 20, 40 -- i.e. MAX grows LINEARLY in R.
//
// What the assertion discriminates: a squared distance would read 82, 362 and 1522 here, ratios to
// R-1 of 9.1, 19.1 and 39.0 that grow instead of converging. One disk cannot tell a distance from
// its square, since either is just a number; the exact form separates them at every radius.
void test_2d_morphology_roi_radius_disk_closed_form_analytic()
{
	for (double R : {10.0, 20.0, 40.0})
	{
		std::vector<std::vector<double>> fvals;
		calculate_disk_radius_values(R, fvals);

		// The nearest boundary pixel to the centre, derived above. Both sides are the square root of
		// the same integer, so this is an equality and not an approximation; the tolerance is
		// float-representation slack, not a convention band.
		const double expected = std::sqrt((R - 1.0) * (R - 1.0) + 1.0);

		SCOPED_TRACE(std::string("ANALYTIC_ORACLE__ROI_RADIUS_MAX__R") + std::to_string((int)R));

		// The feature is named on the assertion line rather than only on the readout:
		// scan_morphology_coverage.py credits a feature from the ASSERTION line and deliberately
		// does not count a line that merely reads a value out of the buffer.
		ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::ROI_RADIUS_MAX)][0], expected, 1e-12 * expected)
			<< "ROI_RADIUS_MAX is not sqrt((R-1)^2+1) = " << expected << " at R = " << R
			<< " -- it is not the distance from the disk's centre to its boundary";
	}
}

// DIAMETER_INSCRIBING_CIRCLE and DIAMETER_CIRCUMSCRIBING_CIRCLE on the r=15 circle
// (calculate_circle_shape_values): twice the smallest and the largest distance from the centroid to
// a contour pixel. The centroid is the centre (21, 21) by symmetry, the contour pixels are the disk's
// edge pixels, and both extremes have closed forms:
//
//   farthest   the axial pixel (R, 0) is inside (R^2 <= R^2) and on the edge, and no disk pixel is
//              farther than R, so the circumscribing diameter is 2R
//   nearest    the edge pixel at offset (1, R-1), derived for ROI_RADIUS_MAX above, so the
//              inscribing diameter is 2 sqrt((R-1)^2 + 1)
//
// Both depend on the contour and the centroid being in one frame: shifting either by a pixel moves
// the two diameters apart by up to 2 sqrt(2).
void test_2d_morphology_circle_diameters_closed_form_analytic()
{
	const double R = 15.0;
	std::vector<std::vector<double>> fvals;
	calculate_circle_shape_values(fvals);

	// the precondition the closed forms rest on, not a claim about the centroid features
	const double cx = fvals[static_cast<int>(Nyxus::Feature2D::CENTROID_X)][0],
		cy = fvals[static_cast<int>(Nyxus::Feature2D::CENTROID_Y)][0];
	ASSERT_EQ(cx, 21.0);
	ASSERT_EQ(cy, 21.0);

	SCOPED_TRACE("ANALYTIC_ORACLE__CIRCLE_DIAMETERS__R15");
	const double circ = 2.0 * R, insc = 2.0 * std::sqrt((R - 1.0) * (R - 1.0) + 1.0);
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE)][0], circ, 1e-12 * circ);
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_INSCRIBING_CIRCLE)][0], insc, 1e-12 * insc);
}

// The two diameters are measured to the ROI's outer borders only, the external contour imea reads:
// a hole's border is not part of the outline. The r=15 disk of the case above with a centred 3x3
// hole keeps its centroid at the centre by symmetry and its outer border unchanged, so both closed
// forms hold as they do for the solid disk. The hole's border is 2 from the centre, and measuring
// to it would give an inscribing diameter of 4.
void test_2d_morphology_circle_diameters_holed_disk_analytic()
{
	const double R = 15.0;
	std::vector<std::vector<double>> fvals;
	calculate_disks_circle_values({ { 21, 21 } }, R, 1, fvals);

	// the precondition the closed forms rest on, not a claim about the centroid features
	const double cx = fvals[static_cast<int>(Nyxus::Feature2D::CENTROID_X)][0],
		cy = fvals[static_cast<int>(Nyxus::Feature2D::CENTROID_Y)][0];
	ASSERT_EQ(cx, 21.0);
	ASSERT_EQ(cy, 21.0);

	SCOPED_TRACE("ANALYTIC_ORACLE__CIRCLE_DIAMETERS__R15_HOLED");
	const double circ = 2.0 * R, insc = 2.0 * std::sqrt((R - 1.0) * (R - 1.0) + 1.0);
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE)][0], circ, 1e-12 * circ);
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_INSCRIBING_CIRCLE)][0], insc, 1e-12 * insc);
}

// A ROI of several fragments has one outer border per fragment, and the diameters are measured from
// the ROI's centroid to all of them. Two disks of radius 10 with centres D = 40 apart have their
// centroid midway; the nearest outer-border pixel is the axial one facing it, D/2 - R away, and the
// farthest the axial one opposite, D/2 + R away, so the diameters are D - 2R = 20 and D + 2R = 60.
void test_2d_morphology_circle_diameters_two_fragments_analytic()
{
	const double R = 10.0, D = 40.0;
	std::vector<std::vector<double>> fvals;
	calculate_disks_circle_values({ { 13, 13 }, { 53, 13 } }, R, -1, fvals);

	// the precondition the closed forms rest on, not a claim about the centroid features
	const double cx = fvals[static_cast<int>(Nyxus::Feature2D::CENTROID_X)][0],
		cy = fvals[static_cast<int>(Nyxus::Feature2D::CENTROID_Y)][0];
	ASSERT_EQ(cx, 33.0);
	ASSERT_EQ(cy, 13.0);

	SCOPED_TRACE("ANALYTIC_ORACLE__CIRCLE_DIAMETERS__TWO_R10_DISKS");
	const double circ = D + 2.0 * R, insc = D - 2.0 * R;
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_CIRCUMSCRIBING_CIRCLE)][0], circ, 1e-12 * circ);
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_INSCRIBING_CIRCLE)][0], insc, 1e-12 * insc);
}
