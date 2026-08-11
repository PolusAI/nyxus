#pragma once

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
