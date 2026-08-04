#pragma once

#include "test_morphology_common.h"

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
void test_shape2d_diameter_equal_perimeter_imea()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	// imea 0.3.5: macro.perimeter_equal_diameter(26.9349412836191)
	const double imea_dep = 8.573658094355881;
	ASSERT_NEAR(fvals[static_cast<int>(Nyxus::Feature2D::DIAMETER_EQUAL_PERIMETER)][0], imea_dep, 1e-9)
		<< "DIAMETER_EQUAL_PERIMETER does not match imea perimeter_equal_diameter(PERIMETER)";
}
