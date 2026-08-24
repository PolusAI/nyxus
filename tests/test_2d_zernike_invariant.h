#pragma once

// SPEC 4.4 invariants for ZERNIKE2D. Every one is forced by the definition of a Zernike moment, so
// they hold for any ROI and claim no oracle. The first two are identities rather than bounds -- a
// moment set whose angular term is off by one repetition fails both, and neither needs a reference
// implementation to state, so they catch a whole class of defect for three lines each.

#include <gtest/gtest.h>

#include "../src/nyx/featureset.h"       // Feature2D
#include "test_2d_zernike_common.h"      // build_zernike_2d_roi, LR, ZernikeFeature
#include "test_main_nyxus.h"             // <cmath> for std::fabs and std::acos

// pi from the library rather than M_PI: this header is included well into test_all.cc's translation
// unit, by which point <cmath> has already been pulled in, so a _USE_MATH_DEFINES here would come
// too late to make the macro appear on MSVC.
static const double ZERNIKE_PI = std::acos(-1.0);

// A_00 is the whole distribution integrated against R_00 = 1, so it is (0+1)/pi times the sum of the
// weights. The weights are I / sum(I) over the pixels inside the unit disk, so when every pixel is
// inside -- which test_2d_zernike_every_pixel_is_inside_the_unit_disk_mechanics asserts for this
// fixture -- they sum to exactly 1 and A_00 is exactly 1/pi, whatever the image contains.
void test_2d_zernike_zeroth_moment_is_one_over_pi_invariant()
{
	LR roidata(101);
	build_zernike_2d_roi(roidata);
	const auto& z = roidata.fvals[(int)Nyxus::Feature2D::ZERNIKE2D];

	ASSERT_EQ(z.size(), (size_t)ZernikeFeature::NUM_FEATURE_VALS) << "ZERNIKE2D";
	ASSERT_NEAR(z[0], 1.0 / ZERNIKE_PI, 1e-14) << "ZERNIKE2D[0] is A(0,0)";
}

// A_11 integrates the distribution against R_11 = r times exp(-i theta), i.e. it IS the first moment
// about the point the unit disk is centred on. mb_zernike2D centres on the intensity centroid, so
// that first moment is zero by construction and A_11 must vanish.
void test_2d_zernike_first_moment_about_the_centroid_vanishes_invariant()
{
	LR roidata(101);
	build_zernike_2d_roi(roidata);
	const auto& z = roidata.fvals[(int)Nyxus::Feature2D::ZERNIKE2D];

	ASSERT_LT(std::fabs(z[1]), 1e-14) << "ZERNIKE2D[1] is A(1,1)";
}

// |R_nm(r)| <= 1 on the unit disk and the weights sum to at most 1, so |A_nm| <= (n+1)/pi. Every
// magnitude is also non-negative, being the modulus of a complex number.
void test_2d_zernike_magnitudes_are_within_their_bound_invariant()
{
	LR roidata(101);
	build_zernike_2d_roi(roidata);
	const auto& z = roidata.fvals[(int)Nyxus::Feature2D::ZERNIKE2D];

	ASSERT_EQ(z.size(), (size_t)ZernikeFeature::NUM_FEATURE_VALS) << "ZERNIKE2D";

	size_t k = 0;
	for (int n = 0; n <= ZernikeFeature::ZERNIKE2D_ORDER; n++)
		for (int m = 0; m <= n; m++)
			if ((n - m) % 2 == 0)
			{
				ASSERT_GE(z[k], 0.0) << "ZERNIKE2D[" << k << "] is A(" << n << "," << m << ")";
				ASSERT_LE(z[k], (n + 1) / ZERNIKE_PI * (1.0 + 1e-12))
					<< "ZERNIKE2D[" << k << "] is A(" << n << "," << m << ")";
				k++;
			}
	ASSERT_EQ(k, z.size()) << "ZERNIKE2D index walk";
}

// The 30 values are the (n, m) pairs with n <= 9, m >= 0 and n - m even. That count is what
// NUM_FEATURE_VALS claims, and it is what any consumer indexing the vector by moment relies on.
void test_2d_zernike_index_set_matches_the_declared_count_invariant()
{
	int count = 0;
	for (int n = 0; n <= ZernikeFeature::ZERNIKE2D_ORDER; n++)
		for (int m = 0; m <= n; m++)
			if ((n - m) % 2 == 0)
				count++;
	ASSERT_EQ(count, (int)ZernikeFeature::NUM_FEATURE_VALS) << "ZERNIKE2D index set";

	LR roidata(101);
	build_zernike_2d_roi(roidata);
	ASSERT_EQ(roidata.fvals[(int)Nyxus::Feature2D::ZERNIKE2D].size(), (size_t)count) << "ZERNIKE2D";
}
