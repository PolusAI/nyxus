#pragma once

// ZERNIKE2D drift guard - Nyxus' own output, pinned at full precision. The correctness claim lives
// in test_2d_zernike_analytic.h, which compares the same 30 magnitudes against the closed form;
// this file only catches movement.
//
// Provenance (SPEC 6.4): Nyxus' output on recipe zernike.shape2d_native at %.17g. Reproduce with
// tests/vetting/oracles/gen_zernike_analytic.py, which computes the same values from the factorial
// series and reports the residual.
//
// The 30 values are one magnitude per (n, m) index with n <= 9, m >= 0 and n - m even, emitted in
// the order zernike.cpp walks them: n ascending, then m ascending. Every entry is asserted
// separately, so a change confined to one index cannot hide inside the rest.

#include <gtest/gtest.h>

#include "../src/nyx/featureset.h"      // Feature2D
#include "test_2d_zernike_common.h"     // calculate_zernike_2d_values;
                                        // <cmath> for std::fabs via test_main_nyxus.h
#include "test_ref_vals.h"              // ref_vals_map, <string>, <vector>

static const ref_vals_map<std::vector<double>> zernike_2d_regression_ref_vals{
	{"ZERNIKE2D", {
		0.31830988618379069, 1.2516784214709243e-17, 0.71985225389069296,
		0.025780497111119398, 0.020669804576956517, 0.02009003454861746,
		0.62194689390734237, 0.086068705661162842, 0.006242585064603394,
		0.082381409770710498, 0.09366075496376644, 0.0047976324379789157,
		0.17461460409333135, 0.12084127682551971, 0.031962211880315075,
		0.0025827535820692675, 0.14801658671869741, 0.21122543105565181,
		0.026366443634871305, 0.00080970804775625372, 0.24088546254504473,
		0.07696640214842293, 0.081733817517752896, 0.01537711025337163,
		0.00035143072544742014, 0.1530929475694654, 0.29425534815914151,
		0.069853801817987116, 0.0048691281137569938, 0.00010157413670881729,
	}},
};

// rel=1e-9: Nyxus' own output pinned to full precision, and a fresh build reproduces all 30 exactly,
// so a drift guard should catch any change at all.
//
// Same assertion shape as test_2d_zernike_analytic.h -- ASSERT_NEAR against |golden|*rel + floor --
// rather than the house agrees_gt, and for the same reason: entry 1 is (n=1,m=1), whose TRUE value is
// zero, because it is the first moment about the centre the feature centres on. What is pinned is the
// ~1.25e-17 of rounding left over from cancelling terms of order 1, and that residue is a property of
// the summation order rather than of the feature. Both toolchains happen to produce the same one
// today; a purely relative band would be asserting that they always will. The floor sits far below
// the smallest real magnitude in the table, 1.0e-4.
static const double ZERNIKE_REGRESSION_REL_TOL = 1e-9;
static const double ZERNIKE_REGRESSION_ABS_FLOOR = 1e-15;

static void assert_zernike_vector_feature_regression(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(zernike_2d_regression_ref_vals.count(feature_name) > 0);
	const auto& actual = fvals[static_cast<int>(feature)];
	const auto& golden_values = zernike_2d_regression_ref_vals.at(feature_name);
	ASSERT_EQ(actual.size(), golden_values.size());
	for (size_t i = 0; i < golden_values.size(); ++i)
		ASSERT_NEAR(actual[i], golden_values[i],
			std::fabs(golden_values[i]) * ZERNIKE_REGRESSION_REL_TOL + ZERNIKE_REGRESSION_ABS_FLOOR)
			<< feature_name << "[" << i << "]";
}

void test_2d_zernike_moments_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_zernike_2d_values(fvals);

	assert_zernike_vector_feature_regression(fvals, Nyxus::Feature2D::ZERNIKE2D, "ZERNIKE2D");
}
