#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/zernike.h"
#include "test_data.h"
#include "test_main_nyxus.h"   // shared config: make_shape2d_settings
#include "test_ref_vals.h"

static ref_vals_map<std::vector<double>> zernike_2d_regression_ref_vals{
	{"ZERNIKE2D", {
		0.02049738595695693, 0.035831084484416686, 0.073953766599300461,
		0.035435050265597692, 0.092323797445497555, 0.011030627605166297,
		0.13199834370886107, 0.13453286019693309, 0.00788523106321295,
		0.082424064819857396, 0.049062071772591059, 0.0040585552756590825,
		0.14488178557089382, 0.23625456011991602, 0.038032570269059741,
		0.0011694758904577424, 0.016507094944884948, 0.10703041567067684,
		0.021302528534918392, 0.00061791897183974015, 0.10313303720229962,
		0.23275354391334316, 0.08692094259111556, 0.0063362223871874139,
		0.00016460740533666494, 0.085700825034398798, 0.15183975656312645,
		0.052012830525298454, 0.0045112452293896111, 0.00015124210515210458,
	}},
};

static void assert_zernike_vector_feature_regression(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double abs_tolerance = 1e-9)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(zernike_2d_regression_ref_vals.count(feature_name) > 0);
	const auto& actual = fvals[static_cast<int>(feature)];
	const auto& golden_values = zernike_2d_regression_ref_vals[feature_name];
	ASSERT_EQ(actual.size(), golden_values.size());
	for (size_t i = 0; i < golden_values.size(); ++i)
		ASSERT_NEAR(actual[i], golden_values[i], abs_tolerance) << feature_name << "[" << i << "]";
}


// The ROI ZERNIKE2D is measured on: the 8x8 shape2d fixture under the shared shape config.
// ZernikeFeature reads the ROI image matrix and nothing any other feature produces, so the loader
// is the whole prerequisite -- this file builds its own ROI rather than borrowing the morphology
// fixture (SPEC 6.3.1).
static void calculate_zernike_moment_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_shape2d_settings();

	LR roidata(101);
	load_masked_test_roi_data(
		roidata,
		shape2d_morphology_intensity,
		shape2d_morphology_mask,
		sizeof(shape2d_morphology_mask) / sizeof(NyxusPixel));
	roidata.initialize_fvals();

	ZernikeFeature zernike;
	zernike.calculate(roidata, s);
	zernike.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

// ---------------------------------------------------------------------------------------------------
// Migrated from test_2d_remaining_features.h (Wave 6): ZERNIKE2D. Per registry decision (§6.1) mahotas
// is not an accepted oracle, so ZERNIKE2D stays analytic/regression -> test_2d_zernike_regression.h.
// The ROI it measures is built above; the shared config is in test_main_nyxus.h.
// ---------------------------------------------------------------------------------------------------

void test_2d_zernike_moments_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_zernike_moment_values(fvals);

	assert_zernike_vector_feature_regression(fvals, Nyxus::Feature2D::ZERNIKE2D, "ZERNIKE2D");
}
