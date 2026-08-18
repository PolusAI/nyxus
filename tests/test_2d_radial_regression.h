#pragma once

// Radial intensity distribution (FRAC_AT_D, MEAN_FRAC, RADIAL_CV) -- family=radial in the registry.
// These lived in test_2d_intensity_histogram_regression.h, which meant a radial_* table and a
// radial-named function sat in a file named for a different family. SPEC 6.3.1 now requires a table
// to live with its assertions, and the honest home for both is a radial file.
//
// Snapshots, not vetting (SPEC 1): the candidate oracle is CellProfiler
// MeasureObjectIntensityDistribution (RadialDistribution_*), not yet wired.

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/radial_distribution.h"
#include "test_data.h"
#include "test_main_nyxus.h"   // shared config: make_shape2d_settings
#include "test_ref_vals.h"

static ref_vals_map<std::vector<double>> radial_2d_regression_ref_vals{
	{"FRAC_AT_D", {
		0.038461538460059175, 0.0, 0.11538461538017751, 0.1538461538402367,
		0.3076923076804734, 0.0, 0.11538461538017751, 0.26923076922041422,
	}},
	{"MEAN_FRAC", {
		50.999999948999999, 0.0, 53.333333315555556, 50.749999987312499,
		47.374999994078124, 0.0, 33.666666655444445, 21.999999996857142,
	}},
	{"RADIAL_CV", {
		2.6457513106495707, 0.0, 1.298797520721114, 1.024429214739045,
		0.64750329537582818, 0.0, 1.3575192606324717, 1.3284260624865412,
	}},
};

static void assert_radial_vector_feature_regression(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double abs_tolerance = 1e-9)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(radial_2d_regression_ref_vals.count(feature_name) > 0);
	const auto& actual = fvals[static_cast<int>(feature)];
	const auto& golden_values = radial_2d_regression_ref_vals[feature_name];
	ASSERT_EQ(actual.size(), golden_values.size());
	for (size_t i = 0; i < golden_values.size(); ++i)
		ASSERT_NEAR(actual[i], golden_values[i], abs_tolerance) << feature_name << "[" << i << "]";
}

// The ROI these three features are measured on: the 8x8 shape2d fixture under the shared shape
// config. RadialDistributionFeature reads the ROI contour, so ContourFeature runs first; nothing
// else in the shape set feeds it, which is why this file builds its own ROI instead of borrowing
// the morphology fixture (SPEC 6.3.1).
static void calculate_radial_distribution_values(std::vector<std::vector<double>>& fvals)
{
	Fsettings s = make_shape2d_settings();

	LR roidata(101);
	load_masked_test_roi_data(
		roidata,
		shape2d_morphology_intensity,
		shape2d_morphology_mask,
		sizeof(shape2d_morphology_mask) / sizeof(NyxusPixel));
	roidata.initialize_fvals();

	ContourFeature contour;
	contour.calculate(roidata, s);
	contour.save_value(roidata.fvals);

	RadialDistributionFeature radial;
	radial.calculate(roidata, s);
	radial.save_value(roidata.fvals);

	fvals = roidata.fvals;
}

// ---------------------------------------------------------------------------------------------------
// Migrated from test_2d_remaining_features.h (Wave 6): radial intensity distribution (FRAC_AT_D,
// MEAN_FRAC, RADIAL_CV). Registry target_test = test_2d_intensity_histogram_regression.h; oracle is
// cellprofiler MeasureObjectIntensityDistribution (RadialDistribution_*), still regression-snapshot
// pending wiring. The ROI it measures is built above; the shared config is in test_main_nyxus.h.
// ---------------------------------------------------------------------------------------------------

void test_2d_radial_distribution_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_radial_distribution_values(fvals);

	assert_radial_vector_feature_regression(fvals, Nyxus::Feature2D::FRAC_AT_D, "FRAC_AT_D");
	assert_radial_vector_feature_regression(fvals, Nyxus::Feature2D::MEAN_FRAC, "MEAN_FRAC");
	assert_radial_vector_feature_regression(fvals, Nyxus::Feature2D::RADIAL_CV, "RADIAL_CV");
}
