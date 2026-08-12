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
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_2d_remaining_common.h"   // fixture: calculate_remaining2d_shape_feature_values
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

static void assert_unvetted_no_direct_oracle_remaining2d_vector_feature(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double abs_tolerance = 1e-9)
{
	SCOPED_TRACE(std::string("UNVETTED_NO_DIRECT_ORACLE__") + feature_name);
	ASSERT_TRUE(radial_2d_regression_ref_vals.count(feature_name) > 0);
	const auto& actual = fvals[static_cast<int>(feature)];
	const auto& golden_values = radial_2d_regression_ref_vals[feature_name];
	ASSERT_EQ(actual.size(), golden_values.size());
	for (size_t i = 0; i < golden_values.size(); ++i)
		ASSERT_NEAR(actual[i], golden_values[i], abs_tolerance) << feature_name << "[" << i << "]";
}

// ---------------------------------------------------------------------------------------------------
// Migrated from test_2d_remaining_features.h (Wave 6): radial intensity distribution (FRAC_AT_D,
// MEAN_FRAC, RADIAL_CV). Registry target_test = test_2d_intensity_histogram_regression.h; oracle is
// cellprofiler MeasureObjectIntensityDistribution (RadialDistribution_*), still regression-snapshot
// pending wiring. Shared fixture/oracle-data lives in test_2d_remaining_common.h.
// ---------------------------------------------------------------------------------------------------

void test_2d_radial_distribution_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_remaining2d_shape_feature_values(fvals);

	assert_unvetted_no_direct_oracle_remaining2d_vector_feature(fvals, Nyxus::Feature2D::FRAC_AT_D, "FRAC_AT_D");
	assert_unvetted_no_direct_oracle_remaining2d_vector_feature(fvals, Nyxus::Feature2D::MEAN_FRAC, "MEAN_FRAC");
	assert_unvetted_no_direct_oracle_remaining2d_vector_feature(fvals, Nyxus::Feature2D::RADIAL_CV, "RADIAL_CV");
}
