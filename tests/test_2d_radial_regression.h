#pragma once

// Radial intensity distribution (FRAC_AT_D, MEAN_FRAC, RADIAL_CV) - family=radial in the registry.
//
// Snapshots, not vetting (SPEC 1). The candidate oracle, CellProfiler
// MeasureObjectIntensityDistribution (RadialDistribution_*), has been run on this fixture and does
// NOT reproduce these three features: Nyxus computes a different quantity under each of the three
// CellProfiler names. The run, the numbers and the six divergences are in
// tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md; regenerate with
// tests/vetting/oracles/gen_radial_cellprofiler.py. The family therefore stays regression.
//
// Each of the three features is a vector of 8 radial bins and every bin is pinned separately, so a
// change confined to one bin cannot hide inside an aggregate.
//
// Provenance (SPEC 6.4): Nyxus' own output on recipe radial.shape2d_native, at full %.17g precision.
// `gen_radial_cellprofiler.py --skip-cellprofiler` reproduces all 24 from a written-down model of the
// implementation and needs no build and no oracle environment.

#include <gtest/gtest.h>

#include "../src/nyx/featureset.h"      // Feature2D
#include "test_2d_radial_common.h"      // calculate_radial_2d_values
#include "test_main_nyxus.h"            // agrees_gt
#include "test_ref_vals.h"              // ref_vals_map, <string>, <vector>

static const ref_vals_map<std::vector<double>> radial_2d_regression_ref_vals{
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

// frac_tolerance = 1e9, i.e. rel=1e-9. These are Nyxus' own output pinned to full precision and the
// measured residual against a fresh build is 0 (report §2), so a drift guard should catch any
// change at all. An empty bin is an exact 0.0 in all three features, which agrees_gt enforces
// exactly: a zero ground truth gives a zero tolerance.
static void assert_radial_vector_feature_regression(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name);
	ASSERT_TRUE(radial_2d_regression_ref_vals.count(feature_name) > 0);
	const auto& actual = fvals[static_cast<int>(feature)];
	const auto& golden_values = radial_2d_regression_ref_vals.at(feature_name);
	ASSERT_EQ(actual.size(), golden_values.size());
	for (size_t i = 0; i < golden_values.size(); ++i)
		ASSERT_TRUE(Nyxus::agrees_gt(actual[i], golden_values[i], 1e9))
			<< feature_name << "[" << i << "]";
}

void test_2d_radial_distribution_regression()
{
	std::vector<std::vector<double>> fvals;
	calculate_radial_2d_values(fvals);

	assert_radial_vector_feature_regression(fvals, Nyxus::Feature2D::FRAC_AT_D, "FRAC_AT_D");
	assert_radial_vector_feature_regression(fvals, Nyxus::Feature2D::MEAN_FRAC, "MEAN_FRAC");
	assert_radial_vector_feature_regression(fvals, Nyxus::Feature2D::RADIAL_CV, "RADIAL_CV");
}
