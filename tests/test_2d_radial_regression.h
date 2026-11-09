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
// Two cases live here. The first asserts the 24 pins. The second characterizes the conventions those
// pins are computed under - properties that hold only because FRAC_AT_D counts pixels and MEAN_FRAC
// is a raw bin mean, so they belong beside the snapshots rather than in test_2d_radial_invariant.h,
// which takes only the properties a re-definition would preserve.
//
// Provenance (SPEC 6.4): Nyxus' own output on recipe radial.shape2d_native, at full %.17g precision.
// `gen_radial_cellprofiler.py --skip-cellprofiler` reproduces all 24 from a written-down model of the
// implementation and needs no build and no oracle environment.

#include <gtest/gtest.h>

#include <algorithm>                    // std::min, std::max

#include "../src/nyx/featureset.h"      // Feature2D
#include "test_2d_radial_common.h"      // build_radial_2d_roi, calculate_radial_2d_values, LR,
                                        // RadialDistributionFeature
#include "test_main_nyxus.h"            // agrees_gt, <cmath> for std::round
#include "test_ref_vals.h"              // ref_vals_map, <string>, <vector>

static const ref_vals_map<std::vector<double>> radial_2d_regression_ref_vals{
	{"FRAC_AT_D", {
		0.038461538460059175, 0.0, 0.11538461538017751, 0.1538461538402367,
		0.38461538460059169, 0.0, 0.1538461538402367, 0.1538461538402367,
	}},
	{"MEAN_FRAC", {
		39.999999959999997, 0.0, 39.666666653444445, 41.249999989687502,
		38.199999996179997, 0.0, 37.999999990500001, 47.499999988124998,
	}},
	{"RADIAL_CV", {
		2.6457513105354402, 0.0, 1.3089567218937439, 1.0287867630146279,
		0.60432094511523338, 0.0, 1.2454904805869986, 1.8152630815185364,
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

// The conventions the 24 pins above are computed under, asserted of the values a build produces
// rather than of the literals. These are characterization, not invariants (SPEC 2): each one holds
// because of a choice this implementation makes, and a re-definition of the feature that the
// CellProfiler comparison would call correct breaks it. FRAC_AT_D counting pixels is what makes
// every entry a whole-pixel fraction; a fraction of intensity would not be. MEAN_FRAC being the
// bin's raw mean intensity is what puts it inside the ROI's intensity range and what makes the two
// tables reconstruct the ROI total; CellProfiler's MeanFrac, normalised by the ROI mean, lands near
// 1 and does neither. So they are drift guards on the current definitions and nothing more - if
// they ever fail, read the goldens above and the report before assuming a regression.
void test_2d_radial_bin_conventions_regression()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];

	ASSERT_EQ(frac.size(), (size_t)RadialDistributionFeature::num_bins) << "FRAC_AT_D";
	ASSERT_EQ(mean.size(), (size_t)RadialDistributionFeature::num_bins) << "MEAN_FRAC";

	const double n_pixels = double(roidata.raw_pixels.size());
	double lo = roidata.raw_pixels[0].inten, hi = roidata.raw_pixels[0].inten, total = 0.0;
	for (const auto& p : roidata.raw_pixels)
	{
		lo = std::min(lo, (double)p.inten);
		hi = std::max(hi, (double)p.inten);
		total += (double)p.inten;
	}

	// FRAC_AT_D divides a pixel count by a pixel count, so every entry is a whole count over n
	double reconstructed = 0.0;
	for (size_t i = 0; i < frac.size(); i++)
	{
		const double count = frac[i] * n_pixels;
		ASSERT_NEAR(count, std::round(count), 1e-6) << "FRAC_AT_D[" << i << "] is not a whole count";

		// MEAN_FRAC is a mean of raw intensities, so a non-empty bin lands inside the ROI's range
		if (frac[i] != 0.0)
		{
			ASSERT_GE(mean[i], lo) << "MEAN_FRAC[" << i << "]";
			ASSERT_LE(mean[i], hi) << "MEAN_FRAC[" << i << "]";
		}

		reconstructed += std::round(count) * mean[i];
	}

	// and the two tables together are a partition of the ROI's intensity
	ASSERT_TRUE(Nyxus::agrees_gt(reconstructed, total, 1e9))
		<< "FRAC_AT_D and MEAN_FRAC do not reconstruct the ROI intensity";
}
