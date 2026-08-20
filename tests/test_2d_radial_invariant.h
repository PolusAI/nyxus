#pragma once

// SPEC 4.4 invariants for the 2D radial intensity distribution. These are properties the three
// features must satisfy for any ROI whatever the radial binning does, so they hold independently of
// the pinned snapshots in test_2d_radial_regression.h and of the CellProfiler divergences recorded
// in tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md. They claim no oracle.

#include <gtest/gtest.h>

#include <algorithm>                 // std::min, std::max

#include "../src/nyx/featureset.h"   // Feature2D
#include "test_2d_radial_common.h"   // build_radial_2d_roi, LR, RadialDistributionFeature
#include "test_main_nyxus.h"         // agrees_gt, <cmath> for std::round and std::sqrt

// FRAC_AT_D is a partition of the ROI's pixels over the radial bins, so every entry is a fraction of
// a whole pixel count and the entries sum to one. The sum falls short of exactly 1 by the feature's
// own epsilon = 1e-9 guard in the denominator, hence the 1e-9 band rather than an exact comparison.
void test_2d_radial_frac_at_d_is_a_partition_invariant()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];

	ASSERT_EQ(frac.size(), (size_t)RadialDistributionFeature::num_bins) << "FRAC_AT_D";

	double sum = 0.0;
	for (size_t i = 0; i < frac.size(); i++)
	{
		ASSERT_GE(frac[i], 0.0) << "FRAC_AT_D[" << i << "]";
		ASSERT_LE(frac[i], 1.0) << "FRAC_AT_D[" << i << "]";
		sum += frac[i];

		const double count = frac[i] * double(roidata.raw_pixels.size());
		ASSERT_NEAR(count, std::round(count), 1e-6) << "FRAC_AT_D[" << i << "] is not a whole count";
	}
	ASSERT_NEAR(sum, 1.0, 1e-9) << "FRAC_AT_D does not sum to one";
}

// MEAN_FRAC is the mean intensity of the pixels in a bin, so a non-empty bin cannot fall outside the
// ROI's own intensity range and an empty one is exactly zero.
void test_2d_radial_mean_frac_is_within_the_roi_intensity_range_invariant()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];

	ASSERT_EQ(mean.size(), (size_t)RadialDistributionFeature::num_bins) << "MEAN_FRAC";

	double lo = roidata.raw_pixels[0].inten, hi = roidata.raw_pixels[0].inten;
	for (const auto& p : roidata.raw_pixels)
	{
		lo = std::min(lo, (double)p.inten);
		hi = std::max(hi, (double)p.inten);
	}

	for (size_t i = 0; i < mean.size(); i++)
	{
		if (frac[i] == 0.0)
			ASSERT_EQ(mean[i], 0.0) << "MEAN_FRAC[" << i << "] is an empty bin (FRAC_AT_D is 0)";
		else
		{
			ASSERT_GE(mean[i], lo) << "MEAN_FRAC[" << i << "]";
			ASSERT_LE(mean[i], hi) << "MEAN_FRAC[" << i << "]";
		}
	}
}

// The two intensity features are tied together: summing each bin's mean intensity over the pixels
// that bin holds must return the ROI's total intensity, whatever the binning.
void test_2d_radial_bin_means_reconstruct_the_roi_intensity_invariant()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];

	double total = 0.0;
	for (const auto& p : roidata.raw_pixels)
		total += (double)p.inten;

	double reconstructed = 0.0;
	for (size_t i = 0; i < frac.size(); i++)
		reconstructed += std::round(frac[i] * double(roidata.raw_pixels.size())) * mean[i];

	const bool reconstructs = Nyxus::agrees_gt(reconstructed, total, 1e9);
	ASSERT_TRUE(reconstructs) << "FRAC_AT_D and MEAN_FRAC do not reconstruct the ROI intensity";
}

// RADIAL_CV is the coefficient of variation of num_bins non-negative wedge sums, computed with the
// population (biased) standard deviation. That is bounded above by sqrt(num_bins - 1), attained when
// a single wedge carries the whole ring; an empty ring is exactly zero.
void test_2d_radial_cv_is_within_its_bound_invariant()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& cv = roidata.fvals[(int)Nyxus::Feature2D::RADIAL_CV];

	ASSERT_EQ(cv.size(), (size_t)RadialDistributionFeature::num_bins) << "RADIAL_CV";
	const double bound = std::sqrt(double(RadialDistributionFeature::num_bins - 1));

	for (size_t i = 0; i < cv.size(); i++)
	{
		ASSERT_GE(cv[i], 0.0) << "RADIAL_CV[" << i << "]";
		ASSERT_LE(cv[i], bound * (1.0 + 1e-9)) << "RADIAL_CV[" << i << "]";
		if (frac[i] == 0.0)
			ASSERT_EQ(cv[i], 0.0) << "RADIAL_CV[" << i << "] is an empty bin (FRAC_AT_D is 0)";
	}
}
