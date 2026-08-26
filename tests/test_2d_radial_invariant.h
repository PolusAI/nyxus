#pragma once

// SPEC 2 invariants for the 2D radial intensity distribution: properties the three features must
// satisfy for any ROI under any radial-binning definition, so they hold independently of the pinned
// snapshots in test_2d_radial_regression.h and of the CellProfiler divergences recorded in
// tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md. They claim no oracle.
//
// Definition-independence is the entry test for this file: an assertion belongs here only if a
// correct re-definition of the feature would still satisfy it. The checks that hold only because of
// the conventions Nyxus currently uses - FRAC_AT_D counting pixels rather than intensity, MEAN_FRAC
// being a raw bin mean rather than a normalised fraction - are characterization and live in
// test_2d_radial_regression.h beside the pins they describe.

#include <gtest/gtest.h>

#include "../src/nyx/featureset.h"   // Feature2D
#include "test_2d_radial_common.h"   // build_radial_2d_roi, LR, RadialDistributionFeature
#include "test_main_nyxus.h"         // <cmath> for std::sqrt

// FRAC_AT_D partitions the ROI over the radial bins, so every entry is a fraction and the entries
// sum to one. That is true of a fraction of pixel count and equally of a fraction of intensity, so
// it survives a change of convention. The sum falls short of exactly 1 by the feature's own
// epsilon = 1e-9 guard in the denominator, hence the 1e-9 band rather than an exact comparison.
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
	}
	ASSERT_NEAR(sum, 1.0, 1e-9) << "FRAC_AT_D does not sum to one";
}

// A radial bin holding no pixels carries no intensity and no dispersion, whichever way the two
// intensity features are defined, so an empty bin is exactly zero in all three tables and the three
// tables agree on which bins are empty.
void test_2d_radial_empty_bins_are_zero_invariant()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	const auto& frac = roidata.fvals[(int)Nyxus::Feature2D::FRAC_AT_D];
	const auto& mean = roidata.fvals[(int)Nyxus::Feature2D::MEAN_FRAC];
	const auto& cv = roidata.fvals[(int)Nyxus::Feature2D::RADIAL_CV];

	ASSERT_EQ(mean.size(), frac.size()) << "MEAN_FRAC";
	ASSERT_EQ(cv.size(), frac.size()) << "RADIAL_CV";

	for (size_t i = 0; i < frac.size(); i++)
	{
		if (frac[i] != 0.0)
			continue;
		ASSERT_EQ(mean[i], 0.0) << "MEAN_FRAC[" << i << "] is an empty bin (FRAC_AT_D is 0)";
		ASSERT_EQ(cv[i], 0.0) << "RADIAL_CV[" << i << "] is an empty bin (FRAC_AT_D is 0)";
	}
}

// RADIAL_CV is a coefficient of variation of non-negative wedge quantities, computed with the
// population (biased) standard deviation. Over num_bins wedges that is bounded above by
// sqrt(num_bins - 1), attained when a single wedge carries the whole ring; restricting the average
// to the m non-empty wedges only lowers the bound to sqrt(m - 1), so the bound holds under either
// convention. A CV is never negative.
void test_2d_radial_cv_is_within_its_bound_invariant()
{
	LR roidata(101);
	build_radial_2d_roi(roidata);
	const auto& cv = roidata.fvals[(int)Nyxus::Feature2D::RADIAL_CV];

	ASSERT_EQ(cv.size(), (size_t)RadialDistributionFeature::num_bins) << "RADIAL_CV";
	const double bound = std::sqrt(double(RadialDistributionFeature::num_bins - 1));

	for (size_t i = 0; i < cv.size(); i++)
	{
		ASSERT_GE(cv[i], 0.0) << "RADIAL_CV[" << i << "]";
		ASSERT_LE(cv[i], bound * (1.0 + 1e-9)) << "RADIAL_CV[" << i << "]";
	}
}
