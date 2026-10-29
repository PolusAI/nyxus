#pragma once

// The 2D first-order percentiles against their closed form, on ROIs small enough to work by hand. The
// rule is numpy.percentile's default ('linear'): the q-quantile of n sorted samples x_0..x_{n-1} sits
// at rank h = (n-1) q and is x_floor(h) + (h - floor(h)) (x_floor(h)+1 - x_floor(h)). PyRadiomics and
// MIRP take their percentiles with it, and it gives the IBSI phantom's reference values. The ROIs run
// through PixelIntensityFeatures, the calculator the production path runs, so each value is asserted
// as the feature reports it.
//
// What this discriminates: an estimator read off a binned histogram lands inside the right bin but
// not on the interpolated rank, and a rank taken as n q instead of (n-1) q moves every percentile
// between samples. Both miss these goldens by far more than the 1e-12 band. P01 and P10 share the
// lower rank 0 on five samples, so a single pass that resolves the ranks out of order is caught too.

#include "test_2d_firstorder_common.h"

// A one-row ROI carrying 'values' in that order; the order is not the sorted one, so nothing the
// test sees is already ranked.
static std::vector<std::vector<double>> firstorder_2d_analytic_values (const std::vector<unsigned int>& values)
{
	std::vector<NyxusPixel> pixels;
	for (size_t k = 0; k < values.size(); k++)
		pixels.push_back ({ k, 0, values[k] });

	Dataset ds;
	ds.dataset_props.push_back (SlideProps ("", ""));
	LR r (100);
	load_test_roi_data (r, pixels.data(), pixels.size());
	r.make_nonanisotropic_aabb();

	PixelIntensityFeatures f;
	EXPECT_NO_THROW (f.calculate (r, Fsettings(), ds));
	r.initialize_fvals();
	f.save_value (r.fvals);
	return r.fvals;
}

// Five distinct samples with one far from the rest, so the upper percentiles interpolate across a
// wide gap: {1, 2, 3, 4, 10}, h = 4q. The robust window [P10, P90] = [1.4, 7.6] holds 2, 3 and 4, whose
// mean is 3 and mean absolute deviation from it 2/3.
void test_2d_firstorder_percentiles_distinct_analytic()
{
	const auto v = firstorder_2d_analytic_values ({ 4, 10, 1, 3, 2 });
	EXPECT_NEAR (v[(int)Feature2D::P01][0], 1.04, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P10][0], 1.4, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P25][0], 2.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P75][0], 4.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P90][0], 7.6, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P99][0], 9.76, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::INTERQUARTILE_RANGE][0], 2.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::QCOD][0], 1. / 3., 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 2. / 3., 1e-12);
}

// Repeated values, which the frequencies hold as one entry each: {5, 5, 5, 9}, h = 3q. Every rank up
// to 2 lands on a 5, so P01 through P25 are 5; P75 sits a quarter of the way from the third 5 to the 9
// and P90 seven tenths of it. The robust window [5, 7.8] holds only the 5s, so their deviation is 0.
void test_2d_firstorder_percentiles_repeated_analytic()
{
	const auto v = firstorder_2d_analytic_values ({ 5, 9, 5, 5 });
	EXPECT_NEAR (v[(int)Feature2D::P01][0], 5.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P10][0], 5.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P25][0], 5.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P75][0], 6.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P90][0], 7.8, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::P99][0], 8.88, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::INTERQUARTILE_RANGE][0], 1.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::QCOD][0], 1. / 11., 1e-12);
	EXPECT_NEAR (v[(int)Feature2D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 0.0, 1e-12);
}
