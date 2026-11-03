#pragma once

// The 3D first-order percentiles against their closed form, on the same hand-worked samples as
// test_2d_firstorder_analytic.h, laid out as a one-row volume. The rule is numpy.percentile's default
// ('linear'): the q-quantile of n sorted samples x_0..x_{n-1} sits at rank h = (n-1) q and is
// x_floor(h) + (h - floor(h)) (x_floor(h)+1 - x_floor(h)). The ROIs run through
// D3_VoxelIntensityFeatures, the calculator the production path runs.
//
// What this discriminates: an estimator read off a binned histogram lands inside the right bin but
// not on the interpolated rank, and a rank taken as n q instead of (n-1) q moves every percentile
// between samples. Both miss these goldens by far more than the 1e-12 band.
//
// The ROI is built by hand rather than loaded from a volume. The 3D grey-level pass reads only
// raw_pixels_3D, aux_min, aux_max, aux_area and slide_idx, and an LR's slide_idx of -1 names no
// scanned slide, so the values are reported as they are.

#include "../src/nyx/featureset.h"
#include "test_3d_firstorder_common.h"

static std::vector<std::vector<double>> firstorder_3d_analytic_values (const std::vector<PixIntens>& values)
{
	LR r (100);
	r.aux_min = values[0];
	r.aux_max = values[0];
	for (size_t k = 0; k < values.size(); k++)
	{
		r.raw_pixels_3D.push_back (Pixel3 ((int) k, 0, 0, values[k]));
		r.aux_min = std::min (r.aux_min, values[k]);
		r.aux_max = std::max (r.aux_max, values[k]);
	}
	r.aux_area = (decltype(r.aux_area)) r.raw_pixels_3D.size();

	Dataset ds;
	ds.dataset_props.push_back (SlideProps ("", ""));
	D3_VoxelIntensityFeatures f;
	EXPECT_NO_THROW (f.calculate (r, Fsettings(), ds));
	r.initialize_fvals();
	f.save_value (r.fvals);
	return r.fvals;
}

// {1, 2, 3, 4, 10}, h = 4q. The robust window [P10, P90] = [1.4, 7.6] holds 2, 3 and 4, whose mean is 3
// and mean absolute deviation from it 2/3.
void test_3d_firstorder_percentiles_distinct_analytic()
{
	const auto v = firstorder_3d_analytic_values ({ 4, 10, 1, 3, 2 });
	EXPECT_NEAR (v[(int)Feature3D::P01][0], 1.04, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P10][0], 1.4, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P25][0], 2.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P75][0], 4.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P90][0], 7.6, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P99][0], 9.76, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::INTERQUARTILE_RANGE][0], 2.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::QCOD][0], 1. / 3., 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 2. / 3., 1e-12);
}

// {5, 5, 5, 9}, h = 3q: P01 through P25 are 5, P75 sits a quarter of the way from the third 5 to the 9
// and P90 seven tenths of it. The robust window [5, 7.8] holds only the 5s.
void test_3d_firstorder_percentiles_repeated_analytic()
{
	const auto v = firstorder_3d_analytic_values ({ 5, 9, 5, 5 });
	EXPECT_NEAR (v[(int)Feature3D::P01][0], 5.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P10][0], 5.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P25][0], 5.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P75][0], 6.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P90][0], 7.8, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::P99][0], 8.88, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::INTERQUARTILE_RANGE][0], 1.0, 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::QCOD][0], 1. / 11., 1e-12);
	EXPECT_NEAR (v[(int)Feature3D::ROBUST_MEAN_ABSOLUTE_DEVIATION][0], 0.0, 1e-12);
}
