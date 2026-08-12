#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/basic_morphology.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/neighbors.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_2d_neighbor_common.h"   // fixture: make_neighbors2d_settings + calculate_neighbor_feature_values
#include "test_ref_vals.h"

static ref_vals_map_by_label<double> neighbor_2d_regression_ref_vals_by_label{
	{1, {
		{"NUM_NEIGHBORS", 4.0},
		{"PERCENT_TOUCHING", 100.0},   // exact_min_sqdist: all 8 contour pixels of the fully-enclosed 3x3 ROI touch a neighbor; was 87.5 (7/8) because the approximate min_sqdist overestimated one pixel's min distance and missed its true 8-adjacency
		{"CLOSEST_NEIGHBOR1_DIST", 2.5},
		{"CLOSEST_NEIGHBOR2_DIST", 2.54950975679639},
	}},
	{2, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 66.6666666666667},   // FIX #13 (neighbors.cpp): deduped adjacency touching pixels / contour length (was 33.33)
		{"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}},
	{3, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 66.6666666666667},
		{"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}},
	{4, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 50.0},
		{"CLOSEST_NEIGHBOR1_DIST", 2.5},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}},
	{5, {
		{"NUM_NEIGHBORS", 1.0},
		{"PERCENT_TOUCHING", 33.3333333333333},
		{"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639},
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
	}}
};

// NOTE: the closest-neighbor angles and the angle-between-neighbors stats are now
// vetted at tight tolerance by the ANALYTIC oracle in test_2d_neighbor_analytic.h
// (independent numpy recomputation of Nyxus' atan2 direction-angle formulas); the
// second-closest distance likewise. This regression file keeps only the fixture
// builder + the PERCENT_TOUCHING snapshot (the one feature with no promotable oracle).


static void assert_neighbor2d_feature(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name,
	double frac_tolerance = 1000.0)
{
	ASSERT_TRUE(neighbor_2d_regression_ref_vals_by_label.count(label) > 0);
	ASSERT_TRUE(neighbor_2d_regression_ref_vals_by_label[label].count(feature_name) > 0);
	ASSERT_TRUE(agrees_gt(roiData.at(label).fvals[static_cast<int>(feature)][0], neighbor_2d_regression_ref_vals_by_label[label][feature_name], frac_tolerance));
}

void test_2d_neighbor_counts_and_touching_regression()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighbor_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::NUM_NEIGHBORS, "NUM_NEIGHBORS");
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::PERCENT_TOUCHING, "PERCENT_TOUCHING");
	}
}

void test_2d_neighbor_closest_neighbors_regression()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighbor_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR1_DIST, "CLOSEST_NEIGHBOR1_DIST");
		assert_neighbor2d_feature(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR2_DIST, "CLOSEST_NEIGHBOR2_DIST");
	}
}

