#pragma once

#include <gtest/gtest.h>

#include "test_2d_neighbor_common.h"  // fixture only: calculate_neighbor_feature_values
#include "test_main_nyxus.h"          // agrees_gt
#include "test_ref_vals.h"            // ref_vals_map_by_label, and <string> for the helper

// Pinned Nyxus output for PERCENT_TOUCHING on the neighborhood2d_scene_labels fixture at
// PIXELDISTANCE=1 -- the same recipe the oracle files use. SPEC 2 regression tier: a drift guard,
// no oracle claim.
//
// PERCENT_TOUCHING is the one feature in this family with no promotable oracle. Nyxus counts contour
// pixels 8-adjacent to a neighbour over contour length; CellProfiler counts object outline pixels
// overlapping a disk(distance+0.5)-dilated neighbour over perimeter, and no CP distance method
// reproduces Nyxus -- the two disagree on 3 of the 5 ROIs. That is a definition difference, not a
// defect, so the values are pinned rather than compared. The bounds the feature must obey regardless
// are asserted in test_2d_neighbor_invariant.h.
//
// The other eight features in the family are vetted at the exact tier against CellProfiler
// (test_2d_neighbor_cellprofiler.h) and the analytic recomputation (test_2d_neighbor_analytic.h) on
// this same fixture, so they are deliberately not duplicated here: a second assertion of the same
// value at a looser band adds no coverage.
static const ref_vals_map_by_label<double> neighbor_2d_regression_ref_vals_by_label{
	{1, {{"PERCENT_TOUCHING", 100.0}}},
	{2, {{"PERCENT_TOUCHING", 66.666666666666671}}},
	{3, {{"PERCENT_TOUCHING", 66.666666666666671}}},
	{4, {{"PERCENT_TOUCHING", 50.0}}},
	{5, {{"PERCENT_TOUCHING", 33.333333333333336}}}
};

static void assert_neighbor2d_regression(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("REGRESSION__") + feature_name + "__L" + std::to_string(label));

	// .at() on a const table: operator[] would default-insert a missing key and compare against
	// the zero it just created, which agrees_gt reads as a demand for exact 0
	ASSERT_TRUE(neighbor_2d_regression_ref_vals_by_label.count(label) > 0) << label;
	const auto& golden = neighbor_2d_regression_ref_vals_by_label.at(label);
	ASSERT_TRUE(golden.count(feature_name) > 0) << feature_name;

	// agrees_gt's default frac_tolerance is rel=1e-3, the drift band these guards want: the pins are
	// recorded at full precision and nothing but a real change should move them
	ASSERT_TRUE(agrees_gt(roiData.at(label).fvals[static_cast<int>(feature)][0],
		golden.at(feature_name), 1000.0)) << feature_name;
}

void test_2d_neighbor_percent_touching_regression()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighbor_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
		assert_neighbor2d_regression(roiData, label, Nyxus::Feature2D::PERCENT_TOUCHING, "PERCENT_TOUCHING");
}
