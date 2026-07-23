#pragma once

// CellProfiler oracle for the 2D neighbor-graph features NUM_NEIGHBORS and
// CLOSEST_NEIGHBOR1_DIST. The real cellprofiler.modules.MeasureObjectNeighbors
// module (Adjacent method) was run on this fixture; CP reproduces Nyxus exactly for
// both features. Goldens + the offline CP run are in
// tests/vetting/oracles/gen_neighbor_cellprofiler.py; asserted at 1e-4.
//
// PERCENT_TOUCHING is NOT vetted here: CP and Nyxus use different definitions
// (Nyxus = contour pixels 8-adjacent to a neighbor / contour length; CP = object
// outline pixels overlapping a disk(distance+0.5)-dilated neighbor / perimeter), and
// no CP distance method reproduces Nyxus (diverges on 3/5 ROIs). It stays a
// regression snapshot in test_neighbor_regression.h.

#include <gtest/gtest.h>

#include "test_neighbor_regression.h"  // shared fixture builder calculate_neighborhood2d_feature_values

static std::unordered_map<int, std::unordered_map<std::string, double>> neighbor2d_cellprofiler_golden_by_label{
	{1, {{"NUM_NEIGHBORS", 4.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.5}}},
	{2, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639}}},
	{3, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639}}},
	{4, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.5}}},
	{5, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.54950975679639}}}
};

static void assert_neighbor2d_cellprofiler(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("CELLPROFILER__") + feature_name + "__L" + std::to_string(label));
	ASSERT_TRUE(neighbor2d_cellprofiler_golden_by_label[label].count(feature_name) > 0);
	ASSERT_NEAR(roiData.at(label).fvals[static_cast<int>(feature)][0],
		neighbor2d_cellprofiler_golden_by_label[label][feature_name], 1e-4);
}

void test_neighborhood2d_cellprofiler_counts_and_first_distance()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighborhood2d_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_neighbor2d_cellprofiler(roiData, label, Nyxus::Feature2D::NUM_NEIGHBORS, "NUM_NEIGHBORS");
		assert_neighbor2d_cellprofiler(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR1_DIST, "CLOSEST_NEIGHBOR1_DIST");
	}
}
