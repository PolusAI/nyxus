#pragma once

// ANALYTIC oracle for the 2D neighbor second-distance + angle features.
// Given the neighbor graph (which CellProfiler vets, see test_neighbor_cellprofiler.h),
// these six features are deterministic closed forms of the ROI centroids, so an
// independent numpy recomputation of Nyxus' documented formulas IS the oracle
// (same analytic-conformance basis as CIRCULARITY / intensity_histogram). Goldens
// and the formula derivation are in tests/vetting/oracles/gen_neighbor_analytic.py,
// which validates them to double precision against these values; asserted at 1e-4.
//
// Why not CellProfiler for these: CP's AngleBetweenNeighbors is the angle SUBTENDED
// at an object by its two neighbors, not Nyxus' absolute atan2 direction angle; and
// CP's SecondClosestDistance is the 2nd-closest of ANY object, whereas Nyxus reports
// the 2nd-closest neighbor WITHIN the search radius (0 when <2 in-radius neighbors).

#include <gtest/gtest.h>

#include "test_neighbor_regression.h"  // shared fixture builder calculate_neighborhood2d_feature_values

static std::unordered_map<int, std::unordered_map<std::string, double>> neighbor2d_analytic_golden_by_label{
	{1, {
		{"CLOSEST_NEIGHBOR2_DIST", 2.54950975679639},
		{"CLOSEST_NEIGHBOR1_ANG", 0.0},
		{"CLOSEST_NEIGHBOR2_ANG", 191.30993247402},
		{"ANG_BW_NEIGHBORS_MEAN", 132.172516881495},
		{"ANG_BW_NEIGHBORS_STDDEV", 115.230018010206},
		{"ANG_BW_NEIGHBORS_MODE", 0.0},
	}},
	{2, {
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
		{"CLOSEST_NEIGHBOR1_ANG", 11.3099324740202},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 11.3099324740202},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 11.0},
	}},
	{3, {
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
		{"CLOSEST_NEIGHBOR1_ANG", 78.6900675259798},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 78.6900675259798},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 79.0},
	}},
	{4, {
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
		{"CLOSEST_NEIGHBOR1_ANG", 180.0},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 180.0},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 180.0},
	}},
	{5, {
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
		{"CLOSEST_NEIGHBOR1_ANG", 258.69006752598},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 258.69006752598},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 259.0},
	}}
};

static void assert_neighbor2d_analytic(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("ANALYTIC__") + feature_name + "__L" + std::to_string(label));
	ASSERT_TRUE(neighbor2d_analytic_golden_by_label[label].count(feature_name) > 0);
	ASSERT_NEAR(roiData.at(label).fvals[static_cast<int>(feature)][0],
		neighbor2d_analytic_golden_by_label[label][feature_name], 1e-4);
}

void test_neighborhood2d_analytic_second_distance_and_angles()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighborhood2d_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_neighbor2d_analytic(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR2_DIST, "CLOSEST_NEIGHBOR2_DIST");
		assert_neighbor2d_analytic(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR1_ANG, "CLOSEST_NEIGHBOR1_ANG");
		assert_neighbor2d_analytic(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR2_ANG, "CLOSEST_NEIGHBOR2_ANG");
		assert_neighbor2d_analytic(roiData, label, Nyxus::Feature2D::ANG_BW_NEIGHBORS_MEAN, "ANG_BW_NEIGHBORS_MEAN");
		assert_neighbor2d_analytic(roiData, label, Nyxus::Feature2D::ANG_BW_NEIGHBORS_STDDEV, "ANG_BW_NEIGHBORS_STDDEV");
		assert_neighbor2d_analytic(roiData, label, Nyxus::Feature2D::ANG_BW_NEIGHBORS_MODE, "ANG_BW_NEIGHBORS_MODE");
	}
}
