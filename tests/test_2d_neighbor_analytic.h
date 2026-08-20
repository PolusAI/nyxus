#pragma once

#include <gtest/gtest.h>

#include "test_2d_neighbor_common.h"  // fixture only: calculate_neighbor_feature_values
#include "test_ref_vals.h"            // ref_vals_map_by_label, and <string> for the helper

// ANALYTIC oracle for the 2D neighbour second-distance and angle features (SPEC 6.4 provenance).
// tool=analytic (numpy); recipe=neighbor.scene2d_radius1, i.e. the neighborhood2d_scene_labels
// fixture from test_data.h at PIXELDISTANCE=1; generator=tests/vetting/oracles/gen_neighbor_analytic.py.
//
// Given the neighbour graph -- which CellProfiler vets independently, see
// test_2d_neighbor_cellprofiler.h -- these six features are deterministic closed forms of the ROI
// centroids, so an independent numpy recomputation of the documented formulas is the oracle (the
// same analytic-conformance basis as CIRCULARITY and the intensity-histogram percentiles).
//
// Agreement is 1.2e-16 worst case: 29 of the 30 values are bit-identical to the recomputation and
// ANG_BW_NEIGHBORS_STDDEV on label 1 differs by one ulp, so these assert at the SPEC 7 exact tier.
//
// Per label, not aggregated: the fixture is five ROIs and each carries its own value, so every one
// is asserted. Eighteen of the thirty goldens are structural zeros -- CLOSEST_NEIGHBOR2_* is 0 when
// fewer than two neighbours lie within the radius, and the sample standard deviation of a single
// angle is 0 -- and agrees_gt turns a zero golden into a demand for bit-exact equality, which is the
// strongest form this assertion takes.
//
// Why not CellProfiler for these: CP's AngleBetweenNeighbors is the angle SUBTENDED at an object by
// its two neighbours, not Nyxus' absolute atan2 direction angle; and CP's SecondClosestDistance is
// the second-closest of ANY object, whereas Nyxus reports the second-closest neighbour WITHIN the
// search radius. Different definitions, so CP cannot vet them.
static const ref_vals_map_by_label<double> neighbor_2d_analytic_ref_vals_by_label{
	{1, {
		{"CLOSEST_NEIGHBOR2_DIST", 2.5495097567963922},
		{"CLOSEST_NEIGHBOR1_ANG", 0.0},
		{"CLOSEST_NEIGHBOR2_ANG", 191.3099324740202},
		{"ANG_BW_NEIGHBORS_MEAN", 132.17251688149494},
		{"ANG_BW_NEIGHBORS_STDDEV", 115.23001801020591},
		{"ANG_BW_NEIGHBORS_MODE", 0.0},
	}},
	{2, {
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
		{"CLOSEST_NEIGHBOR1_ANG", 11.309932474020213},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 11.309932474020213},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 11.0},
	}},
	{3, {
		{"CLOSEST_NEIGHBOR2_DIST", 0.0},
		{"CLOSEST_NEIGHBOR1_ANG", 78.69006752597979},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 78.69006752597979},
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
		{"CLOSEST_NEIGHBOR1_ANG", 258.69006752597977},
		{"CLOSEST_NEIGHBOR2_ANG", 0.0},
		{"ANG_BW_NEIGHBORS_MEAN", 258.69006752597977},
		{"ANG_BW_NEIGHBORS_STDDEV", 0.0},
		{"ANG_BW_NEIGHBORS_MODE", 259.0},
	}}
};

// rel=1e-9: agrees_gt divides the golden by this, so a larger argument is a tighter band
static const double neighbor_2d_analytic_frac_tolerance = 1e9;

static void assert_neighbor2d_analytic(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("ANALYTIC__") + feature_name + "__L" + std::to_string(label));

	// .at() on a const table: operator[] would default-insert a missing key and compare against
	// the zero it just created, which agrees_gt reads as a demand for exact 0
	ASSERT_TRUE(neighbor_2d_analytic_ref_vals_by_label.count(label) > 0) << label;
	const auto& golden = neighbor_2d_analytic_ref_vals_by_label.at(label);
	ASSERT_TRUE(golden.count(feature_name) > 0) << feature_name;

	ASSERT_TRUE(agrees_gt(roiData.at(label).fvals[static_cast<int>(feature)][0],
		golden.at(feature_name), neighbor_2d_analytic_frac_tolerance)) << feature_name;
}

void test_2d_neighbor_second_distance_and_angles_analytic()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighbor_feature_values(roiData);

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
