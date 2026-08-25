#pragma once

#include <gtest/gtest.h>

#include "test_2d_neighbor_common.h"  // fixture only: calculate_neighbor_feature_values
#include "test_ref_vals.h"            // ref_vals_map_by_label, and <string> for the helper

// CellProfiler oracle for the 2D neighbour-graph features NUM_NEIGHBORS and CLOSEST_NEIGHBOR1_DIST
// (SPEC 6.4 provenance).
// tool=CellProfiler 4.2.8, cellprofiler.modules.MeasureObjectNeighbors, Adjacent method;
// env=nyxus_cellprofiler (conda); recipe=neighbor.scene2d_radius1, i.e. the
// neighborhood2d_scene_labels fixture from test_data.h at PIXELDISTANCE=1;
// generator=tests/vetting/oracles/gen_neighbor_cellprofiler.py.
//
// CellProfiler reproduces Nyxus BIT-IDENTICALLY on both features across all five ROIs -- residual
// exactly 0, not merely small -- so these assert at the SPEC 7 exact tier, which is an ABSOLUTE
// 1e-9 band. Per label, not aggregated: each ROI carries its own value and every one is asserted.
//
// PERCENT_TOUCHING is NOT vetted here: CP and Nyxus use different definitions (Nyxus = contour
// pixels 8-adjacent to a neighbour / contour length; CP = object outline pixels overlapping a
// disk(distance+0.5)-dilated neighbour / perimeter), and no CP distance method reproduces Nyxus --
// it diverges on 3 of the 5 ROIs. It is drift-pinned in test_2d_neighbor_regression.h and its
// required bounds are asserted in test_2d_neighbor_invariant.h. Same for CLOSEST_NEIGHBOR2_DIST:
// CP measures the second-closest of ANY object, Nyxus the second-closest neighbour within the
// search radius, so CP's column is a different quantity. See
// tests/vetting/audit/neighbor_2d_cellprofiler_vetting_report.md.
static const ref_vals_map_by_label<double> neighbor_2d_cellprofiler_ref_vals_by_label{
	{1, {{"NUM_NEIGHBORS", 4.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.5}}},
	{2, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.5495097567963922}}},
	{3, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.5495097567963922}}},
	{4, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.5}}},
	{5, {{"NUM_NEIGHBORS", 1.0}, {"CLOSEST_NEIGHBOR1_DIST", 2.5495097567963922}}}
};

// SPEC 7's exact tier verbatim: an absolute band, so ASSERT_NEAR rather than the relative agrees_gt
// the looser-tiered files use. The measured residual over the ten comparisons below is exactly 0 --
// CellProfiler returns the same double Nyxus does -- so the band is what SPEC 7 sets rather than
// what the measurement needs. A relative 1e-9 would also have been the looser of the two here, at
// 2.5e-9 on CLOSEST_NEIGHBOR1_DIST.
static const double neighbor_2d_cellprofiler_abs_tolerance = 1e-9;

static void assert_neighbor2d_cellprofiler(
	const std::unordered_map<int, LR>& roiData,
	int label,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("CELLPROFILER__") + feature_name + "__L" + std::to_string(label));

	// .at() on a const table: operator[] would default-insert a missing key and compare against
	// the zero it just created, so a missing pin would read as a golden of 0
	ASSERT_TRUE(neighbor_2d_cellprofiler_ref_vals_by_label.count(label) > 0) << label;
	const auto& golden = neighbor_2d_cellprofiler_ref_vals_by_label.at(label);
	ASSERT_TRUE(golden.count(feature_name) > 0) << feature_name;

	ASSERT_NEAR(roiData.at(label).fvals[static_cast<int>(feature)][0],
		golden.at(feature_name), neighbor_2d_cellprofiler_abs_tolerance) << feature_name;
}

void test_2d_neighbor_counts_and_first_distance_cellprofiler()
{
	std::unordered_map<int, LR> roiData;
	calculate_neighbor_feature_values(roiData);

	for (int label : {1, 2, 3, 4, 5})
	{
		assert_neighbor2d_cellprofiler(roiData, label, Nyxus::Feature2D::NUM_NEIGHBORS, "NUM_NEIGHBORS");
		assert_neighbor2d_cellprofiler(roiData, label, Nyxus::Feature2D::CLOSEST_NEIGHBOR1_DIST, "CLOSEST_NEIGHBOR1_DIST");
	}
}
