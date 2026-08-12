#pragma once

// Morphology features whose registry rows read status=vetted, oracle=cellprofiler
// (oracle_coverage.csv, target_test=test_2d_morphology_cellprofiler.h): the 5 EDGE_* edge-intensity
// statistics and MASS_DISPLACEMENT. They were asserted inside two mixed-kind functions in
// test_2d_morphology_regression.h; SPEC 2 keeps one kind per file, so they are split out here.
//
// PROVENANCE RECORD MISSING (SPEC 6.4): the values come from the shared golden map in
// test_2d_morphology_common.h, which is still named for a snapshot and carries no CellProfiler version,
// config or generator. The registry is the authority on WHAT vets these features; the map name and
// the missing provenance record are tracked separately in not_covered.md section C.

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

// CellProfiler goldens for the six features whose registry rows read oracle=cellprofiler. Split out
// of morphology_2d_regression_ref_vals, which this file used to assert against through a shared
// helper -- a _cellprofiler function judging itself against a snapshot table.
//
// SPEC 6.4 provenance is still missing: no CellProfiler version, config or generator is recorded, so
// whether these numbers came from CellProfiler or from Nyxus cannot be told from the tree. The split
// preserves the registry's claim and makes the gap local instead of hiding it behind a shared table;
// closing it means a gen_morphology_cellprofiler.py run. Tracked in not_covered.md section C.
static ref_vals_map<double> morphology_2d_cellprofiler_ref_vals
{
	{"MASS_DISPLACEMENT", 0.634476074243407},
	{"EDGE_MEAN_INTENSITY", 41.8333333333333},
	{"EDGE_STDDEV_INTENSITY", 16.7691944455582},
	{"EDGE_MAX_INTENSITY", 68.0},
	{"EDGE_MIN_INTENSITY", 12.0},
	{"EDGE_INTEGRATED_INTENSITY", 753.0}
};

static void assert_morphology_feature_cellprofiler(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("CELLPROFILER__") + feature_name);
	ASSERT_TRUE(morphology_2d_cellprofiler_ref_vals.count(feature_name) > 0) << feature_name;
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_cellprofiler_ref_vals[feature_name], 1000.0));
}

void test_2d_morphology_edge_intensity_cellprofiler()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::MASS_DISPLACEMENT, "MASS_DISPLACEMENT");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_MEAN_INTENSITY, "EDGE_MEAN_INTENSITY");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_STDDEV_INTENSITY, "EDGE_STDDEV_INTENSITY");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_MAX_INTENSITY, "EDGE_MAX_INTENSITY");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_MIN_INTENSITY, "EDGE_MIN_INTENSITY");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, "EDGE_INTEGRATED_INTENSITY");
}
