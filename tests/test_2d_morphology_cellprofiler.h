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

static void assert_morphology_feature_cellprofiler(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("CELLPROFILER__") + feature_name);
	assert_unvetted_no_direct_oracle_shape2d_feature(fvals, feature, feature_name);
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
