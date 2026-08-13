#pragma once

// Morphology features whose registry rows read status=vetted, oracle=cellprofiler
// (oracle_coverage.csv, target_test=test_2d_morphology_cellprofiler.h): four of the EDGE_*
// edge-intensity statistics and MASS_DISPLACEMENT. They were asserted inside two mixed-kind functions
// in test_2d_morphology_regression.h; SPEC 2 keeps one kind per file, so they are split out here.
//
// EDGE_STDDEV_INTENSITY is deliberately not among them -- see the note on the table below.

#include <string>
#include <vector>

#include "test_2d_morphology_common.h"

#include "test_ref_vals.h"

// SPEC 6.4 provenance
//   tool       CellProfiler 4.2.8 (module package) / cellprofiler-core 4.2.8.1, centrosome 1.2.3
//   module     MeasureObjectIntensity, one image and one object set, all settings at defaults
//   fixture    shape2d_morphology_intensity / shape2d_morphology_mask (test_data.h), background-padded
//   generator  tests/vetting/oracles/gen_morphology_cellprofiler.py (offline; CP is never a CI dep)
//   mapping    EDGE_* <- Intensity_{Mean,Max,Min,IntegratedIntensity}Edge_<image>
//              MASS_DISPLACEMENT <- Intensity_MassDisplacement_<image>
//
// The two tools select the same edge pixels, which is what makes this an exact comparison rather
// than an approximate one: CellProfiler's edge is skimage.segmentation.find_boundaries(mode="inner")
// -- an object pixel is an edge pixel unless all four of its N/S/E/W neighbours share its label --
// and on this fixture that is 18 of the 26 ROI pixels, summing to 753 against the ROI's 1048.
//
// Measured agreement is 5.2e-8 relative at worst, and the residual is not a disagreement: CellProfiler
// stores the image as float32, so a raw value round-trips as raw/255 -> float32 -> *255. The band
// below (1e-6 relative) sits above that and far under anything a real divergence would produce.
static const ref_vals_map<double> morphology_2d_cellprofiler_ref_vals
{
	{"MASS_DISPLACEMENT", 0.634476074243407},
	{"EDGE_MEAN_INTENSITY", 41.8333333333333},
	{"EDGE_MAX_INTENSITY", 68.0},
	{"EDGE_MIN_INTENSITY", 12.0},
	{"EDGE_INTEGRATED_INTENSITY", 753.0}
};

// EDGE_STDDEV_INTENSITY is absent by result, not by omission. Over the identical 18 edge pixels the
// two tools use different estimators: Nyxus divides by n-1 (Moments4::std() returns sqrt(M2/(n-1)),
// a helper shared across features, so this is a house convention rather than a local slip) and
// CellProfiler divides by n. The values differ by exactly sqrt(n/(n-1)) = 1.0289915 at n=18, which
// no tolerance should absorb, so the feature keeps its snapshot in
// test_2d_morphology_regression.h and its registry row reads regression.

static void assert_morphology_feature_cellprofiler(
	const std::vector<std::vector<double>>& fvals,
	Nyxus::Feature2D feature,
	const std::string& feature_name)
{
	SCOPED_TRACE(std::string("CELLPROFILER__") + feature_name);
	ASSERT_TRUE(morphology_2d_cellprofiler_ref_vals.count(feature_name) > 0) << feature_name;
	// 1e6 -> a 1e-6 relative band, set from the measured 5.2e-8 float32 residual rather than left
	// at the 0.1% the shared snapshot helper used; a band that loose would pass the sqrt(n/(n-1))
	// estimator gap that disqualified EDGE_STDDEV_INTENSITY.
	ASSERT_TRUE(agrees_gt(fvals[static_cast<int>(feature)][0], morphology_2d_cellprofiler_ref_vals.at(feature_name), 1e6));
}

void test_2d_morphology_edge_intensity_cellprofiler()
{
	std::vector<std::vector<double>> fvals;
	calculate_shape2d_feature_values(fvals);

	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::MASS_DISPLACEMENT, "MASS_DISPLACEMENT");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_MEAN_INTENSITY, "EDGE_MEAN_INTENSITY");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_MAX_INTENSITY, "EDGE_MAX_INTENSITY");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_MIN_INTENSITY, "EDGE_MIN_INTENSITY");
	assert_morphology_feature_cellprofiler(fvals, Nyxus::Feature2D::EDGE_INTEGRATED_INTENSITY, "EDGE_INTEGRATED_INTENSITY");
}
