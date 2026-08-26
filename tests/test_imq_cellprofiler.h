#pragma once

#include <gtest/gtest.h>

#include "test_imq_common.h"                  // fixture: calc_imq_feature, and FeatureIMQ via featureset.h
#include "test_ref_vals.h"                    // ref_vals_map, and <string> for the helper
#include "../src/nyx/features/saturation.h"   // SaturationFeature

// CellProfiler oracle for the image-quality saturation features MIN_SATURATION and MAX_SATURATION
// (SPEC 2 / 6.1: correctness claims live in oracle files, never in _regression files).
// tool=CellProfiler 4.2.8, cellprofiler.modules.MeasureImageQuality.calculate_saturation();
// env=nyxus_cellprofiler (conda); recipe=imq.saturation_observed_extremum;
// generator=tests/vetting/oracles/gen_imq_cellprofiler.py.
//
// Both tools use the same convention -- the fraction of pixels equal to the image's own observed
// extremum, not a fixed bit-depth threshold -- and both count the same 18 and 16 of the 96 pixels.
// MIN_SATURATION agrees bit for bit, because 18/96 = 0.1875 is exact in binary. MAX_SATURATION
// differs by one ulp, 2.8e-17 absolute (1.7e-16 relative): CellProfiler reports a PERCENTAGE, and
// 100.0*16/96 divided by 100.0 is one ulp above the 16/96 Nyxus computes. The pins carry
// CellProfiler's digits, so MAX_SATURATION's last digit is 9 rather than the 6 Nyxus produces. See
// tests/vetting/audit/imq_cellprofiler_vetting_report.md.
//
// Scope of the claim -- two cases where the implementations differ and which this fixture does not
// exercise:
//   * Constant ROI (min == max): CellProfiler counts minimal and maximal independently and reports
//     100% for both; get_percent_max_pixels() uses `else if`, so a pixel equal to both extrema is
//     counted only as maximal. Measured on a constant 4x4 ROI: MIN_SATURATION 0, MAX_SATURATION 1.
//   * Nyxus counts over the ROI's bounding-box image matrix, where in-box out-of-mask pixels are 0
//     and do take part in the extremum; CellProfiler restricts to the mask. They coincide here only
//     because im_quality_mask covers the whole 8x12 box.
//
// CellProfiler's FocusScore / LocalFocusScore are a *different* statistic from Nyxus' (normalized
// variance of the raw image vs variance of the Laplacian), so they are not vetted here -- see
// test_imq_opencv.h.
static const ref_vals_map<double> imq_cellprofiler_ref_vals {
	{"MIN_SATURATION", 0.1875},
	{"MAX_SATURATION", 0.16666666666666669}
};

// SPEC 7's exact tier verbatim: an absolute band, so ASSERT_NEAR rather than the relative agrees_gt
// the looser-tiered files use. The worst residual the band covers is the 2.8e-17 percent-conversion
// ulp on MAX_SATURATION; MIN_SATURATION's is exactly 0.
static const double imq_cellprofiler_abs_tolerance = 1e-9;

static void assert_imq_cellprofiler (Nyxus::FeatureIMQ feature, const std::string& feature_name)
{
	SCOPED_TRACE (std::string("CELLPROFILER__") + feature_name);

	// .at() on a const table: operator[] would default-insert a missing key and compare against
	// the zero it just created, so a missing pin would read as a golden of 0
	ASSERT_TRUE (imq_cellprofiler_ref_vals.count(feature_name) > 0) << feature_name;

	ASSERT_NEAR (calc_imq_feature<SaturationFeature>(feature),
		imq_cellprofiler_ref_vals.at(feature_name), imq_cellprofiler_abs_tolerance) << feature_name;
}

void test_imq_min_saturation_cellprofiler()
{
	assert_imq_cellprofiler (Nyxus::FeatureIMQ::MIN_SATURATION, "MIN_SATURATION");
}

void test_imq_max_saturation_cellprofiler()
{
	assert_imq_cellprofiler (Nyxus::FeatureIMQ::MAX_SATURATION, "MAX_SATURATION");
}
