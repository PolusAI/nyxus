#pragma once

#include <gtest/gtest.h>

#include "test_imq_common.h"                     // fixture: calc_imq_feature, and FeatureIMQ via featureset.h
#include "test_ref_vals.h"                       // ref_vals_map, and <string> for the helper
#include "../src/nyx/features/power_spectrum.h"  // PowerSpectrumFeature
#include "../src/nyx/features/sharpness.h"       // SharpnessFeature

// Snapshot drift guards only -- this file claims no correctness (SPEC 2). FOCUS_SCORE /
// LOCAL_FOCUS_SCORE are vetted in test_imq_opencv.h, MIN_SATURATION / MAX_SATURATION in
// test_imq_cellprofiler.h. Recipe imq.regression_quality_roi; GLCM dissimilarity and correlation,
// which the image-quality documentation also lists, are the GLCM family's and are asserted there.
//
// Both pins are Nyxus' own output at full %.17g precision, and neither feature has an oracle:
//
//   POWER_SPECTRUM_SLOPE is 0 because rps() returns early unless floor(min(h,w)/8) >= 3, i.e.
//   unless the ROI is at least 24 px on its short side; this one is 8 wide. The pin therefore
//   covers the guard and nothing downstream of it, which is why its band is exact rather than
//   float-sized: 0 is a literal the early-return path returns, not a computed quantity.
//
//   SHARPNESS is a DOM (Kumar et al. 2012) port. The published reference implementation does not
//   reproduce it -- 2.1904708385718963 against 0.54592951157710823 on this fixture -- for six
//   structural reasons, none of them numerical; they are enumerated in
//   tests/vetting/audit/imq_pydom_sharpness_vetting_report.md. The pin detects change; it endorses
//   nothing.
static const ref_vals_map<double> imq_regression_ref_vals {
	{"POWER_SPECTRUM_SLOPE", 0.0},
	{"SHARPNESS", 2.1904708385718963}
};

// Absolute bands, per pin. A snapshot compares the program against itself, so movement is the only
// thing it can catch and the band should be no wider than the reproducibility it needs.
// SHARPNESS' is rel=1e-9 at its own magnitude -- 2.1904708385718963e-9 -- rounded UP to two
// significant figures, so the number stays checkable rather than being a band nobody can derive.
// POWER_SPECTRUM_SLOPE's is 0 because the value is a returned literal, so any other value is a
// change of behaviour rather than a float wobble.
static const double imq_regression_power_spectrum_slope_abs_tolerance = 0.0;
static const double imq_regression_sharpness_abs_tolerance = 2.2e-9;

template <class F>
static void assert_imq_regression (Nyxus::FeatureIMQ feature, const std::string& feature_name,
	double abs_tolerance)
{
	SCOPED_TRACE (std::string("REGRESSION__") + feature_name);

	// .at() on a const table: operator[] would default-insert a missing key and compare against
	// the zero it just created, so a missing pin would read as a golden of 0
	ASSERT_TRUE (imq_regression_ref_vals.count(feature_name) > 0) << feature_name;

	ASSERT_NEAR (calc_imq_feature<F>(feature), imq_regression_ref_vals.at(feature_name),
		abs_tolerance) << feature_name;
}

void test_imq_power_spectrum_slope_regression()
{
	assert_imq_regression<PowerSpectrumFeature> (Nyxus::FeatureIMQ::POWER_SPECTRUM_SLOPE,
		"POWER_SPECTRUM_SLOPE", imq_regression_power_spectrum_slope_abs_tolerance);
}

void test_imq_sharpness_regression()
{
	assert_imq_regression<SharpnessFeature> (Nyxus::FeatureIMQ::SHARPNESS, "SHARPNESS",
		imq_regression_sharpness_abs_tolerance);
}
