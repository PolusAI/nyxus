#pragma once

#include <gtest/gtest.h>
#include <vector>                                // the probe ROIs below are built, not literals

#include "test_imq_common.h"                     // fixture: calc_imq_feature, and FeatureIMQ via featureset.h
#include "test_ref_vals.h"                       // ref_vals_map, and <string> for the helper
#include "../src/nyx/features/power_spectrum.h"  // PowerSpectrumFeature
#include "../src/nyx/features/saturation.h"      // SaturationFeature
#include "../src/nyx/features/sharpness.h"       // SharpnessFeature

// Snapshot drift guards only -- this file claims no correctness (SPEC 2). FOCUS_SCORE /
// LOCAL_FOCUS_SCORE are vetted in test_imq_opencv.h, MIN_SATURATION / MAX_SATURATION in
// test_imq_cellprofiler.h. Recipe imq.regression_quality_roi; GLCM dissimilarity and correlation,
// which the image-quality documentation also lists, are the GLCM family's and are asserted there.
//
// Every pin here is Nyxus' own output at full %.17g precision, and no cell in this file has an
// oracle. The two on the 8x12 fixture:
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
//
// The four SATURATION_* pins and POWER_SPECTRUM_SLOPE_LARGE_ROI cover the matrix cells that sit
// outside the two oracle recipes: a constant ROI, a mask narrower than the bounding box, and an ROI
// wide enough for the power-spectrum guard to let the algorithm run. Each is a real config Nyxus
// reaches that no external tool reproduces, so SPEC 5.1 makes them VALID-BUT-PRODUCTION-ONLY and
// they are snapshotted here rather than described in the matrix and left unpinned.
static const ref_vals_map<double> imq_regression_ref_vals {
	{"POWER_SPECTRUM_SLOPE", 0.0},
	{"SHARPNESS", 2.1904708385718963},
	{"MIN_SATURATION_CONSTANT_ROI", 0.0},
	{"MAX_SATURATION_CONSTANT_ROI", 1.0},
	{"MIN_SATURATION_NARROW_MASK", 0.6875},
	{"MAX_SATURATION_NARROW_MASK", 0.0625},
	{"POWER_SPECTRUM_SLOPE_LARGE_ROI", 1.7837481542489078}
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

// ---------------------------------------------------------------------------------------------
// The saturation cells outside the CellProfiler recipe.
//
// tests/vetting/matrix/imq.md classifies both as VALID-BUT-PRODUCTION-ONLY (SPEC 5.1): each is a
// real config Nyxus reaches, and on each one CellProfiler computes something else, so neither can
// carry an oracle claim and both need a drift guard here. The divergences themselves are described
// in test_imq_cellprofiler.h and the matrix; these assertions only hold the current behaviour still
// so a change to it is visible.
//
// The pins are exact fractions of pixel counts, so the band is 0: every value below is k/16 with a
// power-of-two denominator and is therefore representable, and anything else is a behaviour change
// rather than a float wobble.
static const double imq_regression_saturation_abs_tolerance = 0.0;

// A constant ROI, min == max. get_percent_max_pixels() chains `else if`, so a pixel equal to both
// extrema is counted only as maximal: MIN_SATURATION 0 and MAX_SATURATION 1 where CellProfiler
// reports 100% for both.
static std::vector<NyxusPixel> imq_constant_roi_intensity()
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < 4; y++)
		for (size_t x = 0; x < 4; x++)
			px.push_back (NyxusPixel {x, y, 7u});
	return px;
}

// A mask narrower than the bounding box: 5 of the 16 pixels in the 4 x 4 AABB are in the mask, and
// the other 11 sit inside the box unmasked. Nyxus builds its image matrix over the whole box and
// leaves those 11 at 0, so they take part in the extremum and MIN_SATURATION counts them;
// CellProfiler restricts to the mask and never sees them.
static std::vector<NyxusPixel> imq_narrow_mask_intensity()
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < 4; y++)
		for (size_t x = 0; x < 4; x++)
			px.push_back (NyxusPixel {x, y, (unsigned int)(10 + x + 4 * y)});   // all distinct, none 0
	return px;
}

static std::vector<NyxusPixel> imq_narrow_mask_mask()
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < 4; y++)
		for (size_t x = 0; x < 4; x++)
		{
			// the 2 x 2 top-left block plus the far corner: keeps the AABB the full 4 x 4 while
			// leaving 11 in-box pixels out of the mask
			bool in = (x < 2 && y < 2) || (x == 3 && y == 3);
			px.push_back (NyxusPixel {x, y, in ? 1u : 0u});
		}
	return px;
}

static std::vector<NyxusPixel> imq_all_ones_mask (size_t w, size_t h)
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < h; y++)
		for (size_t x = 0; x < w; x++)
			px.push_back (NyxusPixel {x, y, 1u});
	return px;
}

// One assertion per matrix cell. The pin key is the feature plus the cell, not the bare feature
// name: MIN_SATURATION is pinned three times across this family and the cellprofiler file, and a
// key that named only the feature would let two cells share one golden.
//
// The feature enum and the pin key are both on the CALL line for the same reason the rest of the
// family puts them there - audit/scan_imq_coverage.py attributes coverage from the assertion line,
// so a feature named only inside a wrapped expression is a test the coverage artifact cannot see.
template <class F>
static void assert_imq_regression_on (Nyxus::FeatureIMQ feature, const std::string& pin_name,
	const std::vector<NyxusPixel>& intensity, const std::vector<NyxusPixel>& mask,
	double abs_tolerance)
{
	SCOPED_TRACE (std::string("REGRESSION__") + pin_name);

	// .at() on a const table: operator[] would default-insert a missing key and compare against
	// the zero it just created, so a missing pin would read as a golden of 0
	ASSERT_TRUE (imq_regression_ref_vals.count(pin_name) > 0) << pin_name;

	ASSERT_NEAR (calc_imq_feature_on<F> (feature, intensity.data(), mask.data(), mask.size()),
		imq_regression_ref_vals.at(pin_name), abs_tolerance) << pin_name;
}

void test_imq_min_saturation_constant_roi_regression()
{
	assert_imq_regression_on<SaturationFeature> (Nyxus::FeatureIMQ::MIN_SATURATION,
		"MIN_SATURATION_CONSTANT_ROI", imq_constant_roi_intensity(), imq_all_ones_mask (4, 4),
		imq_regression_saturation_abs_tolerance);
}

void test_imq_max_saturation_constant_roi_regression()
{
	assert_imq_regression_on<SaturationFeature> (Nyxus::FeatureIMQ::MAX_SATURATION,
		"MAX_SATURATION_CONSTANT_ROI", imq_constant_roi_intensity(), imq_all_ones_mask (4, 4),
		imq_regression_saturation_abs_tolerance);
}

void test_imq_min_saturation_narrow_mask_regression()
{
	assert_imq_regression_on<SaturationFeature> (Nyxus::FeatureIMQ::MIN_SATURATION,
		"MIN_SATURATION_NARROW_MASK", imq_narrow_mask_intensity(), imq_narrow_mask_mask(),
		imq_regression_saturation_abs_tolerance);
}

void test_imq_max_saturation_narrow_mask_regression()
{
	assert_imq_regression_on<SaturationFeature> (Nyxus::FeatureIMQ::MAX_SATURATION,
		"MAX_SATURATION_NARROW_MASK", imq_narrow_mask_intensity(), imq_narrow_mask_mask(),
		imq_regression_saturation_abs_tolerance);
}

// ---------------------------------------------------------------------------------------------
// POWER_SPECTRUM_SLOPE past the guard.
//
// rps() returns {0.} unless floor(min(h,w)/8) >= 3, so the im_quality fixture's 8 px width pins the
// early return and nothing downstream of it. This 24 x 24 ROI clears the guard and is the only
// reachable cell in which the algorithm itself runs - VALID-BUT-PRODUCTION-ONLY in
// tests/vetting/matrix/imq.md, which is why it is snapshotted here rather than left unpinned.
//
// It endorses nothing. The matrix records what the algorithm does in this cell: the radius axis is
// floor(sqrt(fft coefficient)) + 1 rather than the frequency radius sqrt(kx^2 + ky^2) a log-log
// power-spectrum fit is defined over, and the loop that reads raw_radii[i] is bounded by the padded
// FFT size instead of by raw_radii. This pin exists so a fix to either shows up as a moved golden
// instead of passing unnoticed.
//
// The pattern is a deterministic modular ramp: integer-valued, no RNG, and textured enough that the
// least-squares fit gets more than one surviving point (a smooth ramp leaves too few and the
// function falls through to the same 0 the guard returns, which would make this pin unable to tell
// the two paths apart).
static std::vector<NyxusPixel> imq_large_roi_intensity()
{
	std::vector<NyxusPixel> px;
	for (size_t y = 0; y < 24; y++)
		for (size_t x = 0; x < 24; x++)
			px.push_back (NyxusPixel {x, y, (unsigned int)(1 + (x * 7 + y * 13) % 64)});
	return px;
}

// Measured on this fixture (temporary instrumentation in power_spectrum_slope(), not committed):
// magnitude.size() = 1024 (the 32 x 32 power-of-2 padded FFT) against raw_radii.size() = 24, the
// largest index the loop reached was 3, and 3 points survived to the fit. So the unbounded
// raw_radii[i] read stays in range HERE - the pin is a defined value, not a snapshot of a read past
// the end - and the bound is still missing for inputs that push the index past 23.
//
// A computed slope, not a returned literal, so the band is float-sized rather than 0: rel=1e-9 at
// the pinned magnitude, rounded up to two significant figures.
static const double imq_regression_power_spectrum_slope_large_roi_abs_tolerance = 1.8e-9;

void test_imq_power_spectrum_slope_large_roi_regression()
{
	assert_imq_regression_on<PowerSpectrumFeature> (Nyxus::FeatureIMQ::POWER_SPECTRUM_SLOPE,
		"POWER_SPECTRUM_SLOPE_LARGE_ROI", imq_large_roi_intensity(), imq_all_ones_mask (24, 24),
		imq_regression_power_spectrum_slope_large_roi_abs_tolerance);
}
