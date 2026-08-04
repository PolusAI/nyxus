#pragma once

// CellProfiler oracle for the image-quality saturation features MIN_SATURATION and
// MAX_SATURATION (SPEC 2 / 6.1: correctness claims live in oracle files, never in
// _regression files).
//
// The real cellprofiler.modules.MeasureImageQuality module was run on this fixture;
// its saturation metrics reproduce Nyxus exactly. Both use the same convention -- the
// fraction of pixels equal to the image's own observed extremum, not a fixed bit-depth
// threshold. Goldens + the offline CellProfiler run are in
// tests/vetting/oracles/gen_imq_cellprofiler.py.
//
// Scope of the claim -- two cases where the implementations differ and which this
// fixture does not exercise:
//   * Constant ROI (min == max): CellProfiler counts minimal and maximal independently
//     and reports 100% for both; get_percent_max_pixels() uses `else if`, so
//     MIN_SATURATION comes out 0.
//   * Nyxus counts over the ROI's bounding-box image matrix, where in-box out-of-mask
//     pixels are 0 and do take part in the extremum; CellProfiler restricts to the mask.
//     They coincide here only because im_quality_mask covers the whole 8x12 box.
//
// CellProfiler's FocusScore / LocalFocusScore are a *different* statistic from Nyxus'
// (normalized variance of the raw image vs variance of the Laplacian), so they are not
// vetted here -- see test_imq_opencv.h.

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_feature_calculation.h"

#include "../src/nyx/features/saturation.h"

void test_min_saturation_cellprofiler() {

    SaturationFeature f;
    // CellProfiler 4.2.8 MeasureImageQuality: Image_ImageQuality_PercentMinimal / 100
    // = 18/96. gen_imq_cellprofiler.py; tolerance rel=1e-3 (agreement is exact).
    double truth_value = 0.1875;

    test_feature(f, Nyxus::FeatureIMQ::MIN_SATURATION, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};

void test_max_saturation_cellprofiler() {

    SaturationFeature f;
    // CellProfiler 4.2.8 MeasureImageQuality: Image_ImageQuality_PercentMaximal / 100
    // = 16/96. gen_imq_cellprofiler.py; tolerance rel=1e-3 (agreement is exact).
    double truth_value = 0.16666666666666666;

    test_feature(f, Nyxus::FeatureIMQ::MAX_SATURATION, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};
