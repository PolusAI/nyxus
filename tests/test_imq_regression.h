#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"
#include "test_feature_calculation_common.h"

#include "../src/nyx/features/power_spectrum.h"
#include "../src/nyx/features/sharpness.h"

/* GLCM dissimilarity and correlation for image quality are handled in GLCM tests */

/* Snapshot drift guards only -- this file claims no correctness (SPEC 2).
   FOCUS_SCORE / LOCAL_FOCUS_SCORE are vetted in test_imq_opencv.h,
   MIN_SATURATION / MAX_SATURATION in test_imq_cellprofiler.h. */

void test_imq_power_spectrum_slope_regression() {

    PowerSpectrumFeature f;
    double truth_value = 0.0;

    assert_feature(f, Nyxus::FeatureIMQ::POWER_SPECTRUM_SLOPE, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};


void test_imq_sharpness_regression() {

    SharpnessFeature f;
    double truth_value = 2.19047;

    assert_feature(f, Nyxus::FeatureIMQ::SHARPNESS, 1, im_quality_intensity, im_quality_mask, sizeof(im_quality_mask) / sizeof(NyxusPixel), truth_value);
};
