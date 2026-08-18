#pragma once

// Shared fixture for the 2D NGLDM tests: the settings recipe and the IBSI-phantom scaffolding that
// turns a feature id into one averaged value ready to compare against a golden.
//
// Fixtures only, no reference data (SPEC 6.3.1). This header exists because the mirp and regression
// files reached the assertion scaffolding by including test_2d_ngldm_ibsi.h -- one oracle file
// including another to borrow a helper, which put the IBSI consensus table in scope of every
// assertion that did so, and made the include graph say "mirp depends on IBSI" when only the
// fixture is shared.

// No <gtest/gtest.h> guard needed here: the assertion helper below uses gtest macros, so this
// header includes it like the test files that read it.
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/nyx/environment.h"
#include "../src/nyx/features/ngldm.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// The settings every 2D NGLDM test runs on. Only the IBSI flag varies: the matrix tests exercise
// both modes, while the feature assertions are IBSI-mode only, which is what makes the published
// consensus values and mirp's ngldm_distance=1 / difference_level=0 run comparable.
static Fsettings make_ngldm2d_settings(bool ibsi_mode)
{
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 128;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = ibsi_mode;
    return s;
}

// Featurises the four IBSI digital-phantom slices one at a time and compares the average against
// the caller's table -- the 2D-averaged aggregation the IBSI NGLDM values and the mirp run both
// use. The table, the tolerance and the trace prefix come from the caller, so each file asserts
// against its own goldens at its own tier and this header claims nothing about any of them.
void assert_ngldm_feature_against_golden_values(
    const Feature2D& feature_,
    const std::string& feature_name,
    const std::unordered_map<std::string, double>& feature_reference_values,
    const std::string& review_prefix,
    double frac_tolerance)
{
    Fsettings s = make_ngldm2d_settings(true);

    int feature = int(feature_);

    SCOPED_TRACE(review_prefix + feature_name);
    ASSERT_TRUE(feature_reference_values.count(feature_name) > 0);

    double total = 0;

    //==== image 1

    // Load data (slice #1)
    LR roidata1;
    load_masked_test_roi_data (roidata1, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask, sizeof(ibsi_phantom_z1_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f1;
    ASSERT_NO_THROW (f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value (roidata1.fvals);

    total += roidata1.fvals[feature][0];

    //==== image 2

    // Load data (slice #2)
    LR roidata2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask, sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f2;
    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    total += roidata2.fvals[feature][0];

    //==== image 3

    // Load data (slice #3)
    LR roidata3;
    load_masked_test_roi_data (roidata3, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask, sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f3;
    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value (roidata3.fvals);

    total += roidata3.fvals[feature][0];

    //==== image 4

    // Load data (slice #4)
    LR roidata4;
    load_masked_test_roi_data (roidata4, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask, sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    // Calculate features
    NGLDMfeature f4;
    ASSERT_NO_THROW(f4.calculate(roidata4, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata4.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f4.save_value (roidata4.fvals);

    total += roidata4.fvals[feature][0];

    // Verdict
    double aveTotal = total / 4.0;
    ASSERT_TRUE(agrees_gt(aveTotal, feature_reference_values.at(feature_name), frac_tolerance));
}
