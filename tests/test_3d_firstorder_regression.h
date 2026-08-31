#pragma once

#include "../src/nyx/featureset.h"
#include "test_3d_firstorder_common.h"
#include "test_ref_vals.h"

// Nyxus snapshots only: these values guard current behavior and establish no oracle vetting.
static const ref_vals_map<double> firstorder_3d_regression_ref_vals
{
    // Both ends are now in the volume's own domain, but the ROI range is a difference of integer
    // grey levels while the volume range keeps its fraction, so this ratio still exceeds its
    // implied upper bound of one by that truncation.
    { "3COVERED_IMAGE_INTENSITY_RANGE", 1.0002043207290587 },
    // MATLAB mad(x,1) takes the median absolute deviation; Nyxus takes the mean absolute
    // deviation about its median. Keep regression-only until the intended definition is resolved.
    { "3MEDIAN_ABSOLUTE_DEVIATION", 507.12380480410445 },
    // MATLAB trimmean removes samples by rank; Nyxus selects values through histogram-derived
    // P10/P90 thresholds. Keep regression-only until the intended trimming semantics are resolved.
    { "3ROBUST_MEAN", 953.51896425966447 }
};

static void assert_3d_firstorder_feature_regression(
    const Nyxus::Feature3D& expected_fcode,
    const std::string& fname)
{
    SCOPED_TRACE(std::string("REGRESSION__") + fname);
    ASSERT_TRUE(firstorder_3d_regression_ref_vals.count(fname) > 0) << fname;

    FeatureSet features;
    int fcode = -1;
    ASSERT_TRUE(features.find_3D_FeatureByString(fname, fcode)) << fname;
    ASSERT_EQ(static_cast<int>(expected_fcode), fcode) << fname;

    std::vector<std::vector<double>> values;
    calculate_3d_firstorder_values(values);
    ASSERT_LT(static_cast<std::size_t>(fcode), values.size()) << fname;
    ASSERT_FALSE(values[fcode].empty()) << fname;

    const double expected = firstorder_3d_regression_ref_vals.at(fname);
    const double tolerance = std::max(1.0e-9, std::abs(expected) * 1.0e-6);
    ASSERT_NEAR(values[fcode][0], expected, tolerance) << fname;
}

void test_3d_firstorder_covered_image_intensity_range_regression()
{
    assert_3d_firstorder_feature_regression(
        Nyxus::Feature3D::COVERED_IMAGE_INTENSITY_RANGE,
        "3COVERED_IMAGE_INTENSITY_RANGE");
}

void test_3d_firstorder_median_absolute_deviation_regression()
{
    assert_3d_firstorder_feature_regression(
        Nyxus::Feature3D::MEDIAN_ABSOLUTE_DEVIATION,
        "3MEDIAN_ABSOLUTE_DEVIATION");
}

void test_3d_firstorder_robust_mean_regression()
{
    assert_3d_firstorder_feature_regression(
        Nyxus::Feature3D::ROBUST_MEAN,
        "3ROBUST_MEAN");
}
