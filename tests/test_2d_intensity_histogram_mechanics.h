#pragma once

// Intensity-histogram GATING, not values: IH features are only computed when the IBSI path is on, and
// these two assertions pin that contract - features return NaN with ibsi=false, and the required
// predicate reports what the family needs. Mechanics per SPEC 2, so they live here rather than in
// test_2d_intensity_histogram_regression.h, whose assertions are pinned feature values.

#include "test_2d_intensity_histogram_regression.h"   // shared fixture: run_intensity_histogram_fixture

// 3) IBSI gate: with IBSI off the family returns the soft-NaN sentinel for all 46.
void test_2d_intensity_histogram_gate_off_returns_nan_mechanics()
{
    const double sentinel = -7777.0;
    std::vector<std::vector<double>> fv;
    run_intensity_histogram_fixture(fv, ih_make_settings(3, /*ibsi*/ false, sentinel));

    for (auto fc : IntensityHistogramFeatures::featureset)
        ASSERT_DOUBLE_EQ(fv[(int)fc][0], sentinel);
}
// 5) required(): the class is only "required" when at least one IH feature is enabled.
void test_2d_intensity_histogram_required_predicate_mechanics()
{
    FeatureSet fs;
    fs.enableAll(false);
    ASSERT_FALSE(IntensityHistogramFeatures::required(fs));
    fs.enableFeature((int)Feature2D::IH_ENTROPY_VAL);
    ASSERT_TRUE(IntensityHistogramFeatures::required(fs));
}
