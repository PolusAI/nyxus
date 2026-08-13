#pragma once

// First-order features vetted against MATLAB on the canonical ROI
// (tests/test_data.h : pixelIntensityFeaturesTestData), at default settings.
//
// oracle_coverage.csv places these here: the 33 2D firstorder rows with oracle=matlab name
// test_2d_firstorder_matlab.h in target_test. Same shape as test_2d_firstorder_ibsi.h - one keyed golden
// map, one assert_ helper, one one-line test per feature - so a reader can see at a glance which
// reference each value comes from and every feature has its own reportable case.
//
// All 34 values below are MATLAB reference values - that is what oracle=matlab in the registry
// records. The pre-rename constant names said so directly: oracle_3p_matlab_* named the tool, and
// oracle_3p_builtin_* meant MATLAB's built-in statistics functions.
//
// WHAT IS MISSING is the SPEC 6.4 record, not the origin: no MATLAB version, no exact config and no
// generator script path is written down at these goldens - only the numbers. Adding that record (the
// Octave harness of MIGRATION 5.13 reproduces MATLAB for first-order) is tracked in not_covered.md
// section C, so a future reader can regenerate the values rather than trust them.
//
// REGISTRY CONFLICT, deliberately not propagated: the UNIFORMITY row reads oracle=pyradiomics with
// target_test=test_2d_firstorder_regression.h, yet the value below is MATLAB's. Moving it into a file
// named _regression would restate the mislabelling this wave removed, so the assertion stays here
// under its true name and the row is flagged for correction in not_covered.md.

#include "test_2d_firstorder_common.h"
#include "test_ref_vals.h"

static const ref_vals_map<double> firstorder_2d_matlab_ref_vals {
    {"HYPERSKEWNESS",                 1.978293086605381},
    {"HYPERFLATNESS",                 5.126659243028459},
    {"UNIFORMITY",                    0.0647664},
    {"UNIFORMITY_PIU",                29.477577192725725},
    {"COVERED_IMAGE_INTENSITY_RANGE", 8.088960097657740e-01},
    {"ROBUST_MEAN",                   3.142136800000000e+04},
    {"INTEGRATED_INTENSITY",          5015224},
    {"MIN",                           11079},
    {"MAX",                           64090},
    {"RANGE",                         53011},
    {"MEAN",                          3.256638961038961e+04},
    {"MEDIAN",                        2.980350000000000e+04},
    {"MODE",                          19552},
    {"STANDARD_DEVIATION",            1.473096831710767e+04},
    {"SKEWNESS",                      0.450256759704494},
    {"EXCESS_KURTOSIS",               1.927888720710090 - 3},
    {"KURTOSIS",                      1.927888720710090},
    {"MEAN_ABSOLUTE_DEVIATION",       1.283308449991567e+04},
    {"STANDARD_ERROR",                1.187055255225567e+03},
    {"ROOT_MEAN_SQUARED",             3.572341052638121e+04},
    {"ENERGY",                        1.965289571840000e+11},
    {"P01",                           1.189536940000000e+04},
    {"P10",                           1.610747200000000e+04},
    {"P25",                           1.907482583333333e+04},
    {"P75",                           4.580120500000000e+04},
    {"P90",                           5.338177800000000e+04},
    {"P99",                           6.341676030000000e+04},
    {"INTERQUARTILE_RANGE",           2.672637916666667e+04},
    {"COV",                           4.523365498399634e-01},
    {"MEDIAN_ABSOLUTE_DEVIATION",     1.269384415584416e+04},
    {"QCOD",                          4.119607630640470e-01},
    {"STANDARD_DEVIATION_BIASED",     1.468306260221863e+04},
    {"VARIANCE",                      2.170014275596299e+08},
    {"VARIANCE_BIASED",               2.155923273806713e+08},
};

// Computes the canonical ROI once and compares one feature against its MATLAB golden.
// frac_tolerance follows agrees_gt: larger is tighter; 100. is the 1% cross-tool tier (SPEC 7).
void assert_firstorder_feature_matlab(const Feature2D& feature, const std::string& feature_name,
                                      double frac_tolerance = 1000.,
                                      Fsettings s = Fsettings(),
                                      int slide_idx = -1, double slide_min = -1.0, double slide_max = -1.0)
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals, s, slide_idx, slide_min, slide_max);

    ASSERT_TRUE(agrees_gt(fvals[(int)feature][0],
                          firstorder_2d_matlab_ref_vals.at(feature_name),
                          frac_tolerance));
}

void test_2d_firstorder_hyperskewness_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::HYPERSKEWNESS, "HYPERSKEWNESS");
}
void test_2d_firstorder_hyperflatness_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::HYPERFLATNESS, "HYPERFLATNESS");
}
// UNIFORMITY is histogram-based, so the bin count is part of the comparison: MATLAB was matched at
// GREYDEPTH=20 with the IBSI path off (SPEC 5 config recipe). Asserted at the 1% cross-tool tier.
void test_2d_firstorder_uniformity_matlab()
{
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = 20;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::IBSI].bval = false;

    assert_firstorder_feature_matlab(Nyxus::Feature2D::UNIFORMITY, "UNIFORMITY", 100., s);
}
void test_2d_firstorder_uniformity_piu_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::UNIFORMITY_PIU, "UNIFORMITY_PIU");
}
// COVERED_IMAGE_INTENSITY_RANGE is the ROI range as a fraction of the SLIDE dynamic range, so the
// fixture must carry slide properties: slide 0 spanning 0..65535 (SPEC 5 config recipe).
void test_2d_firstorder_covered_image_intensity_range_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::COVERED_IMAGE_INTENSITY_RANGE,
                                     "COVERED_IMAGE_INTENSITY_RANGE", 1000., Fsettings(), 0, 0.0, 65535.0);
}
void test_2d_firstorder_robust_mean_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::ROBUST_MEAN, "ROBUST_MEAN");
}
void test_2d_firstorder_integrated_intensity_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::INTEGRATED_INTENSITY, "INTEGRATED_INTENSITY");
}
void test_2d_firstorder_min_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::MIN, "MIN");
}
void test_2d_firstorder_max_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::MAX, "MAX");
}
void test_2d_firstorder_range_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::RANGE, "RANGE");
}
void test_2d_firstorder_mean_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::MEAN, "MEAN");
}
void test_2d_firstorder_median_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::MEDIAN, "MEDIAN");
}
void test_2d_firstorder_mode_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::MODE, "MODE");
}
void test_2d_firstorder_standard_deviation_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::STANDARD_DEVIATION, "STANDARD_DEVIATION");
}
void test_2d_firstorder_skewness_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::SKEWNESS, "SKEWNESS");
}
void test_2d_firstorder_excess_kurtosis_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::EXCESS_KURTOSIS, "EXCESS_KURTOSIS");
}
void test_2d_firstorder_kurtosis_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::KURTOSIS, "KURTOSIS");
}
void test_2d_firstorder_mean_absolute_deviation_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::MEAN_ABSOLUTE_DEVIATION, "MEAN_ABSOLUTE_DEVIATION");
}
void test_2d_firstorder_standard_error_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::STANDARD_ERROR, "STANDARD_ERROR");
}
void test_2d_firstorder_root_mean_squared_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::ROOT_MEAN_SQUARED, "ROOT_MEAN_SQUARED");
}
void test_2d_firstorder_energy_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::ENERGY, "ENERGY");
}
void test_2d_firstorder_p01_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::P01, "P01");
}
void test_2d_firstorder_p10_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::P10, "P10");
}
void test_2d_firstorder_p25_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::P25, "P25");
}
void test_2d_firstorder_p75_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::P75, "P75");
}
void test_2d_firstorder_p90_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::P90, "P90");
}
void test_2d_firstorder_p99_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::P99, "P99");
}
void test_2d_firstorder_interquartile_range_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::INTERQUARTILE_RANGE, "INTERQUARTILE_RANGE");
}
void test_2d_firstorder_cov_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::COV, "COV");
}
void test_2d_firstorder_median_absolute_deviation_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::MEDIAN_ABSOLUTE_DEVIATION, "MEDIAN_ABSOLUTE_DEVIATION");
}
void test_2d_firstorder_qcod_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::QCOD, "QCOD");
}
void test_2d_firstorder_standard_deviation_biased_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::STANDARD_DEVIATION_BIASED, "STANDARD_DEVIATION_BIASED");
}
void test_2d_firstorder_variance_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::VARIANCE, "VARIANCE");
}
void test_2d_firstorder_variance_biased_matlab()
{
    assert_firstorder_feature_matlab(Nyxus::Feature2D::VARIANCE_BIASED, "VARIANCE_BIASED");
}
