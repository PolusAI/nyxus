#pragma once

// 2D first-order drift guards: pinned Nyxus output, no external reference.
// Provenance and vetting history: tests/vetting/audit/firstorder_2d_matlab_vetting_report.md.

#include "test_2d_firstorder_common.h"
#include "test_ref_vals.h"

static const ref_vals_map<double> firstorder_2d_regression_ref_vals {
    {"ENTROPY",     4.12733},
    {"P01",         1.189536940000000e+04},
    {"P25",         1.907482583333333e+04},
    {"P75",         4.580120500000000e+04},
    {"P99",         6.341676030000000e+04},
    {"QCOD",        4.119607630640470e-01},
    {"ROBUST_MEAN", 3.142136800000000e+04},
};

// Computes the canonical ROI once and compares one feature against its pinned Nyxus value.
void assert_firstorder_feature_regression(const Feature2D& feature, const std::string& feature_name,
                                          double frac_tolerance = 1000.,
                                          Fsettings s = Fsettings())
{
    std::vector<std::vector<double>> fvals;
    calculate_pixel_intensity_feature_values(fvals, s);

    ASSERT_TRUE(agrees_gt(fvals[(int)feature][0],
                          firstorder_2d_regression_ref_vals.at(feature_name),
                          frac_tolerance));
}

// Pins the non-IBSI GREYDEPTH=20 histogram path (the pyradiomics-oracle ENTROPY test uses binCount=64).
void test_2d_firstorder_entropy_regression()
{
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::GREYDEPTH].ival = 20;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::IBSI].bval = false;

    assert_firstorder_feature_regression(Nyxus::Feature2D::ENTROPY, "ENTROPY", 1000., s);
}

void test_2d_firstorder_p01_regression()
{
    assert_firstorder_feature_regression(Nyxus::Feature2D::P01, "P01");
}
void test_2d_firstorder_p25_regression()
{
    assert_firstorder_feature_regression(Nyxus::Feature2D::P25, "P25");
}
void test_2d_firstorder_p75_regression()
{
    assert_firstorder_feature_regression(Nyxus::Feature2D::P75, "P75");
}
void test_2d_firstorder_p99_regression()
{
    assert_firstorder_feature_regression(Nyxus::Feature2D::P99, "P99");
}
void test_2d_firstorder_qcod_regression()
{
    assert_firstorder_feature_regression(Nyxus::Feature2D::QCOD, "QCOD");
}
void test_2d_firstorder_robust_mean_regression()
{
    assert_firstorder_feature_regression(Nyxus::Feature2D::ROBUST_MEAN, "ROBUST_MEAN");
}
