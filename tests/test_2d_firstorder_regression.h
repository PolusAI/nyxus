#pragma once

// 2D first-order drift guards: pinned Nyxus output, no external reference.
// Provenance and vetting history: tests/vetting/audit/firstorder_2d_matlab_vetting_report.md.

#include "test_2d_firstorder_common.h"
#include "test_ref_vals.h"

static const ref_vals_map<double> firstorder_2d_regression_ref_vals {
	{"ENTROPY", 4.12733},
	// MATLAB mad(x,1) is the median absolute deviation; Nyxus takes the mean absolute
	// deviation about its median. Keep regression-only until the intended definition is resolved.
	{"MEDIAN_ABSOLUTE_DEVIATION", 1.269384415584416e+04},
	// MATLAB trimmean removes samples by rank; Nyxus selects values through histogram-derived
	// P10/P90 thresholds. Keep regression-only until the intended trimming semantics are resolved.
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

void test_2d_firstorder_median_absolute_deviation_regression()
{
	assert_firstorder_feature_regression(
		Nyxus::Feature2D::MEDIAN_ABSOLUTE_DEVIATION,
		"MEDIAN_ABSOLUTE_DEVIATION");
}
void test_2d_firstorder_robust_mean_regression()
{
    assert_firstorder_feature_regression(Nyxus::Feature2D::ROBUST_MEAN, "ROBUST_MEAN");
}
