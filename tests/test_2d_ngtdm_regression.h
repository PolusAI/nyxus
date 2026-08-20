#pragma once

#include "test_2d_ngtdm_common.h"   // gtest, <string>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// Pinned Nyxus output on the four IBSI phantom slices in DEFAULT mode -- ibsi=false with
// NGTDMFeature::n_levels set to 100 -- averaged. SPEC 2 regression tier: a drift guard, no oracle
// claim.
//
// This is a different config point from the oracle files, not a duplicate of them. In default mode
// Nyxus bins the intensities to a fixed grey count instead of using the phantom's own levels, and no
// reference implementation reproduces that: on this same fixture NGTDM_CONTRAST is 0.925 in IBSI
// mode and 3169.93 here. That is the mode a caller gets without asking for IBSI compliance, so it is
// worth a drift guard even though nothing vets it. The IBSI-mode values are vetted against mirp in
// test_2d_ngtdm_mirp.h and against the published consensus in test_2d_ngtdm_ibsi.h.
//
// The 100 matters and is passed explicitly rather than assigned to the static and left: these values
// exist only at that grey count. At the default 0 the same fixture gives NGTDM_CONTRAST 6634.50,
// more than twice this pin, so a leaked or missing setting would move every number here.
static const ref_vals_map<double> ngtdm_2d_regression_ref_vals
{
    {"NGTDM_COARSENESS", 0.0083740684013098136},
    {"NGTDM_CONTRAST", 3169.9290851041665},
    {"NGTDM_BUSYNESS", 1.4445714038157604},
    {"NGTDM_COMPLEXITY", 3608.3891877968335},
    {"NGTDM_STRENGTH", 52.07664266981719}
};

// the grey count these pins were recorded at; see the table comment
static const int ngtdm_2d_regression_n_levels = 100;

void assert_ngtdm_feature_regression (const Feature2D& feature_, const std::string& feature_name)
{
    // agrees_gt's default frac_tolerance is rel=1e-3, which is the drift band these guards want:
    // the quantities are means over four slices and nothing but a real change should move them
    assert_ngtdm_feature_against_golden_values (feature_, feature_name, ngtdm_2d_regression_ref_vals,
                                                "regression ", 1000.,
                                                make_ngtdm2d_settings (false),
                                                ngtdm_2d_regression_n_levels);
}

void test_2d_ngtdm_coarseness_regression()
{
    assert_ngtdm_feature_regression (Nyxus::Feature2D::NGTDM_COARSENESS, "NGTDM_COARSENESS");
}

void test_2d_ngtdm_contrast_regression()
{
    assert_ngtdm_feature_regression (Nyxus::Feature2D::NGTDM_CONTRAST, "NGTDM_CONTRAST");
}

void test_2d_ngtdm_busyness_regression()
{
    assert_ngtdm_feature_regression (Nyxus::Feature2D::NGTDM_BUSYNESS, "NGTDM_BUSYNESS");
}

void test_2d_ngtdm_complexity_regression()
{
    assert_ngtdm_feature_regression (Nyxus::Feature2D::NGTDM_COMPLEXITY, "NGTDM_COMPLEXITY");
}

void test_2d_ngtdm_strength_regression()
{
    assert_ngtdm_feature_regression (Nyxus::Feature2D::NGTDM_STRENGTH, "NGTDM_STRENGTH");
}
