#pragma once

#include "test_2d_ngtdm_common.h"   // gtest, <string>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// Digital phantom values for the 2D NGTDM family
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021.
// https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
//
// These are published to three significant figures, which is what sets this file's rel=1e-2
// tolerance -- the worst residual against the full-precision values is 0.41% on NGTDM_COARSENESS.
// The exact digits are pinned against mirp in test_2d_ngtdm_mirp.h, per slice as well as averaged;
// the two files are complementary, IBSI fixing the definition and mirp the digits.
//
// Every entry here carries three significant figures, which is the precision the reference manual
// publishes at: a longer literal in this table is not a transcription of anything IBSI printed. See
// tests/vetting/audit/ngtdm_2d_ibsi_vetting_report.md.
static const ref_vals_map<double> ngtdm_2d_ibsi_ref_vals
{
    {"NGTDM_COARSENESS", 0.121},  // Coarseness
    {"NGTDM_CONTRAST",   0.925},  // Contrast
    {"NGTDM_BUSYNESS",   2.99},   // Busyness
    {"NGTDM_COMPLEXITY", 10.4},   // Complexity
    {"NGTDM_STRENGTH",   2.88}    // Strength
};

// rel=1e-2: the published values carry three significant figures, and the worst residual against
// them is 0.41% (NGTDM_COARSENESS, 0.121 published against 0.1205106 computed)
static const double ngtdm_2d_ibsi_frac_tolerance = 100.;

void assert_ngtdm_feature_ibsi (const Feature2D& feature_, const std::string& feature_name)
{
    assert_ngtdm_feature_against_golden_values (feature_, feature_name, ngtdm_2d_ibsi_ref_vals,
                                                "ibsi ", ngtdm_2d_ibsi_frac_tolerance,
                                                make_ngtdm2d_settings (true), 0);
}

void test_2d_ngtdm_coarseness_ibsi()
{
    assert_ngtdm_feature_ibsi (Nyxus::Feature2D::NGTDM_COARSENESS, "NGTDM_COARSENESS");
}

void test_2d_ngtdm_contrast_ibsi()
{
    assert_ngtdm_feature_ibsi (Nyxus::Feature2D::NGTDM_CONTRAST, "NGTDM_CONTRAST");
}

void test_2d_ngtdm_busyness_ibsi()
{
    assert_ngtdm_feature_ibsi (Nyxus::Feature2D::NGTDM_BUSYNESS, "NGTDM_BUSYNESS");
}

void test_2d_ngtdm_complexity_ibsi()
{
    assert_ngtdm_feature_ibsi (Nyxus::Feature2D::NGTDM_COMPLEXITY, "NGTDM_COMPLEXITY");
}

void test_2d_ngtdm_strength_ibsi()
{
    assert_ngtdm_feature_ibsi (Nyxus::Feature2D::NGTDM_STRENGTH, "NGTDM_STRENGTH");
}
