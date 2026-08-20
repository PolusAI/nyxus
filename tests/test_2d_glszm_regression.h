#pragma once

#include "test_2d_glszm_common.h"   // gtest, <string>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// Pinned Nyxus output on the four IBSI phantom slices in DEFAULT mode -- ibsi=false at 64 grey
// levels -- averaged. SPEC 2 regression tier: a drift guard, no oracle claim.
//
// This is a different config point from the oracle files, not a duplicate of them. In default mode
// Nyxus weights the grey-level-dependent features by raw intensity instead of by the grey-level
// index, which no reference implementation reproduces: on this same fixture GLSZM_HGLZE is 16.44 in
// IBSI mode and 1497.57 here. That is the mode a caller gets without asking for IBSI compliance, so
// it is worth a drift guard even though nothing vets it. The IBSI-mode values are vetted against
// mirp in test_2d_glszm_mirp.h and against the published consensus in test_2d_glszm_ibsi.h.
//
// GLSZM_ZE goes through Nyxus::fast_log10 here as it does in IBSI mode, so this pin records the
// approximation's output; see test_2d_glszm_mirp.h for the measurement.
static const ref_vals_map<double> glszm_2d_regression_ref_vals
{
    {"GLSZM_SAE", 0.38873817495748297},
    {"GLSZM_LAE", 32.5},
    {"GLSZM_LGLZE", 0.19625508400750141},
    {"GLSZM_HGLZE", 1497.5749999999998},
    {"GLSZM_SALGLE", 0.10956092298348039},
    {"GLSZM_SAHGLE", 881.47065250318883},
    {"GLSZM_LALGLE", 0.76935847487393461},
    {"GLSZM_LAHGLE", 11048.975},
    {"GLSZM_GLN", 1.4874999999999998},
    {"GLSZM_GLNN", 0.27718750000000003},
    {"GLSZM_SZN", 1.9624999999999999},
    {"GLSZM_SZNN", 0.35968749999999999},
    {"GLSZM_ZP", 0.27500000000000002},
    {"GLSZM_GLV", 551.70375000000001},
    {"GLSZM_ZV", 16.6875},
    {"GLSZM_ZE", 2.2842901945114136}
};

/// @brief Pins one GLSZM feature against its recorded default-mode value on the IBSI phantom
void assert_glszm_feature_regression (const Feature2D& feature_, const std::string& feature_name)
{
    // agrees_gt's default frac_tolerance is rel=1e-3, which is the drift band these guards want:
    // the quantities are means over four slices and nothing but a real change should move them
    assert_glszm_feature_against_golden_values (feature_, feature_name, glszm_2d_regression_ref_vals,
                                                "regression ", 1000.,
                                                make_glszm2d_settings (false, 64));
}

void test_2d_glszm_sae_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_SAE, "GLSZM_SAE");
}

void test_2d_glszm_lae_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_LAE, "GLSZM_LAE");
}

void test_2d_glszm_lglze_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_LGLZE, "GLSZM_LGLZE");
}

void test_2d_glszm_hglze_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_HGLZE, "GLSZM_HGLZE");
}

void test_2d_glszm_salgle_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_SALGLE, "GLSZM_SALGLE");
}

void test_2d_glszm_sahgle_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_SAHGLE, "GLSZM_SAHGLE");
}

void test_2d_glszm_lalgle_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_LALGLE, "GLSZM_LALGLE");
}

void test_2d_glszm_lahgle_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_LAHGLE, "GLSZM_LAHGLE");
}

void test_2d_glszm_gln_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_GLN, "GLSZM_GLN");
}

void test_2d_glszm_glnn_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_GLNN, "GLSZM_GLNN");
}

void test_2d_glszm_szn_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_SZN, "GLSZM_SZN");
}

void test_2d_glszm_sznn_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_SZNN, "GLSZM_SZNN");
}

void test_2d_glszm_zp_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_ZP, "GLSZM_ZP");
}

void test_2d_glszm_glv_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_GLV, "GLSZM_GLV");
}

void test_2d_glszm_zv_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_ZV, "GLSZM_ZV");
}

void test_2d_glszm_ze_regression()
{
    assert_glszm_feature_regression (Nyxus::Feature2D::GLSZM_ZE, "GLSZM_ZE");
}
