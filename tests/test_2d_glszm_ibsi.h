#pragma once

#include "test_2d_glszm_common.h"   // gtest, <string>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// Digital phantom values for the 2D GLSZM family
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021.
// https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
//
// These are published to three significant figures, which is what sets this file's rel=1e-2
// tolerance -- the worst residual against the full-precision values is 0.42% on GLSZM_LAHGLE. The
// exact digits are pinned against mirp in test_2d_glszm_mirp.h, per slice as well as averaged; the
// two files are complementary, IBSI fixing the definition and mirp the digits.
//
// Every entry here carries three significant figures, which is the precision the reference manual
// publishes at: a longer literal in this table is not a transcription of anything IBSI printed. See
// tests/vetting/audit/glszm_2d_ibsi_vetting_report.md.
static const ref_vals_map<double> glszm_2d_ibsi_ref_vals
{
    {"GLSZM_SAE",       0.363},  // Small area emphasis
    {"GLSZM_LAE",       43.9},   // Large area emphasis
    {"GLSZM_LGLZE",     0.371},  // Low grey level zone emphasis
    {"GLSZM_HGLZE",     16.4},   // High grey level zone emphasis
    {"GLSZM_SALGLE",    0.0259}, // Small area low grey level emphasis
    {"GLSZM_SAHGLE",    10.3},   // Small area high grey level emphasis
    {"GLSZM_LALGLE",    40.4},   // Large area low grey level emphasis
    {"GLSZM_LAHGLE",    113},    // Large area high grey level emphasis
    {"GLSZM_GLN",       1.41},   // Grey level non-uniformity
    {"GLSZM_GLNN",      0.323},  // Normalised grey level non-uniformity
    {"GLSZM_SZN",       1.49},   // Zone size non-uniformity
    {"GLSZM_SZNN",      0.333},  // Normalised zone size non-uniformity
    {"GLSZM_ZP",        0.24},   // Zone percentage
    {"GLSZM_GLV",       3.97},   // Grey level variance
    {"GLSZM_ZV",        21},     // Zone size variance
    {"GLSZM_ZE",        1.93}    // Zone size entropy
};

// rel=1e-2: the published values carry three significant figures, and the worst residual against
// them is 0.42% (GLSZM_LAHGLE, 113 published against 112.52142857 computed)
static const double glszm_2d_ibsi_frac_tolerance = 100.;

void assert_glszm_feature_ibsi (const Feature2D& feature_, const std::string& feature_name)
{
    assert_glszm_feature_against_golden_values (feature_, feature_name, glszm_2d_ibsi_ref_vals,
                                                "ibsi ", glszm_2d_ibsi_frac_tolerance,
                                                make_glszm2d_settings (true, 128));
}

void test_2d_glszm_sae_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_SAE, "GLSZM_SAE");
}

void test_2d_glszm_lae_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_LAE, "GLSZM_LAE");
}

void test_2d_glszm_lglze_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_LGLZE, "GLSZM_LGLZE");
}

void test_2d_glszm_hglze_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_HGLZE, "GLSZM_HGLZE");
}

void test_2d_glszm_salgle_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_SALGLE, "GLSZM_SALGLE");
}

void test_2d_glszm_sahgle_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_SAHGLE, "GLSZM_SAHGLE");
}

void test_2d_glszm_lalgle_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_LALGLE, "GLSZM_LALGLE");
}

void test_2d_glszm_lahgle_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_LAHGLE, "GLSZM_LAHGLE");
}

void test_2d_glszm_gln_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_GLN, "GLSZM_GLN");
}

void test_2d_glszm_glnn_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_GLNN, "GLSZM_GLNN");
}

void test_2d_glszm_szn_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_SZN, "GLSZM_SZN");
}

void test_2d_glszm_sznn_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_SZNN, "GLSZM_SZNN");
}

void test_2d_glszm_zp_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_ZP, "GLSZM_ZP");
}

void test_2d_glszm_glv_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_GLV, "GLSZM_GLV");
}

void test_2d_glszm_zv_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_ZV, "GLSZM_ZV");
}

void test_2d_glszm_ze_ibsi()
{
    assert_glszm_feature_ibsi (Nyxus::Feature2D::GLSZM_ZE, "GLSZM_ZE");
}
