#pragma once

#include "test_2d_gldzm_common.h"   // gtest, <string>, <vector>, the fixture and the mean helper
#include "test_ref_vals.h"          // ref_vals_map

// Digital phantom values for the 2D GLDZM family
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021.
// https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
//
// These are published to three significant figures, which is what sets this file's rel=1e-2
// tolerance -- the worst residual against the full-precision values is 0.35% on GLDZM_LDE. The
// exact digits are pinned against mirp in test_2d_gldzm_mirp.h, per slice as well as averaged; the
// two files are complementary, IBSI fixing the definition and mirp the digits.
//
// Four features are deliberately absent. GLDZM_SDLGLE and GLDZM_ZDV are covered by mirp at the
// exact tier; GLDZM_GLM and GLDZM_ZDM are not IBSI GLDZM features and have no mirp column, so they
// are drift-pinned in test_2d_gldzm_regression.h. Every entry here carries three significant
// figures, which is the precision the reference manual publishes at: a longer literal in this table
// is not a transcription of anything IBSI printed. See
// tests/vetting/audit/gldzm_2d_ibsi_vetting_report.md.
static const ref_vals_map<double> gldzm_2d_ibsi_ref_vals
{
    {"GLDZM_SDE",       0.946}, // Small distance emphasis
    {"GLDZM_LDE",       1.21},  // Large distance emphasis
    {"GLDZM_LGLZE",     0.371}, // Low grey level zone emphasis
    {"GLDZM_HGLZE",     16.4},  // High grey level zone emphasis
    {"GLDZM_SDHGLE",    15.2},  // Small distance high grey level emphasis
    {"GLDZM_LDLGLE",    0.386}, // Large distance low grey level emphasis
    {"GLDZM_LDHGLE",    21.3},  // Large distance high grey level emphasis
    {"GLDZM_GLNU",      1.41},  // Grey level non-uniformity
    {"GLDZM_GLNUN",     0.323}, // Normalised grey level non-uniformity
    {"GLDZM_ZDNU",      3.79},  // Zone distance non-uniformity
    {"GLDZM_ZDNUN",     0.898}, // Normalised zone distance non-uniformity
    {"GLDZM_ZP",        0.24},  // Zone percentage
    {"GLDZM_GLV",       3.97},  // Grey level variance
    {"GLDZM_ZDE",       1.73}   // Zone distance entropy
};

// rel=1e-2: the published values carry three significant figures, and the worst residual against
// them is 0.35% (GLDZM_LDE, 1.21 published against 1.2142857 computed)
static const double gldzm_2d_ibsi_frac_tolerance = 100.;

// The IBSI GLDZ-matrix worked example (figure 3.17): a 4x4 ROI whose reference matrix is pinned in
// test_data.h. This checks the matrix itself rather than the features built from it, so it runs on
// its own fixture and its own settings.
void assert_gldzm_matrix_ibsi()
{
    const Fsettings s = make_gldzm2d_settings (true);

    // Load a test image
    LR roidata;
    load_masked_test_roi_data (roidata, ibsi_fig3_17a_gldzm_sample_image_int, ibsi_fig3_17a_gldzm_sample_image_mask, sizeof(ibsi_fig3_17a_gldzm_sample_image_mask) / sizeof(NyxusPixel));

    // In this test, we only calculate and examine the GLDZ-matrix without calculating features
    GLDZMFeature f;

    // Have the feature object to create the GLDZM matrix kit (matrix itself, LUT of grey tones (0-max in IBSI mode, unique otherwise), and GLDZM's dimensions)
    std::vector<PixIntens> greysLUT;
    SimpleMatrix<unsigned int> GLDZM;
    int Ng,	// number of grey levels
        Nd;	// maximum number of non-zero dependencies
    ASSERT_NO_THROW(f.prepare_GLDZM_matrix_kit (GLDZM, Ng, Nd, greysLUT, roidata, STNGS_NGREYS(s), STNGS_IBSI(s)));

    // Count discrepancies
    int n_mismatches = 0;
    for (int g = 0; g < Ng; g++)
        for (int d = 0; d < Nd; d++)
        {
            auto ibsi_reference_matrix_value = ibsi_fig3_17c_gldzm_reference_matrix [g * Nd + d];
            auto actual = GLDZM.yx (g,d);
            if (ibsi_reference_matrix_value != actual)
            {
                n_mismatches++;
                std::cout << "GLDZ-matrix mismatch! Expecting [g=" << g << ", d=" << d << "] = " << ibsi_reference_matrix_value << " not " << actual << "\n";
            }
        }

    ASSERT_TRUE(n_mismatches == 0);
}

static void assert_gldzm_feature_ibsi (const Feature2D& feature_, const std::string& feature_name)
{
    assert_gldzm_feature_against_golden_values (feature_, feature_name, gldzm_2d_ibsi_ref_vals,
                                                "ibsi ", gldzm_2d_ibsi_frac_tolerance);
}

void test_2d_gldzm_matrix_correctness_ibsi()
{
    assert_gldzm_matrix_ibsi();
}

void test_2d_gldzm_sde_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_SDE, "GLDZM_SDE");
}

void test_2d_gldzm_lde_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_LDE, "GLDZM_LDE");
}

void test_2d_gldzm_lglze_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_LGLZE, "GLDZM_LGLZE");
}

void test_2d_gldzm_hglze_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_HGLZE, "GLDZM_HGLZE");
}

void test_2d_gldzm_sdhgle_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_SDHGLE, "GLDZM_SDHGLE");
}

void test_2d_gldzm_ldlgle_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_LDLGLE, "GLDZM_LDLGLE");
}

void test_2d_gldzm_ldhgle_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_LDHGLE, "GLDZM_LDHGLE");
}

void test_2d_gldzm_glnu_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_GLNU, "GLDZM_GLNU");
}

void test_2d_gldzm_glnun_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_GLNUN, "GLDZM_GLNUN");
}

void test_2d_gldzm_zdnu_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_ZDNU, "GLDZM_ZDNU");
}

void test_2d_gldzm_zdnun_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_ZDNUN, "GLDZM_ZDNUN");
}

void test_2d_gldzm_zp_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_ZP, "GLDZM_ZP");
}

void test_2d_gldzm_glv_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_GLV, "GLDZM_GLV");
}

void test_2d_gldzm_zde_ibsi()
{
    assert_gldzm_feature_ibsi(Nyxus::Feature2D::GLDZM_ZDE, "GLDZM_ZDE");
}
