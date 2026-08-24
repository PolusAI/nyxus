#pragma once

#include "test_2d_gldm_common.h"   // gtest, <string>, the slice featuriser and the mean assertion
#include "test_ref_vals.h"         // ref_vals_map

// Digital phantom values for the 2D GLDM family
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021.
// https://ibsi.readthedocs.io/en/latest/03_Image_features.html
// Dataset: dig phantom. Aggr. method: 2D, averaged)
//
// IBSI publishes these under the NGLDM name: a GLDM dependence count is 1 + the number of
// 8-neighbours sharing the centre's grey level, which is IBSI's j = k + 1 at coarseness alpha=0,
// distance d=1, so the two families' features are the same quantities under different names. The
// dependence axis is spelled small/large here and low/high there, hence GLDM_SDE against the
// low-dependence emphasis value and GLDM_LDE against the high-dependence one. The mapping and the
// measurement establishing it are in tests/vetting/audit/gldm_2d_pyradiomics_vetting_report.md.
//
// These are published to three significant figures, which is what sets this file's rel=1e-2
// tolerance. The full-precision digits are pinned against PyRadiomics in test_2d_gldm_pyradiomics.h.
static const ref_vals_map<double> gldm_2d_ibsi_ref_vals {
    {"GLDM_SDE", 0.158},        // Low dependence emphasis, p.120
    {"GLDM_LDE", 19.2},         // High dependence emphasis, p.121
    {"GLDM_LGLE", 0.702},       // Low grey level count emphasis, p.121
    {"GLDM_HGLE", 7.49},        // High grey level count emphasis, p.122
    {"GLDM_SDLGLE", 0.0473},    // Low dependence low grey level emphasis, p.122
    {"GLDM_SDHGLE", 3.06},      // Low dependence high grey level emphasis, p.123
    {"GLDM_LDLGLE", 17.6},      // High dependence low grey level emphasis, p.123
    {"GLDM_LDHGLE", 49.5},      // High dependence high grey level emphasis, p.124
    {"GLDM_GLN", 10.2},         // Grey level non-uniformity, p.124
    {"GLDM_DN", 3.96},          // Dependence count non-uniformity, p.125
    {"GLDM_DNN", 0.212},        // Normalised dependence count non-uniformity, p.125
    {"GLDM_GLV", 2.7},          // Grey level variance, p.127
    {"GLDM_DV", 2.73},          // Dependence count variance, p.127
    {"GLDM_DE", 2.71}           // Dependence count entropy, p.128
};

// rel=1e-2: the published values carry three significant figures, and the worst residual against
// them is 0.45% (GLDM_GLN, 10.2 published against 10.2464 computed)
static const double gldm_2d_ibsi_frac_tolerance = 100.;

static void assert_gldm_feature_ibsi (const Feature2D& feature_, const std::string& feature_name)
{
    // the published consensus values are means only, so the mean is all there is to assert here
    assert_gldm_mean_against_golden_values (gldm_2d_phantom_slice_values (feature_), feature_name,
                                            gldm_2d_ibsi_ref_vals, "ibsi ",
                                            gldm_2d_ibsi_frac_tolerance);
}

void test_2d_gldm_sde_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_SDE, "GLDM_SDE");
}

void test_2d_gldm_lde_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_LDE, "GLDM_LDE");
}

void test_2d_gldm_lgle_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_LGLE, "GLDM_LGLE");
}

void test_2d_gldm_hgle_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_HGLE, "GLDM_HGLE");
}

void test_2d_gldm_sdlgle_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_SDLGLE, "GLDM_SDLGLE");
}

void test_2d_gldm_sdhgle_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_SDHGLE, "GLDM_SDHGLE");
}

void test_2d_gldm_ldlgle_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_LDLGLE, "GLDM_LDLGLE");
}

void test_2d_gldm_ldhgle_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_LDHGLE, "GLDM_LDHGLE");
}

void test_2d_gldm_gln_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_GLN, "GLDM_GLN");
}

void test_2d_gldm_dn_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_DN, "GLDM_DN");
}

void test_2d_gldm_dnn_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_DNN, "GLDM_DNN");
}

void test_2d_gldm_glv_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_GLV, "GLDM_GLV");
}

void test_2d_gldm_dv_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_DV, "GLDM_DV");
}

void test_2d_gldm_de_ibsi()
{
    assert_gldm_feature_ibsi (Nyxus::Feature2D::GLDM_DE, "GLDM_DE");
}
