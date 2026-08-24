#pragma once

#include "test_2d_gldm_common.h"   // gtest, <string>, make_gldm2d_settings, gldm_2d_feature_value
#include "test_ref_vals.h"         // ref_vals_map

// Pinned Nyxus output on the cat2500 fixture at the settings assert_gldm_feature_regression uses
// (GREYDEPTH=128, PIXELDISTANCE=5, ibsi=false). SPEC 2 regression tier: a drift guard, no oracle
// claim -- GLDM 2D is vetted against PyRadiomics in test_2d_gldm_pyradiomics.h and against the IBSI
// consensus values in test_2d_gldm_ibsi.h, both on the digital phantom in IBSI mode. This fixture
// and this discretisation are the production default, which no oracle in the tree covers.
static const ref_vals_map<double> gldm_2d_regression_ref_vals {
    {"GLDM_SDE", 0.43899590049484488},
    {"GLDM_LDE", 24.932266009852217},
    {"GLDM_LGLE", 0.01175642780523556},
    {"GLDM_HGLE", 11512.041256157636},
    {"GLDM_SDLGLE", 0.011708932179969746},
    {"GLDM_SDHGLE", 2715.4140745395039},
    {"GLDM_LDLGLE", 0.013334084084609749},
    {"GLDM_LDHGLE", 400134.37623152707},
    {"GLDM_GLN", 453.0615763546798},
    {"GLDM_DN", 378.68719211822662},
    {"GLDM_DNN", 0.23318176854570605},
    {"GLDM_GLV", 1896.8138388307173},
    {"GLDM_DV", 8.4758547890024012},
    {"GLDM_DE", 5.3430148357241016}
};

/// @brief Pins one GLDM feature against its recorded value on the cat2500 fixture
void assert_gldm_feature_regression (const Feature2D& feature_, const std::string& feature_name)
{
    SCOPED_TRACE ("regression " + feature_name);

    // a missing key would otherwise be compared against a default-inserted zero
    ASSERT_TRUE (gldm_2d_regression_ref_vals.count(feature_name) > 0) << feature_name;

    const Fsettings s = make_gldm2d_settings (false);
    const double fval = gldm_2d_feature_value (cat2500_int, cat2500_seg,
                                               sizeof(cat2500_seg) / sizeof(NyxusPixel), s, feature_);

    ASSERT_TRUE (agrees_gt (fval, gldm_2d_regression_ref_vals.at(feature_name))) << feature_name;
}

void test_2d_gldm_sde_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_SDE, "GLDM_SDE");
}

void test_2d_gldm_lde_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_LDE, "GLDM_LDE");
}

void test_2d_gldm_lgle_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_LGLE, "GLDM_LGLE");
}

void test_2d_gldm_hgle_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_HGLE, "GLDM_HGLE");
}

void test_2d_gldm_sdlgle_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_SDLGLE, "GLDM_SDLGLE");
}

void test_2d_gldm_sdhgle_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_SDHGLE, "GLDM_SDHGLE");
}

void test_2d_gldm_ldlgle_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_LDLGLE, "GLDM_LDLGLE");
}

void test_2d_gldm_ldhgle_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_LDHGLE, "GLDM_LDHGLE");
}

void test_2d_gldm_gln_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_GLN, "GLDM_GLN");
}

void test_2d_gldm_dn_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_DN, "GLDM_DN");
}

void test_2d_gldm_dnn_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_DNN, "GLDM_DNN");
}

void test_2d_gldm_glv_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_GLV, "GLDM_GLV");
}

void test_2d_gldm_dv_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_DV, "GLDM_DV");
}

void test_2d_gldm_de_regression()
{
    assert_gldm_feature_regression (Nyxus::Feature2D::GLDM_DE, "GLDM_DE");
}
