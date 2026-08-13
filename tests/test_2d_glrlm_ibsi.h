#pragma once

#include <gtest/gtest.h>

#include <string>

#include "test_2d_glrlm_common.h"   // the phantom fixture and calc_2d_glrlm_phantom_feature
#include "test_main_nyxus.h"        // agrees_gt
#include "test_ref_vals.h"          // ref_vals_map

// IBSI digital phantom, 2D direction- and slice-averaged consensus values.
// (Reference: IBSI Documentation, Release 0.0.1dev Dec 13, 2021. Dataset: dig phantom.)
// Verified against fresh PyRadiomics and MIRP runs on the phantom pixels checked into test_data.h -
// see tests/vetting/audit/glrlm_2d_ibsi_vetting_report.md.
static ref_vals_map<double> glrlm_2d_ibsi_ref_vals {
    {"GLRLM_SRE", 0.641},
    {"GLRLM_LRE", 3.78},
    {"GLRLM_LGLRE", 0.604},
    {"GLRLM_HGLRE", 9.82},
    {"GLRLM_SRLGLE", 0.294},
    {"GLRLM_SRHGLE", 8.57},
    {"GLRLM_LRLGLE", 3.14},
    {"GLRLM_LRHGLE", 17.4},
    {"GLRLM_GLN", 5.2},
    {"GLRLM_GLNN", 0.46},
    {"GLRLM_RLN", 6.12},
    {"GLRLM_RLNN", 0.492},
    {"GLRLM_RP", 0.627},
    {"GLRLM_GLV", 3.35},
    {"GLRLM_RV", 0.761},
    {"GLRLM_RE", 2.17}
};

// A base feature and its _AVE twin are the same quantity aggregated at different points, so they
// read the same golden; the table is keyed by the base name.
static std::string glrlm_ibsi_golden_key (const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    return is_2d_glrlm_ave_feature (feature_name)
        ? feature_name.substr (0, feature_name.size() - ave_suffix.size())
        : feature_name;
}

void assert_glrlm_feature_ibsi (const std::string& feature_name)
{
    // A key absent from the table would otherwise be default-inserted as 0, and agrees_gt turns a
    // ground truth of 0 into a tolerance of 0 - an assertion that passes only on an exact 0.
    auto golden_it = glrlm_2d_ibsi_ref_vals.find (glrlm_ibsi_golden_key (feature_name));
    ASSERT_TRUE (golden_it != glrlm_2d_ibsi_ref_vals.end()) << feature_name;

    double value = 0;
    ASSERT_TRUE (calc_2d_glrlm_phantom_feature (feature_name, value)) << feature_name;
    ASSERT_TRUE (agrees_gt (value, golden_it->second, 100.)) << feature_name;
}

void test_2d_glrlm_sre_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_SRE");
}

void test_2d_glrlm_lre_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LRE");
}

void test_2d_glrlm_lglre_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LGLRE");
}

void test_2d_glrlm_hglre_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_HGLRE");
}

void test_2d_glrlm_srlgle_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_SRLGLE");
}

void test_2d_glrlm_srhgle_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_SRHGLE");
}

void test_2d_glrlm_lrlgle_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LRLGLE");
}

void test_2d_glrlm_lrhgle_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LRHGLE");
}

void test_2d_glrlm_gln_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_GLN");
}

void test_2d_glrlm_glnn_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_GLNN");
}

void test_2d_glrlm_rln_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RLN");
}

void test_2d_glrlm_rlnn_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RLNN");
}

void test_2d_glrlm_rp_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RP");
}

void test_2d_glrlm_glv_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_GLV");
}

void test_2d_glrlm_rv_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RV");
}

void test_2d_glrlm_re_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RE");
}

// The _AVE features are what Nyxus reports for the direction-averaged aggregation the IBSI values
// are quoted at. Without these the averaging step itself is asserted nowhere: the checks above read
// the 4 directional values and average them in the test, never touching the aggregated value.
void test_2d_glrlm_sre_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_SRE_AVE");
}

void test_2d_glrlm_lre_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LRE_AVE");
}

void test_2d_glrlm_lglre_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LGLRE_AVE");
}

void test_2d_glrlm_hglre_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_HGLRE_AVE");
}

void test_2d_glrlm_srlgle_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_SRLGLE_AVE");
}

void test_2d_glrlm_srhgle_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_SRHGLE_AVE");
}

void test_2d_glrlm_lrlgle_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LRLGLE_AVE");
}

void test_2d_glrlm_lrhgle_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_LRHGLE_AVE");
}

void test_2d_glrlm_gln_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_GLN_AVE");
}

void test_2d_glrlm_glnn_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_GLNN_AVE");
}

void test_2d_glrlm_rln_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RLN_AVE");
}

void test_2d_glrlm_rlnn_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RLNN_AVE");
}

void test_2d_glrlm_rp_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RP_AVE");
}

void test_2d_glrlm_glv_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_GLV_AVE");
}

void test_2d_glrlm_rv_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RV_AVE");
}

void test_2d_glrlm_re_ave_ibsi()
{
    assert_glrlm_feature_ibsi ("GLRLM_RE_AVE");
}
