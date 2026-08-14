#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "../src/nyx/featureset.h"   // UserFacingFeatureNames
#include "test_2d_glcm_common.h"     // the dense phantom and calc_2d_glcm_dense_feature
#include "test_main_nyxus.h"         // agrees_gt
#include "test_ref_vals.h"           // ref_vals_map

// mirp 2.6.0, recipe glcm.ibsi_identity (by_slice + 2d_average + no discretisation), dense 8x8 phantom
// Generated offline by tests/vetting/oracles/gen_glcm_mirp.py (SPEC 6.4) on the dense phantom
// test_2d_glcm_common.h builds, at recipe glcm.ibsi_identity: by_slice=True,
// base_discretisation_method="none", glcm_distance=1, glcm_spatial_method="2d_average".
//
// MIRP is the IBSI reference implementation, so it reports the two quantities PyRadiomics dropped
// as duplicates - dissimilarity and sum variance - in their own right, and GLCM_DIS and
// GLCM_SUMVARIANCE are vetted here against those rather than through an equality argument.
static ref_vals_map<double> glcm_2d_mirp_ref_vals
{
    {"GLCM_ACOR", 20.512755102040813},   // cm_auto_corr
    {"GLCM_ACOR_AVE", 20.512755102040813},   // cm_auto_corr
    {"GLCM_CLUPROM", 263.28733919002582},   // cm_clust_prom
    {"GLCM_CLUPROM_AVE", 263.28733919002582},   // cm_clust_prom
    {"GLCM_CLUSHADE", 0.017543710528775458},   // cm_clust_shade
    {"GLCM_CLUSHADE_AVE", 0.017543710528775458},   // cm_clust_shade
    {"GLCM_CLUTEND", 10.898896293211161},   // cm_clust_tend
    {"GLCM_CLUTEND_AVE", 10.898896293211161},   // cm_clust_tend
    {"GLCM_CONTRAST", 10.219387755102041},   // cm_contrast
    {"GLCM_CONTRAST_AVE", 10.219387755102041},   // cm_contrast
    {"GLCM_CORRELATION", 0.032164714988198242},   // cm_corr
    {"GLCM_CORRELATION_AVE", 0.032164714988198242},   // cm_corr
    {"GLCM_DIFAVE", 2.5586734693877551},   // cm_diff_avg
    {"GLCM_DIFAVE_AVE", 2.5586734693877551},   // cm_diff_avg
    {"GLCM_DIFENTRO", 0.71070036077609355},   // cm_diff_entr
    {"GLCM_DIFENTRO_AVE", 0.71070036077609355},   // cm_diff_entr
    {"GLCM_DIFVAR", 2.9454133694294038},   // cm_diff_var
    {"GLCM_DIFVAR_AVE", 2.9454133694294038},   // cm_diff_var
    {"GLCM_DIS", 2.5586734693877551},   // cm_dissimilarity
    {"GLCM_DIS_AVE", 2.5586734693877551},   // cm_dissimilarity
    {"GLCM_ASM", 0.063534725114535603},   // cm_energy
    {"GLCM_ASM_AVE", 0.063534725114535603},   // cm_energy
    {"GLCM_ENERGY", 0.063534725114535603},   // cm_energy
    {"GLCM_ENERGY_AVE", 0.063534725114535603},   // cm_energy
    {"GLCM_INFOMEAS1", -0.67050298521075136},   // cm_info_corr1
    {"GLCM_INFOMEAS1_AVE", -0.67050298521075136},   // cm_info_corr1
    {"GLCM_INFOMEAS2", 0.9910039918016994},   // cm_info_corr2
    {"GLCM_INFOMEAS2_AVE", 0.9910039918016994},   // cm_info_corr2
    {"GLCM_ID", 0.35283801020408162},   // cm_inv_diff
    {"GLCM_ID_AVE", 0.35283801020408162},   // cm_inv_diff
    {"GLCM_HOM1", 0.35283801020408162},   // cm_inv_diff
    {"GLCM_HOM1_AVE", 0.35283801020408162},   // cm_inv_diff
    {"GLCM_IDM", 0.27853769782341209},   // cm_inv_diff_mom
    {"GLCM_IDM_AVE", 0.27853769782341209},   // cm_inv_diff_mom
    {"GLCM_HOM2", 0.27853769782341209},   // cm_inv_diff_mom
    {"GLCM_IDMN", 0.88734163285067313},   // cm_inv_diff_mom_norm
    {"GLCM_IDMN_AVE", 0.88734163285067313},   // cm_inv_diff_mom_norm
    {"GLCM_IDN", 0.77947925090782233},   // cm_inv_diff_norm
    {"GLCM_IDN_AVE", 0.77947925090782233},   // cm_inv_diff_norm
    {"GLCM_IV", 0.50863378684807259},   // cm_inv_var
    {"GLCM_IV_AVE", 0.50863378684807259},   // cm_inv_var
    {"GLCM_JAVE", 4.5102040816326534},   // cm_joint_avg
    {"GLCM_JAVE_AVE", 4.5102040816326534},   // cm_joint_avg
    {"GLCM_JE", 3.9879003821350416},   // cm_joint_entr
    {"GLCM_JE_AVE", 3.9879003821350416},   // cm_joint_entr
    {"GLCM_ENTROPY", 3.9879003821350416},   // cm_joint_entr
    {"GLCM_ENTROPY_AVE", 3.9879003821350416},   // cm_joint_entr
    {"GLCM_JMAX", 0.069196428571428575},   // cm_joint_max
    {"GLCM_JMAX_AVE", 0.069196428571428575},   // cm_joint_max
    {"GLCM_JVAR", 5.2795710120783008},   // cm_joint_var
    {"GLCM_JVAR_AVE", 5.2795710120783008},   // cm_joint_var
    {"GLCM_VARIANCE", 5.2795710120783008},   // cm_joint_var
    {"GLCM_VARIANCE_AVE", 5.2795710120783008},   // cm_joint_var
    {"GLCM_SUMAVERAGE", 9.0204081632653068},   // cm_sum_avg
    {"GLCM_SUMAVERAGE_AVE", 9.0204081632653068},   // cm_sum_avg
    {"GLCM_SUMENTROPY", 2.5596639927154166},   // cm_sum_entr
    {"GLCM_SUMENTROPY_AVE", 2.5596639927154166},   // cm_sum_entr
    {"GLCM_SUMVARIANCE", 10.898896293211161},   // cm_sum_var
    {"GLCM_SUMVARIANCE_AVE", 10.898896293211161},   // cm_sum_var
};

// Same split as the PyRadiomics goldens and for the same reason - Nyxus sums logarithms through
// fast_log10 with an EPSILON guard. Measured worst case DIFENTRO 2.8e-3; everything else agrees
// with MIRP to 1e-13 or better.
static double glcm_2d_mirp_frac_tolerance (const std::string& feature_name)
{
    static const std::unordered_set<std::string> log_based {
        "GLCM_DIFENTRO", "GLCM_ENTROPY", "GLCM_INFOMEAS1", "GLCM_INFOMEAS2", "GLCM_JE",
        "GLCM_SUMENTROPY", "GLCM_DIFENTRO_AVE", "GLCM_ENTROPY_AVE", "GLCM_INFOMEAS1_AVE",
        "GLCM_INFOMEAS2_AVE", "GLCM_JE_AVE", "GLCM_SUMENTROPY_AVE"
    };
    return log_based.count (feature_name) ? 200. : 1.e9;
}

void assert_2d_glcm_feature_mirp (const std::string& feature_name, double golden)
{
    double value = 0;
    ASSERT_TRUE (calc_2d_glcm_dense_feature (feature_name, value)) << feature_name;
    ASSERT_TRUE (Nyxus::agrees_gt (value, golden, glcm_2d_mirp_frac_tolerance (feature_name)))
        << feature_name;
}

void test_2d_glcm_family_mirp()
{
    // Every 2D GLCM feature the build exposes has to be pinned here. Without this a feature added
    // to the family later is vetted by nothing while this test still passes over the table it has.
    for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
        if (name.rfind ("GLCM_", 0) == 0)
            ASSERT_TRUE (glcm_2d_mirp_ref_vals.count (name)) << name << " has no mirp golden";

    for (const auto& [feature_name, golden] : glcm_2d_mirp_ref_vals)
        assert_2d_glcm_feature_mirp (feature_name, golden);
}
