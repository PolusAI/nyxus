#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "../src/nyx/featureset.h"   // UserFacingFeatureNames
#include "test_2d_glrlm_common.h"    // the phantom fixture and calc_2d_glrlm_phantom_feature
#include "test_main_nyxus.h"         // agrees_gt
#include "test_ref_vals.h"           // ref_vals_map

// mirp 2.6.0, recipe glrlm.ibsi_ng128 (by_slice + 2d_average + no discretisation), IBSI digital phantom
// Generated offline by tests/vetting/oracles/gen_glrlm_mirp.py (SPEC 6.4) on the IBSI digital
// phantom as tests/test_data.h stores it, at recipe glrlm.ibsi_ng128.
// MIRP side: by_slice=True, base_discretisation_method="none", glrlm_spatial_method=
// "2d_average". MIRP is the IBSI reference implementation, so it is the second opinion
// that is not PyRadiomics-shaped.
//
// Each quantity is pinned twice: once for the per-direction feature, once for its _AVE twin. The
// tool reports one value over the 4 in-slice directions, which is what _AVE holds and what
// averaging the 4 directional values of the base feature produces.
static const ref_vals_map<double> glrlm_2d_mirp_ref_vals
{
    {"GLRLM_GLV", 3.3530283911078764},   // rlm_gl_var
    {"GLRLM_GLV_AVE", 3.3530283911078764},   // rlm_gl_var
    {"GLRLM_GLN", 5.1970624045072569},   // rlm_glnu
    {"GLRLM_GLN_AVE", 5.1970624045072569},   // rlm_glnu
    {"GLRLM_GLNN", 0.45972933667496529},   // rlm_glnu_norm
    {"GLRLM_GLNN_AVE", 0.45972933667496529},   // rlm_glnu_norm
    {"GLRLM_HGLRE", 9.8242744540998217},   // rlm_hgre
    {"GLRLM_HGLRE_AVE", 9.8242744540998217},   // rlm_hgre
    {"GLRLM_LGLRE", 0.60435790773853726},   // rlm_lgre
    {"GLRLM_LGLRE_AVE", 0.60435790773853726},   // rlm_lgre
    {"GLRLM_LRE", 3.7783842230073845},   // rlm_lre
    {"GLRLM_LRE_AVE", 3.7783842230073845},   // rlm_lre
    {"GLRLM_LRHGLE", 17.387027111981155},   // rlm_lrhge
    {"GLRLM_LRHGLE_AVE", 17.387027111981155},   // rlm_lrhge
    {"GLRLM_LRLGLE", 3.1444829924551891},   // rlm_lrlge
    {"GLRLM_LRLGLE_AVE", 3.1444829924551891},   // rlm_lrlge
    {"GLRLM_RP", 0.62709945820433433},   // rlm_r_perc
    {"GLRLM_RP_AVE", 0.62709945820433433},   // rlm_r_perc
    {"GLRLM_RE", 2.1695507975592458},   // rlm_rl_entr
    {"GLRLM_RE_AVE", 2.1695507975592458},   // rlm_rl_entr
    {"GLRLM_RV", 0.76147506092087514},   // rlm_rl_var
    {"GLRLM_RV_AVE", 0.76147506092087514},   // rlm_rl_var
    {"GLRLM_RLN", 6.1228637477718362},   // rlm_rlnu
    {"GLRLM_RLN_AVE", 6.1228637477718362},   // rlm_rlnu
    {"GLRLM_RLNN", 0.49174138091394598},   // rlm_rlnu_norm
    {"GLRLM_RLNN_AVE", 0.49174138091394598},   // rlm_rlnu_norm
    {"GLRLM_SRE", 0.6406243545397956},   // rlm_sre
    {"GLRLM_SRE_AVE", 0.6406243545397956},   // rlm_sre
    {"GLRLM_SRHGLE", 8.5731367652806085},   // rlm_srhge
    {"GLRLM_SRHGLE_AVE", 8.5731367652806085},   // rlm_srhge
    {"GLRLM_SRLGLE", 0.29396584677839466},   // rlm_srlge
    {"GLRLM_SRLGLE_AVE", 0.29396584677839466},   // rlm_srlge
};

// Nyxus reproduces this tool to double precision on 15 of the 16 quantities. The exception is run
// entropy, the family's only sum over logarithms, which Nyxus evaluates through fast_log10 with an
// EPSILON guard: measured 1.1e-3 away, so it is held to 5e-3 and everything else to 1e-9. See
// tests/vetting/audit/glrlm_2d_mirp_vetting_report.md.
static double glrlm_2d_mirp_frac_tolerance (const std::string& feature_name)
{
    static const std::unordered_set<std::string> log_based { "GLRLM_RE", "GLRLM_RE_AVE" };
    return log_based.count (feature_name) ? 200. : 1.e9;
}

void assert_2d_glrlm_feature_mirp (const std::string& feature_name, double golden)
{
    double value = 0;
    ASSERT_TRUE (calc_2d_glrlm_phantom_feature (feature_name, value)) << feature_name;
    ASSERT_TRUE (Nyxus::agrees_gt (value, golden, glrlm_2d_mirp_frac_tolerance (feature_name)))
        << feature_name;
}

void test_2d_glrlm_family_mirp()
{
    // Every 2D GLRLM feature the build exposes has to be pinned here. Without this a feature added
    // to the family later is vetted by nothing while this test still passes over the table it has.
    for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
        if (name.rfind ("GLRLM_", 0) == 0)
            ASSERT_TRUE (glrlm_2d_mirp_ref_vals.count (name)) << name << " has no mirp golden";

    for (const auto& [feature_name, golden] : glrlm_2d_mirp_ref_vals)
        assert_2d_glrlm_feature_mirp (feature_name, golden);
}
