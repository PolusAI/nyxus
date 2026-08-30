#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "../src/nyx/featureset.h"   // UserFacingFeatureNames
#include "test_2d_glrlm_common.h"    // the phantom fixture and calc_2d_glrlm_phantom_feature
#include "test_main_nyxus.h"         // agrees_gt
#include "test_ref_vals.h"           // ref_vals_map

// pyradiomics v3.0.1, recipe glrlm.ibsi_ng128, IBSI digital phantom (4 slices, averaged)
// Generated offline by tests/vetting/oracles/gen_glrlm_pyradiomics.py (SPEC 6.4) on the IBSI digital
// phantom as tests/test_data.h stores it, at recipe glrlm.ibsi_ng128.
// PyRadiomics side: binWidth=1 (identity on this integer phantom, so neither
// tool discretises), distances=[1], force2D=True, force2Ddimension=0,
// weightingNorm=None, label=1; one value per feature over the 4 directions.
//
// Each quantity is pinned twice: once for the per-direction feature, once for its _AVE twin. The
// tool reports one value over the 4 in-slice directions, which is what _AVE holds and what
// averaging the 4 directional values of the base feature produces.
static const ref_vals_map<double> glrlm_2d_pyradiomics_ref_vals
{
    {"GLRLM_GLN", 5.1970624045072578},   // GrayLevelNonUniformity
    {"GLRLM_GLN_AVE", 5.1970624045072578},   // GrayLevelNonUniformity
    {"GLRLM_GLNN", 0.45972933667496541},   // GrayLevelNonUniformityNormalized
    {"GLRLM_GLNN_AVE", 0.45972933667496541},   // GrayLevelNonUniformityNormalized
    {"GLRLM_GLV", 3.353028391107876},   // GrayLevelVariance
    {"GLRLM_GLV_AVE", 3.353028391107876},   // GrayLevelVariance
    {"GLRLM_HGLRE", 9.8242744540998217},   // HighGrayLevelRunEmphasis
    {"GLRLM_HGLRE_AVE", 9.8242744540998217},   // HighGrayLevelRunEmphasis
    {"GLRLM_LRE", 3.7783842230073845},   // LongRunEmphasis
    {"GLRLM_LRE_AVE", 3.7783842230073845},   // LongRunEmphasis
    {"GLRLM_LRHGLE", 17.387027111981155},   // LongRunHighGrayLevelEmphasis
    {"GLRLM_LRHGLE_AVE", 17.387027111981155},   // LongRunHighGrayLevelEmphasis
    {"GLRLM_LRLGLE", 3.1444829924551891},   // LongRunLowGrayLevelEmphasis
    {"GLRLM_LRLGLE_AVE", 3.1444829924551891},   // LongRunLowGrayLevelEmphasis
    {"GLRLM_LGLRE", 0.60435790773853737},   // LowGrayLevelRunEmphasis
    {"GLRLM_LGLRE_AVE", 0.60435790773853737},   // LowGrayLevelRunEmphasis
    {"GLRLM_RE", 2.1695507975592441},   // RunEntropy
    {"GLRLM_RE_AVE", 2.1695507975592441},   // RunEntropy
    {"GLRLM_RLN", 6.1228637477718353},   // RunLengthNonUniformity
    {"GLRLM_RLN_AVE", 6.1228637477718353},   // RunLengthNonUniformity
    {"GLRLM_RLNN", 0.49174138091394598},   // RunLengthNonUniformityNormalized
    {"GLRLM_RLNN_AVE", 0.49174138091394598},   // RunLengthNonUniformityNormalized
    {"GLRLM_RP", 0.62709945820433444},   // RunPercentage
    {"GLRLM_RP_AVE", 0.62709945820433444},   // RunPercentage
    {"GLRLM_RV", 0.76147506092087514},   // RunVariance
    {"GLRLM_RV_AVE", 0.76147506092087514},   // RunVariance
    {"GLRLM_SRE", 0.6406243545397956},   // ShortRunEmphasis
    {"GLRLM_SRE_AVE", 0.6406243545397956},   // ShortRunEmphasis
    {"GLRLM_SRHGLE", 8.5731367652806068},   // ShortRunHighGrayLevelEmphasis
    {"GLRLM_SRHGLE_AVE", 8.5731367652806068},   // ShortRunHighGrayLevelEmphasis
    {"GLRLM_SRLGLE", 0.29396584677839466},   // ShortRunLowGrayLevelEmphasis
    {"GLRLM_SRLGLE_AVE", 0.29396584677839466},   // ShortRunLowGrayLevelEmphasis
};

// Nyxus reproduces this tool to double precision on 15 of the 16 quantities. The exception is run
// entropy, the family's only sum over logarithms, which Nyxus evaluates through fast_log10 with an
// EPSILON guard: measured 1.1e-3 away, so it is held to 5e-3 and everything else to 1e-9. See
// tests/vetting/audit/glrlm_2d_pyradiomics_vetting_report.md.
static double glrlm_2d_pyradiomics_frac_tolerance (const std::string& feature_name)
{
    static const std::unordered_set<std::string> log_based { "GLRLM_RE", "GLRLM_RE_AVE" };
    return log_based.count (feature_name) ? 200. : 1.e9;
}

void assert_2d_glrlm_feature_pyradiomics (const std::string& feature_name, double golden)
{
    double value = 0;
    ASSERT_TRUE (calc_2d_glrlm_phantom_feature (feature_name, value)) << feature_name;
    ASSERT_TRUE (Nyxus::agrees_gt (value, golden, glrlm_2d_pyradiomics_frac_tolerance (feature_name)))
        << feature_name;
}

void test_2d_glrlm_family_pyradiomics()
{
    // Every 2D GLRLM feature the build exposes has to be pinned here. Without this a feature added
    // to the family later is vetted by nothing while this test still passes over the table it has.
    for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
        if (name.rfind ("GLRLM_", 0) == 0)
            ASSERT_TRUE (glrlm_2d_pyradiomics_ref_vals.count (name)) << name << " has no pyradiomics golden";

    for (const auto& [feature_name, golden] : glrlm_2d_pyradiomics_ref_vals)
        assert_2d_glrlm_feature_pyradiomics (feature_name, golden);
}
