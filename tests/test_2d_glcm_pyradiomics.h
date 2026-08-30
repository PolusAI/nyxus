#pragma once

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "../src/nyx/featureset.h"   // UserFacingFeatureNames
#include "test_2d_glcm_common.h"     // the dense phantom and calc_2d_glcm_dense_feature
#include "test_main_nyxus.h"         // agrees_gt
#include "test_ref_vals.h"           // ref_vals_map

// pyradiomics v3.0.1, recipe glcm.ibsi_identity, dense 8x8 phantom
// Generated offline by tests/vetting/oracles/gen_glcm_pyradiomics.py (SPEC 6.4) on the dense
// phantom test_2d_glcm_common.h builds. Recipe glcm.ibsi_identity on the Nyxus side; PyRadiomics
// run at the settings that make it the same configuration: symmetricalGLCM=True, binWidth=1
// (identity on this integer image, so neither tool discretises), distances=[1], force2D=True,
// force2Ddimension=0, weightingNorm=None, label=1. Not glcm.pyradiomics_symmetric - that recipe is
// the non-IBSI path at a fixed bin count, where the re-binning moves the absolute grey levels and
// the level-dependent features stop being comparable.
//
// Nyxus keeps several names for one PyRadiomics quantity, and each is pinned here rather than
// asserted through the other: ENERGY and ASM are JointEnergy, HOM1 is Id, HOM2 is Idm, VARIANCE
// and JVAR are the joint variance PyRadiomics calls SumSquares. PyRadiomics 3.x dropped
// Dissimilarity and SumVariance as duplicates of DifferenceAverage and ClusterTendency, which is
// where the goldens for GLCM_DIS and GLCM_SUMVARIANCE come from; MIRP reports those two as
// quantities of their own and test_2d_glcm_mirp.h pins them there.
static const ref_vals_map<double> glcm_2d_pyradiomics_ref_vals
{
    {"GLCM_ACOR", 20.512755102040817},   // Autocorrelation
    {"GLCM_ACOR_AVE", 20.512755102040817},   // Autocorrelation
    {"GLCM_CLUPROM", 263.28733919002576},   // ClusterProminence
    {"GLCM_CLUPROM_AVE", 263.28733919002576},   // ClusterProminence
    {"GLCM_CLUSHADE", 0.017543710528812317},   // ClusterShade
    {"GLCM_CLUSHADE_AVE", 0.017543710528812317},   // ClusterShade
    {"GLCM_CLUTEND", 10.898896293211161},   // ClusterTendency
    {"GLCM_CLUTEND_AVE", 10.898896293211161},   // ClusterTendency
    {"GLCM_SUMVARIANCE", 10.898896293211161},   // ClusterTendency
    {"GLCM_SUMVARIANCE_AVE", 10.898896293211161},   // ClusterTendency
    {"GLCM_CONTRAST", 10.219387755102041},   // Contrast
    {"GLCM_CONTRAST_AVE", 10.219387755102041},   // Contrast
    {"GLCM_CORRELATION", 0.032164714988198187},   // Correlation
    {"GLCM_CORRELATION_AVE", 0.032164714988198187},   // Correlation
    {"GLCM_DIFAVE", 2.5586734693877551},   // DifferenceAverage
    {"GLCM_DIFAVE_AVE", 2.5586734693877551},   // DifferenceAverage
    {"GLCM_DIS", 2.5586734693877551},   // DifferenceAverage
    {"GLCM_DIS_AVE", 2.5586734693877551},   // DifferenceAverage
    {"GLCM_DIFENTRO", 0.710700360776093},   // DifferenceEntropy
    {"GLCM_DIFENTRO_AVE", 0.710700360776093},   // DifferenceEntropy
    {"GLCM_DIFVAR", 2.9454133694294047},   // DifferenceVariance
    {"GLCM_DIFVAR_AVE", 2.9454133694294047},   // DifferenceVariance
    {"GLCM_ID", 0.35283801020408156},   // Id
    {"GLCM_ID_AVE", 0.35283801020408156},   // Id
    {"GLCM_HOM1", 0.35283801020408156},   // Id
    {"GLCM_HOM1_AVE", 0.35283801020408156},   // Id
    {"GLCM_IDM", 0.27853769782341203},   // Idm
    {"GLCM_IDM_AVE", 0.27853769782341203},   // Idm
    {"GLCM_HOM2", 0.27853769782341203},   // Idm
    {"GLCM_IDMN", 0.88734163285067313},   // Idmn
    {"GLCM_IDMN_AVE", 0.88734163285067313},   // Idmn
    {"GLCM_IDN", 0.77947925090782222},   // Idn
    {"GLCM_IDN_AVE", 0.77947925090782222},   // Idn
    {"GLCM_INFOMEAS1", -0.67050298521074692},   // Imc1
    {"GLCM_INFOMEAS1_AVE", -0.67050298521074692},   // Imc1
    {"GLCM_INFOMEAS2", 0.99100399180169929},   // Imc2
    {"GLCM_INFOMEAS2_AVE", 0.99100399180169929},   // Imc2
    {"GLCM_IV", 0.50863378684807248},   // InverseVariance
    {"GLCM_IV_AVE", 0.50863378684807248},   // InverseVariance
    {"GLCM_JAVE", 4.5102040816326525},   // JointAverage
    {"GLCM_JAVE_AVE", 4.5102040816326525},   // JointAverage
    {"GLCM_ASM", 0.063534725114535603},   // JointEnergy
    {"GLCM_ASM_AVE", 0.063534725114535603},   // JointEnergy
    {"GLCM_ENERGY", 0.063534725114535603},   // JointEnergy
    {"GLCM_ENERGY_AVE", 0.063534725114535603},   // JointEnergy
    {"GLCM_JE", 3.9879003821350367},   // JointEntropy
    {"GLCM_JE_AVE", 3.9879003821350367},   // JointEntropy
    {"GLCM_ENTROPY", 3.9879003821350367},   // JointEntropy
    {"GLCM_ENTROPY_AVE", 3.9879003821350367},   // JointEntropy
    {"GLCM_JMAX", 0.069196428571428575},   // MaximumProbability
    {"GLCM_JMAX_AVE", 0.069196428571428575},   // MaximumProbability
    {"GLCM_SUMAVERAGE", 9.020408163265305},   // SumAverage
    {"GLCM_SUMAVERAGE_AVE", 9.020408163265305},   // SumAverage
    {"GLCM_SUMENTROPY", 2.5596639927154143},   // SumEntropy
    {"GLCM_SUMENTROPY_AVE", 2.5596639927154143},   // SumEntropy
    {"GLCM_JVAR", 5.2795710120783008},   // SumSquares
    {"GLCM_JVAR_AVE", 5.2795710120783008},   // SumSquares
    {"GLCM_VARIANCE", 5.2795710120783008},   // SumSquares
    {"GLCM_VARIANCE_AVE", 5.2795710120783008},   // SumSquares
};

// Nyxus reproduces PyRadiomics to double precision on this fixture except in the four features
// whose sums are over logarithms: it computes those through fast_log10 with an EPSILON guard, so
// they land near the reference rather than on it. Measured worst case DIFENTRO 2.8e-3, then
// INFOMEAS1 9.6e-4, SUMENTROPY 7.6e-4, JE/ENTROPY 5.1e-4, INFOMEAS2 3.5e-5 - so 5e-3 keeps the
// band under twice the largest real deviation, while everything else is held to 1e-9, ~500x
// tighter than the 1.9e-12 worst case among the exact features.
static double glcm_2d_pyradiomics_frac_tolerance (const std::string& feature_name)
{
    static const std::unordered_set<std::string> log_based {
        "GLCM_DIFENTRO", "GLCM_ENTROPY", "GLCM_INFOMEAS1", "GLCM_INFOMEAS2", "GLCM_JE",
        "GLCM_SUMENTROPY", "GLCM_DIFENTRO_AVE", "GLCM_ENTROPY_AVE", "GLCM_INFOMEAS1_AVE",
        "GLCM_INFOMEAS2_AVE", "GLCM_JE_AVE", "GLCM_SUMENTROPY_AVE"
    };
    return log_based.count (feature_name) ? 200. : 1.e9;
}

void assert_2d_glcm_feature_pyradiomics (const std::string& feature_name, double golden)
{
    double value = 0;
    ASSERT_TRUE (calc_2d_glcm_dense_feature (feature_name, value)) << feature_name;
    ASSERT_TRUE (Nyxus::agrees_gt (value, golden, glcm_2d_pyradiomics_frac_tolerance (feature_name)))
        << feature_name;
}

void test_2d_glcm_family_pyradiomics()
{
    // Every 2D GLCM feature the build exposes has to be pinned here. Without this a feature added
    // to the family later is vetted by nothing while this test still passes over the table it has.
    for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
        if (name.rfind ("GLCM_", 0) == 0)
            ASSERT_TRUE (glcm_2d_pyradiomics_ref_vals.count (name)) << name << " has no pyradiomics golden";

    for (const auto& [feature_name, golden] : glcm_2d_pyradiomics_ref_vals)
        assert_2d_glcm_feature_pyradiomics (feature_name, golden);
}
