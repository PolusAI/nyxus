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


// ---------------------------------------------------------------------------------------------
// Second fixture: grey levels that leave bins empty
//
// A 12x12 box cut from inside one segmented structure of a contrast CT slice, Hounsfield units
// shifted so the box minimum is 1, the whole box one ROI. At PyRadiomics binCount 20
// (GLCM_GREYDEPTH=-20) its pixels occupy levels {1..8, 11, 13, 20} only: interior bins are empty
// and a few pixels sit alone in the top bin, which is what CT ROIs look like at this binning. Nyxus
// then builds the matrix over the levels present, so matrix row x stands for level I[x] and the two
// differ. An implementation that reads x where it means I[x] - in the sum and difference
// distributions, the homogeneity and inverse-difference weights, N_g, or the joint variance - lands
// far from these goldens; on the dense phantom above x + 1 == I[x] and it would not.
// Goldens: tests/vetting/oracles/gen_glcm_gap_levels_pyradiomics.py, which reads this array.
static const unsigned int glcm_2d_gap_levels_side = 12;
static const unsigned int glcm_2d_gap_levels_pixels[] =
{
	  76,   77,  120,  257,  131,  138,  116,   85,  142,  163,  162,  164,
	  59,    1,   50,  194,  119,  160,  140,   94,   91,  157,  155,   61,
	  93,   82,  113,   84,   77,  127,   79,   89,   85,  141,  214,  211,
	 218,  183,  129,   45,    9,  160,  255,  190,  207,  127,  115,   51,
	  94,  108,   86,  144,  105,   69,  117,  152,  207,  168,  184,  205,
	  69,   61,   84,  142,  154,  143,   91,  122,  158,  127,  132,  138,
	 112,   74,  100,   84,  154,  136,  153,   65,  112,  103,  191,  163,
	 104,   83,   57,   83,   74,  115,  154,   88,   87,   60,  121,  154,
	 135,   93,  132,  161,  121,  123,  158,  141,  163,  113,  243,  261,
	 134,  177,  161,  138,  159,  157,  186,  192,  173,  134,  235,  460,
	 361,  314,  200,  175,  147,  213,  266,  197,  275,  391,  544,  891,
	 434,  391,  408,  381,  341,  322,  319,  367,  448,  623, 1045, 1645,
};

// pyradiomics v3.0.1, recipe glcm.pyradiomics_gap_levels_bincount20, 4 directions
static const ref_vals_map<double> glcm_2d_pyradiomics_gap_levels_ref_vals
{
    {"GLCM_ACOR", 8.2956267217630852},
    {"GLCM_ACOR_AVE", 8.2956267217630852},
    {"GLCM_ASM", 0.16575681211058735},
    {"GLCM_ASM_AVE", 0.16575681211058735},
    {"GLCM_CLUPROM", 3237.0785824445866},
    {"GLCM_CLUPROM_AVE", 3237.0785824445866},
    {"GLCM_CLUSHADE", 147.87658800367043},
    {"GLCM_CLUSHADE_AVE", 147.87658800367043},
    {"GLCM_CLUTEND", 10.372116165410681},
    {"GLCM_CLUTEND_AVE", 10.372116165410681},
    {"GLCM_CONTRAST", 2.1391184573002757},
    {"GLCM_CONTRAST_AVE", 2.1391184573002757},
    {"GLCM_CORRELATION", 0.65685092812216161},
    {"GLCM_CORRELATION_AVE", 0.65685092812216161},
    {"GLCM_DIFAVE", 0.8150826446280991},
    {"GLCM_DIFAVE_AVE", 0.8150826446280991},
    {"GLCM_DIFENTRO", 1.6181356521862624},
    {"GLCM_DIFENTRO_AVE", 1.6181356521862624},
    {"GLCM_DIFVAR", 1.4548276719106921},
    {"GLCM_DIFVAR_AVE", 1.4548276719106921},
    {"GLCM_DIS", 0.8150826446280991},
    {"GLCM_DIS_AVE", 0.8150826446280991},
    {"GLCM_ENERGY", 0.16575681211058735},
    {"GLCM_ENERGY_AVE", 0.16575681211058735},
    {"GLCM_ENTROPY", 3.498688215631049},
    {"GLCM_ENTROPY_AVE", 3.498688215631049},
    {"GLCM_HOM1", 0.69574212088416632},
    {"GLCM_HOM1_AVE", 0.69574212088416632},
    {"GLCM_HOM2", 0.6777890762656642},
    {"GLCM_ID", 0.69574212088416632},
    {"GLCM_IDM", 0.6777890762656642},
    {"GLCM_IDMN", 0.99515942458445095},
    {"GLCM_IDMN_AVE", 0.99515942458445095},
    {"GLCM_IDM_AVE", 0.6777890762656642},
    {"GLCM_IDN", 0.96353515722255922},
    {"GLCM_IDN_AVE", 0.96353515722255922},
    {"GLCM_ID_AVE", 0.69574212088416632},
    {"GLCM_INFOMEAS1", -0.23431479796713328},
    {"GLCM_INFOMEAS1_AVE", -0.23431479796713328},
    {"GLCM_INFOMEAS2", 0.76628824994865963},
    {"GLCM_INFOMEAS2_AVE", 0.76628824994865963},
    {"GLCM_IV", 0.44134759878405111},
    {"GLCM_IV_AVE", 0.44134759878405111},
    {"GLCM_JAVE", 2.4965564738292012},
    {"GLCM_JAVE_AVE", 2.4965564738292012},
    {"GLCM_JE", 3.498688215631049},
    {"GLCM_JE_AVE", 3.498688215631049},
    {"GLCM_JMAX", 0.35519972451790638},
    {"GLCM_JMAX_AVE", 0.35519972451790638},
    {"GLCM_JVAR", 3.1278086556777396},
    {"GLCM_JVAR_AVE", 3.1278086556777396},
    {"GLCM_SUMAVERAGE", 4.9931129476584033},
    {"GLCM_SUMAVERAGE_AVE", 4.9931129476584033},
    {"GLCM_SUMENTROPY", 2.6604353207607674},
    {"GLCM_SUMENTROPY_AVE", 2.6604353207607674},
    {"GLCM_SUMVARIANCE", 10.372116165410681},
    {"GLCM_SUMVARIANCE_AVE", 10.372116165410681},
    {"GLCM_VARIANCE", 3.1278086556777396},
    {"GLCM_VARIANCE_AVE", 3.1278086556777396},
};

void test_2d_glcm_gap_levels_pyradiomics()
{
    // Every 2D GLCM feature the build exposes has to be pinned here, as on the dense phantom.
    for (const auto& [name, code] : Nyxus::UserFacingFeatureNames)
        if (name.rfind ("GLCM_", 0) == 0)
            ASSERT_TRUE (glcm_2d_pyradiomics_gap_levels_ref_vals.count (name)) << name << " has no pyradiomics golden";

    const size_t side = glcm_2d_gap_levels_side;
    std::vector<NyxusPixel> intensity, mask;
    for (size_t y = 0; y < side; y++)
        for (size_t x = 0; x < side; x++)
        {
            intensity.push_back (NyxusPixel {x, y, glcm_2d_gap_levels_pixels [y * side + x]});
            mask.push_back (NyxusPixel {x, y, 1u});
        }

    // recipe glcm.pyradiomics_gap_levels_bincount20: the non-IBSI path at PyRadiomics binCount 20
    Fsettings s = make_2d_glcm_dense_settings();
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;   // negative: PyRadiomics binCount binning

    // the angle set is a static; restore it however this test exits
    struct Restore { std::vector<int> a; ~Restore() { GLCMFeature::angles = a; } } restore { GLCMFeature::angles };
    GLCMFeature::angles = {0, 45, 90, 135};
    LR roidata;
    GLCMFeature f;
    Nyxus::load_masked_test_roi_data (roidata, intensity.data(), mask.data(), intensity.size());
    f.calculate (roidata, s);
    roidata.initialize_fvals();
    f.save_value (roidata.fvals);

    for (const auto& [feature_name, golden] : glcm_2d_pyradiomics_gap_levels_ref_vals)
    {
        auto it_code = Nyxus::UserFacingFeatureNames.find (feature_name);
        ASSERT_TRUE (it_code != Nyxus::UserFacingFeatureNames.end()) << feature_name;
        const auto& fv = roidata.fvals [(int) it_code->second];
        const double value = is_2d_glcm_ave_feature (feature_name)
            ? fv[0]
            : (fv[0] + fv[1] + fv[2] + fv[3]) / 4.;
        EXPECT_TRUE (Nyxus::agrees_gt (value, golden, glcm_2d_pyradiomics_frac_tolerance (feature_name)))
            << feature_name << ": nyxus " << value << ", pyradiomics " << golden;
    }
}
