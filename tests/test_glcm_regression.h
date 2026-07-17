#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include <unordered_map> 

// Nyxus-convention GLCM regression snapshot. These values pin current Nyxus output to catch
// drift. Calculated at 100 grey levels, offset 1, via the MATLAB-binning path with
// symmetric_glcm=false, i.e. on an *asymmetric* cooc matrix. That is a configuration choice, NOT
// an inherent limitation: nyxus symmetrizes on the IBSI and radiomics paths (matching PyRadiomics'
// symmetricalGLCM=True). As configured here the matrix increments only (a,b), so the
// convention-sensitive Haralick family is transpose-sensitive and diverges from a symmetric-matrix
// tool, while the symmetrization-invariant keys coincide with one.
// REFRESHED 2026-06 after the GLCM background-pollution fix: the non-IBSI (MATLAB binning) path now
// excludes out-of-ROI background pixels (matching the IBSI path), so the phantom slices z2-z4
// (which contain masked-out pixels) yield corrected snapshot values. CORRELATION/INFOMEAS1 are
// softNAN(=0)-guarded on the degenerate (single-grey-marginal) phantom directions.
//
// ORACLE-VETTED 2026-07 (corrected 2026-07-09): the 10 keys below were run against a third-party
// symmetric-matrix oracle (PyRadiomics v3.0.1) on the *same* per-slice matlab-binned phantom images,
// aggregated the same way (mean over 4 slices x 4 angles). Nine of them depend only on the grey-level
// DIFFERENCE p_{x-y} / |i-j| (CONTRAST/DIFAVE/DIS/DIFENTRO/DIFVAR/ID/HOM1/IDM/IV); SUMENTROPY is the
// dimensionless entropy of the SUM distribution p_{x+y}. Both kinds are invariant to matrix
// symmetrization and to a level RELABELING (origin shift) -- and this phantom's binning relabels
// levels without RESCALING them, so they coincide with the symmetric-matrix oracle -> PyRadiomics
// agrees within the test's 1% tolerance. (Difference-based features are NOT invariant to level
// *scaling*; that only holds here because the binning does not rescale.) These are VETTED against an
// external definition, not merely pinned.
// CORRECTION (2026-07-09, PR #356 review): ACOR/IDN/IDMN/SUMAVERAGE were previously listed here as
// oracle-vetted -- that was WRONG for this ibsi=False / matlab-binning config. They depend on the
// absolute grey-level values / Ng, which matlab binning re-maps, so under this config they diverge
// from PyRadiomics by up to ~43% (ACOR; measured on a dense 8-level phantom: Nyxus ibsi=False ACOR
// 29.25 vs oracle 20.51). They were therefore MOVED to the unvetted snapshot below. They ARE
// genuinely third-party-vetted on the IBSI path (symmetric matrix, identity binning), where Nyxus
// ibsi=True == PyRadiomics exactly (ACOR 20.512755, SUMAVERAGE 9.020408, IDN 0.779479, IDMN
// 0.887342) -- covered by the dense-phantom oracle test in tests/python/test_glcm_oracle.py.
static std::unordered_map<std::string, double> vetted_nyxus_convention_regression_glcm_feature_golden_values
{
    {"GLCM_CONTRAST", 1.4448130208333334e+03},
    {"GLCM_DIFAVE", 23.6493},
    {"GLCM_DIFENTRO", 1.44004},
    {"GLCM_DIFVAR", 801.208},
    {"GLCM_DIS", 23.6493},
    {"GLCM_HOM1", 0.580526},
    {"GLCM_ID", 0.580526},
    {"GLCM_IDM", 0.572168},
    {"GLCM_IV", 0.000206466},
    {"GLCM_SUMENTROPY", 1.61957}
};

// These keys are NOT oracle-matched under this config, for one of two reasons.
// (a) Transpose-sensitive: they depend on individual matrix entries or on the grey-tone marginal
//     means (mu_x/mu_y), which differ between the asymmetric matrix used here and a symmetric one.
//     As configured (symmetric_glcm=false) they diverge from any symmetric oracle by >1% (verified:
//     ASM/ENERGY 3.7%, CLUSHADE 46%, CLUTEND/SUMVARIANCE 3.2%, JE 9.3%, JVAR/VARIANCE ~10%, ...), and
//     ENTROPY/HOM2 are computed from raw counts (un-normalized), so they have no probability-
//     normalized oracle counterpart.
// (b) Absolute-level / Ng-dependent (ACOR/IDN/IDMN/SUMAVERAGE, moved here from the vetted map on
//     2026-07-09 per PR #356 review): matlab binning re-maps the absolute grey levels and Ng, so on
//     this ibsi=False config they diverge from PyRadiomics (up to ~43% for ACOR). They ARE genuinely
//     oracle-vetted on the IBSI path (symmetric matrix, identity binning) -- see the dense-phantom
//     test in tests/python/test_glcm_oracle.py, which pins them tightly against PyRadiomics.
// All of the above remain UNVETTED snapshots on this config; the transpose-sensitive group could be
// vetted by rerunning with symmetric_glcm=true (or the radiomics/IBSI path). (Note: CLUTEND/
// SUMVARIANCE are NOT symmetrization-invariant in Nyxus because they use the single row-marginal mean
// by_row_mean; the earlier comment listing them as invariant was inaccurate.)
static std::unordered_map<std::string, double> unvetted_nyxus_convention_regression_glcm_feature_golden_values
{
    {"GLCM_ACOR", 1437.33},                 // moved from vetted 2026-07-09: absolute-level-dependent, ibsi=False != oracle (IBSI-path vetted)
    {"GLCM_IDMN", 9.0029152005531590e-01},  // moved from vetted 2026-07-09: Ng-dependent, ibsi=False != oracle (IBSI-path vetted)
    {"GLCM_IDN", 8.4432100308124380e-01},   // moved from vetted 2026-07-09: Ng-dependent, ibsi=False != oracle (IBSI-path vetted)
    {"GLCM_SUMAVERAGE", 72.0369},           // moved from vetted 2026-07-09: absolute-level-dependent, ibsi=False != oracle (IBSI-path vetted)
    {"GLCM_ASM", 0.381801},
    {"GLCM_CLUPROM", 6.1972e+06},
    {"GLCM_CLUSHADE", 21905.3},
    {"GLCM_CLUTEND", 1.5639042057291665e+03},
    {"GLCM_CORRELATION", 0.000690135},
    {"GLCM_ENERGY", 0.381801},
    {"GLCM_ENTROPY", 1.87602},   // FIX: was buggy unnormalized -20.1735; post /sum_p fix == GLCM_JE (joint entropy)
    {"GLCM_HOM2", 0.572168},     // FIX: was buggy unnormalized 6.81505; post /sum_p fix == GLCM_IDM (homogeneity in [0,1])
    {"GLCM_INFOMEAS1", -0.184406},
    {"GLCM_INFOMEAS2", 0.495817},
    {"GLCM_JAVE", 35.5215},
    {"GLCM_JE", 1.87602},
    {"GLCM_JMAX", 0.527914},
    {"GLCM_JVAR", 828.383},
    {"GLCM_SUMVARIANCE", 1.5639042057291665e+03},
    {"GLCM_VARIANCE", 674.871}
};

// A GLCM golden value now lives in exactly one of the two snapshots above. Look it up wherever it is.
static double glcm_golden_value(const std::string& golden_key, bool& found)
{
    auto itv = vetted_nyxus_convention_regression_glcm_feature_golden_values.find(golden_key);
    if (itv != vetted_nyxus_convention_regression_glcm_feature_golden_values.end())
    {
        found = true;
        return itv->second;
    }
    auto itu = unvetted_nyxus_convention_regression_glcm_feature_golden_values.find(golden_key);
    if (itu != unvetted_nyxus_convention_regression_glcm_feature_golden_values.end())
    {
        found = true;
        return itu->second;
    }
    found = false;
    return 0.0;
}

static std::string glcm_golden_key(const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    if (feature_name.size() > ave_suffix.size() &&
        feature_name.compare(feature_name.size() - ave_suffix.size(), ave_suffix.size(), ave_suffix) == 0)
        return feature_name.substr(0, feature_name.size() - ave_suffix.size());

    return feature_name;
}

void test_glcm_feature(const Feature2D& feature_, const std::string& feature_name) 
{
    // featue settings for this particular test
    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;   // important
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    //

    // Set feature's state
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 100;   // important
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;   // important
    GLCMFeature::symmetric_glcm = false;
    GLCMFeature::angles = { 0, 45, 90, 135 };

    int feature = int(feature_);
    const std::string golden_key = glcm_golden_key(feature_name);
    // Golden lives in either the vetted or the (remaining) unvetted snapshot; require it in one.
    bool golden_found = false;
    const double golden = glcm_golden_value(golden_key, golden_found);
    ASSERT_TRUE(golden_found);
    const bool is_ave_feature = golden_key != feature_name;

    double total = 0;

    // image 1

     LR roidata;
    GLCMFeature f;   
    load_masked_test_roi_data (roidata, ibsi_phantom_z1_intensity, ibsi_phantom_z1_mask,  sizeof(ibsi_phantom_z1_mask) / sizeof(NyxusPixel));
    ASSERT_NO_THROW(f.calculate(roidata, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);
 
    if (is_ave_feature)
        total += roidata.fvals[feature][0];
    else
    {
        total += roidata.fvals[feature][0];
        total += roidata.fvals[feature][1];
        total += roidata.fvals[feature][2];
        total += roidata.fvals[feature][3];
    }

    // image 2

    LR roidata1;
    GLCMFeature f1;
    load_masked_test_roi_data (roidata1, ibsi_phantom_z2_intensity, ibsi_phantom_z2_mask,  sizeof(ibsi_phantom_z2_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f1.calculate(roidata1, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata1.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f1.save_value(roidata1.fvals);

    if (is_ave_feature)
        total += roidata1.fvals[feature][0];
    else
    {
        total += roidata1.fvals[feature][0];
        total += roidata1.fvals[feature][1];
        total += roidata1.fvals[feature][2];
        total += roidata1.fvals[feature][3];
    }
    
    // image 3

    LR roidata2;
    GLCMFeature f2;
    load_masked_test_roi_data (roidata2, ibsi_phantom_z3_intensity, ibsi_phantom_z3_mask,  sizeof(ibsi_phantom_z3_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f2.calculate(roidata2, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata2.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f2.save_value(roidata2.fvals);

    if (is_ave_feature)
        total += roidata2.fvals[feature][0];
    else
    {
        total += roidata2.fvals[feature][0];
        total += roidata2.fvals[feature][1];
        total += roidata2.fvals[feature][2];
        total += roidata2.fvals[feature][3];
    }
    
    // image 4
    
    LR roidata3;
    GLCMFeature f3;
    load_masked_test_roi_data (roidata3, ibsi_phantom_z4_intensity, ibsi_phantom_z4_mask,  sizeof(ibsi_phantom_z4_intensity) / sizeof(NyxusPixel));

    ASSERT_NO_THROW(f3.calculate(roidata3, s));

    // Initialize per-ROI feature value buffer with zeros
    roidata3.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f3.save_value(roidata3.fvals);

    // Check the feature values vs ground truth
    if (is_ave_feature)
        total += roidata3.fvals[feature][0];
    else
    {
        total += roidata3.fvals[feature][0];
        total += roidata3.fvals[feature][1];
        total += roidata3.fvals[feature][2];
        total += roidata3.fvals[feature][3];
    }

    // Verdict
    const double divisor = is_ave_feature ? 4.0 : 16.0;
    ASSERT_TRUE(agrees_gt(total / divisor, golden, 100.));
}

void test_glcm_ACOR()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ACOR, "GLCM_ACOR");
}

void test_glcm_angular_2d_moment()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ASM, "GLCM_ASM");
}

void test_glcm_CLUPROM()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUPROM, "GLCM_CLUPROM");
}

void test_glcm_CLUSHADE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUSHADE, "GLCM_CLUSHADE");
}

void test_glcm_CLUTEND()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUTEND, "GLCM_CLUTEND");
}

void test_glcm_contrast()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_CONTRAST, "GLCM_CONTRAST");
}

void test_glcm_correlation()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CORRELATION, "GLCM_CORRELATION");
}

void test_glcm_difference_average()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFAVE, "GLCM_DIFAVE");
}

void test_glcm_difference_entropy()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFENTRO, "GLCM_DIFENTRO");
}

void test_glcm_difference_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFVAR, "GLCM_DIFVAR");
}

void test_glcm_DIS()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIS, "GLCM_DIS");
}

void test_glcm_energy()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENERGY, "GLCM_ENERGY");
}

void test_glcm_entropy()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENTROPY, "GLCM_ENTROPY");
}

void test_glcm_hom1()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_HOM1, "GLCM_HOM1");
}

void test_glcm_hom2()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_HOM2, "GLCM_HOM2");
}

void test_glcm_ID()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ID, "GLCM_ID");
}

void test_glcm_IDN()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDN, "GLCM_IDN");
}

void test_glcm_IDM()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDM, "GLCM_IDM");
}

void test_glcm_IDMN()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDMN, "GLCM_IDMN");
}

void test_glcm_infomeas1()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS1, "GLCM_INFOMEAS1");
}

void test_glcm_infomeas2()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS2, "GLCM_INFOMEAS2");
}

void test_glcm_IV()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IV, "GLCM_IV");
}

void test_glcm_JAVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JAVE, "GLCM_JAVE");
}

void test_glcm_JE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JE, "GLCM_JE");
}

void test_glcm_JMAX()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JMAX, "GLCM_JMAX");
}

void test_glcm_JVAR()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JVAR, "GLCM_JVAR");
}

void test_glcm_sum_average()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMAVERAGE, "GLCM_SUMAVERAGE");
}

void test_glcm_sum_entropy()
{
   test_glcm_feature(Nyxus::Feature2D::GLCM_SUMENTROPY, "GLCM_SUMENTROPY");
}

void test_glcm_sum_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMVARIANCE, "GLCM_SUMVARIANCE");
}

void test_glcm_variance()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_VARIANCE, "GLCM_VARIANCE");
}

void test_glcm_ASM_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ASM_AVE, "GLCM_ASM_AVE");
}

void test_glcm_ACOR_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ACOR_AVE, "GLCM_ACOR_AVE");
}

void test_glcm_CLUPROM_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUPROM_AVE, "GLCM_CLUPROM_AVE");
}

void test_glcm_CLUSHADE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUSHADE_AVE, "GLCM_CLUSHADE_AVE");
}

void test_glcm_CLUTEND_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CLUTEND_AVE, "GLCM_CLUTEND_AVE");
}

void test_glcm_CONTRAST_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CONTRAST_AVE, "GLCM_CONTRAST_AVE");
}

void test_glcm_CORRELATION_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_CORRELATION_AVE, "GLCM_CORRELATION_AVE");
}

void test_glcm_DIFAVE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFAVE_AVE, "GLCM_DIFAVE_AVE");
}

void test_glcm_DIFENTRO_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFENTRO_AVE, "GLCM_DIFENTRO_AVE");
}

void test_glcm_DIFVAR_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIFVAR_AVE, "GLCM_DIFVAR_AVE");
}

void test_glcm_DIS_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_DIS_AVE, "GLCM_DIS_AVE");
}

void test_glcm_ENERGY_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENERGY_AVE, "GLCM_ENERGY_AVE");
}

void test_glcm_ENTROPY_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ENTROPY_AVE, "GLCM_ENTROPY_AVE");
}

void test_glcm_HOM1_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_HOM1_AVE, "GLCM_HOM1_AVE");
}

void test_glcm_ID_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_ID_AVE, "GLCM_ID_AVE");
}

void test_glcm_IDN_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDN_AVE, "GLCM_IDN_AVE");
}

void test_glcm_IDM_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDM_AVE, "GLCM_IDM_AVE");
}

void test_glcm_IDMN_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IDMN_AVE, "GLCM_IDMN_AVE");
}

void test_glcm_IV_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_IV_AVE, "GLCM_IV_AVE");
}

void test_glcm_JAVE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JAVE_AVE, "GLCM_JAVE_AVE");
}

void test_glcm_JE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JE_AVE, "GLCM_JE_AVE");
}

void test_glcm_INFOMEAS1_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS1_AVE, "GLCM_INFOMEAS1_AVE");
}

void test_glcm_INFOMEAS2_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_INFOMEAS2_AVE, "GLCM_INFOMEAS2_AVE");
}

void test_glcm_VARIANCE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_VARIANCE_AVE, "GLCM_VARIANCE_AVE");
}

void test_glcm_JMAX_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JMAX_AVE, "GLCM_JMAX_AVE");
}

void test_glcm_JVAR_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_JVAR_AVE, "GLCM_JVAR_AVE");
}

void test_glcm_SUMAVERAGE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMAVERAGE_AVE, "GLCM_SUMAVERAGE_AVE");
}

void test_glcm_SUMENTROPY_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMENTROPY_AVE, "GLCM_SUMENTROPY_AVE");
}

void test_glcm_SUMVARIANCE_AVE()
{
    test_glcm_feature(Nyxus::Feature2D::GLCM_SUMVARIANCE_AVE, "GLCM_SUMVARIANCE_AVE");
}
