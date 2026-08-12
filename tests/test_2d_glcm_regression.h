#pragma once

#include <gtest/gtest.h>

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

#include "test_ref_vals.h"

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
// These are snapshots and establish no vetting (SPEC 1) -- which is why they are one table. The file
// previously split them in two, "vetted convention" and "unvetted convention", by whether a key also
// happened to agree with PyRadiomics under this config. That split drove nothing: both halves fed the
// one lookup below and the one 1% assertion at the bottom of assert_glcm_feature_regression, so it
// was documentation wearing a data structure. It had also gone stale as documentation -- every 2D GLCM
// row in oracle_coverage.csv now reads status=vetted, ACOR/ENTROPY/HOM2/JVAR among them, so a table
// named "unvetted" contradicted the registry. Vetting is a property of a (feature x config x oracle)
// assertion and lives in the registry and the oracle files, not in a boundary between two snapshot
// tables. The per-key notes that split carried are kept below, where they describe a key rather than
// a group.
//
// Which keys coincide with a symmetric-matrix oracle under THIS config, and why, is still worth
// knowing, so it is recorded per key as "sym-invariant":
//   Marked keys depend only on the grey-level DIFFERENCE p_{x-y} / |i-j| (CONTRAST, DIFAVE, DIS,
//   DIFENTRO, DIFVAR, ID, HOM1, IDM, IV) or on the SUM distribution p_{x+y} (SUMENTROPY). Both kinds
//   are invariant to matrix symmetrization and to a level RELABELING, and this phantom's binning
//   relabels levels without RESCALING them, so they land within 1% of PyRadiomics v3.0.1 run on the
//   same per-slice matlab-binned images with the same mean-over-4-slices-x-4-angles aggregation.
//   (They are NOT invariant to level *scaling*; that only holds here because the binning does not
//   rescale.) Unmarked keys are transpose-sensitive -- they read individual matrix entries or the
//   grey-tone marginal means mu_x/mu_y -- and diverge from any symmetric oracle by >1% as configured
//   (measured: ASM/ENERGY 3.7%, CLUSHADE 46%, CLUTEND/SUMVARIANCE 3.2%, JE 9.3%, JVAR/VARIANCE ~10%).
//   CLUTEND/SUMVARIANCE are among them because Nyxus computes them from the single row-marginal mean
//   by_row_mean. Rerunning with symmetric_glcm=true (or the radiomics/IBSI path) is what would let
//   that group be compared to a symmetric oracle at all.
static ref_vals_map<double> glcm_2d_regression_ref_vals
{
    {"GLCM_ACOR", 1437.33},                 // absolute-level-dependent: matlab binning re-maps levels, so ibsi=False diverges from a symmetric oracle by ~43% (2026-07-09, PR #356 review). Vetted instead on the IBSI path -- see below.
    {"GLCM_ASM", 0.381801},
    {"GLCM_CLUPROM", 6.1972e+06},
    {"GLCM_CLUSHADE", 21905.3},
    {"GLCM_CLUTEND", 1.5639042057291665e+03},
    {"GLCM_CONTRAST", 1.4448130208333334e+03},   // sym-invariant
    {"GLCM_CORRELATION", 0.000690135},
    {"GLCM_DIFAVE", 23.6493},                    // sym-invariant
    {"GLCM_DIFENTRO", 1.44004},                  // sym-invariant
    {"GLCM_DIFVAR", 801.208},                    // sym-invariant
    {"GLCM_DIS", 23.6493},                       // sym-invariant
    {"GLCM_ENERGY", 0.381801},
    {"GLCM_ENTROPY", 1.87602},   // was buggy unnormalized -20.1735; post /sum_p fix == GLCM_JE (joint entropy)
    {"GLCM_HOM1", 0.580526},                     // sym-invariant
    {"GLCM_HOM2", 0.572168},     // was buggy unnormalized 6.81505; post /sum_p fix == GLCM_IDM (homogeneity in [0,1])
    {"GLCM_ID", 0.580526},                       // sym-invariant
    {"GLCM_IDM", 0.572168},                      // sym-invariant
    {"GLCM_IDMN", 9.0029152005531590e-01},  // Ng-dependent, same story as ACOR (2026-07-09, PR #356 review)
    {"GLCM_IDN", 8.4432100308124380e-01},   // Ng-dependent, same story as ACOR (2026-07-09, PR #356 review)
    {"GLCM_INFOMEAS1", -0.184406},
    {"GLCM_INFOMEAS2", 0.495817},
    {"GLCM_IV", 0.000206466},                    // sym-invariant
    {"GLCM_JAVE", 35.5215},
    {"GLCM_JE", 1.87602},
    {"GLCM_JMAX", 0.527914},
    {"GLCM_JVAR", 828.383},
    {"GLCM_SUMAVERAGE", 72.0369},           // absolute-level-dependent, same story as ACOR (2026-07-09, PR #356 review)
    {"GLCM_SUMENTROPY", 1.61957},                // sym-invariant
    {"GLCM_SUMVARIANCE", 1.5639042057291665e+03},
    {"GLCM_VARIANCE", 674.871}
};
// ACOR/IDN/IDMN/SUMAVERAGE are genuinely third-party-vetted on the IBSI path (symmetric matrix,
// identity binning), where Nyxus ibsi=True == PyRadiomics exactly (ACOR 20.512755, SUMAVERAGE
// 9.020408, IDN 0.779479, IDMN 0.887342). That evidence lives in the dense-phantom oracle test
// tests/python/test_glcm_oracle.py, not here.

static double glcm_golden_value(const std::string& golden_key, bool& found)
{
    auto it = glcm_2d_regression_ref_vals.find(golden_key);
    found = it != glcm_2d_regression_ref_vals.end();
    return found ? it->second : 0.0;
}

static std::string glcm_golden_key(const std::string& feature_name)
{
    static const std::string ave_suffix = "_AVE";
    if (feature_name.size() > ave_suffix.size() &&
        feature_name.compare(feature_name.size() - ave_suffix.size(), ave_suffix.size(), ave_suffix) == 0)
        return feature_name.substr(0, feature_name.size() - ave_suffix.size());

    return feature_name;
}

void assert_glcm_feature_regression(const Feature2D& feature_, const std::string& feature_name) 
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
    // Every asserted feature must be pinned; an unpinned key is a silently unguarded assertion.
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

void test_2d_glcm_acor_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ACOR, "GLCM_ACOR");
}

void test_2d_glcm_cluprom_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CLUPROM, "GLCM_CLUPROM");
}

void test_2d_glcm_clushade_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CLUSHADE, "GLCM_CLUSHADE");
}

void test_2d_glcm_clutend_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CLUTEND, "GLCM_CLUTEND");
}

void test_2d_glcm_difference_average_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIFAVE, "GLCM_DIFAVE");
}

void test_2d_glcm_difference_entropy_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIFENTRO, "GLCM_DIFENTRO");
}

void test_2d_glcm_difference_variance_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIFVAR, "GLCM_DIFVAR");
}

void test_2d_glcm_dis_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIS, "GLCM_DIS");
}

void test_2d_glcm_entropy_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ENTROPY, "GLCM_ENTROPY");
}

void test_2d_glcm_hom2_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_HOM2, "GLCM_HOM2");
}

void test_2d_glcm_id_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ID, "GLCM_ID");
}

void test_2d_glcm_idn_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IDN, "GLCM_IDN");
}

void test_2d_glcm_idm_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IDM, "GLCM_IDM");
}

void test_2d_glcm_idmn_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IDMN, "GLCM_IDMN");
}

void test_2d_glcm_infomeas1_regression()
{
   assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_INFOMEAS1, "GLCM_INFOMEAS1");
}

void test_2d_glcm_infomeas2_regression()
{
   assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_INFOMEAS2, "GLCM_INFOMEAS2");
}

void test_2d_glcm_iv_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IV, "GLCM_IV");
}

void test_2d_glcm_jave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JAVE, "GLCM_JAVE");
}

void test_2d_glcm_je_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JE, "GLCM_JE");
}

void test_2d_glcm_jmax_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JMAX, "GLCM_JMAX");
}

void test_2d_glcm_jvar_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JVAR, "GLCM_JVAR");
}

void test_2d_glcm_sum_average_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_SUMAVERAGE, "GLCM_SUMAVERAGE");
}

void test_2d_glcm_sum_entropy_regression()
{
   assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_SUMENTROPY, "GLCM_SUMENTROPY");
}

void test_2d_glcm_sum_variance_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_SUMVARIANCE, "GLCM_SUMVARIANCE");
}

void test_2d_glcm_variance_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_VARIANCE, "GLCM_VARIANCE");
}

void test_2d_glcm_acor_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ACOR_AVE, "GLCM_ACOR_AVE");
}

void test_2d_glcm_cluprom_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CLUPROM_AVE, "GLCM_CLUPROM_AVE");
}

void test_2d_glcm_clushade_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CLUSHADE_AVE, "GLCM_CLUSHADE_AVE");
}

void test_2d_glcm_clutend_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CLUTEND_AVE, "GLCM_CLUTEND_AVE");
}

void test_2d_glcm_difference_average_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIFAVE_AVE, "GLCM_DIFAVE_AVE");
}

void test_2d_glcm_difference_entropy_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIFENTRO_AVE, "GLCM_DIFENTRO_AVE");
}

void test_2d_glcm_difference_variance_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIFVAR_AVE, "GLCM_DIFVAR_AVE");
}

void test_2d_glcm_dis_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_DIS_AVE, "GLCM_DIS_AVE");
}

void test_2d_glcm_entropy_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ENTROPY_AVE, "GLCM_ENTROPY_AVE");
}

void test_2d_glcm_id_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ID_AVE, "GLCM_ID_AVE");
}

void test_2d_glcm_idn_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IDN_AVE, "GLCM_IDN_AVE");
}

void test_2d_glcm_idm_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IDM_AVE, "GLCM_IDM_AVE");
}

void test_2d_glcm_idmn_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IDMN_AVE, "GLCM_IDMN_AVE");
}

void test_2d_glcm_iv_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_IV_AVE, "GLCM_IV_AVE");
}

void test_2d_glcm_jave_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JAVE_AVE, "GLCM_JAVE_AVE");
}

void test_2d_glcm_je_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JE_AVE, "GLCM_JE_AVE");
}

void test_2d_glcm_infomeas1_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_INFOMEAS1_AVE, "GLCM_INFOMEAS1_AVE");
}

void test_2d_glcm_infomeas2_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_INFOMEAS2_AVE, "GLCM_INFOMEAS2_AVE");
}

void test_2d_glcm_variance_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_VARIANCE_AVE, "GLCM_VARIANCE_AVE");
}

void test_2d_glcm_jmax_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JMAX_AVE, "GLCM_JMAX_AVE");
}

void test_2d_glcm_jvar_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_JVAR_AVE, "GLCM_JVAR_AVE");
}

void test_2d_glcm_sum_average_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_SUMAVERAGE_AVE, "GLCM_SUMAVERAGE_AVE");
}

void test_2d_glcm_sum_entropy_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_SUMENTROPY_AVE, "GLCM_SUMENTROPY_AVE");
}

void test_2d_glcm_sum_variance_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_SUMVARIANCE_AVE, "GLCM_SUMVARIANCE_AVE");
}

// ASM/CONTRAST/CORRELATION/ENERGY/HOM1 and their _AVE twins, moved back here from
// test_2d_glcm_matlab.h. That file was named for an oracle it never ran: its ten functions called
// this file's assert_glcm_feature_regression, on this file's fixture, against this file's snapshot
// table, at this file's 1% tolerance -- the only thing distinguishing them from a regression
// assertion was the word "matlab" in their names. The values cannot be MATLAB's: the table pins
// Nyxus output and was refreshed in 2026-06 to follow a Nyxus bug fix, which no external golden
// could survive. Three of the five (ASM, ENERGY, CORRELATION) are in the transpose-sensitive group
// documented above, measured to diverge from a symmetric-matrix tool by 3.7% and more.
// The registry rows now read status=regression to match. Vetting these against MATLAB graycoprops
// remains open work; it needs goldens generated from graycoprops itself (SPEC 6.4), not a rename.
void test_2d_glcm_asm_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ASM, "GLCM_ASM");
}

void test_2d_glcm_contrast_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CONTRAST, "GLCM_CONTRAST");
}

void test_2d_glcm_correlation_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CORRELATION, "GLCM_CORRELATION");
}

void test_2d_glcm_energy_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ENERGY, "GLCM_ENERGY");
}

void test_2d_glcm_hom1_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_HOM1, "GLCM_HOM1");
}

void test_2d_glcm_asm_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ASM_AVE, "GLCM_ASM_AVE");
}

void test_2d_glcm_contrast_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CONTRAST_AVE, "GLCM_CONTRAST_AVE");
}

void test_2d_glcm_correlation_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_CORRELATION_AVE, "GLCM_CORRELATION_AVE");
}

void test_2d_glcm_energy_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_ENERGY_AVE, "GLCM_ENERGY_AVE");
}

void test_2d_glcm_hom1_ave_regression()
{
    assert_glcm_feature_regression(Nyxus::Feature2D::GLCM_HOM1_AVE, "GLCM_HOM1_AVE");
}
