#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/raw_nifti.h"

#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"

// PyRadiomics goldens for the 3D GLCM family (SPEC 6.4 provenance).
// tool=pyradiomics 3.0.1 (SimpleITK 2.3.1, Python 3.8); env=nyxus_oracle (conda, needs Python <=3.9);
// recipe=glcm3d.pyradiomics_bincount20; generator=tests/vetting/oracles/gen_glcm3d_pyradiomics.py.
//
// Fixture: the COMPAT phantom -- data/nifti/compat_int/compat_int_mri.nii +
// compat_seg/compat_seg_liver.nii, label 1 -- which is what get_3d_compat_phantom() returns and what
// every assertion below uses. (The ut_ phantom named in the older version of this comment belongs to
// test_3d_glcm_regression.h, at a different bin count; the two benchmarks are not comparable.)
//
// Nyxus side: 100 grey levels, GLCM offset 1, asymmetric cooc matrix. PyRadiomics' GLCM is symmetric
// by default, which is the convention gap the 10% band covers.
//
// PyRadiomics reports ONE value per feature over its whole direction set, i.e. the Nyxus *_AVE
// aggregation over the 13 3D angles -- not a per-angle value. Each golden below is therefore the
// reference for both the per-angle base feature (through calc_ave) and the stored *_AVE feature.
//
// Getting Pyradiomics ground truth values (the generator does exactly this):
//      pyradiomics <intensity>.nii <mask>.nii --param settings1.yaml
// 
// where file "settings1.yaml" is:
// 
//  setting:
//  #disabled - binWidth: 25
//  binCount : 20
//  label : 1
//  interpolator : 'sitkBSpline'
//  resampledPixelSpacing :
//  weightingNorm: 
//
//  imageType :
//        Original : {} 
//  featureClass :
//      glcm:
//        - 'Autocorrelation'
//        - 'JointAverage'
//        - 'ClusterProminence'
//        - 'ClusterShade'
//        - 'ClusterTendency'
//        - 'Contrast'
//        - 'Correlation'
//        - 'DifferenceAverage'
//        - 'DifferenceEntropy'
//        - 'DifferenceVariance'
//        - 'JointEnergy'
//        - 'JointEntropy'
//        - 'Imc1'
//        - 'Imc2'
//        - 'Idm'
//        - 'Idmn'
//        - 'Id'
//        - 'Idn'
//        - 'InverseVariance'
//        - 'MaximumProbability'
//        - 'SumAverage'
//        - 'SumEntropy'
//        - 'SumSquares'
//

static ref_vals_map<double> glcm_3d_pyradiomics_ref_vals
{
    {"3GLCM_ACOR", 122.14708306342365},         // Case-1_original_glcm_Autocorrelation
    {"3GLCM_ASM", 0.0143339715631298},          // Case-1_original_glcm_JointEnergy
    {"3GLCM_CLUPROM", 1870.7687419551776},      // Case-1_original_glcm_ClusterProminence
    {"3GLCM_CLUSHADE", 8.755242780815239},      // Case-1_original_glcm_ClusterShade
    {"3GLCM_CLUTEND", 23.113911920055934},      // Case-1_original_glcm_ClusterTendency
    {"3GLCM_CONTRAST", 8.76143159022662},       // Case-1_original_glcm_Contrast
    {"3GLCM_CORRELATION", 0.43309121847659515},  // Case-1_original_glcm_Correlation
    {"3GLCM_DIFAVE", 2.2143984613019545},       // Case-1_original_glcm_DifferenceAverage
    {"3GLCM_DIFENTRO", 2.645537347146111},      // Case-1_original_glcm_DifferenceEntropy
    {"3GLCM_DIFVAR", 3.4395235149928194},       // Case-1_original_glcm_DifferenceVariance
    // No 3GLCM_DIS entry: PyRadiomics deprecates Dissimilarity as equivalent to DifferenceAverage
    // and does not report it, so it is vetted through the DIS == DIFAVE identity instead. (The value
    // that used to sit here commented out, 27.5, was not a PyRadiomics number at all -- it is the
    // stale ut_-phantom regression pin, at a different bin count.)
    {"3GLCM_ID", 0.4459415317170447},          // Case-1_original_glcm_Id
    {"3GLCM_IDN", 0.9067759330416398},          // Case-1_original_glcm_Idn
    {"3GLCM_IDM", 0.3726945904589868},          // Case-1_original_glcm_Idm
    {"3GLCM_IDMN", 0.9797065356412845},         // Case-1_original_glcm_Idmn
    {"3GLCM_INFOMEAS1", -0.09924883901268647},  // Case-1_original_glcm_Imc1
    {"3GLCM_INFOMEAS2", 0.5781205730305887},    // Case-1_original_glcm_Imc2
    {"3GLCM_IV", 0.36184532347527026},          // Case-1_original_glcm_InverseVariance
    {"3GLCM_JAVE", 10.888107083238083},         // Case-1_original_glcm_JointAverage
    {"3GLCM_JE", 6.701464036118752},            // Case-1_original_glcm_JointEntropy
    // only in pyRadiomics: Case-1_original_glcm_MCC
    {"3GLCM_JMAX", 0.036309525310650057},       // 1_original_glcm_MaximumProbability
    {"3GLCM_JVAR", 7.968835877570637},          // Case-1_original_glcm_SumSquares
    {"3GLCM_SUMAVERAGE", 21.776214166476173},   // Case-1_original_glcm_SumAverage
    {"3GLCM_SUMENTROPY", 4.27263829307018}      // Case-1_original_glcm_SumEntropy
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/phantoms/ut_inten.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/phantoms/ut_mask57.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    // ROI sitting in the mask phantom
    return { ipath, mpath, 57 };
}

static std::tuple<std::string, std::string, int> get_3d_compat_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/compat_int/compat_int_mri.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());

    fs::path f2("/data/nifti/compat_seg/compat_seg_liver.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    return { ipath, mpath, 1 };
}

void assert_3d_glcm_feature_pyradiomics (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare
    
    // check that requested feature exists
    auto iter = glcm_3d_pyradiomics_ref_vals.find(fname);
    ASSERT_TRUE(iter != glcm_3d_pyradiomics_ref_vals.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // (2) mock the 3D workflow

    Environment e;

    // slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    // properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));

    // voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));

    // buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (3) common feature extraction settings

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;

    // (4) GLCM-specific feature settings mocking default pyRadiomics settings

    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;  // intentionally negative to activate radiomics binCount-based grey-binning
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    // (5) feature extraction

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(e.theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // extract the feature
    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));

    // (6) get values

    f.save_value(r.fvals);

    // (7) aggregate angled subfeatures
    double atot = f.calc_ave (r.fvals[fcode]);

    // (8) verdict
    ASSERT_TRUE(agrees_gt(atot, glcm_3d_pyradiomics_ref_vals[fname], 10.));
}

// Deep-dive: verify the 7 config-sensitive 3D GLCM features equal their already-vetted twins
// (numerically, same fixture/config), so they can be promoted by equivalence. Dumps calc_ave pairs.
void test_3d_glcm_equivalence_dump_pyradiomics()
{
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    using F = Nyxus::Feature3D;
    struct Pair { const char* a; F fa; const char* b; F fb; };
    std::vector<Pair> pairs = {
        {"3GLCM_DIS", F::GLCM_DIS, "3GLCM_DIFAVE", F::GLCM_DIFAVE},
        {"3GLCM_ENERGY", F::GLCM_ENERGY, "3GLCM_ASM", F::GLCM_ASM},
        {"3GLCM_ENTROPY", F::GLCM_ENTROPY, "3GLCM_JE", F::GLCM_JE},
        {"3GLCM_HOM1", F::GLCM_HOM1, "3GLCM_ID", F::GLCM_ID},
        {"3GLCM_HOM2", F::GLCM_HOM2, "3GLCM_IDM", F::GLCM_IDM},
        {"3GLCM_SUMVARIANCE", F::GLCM_SUMVARIANCE, "3GLCM_CLUTEND", F::GLCM_CLUTEND},
        {"3GLCM_VARIANCE", F::GLCM_VARIANCE, "3GLCM_JVAR", F::GLCM_JVAR},
    };
    for (auto& p : pairs)
    {
        double va = f.calc_ave(r.fvals[(int)p.fa]);
        double vb = f.calc_ave(r.fvals[(int)p.fb]);
        double m = std::max(std::abs(va), std::abs(vb)); if (m < 1e-12) m = 1e-12;
        double rel = std::abs(va - vb) / m;
        std::cout << "[3DGLCM-EQ] " << p.a << "=" << va << "  " << p.b << "=" << vb
                  << "  rel=" << rel << (rel < 1e-6 ? "  EQUAL" : "  DIFFER") << "\n";
        // Config-sensitive 3D GLCM features are numerically identical to their pyradiomics-vetted
        // twins (also guards the HOM2/ENTROPY /sum_p fix: pre-fix ENTROPY!=JE, HOM2!=IDM).
        EXPECT_NEAR(va, vb, std::abs(vb) * 1e-6 + 1e-9) << p.a << " != " << p.b;
    }
}

void test_3d_glcm_acor_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_ACOR, "3GLCM_ACOR");
}

void test_3d_glcm_asm_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_ASM, "3GLCM_ASM");
}

void test_3d_glcm_cluprom_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUPROM, "3GLCM_CLUPROM");
}

void test_3d_glcm_clushade_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUSHADE, "3GLCM_CLUSHADE");
}

void test_3d_glcm_clutend_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUTEND, "3GLCM_CLUTEND");
}

void test_3d_glcm_contrast_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CONTRAST, "3GLCM_CONTRAST");
}

void test_3d_glcm_correlation_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_CORRELATION, "3GLCM_CORRELATION");
}

void test_3d_glcm_difference_average_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFAVE, "3GLCM_DIFAVE");
}

void test_3d_glcm_difference_entropy_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFENTRO, "3GLCM_DIFENTRO");
}

void test_3d_glcm_difference_variance_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFVAR, "3GLCM_DIFVAR");
}

void test_3d_glcm_id_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_ID, "3GLCM_ID");
}

void test_3d_glcm_idn_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDN, "3GLCM_IDN");
}

void test_3d_glcm_idm_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDM, "3GLCM_IDM");
}

void test_3d_glcm_idmn_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDMN, "3GLCM_IDMN");
}

void test_3d_glcm_infomeas1_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS1, "3GLCM_INFOMEAS1");
}

void test_3d_glcm_infomeas2_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS2, "3GLCM_INFOMEAS2");
}

void test_3d_glcm_iv_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_IV, "3GLCM_IV");
}

void test_3d_glcm_jave_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JAVE, "3GLCM_JAVE");
}

void test_3d_glcm_je_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JE, "3GLCM_JE");
}

void test_3d_glcm_jmax_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JMAX, "3GLCM_JMAX");
}

void test_3d_glcm_jvar_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_JVAR, "3GLCM_JVAR");
}

void test_3d_glcm_sum_average_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMAVERAGE, "3GLCM_SUMAVERAGE");
}

void test_3d_glcm_sum_entropy_pyradiomics()
{
    assert_3d_glcm_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMENTROPY, "3GLCM_SUMENTROPY");
}



// ---------------------------------------------------------------------------------------------
// The _AVE features. PyRadiomics reports one value per feature over its whole direction set, which
// is the Nyxus *_AVE aggregation -- assert_3d_glcm_feature_pyradiomics() above already compares that
// quantity, but it recomputes it with calc_ave() and books the result against the per-angle base
// feature. Nothing checked the *_AVE features that save_value() actually writes
// (fvals[..._AVE][0] = calc_ave(...) in 3d_glcm.cpp), which is what the registry's *_AVE rows name.
//
// These read the stored feature, so a defect in how save_value populates *_AVE fails here and
// nowhere else.
// ---------------------------------------------------------------------------------------------

void assert_3d_glcm_ave_feature_pyradiomics (const Nyxus::Feature3D& ave_fcode,
    const std::string& base_fname)
{
    auto iter = glcm_3d_pyradiomics_ref_vals.find(base_fname);
    ASSERT_TRUE(iter != glcm_3d_pyradiomics_ref_vals.end()) << base_fname;

    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    SCOPED_TRACE(std::string("PYRADIOMICS_ORACLE__") + base_fname + "_AVE");
    ASSERT_TRUE(agrees_gt(r.fvals[(int)ave_fcode][0], glcm_3d_pyradiomics_ref_vals[base_fname], 10.));
}

void test_3d_glcm_acor_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_ACOR_AVE, "3GLCM_ACOR");
}

void test_3d_glcm_asm_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_ASM_AVE, "3GLCM_ASM");
}

void test_3d_glcm_cluprom_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUPROM_AVE, "3GLCM_CLUPROM");
}

void test_3d_glcm_clushade_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUSHADE_AVE, "3GLCM_CLUSHADE");
}

void test_3d_glcm_clutend_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CLUTEND_AVE, "3GLCM_CLUTEND");
}

void test_3d_glcm_contrast_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CONTRAST_AVE, "3GLCM_CONTRAST");
}

void test_3d_glcm_correlation_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_CORRELATION_AVE, "3GLCM_CORRELATION");
}

void test_3d_glcm_difave_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFAVE_AVE, "3GLCM_DIFAVE");
}

void test_3d_glcm_difentro_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFENTRO_AVE, "3GLCM_DIFENTRO");
}

void test_3d_glcm_difvar_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_DIFVAR_AVE, "3GLCM_DIFVAR");
}

void test_3d_glcm_id_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_ID_AVE, "3GLCM_ID");
}

void test_3d_glcm_idm_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDM_AVE, "3GLCM_IDM");
}

void test_3d_glcm_idmn_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDMN_AVE, "3GLCM_IDMN");
}

void test_3d_glcm_idn_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IDN_AVE, "3GLCM_IDN");
}

void test_3d_glcm_infomeas1_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS1_AVE, "3GLCM_INFOMEAS1");
}

void test_3d_glcm_infomeas2_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_INFOMEAS2_AVE, "3GLCM_INFOMEAS2");
}

void test_3d_glcm_iv_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_IV_AVE, "3GLCM_IV");
}

void test_3d_glcm_jave_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JAVE_AVE, "3GLCM_JAVE");
}

void test_3d_glcm_je_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JE_AVE, "3GLCM_JE");
}

void test_3d_glcm_jmax_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JMAX_AVE, "3GLCM_JMAX");
}

void test_3d_glcm_jvar_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_JVAR_AVE, "3GLCM_JVAR");
}

void test_3d_glcm_sumaverage_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMAVERAGE_AVE, "3GLCM_SUMAVERAGE");
}

void test_3d_glcm_sumentropy_ave_pyradiomics()
{
    assert_3d_glcm_ave_feature_pyradiomics (Nyxus::Feature3D::GLCM_SUMENTROPY_AVE, "3GLCM_SUMENTROPY");
}


// The six *_AVE features PyRadiomics does not report under their own name. Each is numerically
// identical to a twin that does, and the identity is not an assumption: it is asserted at 1e-6 on
// the per-angle values by test_3d_glcm_equivalence_dump_pyradiomics(). This pins the same identity
// on the stored *_AVE features, so the twin's PyRadiomics golden carries both.
void test_3d_glcm_ave_equivalence_pyradiomics()
{
    auto [ipath, mpath, label] = get_3d_compat_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    Environment e;
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();

    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0, ipath, mpath, 0));
    std::vector<int> batch = { label };
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    Fsettings s;
    s.resize((int)NyxSetting::__COUNT__);
    s[(int)NyxSetting::SOFTNAN].rval = 0.0;
    s[(int)NyxSetting::TINY].rval = 0.0;
    s[(int)NyxSetting::SINGLEROI].bval = false;
    s[(int)NyxSetting::GREYDEPTH].ival = 100;
    s[(int)NyxSetting::PIXELSIZEUM].rval = 100;
    s[(int)NyxSetting::PIXELDISTANCE].ival = 5;
    s[(int)NyxSetting::USEGPU].bval = false;
    s[(int)NyxSetting::VERBOSLVL].ival = 0;
    s[(int)NyxSetting::IBSI].bval = false;
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -20;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    LR& r = e.roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r, s));
    f.save_value(r.fvals);

    using F = Nyxus::Feature3D;
    struct Pair { F a; F b; const char* an; const char* bn; const char* golden; };
    std::vector<Pair> pairs = {
        { F::GLCM_DIS_AVE, F::GLCM_DIFAVE_AVE, "3GLCM_DIS_AVE", "3GLCM_DIFAVE_AVE", "3GLCM_DIFAVE" },
        { F::GLCM_ENERGY_AVE, F::GLCM_ASM_AVE, "3GLCM_ENERGY_AVE", "3GLCM_ASM_AVE", "3GLCM_ASM" },
        { F::GLCM_ENTROPY_AVE, F::GLCM_JE_AVE, "3GLCM_ENTROPY_AVE", "3GLCM_JE_AVE", "3GLCM_JE" },
        { F::GLCM_HOM1_AVE, F::GLCM_ID_AVE, "3GLCM_HOM1_AVE", "3GLCM_ID_AVE", "3GLCM_ID" },
        { F::GLCM_SUMVARIANCE_AVE, F::GLCM_CLUTEND_AVE, "3GLCM_SUMVARIANCE_AVE", "3GLCM_CLUTEND_AVE", "3GLCM_CLUTEND" },
        { F::GLCM_VARIANCE_AVE, F::GLCM_JVAR_AVE, "3GLCM_VARIANCE_AVE", "3GLCM_JVAR_AVE", "3GLCM_JVAR" }
    };

    for (auto& p : pairs)
    {
        SCOPED_TRACE(std::string("PYRADIOMICS_ORACLE__") + p.an + " via " + p.bn);
        double va = r.fvals[(int)p.a][0], vb = r.fvals[(int)p.b][0];
        // (1) the identity holds on the stored _AVE features
        EXPECT_NEAR(va, vb, std::abs(vb) * 1e-6 + 1e-9) << p.an << " != " << p.bn;
        // (2) and the twin still matches its PyRadiomics golden, so the identity carries the claim
        ASSERT_TRUE(agrees_gt(vb, glcm_3d_pyradiomics_ref_vals[p.golden], 10.));
    }
}
