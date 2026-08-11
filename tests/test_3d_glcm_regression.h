#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glcm.h"
#include "../src/nyx/helpers/fsystem.h"
#include "test_ref_vals.h"

// Feature values calculated on intensity ut_inten.nii and mask ut_inten.nii, label 57:
// (100 grey levels, offset 1, and asymmetric cooc matrix)
static ref_vals_map<float> glcm_3d_regression_ref_vals
{
    {"3GLCM_ACOR", 8686.0},
    {"3GLCM_ASM", 0.87},
    {"3GLCM_CLUPROM", 128972000.0},
    {"3GLCM_CLUSHADE", 390252.0},
    {"3GLCM_CLUTEND", 18057.0},
    {"3GLCM_CONTRAST", 1841.0},
    {"3GLCM_CORRELATION", 3.26},
    {"3GLCM_DIFAVE", 27.5},
    {"3GLCM_DIFENTRO", 11.8},
    {"3GLCM_DIFVAR", 1641.0},
    {"3GLCM_DIS", 27.5},
    {"3GLCM_ID", 2.5},
    {"3GLCM_IDN", 3.8},
    {"3GLCM_IDM", 2.4},
    {"3GLCM_IDMN", 3.9},
    {"3GLCM_INFOMEAS1", -1.7},
    {"3GLCM_INFOMEAS2", 3.9},
    {"3GLCM_IV", 0.45},
    {"3GLCM_JAVE", 136.1},
    {"3GLCM_JE", 25.8},
    {"3GLCM_JMAX", 1.86},
    {"3GLCM_JVAR", 4974.61},
    {"3GLCM_SUMAVERAGE", 272.2},
    {"3GLCM_SUMENTROPY", 18.86},
    {"3GLCM_SUMVARIANCE", 18057.4}
};

static std::tuple<std::string, std::string, int> get_3d_segmented_phantom()
{
    // physical paths of the phantoms
    fs::path this_fpath(__FILE__);
    fs::path pp = this_fpath.parent_path();

    fs::path f1("/data/nifti/phantoms/ut_inten.nii");
    fs::path i_phys_path = (pp.string() + f1.make_preferred().string());
    //ASSERT_TRUE(fs::exists(i_phys_path));

    fs::path f2("/data/nifti/phantoms/ut_mask57.nii");
    fs::path m_phys_path = (pp.string() + f2.make_preferred().string());
    //ASSERT_TRUE(fs::exists(m_phys_path));

    std::string ipath = i_phys_path.string(),
        mpath = m_phys_path.string();

    // ROI sitting in the mask phantom
    return {ipath, mpath, 57};
}

void assert_3d_glcm_feature_regression (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // (1) prepare

    // check that requested feature exists
    auto iter = glcm_3d_regression_ref_vals.find(fname);
    ASSERT_TRUE(iter != glcm_3d_regression_ref_vals.end());

    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
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

    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = -100;  // intentionally negative to activate radiomics binCount-based grey-binning
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

    // aggregate angled subfeatures (13 angles for 3D)
    double atot = f.calc_ave(r.fvals[fcode]);

    // (7) verdict
    ASSERT_TRUE(agrees_gt(atot, glcm_3d_regression_ref_vals[fname], 10.));
}

void test_3d_glcm_acor_regression()
{
    assert_3d_glcm_feature_regression (Nyxus::Feature3D::GLCM_ACOR, "3GLCM_ACOR");
}

void test_3d_glcm_asm_regression()
{
    assert_3d_glcm_feature_regression (Nyxus::Feature3D::GLCM_ASM, "3GLCM_ASM");
}

void test_3d_glcm_cluprom_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CLUPROM, "3GLCM_CLUPROM");
}

void test_3d_glcm_clushade_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CLUSHADE, "3GLCM_CLUSHADE");
}

void test_3d_glcm_clutend_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CLUTEND, "3GLCM_CLUTEND");
}

void test_3d_glcm_contrast_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CONTRAST, "3GLCM_CONTRAST");
}

void test_3d_glcm_correlation_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_CORRELATION, "3GLCM_CORRELATION");
}

void test_3d_glcm_difference_average_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIFAVE, "3GLCM_DIFAVE");
}

void test_3d_glcm_difference_entropy_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIFENTRO, "3GLCM_DIFENTRO");
}

void test_3d_glcm_difference_variance_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIFVAR, "3GLCM_DIFVAR");
}

void test_3d_glcm_dis_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_DIS, "3GLCM_DIS");
}

void test_3d_glcm_id_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_ID, "3GLCM_ID");
}

void test_3d_glcm_idn_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IDN, "3GLCM_IDN");
}

void test_3d_glcm_idm_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IDM, "3GLCM_IDM");
}

void test_3d_glcm_idmn_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IDMN, "3GLCM_IDMN");
}

void test_3d_glcm_infomeas1_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_INFOMEAS1, "3GLCM_INFOMEAS1");
}

void test_3d_glcm_infomeas2_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_INFOMEAS2, "3GLCM_INFOMEAS2");
}

void test_3d_glcm_iv_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_IV, "3GLCM_IV");
}

void test_3d_glcm_jave_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JAVE, "3GLCM_JAVE");
}

void test_3d_glcm_je_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JE, "3GLCM_JE");
}

void test_3d_glcm_jmax_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JMAX, "3GLCM_JMAX");
}

void test_3d_glcm_jvar_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_JVAR, "3GLCM_JVAR");
}

void test_3d_glcm_sum_average_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_SUMAVERAGE, "3GLCM_SUMAVERAGE");
}

void test_3d_glcm_sum_entropy_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_SUMENTROPY, "3GLCM_SUMENTROPY");
}

void test_3d_glcm_sum_variance_regression()
{
    assert_3d_glcm_feature_regression(Nyxus::Feature3D::GLCM_SUMVARIANCE, "3GLCM_SUMVARIANCE");
}

