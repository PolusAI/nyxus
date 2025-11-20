#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glcm.h"

// Feature values calculated on intensity ut_inten.nii and mask ut_inten.nii, label 57:
// (100 grey levels, offset 1, and asymmetric cooc matrix)
//
// Getting Pyradiomics ground truth values:
//      pyradiomics mri.nii.gz liver.nii.gz
static std::unordered_map<std::string, float> compat_d3glcm_GT
{
    {"3GLCM_ACOR", 106.01884203801286},     // Case-1_original_glcm_Autocorrelation
    {"3GLCM_ASM", 0.87},        // Case-1_original_glcm_JointEnergy
    {"3GLCM_CLUPROM", 1130.1440778504054},     // Case-1_original_glcm_ClusterProminence
    {"3GLCM_CLUSHADE", 6.354248418538878},       // Case-1_original_glcm_ClusterShade
    {"3GLCM_CLUTEND", 17.944878549169108},         // Case-1_original_glcm_ClusterTendency
    {"3GLCM_CONTRAST", 6.849306098789943},         // Case-1_original_glcm_Contrast
    {"3GLCM_CORRELATION", 0.4305477709920443},        // Case-1_original_glcm_Correlation
    {"3GLCM_DIFAVE", 1.9565587912776674},     // Case-1_original_glcm_DifferenceAverage
    {"3GLCM_DIFENTRO", 2.490160784136734},   // Case-1_original_glcm_DifferenceEntropy
    {"3GLCM_DIFVAR", 2.6901445354782387},   // Case-1_original_glcm_DifferenceVariance
    {"3GLCM_DIS", 27.5},
    {"3GLCM_ID", 0.47211428859469606},          // Case-1_original_glcm_Id
    {"3GLCM_IDN", 0.9822362042997563},         
    {"3GLCM_IDM", 0.4040020605537021},         // Case-1_original_glcm_Idm
    {"3GLCM_IDMN", 0.9822362042997563},     // Case-1_original_glcm_Idmn
    {"3GLCM_INFOMEAS1", -1.7},  // Case-1_original_glcm_Imc1
    {"3GLCM_INFOMEAS2", 3.9},   // Case-1_original_glcm_Imc2
    {"3GLCM_IV", 0.45},         // Case-1_original_glcm_InverseVariance
    {"3GLCM_JAVE", 136.1},      // Case-1_original_glcm_JointAverage
    {"3GLCM_JE", 25.8},         // Case-1_original_glcm_JointEntropy
    // only in pyRadiomics: Case-1_original_glcm_MCC
    // only in pyRadiomics: Case-
    {"3GLCM_JMAX", 1.86},       // 1_original_glcm_MaximumProbability
    {"3GLCM_JVAR", 4974.61},    // Case-1_original_glcm_SumSquares
    {"3GLCM_SUMAVERAGE", 272.2},    // Case-1_original_glcm_SumAverage
    {"3GLCM_SUMENTROPY", 18.86},    // Case-1_original_glcm_SumEntropy
    {"3GLCM_SUMVARIANCE", 18057.4}  // Case-1_original_glcm_SumSquares
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
    return { ipath, mpath, 57 };
}

void test_compat_3glcm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    Environment e;
    // (1) slide -> dataset -> prescan 
    e.dataset.dataset_props.reserve(1);
    SlideProps& sp = e.dataset.dataset_props.emplace_back(ipath, mpath);
    ASSERT_TRUE(scan_slide_props(sp, 3, e.anisoOptions, e.resultOptions.need_annotation()));
    e.dataset.update_dataset_props_extrema();
    // (2) properties of specific ROIs sitting in 'e.uniqueLabels'
    clear_slide_rois(e.uniqueLabels, e.roiData);
    ASSERT_TRUE(gatherRoisMetrics_3D(e, 0/*slide_index*/, ipath, mpath, 0/*t_index*/));
    // (3) voxel clouds
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(e, batch, ipath, mpath, 0/*t_index*/));
    // (4) buffers
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch, e.roiData, e.hostCache));

    // (5) common feature settings
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

    // (6) GLCM-specific feature settings mocking default pyRadiomics settings
    s[(int)NyxSetting::GLCM_GREYDEPTH].ival = 128;
    s[(int)NyxSetting::GLCM_OFFSET].ival = 1;
    s[(int)NyxSetting::GLCM_NUMANG].ival = 8;
    s[(int)NyxSetting::GLCM_SPARSEINTENS].bval = true;

    // (7) feature extraction

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

    // (8) saving values

    f.save_value(r.fvals);

    // aggregate subfeatures
    double atot = r.fvals[fcode][0] + r.fvals[fcode][1] + r.fvals[fcode][2] + r.fvals[fcode][3];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, compat_d3glcm_GT[fname], 10.));
}

void test_compat_3glcm_ACOR()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_ACOR, "3GLCM_ACOR");
}

void test_compat_3glcm_angular_2d_moment()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_ASM, "3GLCM_ASM");
}

void test_compat_3glcm_CLUPROM()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CLUPROM, "3GLCM_CLUPROM");
}

void test_compat_3glcm_CLUSHADE()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CLUSHADE, "3GLCM_CLUSHADE");
}

void test_compat_3glcm_CLUTEND()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CLUTEND, "3GLCM_CLUTEND");
}

void test_compat_3glcm_contrast()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CONTRAST, "3GLCM_CONTRAST");
}

void test_compat_3glcm_correlation()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_CORRELATION, "3GLCM_CORRELATION");
}

void test_compat_3glcm_difference_average()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_DIFAVE, "3GLCM_DIFAVE");
}

void test_compat_3glcm_difference_entropy()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_DIFENTRO, "3GLCM_DIFENTRO");
}

void test_compat_3glcm_difference_variance()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_DIFVAR, "3GLCM_DIFVAR");
}

void test_compat_3glcm_DIS()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_DIS, "3GLCM_DIS");
}

void test_compat_3glcm_ID()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_ID, "3GLCM_ID");
}

void test_compat_3glcm_IDN()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IDN, "3GLCM_IDN");
}

void test_compat_3glcm_IDM()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IDM, "3GLCM_IDM");
}

void test_compat_3glcm_IDMN()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IDMN, "3GLCM_IDMN");
}

void test_compat_3glcm_infomeas1()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_INFOMEAS1, "3GLCM_INFOMEAS1");
}

void test_compat_3glcm_infomeas2()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_INFOMEAS2, "3GLCM_INFOMEAS2");
}

void test_compat_3glcm_IV()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_IV, "3GLCM_IV");
}

void test_compat_3glcm_JAVE()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JAVE, "3GLCM_JAVE");
}

void test_compat_3glcm_JE()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JE, "3GLCM_JE");
}

void test_compat_3glcm_JMAX()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JMAX, "3GLCM_JMAX");
}

void test_compat_3glcm_JVAR()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_JVAR, "3GLCM_JVAR");
}

void test_compat_3glcm_sum_average()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_SUMAVERAGE, "3GLCM_SUMAVERAGE");
}

void test_compat_3glcm_sum_entropy()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_SUMENTROPY, "3GLCM_SUMENTROPY");
}

void test_compat_3glcm_sum_variance()
{
    test_compat_3glcm_feature (Nyxus::Feature3D::GLCM_SUMVARIANCE, "3GLCM_SUMVARIANCE");
}

