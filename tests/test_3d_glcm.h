#pragma once

#include <gtest/gtest.h>
#include <unordered_map> 
#include "../src/nyx/environment.h"
#include "../src/nyx/featureset.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/3d_glcm.h"

// Feature values calculated on intensity ut_inten.nii and mask ut_inten.nii, label 57:
// (100 grey levels, offset 1, and asymmetric cooc matrix)
static std::unordered_map<std::string, float> d3glcm_GT
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

void test_3glcm_feature (const Nyxus::Feature3D& expecting_fcode, const std::string& fname)
{
    // get segment info
    auto [ipath, mpath, label] = get_3d_segmented_phantom();
    ASSERT_TRUE(fs::exists(ipath));
    ASSERT_TRUE(fs::exists(mpath));

    // mock the 3D workflow
    clear_slide_rois();
    ASSERT_TRUE(gatherRoisMetrics_3D(ipath, mpath));
    std::vector<int> batch = { label };   // expecting this roi label after metrics gathering
    ASSERT_TRUE(scanTrivialRois_3D(batch, ipath, mpath));
    ASSERT_NO_THROW(allocateTrivialRoisBuffers_3D(batch));

    // make it find the feature code by name
    int fcode = -1;
    ASSERT_TRUE(theFeatureSet.find_3D_FeatureByString(fname, fcode));
    // ... and that it's the feature we expect
    ASSERT_TRUE((int)expecting_fcode == fcode);

    // set feature's state
    D3_GLCM_feature::n_levels = 100;
    D3_GLCM_feature::offset = 1;
    D3_GLCM_feature::symmetric_glcm = false;
    D3_GLCM_feature::angles = { 0, 45, 90, 135 };
    Environment::ibsi_compliance = false;

    // extract the feature
    LR & r = Nyxus::roiData[label];
    ASSERT_NO_THROW(r.initialize_fvals());
    D3_GLCM_feature f;
    ASSERT_NO_THROW(f.calculate(r));
    f.save_value(r.fvals);

    // aggregate all the angles
    double atot = r.fvals[fcode][0] + r.fvals[fcode][1] + r.fvals[fcode][2] + r.fvals[fcode][3];

    // verdict
    ASSERT_TRUE(agrees_gt(atot, d3glcm_GT[fname], 10.));
}

void test_3glcm_ACOR()
{
    test_3glcm_feature (Nyxus::Feature3D::GLCM_ACOR, "3GLCM_ACOR");
}

void test_3glcm_angular_2d_moment()
{
    test_3glcm_feature (Nyxus::Feature3D::GLCM_ASM, "3GLCM_ASM");
}

void test_3glcm_CLUPROM()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_CLUPROM, "3GLCM_CLUPROM");
}

void test_3glcm_CLUSHADE()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_CLUSHADE, "3GLCM_CLUSHADE");
}

void test_3glcm_CLUTEND()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_CLUTEND, "3GLCM_CLUTEND");
}

void test_3glcm_contrast()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_CONTRAST, "3GLCM_CONTRAST");
}

void test_3glcm_correlation()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_CORRELATION, "3GLCM_CORRELATION");
}

void test_3glcm_difference_average()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_DIFAVE, "3GLCM_DIFAVE");
}

void test_3glcm_difference_entropy()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_DIFENTRO, "3GLCM_DIFENTRO");
}

void test_3glcm_difference_variance()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_DIFVAR, "3GLCM_DIFVAR");
}

void test_3glcm_DIS()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_DIS, "3GLCM_DIS");
}

void test_3glcm_ID()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_ID, "3GLCM_ID");
}

void test_3glcm_IDN()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_IDN, "3GLCM_IDN");
}

void test_3glcm_IDM()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_IDM, "3GLCM_IDM");
}

void test_3glcm_IDMN()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_IDMN, "3GLCM_IDMN");
}

void test_3glcm_infomeas1()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_INFOMEAS1, "3GLCM_INFOMEAS1");
}

void test_3glcm_infomeas2()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_INFOMEAS2, "3GLCM_INFOMEAS2");
}

void test_3glcm_IV()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_IV, "3GLCM_IV");
}

void test_3glcm_JAVE()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_JAVE, "3GLCM_JAVE");
}

void test_3glcm_JE()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_JE, "3GLCM_JE");
}

void test_3glcm_JMAX()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_JMAX, "3GLCM_JMAX");
}

void test_3glcm_JVAR()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_JVAR, "3GLCM_JVAR");
}

void test_3glcm_sum_average()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_SUMAVERAGE, "3GLCM_SUMAVERAGE");
}

void test_3glcm_sum_entropy()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_SUMENTROPY, "3GLCM_SUMENTROPY");
}

void test_3glcm_sum_variance()
{
    test_3glcm_feature(Nyxus::Feature3D::GLCM_SUMVARIANCE, "3GLCM_SUMVARIANCE");
}

